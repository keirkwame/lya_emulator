"""Module for computing the likelihood function for the forest emulator."""
import math
from datetime import datetime
import mpmath as mmh
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import numpy.testing as npt
import scipy.optimize as spo
import scipy.interpolate
import emcee
from . import coarse_grid
from . import flux_power
from . import lyman_data
from . import mean_flux as mflux
from .latin_hypercube import map_to_unit_cube, map_from_unit_cube
from .quadratic_emulator import QuadraticEmulator

def _siIIIcorr(kf):
    """For precomputing the shape of the SiIII correlation"""
    #Compute bin boundaries in logspace.
    kmids = np.zeros(np.size(kf)+1)
    kmids[1:-1] = np.exp((np.log(kf[1:])+np.log(kf[:-1]))/2.)
    #arbitrary final point
    kmids[-1] = 2*math.pi/2271 + kmids[-2]
    # This is the average of cos(2271k) across the k interval in the bin
    siform = np.zeros_like(kf)
    siform = (np.sin(2271*kmids[1:])-np.sin(2271*kmids[:-1]))/(kmids[1:]-kmids[:-1])/2271.
    #Correction for the zeroth bin, because the integral is oscillatory there.
    siform[0] = np.cos(2271*kf[0])
    return siform

def SiIIIcorr(fSiIII, tau_eff, kf):
    """The correction for SiIII contamination, as per McDonald."""
    assert tau_eff > 0
    aa = fSiIII/(1-np.exp(-tau_eff))
    return 1 + aa**2 + 2 * aa * _siIIIcorr(kf)

def gelman_rubin(chain):
    """Compute the Gelman-Rubin statistic for a chain"""
    ssq = np.var(chain, axis=1, ddof=1)
    W = np.mean(ssq, axis=0)
    tb = np.mean(chain, axis=1)
    tbb = np.mean(tb, axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n / (m - 1) * np.sum((tbb - tb)**2, axis=0)
    var_t = (n - 1) / n * W + 1 / n * B
    R = np.sqrt(var_t / W)
    return R

def invert_block_diagonal_covariance(full_covariance_matrix, n_blocks):
    """Efficiently invert block diagonal covariance matrix"""
    inverse_covariance_matrix = np.zeros_like(full_covariance_matrix)
    nz = n_blocks
    nk = int(full_covariance_matrix.shape[0] / nz)
    for z in range(nz): #Loop over blocks by redshift
        start_index = nk * z
        end_index = nk * (z + 1)
        inverse_covariance_block = npl.inv(full_covariance_matrix[start_index: end_index, start_index: end_index])
        inverse_covariance_matrix[start_index: end_index, start_index: end_index] = inverse_covariance_block
    return inverse_covariance_matrix

def load_data(datadir, *, kf, max_z=4.2, redshifts=None, pixel_resolution_km_s='default', t0=1., mean_flux_model='low_z'):
    """Load and initialise a "fake data" flux power spectrum"""
    #Load the data directory
    myspec = flux_power.MySpectra(max_z=max_z, redshifts=redshifts, pixel_resolution_km_s=pixel_resolution_km_s)
    pps = myspec.get_snapshot_list(datadir)
    #self.data_fluxpower is used in likelihood.
    if mean_flux_model == 'low_z':
        mean_flux_function = mflux.obs_mean_tau
    elif mean_flux_model == 'high_z':
        mean_flux_function = mflux.obs_mean_tau_high_z
    else:
        raise ValueError('Mean flux model not recognised')
    data_fluxpower = pps.get_power(kf=kf, mean_fluxes=np.exp(-t0*mean_flux_function(myspec.zout, amp=0)))
    assert np.size(data_fluxpower) % np.size(kf) == 0
    return data_fluxpower

class LikelihoodClass:
    """Class to contain likelihood computations."""
    def __init__(self, basedir, mean_flux='s', max_z = 4.2, redshifts=None, pixel_resolution_km_s='default',
                 emulator_class="standard", t0_training_value = 1., optimise_GP=True,
                 emulator_json_file='emulator_params.json', use_measured_parameters=False,
                 redshift_dependent_parameters=False, data_class='BOSS'):
        """Initialise the emulator by loading the flux power spectra from the simulations."""

        #Stored covariance matrix
        self._inverse_covariance_full = None
        #Use the covariance matrix
        if data_class == 'BOSS':
            self.lyman_data_instance = lyman_data.BOSSData()
        elif data_class == 'Boera':
            self.lyman_data_instance = lyman_data.BoeraData()
        else:
            raise ValueError('Data class not recognised')
        #'Data' now is a simulation
        self.max_z = max_z
        self.data_redshifts = redshifts
        self.pixel_resolution_km_s = pixel_resolution_km_s
        myspec = flux_power.MySpectra(max_z=max_z, redshifts=self.data_redshifts, pixel_resolution_km_s=self.pixel_resolution_km_s)
        self.zout = myspec.zout
        self.kf = self.lyman_data_instance.get_kf()

        self.t0_training_value = t0_training_value
        #Load data vector
        self.lyman_data_flux_power = self.lyman_data_instance.pf.reshape(-1, self.kf.shape[0])[:self.zout.shape[0]][::-1] #km / s; n_z * n_k

        self.mf_slope = False
        #Param limits on t0
        t0_factor = np.array([0.75,1.25])
        t0_slope = np.array([-0.25, 0.25])

        if (mean_flux == 'c') or (mean_flux == 's'):
            self.mean_flux_model = 'low_z'
            if mean_flux == 'c':
                mf = mflux.ConstMeanFlux(value=t0_training_value)
            elif mean_flux == 's':
                #Add a slope to the parameter limits
                self.mf_slope = True
                slopehigh = np.max(mflux.mean_flux_slope_to_factor(np.linspace(2.2, max_z, 11),0.25))
                slopelow = np.min(mflux.mean_flux_slope_to_factor(np.linspace(2.2, max_z, 11),-0.25))
                dense_limits = np.array([np.array(t0_factor) * np.array([slopelow, slopehigh])])
                mf = mflux.MeanFluxFactor(dense_limits = dense_limits)
        elif (mean_flux == 'c_high_z') or (mean_flux == 's_high_z') or (mean_flux == 'free_high_z'):
            self.mean_flux_model = 'high_z'
            if mean_flux == 'c_high_z':
                mf = mflux.ConstMeanFluxHighRedshift(value=t0_training_value)
            #As each redshift bin is independent, for redshift-dependent mean flux models
            #we just need to convert the input parameters to a list of mean flux scalings
            #in each redshift bin.
            #This is an example which parametrises the mean flux as an amplitude and slope.
            elif mean_flux == 's_high_z':
                self.mf_slope = True
                mf = mflux.MeanFluxFactorHighRedshift()
            elif mean_flux == 'free_high_z':
                mf = mflux.FreeMeanFlux()
        else:
            mf = mflux.MeanFluxFactor()
        self.mean_flux_instance = mf

        if emulator_class == "standard":
            self.emulator = coarse_grid.Emulator(basedir, kf=self.kf, mf=mf)
        elif emulator_class == "knot":
            self.emulator = coarse_grid.KnotEmulator(basedir, kf=self.kf, mf=mf)
        elif emulator_class == "quadratic":
            self.emulator = QuadraticEmulator(basedir, kf=self.kf, mf=mf)
        elif emulator_class == 'nCDM':
            self.emulator = coarse_grid.nCDMEmulator(basedir, kf=self.kf, mf=mf)
        else:
            raise ValueError("Emulator class not recognised")
        self.emulator.load(dumpfile=emulator_json_file)
        self.use_measured_parameters = use_measured_parameters
        self.param_limits = self.emulator.get_param_limits(include_dense=True, use_measured=self.use_measured_parameters)
        if (mean_flux == 's') or (mean_flux == 's_high_z'):
            #Add a slope to the parameter limits
            self.param_limits = np.vstack([t0_slope, self.param_limits])
            #Shrink param limits t0 so that even with
            #a slope they are within the emulator range
            self.param_limits[1,:] = t0_factor
        self.ndim = np.shape(self.param_limits)[0]
        assert np.shape(self.param_limits)[1] == 2
        print('Beginning to generate emulator at', str(datetime.now()))
        if optimise_GP:
            if emulator_class == 'nCDM':
                self.gpemu = self.emulator.get_emulator(redshifts=self.data_redshifts,
                                pixel_resolution_km_s=self.pixel_resolution_km_s,
                                use_measured_parameters=use_measured_parameters,
                                redshift_dependent_parameters=redshift_dependent_parameters,
                                k_max_emulated_h_Mpc=self._get_k_max_emulated_h_Mpc())
            else:
                self.gpemu = self.emulator.get_emulator(max_z=max_z, use_measured_parameters=use_measured_parameters,
                                                        redshift_dependent_parameters=redshift_dependent_parameters)
        print('Finished generating emulator at', str(datetime.now()))

    def _get_k_max_emulated_h_Mpc(self):
        """Calculate the maximum comoving wavenumber (in h/Mpc) that needs to be emulated"""
        omega_m_index = self.emulator._get_parameter_index_number('omega_m', include_mean_flux=False)
        omega_m_max = self.emulator.get_param_limits(include_dense=False)[omega_m_index, 1]
        k_max = np.max(self.kf) * flux_power.velocity_factor(np.max(self.zout), omega_m_max)
        print('k_max_emulated_h_Mpc =', k_max, np.max(self.kf), np.max(self.zout), omega_m_max)
        return k_max

    def log_gaussian_prior(self, parameter_vector, parameter_names, means, standard_deviations):
        """The natural logarithm of an un-normalised (multi-variate) Gaussian prior distribution"""
        mean_vector = np.ones_like(parameter_vector)
        inverse_variance_vector = np.zeros_like(parameter_vector)
        for i, parameter_name in enumerate(parameter_names):
            parameter_index_number = self.emulator._get_parameter_index_number(parameter_name,
                                        use_measured_parameters=self.use_measured_parameters,
                                        include_mean_flux_slope=self.mf_slope)
            mean_vector[parameter_index_number] = means[i]
            inverse_variance_vector[parameter_index_number] = 1. / (standard_deviations[i] ** 2)
        return -0.5 * np.sum(((parameter_vector - mean_vector) ** 2) * inverse_variance_vector)

    def log_uniform_prior(self, parameter_vector):
        """The natural logarithm of an un-normalised uniform prior distribution"""
        return 0.

    def get_predicted(self, params, use_updated_training_set=False):
        """Helper function to get the predicted flux power spectrum and error, rebinned to match the desired kbins."""
        nparams = params
        if self.mf_slope:
            # tau_0_i[z] @dtau_0 / tau_0_i[z] @[dtau_0 = 0]
            # Divided by lowest redshift case
            tau0_fac = mflux.mean_flux_slope_to_factor(self.zout, params[0], redshift_pivot=self.mean_flux_instance.redshift_pivot)
            nparams = params[1:] #Keep only t0 sampling parameter (of mean flux parameters)
        else: #Otherwise bug if choose mean_flux = 'c'
            tau0_fac = None
        # .predict should take [{list of parameters: t0; cosmo.; thermal},]
        # Here: emulating @ cosmo.; thermal; sampled t0 * [tau0_fac from above]
        predicted_nat, std_nat = self.gpemu.predict(np.array(nparams).reshape(1,-1), tau0_factors = tau0_fac, use_updated_training_set=use_updated_training_set)

        hubble_parameter_name, omega_m_parameter_name = ('hub', 'omega_m')
        if self.emulator.param_names.get(hubble_parameter_name, None) is not None:
            hindex = self.emulator._get_parameter_index_number(hubble_parameter_name, use_measured_parameters=self.use_measured_parameters)
            hubble = nparams[hindex]
            omega_m = self.emulator.omegamh2/hubble**2
        elif self.emulator.param_names.get(omega_m_parameter_name, None) is not None:
            omega_m_index = self.emulator._get_parameter_index_number(omega_m_parameter_name,
                                                               use_measured_parameters=self.use_measured_parameters)
            omega_m = nparams[omega_m_index]
            hubble = np.sqrt(self.emulator.omegamh2 / omega_m)
        else:
            raise ValueError('Neither Hubble parameter nor matter density are specified!')
        assert 0.5 < hubble < 1
        okf, predicted = flux_power.rebin_power_to_kms(kfkms=self.kf, kfmpc=self.gpemu.kf, flux_powers = predicted_nat[0], zbins=self.zout, omega_m = omega_m)
        _, std= flux_power.rebin_power_to_kms(kfkms=self.kf, kfmpc=self.gpemu.kf, flux_powers = std_nat[0], zbins=self.zout, omega_m = omega_m)
        return okf, predicted, std

    def likelihood(self, params, include_emu=True, data_power=None, use_updated_training_set=False):
        """A simple likelihood function for the Lyman-alpha forest.
        Assumes data is quadratic with a covariance matrix.
        The covariance for the emulator points is assumed to be
        completely correlated with each z bin, as the emulator
        parameters are estimated once per z bin."""
        if data_power is None:
            data_power = self.data_fluxpower
        #Set parameter limits as the hull of the original emulator.
        if np.any(params >= self.param_limits[:,1]) or np.any(params <= self.param_limits[:,0]):
            return -np.inf

        okf, predicted, std = self.get_predicted(params, use_updated_training_set=use_updated_training_set)

        nkf = int(np.size(self.kf))
        nz = np.shape(predicted)[0]
        assert nz == int(np.size(data_power)/nkf)
        #Likelihood using full covariance matrix
        chi2 = 0

        for bb in range(nz):
            idp = np.where(self.kf >= okf[bb][0])
            diff_bin = predicted[bb] - data_power[nkf*bb:nkf*(bb+1)][idp]
            std_bin = std[bb]
            bindx = np.min(idp)
            covar_bin = self.get_data_covariance(bb)[bindx:, bindx:]

            assert np.shape(np.outer(std_bin,std_bin)) == np.shape(covar_bin)
            if include_emu:
                #Assume each k bin is independent
                covar_emu = np.diag(std_bin**2)
                #Assume completely correlated emulator errors within this bin
                #covar_emu = np.outer(std_bin, std_bin)
                covar_bin += covar_emu
            icov_bin = np.linalg.inv(covar_bin)
            (_, cdet) = np.linalg.slogdet(covar_bin)
            dcd = - np.dot(diff_bin, np.dot(icov_bin, diff_bin),)/2.
            chi2 += dcd -0.5* cdet
            assert 0 > chi2 > -2**31
            assert not np.isnan(chi2)
        return chi2

    def load(self, savefile):
        """Load the chain from a savefile"""
        self.flatchain = np.loadtxt(savefile)

    def log_likelihood_marginalised_mean_flux(self, params, include_emu=True, integration_bounds='default', integration_options='gauss-legendre', verbose=True, integration_method='Quadrature'): #marginalised_axes=(0, 1)
        """Evaluate (Gaussian) likelihood marginalised over mean flux parameter axes: (dtau0, tau0)"""
        #assert len(marginalised_axes) == 2
        assert self.mf_slope
        if integration_bounds == 'default':
            integration_bounds = [list(self.param_limits[0]), list(self.param_limits[1])]

        likelihood_function = lambda dtau0, tau0: mmh.exp(self.likelihood(np.concatenate(([dtau0, tau0], params)), include_emu=include_emu))
        if integration_method == 'Quadrature':
            integration_output = mmh.quad(likelihood_function, integration_bounds[0], integration_bounds[1], method=integration_options, error=True, verbose=verbose)
        elif integration_method == 'Monte-Carlo':
            integration_output = (self._do_Monte_Carlo_marginalisation(likelihood_function, n_samples=integration_options),)
        print(integration_output)
        return float(mmh.log(integration_output[0]))

    def _do_Monte_Carlo_marginalisation(self, function, n_samples=6000):
        """Marginalise likelihood by Monte-Carlo integration"""
        random_samples = self.param_limits[:2, 0, np.newaxis] + (self.param_limits[:2, 1, np.newaxis] - self.param_limits[:2, 0, np.newaxis]) * npr.rand(2, n_samples)
        function_sum = 0.
        for i in range(n_samples):
            print('Likelihood function evaluation number =', i + 1)
            function_sum += function(random_samples[0, i], random_samples[1, i])
        volume_factor = (self.param_limits[0, 1] - self.param_limits[0, 0]) * (self.param_limits[1, 1] - self.param_limits[1, 0])
        return volume_factor * function_sum / n_samples

    def get_data_covariance(self, zbin):
        """Get the covariance matrix error."""
        #Redshifts
        lyman_data_redshifts = self.lyman_data_instance.get_redshifts()
        #Fix maximum redshift bug
        lyman_data_redshifts = lyman_data_redshifts[lyman_data_redshifts <= self.max_z]
        #Important assertion
        npt.assert_allclose(lyman_data_redshifts, self.zout, atol=1.e-16)
        #print('SDSS redshifts are', lyman_data_redshifts)
        if zbin < 0:
            covar_bin = self.lyman_data_instance.get_covar(lyman_data_redshifts)
        else:
            covar_bin = self.lyman_data_instance.get_covar(lyman_data_redshifts[zbin])
        print(covar_bin.shape)
        return covar_bin

    def log_posterior(self, parameter_vector, prior_function='uniform', include_emulator_error=True):
        """Evaluate the natural logarithm of the posterior distribution"""
        if prior_function == 'uniform':
            prior_function = self.log_uniform_prior
        return self.likelihood(parameter_vector, include_emu=include_emulator_error) + prior_function(parameter_vector)

    def do_sampling(self, savefile, datadir, nwalkers=150, burnin=3000, nsamples=3000, prior_function='uniform',
                    while_loop=True, include_emulator_error=True, maxsample=20, n_threads=1):
        """Initialise and run emcee."""
        pnames = self.emulator.print_pnames(use_measured_parameters=self.use_measured_parameters)
        #Load the data directory
        self.data_fluxpower = load_data(datadir, kf=self.kf, max_z=self.max_z, redshifts=self.data_redshifts,
                                        pixel_resolution_km_s=self.pixel_resolution_km_s, t0=self.t0_training_value,
                                        mean_flux_model=self.mean_flux_model)
        #Set up mean flux
        if self.mf_slope:
            pnames = np.concatenate((np.array([['dtau0',r'd\tau_0'],]), pnames), axis=0)
        with open(savefile+"_names.txt",'w') as ff:
            for pp in pnames:
                ff.write("%s %s\n" % tuple(pp))
        #Limits: we need to hard-prior to the volume of our emulator.
        pr = (self.param_limits[:,1]-self.param_limits[:,0])
        #Priors are assumed to be in the middle.
        cent = (self.param_limits[:,1]+self.param_limits[:,0])/2.
        p0 = [cent+2*pr/16.*np.random.rand(self.ndim)-pr/16. for _ in range(nwalkers)]
        assert np.all([np.isfinite(self.log_posterior(pp, prior_function=prior_function, include_emulator_error=include_emulator_error)) for pp in p0])
        emcee_sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_posterior,
                            kwargs={'prior_function': prior_function, 'include_emulator_error': include_emulator_error},
                            threads=n_threads)
        pos, _, _ = emcee_sampler.run_mcmc(p0, burnin)
        #Check things are reasonable
        print('The fraction of proposed steps that were accepted =', emcee_sampler.acceptance_fraction)
        #assert np.all(emcee_sampler.acceptance_fraction > 0.01)
        emcee_sampler.reset()
        self.cur_results = emcee_sampler
        gr = 10.
        count = 0
        while np.any(gr > 1.01) and count < maxsample:
            emcee_sampler.run_mcmc(pos, nsamples)
            gr = gelman_rubin(emcee_sampler.chain)
            print("Total samples:",nsamples," Gelman-Rubin: ",gr)
            np.savetxt(savefile, emcee_sampler.flatchain)
            count += 1
            if while_loop is False:
                break
        self.flatchain = emcee_sampler.flatchain
        return emcee_sampler

    def new_parameter_limits(self, confidence=0.99, include_dense=False):
        """Find a square region which includes coverage of the parameters in each direction, for refinement.
        Confidence must be 0.68, 0.95 or 0.99."""
        #Use the marginalised distributions to find the square region.
        #If there are strong degeneracies this will be very inefficient.
        #We could rotate the parameters here,
        #but ideally we would do that before running the coarse grid anyway.
        #Get marginalised statistics.
        limits = np.percentile(self.flatchain, [100-100*confidence, 100*confidence], axis=0).T
        #Discard dense params
        ndense = len(self.emulator.mf.dense_param_names)
        if self.mf_slope:
            ndense+=1
        if include_dense:
            ndense = 0
        lower = limits[ndense:,0]
        upper = limits[ndense:, 1]
        assert np.all(lower < upper)
        new_par = limits[ndense:,:]
        return new_par

    def get_covar_det(self, params, include_emu):
        """Get the determinant of the covariance matrix.for certain parameters"""
        if np.any(params >= self.param_limits[:,1]) or np.any(params <= self.param_limits[:,0]):
            return -np.inf
        lyman_data_redshifts = self.lyman_data_instance.get_redshifts()
        #Fix maximum redshift bug
        lyman_data_redshifts = lyman_data_redshifts[lyman_data_redshifts <= self.max_z]
        nz = lyman_data_redshifts.size
        if include_emu:
            okf, _, std = self.get_predicted(params)
        detc = 1
        for bb in range(nz):
            covar_bin = self.lyman_data_instance.get_covar(lyman_data_redshifts[bb])
            if include_emu:
                idp = np.where(self.kf >= okf[bb][0])
                std_bin = std[bb]
                #Assume completely correlated emulator errors within this bin
                covar_emu = np.outer(std_bin, std_bin)
                covar_bin[idp,idp] += covar_emu
            _, det_bin = np.linalg.slogdet(covar_bin)
            #We have a block diagonal covariance
            detc *= det_bin
        return detc

    def refine_metric(self, params):
        """This evaluates the 'refinement metric':
           the extent to which the emulator error dominates the covariance.
           The idea is that when it is > 1, refinement is necessary"""
        detnoemu = self.get_covar_det(params, False)
        detemu = self.get_covar_det(params, True)
        return detemu/detnoemu

    def _get_emulator_error_averaged_mean_flux(self, params, use_updated_training_set=False):
        """Get the emulator error having averaged over the mean flux parameter axes: (dtau0, tau0)"""
        n_samples = 10
        emulator_error_total = 0.
        for dtau0 in np.linspace(self.param_limits[0, 0], self.param_limits[0, 1], num=n_samples):
            for tau0 in np.linspace(self.param_limits[1, 0], self.param_limits[1, 1], num=n_samples):
                _,_,std = self.get_predicted(np.concatenate([[dtau0, tau0], params]),use_updated_training_set=use_updated_training_set)
                emulator_error_total += std
        return emulator_error_total / (n_samples ** 2)

    def _get_GP_UCB_exploitation_term(self, objective_function, exploitation_weight=1.):
        """Evaluate the exploitation term of the GP-UCB acquisition function"""
        return objective_function * exploitation_weight

    def _get_GP_UCB_exploration_term(self, data_vector_emulator_error, n_emulated_params, iteration_number=1, delta=0.5, nu=1.):
        """Evaluate the exploration term of the GP-UCB acquisition function"""
        exploration_weight = math.sqrt(nu * 2. * math.log((iteration_number**((n_emulated_params / 2.) + 2.)) * (math.pi**2) / 3. / delta))
        if self._inverse_covariance_full is None:
            self._inverse_covariance_full = invert_block_diagonal_covariance(self.get_data_covariance(-1), self.zout.shape[0])
        posterior_estimated_error = np.dot(data_vector_emulator_error, np.dot(self._inverse_covariance_full, data_vector_emulator_error))
        return exploration_weight * posterior_estimated_error

    def acquisition_function_GP_UCB(self, params, iteration_number=1, delta=0.5, nu=1., exploitation_weight=1.):
        """Evaluate the GP-UCB at given parameter vector. This is an acquisition function for determining where to run
        new training simulations"""
        assert iteration_number >= 1.
        assert 0. < delta < 1.
        if self.mf_slope:
            n_emulated_params = params.shape[0] - 1
        else:
            n_emulated_params = params.shape[0]
        #exploitation_term = self.likelihood(params) * exploitation_weight #Log-posterior [weighted]

        exploitation = self._get_GP_UCB_exploitation_term(self.likelihood(params), exploitation_weight)
        _,_,std = self.get_predicted(params)
        exploration = self._get_GP_UCB_exploration_term(std, n_emulated_params, iteration_number=iteration_number, delta=delta, nu=nu)
        return exploitation + exploration

    def acquisition_function_GP_UCB_marginalised_mean_flux(self, params, iteration_number=1, delta=0.5, nu=1., exploitation_weight=1., integration_bounds='default', integration_options='gauss-legendre', use_updated_training_set=False):
        """Evaluate the GP-UCB acquisition function, having marginalised over mean flux parameter axes: (dtau0, tau0)"""
        if exploitation_weight is None:
            print('No exploitation term')
            exploitation = 0.
        else:
            exploitation = self._get_GP_UCB_exploitation_term(self.log_likelihood_marginalised_mean_flux(params, integration_bounds=integration_bounds, integration_options=integration_options), exploitation_weight=exploitation_weight)
        exploration = self._get_GP_UCB_exploration_term(self._get_emulator_error_averaged_mean_flux(params, use_updated_training_set=use_updated_training_set), params.size, iteration_number=iteration_number, delta=delta, nu=nu)
        return exploitation + exploration

    def optimise_acquisition_function(self, starting_params, optimisation_bounds='default', optimisation_method=None, iteration_number=1, delta=0.5, nu=1., exploitation_weight=1., integration_bounds='default'):
        """Find parameter vector (marginalised over mean flux parameters) at maximum of (GP-UCB) acquisition function"""
        if optimisation_bounds == 'default': #Default to prior bounds
            #optimisation_bounds = [tuple(self.param_limits[2 + i]) for i in range(starting_params.shape[0])]
            optimisation_bounds = [(1.e-7, 1. - 1.e-7) for i in range(starting_params.shape[0])] #Might get away with 1.e-7
        optimisation_function = lambda parameter_vector: -1. * self.acquisition_function_GP_UCB_marginalised_mean_flux(map_from_unit_cube(parameter_vector, self.param_limits), iteration_number=iteration_number, delta=delta, nu=nu, exploitation_weight=exploitation_weight, integration_bounds=integration_bounds)
        return spo.minimize(optimisation_function, map_to_unit_cube(starting_params, self.param_limits), method=optimisation_method, bounds=optimisation_bounds)

    def check_for_refinement(self, conf = 0.95, thresh = 1.05):
        """Crude check for refinement: check whether the likelihood is dominated by
           emulator error at the 1 sigma contours."""
        limits = self.new_parameter_limits(confidence=conf, include_dense = True)
        while True:
            #Do the check
            uref = self.refine_metric(limits[:,0])
            lref = self.refine_metric(limits[:,1])
            #This should be close to 1.
            print("up =",uref," low=",lref)
            if (uref < thresh) and (lref < thresh):
                break
            #Iterate by moving each limit 40% outwards.
            midpt = np.mean(limits, axis=1)
            limits[:,0] = 1.4*(limits[:,0] - midpt) + midpt
            limits[:,0] = np.max([limits[:,0], self.param_limits[:,0]],axis=0)
            limits[:,1] = 1.4*(limits[:,1] - midpt) + midpt
            limits[:,1] = np.min([limits[:,1], self.param_limits[:,1]],axis=0)
            if np.all(limits == self.param_limits):
                break
        return limits

    def refinement(self,nsamples,confidence=0.99):
        """Do the refinement step."""
        new_limits = self.new_parameter_limits(confidence=confidence)
        new_samples = self.emulator.build_params(nsamples=nsamples,limits=new_limits, use_existing=True)
        assert np.shape(new_samples)[0] == nsamples
        self.emulator.gen_simulations(nsamples=nsamples, samples=new_samples)

    def make_err_grid(self, i, j, samples = 30000):
        """Make an error grid"""
        ndim = np.size(self.param_limits[:,0])
        rr = lambda x : np.random.rand(ndim)*(self.param_limits[:,1]-self.param_limits[:,0]) + self.param_limits[:,0]
        rsamples = np.array([rr(i) for i in range(samples)])
        randscores = [self.refine_metric(rr) for rr in rsamples]
        grid_x, grid_y = np.mgrid[0:1:200j, 0:1:200j]
        grid_x = grid_x * (self.param_limits[i,1] - self.param_limits[i,0]) + self.param_limits[i,0]
        grid_y = grid_y * (self.param_limits[j,1] - self.param_limits[j,0]) + self.param_limits[j,0]
        grid = scipy.interpolate.griddata(rsamples[:,(i,j)], randscores,(grid_x,grid_y),fill_value = 0)
        return grid
