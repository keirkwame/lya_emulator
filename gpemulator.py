"""Building a surrogate using a Gaussian Process."""
# from datetime import datetime
import numpy as np
from latin_hypercube import map_to_unit_cube
from mean_flux import d_tau_emulated_d_tau0_sampled
from mean_flux import d_tau_emulated_d_dtau0_sampled
#Make sure that we don't accidentally
#get another backend when we import GPy.
import matplotlib
matplotlib.use('PDF')
import GPy

class MultiBinGP(object):
    """A wrapper around the emulator that constructs a separate emulator for each bin.
    Each one has a separate mean flux parameter.
    The t0 parameter fed to the emulator should be constant factors."""
    def __init__(self, *, params, kf, powers, param_limits, coreg=False):
        #Build an emulator for each redshift separately. This means that the
        #mean flux for each bin can be separated.
        self.kf = kf
        self.nk = np.size(kf)
        assert np.shape(powers)[1] % self.nk == 0
        self.nz = int(np.shape(powers)[1]/self.nk)
        self.coreg = coreg
        gp = lambda i: SkLearnGP(params=params, powers=powers[:,i*self.nk:(i+1)*self.nk], param_limits = param_limits, coreg=coreg)
        print('Number of redshifts for emulator generation =', self.nz)
        self.gps = [gp(i) for i in range(self.nz)]

    def _renormalise_parameter_array(self, params, tau0_factors, i):
        zparams = np.array(params)
        if tau0_factors is not None:
            zparams[0][0] *= tau0_factors[i]  # Multiplying t0[z] by "tau0_factors"[z]
        return zparams

    def predict(self,params, tau0_factors = None):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        std = np.zeros([1 + self.coreg*(np.shape(params)[1]-1),self.nk*self.nz])
        means = np.zeros([1,self.nk*self.nz])
        for i, gp in enumerate(self.gps): #Looping over redshifts
            #Adjust the slope of the mean flux for this bin
            zparams = self._renormalise_parameter_array(params, tau0_factors, i)
            (m, s) = gp.predict(zparams)
            means[0,i*self.nk:(i+1)*self.nk] = m
            std[:,i*self.nk:(i+1)*self.nk] = s
        return means, std

    def get_predictive_gradients(self, params, tau0_factors=None, mean_flux_slope=0., redshifts=None, pivot_redshift=2.2):
        """Get the predictive gradients of the flux power spectrum and covariance wrt. simulation parameters, evaluated
        at a given parameter value (or list of parameter values). As a function of redshift; here, if necessary, the
        emulated mean optical depth parameters are reparameterised into the sampling parameters."""
        n_params = params.shape[1]
        if tau0_factors is None:
            mean_flux_parameters = None
            redshifts = np.array([None,] * self.nz)
        else:
            n_params += 1 #We actually have two mean flux sampling parameters
            mean_flux_parameters = np.array([mean_flux_slope, params[0, 0]])

        gradient_of_mean = np.zeros((1, n_params, self.nk*self.nz))
        gradient_of_variance = np.zeros_like(gradient_of_mean)
        for i, gp in enumerate(self.gps): #Looping over redshifts
            params_renorm = self._renormalise_parameter_array(params, tau0_factors, i)
            gom, gov = gp.get_predictive_gradients(params_renorm, mean_flux_sampling_parameters=mean_flux_parameters, redshift=redshifts[i], pivot_redshift=pivot_redshift)
            start_index = i*self.nk
            end_index = (i+1)*self.nk
            gradient_of_mean[0, :, start_index: end_index] = gom
            gradient_of_variance[0, :, start_index: end_index] = gov
        return gradient_of_mean, gradient_of_variance

class SkLearnGP(object):
    """An emulator using the one in Scikit-learn.
       Parameters: params is a list of parameter vectors.
                   powers is a list of flux power spectra (same shape as params).
                   param_limits is a list of parameter limits (shape 2,params).
                   coreg is a flag to enable GPy's coregionalisation (not helpful)."""
    def __init__(self, *, params, powers,param_limits, coreg=False):
        self.params = params
        self.param_limits = param_limits
        self.intol = 3e-5
        #Should we test the built emulator?
        self._test_interp = False
        self.coreg=coreg
        #Get the flux power and build an emulator
        self._get_interp(flux_vectors=powers)
        #In case we need it, we can rescale the errors using cross-validation.
        #self.sdscale = np.mean([self._get_cv_one(powers, exclude) for exclude in range(len(self.powers))])

#     def _get_cv_one(self, powers, exclude):
#         """Get the prediction error for one point when
#         excluding that point from the emulator."""
#         self._get_interp(flux_vectors=powers, exclude=exclude)
#         test_exact = powers[exclude]
#         return self.get_predict_error(self.params[exclude], test_exact)

    def _get_interp(self, flux_vectors):
        """Build the actual interpolator."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        nparams = np.shape(self.params)[1]
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in self.params])
        #print('Normalised parameter values =', params_cube)
        #Normalise the flux vectors by the median power spectrum.
        #This ensures that the GP prior (a zero-mean input) is close to true.
        medind = np.argsort(np.mean(flux_vectors, axis=1))[np.shape(flux_vectors)[0]//2]
        self.scalefactors = flux_vectors[medind,:]
        self.paramzero = params_cube[medind,:]
        #Normalise by the median value
        normspectra = flux_vectors/self.scalefactors -1.

        #Extracting flux power vectors to disk
        #date_and_time = str(datetime.now())
        #savefile = "/Users/kwame/Simulations/emulator/training_flux_power_" + date_and_time + ".npz"
        #print("Extracting flux power vectors to disk at", date_and_time)
        #np.savez(savefile, flux_vectors, self.scalefactors, self.paramzero, medind)

        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        kernel = GPy.kern.Linear(nparams)
        print(kernel)
        kernel += GPy.kern.RBF(nparams)

        #Try rational quadratic kernel
        #kernel += GPy.kern.RatQuad(nparams)

        print(kernel)
        noutput = np.shape(normspectra)[1]
        if self.coreg and noutput > 1:
            coreg = GPy.kern.Coregionalize(input_dim=nparams,output_dim=noutput)
            kernel = kernel.prod(coreg,name='coreg.kern')

        #Set priors on hyperparameters
        #kernel.rbf.lengthscale.constrain_bounded(1.e-2, np.inf)
        #kernel.Gaussian_noise.variance.constrain_bounded(0., 1.e-9)

        self.gp = GPy.models.GPRegression(params_cube, normspectra,kernel=kernel, noise_var=1e-10)

        #Let's see if there's any output
        self.gp.optimize(messages=True) #False
        print(self.gp)
        #print('Gradients of model hyperparameters [after optimisation] =', self.gp.gradient)
        #Let's check that hyperparameter optimisation is converged
        #self.gp.optimize_restarts(num_restarts=10)
        #print(self.gp)
        #print('Gradients of model hyperparameters [after second optimisation (x 10)] =', self.gp.gradient)

        #Check we reproduce the input
        if self._test_interp:
            test,_ = self.predict(self.params[0,:].reshape(1,-1))
            worst = np.abs(test[0] / flux_vectors[0,:]-1)
            if np.max(worst) > self.intol:
                print("Bad interpolation at:",np.where(worst > np.max(worst)*0.9), np.max(worst))
                assert np.max(worst) < self.intol
            self._test_interp = False

    def predict(self, params):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in params])
        flux_predict, var = self.gp.predict(params_cube)

        #Extract gradients of posterior mean and variance wrt. parameters (hopefully not hyperparameters)
        #print('Posterior variance as output from Gaussian process (for normalised data vector) =', var)
        #print('Gradients of posterior mean and variance wrt. parameters =', self.gp.predictive_gradients(params_cube))

        mean = (flux_predict+1)*self.scalefactors
        std = np.sqrt(var) * self.scalefactors

        return mean, std

    def get_predictive_gradients(self, params, mean_flux_sampling_parameters=None, redshift=None, pivot_redshift=2.2):
        """Get the predictive gradients of the flux power spectrum and covariance wrt. simulation parameters, evaluated
        at a given parameter value (or list of parameter values).
        mean_flux_sampling_parameters: [dtau0, tau0]"""
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in params])

        d_unit_d_parameters = 1. / (self.param_limits[:, 1] - self.param_limits[:, 0])  # dims: n_params
        gradient_of_mean, gradient_of_variance = self.gp.predictive_gradients(params_cube)
        gradient_of_mean_renorm = gradient_of_mean * self.scalefactors * d_unit_d_parameters[np.newaxis, :, np.newaxis]
        # dims: 1 x n_params x len(data)
        gradient_of_variance_renorm = gradient_of_variance[:, :, np.newaxis] * (self.scalefactors**2) * d_unit_d_parameters[
                                                                                                   np.newaxis, :,
                                                                                                   np.newaxis]
        if mean_flux_sampling_parameters is not None:
            gradient_of_mean_renorm = np.insert(gradient_of_mean_renorm, 0., 0., axis=1)
            gradient_of_variance_renorm = np.insert(gradient_of_variance_renorm, 0., 0., axis=1)
            mean_flux_slope_correction = d_tau_emulated_d_dtau0_sampled(redshift, tau0=mean_flux_sampling_parameters[1], slope=mean_flux_sampling_parameters[0], pivot_redshift=pivot_redshift)
            mean_flux_amplitude_correction = d_tau_emulated_d_tau0_sampled(redshift, slope=mean_flux_sampling_parameters[0], pivot_redshift=pivot_redshift)
            gradient_of_mean_renorm[:, 0, :] = gradient_of_mean_renorm[:, 1, :] * mean_flux_slope_correction
            gradient_of_variance_renorm[:, 1, :] *= mean_flux_amplitude_correction
        return gradient_of_mean_renorm, gradient_of_variance_renorm

    def get_predict_error(self, test_params, test_exact):
        """Get the difference between the predicted GP
        interpolation and some exactly computed test parameters."""
        #Note: this is not used anywhere
        test_exact = test_exact.reshape(np.shape(test_params)[0],-1)
        predict, sigma = self.predict(test_params)
        #The transposes are because of numpy broadcasting rules only doing the last axis
        return ((test_exact - predict).T/np.sqrt(sigma)).T
