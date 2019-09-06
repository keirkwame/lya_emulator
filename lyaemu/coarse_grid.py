"""Generate a coarse grid for the emulator and test it"""
from __future__ import print_function
import os
import os.path
import shutil
import glob
import string
import math
import json
import numpy as np
import h5py
from .SimulationRunner.SimulationRunner import lyasimulation
from .SimulationRunner.SimulationRunner import clusters
from . import latin_hypercube
from . import flux_power
from . import lyman_data
from . import gpemulator
from .mean_flux import ConstMeanFlux
from .mean_flux import ConstMeanFluxHighRedshift

def get_latex(key):
    """Get a latex name if it exists, otherwise return the key."""
    #Names for pretty-printing some parameters in Latex
    print_names = { 'ns': r'n_\mathrm{s}', 'As': r'A_\mathrm{s}', 'heat_slope': r'H_\mathrm{S}', 'heat_amp': r'H_\mathrm{A}', 'hub':'h', 'omega_m': r'\Omega_\mathrm{m}', 'alpha': r'\alpha', 'beta': r'\beta', 'gamma': r'\gamma', 'T_0_z_5.0': r'T_0(z=5)', 'T_0_z_4_6': r'T_0(z=4.6)', 'T_0_z_4_2': r'T_0(z=4.2)', 'gamma_z_5.0': r'\gamma(z=5)', 'gamma_z_4.6': r'\gamma(z=4.6)','gamma_z_4.2': r'\gamma(z=4.2)', 'u_0_z_5.0': r'u_0(z=5)', 'u_0_z_4.6': r'u_0(z=4.6)', 'u_0_z_4.2': r'u_0(z=4.2)', 'tau0':r'\tau_0', 'dtau0':r'd\tau_0'}
    try:
        return print_names[key]
    except KeyError:
        return key

class Emulator:
    """Small wrapper class to store parameter names and limits, generate simulations and get an emulator.
    """
    def __init__(self, basedir, param_names=None, param_limits=None, kf=None, mf=None, z=None, omegamh2=0.1199):
        if param_names is None:
            self.param_names = {'ns':0, 'As':1, 'heat_slope':2, 'heat_amp':3, 'hub':4}
        else:
            self.param_names = param_names
        if param_limits is None:
            self.param_limits = np.array([[0.8, 0.995], [1.2e-09, 2.6e-09], [-0.7, 0.1], [0.4, 1.4], [0.65, 0.75]])
        else:
            self.param_limits = param_limits
        if mf is None:
            self.mf = ConstMeanFlux(None)
        else:
            self.mf = mf

        if (kf is None) or (z is None):
            data_instance = lyman_data.BOSSData()
            if kf is None:
                kf = data_instance.get_kf()
            if z is None:
                z = data_instance.get_redshifts()
        self.kf = kf
        self.redshifts = z

        #We fix omega_m h^2 = 0.1199 (Planck best-fit) and vary omega_m and h^2 to match it.
        #h^2 itself has little effect on the forest.
        self.omegamh2 = omegamh2
        #Corresponds to omega_m = (0.23, 0.31) which should be enough.

        self.maxz = np.max(z)
        self.set_maxk()

        self.sample_params = []
        self.basedir = os.path.expanduser(basedir)
        if not os.path.exists(basedir):
            os.mkdir(basedir)

        self.measured_param_names = {}
        self.measured_param_limits = 'None'
        self.measured_sample_params = 'None'
        self.remove_simulation_params = np.array([], dtype=np.int)
        self.redshift_sensitivity = 'None'

    def set_maxk(self):
        """Get the maximum k in Mpc/h that we will need."""
        #Corresponds to omega_m = (0.23, 0.31) which should be enough.

        #Maximal velfactor: the h dependence cancels but there is an omegam
        if self.param_names.get('omega_m', None) is None:
            minhub = self.param_limits[self.param_names['hub'], 0]
            omegam_max = self.omegamh2 / (minhub ** 2)
        else:
            omegam_max = self.param_limits[self.param_names['omega_m'], 1]
        velfac = lambda a: a * 100. * np.sqrt((omegam_max / (a ** 3)) + (1 - omegam_max))

        #Maximum k value to use in comoving Mpc/h.
        #Comes out to k ~ 5, which is a bit larger than strictly necessary.
        self.maxk = np.max(self.kf) * velfac(1/(1+self.maxz)) * 2

    def build_dirname(self,params, include_dense=False, strsz=3):
        """Make a directory name for a given set of parameter values"""
        ndense = include_dense * len(self.mf.dense_param_names)
        parts = ['',]*(len(self.param_names) + ndense)
        #Transform the dictionary into a list of string parts,
        #sorted in the same way as the parameter array.
        fstr = "%."+str(strsz)+"g"
        for nn,val in self.mf.dense_param_names.items():
            parts[val] = nn+fstr % params[val]
        for nn,val in self.param_names.items():
            parts[ndense+val] = nn+fstr % params[ndense+val]
        name = ''.join(str(elem) for elem in parts)
        return name

    def print_pnames(self, use_measured_parameters=False):
        """Get parameter names for printing"""
        n_latex = []
        sort_names = sorted(list(self.mf.dense_param_names.items()), key=lambda k:(k[1],k[0]))
        for key, _ in sort_names:
            n_latex.append((key, get_latex(key)))

        sort_names = sorted(list(self.param_names.items()), key=lambda k:(k[1],k[0]))
        for key, _ in sort_names:
            n_latex.append((key, get_latex(key)))

        if use_measured_parameters:
            sort_names = sorted(list(self.measured_param_names.items()), key=lambda k:(k[1],k[0]))
            for key, _ in sort_names:
                n_latex.append((key, get_latex(key)))
            n_latex = np.array(n_latex)
            n_latex = np.delete(n_latex, self.remove_simulation_params + len(self.mf.dense_param_names), axis=0)
        else:
            n_latex = np.array(n_latex)

        return n_latex

    def _fromarray(self):
        """Convert the data stored as lists back to arrays."""
        for arr in self.really_arrays:
            self.__dict__[arr] = np.array(self.__dict__[arr])
        self.really_arrays = []

    def _recon_one(self, pdir):
        """Get the parameters of a simulation from the SimulationICs.json file"""
        with open(os.path.join(pdir, "SimulationICs.json"), 'r') as jsin:
            sics = json.load(jsin)
        ev = np.zeros_like(self.param_limits[:,0])
        pn = self.param_names
        ev[pn['heat_slope']] = sics["rescale_slope"]
        ev[pn['heat_amp']] = sics["rescale_amp"]
        if self.param_names.get('hub', None) is not None:
            ev[pn['hub']] = sics["hubble"]
        if self.param_names.get('omega_m', None) is not None:
            ev[pn['omega_m']] = sics["omega_m"]
        ev[pn['ns']] = sics["ns"]
        wmap = sics["scalar_amp"]
        #Convert pivot of the scalar amplitude from amplitude
        #at 8 Mpc (k = 0.78) to pivot scale of 0.05
        conv = (0.05/(2*math.pi/8.))**(sics["ns"]-1.)
        ev[pn['As']] = wmap / conv
        return ev

    def reconstruct(self):
        """Reconstruct the parameters of an emulator by loading the parameters of each simulation in turn."""
        dirs = glob.glob(os.path.join(self.basedir, "*"))
        self.sample_params = np.array([self._recon_one(pdir) for pdir in dirs])
        assert np.shape(self.sample_params) == (len(dirs), np.size(self.param_limits[:,0]))

    def dump_measured_parameters(self, measured_parameter_names, measured_sample_parameters,
                                 remove_simulation_parameters, measured_parameter_limits='default',
                                 redshift_sensitivity='None', dumpfile='emulator_params.json'):
        """Dump measured parameters [e.g., T_0(z); gamma(z); u_0(z)] to a textfile"""
        measured_parameter_indices = np.arange(len(self.measured_param_names), len(self.measured_param_names) + len(measured_parameter_names))
        for i, measured_parameter_name in enumerate(measured_parameter_names):
            self.measured_param_names.update({measured_parameter_name: int(measured_parameter_indices[i])})

        if self.measured_sample_params == 'None':
            self.measured_sample_params = measured_sample_parameters
        else:
            self.measured_sample_params = np.concatenate((self.measured_sample_params, measured_sample_parameters), axis=1)

        if measured_parameter_limits == 'default':
            measured_parameter_minima = np.min(measured_sample_parameters, axis=0).reshape(-1, 1)
            measured_parameter_maxima = np.max(measured_sample_parameters, axis=0).reshape(-1, 1)
            measured_parameter_limits = np.concatenate((measured_parameter_minima, measured_parameter_maxima), axis=1)
        if self.measured_param_limits == 'None':
            self.measured_param_limits = measured_parameter_limits
        else:
            self.measured_param_limits = np.concatenate((self.measured_param_limits, measured_parameter_limits), axis=0)

        self.remove_simulation_params = np.sort(np.concatenate((self.remove_simulation_params, remove_simulation_parameters))).astype(np.int)
        self.redshift_sensitivity = redshift_sensitivity

        self.dump(dumpfile=dumpfile)

    def dump(self, dumpfile="emulator_params.json"):
        """Dump parameters to a textfile."""
        #Backup existing parameter file
        fdump = os.path.join(self.basedir, dumpfile)
        if os.path.exists(fdump):
            backup = fdump + ".backup"
            r=1
            while os.path.exists(backup):
                backup = fdump + "_r"+str(r)+".backup"
                r+=1
            shutil.move(fdump, backup)
        #Arrays can't be serialised so convert them back and forth to lists
        self.really_arrays = []
        mf = self.mf
        self.mf = []
        for nn, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                self.__dict__[nn] = val.tolist()
                self.really_arrays.append(nn)
        with open(fdump, 'w') as jsout:
            json.dump(self.__dict__, jsout)
        self._fromarray()
        self.mf = mf

    def load(self,dumpfile="emulator_params.json"):
        """Load parameters from a textfile."""
        kf = self.kf
        mf = self.mf
        real_basedir = self.basedir
        with open(os.path.join(real_basedir, dumpfile), 'r') as jsin:
            indict = json.load(jsin)
        self.__dict__ = indict
        self._fromarray()
        self.kf = kf
        self.mf = mf
        self.basedir = real_basedir
        self.set_maxk()

    def get_outdir(self, pp, strsz=3):
        """Get the simulation output directory path for a parameter set."""
        return os.path.join(os.path.join(self.basedir, self.build_dirname(pp, strsz=strsz)),"output")

    def get_parameters(self):
        """Get the list of parameter vectors in this emulator."""
        return self.sample_params

    def get_measured_parameters(self):
        """Get the list of measured parameter vectors in this emulator"""
        return self.measured_sample_params

    def get_combined_params(self):
        """Get the list of parameter vectors (combined input and measured) in this emulator"""
        input_parameters = np.delete(self.get_parameters(), self.remove_simulation_params, axis=1)
        return np.concatenate((input_parameters, self.get_measured_parameters()), axis=1)

    def build_params(self, nsamples,limits = None, use_existing=False):
        """Build a list of directories and parameters from a hypercube sample"""
        if limits is None:
            limits = self.param_limits
        #Consider only prior points inside the limits
        prior_points = None
        if use_existing:
            ii = np.where(np.all(self.sample_params > limits[:,0],axis=1)*np.all(self.sample_params < limits[:,1],axis=1))
            prior_points = self.sample_params[ii]
        return latin_hypercube.get_hypercube_samples(limits, nsamples,prior_points=prior_points)

    def gen_simulations(self, nsamples, npart=256.,box=40,samples=None):
        """Initialise the emulator by generating simulations for various parameters."""
        if nsamples is not None:
            if len(self.sample_params) == 0:
                self.sample_params = self.build_params(nsamples)
            if samples is None:
                samples = self.sample_params
            else:
                self.sample_params = np.vstack([self.sample_params, samples])
        else:
            self.sample_params = samples
        #Generate ICs for each set of parameter inputs
        for ev in samples:
            self._do_ic_generation(ev, npart, box)
        self.dump()

    def _do_ic_generation(self,ev,npart,box):
        """Do the actual IC generation."""
        outdir = os.path.join(self.basedir, self.build_dirname(ev))
        pn = self.param_names
        rescale_slope = ev[pn['heat_slope']]
        rescale_amp = ev[pn['heat_amp']]
        hub = ev[pn['hub']]
        #Convert pivot of the scalar amplitude from amplitude
        #at 8 Mpc (k = 0.78) to pivot scale of 0.05
        ns = ev[pn['ns']]
        wmap = (0.05/(2*math.pi/8.))**(ns-1.) * ev[pn['As']]
        ss = lyasimulation.LymanAlphaSim(outdir=outdir, box=box,npart=npart, ns=ns, scalar_amp=wmap, rescale_gamma=True, rescale_slope = rescale_slope, redend=2.2, rescale_amp = rescale_amp, hubble=hub, omega0=self.omegamh2/hub**2, omegab=0.0483,unitary=True)
        try:
            ss.make_simulation()
            fpfile = os.path.join(os.path.dirname(__file__),"flux_power.py")
            shutil.copy(fpfile, os.path.join(outdir, "flux_power.py"))
            ss._cluster.generate_spectra_submit(outdir)
        except RuntimeError as e:
            print(str(e), " while building: ",outdir)

    def get_param_limits(self, include_dense=True, use_measured=False):
        """Get the reprocessed limits on the parameters for the likelihood."""
        plimits = self.param_limits

        if use_measured:
            plimits = np.delete(plimits, self.remove_simulation_params, axis=0)
            plimits = np.vstack((plimits, self.measured_param_limits))

        if include_dense:
            dlim = self.mf.get_limits()
            if dlim is not None:
                #Dense parameters go first as they are 'slow'
                plimits = np.vstack([dlim, plimits])
                assert np.shape(plimits)[1] == 2

        return plimits

    def get_nsample_params(self):
        """Get the number of sparse parameters, those sampled by simulations."""
        return np.shape(self.param_limits)[0]

    def _get_fv(self, pp,myspec):
        """Helper function to get a single flux vector."""
        di = self.get_outdir(pp, strsz=3)
        if not os.path.exists(di):
            di = self.get_outdir(pp, strsz=2)
        powerspectra = myspec.get_snapshot_list(base=di)
        return powerspectra

    def get_emulator(self, max_z=4.2, use_measured_parameters=False, redshift_dependent_parameters=False):
        """ Build an emulator for the desired k_F and our simulations.
            kf gives the desired k bins in s/km.
            Mean flux rescaling is handled (if mean_flux=True) as follows:
            1. A set of flux power spectra are generated for every one of a list of possible mean flux values.
            2. Each flux power spectrum in the set is rescaled to the same mean flux.
            3.
        """
        gp = self._get_custom_emulator(emuobj=None, max_z=max_z, use_measured_parameters=use_measured_parameters,
                                       redshift_dependent_parameters=redshift_dependent_parameters)
        return gp

    def get_flux_vectors(self, max_z=4.2, kfunits="kms", redshifts=None, pixel_resolution_km_s='default',
                         use_measured_parameters=False, fix_mean_flux_samples=False):
        """Get the desired flux vectors and their parameters"""
        pvals = self.get_parameters()
        nparams = np.shape(pvals)[1]
        nsims = np.shape(pvals)[0]
        assert nparams == len(self.param_names)
        myspec = flux_power.MySpectra(max_z=max_z, max_k=self.maxk, redshifts=redshifts, pixel_resolution_km_s=pixel_resolution_km_s)
        aparams = pvals
        #Note this gets tau_0 as a linear scale factor from the observed power law
        dpvals = self.mf.get_params()
        nuggets = np.zeros_like(pvals[:,0])
        #Savefile prefix
        mfc = "cc"
        if dpvals is not None:
            #Add a small offset to the mean flux in each simulation to improve support
            if not fix_mean_flux_samples:
                nuggets = np.arange(nsims)/nsims * (dpvals[-1] - dpvals[0])/(np.size(dpvals)+1)
            newdp = dpvals[0] + (dpvals-dpvals[0]) / (np.size(dpvals)+1) * np.size(dpvals)
            #Make sure we don't overflow the parameter limits
            assert (newdp[-1] + nuggets[-1] < dpvals[-1]) and (newdp[0] + nuggets[0] >= dpvals[0])
            dpvals = newdp
            aparams = np.array([np.concatenate([dp+nuggets[i],pvals[i]]) for dp in dpvals for i in range(nsims)])
            mfc = "mf10"
        print('dpvals =', dpvals)
        print('nuggets =', nuggets)
        print('mean_flux =', self.mf.get_mean_flux(myspec.zout, params=dpvals))
        try:
            kfmpc, kfkms, flux_vectors = self.load_flux_vectors(aparams, mfc=mfc)
        except (AssertionError, OSError):
            print("Could not load flux vectors, regenerating from disc")
            powers = [self._get_fv(pp, myspec) for pp in pvals]
            mef = lambda pp: self.mf.get_mean_flux(myspec.zout, params=pp)[0]
            if dpvals is not None:
                flux_vectors = np.array([powers[i].get_power_native_binning(mean_fluxes = mef(dp+nuggets[i])) for dp in dpvals for i in range(nsims)])
                #'natively' binned k values in km/s units as a function of redshift
                kfkms = [ps.get_kf_kms() for _ in dpvals for ps in powers]
            else:
                flux_vectors = np.array([powers[i].get_power_native_binning(mean_fluxes = mef(dpvals)) for i in range(nsims)])
                #'natively' binned k values in km/s units as a function of redshift
                kfkms = [ps.get_kf_kms() for ps in powers]
            #Same in all boxes
            kfmpc = powers[0].kf
            assert np.all(np.abs(powers[0].kf/ powers[-1].kf-1) < 1e-6)
            self.save_flux_vectors(aparams, kfmpc, kfkms, flux_vectors, mfc=mfc)

        if use_measured_parameters:
            if dpvals is not None:
                index_adjustment = 1
                measured_parameters = np.repeat(self.measured_sample_params, dpvals.shape[0], axis=0)
            else:
                index_adjustment = 0
                measured_parameters = self.measured_sample_params
            aparams = np.delete(aparams, self.remove_simulation_params + index_adjustment, axis=1)
            aparams = np.concatenate((aparams, measured_parameters), axis=1)

        assert np.shape(flux_vectors)[0] == np.shape(aparams)[0]
        if kfunits == "kms":
            kf = kfkms
        else:
            kf = kfmpc
        return aparams, kf, flux_vectors

    def save_flux_vectors(self, aparams, kfmpc, kfkms, flux_vectors, mfc="mf", savefile="emulator_flux_vectors.hdf5"):
        """Save the flux vectors and parameters to a file, which is the only thing read on reload."""
        save = h5py.File(os.path.join(self.basedir, mfc+"_"+savefile), 'w')
        save.attrs["classname"] = str(self.__class__)
        save["params"] = aparams
        save["flux_vectors"] = flux_vectors
        #Save in both km/s and Mpc/h units.
        save["kfkms"] = kfkms
        save["kfmpc"] = kfmpc
        save.close()

    def load_flux_vectors(self, aparams, mfc="mf", savefile="emulator_flux_vectors.hdf5"):
        """Save the flux vectors and parameters to a file, which is the only thing read on reload."""
        load = h5py.File(os.path.join(self.basedir, mfc+"_"+savefile), 'r')
        inparams = np.array(load["params"])
        flux_vectors = np.array(load["flux_vectors"])
        kfkms = np.array(load["kfkms"])
        kfmpc = np.array(load["kfmpc"])
        name = str(load.attrs["classname"])
        load.close()
        assert name.split(".")[-1] == str(self.__class__).split(".")[-1]
        assert np.shape(inparams) == np.shape(aparams)
        assert np.all(inparams - aparams < 1e-3)
        return kfmpc, kfkms, flux_vectors

    def _get_custom_emulator(self, *, emuobj, max_z=4.2, redshifts=None, pixel_resolution_km_s='default',
                             use_measured_parameters=False, redshift_dependent_parameters=False):
        """Helper to allow supporting different emulators."""
        aparams, kf, flux_vectors = self.get_flux_vectors(max_z=max_z, kfunits="mpc", redshifts=redshifts,
                                        pixel_resolution_km_s=pixel_resolution_km_s,
                                        use_measured_parameters=use_measured_parameters)
        plimits = self.get_param_limits(include_dense=True, use_measured=use_measured_parameters)
        if redshift_dependent_parameters:
            redshift_sensitivity = self.redshift_sensitivity
        else:
            redshift_sensitivity = np.ones((int(flux_vectors.shape[1] / kf.shape[0]), aparams.shape[1]), dtype=np.bool)
        gp = gpemulator.MultiBinGP(params=aparams, kf=kf, powers = flux_vectors, param_limits = plimits,
                                   singleGP=emuobj, redshift_sensitivity=redshift_sensitivity)
        return gp


class KnotEmulator(Emulator):
    """Specialise parameter class for an emulator using knots.
    Thermal parameters turned off."""
    def __init__(self, basedir, nknots=4, kf=None, mf=None):
        param_names = {'heat_slope':nknots, 'heat_amp':nknots+1, 'hub':nknots+2}
        #Assign names like AA, BB, etc.
        for i in range(nknots):
            param_names[string.ascii_uppercase[i]*2] = i
        self.nknots = nknots
        param_limits = np.append(np.repeat(np.array([[0.6,1.5]]),nknots,axis=0),[[-0.5, 0.5],[0.5,1.5],[0.65,0.75]],axis=0)
        super().__init__(basedir=basedir, param_names = param_names, param_limits = param_limits, kf=kf, mf=mf)
        #Linearly spaced knots in k space:
        #these do not quite hit the edges of the forest region, because we want some coverage over them.
        self.knot_pos = np.linspace(0.15, 1.5,nknots)
        #Used for early iterations.
        #self.knot_pos = [0.15,0.475,0.75,1.19]

    def _do_ic_generation(self,ev,npart,box):
        """Do the actual IC generation."""
        outdir = os.path.join(self.basedir, self.build_dirname(ev))
        pn = self.param_names
        rescale_slope = ev[pn['heat_slope']]
        rescale_amp = ev[pn['heat_amp']]
        hub = ev[pn['hub']]
        ss = lyasimulation.LymanAlphaKnotICs(outdir=outdir, box=box,npart=npart, knot_pos = self.knot_pos,
                                             knot_val=ev[0:self.nknots],hubble=hub, rescale_gamma=True, redend=2.2,
                                             rescale_slope = rescale_slope, rescale_amp = rescale_amp,
                                             omega0=self.omegamh2/hub**2, omegab=0.0483,unitary=True)
        try:
            ss.make_simulation()
        except RuntimeError as e:
            print(str(e), " while building: ",outdir)


class nCDMEmulator(Emulator):
    """Specialise parameter class for an emulator for nCDM models. Defaults to Planck 2018 Omega_m h**2 & Omega_b."""
    def __init__(self, basedir, kf=None, mf=None, z=None, omegamh2=0.14345, omegab=0.04950):
        param_names = {'ns': 0, 'As': 1, 'heat_slope': 2, 'heat_amp': 3, 'omega_m': 4, 'alpha': 5, 'beta': 6, 'gamma': 7, 'z_rei': 8, 'T_rei': 9}
        param_limits = np.array([[0.9, 0.995], [1.2e-9, 2.5e-9], [-1.3, 0.7], [0.05, 3.5], [0.26, 0.33],
                                 [0., 0.1], [1., 10.], [-10., 0.], [6., 15.], [1.5e+4, 4.e+4]])
        if (kf is None) or (z is None):
            data_instance = lyman_data.BoeraData()
            if kf is None:
                kf = data_instance.get_kf()
            if z is None:
                z = data_instance.get_redshifts()
        if mf is None:
            mf = ConstMeanFluxHighRedshift(None)
        self.omegab = omegab
        self._scalar_pivot_scale_ratio = 0.05 / 2. #Ratio between CMB and Lyman-a forest scalar power spectrum pivots
        super().__init__(basedir=basedir, param_names=param_names, param_limits=param_limits, kf=kf, mf=mf, z=z, omegamh2=omegamh2)

    def _do_ic_generation(self, ev, npart, box): #To be modified!!!
        """Generate initial conditions"""
        outdir = os.path.join(self.basedir, self.build_dirname(ev))
        pn = self.param_names
        rescale_slope = ev[pn['heat_slope']]
        rescale_amp = ev[pn['heat_amp']]
        z_rei = ev[pn['z_rei']]
        T_rei = ev[pn['T_rei']]
        alpha = ev[pn['alpha']]
        beta = ev[pn['beta']]
        gamma = ev[pn['gamma']]
        omega_m = ev[pn['omega_m']]
        hubble = np.sqrt(self.omegamh2/omega_m)
        # Convert pivot of the scalar amplitude from amplitude at k = 2 [appropriate pivot scale for high-resolution
        # data] to pivot scale of 0.05
        ns = ev[pn['ns']]
        wmap = self._scalar_pivot_scale_ratio ** (ns - 1.) * ev[pn['As']]
        ss = lyasimulation.LymanAlphaNCDMSim(outdir=outdir, box=box, npart=npart, alpha=alpha, beta=beta, gamma=gamma,
                                             ns=ns, scalar_amp=wmap, rescale_gamma=True, rescale_slope=rescale_slope,
                                             rescale_amp=rescale_amp, z_rei=z_rei, delta_T_HI_K=T_rei,
                                             hubble=hubble, omega0=omega_m, omegab=self.omegab,
                                             unitary=True, cluster_class=clusters.HypatiaClass,
                                             MPGadget_directory=os.path.expanduser("~/Software/MP-Gadget-master/"))
        try:
            ss.make_simulation(do_build=False)
            fpfile = os.path.join(os.path.dirname(__file__), "flux_power.py")
            shutil.copy(fpfile, os.path.join(outdir, "flux_power.py"))
            extra_options = '-pixel_resolution_km_s 1.0 -redshifts'
            snapshot_numbers = np.zeros_like(self.redshifts)
            for i, redshift in enumerate(self.redshifts):
                extra_options += ' %.3f'%redshift
                snapshot_numbers[i] = ss.get_snapshot_number(redshift)
            ss._cluster.generate_spectra_submit(outdir, extra_options=extra_options)
            ss._cluster.generate_GenPK_submit(outdir, os.path.expanduser('~/Software/GenPK'), snapshot_numbers,
                                              box*1000., hubble)
        except RuntimeError as e:
            print(str(e), " while building: ", outdir)

    def _recon_one(self, pdir):
        """Get the parameters of a simulation from the SimulationICs.json file"""
        with open(os.path.join(pdir, "SimulationICs.json"), 'r') as jsin:
            sics = json.load(jsin)
        ev = np.zeros_like(self.param_limits[:,0])
        pn = self.param_names
        ev[pn['alpha']] = sics['alpha']
        ev[pn['beta']] = sics['beta']
        ev[pn['gamma']] = sics['gamma']
        ev[pn['heat_slope']] = sics["rescale_slope"]
        ev[pn['heat_amp']] = sics["rescale_amp"]
        ev[pn['z_rei']] = sics['z_rei']
        ev[pn['T_rei']] = sics['T_rei']
        if self.param_names.get('hub', None) is not None:
            ev[pn['hub']] = sics["hubble"]
        if self.param_names.get('omega_m', None) is not None:
            ev[pn['omega_m']] = sics["omega_m"]
        ev[pn['ns']] = sics["ns"]
        wmap = sics["scalar_amp"]
        #Convert pivot of the scalar amplitude from amplitude
        #at 8 Mpc (k = 0.78) to pivot scale of 0.05
        conv = self._scalar_pivot_scale_ratio**(sics["ns"]-1.)
        ev[pn['As']] = wmap / conv
        return ev

    def get_emulator(self, max_z=None, redshifts='default', pixel_resolution_km_s=1., use_measured_parameters=False,
                     redshift_dependent_parameters=False):
        """ Build an emulator for the desired k_F and our simulations.
            kf gives the desired k bins in s/km.
            Mean flux rescaling is handled (if mean_flux=True) as follows:
            1. A set of flux power spectra are generated for every one of a list of possible mean flux values.
            2. Each flux power spectrum in the set is rescaled to the same mean flux.
            3.
        """
        if redshifts is 'default':
            redshifts = self.redshifts
        gp = self._get_custom_emulator(emuobj=None, max_z=max_z, redshifts=redshifts,
                                       pixel_resolution_km_s=pixel_resolution_km_s, use_measured_parameters=use_measured_parameters,
                                       redshift_dependent_parameters=redshift_dependent_parameters)
        return gp


def get_simulation_parameters_knots(base):
    """Get the parameters of a knot-based simulation from the SimulationICs JSON file."""
    jsin = open(os.path.join(base, "SimulationICs.json"), 'r')
    pp = json.load(jsin)
    knv = pp["knot_val"]
    #This will fail!
    slope, amp = _therm_params(pp)
    parvec = [0., 1., *knv, slope, amp, pp["hubble"]]
    return parvec

def _therm_params(pp):
    """Helper to get thermal parameters from a json dictionary."""
    try:
        #Old-style emulator
        assert pp["code_args"]["rescale_gamma"] is True
        slope = pp["code_args"]["rescale_slope"]
        amp = pp["code_args"]["rescale_amp"]
    except KeyError:
        assert pp["rescale_gamma"] is True
        slope = pp["rescale_slope"]
        amp = pp["rescale_amp"]
    return slope, amp

def get_simulation_parameters_s8(base, dt0=0, t0=1, pivot=0.05):
    """Get the parameters of a sigma8-ns-based simulation from the SimulationICs JSON file."""
    jsin = open(os.path.join(base, "SimulationICs.json"), 'r')
    pp = json.load(jsin)
    slope, amp = _therm_params(pp)
    #Change the pivot value
    As = pp['scalar_amp'] / (pivot/(2*np.pi/8.))**(pp['ns']-1.)
    parvec = [dt0, t0, pp['ns'], As, slope, amp, pp["hubble"]]
    return parvec
