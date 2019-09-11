"""Module for holding different mean flux models"""

import math as mh
import numpy as np

def obs_mean_tau(redshift, amp=0, slope=0):
    """The mean flux from 0711.1862: is (0.0023±0.0007) (1+z)^(3.65±0.21)
    Note we constrain this much better from the SDSS data itself:
    this is a weak prior"""
    return (2.3+amp)*1e-3*(1.0+redshift)**(3.65+slope)

def obs_mean_tau_high_z(redshift, amp=0, slope=0):
    """The mean flux as used in 1809.06980 is 0.0014(1+z)^4.0
    Note we constrain this much better from data itself:
    this is a weak prior"""
    return (1.4+amp)*1e-3*(1.0+redshift)**(4.+slope)

class ConstMeanFlux(object):
    """Object which implements different mean flux models. This model fixes the mean flux to a constant value.
    """
    def __init__(self, value = 1.):
        self.value = value
        self.dense_param_names = {}

    def get_t0(self, zzs, params=None):
        """Get mean optical depth."""
        if params is None:
            params = self.value
        if params is None:
            return np.array([None,])
        return np.array([params * obs_mean_tau(zzs),])

    def get_mean_flux(self, zzs, params=None):
        """Get mean flux"""
        t0 = self.get_t0(zzs, params=params)
        if t0[0] is None:
            return t0
        return np.exp(-1 * t0)

    def get_params(self):
        """Returns a list of parameters where the mean flux is evaluated."""
        return None

    def get_limits(self):
        """Get limits on the dense parameters"""
        return None


class ConstMeanFluxHighRedshift(ConstMeanFlux):
    """Class which implements a mean flux model appropriate for high redshifts (z > 4) -- fixed to a fiducial redshift
    dependence"""
    def __init__(self, value=1.):
        super().__init__(value=value)

    def get_t0(self, zzs, params=None):
        """Get mean optical depth."""
        if params is None:
            params = self.value
        if params is None:
            return np.array([None,])
        return np.array([params * obs_mean_tau_high_z(zzs)])


class MeanFluxFactor(ConstMeanFlux):
    """Object which implements different mean flux models. This model parametrises
    uncertainty in the mean flux with a simple scaling factor.
    """
    def __init__(self, dense_samples = 10, dense_limits = None, dense_parameter_names=None, redshift_pivot=3.):
        #Limits on factors to multiply the thermal history by.
        #Mean flux is known to about 10% from SDSS, so we don't need a big range.
        self.redshift_pivot = redshift_pivot
        if dense_limits is None:
            slopehigh = np.max(mean_flux_slope_to_factor(np.linspace(2.2, 4.2, 11),0.25, redshift_pivot=self.redshift_pivot))
            slopelow = np.min(mean_flux_slope_to_factor(np.linspace(2.2, 4.2, 11),-0.25))
            self.dense_param_limits = np.array([[0.75,1.25]]) * np.array([slopelow, slopehigh])
        else:
            self.dense_param_limits = dense_limits
        self.dense_samples = dense_samples
        if dense_parameter_names is None:
            self.dense_param_names = { 'tau0': 0, }
        else:
            self.dense_param_names = dense_parameter_names

    def get_t0(self, zzs, params=None):
        """Get the mean optical depth as a function of redshift for all parameters."""
        if params is None:
            params = self.get_params()
        return np.array([t0 * obs_mean_tau(zzs) for t0 in params])

    def get_params(self):
        """Returns a list of parameters where the mean flux is evaluated."""
        #Number of dense parameters
        ndense = np.shape(self.dense_param_limits)[0]
        #This grid will hold the expanded grid of parameters: dense parameters are on the end.
        #Initially make it NaN as a poisoning technique.
        pvals = np.nan*np.zeros((self.dense_samples, ndense))
        for dd in range(ndense):
            #Build grid of mean fluxes
            dlim = self.dense_param_limits[dd]
            dense = np.linspace(dlim[0], dlim[1], self.dense_samples)
            pvals[:,dd] = dense
        assert not np.any(np.isnan(pvals))
        return pvals

    def get_limits(self):
        """Get limits on the dense parameters"""
        return self.dense_param_limits


class MeanFluxFactorHighRedshift(MeanFluxFactor):
    """Class which implements a mean flux model appropriate for high redshifts (z > 4)"""
    def __init__(self, dense_samples=10, dense_limits=None, dense_parameter_names=None, redshifts=np.array([4.24, 4.58, 4.95])):
        redshift_pivot = redshifts[mh.floor(redshifts.size / 2)]
        if dense_limits is None:
            tau_factor_maximum = np.max(mean_flux_slope_to_factor(redshifts, 0.25, redshift_pivot=redshift_pivot))
            tau_factor_minimum = np.min(mean_flux_slope_to_factor(redshifts, -0.25, redshift_pivot=redshift_pivot))
            dense_limits = np.array([[0.75, 1.25]]) * np.array([tau_factor_minimum, tau_factor_maximum])
        super().__init__(dense_samples=dense_samples, dense_limits=dense_limits,
                         dense_parameter_names=dense_parameter_names, redshift_pivot=redshift_pivot)

    def get_t0(self, zzs, params=None):
        """Get the mean optical depth as a function of redshift for all parameters."""
        if params is None:
            params = self.get_params()
        return np.array([t0 * obs_mean_tau_high_z(zzs) for t0 in params])


class FreeMeanFlux(MeanFluxFactorHighRedshift):
    """Class which implements a mean flux model where each redshift has its own parameter"""
    def __init__(self, dense_samples=10, dense_limits=None, dense_parameter_names=None, redshifts=np.array([4.24, 4.58, 4.95])):
        if dense_parameter_names is None:
            dense_parameter_names = {'tau0_%.2f'%redshift : i for i, redshift in enumerate(np.sort(redshifts)[::-1])}
        super().__init__(dense_samples=dense_samples, dense_limits=dense_limits,
                         dense_parameter_names=dense_parameter_names, redshifts=redshifts)


def mean_flux_slope_to_factor(zzs, slope, redshift_pivot=3.):
    """Convert a mean flux slope into a list of mean flux amplitudes."""
    #tau_0_i[z] @dtau_0 / tau_0_i[z] @[dtau_0 = 0]
    taus = obs_mean_tau(zzs, amp=0, slope=slope)/obs_mean_tau(zzs, amp=0, slope=0)
    ii = np.argmin(np.abs(zzs-redshift_pivot))
    #Divide by redshift pivot bin
    return taus / taus[ii]
