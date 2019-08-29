"""File to calculate integrated heating"""

import copy as cp
import numpy as np
import scipy.integrate as spi
import astropy.units as u

from .SimulationRunner.SimulationRunner import gen_UVB as uvb
#import SimulationRunner.gen_UVB as uvb

def get_integrated_heating(z_min, z_max, TREECOOL, number_to_mass_density_ratios, hubble, omega_m):
    """Calculate the integrated heating [eV] per unit [proton] mass. Number to mass density ratio in [1 / m_p]"""
    z = (10 ** TREECOOL[:, 0]) - 1.
    slice_array = (z >= z_min) * (z <= z_max)

    species_integrand = TREECOOL[:, 4:] * number_to_mass_density_ratios[:, 1:] / hubble_factor(
        z, hubble, omega_m)[:, np.newaxis].to(1. / u.s).value / (1. + z[:, np.newaxis])
    integrated_heating_per_species = (spi.trapz(species_integrand[slice_array], z[slice_array], axis=0) * u.erg).to(u.eV)
    print('Integrated heating [eV] per proton mass [HI, HeI, HeII] =', integrated_heating_per_species)

    return np.sum(integrated_heating_per_species)

def species_fraction_to_density_ratio(species_fractions, helium_mass_fraction):
    """Convert species [HI, HeI, HeII] fractions to number to mass density ratios in [1 / m_p]"""
    density_ratio = cp.deepcopy(species_fractions)
    density_ratio[:, 1] *= 1. - helium_mass_fraction
    density_ratio[:, 2:] *= helium_mass_fraction / 2.
    return density_ratio

def get_species_fraction_simple_model(z, z_rei, delta_z_rei=0.5):
    """Simple model (Planck "tanh" model) for species [HI, HeI, HeII] fractions"""
    y = lambda a: (1. + a) ** 1.5
    delta_y = 1.5 * ((1. + z_rei) ** 0.5) * delta_z_rei

    species_fractions = 1. - (0.5 * (1. + np.tanh((y(z_rei)[np.newaxis, :] - y(z)[:, np.newaxis]) / delta_y[np.newaxis, :])))
    species_fractions[:, 2] -= species_fractions[:, 1] #Correct HeII fraction
    return np.hstack((z.reshape(-1, 1), species_fractions))

def get_species_fraction_gen_UVB(z, z_rei):
    """Simple model (as used in the gen_UVB module) for species [HI, HeI, HeII] fractions"""
    species_fractions = np.zeros((z.size, 3))
    for i in range(2):
        species_fractions[:, i] = 1. - uvb.myfQHII_2(z, z_rei[i])
    species_fractions[:, 2] = 1. - uvb.volume_filling_factor_HeIII(z, z_rei[2]) - species_fractions[:, 1]
    return np.hstack((z.reshape(-1, 1), species_fractions))

def _get_single_species_fraction_Onorbe_analytical(z, TREECOOL, recombination_rate, C_ionised_species, hubble, omega_b,
                                                   helium_mass_fraction):
    """Get one of the species [HI, HeI, HeII] fractions as predicted analytically (HI; HeI: eq. A15 in 1607.04218)"""
    number_density_H_physical = uvb.calc_nH(cosmo=[hubble, omega_b, None, None, 1. - helium_mass_fraction]) * ((1. + z) ** 3)
    chi = helium_mass_fraction / 4. / (1. - helium_mass_fraction)
    A = C_ionised_species * number_density_H_physical * recombination_rate * (1. + chi)
    return ((2. * A) + TREECOOL[:, 1] - np.sqrt((TREECOOL[:, 1] ** 2) + (4. * A * TREECOOL[:, 1]))) / 2. / A

def get_species_fraction_Onorbe_analytical(z, TREECOOL, hubble, omega_b, helium_mass_fraction, T0=2.e+4*u.K, z_rei_HeII=3.):
    """Species [HI, HeI, HeII] fractions as predicted analytically (HI: eq. A15 in 1607.04218)"""
    species_fractions = np.zeros((z.size, 3))

    C_H_II = np.ones_like(z) * 1.5
    #C_H_II[z < 10.] = 2.
    species_fractions[:, 0] = _get_single_species_fraction_Onorbe_analytical(z, TREECOOL,
                                uvb.calc_alphaB(T0=(T0.to(u.K)).value), C_H_II, hubble, omega_b, helium_mass_fraction)

    #species_fractions[:, 1] = cp.deepcopy(species_fractions[:, 0])
    C_He_II = cp.deepcopy(C_H_II)
    species_fractions[:, 1] = _get_single_species_fraction_Onorbe_analytical(z, TREECOOL,
                                uvb.calc_alphaBHeII(T0=(T0.to(u.K)).value), C_He_II, hubble, omega_b, helium_mass_fraction)

    #species_fractions[:, 2] = get_species_fraction_gen_UVB(z, np.array([15., 15., z_rei_HeII]))[:, 3]
    species_fractions[:, 2] = 1. - species_fractions[:, 1] - uvb.volume_filling_factor_HeIII(z, z_rei_HeII)

    return np.hstack((z.reshape(-1, 1), species_fractions))

def hubble_factor(z, hubble, omega_m):
    """Get the Hubble factor"""
    return hubble * np.sqrt((omega_m * ((1. + z) ** 3)) + (1. - omega_m)) * 100. * u.km / u.s / u.Mpc

def simulation_parameters_to_integrated_heating(z_range, TREECOOL, heat_amp, hubble, omega_m, omega_b,
                                                helium_mass_fraction, T0, z_rei_HeII=3.):
    """Calculate the integrated heating [eV] per unit [proton] mass, given some simulation parameters"""
    print(z_range, heat_amp, hubble, omega_m, omega_b, helium_mass_fraction, T0, z_rei_HeII)
    TREECOOL[:, 4:] *= heat_amp
    z = (10 ** TREECOOL[:, 0]) - 1.

    species_fractions = get_species_fraction_Onorbe_analytical(z, TREECOOL, hubble, omega_b, helium_mass_fraction,
                                                               T0=T0.to(u.K), z_rei_HeII=z_rei_HeII)
    number_to_mass_density_ratios = species_fraction_to_density_ratio(species_fractions, helium_mass_fraction)
    return get_integrated_heating(np.min(z_range), np.max(z_range), TREECOOL, number_to_mass_density_ratios, hubble, omega_m)

if __name__ == "__main__":
    #TREECOOL = np.loadtxt('/Users/kwame/Software/SimulationRunner/SimulationRunner/TREECOOL_hm_2012')
    TREECOOL = np.loadtxt('/Users/kwame/Simulations/nCDM/nCDM_test_512/TREECOOL_Trei4e+04_HM12')
    heat_amp_factor = 1.
    omega_m = 0.3209 #0.308

    hubble = np.sqrt(0.14345 / omega_m) #0.678
    z_rei = np.array([15., 15., 3.]) #8., 8., 3.])  # HI, HeI, HeII
    z_min, z_max = (6., 13.)
    #z_min, z_max = (4.6, 13.)
    #z_min, z_max = (4.2, 12.)

    omega_b = 0.04950 #0.0482
    helium_mass_fraction = 0.241 #0.24

    TREECOOL[:, 4:] *= heat_amp_factor
    z = (10 ** TREECOOL[:, 0]) - 1.
    #species_fractions = get_species_fraction_Onorbe_analytical(z, TREECOOL, hubble, omega_b, helium_mass_fraction, T0=8070.*u.K)
    #get_species_fraction_gen_UVB(z, z_rei)
    #get_species_fraction_simple_model(z, z_rei, delta_z_rei=1.5)

    #number_to_mass_density_ratios = species_fraction_to_density_ratio(species_fractions, helium_mass_fraction)
    #integrated_heating = get_integrated_heating(z_min, z_max, TREECOOL, number_to_mass_density_ratios, hubble, omega_m)
    #print('Total integrated heating [eV] per proton mass =', integrated_heating)
