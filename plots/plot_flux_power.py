"""Plot 1D flux power spectra"""

import os
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

import lyaemu.coarse_grid as cg
import lyaemu.mean_flux as mef

if __name__ == "__main__":
    emulator_base_directory = '/share/data2/keir/Simulations'
    emulator_name = 'nCDM_test_thermal2_corners'
    emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, emulator_name), mf=mef.MeanFluxFactorHighRedshift(dense_samples=2))
    emulator_instance.load()
    n_simulations = emulator_instance.get_parameters().shape[0]
    default_simulation_index = 0

    savefile = os.path.join(emulator_base_directory, emulator_name, 'flux_power_nCDM_test_thermal2_corners.pdf')
    figure, axes = plt.subplots(nrows=n_simulations, ncols=2, figsize=(20., 20.))

    input_parameters_all, k_parallel, flux_powers = emulator_instance.get_flux_vectors(
        redshifts=emulator_instance.redshifts, pixel_resolution_km_s=1., fix_mean_flux_samples=True)
    print('Array shapes =', len(input_parameters_all), len(k_parallel), len(flux_powers), input_parameters_all.shape, k_parallel[0].shape, flux_powers.shape)
    print(input_parameters_all, k_parallel)
    flux_powers = flux_powers.reshape(k_parallel.shape)

    #Cut out a single mean flux value
    input_parameters_all = input_parameters_all[:n_simulations]
    k_parallel = k_parallel[:n_simulations]
    flux_powers = flux_powers[:n_simulations]

    for i, input_parameters in enumerate(input_parameters_all): #Loop over simulations
        axes[i, 0].set_ylabel(r'log10($k P(k) / \pi$)')
        axes[i, 1].set_ylabel(r'$P(k) / P_\mathrm{fiducial}(k)$')
        #axes[i, 0].set_xscale('log')
        #axes[i, 1].set_xscale('log')
        #axes[i, 0].set_yscale('log')
        axes[i, 0].set_title(input_parameters)
        #axes[i, 0].axvline(x = 0.2, ls=':', color='black', lw=0.5)
        #axes[i, 1].axvline(x = 0.2, ls=':', color='black', lw=0.5)
        axes[i, 0].axvline(x = -2.2, ls=':', color='black', lw=0.5)
        axes[i, 1].axvline(x = -2.2, ls=':', color='black', lw=0.5)
        axes[i, 1].axhline(y = 1., ls=':', color='black', lw=0.5)

        for j, redshift in enumerate(emulator_instance.redshifts): #Loop over redshifts
            k_cut = k_parallel[i, j] < 0.2
            print(flux_powers[i, j])
            axes[i][0].plot(np.log10(k_parallel[i, j][k_cut]), np.log10(flux_powers[i, j] * k_parallel[i, j] / np.pi)[k_cut], label=r'$z = %.2f$'%redshift)
            axes[i][1].plot(np.log10(k_parallel[i, j][k_cut]), (flux_powers[i, j] / flux_powers[default_simulation_index, j])[k_cut],
                            label=r'$z = %.2f$' % redshift)

    axes[0, 0].legend(frameon=False, fontsize=7.)
    x_label = r'log10($k_{||}$ [s/km])'
    axes[-1, 0].set_xlabel(x_label)
    axes[-1, 1].set_xlabel(x_label)

    plt.savefig(savefile)
