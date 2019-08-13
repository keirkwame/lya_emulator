"""Plot 1D flux power spectra"""

import os
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

import lyaemu.coarse_grid as cg

if __name__ == "__main__":
    emulator_base_directory = '/share/data2/keir/Simulations'
    emulator_name = 'nCDM_test_thermal'
    emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, emulator_name))
    emulator_instance.load()
    n_simulations = emulator_instance.get_parameters().shape[0]
    default_simulation_index = 0

    savefile = os.path.join(emulator_base_directory, emulator_name, 'flux_power.pdf')
    figure, axes = plt.subplots(nrows=n_simulations, ncols=2)

    input_parameters_all, k_parallel, flux_powers = emulator_instance.get_flux_vectors(
        redshifts=emulator_instance.redshifts, pixel_resolution_km_s=1.)
    flux_powers = flux_powers.reshape(k_parallel.shape)

    for i, input_parameters in enumerate(input_parameters_all): #Loop over simulations
        axes[i, 0].set_ylabel(r'$k P(k) / \pi$')
        axes[i, 1].set_ylabel(r'$P(k) / P_\mathrm{fiducial}(k)$')
        axes[i, 0].set_xscale('log')
        axes[i, 1].set_xscale('log')
        axes[i, 0].set_yscale('log')

        for j, redshift in enumerate(emulator_instance.redshifts): #Loop over redshifts
            axes[i][0].plot(k_parallel[i, j], flux_powers[i, j] * k_parallel[i, j] / np.pi, label=r'$z = %.2f$'%redshift)
            axes[i][1].plot(k_parallel[i, j], flux_powers[i, j] / flux_powers[default_simulation_index, j],
                            label=r'$z = %.2f$' % redshift)

    axes[0, 0].legend(frameon=False, fontsize=7.)
    x_label = r'$k_{||}$ (s/km)'
    axes[-1, 0].set_xlabel(x_label)
    axes[-1, 1].set_xlabel(x_label)

    plt.savefig(savefile)
