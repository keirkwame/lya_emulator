"""Plot 1D flux power spectra"""

import os
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

import lyaemu.coarse_grid as cg
import lyaemu.mean_flux as mef

if __name__ == "__main__":
    emulator_base_directory = ['/share/data2/keir/Simulations', '/share/rcifdata/keir/Simulations']
    emulator_names = ['nCDM_convergence_512_256', 'nCDM_convergence_768_WDM']
    emulator_jsons = ['emulator_params.json', 'emulator_params_4.json']
    flux_vectors = ['emulator_flux_vectors.hdf5', 'emulator_4_flux_vectors.hdf5']
    simulation_indices = [np.arange(2), np.arange(4)] #, np.arange(1)] #np.arange(2)]
    interpolate_to_same_k = False

    default_emulator_index = 0
    default_simulation_index = 1
    n_simulations = np.sum(
        [np.size(simulation_indices_single_emulator) for simulation_indices_single_emulator in simulation_indices])

    savefile = os.path.join(emulator_base_directory[0], emulator_names[0], 'flux_power_nCDM_convergence_mfraw.pdf')
    figure, axes = plt.subplots(nrows=n_simulations, ncols=2, figsize=(20., 20. * 6. / 10.))

    plot_start_index = 0
    input_parameters_all = []
    k_parallel = []
    flux_powers = []
    for a, emulator_name in enumerate(emulator_names):
        emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory[a], emulator_name), mf=mef.ConstMeanFluxHighRedshift(value=1.))
        emulator_instance.load(dumpfile=emulator_jsons[a]) #'emulator_params_batch4_1_TDR_u0.json')

        input_parameters_all_single_emulator, k_parallel_single_emulator, flux_powers_single_emulator = emulator_instance.get_flux_vectors(
            redshifts=emulator_instance.redshifts, pixel_resolution_km_s=1., fix_mean_flux_samples=True,
            no_mean_flux_rescaling=True, savefile=flux_vectors[a], parallel=True, n_process=8)
        print(flux_powers_single_emulator.shape)
        flux_powers_single_emulator = flux_powers_single_emulator.reshape(k_parallel_single_emulator.shape)

        #Cut out certain simulations
        input_parameters_all.append(input_parameters_all_single_emulator[simulation_indices[a]])
        k_parallel.append(k_parallel_single_emulator[simulation_indices[a]])
        flux_powers.append(flux_powers_single_emulator[simulation_indices[a]])
        print(len(flux_powers), flux_powers[a].shape)

        for b, input_parameters in enumerate(input_parameters_all[a]): #Loop over simulations
            i = plot_start_index + b
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
                k_cut = k_parallel[a][b, j] < 0.2
                k_plot = k_parallel[a][b, j]
                if interpolate_to_same_k:
                    flux_powers_default = 10 ** np.interp(np.log10(k_plot[k_cut]),
                                                    np.log10(k_parallel[default_emulator_index][default_simulation_index, j]),
                                                    np.log10(flux_powers[default_emulator_index][default_simulation_index, j]))
                else:
                    flux_powers_default = flux_powers[default_emulator_index][default_simulation_index, j][k_cut]
                axes[i][0].plot(np.log10(k_plot[k_cut]), np.log10(flux_powers[a][b, j] * k_plot / np.pi)[k_cut], label=r'$z = %.2f$'%redshift)
                axes[i][1].plot(np.log10(k_plot[k_cut]), flux_powers[a][b, j][k_cut] / flux_powers_default,
                                label=r'$z = %.2f$' % redshift)
        plot_start_index += simulation_indices[a].size

    axes[0, 0].legend(frameon=False, fontsize=7.)
    x_label = r'log10($k_{||}$ [s/km])'
    axes[-1, 0].set_xlabel(x_label)
    axes[-1, 1].set_xlabel(x_label)

    plt.savefig(savefile)
