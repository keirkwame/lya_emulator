"""Plot temperature-density parameters"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

import lyaemu.coarse_grid as cg
import lyaemu.tempdens as td

if __name__ == "__main__":
    emulator_base_directory = sys.argv[1] #'/share/data2/keir/Simulations'
    emulator_name = sys.argv[2] #'nCDM_test_emulator'
    load_file = sys.argv[3]
    dump_file = sys.argv[4]
    emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, emulator_name))
    emulator_instance.load(dumpfile=load_file)
    optimisation_index = int(sys.argv[5]) #0

    snapshot_numbers = np.array([7, 8, 10]) #- 4
    redshifts = emulator_instance.redshifts

    savefile = os.path.join(emulator_base_directory, emulator_name, 'temperature_density_%s_%s.pdf'
                            %(emulator_name, optimisation_index)) #test_emulator.pdf')
    figure, axes = plt.subplots(nrows=2, ncols=1)

    T0 = np.zeros((emulator_instance.get_parameters()[optimisation_index:].shape[0], snapshot_numbers.shape[0]))
    gamma = np.zeros_like(T0)

    for i, input_parameters in enumerate(emulator_instance.get_parameters()[optimisation_index:]):
        print(input_parameters)
        simulation_directory = emulator_instance.get_outdir(input_parameters, extra_flag=optimisation_index+i+1)
        for j, snapshot_number in enumerate(snapshot_numbers):
            T0[i, j], gamma[i, j] = td.fit_td_rel_plot(snapshot_number, simulation_directory, plot=False)
        axes[0].plot(redshifts, T0[i], label=i) #input_parameters)
        axes[1].plot(redshifts, gamma[i], label=i) #input_parameters)

    axes[0].legend(frameon=False, fontsize=7.)
    axes[1].legend(frameon=False, fontsize=7.)

    axes[0].set_ylabel(r'$T_0 (z)$')
    axes[1].set_ylabel(r'$\gamma (z)$')
    axes[1].set_xlabel(r'$z$')

    plt.savefig(savefile)

    #Scatter plot T0, gamma
    savefile = os.path.join(emulator_base_directory, emulator_name, 'temperature_density_scatter_%s_%s.pdf'
                            %(emulator_name, optimisation_index)) #test_emulator.pdf')
    figure, axes = plt.subplots(nrows=redshifts.size, ncols=1)
    for i, redshift in enumerate(redshifts):
        axes[i].scatter(T0[:, i], gamma[:, i], label=r'$z = %.2f$'%redshift)
        axes[i].legend(frameon=False, fontsize=7.)
        axes[i].set_ylabel(r'$\gamma (z)$')
    axes[-1].set_xlabel(r'$T_0 (z)$')
    plt.savefig(savefile)

    #Dump data to JSON
    measured_parameter_names = np.array(['T_0_z_5.0', 'T_0_z_4.6', 'T_0_z_4.2', 'gamma_z_5.0', 'gamma_z_4.6', 'gamma_z_4.2'])
    measured_sample_parameters = np.concatenate((T0, gamma), axis=1)
    remove_simulation_parameters = np.array([2, 3])
    emulator_instance.dump_measured_parameters(measured_parameter_names, measured_sample_parameters,
                                               remove_simulation_parameters, dumpfile=dump_file,
                                               add_optimisation=optimisation_index)
