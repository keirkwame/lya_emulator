"""Plot temperature-density parameters"""

import os
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

import lyaemu.coarse_grid as cg
import lyaemu.tempdens as td

if __name__ == "__main__":
    emulator_base_directory = '/share/data2/keir/Simulations'
    emulator_name = 'nCDM_test_thermal'
    emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, emulator_name))
    emulator_instance.load()

    snapshot_numbers = np.array([3, 4, 6])
    redshifts = emulator_instance.redshifts

    savefile = os.path.join(emulator_base_directory, emulator_name, 'temperature_density.pdf')
    figure, axes = plt.subplots(nrows=2, ncols=1)

    T0 = np.zeros((emulator_instance.get_parameters().shape[0], snapshot_numbers.shape[0]))
    gamma = np.zeros_like(T0)

    for i, input_parameters in enumerate(emulator_instance.get_parameters()):
        print(input_parameters)
        simulation_directory = emulator_instance.get_outdir(input_parameters)
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
