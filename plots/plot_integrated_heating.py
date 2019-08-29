"""Plot the integrated heating"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

import lyaemu.coarse_grid as cg
import lyaemu.integrated_heating as ih

if __name__ == "__main__":
    emulator_base_directory = sys.argv[1] #'/share/data2/keir/Simulations'
    emulator_name = sys.argv[2] #'nCDM_test_emulator'
    emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, emulator_name))
    emulator_instance.load()

    z_ranges = [[6., 13.], [4.6, 13.], [4.2, 12.]]
    helium_mass_fraction = 0.2453
    z_rei_HeII = 3.
    omega_b = emulator_instance.omegab

    integrated_heating = np.zeros((emulator_instance.get_parameters().shape[0], emulator_instance.redshifts.shape[0]))
    for i, input_parameters in enumerate(emulator_instance.get_parameters()):
        simulation_directory = emulator_instance.get_outdir(input_parameters)
        TREECOOL_path = os.path.join(simulation_directory, 'TREECOOL')
        TREECOOL = np.loadtxt(TREECOOL_path)
        heat_amp = input_parameters[emulator_instance.param_names['heat_amp']]
        omega_m = input_parameters[emulator_instance.param_names['omega_m']]
        hubble = np.sqrt(emulator_instance.omegamh2 / omega_m)
        T0 = emulator_instance.get_measured_parameters()[i, emulator_instance.measured_param_names['T_0_z_5.0']] * u.K

        for j, redshift in enumerate(emulator_instance.redshifts):
            integrated_heating[i, j] = ih.simulation_parameters_to_integrated_heating(z_ranges[j], TREECOOL, heat_amp,
                                            hubble, omega_m, omega_b, helium_mass_fraction, T0, z_rei_HeII=z_rei_HeII)

    #Scatter plot integrated heating
    savefile = os.path.join(emulator_base_directory, emulator_name, 'integrated_heating_scatter_%s.pdf' % emulator_name)
    figure, axes = plt.subplots(nrows=emulator_instance.redshifts.shape[0], ncols=2)
    for i, redshift in enumerate(emulator_instance.redshifts):
        axes[i, 0].scatter(integrated_heating[:, i], emulator_instance.get_measured_parameters()[:, i], label=r'$z = %.2f$'%redshift)
        axes[i, 1].scatter(integrated_heating[:, i], emulator_instance.get_measured_parameters()[:, i+3])
        axes[i, 0].legend(frameon=True, fontsize=7.)
        axes[i, 0].set_ylabel(r'$T_0 (z)$')
        axes[i, 1].set_ylabel(r'$\gamma (z)$')
    x_label = r'$u_0 (z)$'
    axes[-1, 0].set_xlabel(x_label)
    axes[-1, 1].set_xlabel(x_label)
    plt.savefig(savefile)

    #Dump data to JSON
    measured_parameter_names = np.array(['u_0_z_5.0', 'u_0_z_4.6', 'u_0_z_4.2'])
    remove_simulation_parameters = np.array([8, 9])

    redshift_sensitivity = np.zeros((emulator_instance.redshifts.shape[0], 16), dtype=np.bool)
    redshift_sensitivity[:, :7] = True
    redshift_sensitivity[0, np.array([7, 10, 13])] = True
    redshift_sensitivity[1, np.array([8, 11, 14])] = True
    redshift_sensitivity[2, np.array([9, 12, 15])] = True

    emulator_instance.dump_measured_parameters(measured_parameter_names, integrated_heating.value,
                                               remove_simulation_parameters, redshift_sensitivity=redshift_sensitivity)
