"""Script containing details of nCDM emulators"""

import os
import sys
import numpy as np

import lyaemu.coarse_grid as cg
import lyaemu.likelihood as like

if __name__ == '__main__':
    emulator_base_directory = sys.argv[1]  # '/share/data2/keir/Simulations'
    emulator_name = sys.argv[2] #'nCDM_emulator_512'

    #nCDM_test
    '''n_simulations = 10
    samples_fiducial_nCDM = [0.9635, 1.8296e-9, 0., 1., 0.3209, 0., 1. ,-1., 8.]
    samples_perturbation_nCDM = np.array([0.9, 2.5e-9, -1.1, 1.4, 0.26, 0.1, 10., -10., 15.])
    samples_nCDM_test = np.array(samples_fiducial_nCDM * n_simulations).reshape(n_simulations, -1)
    samples_nCDM_test[1:, :][np.diag_indices(n_simulations - 1)] = samples_perturbation_nCDM

    #Beta & gamma perturbations need alpha=0.1 to have any effect
    samples_nCDM_test[7, 5] = 0.1
    samples_nCDM_test[8, 5] = 0.1

    emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, 'nCDM_test2'))
    emulator_instance.gen_simulations(None, npart=256, box=10., samples=samples_nCDM_test)'''

    #nCDM_test_thermal
    '''n_simulations = 9
    samples_fiducial_nCDM = [0.9635, 1.8296e-9, 0., 1., 0.3209, 0., 1., -1., 8., 2.e+4]
    samples_perturbation_nCDM_TDR_plus = np.array([0.9635, 1.8296e-9, 1.4, 1.9, 0.3209, 0., 1., -1., 8., 2.e+4])
    samples_perturbation_nCDM_TDR_minus = np.array([0.9635, 1.8296e-9, -1.4, 0.05, 0.3209, 0., 1., -1., 8., 2.e+4])
    samples_perturbation_nCDM_T_rei = np.array([0.9635, 1.8296e-9, 0., 1., 0.3209, 0., 1., -1., 8., 4.e+4])

    samples_nCDM_test = np.array(samples_fiducial_nCDM * n_simulations).reshape(n_simulations, -1)
    samples_nCDM_test[:3, 2] = samples_perturbation_nCDM_TDR_plus[2]
    samples_nCDM_test[3:6, 2] = samples_perturbation_nCDM_TDR_minus[2]
    samples_nCDM_test[np.array([0, 3, 6]), 3] = samples_perturbation_nCDM_TDR_plus[3]
    samples_nCDM_test[np.array([2, 5, 7]), 3] = samples_perturbation_nCDM_TDR_minus[3]
    samples_nCDM_test[8] = samples_perturbation_nCDM_T_rei

    emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, 'nCDM_test_thermal'))
    emulator_instance.gen_simulations(None, npart=256, box=10., samples=samples_nCDM_test)'''

    #nCDM_test_512_HM12
    '''n_simulations = 3
    samples_fiducial_nCDM = [0.9635, 1.8296e-9, 0., 1., 0.3209, 0., 1., -1., 8., 2.e+4]
    #samples_perturbation_nCDM_z_rei = 15.
    samples_nCDM_test = np.array(samples_fiducial_nCDM * n_simulations).reshape(n_simulations, -1)
    #samples_nCDM_test[1, 8] = samples_perturbation_nCDM_z_rei
    samples_nCDM_test[0, 9] = 4.e+4

    samples_nCDM_test[1, 3] = 3.5
    samples_nCDM_test[1, 8] = 15.
    samples_nCDM_test[1, 9] = 4.e+4

    samples_nCDM_test[2, 3] = 0.05
    samples_nCDM_test[2, 8] = 6.
    samples_nCDM_test[2, 9] = 1.5e+4

    emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, 'nCDM_test_512_filtering'))
    emulator_instance.gen_simulations(None, npart=512, box=10., samples=samples_nCDM_test)'''

    #nCDM_test_thermal2_corners
    '''n_simulations = 4
    samples_fiducial_nCDM = [0.9635, 1.8296e-9, 0., 1., 0.3209, 0., 1., -1., 8., 2.e+4]
    samples_nCDM_test = np.array(samples_fiducial_nCDM * n_simulations).reshape(n_simulations, -1)
    #B
    samples_nCDM_test[0, 2] = 1.75
    samples_nCDM_test[1, 2] = 1.75
    samples_nCDM_test[2, 2] = -1.75
    samples_nCDM_test[3, 2] = -1.75
    #A
    samples_nCDM_test[0, 3] = 3.5
    samples_nCDM_test[1, 3] = 0.05
    samples_nCDM_test[2, 3] = 3.5
    samples_nCDM_test[3, 3] = 0.05

    emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, 'nCDM_test_thermal2_corners'))
    emulator_instance.gen_simulations(None, npart=256, box=10., samples=samples_nCDM_test)'''

    #nCDM_test_emulator
    #emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, 'nCDM_emulator_512'))
    #emulator_instance.gen_simulations(50, npart=512, box=10.)

    #nCDM_emulator_512_refined
    '''emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, emulator_name))
    emulator_instance.load(dumpfile='emulator_params_input_parameters.json')
    emulator_instance.gen_simulations(101, npart=512, box=10., fill_in=True)'''

    #nCDM_test_convergence
    #nCDM_convergence_768_WDM
    '''n_simulations = 5
    samples_fiducial_nCDM = [0.9635, 1.8296e-9, 0., 1., 0.3209, 0., 1., -1., 8., 2.e+4]
    samples_nCDM_test = np.array(samples_fiducial_nCDM * n_simulations).reshape(n_simulations, -1)
    #nCDM
    samples_nCDM_test[:3, 5] = 0.0227
    samples_nCDM_test[:3, 6] = 2.24
    samples_nCDM_test[:3, 7] = -4.46
    #npart & box
    npart = np.array([256, 512, 768, 768, 768])
    box = np.array([10., 10., 10., 10., 15.])

    emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, 'nCDM_convergence_768_WDM'))
    emulator_instance.gen_simulations(None, npart=npart, box=box, samples=samples_nCDM_test)'''

    #nCDM_convergence_512_256
    '''n_simulations = 2
    samples_fiducial_nCDM = [0.9635, 1.8296e-9, 0., 1., 0.3209, 0., 1., -1., 8., 2.e+4]
    samples_nCDM_test = np.array(samples_fiducial_nCDM * n_simulations).reshape(n_simulations, -1)
    npart = np.array([512, 256])
    box = np.array([10., 10.])

    emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, 'nCDM_convergence_512_256'))
    emulator_instance.gen_simulations(None, npart=npart, box=box, samples=samples_nCDM_test)'''

    #nCDM_convergence_896
    '''n_simulations = 1
    samples_fiducial_nCDM = [0.9635, 1.8296e-9, 0., 1., 0.3209, 0., 1., -1., 8., 2.e+4]
    samples_nCDM_test = np.array(samples_fiducial_nCDM * n_simulations).reshape(n_simulations, -1)
    npart = 896
    box = 17.5

    emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, emulator_name))
    emulator_instance.gen_simulations(None, npart=npart, box=box, samples=samples_nCDM_test)'''

    #bDM_test
    n_simulations = 2
    samples_fiducial_nCDM = [0.9635, 1.8296e-9, 0., 1., 0.3209, 0., 1., -1., 8., 2.e+4]
    samples_nCDM_test = np.array(samples_fiducial_nCDM * n_simulations).reshape(n_simulations, -1)
    nCDM_test_params = like.bDM_numerical_model([9., -27.], np.array([[0., 0.1], [1., 10.], [-10., 0.]]))
    samples_nCDM_test[1, 5:8] = nCDM_test_params
    print(samples_nCDM_test)
    npart = 512
    box = 10.

    emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, emulator_name))
    emulator_instance.gen_simulations(None, npart=npart, box=box, samples=samples_nCDM_test)
