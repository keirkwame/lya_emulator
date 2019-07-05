"""Script containing details of nCDM emulators"""

import os
import numpy as np

import lyaemu.coarse_grid as cg

if __name__ == '__main__':
    emulator_base_directory = '/share/data2/keir/Simulations'

    #nCDM_test
    n_simulations = 10
    samples_fiducial_nCDM = [0.9635, 1.8296e-9, 0., 1., 0.3209, 0., 1. ,-1., 8.]
    samples_perturbation_nCDM = np.array([0.9, 2.5e-9, -1.1, 1.4, 0.26, 0.1, 10., -10., 15.])
    samples_nCDM_test = np.array(samples_fiducial_nCDM * n_simulations).reshape(n_simulations, -1)
    samples_nCDM_test[1:, :][np.diag_indices(n_simulations - 1)] = samples_perturbation_nCDM

    #Beta & gamma perturbations need alpha=0.1 to have any effect
    samples_nCDM_test[7, 5] = 0.1
    samples_nCDM_test[8, 5] = 0.1

    emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, 'nCDM_test2'))
    emulator_instance.gen_simulations(None, npart=256, box=10., samples=samples_nCDM_test)
