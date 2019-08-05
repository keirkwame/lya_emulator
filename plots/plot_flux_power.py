"""Plot 1D flux power spectra"""

import os

import lyaemu.coarse_grid as cg

if __name__ == "__main__":
    emulator_base_directory = '/share/data2/keir/Simulations'
    emulator_instance = cg.nCDMEmulator(os.path.join(emulator_base_directory, 'nCDM_test2'))
    emulator_instance.load()

    flux_powers = emulator_instance.get_flux_vectors(redshifts=emulator_instance.redshifts, pixel_resolution_km_s=1.)
