"""Make some plots"""
import sys

from make_paper_plots import *
from coarse_grid import *
from coarse_grid_plot import *
from plot_likelihood import *
from plot_latin_hypercube import *

if __name__ == "__main__":
    sim_rootdir = sys.argv[1]
    savedir = sys.argv[2]
    plotname = sys.argv[3]
    chain_savedir = sys.argv[4]

    testdir = sim_rootdir + '/refinement_tight_validation' #'/hires_s8_test' #'/hot_cold_test' #/share/hypatia/sbird
    emudir = sim_rootdir + '/refinement_big' #'/hires_s8' #'/hot_cold'

    simulation_sub_directory1 = '/ns0.96As2.6e-09heat_slope-0.19heat_amp1hub0.74/output'
    #'/ns0.97As2.2e-09heat_slope0.083heat_amp0.92hub0.69/output' #'/HeliumHeatAmp0.9/output'
    simulation_sub_directory2 = '/test/output' #'/HeliumHeatAmp1.1/output'

    likelihood_samples_plot_savefile = savedir + '/likelihood_samples_' + plotname + '.pdf'
    flux_power_plot_savefile = savedir + '/flux_power_' + plotname + '.pdf'
    compare_plot_savefile = savedir + '/flux_power_comparison_' + plotname + '.pdf'
    emulator_error_plot_savefile = savedir + '/emulator_error_' + plotname + '.pdf'
    initial_parameter_samples_plot_savefile = savedir + '/initial_parameter_samples_' + plotname + '.pdf'

    #new_simulation_parameters = np.array([[9.83826422e-01, 1.53457828e-09, -1.12879347e-01, 1.30000000e+00, 6.61319870e-01],])
    #new_simulation_parameters = np.array([[9.86695393e-01, 1.42783510e-09, -1.03117089e-01, 1.34173988e+00, 6.5374e-01],]) #6.53651311e-01
    #new_simulation_parameters = np.array([[9.46829601e-01, 2.25988354e-09, 1.34815249e-01, 9.13357137e-01, 7.12787210e-01],])
    #new_simulation_parameters = np.array([[9.57931637e-01, 2.24577734e-09, 1.47123460e-01, 9.23891652e-01, 6.91849447e-01],])
    #new_simulation_parameters = np.array([[9.59915872e-01, 2.22953551e-09, 1.91087802e-01, 9.51346570e-01, 6.95725678e-01],])
    #new_simulation_parameters = np.array([[9.70874564e-01, 2.31612328e-09, 1.72823711e-01, 1.00025673e+00, 6.87801539e-01],])
    #new_simulation_parameters = np.array([[9.68576576e-01, 2.27767175e-09, 1.50622468e-01, 9.92427247e-01, 6.75640312e-01],])

    #new_simulation_parameters = np.array([[9.85879594e-01, 2.41643810e-09, 1.77867781e-01, 1.07567171e+00, 6.66582572e-01],])

    #new_simulation_parameters = np.array([[9.75837010e-01, 2.09004645e-09, -2.01500942e-02, 1.21652183e+00, 6.84541976e-01],])
    #new_simulation_parameters = np.array([[9.78768076e-01, 2.06522887e-09, 1.74467822e-02, 1.21296778e+00, 6.80533609e-01],])
    new_simulation_parameters = np.array([[9.70352841e-01, 2.06333254e-09, 1.15731455e-01, 1.10472541e+00, 6.91902152e-01],])
    print(new_simulation_parameters.shape)
    emulator_parameter_limits = np.array([[0.9, 0.99], [1.5e-09, 2.8e-09], [-0.4, 0.4], [0.6, 1.4], [0.65, 0.75]]) #Big emulator
    #emulator_parameter_limits = np.array([[0.8, 1.05], [1.2e-09, 3e-09], [-0.5, 0.5], [0.5, 1.5], [0.6, 0.8]]) #Small emulator

    #test_knot_plots(testdir=testdir, emudir=emudir, plotdir=savedir, plotname=plotname, mf=2, kf_bin_nums=None, data_err=False, max_z=4.2)
    #plot_test_interpolate_kf_bin_loop(emudir, testdir, savedir=savedir, plotname="_Two_loop", kf_bin_nums=np.arange(2))

    #make_plot(chain_savedir + '/AA0.97BB1.3_chain_20000_MeanFluxFactor.txt', likelihood_samples_plot_savefile)
    #output = make_plot_flux_power_spectra(testdir, emudir, flux_power_plot_savefile, mean_flux_label='s', rescale_data_error=True, fix_error_ratio=False, error_ratio=100.)
    #make_plot_compare_two_simulations(emudir, emudir, simulation_sub_directory1, simulation_sub_directory2, compare_plot_savefile)
    #make_plot_emulator_error(emudir, emulator_error_plot_savefile, mean_flux_label='s') #, max_z=2.6)
    #output = make_plot_initial_parameter_samples(initial_parameter_samples_plot_savefile)
    #generate_emulator_submissions(emudir, new_simulation_parameters, emulator_parameter_limits, hypatia_queue='cores24', refinement=True)
    output = run_and_plot_likelihood_samples(testdir, emudir, likelihood_samples_plot_savefile, plotname, plot_posterior=True,
                                             chain_savedir=chain_savedir, n_burn_in_steps=500, n_steps=1500,
                                             while_loop=False, mean_flux_label='s', return_class_only=False,
                                             rescale_data_error=True, fix_error_ratio=False, error_ratio=100.,
                                             include_emulator_error=True) #, emulator_json_file='emulator_params_refine5star.json')  # , max_z=2.6)
    #output = run_simulations(testdir, emudir, new_simulation_parameters, simulation_sub_directory=simulation_sub_directory1, optimise_GP=False)
    #make_emulator_latin_hypercube(emudir, 21, emulator_parameter_limits, hypatia_queue='cores24')
    #generate_emulator_submissions(emudir, new_simulation_parameters, emulator_parameter_limits, hypatia_queue='cores24', refinement=True)
