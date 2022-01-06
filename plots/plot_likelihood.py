"""Module for plotting generated likelihood chains"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import re
import glob
import random
import string
import itertools
import math as mh
from datetime import datetime
import numpy as np
import multiprocessing as mg
import lyaemu.distinct_colours_py3 as dc
import lyaemu.lyman_data as lyman_data
import lyaemu.likelihood as likeh
import lyaemu.coarse_grid as cg
from lyaemu.coarse_grid import get_simulation_parameters_s8
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import getdist as gd
import getdist.plots as gdp

### A helper for letting the forked processes use data without pickling.
_data_name_cands = (
    '_data_' + ''.join(random.sample(string.ascii_lowercase, 10))
    for _ in itertools.count())
class ForkedData(object):
    '''
    Class used to pass data to child processes in multiprocessing without
    really pickling/unpickling it. Only works on POSIX.
    Intended use:
        - The master process makes the data somehow, and does e.g.
          data = ForkedData(the_value)
        - The master makes sure to keep a reference to the ForkedData object
          until the children are all done with it, since the global reference
          is deleted to avoid memory leaks when the ForkedData object dies.
        - Master process constructs a multiprocessing.Pool *after*
          the ForkedData construction, so that the forked processes
          inherit the new global.
        - Master calls e.g. pool.map with data as an argument.
        - Child gets the real value through data.value, and uses it read-only.
    '''
    # TODO: does data really need to be used read-only? don't think so...
    # TODO: more flexible garbage collection options
    def __init__(self, val):
        g = globals()
        self.name = next(n for n in _data_name_cands if n not in g)
        g[self.name] = val
        self.master_pid = os.getpid()
    def __getstate__(self):
        if os.name != 'posix':
            raise RuntimeError("ForkedData only works on OSes with fork()")
        return self.__dict__
    @property
    def value(self):
        return globals()[self.name]
    def __del__(self):
        if os.getpid() == self.master_pid:
            del globals()[self.name]
class NoDaemonProcess(mg.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)
# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(mg.pool.Pool):
    Process = NoDaemonProcess


def get_k_z(likelihood_instance):
    """Get k and z bins"""
    k_los = likelihood_instance.gpemu.kf
    n_k_los = k_los.size
    z = likelihood_instance.zout #Highest redshift first
    n_z = z.size
    return k_los, z, n_k_los, n_z

def make_plot_flux_power_spectra(like, params, datadir, savefile, t0=1., data_class='BOSS',
                                 pixel_resolution_km_s='default', mean_flux_label='s'):
    """Make a plot of the power spectra, with redshift, the data power and the sigmas. Four plots stacked."""
    if data_class == 'BOSS':
        lyman_data_instance = lyman_data.BOSSData()
    elif data_class == 'Boera':
        lyman_data_instance = lyman_data.BoeraData()
    else:
        raise ValueError('Data class not recognised')

    if (mean_flux_label == 'c') or (mean_flux_label == 's'):
        mean_flux_model = 'low_z'
    elif (mean_flux_label == 'c_high_z') or (mean_flux_label == 's_high_z') or (mean_flux_label == 'free_high_z'):
        mean_flux_model = 'high_z'
    else:
        raise ValueError('Mean flux label not recognised')

    #'Data' now is a simulation
    k_los = lyman_data_instance.get_kf()
    n_k_los = k_los.size
    z = like.zout #Highest redshift first
    n_z = z.size

    assert params[1] == t0
    data_fluxpower = likeh.load_data(datadir, kf=k_los, max_z=np.max(z), redshifts=z,
                                     pixel_resolution_km_s=pixel_resolution_km_s, t0=t0,
                                     mean_flux_model=mean_flux_model)
    exact_flux_power = data_fluxpower.reshape(n_z, n_k_los)

    print('Exact parameters =', params)
    ekf, emulated_flux_power, emulated_flux_power_std = like.get_predicted(params)

    data_flux_power = like.lyman_data_flux_power
    #like.lyman_data_instance.pf.reshape(-1, n_k_los)[:n_z][::-1]

    figure, axes = plt.subplots(nrows=4, ncols=1, figsize=(6.4*2., 10.))
    distinct_colours = dc.get_distinct(n_z)
    for i in range(n_z):
        idp = np.where(k_los >= ekf[i][0])

        scaling_factor = ekf[i]/ mh.pi
        data_flux_power_std_single_z = np.sqrt(like.lyman_data_instance.get_covar(z[i]).diagonal())
        exact_flux_power_std_single_z = np.sqrt(np.diag(like.get_data_covariance(i)))

        line_width = 0.5
        axes[0].plot(ekf[i], exact_flux_power[i][idp]*scaling_factor, color=distinct_colours[i], ls='-', lw=line_width,
                     label=r'$z = %.1f$'%z[i])
        axes[0].plot(ekf[i], emulated_flux_power[i]*scaling_factor, color=distinct_colours[i], ls='--', lw=line_width)
        axes[0].errorbar(ekf[i], emulated_flux_power[i]*scaling_factor, yerr=emulated_flux_power_std[i]*scaling_factor,
                         ecolor=distinct_colours[i], ls='')

        axes[1].plot(ekf[i], data_flux_power[i][idp]*scaling_factor, color=distinct_colours[i], lw=line_width)
        axes[1].errorbar(ekf[i], data_flux_power[i][idp]*scaling_factor,
                         yerr=data_flux_power_std_single_z[idp]*scaling_factor, ecolor=distinct_colours[i], ls='')

        axes[2].plot(ekf[i], exact_flux_power_std_single_z[idp] / exact_flux_power[i][idp], color=distinct_colours[i],
                     ls='-', lw=line_width)
        axes[2].plot(ekf[i], emulated_flux_power_std[i] / exact_flux_power[i][idp], color=distinct_colours[i], ls='--',
                     lw=line_width)

        #axes[3].plot(ekf[i], data_flux_power_std_single_z / data_flux_power[i], color=distinct_colours[i], ls='-', lw=line_width)
        axes[3].plot(ekf[i], emulated_flux_power[i] / exact_flux_power[i][idp], color=distinct_colours[i], ls='-',
                     lw=line_width)
    print('z=%.2g Max frac overestimation of P_f =' % z[i],
          np.max((emulated_flux_power[i] / exact_flux_power[i][idp]) - 1.))
    print('z=%.2g Min frac underestimation of P_f =' % z[i] ,
          np.min((emulated_flux_power[i] / exact_flux_power[i][idp]) - 1.))

    fontsize = 7.
    xlim = [1.e-3, 0.022]
    xlabel = r'$k$ ($\mathrm{s}\,\mathrm{km}^{-1}$)'
    ylabel = r'$k P(k) / \pi$'

    axes[0].plot([], color='gray', ls='-', label=r'Exact')
    axes[0].plot([], color='gray', ls='--', label=r'Emulated')
    axes[0].legend(frameon=False, fontsize=fontsize)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    axes[1].plot([], color='gray', label=r'Data')
    axes[1].legend(frameon=False, fontsize=fontsize)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)

    axes[2].plot([], color='gray', ls='-', label=r'Data sigma (frac of exact power)')
    axes[2].plot([], color='gray', ls='--', label=r'Emulated sigma')
    axes[2].legend(frameon=False, fontsize=fontsize)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(r'Sigma / Exact P(k)')

    axes[3].axhline(y=1., color='black', ls=':', lw=line_width)
    axes[3].set_xscale('log')
    axes[3].set_xlabel(xlabel)
    axes[3].set_ylabel(r'Emulated P(k) / Exact P(k)')

    figure.subplots_adjust(hspace=0)
    plt.savefig(savefile)

    return like

def make_plot(chainfile, savefile, true_parameter_values=None, pnames=None, ranges=None, parameter_indices=None):
    """Make a getdist plot"""
    samples = np.loadtxt(chainfile)
    #A_s hack
    #samples = samples[samples[:, 4] > 2.05e-9, :]

    if parameter_indices is not None:
        samples = samples[:, parameter_indices]
        true_parameter_values = true_parameter_values[parameter_indices]
        pnames = pnames[parameter_indices]
        ranges = ranges[parameter_indices]

    ticks = {}
    if pnames is None:
        #Default emulator parameters
        pnames = [r"d\tau_0", r"\tau_0", r"n_s", r"A_\mathrm{P} \times 10^9", r"H_S", r"H_A", r"h"]
        samples[:,3] *= 1e9
        true_parameter_values[3] *= 1e9
        #Ticks we want to show for each parameter
        ticks = {pnames[3]: [1.5, 2.0, 2.5], pnames[4]: [-0.6,-0.3, 0.], pnames[5]: [0.5,0.7,1.0,1.3], pnames[6]: [0.66, 0.70, 0.74]}
    prange = None
    if ranges is not None:
        prange = {pnames[i] : ranges[i] for i in range(len(pnames))}
    posterior_MCsamples = gd.MCSamples(samples=samples, names=pnames, labels=pnames, label='', ranges=prange)

    print("Corner plot =", savefile)
    #Get and print the confidence limits
    for i in range(len(pnames)):
        strr = pnames[i] + " 1-sigma, 2-sigma: "
        #if i == 6:
        #    for j in (0.32, 0.05):
        #        strr += str(round(posterior_MCsamples.confidence(i, j, upper=True),5)) + " "
        #else:
        for j in (0.16, 1.-0.16, 0.025, 1.-0.025):
            strr += str(round(posterior_MCsamples.confidence(i, j), 5)) + " "
        print(strr)
    subplot_instance = gdp.getSubplotPlotter()
    subplot_instance.triangle_plot([posterior_MCsamples], filled=True)
#     colour_array = np.array(['black', 'red', 'magenta', 'green', 'green', 'purple', 'turquoise', 'gray', 'red', 'blue'])

    for pi in range(samples.shape[1]):
        for pi2 in range(pi + 1):
            #Place horizontal and vertical lines for the true point
            ax = subplot_instance.subplots[pi, pi2]
            ax.yaxis.label.set_size(16)
            ax.xaxis.label.set_size(16)
            if pi == samples.shape[1]-1 and pnames[pi2] in ticks:
                ax.set_xticks(ticks[pnames[pi2]])
            if pi2 == 0 and pnames[pi] in ticks:
                ax.set_yticks(ticks[pnames[pi]])
            if true_parameter_values is not None:
                ax.axvline(true_parameter_values[pi2], color='gray', ls='--', lw=2)
                if pi2 < pi:
                    ax.axhline(true_parameter_values[pi], color='gray', ls='--', lw=2)
#                #Plot the emulator points
#                 if parameter_index > 1:
#                     ax.scatter(simulation_parameters_latin[:, parameter_index2 - 2], simulation_parameters_latin[:, parameter_index - 2], s=54, color=colour_array[-1], marker='+')
#
#     legend_labels = ['+ Initial Latin hypercube']
#     subplot_instance.add_legend(legend_labels, legend_loc='upper right', colored_text=True, figure=True)
    plt.savefig(savefile)

def run_likelihood_test(testdir, emudir, savedir=None, prior_function='uniform', prior_function_args=None,
                        test_simulation_parameters=None, plot=True, mean_flux_label='s', max_z=4.2, redshifts=None,
                        pixel_resolution_km_s='default', t0_training_value=1., emulator_class="standard",
                        use_measured_parameters=False, redshift_dependent_parameters=False, data_class='BOSS',
                        plot_parameter_indices=None, emulator_json_file='emulator_params.json', n_threads_mcmc=1,
                        leave_out_validation=None):
    """Generate some likelihood samples"""
    #Find all subdirectories
    if test_simulation_parameters is None:
        subdirs = glob.glob(testdir + "/*/")
        assert len(subdirs) > 1
    else:
        subdirs = [testdir,]

    #Measured parameter redshift model
    measured_parameter_names_z_model = None #np.array(['T_0', 'gamma']) #'T_0', 'u_0'
    measured_parameter_z_model_parameter_limits = None
    #np.array([[5000., 12000.], [-0.5, 0.5], [0.75, 1.75], [-0.5, 0.5]]) #A, S #[5000., 12000.], [-1., 1.]

    log_mass_DM_eV = 9.
    fix_parameters = {'u_0_z_4.2': 8., 'u_0_z_4.6': 8., 'gamma_z_4.2': 1.6, 'gamma_z_4.6': 1.6, 'T_0_z_4.2': 10000.,
                      'T_0_z_4.6': 10000., 'omega_m': 0.3209, 'tau0_2': 1., 'tau0_1': 1.}
    like = likeh.BaryonDarkMatterLikelihoodClass(basedir=emudir, mean_flux=mean_flux_label, #log_mass_DM_eV=log_mass_DM_eV,
                                 measured_parameter_names_z_model=measured_parameter_names_z_model, max_z=max_z,
                                 redshifts=redshifts, pixel_resolution_km_s=pixel_resolution_km_s,
                                 t0_training_value = t0_training_value, t0_parameter_limits=np.array([0.75, 1.25]),
                                 emulator_class=emulator_class, emulator_json_file=emulator_json_file,
                                 use_measured_parameters=use_measured_parameters,
                                 redshift_dependent_parameters=redshift_dependent_parameters,
                                 flux_power_savefile='bDM_batch11_emulator_flux_vectors.hdf5',
                                 flux_power_parallel=True, flux_power_n_process=35, data_class=data_class,
                                 measured_parameter_z_model_parameter_limits=measured_parameter_z_model_parameter_limits,
                                 fix_parameters=fix_parameters, leave_out_validation=leave_out_validation) #,
    #                             dark_matter_parameter_limits=np.array([[-31., -26.],])) #,
    #                             dark_matter_model=likeh.ultra_light_axion_numerical_model,
    #                             dark_matter_parameter_limits=np.array([[-22., -19.],]))
    #UltraLightAxionLikelihoodClass

    #Prior functions
    if prior_function == 'Gaussian':
        prior_function = {'parameter_names': prior_function_args[0], 'means': prior_function_args[1],
                            'standard_deviations': prior_function_args[2]}

    #Convex hull prior
    parameter_names_convex_hull = [['T_0_z_5.0', 'u_0_z_5.0'], #['T_0_z_4.6', 'u_0_z_4.6'], ['T_0_z_4.2', 'u_0_z_4.2'],
                                   ['T_0_z_5.0', 'gamma_z_5.0']] #, ['T_0_z_4.6', 'gamma_z_4.6'],
    #                               ['T_0_z_4.2', 'gamma_z_4.2']]
    prior_function_convex_hull = {'parameter_names': parameter_names_convex_hull,
                                  'use_likelihood_parameter_limits': True}

    #Maximum jumps prior
    parameter_names_maximum_jump = np.array(['T_0', 'u_0'])
    maximum_jumps = np.array([5000., 10.])
    prior_function_maximum_jump = {'parameter_names': parameter_names_maximum_jump,
                                   'maximum_differences': maximum_jumps}

    prior_functions = [prior_function_convex_hull, prior_function] #, prior_function_maximum_jump]
    like.set_log_prior(['convex_hull', 'Gaussian'], prior_functions) #, 'maximum_jump'

    for sdir in subdirs:
        single_likelihood_plot(sdir, like, savedir=savedir, plot=plot, t0=t0_training_value,
                               true_parameter_values=test_simulation_parameters,
                               plot_parameter_indices=plot_parameter_indices, leave_out_validation=leave_out_validation,
                               data_class=data_class, mean_flux_label=mean_flux_label, log_mass_DM_eV=log_mass_DM_eV)
    return like

def single_likelihood_plot(sdir, like, savedir, plot=True, t0=1., true_parameter_values=None,
                           plot_parameter_indices=None, leave_out_validation=None, data_class='Boera',
                           mean_flux_label='free_high_z', log_mass_DM_eV=9.):
    """Make a likelihood and error plot for a single simulation."""
    sname = os.path.basename(os.path.abspath(sdir))
    if t0 != 1.0:
        sname = re.sub(r"\.", "_", "tau0%.3g" % t0) + sname

    if leave_out_validation is None:
        validation_suffix = ''
    else:
        validation_suffix = '_' + str(leave_out_validation[0])
    filename_suffix = '_vary_mass_400_bDM_z_test_full' #%int(log_mass_DM_eV)
    filename_suffix += validation_suffix
    chainfile = os.path.join(savedir, 'chain_' + sname + filename_suffix + '.txt')
    sname = re.sub(r"\.", "_", sname)
    datadir = os.path.join(sdir, "output")

    if true_parameter_values is None:
        true_parameter_values = get_simulation_parameters_s8(sdir, t0=t0)

    # Change prior
    x = 0
    '''like.param_limits[5 + x, 1] = 15000. #12000.
    like.param_limits[np.array([6, 7]), 1] = 15000.
    like.param_limits[np.array([8, 9, 10]) + x, 0] = 0.9
    like.param_limits[11 + x, 1] = 18. #12.
    like.param_limits[np.array([12, 13]) + x, 1] = 18.
    '''
    like.param_limits[3, 1] = 15000.
    like.param_limits[4, 0] = 0.9
    like.param_limits[5, 1] = 18.

    if not os.path.exists(chainfile):
        print('Beginning to sample likelihood at', str(datetime.now()))

        like.do_sampling(chainfile, datadir='use_real_data', nwalkers=150, burnin=100, nsamples=400, #1000, 40000
                         while_loop=False, k_data_max=None, include_emulator_error=True, pool=None) #datadir
        print('Done sampling likelihood at', str(datetime.now()))

    if plot is True:
        if like.use_dark_matter_model:
            #DM_params = likeh.ultra_light_axion_numerical_model_inverse(true_parameter_values[np.arange(6, 9)])
            true_parameter_values = np.delete(true_parameter_values, np.arange(6, 9))
            true_parameter_values = np.concatenate((true_parameter_values, np.array([9., -28.]))) #9.,
            #np.array([DM_params,]))) #np.array([-20.,])
        #omega_m fixed
        true_parameter_values = np.delete(true_parameter_values, 5, axis=0)

        fp_savefile = os.path.join(savedir, 'flux_power_' + sname + ".pdf")
        #make_plot_flux_power_spectra(like, true_parameter_values, datadir, savefile=fp_savefile, t0=t0,
        #                             data_class=data_class, pixel_resolution_km_s=pixel_resolution_km_s,
        #                             mean_flux_label=mean_flux_label)

        savefile = os.path.join(savedir, 'corner_' + sname + filename_suffix + ".pdf")
        plot_parameter_names = like.likelihood_parameter_names[:, 1]
        plot_parameter_limits = like.param_limits
        make_plot(chainfile, savefile, true_parameter_values=None, pnames=plot_parameter_names, #true_parameter_values
                  ranges=plot_parameter_limits, parameter_indices=plot_parameter_indices)


if __name__ == "__main__":
    emulator_base_directory = sys.argv[1] #'/share/data2/keir/Simulations'
    emulator_name = sys.argv[2] #'nCDM_test_emulator'
    test_name = sys.argv[3] #'nCDM_test_thermal2'
    parameters_json = sys.argv[4] #'emulator_params_measured_TDR.json'
    use_measured_parameters = (sys.argv[5].lower() == 'true')
    leave_out_validation = None #np.array([int(sys.argv[6]),])
    redshift_dependent_parameters = True #True #(sys.argv[6].lower() == 'true')

    plotdir = 'Plots'
    gpsavedir=os.path.join(plotdir,"nCDM")
    emud = os.path.join(emulator_base_directory, emulator_name)
    testdirs = os.path.join('/share/data2/keir/Simulations', test_name)

    lyman_data_instance = lyman_data.BoeraData()
    redshifts = lyman_data_instance.redshifts_unique[::-1]
    #redshifts = np.array([4.24,])
    max_z = np.max(redshifts)
    pixel_resolution_km_s = 1.

    # Get test simulation parameters
    t0_test_value = 1.
    test_simulation_number = 0
    test_emulator_instance = cg.nCDMEmulator(testdirs)
    test_emulator_instance.load(dumpfile='emulator_params_TDR_u0_original.json')
    test_simulation_directory = test_emulator_instance.get_outdir(test_emulator_instance.get_parameters()
                                                    [test_simulation_number], extra_flag=test_simulation_number+1)[:-7]

    test_simulation_parameters = test_emulator_instance.get_combined_params()[test_simulation_number]
    #test_simulation_parameters = test_emulator_instance.get_parameters()[test_simulation_number]
    #test_simulation_parameters = np.concatenate((np.array([0., t0_test_value]), test_simulation_parameters))
    test_simulation_parameters = np.concatenate((np.array([t0_test_value,] * 3), test_simulation_parameters))
    #T_0; gamma power laws
    #test_simulation_parameters = np.concatenate((np.array([t0_test_value,] * 3), test_simulation_parameters[:-6], np.array([test_simulation_parameters[-5], 0., test_simulation_parameters[-2], 0.])))
    #gamma power law
    #test_simulation_parameters = np.concatenate((np.array([t0_test_value,] * 3), test_simulation_parameters[:-3], np.array([test_simulation_parameters[-2], 0.])))

    #Prior distribution
    prior_parameter_names = np.array(['tau0_0', 'ns', 'As']) #'tau0_1', 'tau0_2',
    #, 'gamma_z_5.0', 'gamma_z_4.6', 'gamma_z_4.2']) #'T_0_z_5.0', 'T_0_z_4.6', 'T_0_z_4.2'])
    #tau0_0', 'tau0_1', 'tau0_2', 'ns', 'As', 'omega_m', 'T_0_z_5.0', 'T_0_z_4.6', 'T_0_z_4.2', 'gamma_z_5.0', 'gamma_z_4.6', 'gamma_z_4.2'])
    prior_means = test_simulation_parameters[np.array([0, 3, 4])] #1, 2,
    #, 12, 13, 14])] #9, 10, 11])] #0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14])] #, 15, 16])] #7
    print('Gaussian prior means =', prior_means)
    prior_standard_deviations = np.array([0.05, 0.0057, 0.030 * 1.e-9]) #0.05, 0.05,
    #, 0.25, 0.25, 0.25]) #3000., 3000., 3000.]) #, 0.3, 0.3, 0.3]) #, 0.5, 0.5, 0.5])
    #0.05, 0.05, 0.05, 0.0057, 0.030 * 1.e-9, 0.001, 2000., 2000., 2000., 0.25, 0.25, 0.25]) #0.013])
    #0.1, 0.1 * 1.e-9, 0.1])
    prior_function_args = (prior_parameter_names, prior_means, prior_standard_deviations)
    #omega_m fixed
    #test_simulation_parameters = np.delete(test_simulation_parameters, 5, axis=0)

    gplike09 = run_likelihood_test(test_simulation_directory, emud, savedir=gpsavedir, prior_function='Gaussian',
                                   prior_function_args=prior_function_args,
                                   test_simulation_parameters=test_simulation_parameters, plot=True,
                                   mean_flux_label='free_high_z', max_z=max_z, redshifts=redshifts,
                                   pixel_resolution_km_s=pixel_resolution_km_s, t0_training_value=t0_test_value,
                                   emulator_class='nCDM', use_measured_parameters=use_measured_parameters,
                                   redshift_dependent_parameters=redshift_dependent_parameters, data_class='Boera',
                                   emulator_json_file=parameters_json, n_threads_mcmc=23,
                                   leave_out_validation=leave_out_validation)
