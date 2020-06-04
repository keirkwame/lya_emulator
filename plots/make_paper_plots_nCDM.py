import copy as cp
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

import lyaemu.distinct_colours_py3 as lyc
#import lyaemu.coarse_grid as lyc
#import lyaemu.mean_flux as lym
#from lyaemu.likelihood import transfer_function_nCDM

#Define global variables
alpha_model_parameters = np.array([5.54530089e-03, 3.31718138e-01, 6.16422310e+00, 3.31219369e+01])
beta_model_parameters = np.array([-0.02576259, -0.82153762, -0.45096863])
gamma_model_parameters = np.array([-1.29071567e-02, -7.52873377e-01, -1.47076333e+01, -9.60752318e+01])
h_planck = 0.6686
nCDM_parameter_limits = np.array([[0., 0.1], [1., 10.], [-10., 0.]])

def transfer_function_nCDM(k, alpha, beta, gamma):
    """Square root of ratio of linear power spectrum in presence of nCDM with respect to that in presence of CDM."""
    return (1. + ((alpha * k) ** beta)) ** gamma

def ultra_light_axion_alpha_model(log_mass, b, a, m, c):
    """Model for alpha as a function of log ULA mass"""
    return 10. ** ((b * (log_mass ** 3)) + (a * (log_mass ** 2)) + (m * log_mass) + c)

def ultra_light_axion_beta_model(log_mass, a, m, c):
    """Model for beta as a function of log ULA mass"""
    return (a * (log_mass ** 2)) + (m * log_mass) + c

def ultra_light_axion_gamma_model(log_mass, b, a, m, c):
    """Model for gamma as a function of log ULA mass"""
    return -1. * (10. ** ((b * (log_mass ** 3)) + (a * (log_mass ** 2)) + (m * log_mass) + c))

def ultra_light_axion_numerical_model(ultra_light_axion_parameters, nCDM_parameter_limits, h=0.6686):
    """Model to map ultra-light axion parameters to nCDM parameters using a fit to a numerical Einstein-Boltzmann
    solver. Valid for -22 < log ULA mass [eV] < -18"""
    log_mass = ultra_light_axion_parameters[0]
    alpha = ultra_light_axion_alpha_model(log_mass, *alpha_model_parameters)
    beta = ultra_light_axion_beta_model(log_mass, *beta_model_parameters)
    gamma = ultra_light_axion_gamma_model(log_mass, *gamma_model_parameters)
    nCDM_parameters = np.array([alpha * h / h_planck, beta, gamma])

    for i in range(3):
        if nCDM_parameters[i] < nCDM_parameter_limits[i, 0]:
            nCDM_parameters[i] = nCDM_parameter_limits[i, 0]
        if nCDM_parameters[i] > nCDM_parameter_limits[i, 1]:
            nCDM_parameters[i] = nCDM_parameter_limits[i, 1]
    return nCDM_parameters

def plot_transfer_function():
    """Plot the nCDM transfer function."""
    k_log = np.linspace(-1.1, 1.5, num=1000)

    nCDM_ULA = ultra_light_axion_numerical_model(np.array([-22.,]), nCDM_parameter_limits)
    print('nCDM_ULA =', nCDM_ULA)
    alphas = [0., 0.0227, nCDM_ULA[0], 0.1, 0.1, 0.1]
    betas = [1., 2.24, nCDM_ULA[1], 1., 10., 1.]
    gammas = [-1., -4.46, nCDM_ULA[2], -1., -1., -10.]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 6.4)) #5.6))#4.8))
    plot_labels = lambda i: r'$[\alpha, \beta, \gamma] = [%.1f, %i, %i]$' % (alphas[i], betas[i], gammas[i])
    colours = lyc.get_distinct(len(alphas) - 3)
    line_styles = ['-',] * len(alphas)
    line_weights = [2.5,] * len(alphas)
    for i, alpha in enumerate(alphas):
        if i == 0:
            plot_label = r'CDM $[\alpha = 0]$'
            plot_colour = 'black'
        elif i == 1:
            plot_label = r'WDM (2 keV)' #+ r'$[\alpha, \beta, \gamma] = [%.2f, %.1f, %.1f]$'%(alpha, betas[i], gammas[i])
            plot_colour = 'gray'
            line_styles[i] = '--'
        elif i == 2:
            plot_label = r'ULA ($10^{-22}\,\mathrm{eV}$)' #+ plot_labels(i)
            plot_colour = 'gray'
            line_styles[i] = ':'
        else:
            plot_label = plot_labels(i)
            plot_colour = colours[i - 3]
        ax.plot(k_log, transfer_function_nCDM(10. ** k_log, alpha, betas[i], gammas[i]), label=plot_label,
                color=plot_colour, ls=line_styles[i], lw=line_weights[i])

    ax.set_xlabel(r'$\mathrm{log} (k [h\,\mathrm{Mpc}^{-1}])$')
    ax.set_ylabel(r'$T(k)$')
    ax.set_xlim([-1.2, 1.6])
    ax.set_ylim([-0.1, 1.05])
    ax.legend(fontsize=16., frameon=False) #fontsize=16.)
    fig.subplots_adjust(top=0.95, bottom=0.15, right=0.95)
    plt.savefig('/Users/keir/Documents/emulator_paper_axions/transfer.pdf')

def make_error_distribution():
    """Calculate the emulator error distribution for leave-one-out cross-validation."""
    emudir = '/share/data2/keir/Simulations/nCDM_emulator_512'
    emu_json = 'emulator_params_batch18_2_TDR_u0.json'
    flux_power_file = 'batch18_2_emulator_flux_vectors.hdf5'
    n_sims = 93
    mf_instance = lym.FreeMeanFlux()

    emu_instance_full = lyc.nCDMEmulator(emudir, mf=mf_instance)
    emu_instance_full.load(dumpfile=emu_json)
    training_parameters, k, training_flux_powers = emu_instance_full.get_flux_vectors(kfunits='mpc',
                                                    redshifts=emu_instance_full.redshifts, pixel_resolution_km_s=1.,
                                                    use_measured_parameters=True, savefile=flux_power_file)

    GP_mean = [None] * training_flux_powers.shape[0] #np.zeros_like(training_flux_powers)
    GP_std = [None] * training_flux_powers.shape[0] #np.zeros_like(GP_mean)
    for i in range(n_sims):
        print('Getting data for simulation number', i)
        emu_instance = lyc.nCDMEmulator(emudir, mf=mf_instance, leave_out_validation=np.array([i,]))
        emu_instance.load(dumpfile=emu_json)
        test_parameters = emu_instance.get_combined_params(use_all=True)[i]
        test_parameters = np.concatenate((np.array([[1., ], ]), test_parameters.reshape(1, -1)), axis=1)

        GP_instance = emu_instance.get_emulator(use_measured_parameters=True, redshift_dependent_parameters=True,
                                                savefile=flux_power_file)
        test_parameters_tau0 = training_parameters[np.arange(i, training_parameters.shape[0], n_sims)]
        for j, test_parameters_tau0_single in enumerate(test_parameters_tau0):
            print('Getting data for mean flux sample number', j)
            npt.assert_array_equal(test_parameters_tau0_single[1:], test_parameters[0, 1:])
            tau0 = np.ones(emu_instance_full.redshifts.size) * test_parameters_tau0_single[0]
            GP_mean_single, GP_std_single = GP_instance.predict(test_parameters, tau0_factors=tau0)
            idx = (j * n_sims) + i
            GP_mean[idx] = GP_mean_single[0]
            GP_std[idx] = GP_std_single[0]

    return k, emu_instance_full.redshifts, training_parameters, training_flux_powers, np.array(GP_mean), np.array(GP_std), emu_instance_full._get_k_max_emulated_h_Mpc()

def violinplot_error_distribution(distribution='validation'):
    """Make a violin-plot of the emulator error distribution for leave-one-out cross-validation."""
    n_sims = 93
    n_LH = 50
    n_BO = n_sims - n_LH
    n_mf = 2
    n_k_cut = 45
    n_k_data = 16
    k_bins = np.concatenate((np.repeat(1, 15), np.repeat(2, 15), np.repeat(3, 15)))

    #Load data
    validation_data = np.load('/Users/keir/Software/lya_emulator/plots/cross_validation.npz')
    k = validation_data['k']
    redshifts = validation_data['z']
    p = validation_data['p']
    f = validation_data['f']
    m = validation_data['m']
    s = validation_data['s']
    k_max = validation_data['k_max']
    f_cut = np.concatenate((f[:, :n_k_cut], f[:, k.size:k.size+n_k_cut], f[:, 2*k.size:(2*k.size)+n_k_cut]), axis=1)

    #Load Boera+ data
    s_data = np.zeros(n_k_data * redshifts.size)
    for i, z in enumerate(redshifts):
        data_all = np.genfromtxt('/Users/keir/Software/lya_emulator/lyaemu/data/Boera_HIRES_UVES_flux_power/flux_power_z_%.1f.dat'%z,
                                    skip_header=5, skip_footer=1)
        s_data[(i * n_k_data): ((i + 1) * n_k_data)] = data_all[:, 3]
        if i == 0:
            k_data = 10. ** data_all[:, 0]

    k_expand = np.tile(k[:n_k_cut], (n_sims * n_mf, redshifts.size))
    vel_fac = (1. + redshifts[np.newaxis, :]) / (100. *
                np.sqrt((p[:(n_sims*n_mf), 3][:, np.newaxis] *
                ((1. + redshifts[np.newaxis, :]) ** 3.)) + (1. - p[:(n_sims*n_mf), 3][:, np.newaxis])))
    vel_fac = np.repeat(vel_fac, n_k_cut, axis=1)
    k_kms = k_expand * vel_fac

    k_nearest_data = np.ones_like(k_kms) * k_data[0]
    s_data_expand = np.zeros_like(k_kms)
    #k_index = np.zeros_like(k_kms)
    for i, k_datum in enumerate(k_data):
        for j, z in enumerate(redshifts):
            slice_array = np.absolute(k_kms[:, (j * n_k_cut): ((j + 1) * n_k_cut)] - k_datum) <=\
                          np.absolute(k_kms - k_nearest_data)[:, (j * n_k_cut): ((j + 1) * n_k_cut)]
            k_nearest_data[:, (j * n_k_cut): ((j + 1) * n_k_cut)][slice_array] = k_datum
            s_data_expand[:, (j * n_k_cut): ((j + 1) * n_k_cut)][slice_array] = s_data[(j * n_k_data) + i]

    #LH-BO cut
    if distribution == 'validation':
        errors = (m - f_cut) / s
        kernel_bw = 0.5
        colours = lyc.get_distinct(3)
        ylim = [-4.9, 4.9]
        ylabel = r'$(\mathrm{Mean - Truth})\,/\,\sigma$'
        text_height = 0.85
        legend_loc = 'upper left'
        save_file = 'validation.pdf'
    elif distribution == 'data':
        errors = np.log10(s[:(n_sims * n_mf)] / s_data_expand) #np.log10(
        errors_real = np.log10(np.absolute(m - f_cut)[:(n_sims * n_mf)] / s_data_expand)
        kernel_bw = 'scott' #0.01
        colours = lyc.get_distinct(4)
        ylim = [-5., 2.5]
        ylabel = r'$\mathrm{log}\,(\sigma_\mathrm{Theory}\,/\,\sigma_\mathrm{Data})$'
        text_height = 0.1
        legend_loc = 'lower center'
        save_file = 'data_error.pdf'
    LH_cut = np.sort([np.arange(i, i+(n_LH + n_BO), 1) for i in range(0, n_sims*n_mf, n_sims)], axis=None)
    BO_cut = np.sort([np.arange(i+n_LH, i+n_sims, 1) for i in range(0, n_sims*n_mf, n_sims)], axis=None)
    errors_LH = errors[LH_cut]
    errors_BO = errors[BO_cut]
    errors_list = [errors_BO, errors_LH]
    if distribution == 'data':
        errors_list_real = [errors_real[BO_cut], errors_real[LH_cut]]

    #k bins
    #k_bins_input = k[:n_k_cut]
    k_bins_input = cp.deepcopy(k_bins)
    k_bin_LH = np.tile(k_bins_input, ((n_LH + n_BO) * n_mf, redshifts.size))
    k_bin_BO = np.tile(k_bins_input, (n_BO * n_mf, redshifts.size))
    k_bin_list = [k_bin_BO, k_bin_LH]

    #BO
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(6.4*2., 6.4*2.5)) #6.4*1.))
    data_frames = [None] * redshifts.size * len(errors_list)
    for i, z in enumerate(redshifts):
        #z cut
        z_cut = np.arange(i*n_k_cut, (i+1)*n_k_cut, 1)
        for j in range(len(errors_list)):
            idx = (j * redshifts.size) + i
            if distribution == 'validation':
                k_bin_df = np.tile(np.ravel(k_bin_list[j][:, z_cut]), 21)
                errors_df = np.concatenate((np.ravel(errors_list[j][:, z_cut]), npr.normal(size=errors_list[j][:, z_cut].size * 20)))
                if j == 0:
                    samples_label = r'Validation test [optimisation simulations]'
                else:
                    samples_label = r'Validation test [all simulations]'
                samples_label_real = 'Unit Gaussian model'
                if j == 0:
                    violin_colours = {samples_label: colours[0], samples_label_real: colours[2]}
                else:
                    violin_colours = {samples_label: colours[1], samples_label_real: colours[2]}
                split_cut_df = ([samples_label,] * errors_list[j][:, z_cut].size) + ([samples_label_real,] * errors_list[j][:, z_cut].size * 20)
            elif distribution == 'data':
                k_bin_df = np.tile(np.ravel(k_bin_list[j][:, z_cut]), 2)
                errors_df = np.concatenate(
                    (np.ravel(errors_list[j][:, z_cut]), np.ravel(errors_list_real[j][:, z_cut])))
                if j == 0:
                    samples_label = r'Emulator / Data [optimisation simulations]'
                    samples_label_real = r'$|\mathrm{Mean - Truth}|$ / Data [optimisation simulations]'
                    violin_colours = {samples_label: colours[0], samples_label_real: colours[1]}
                else:
                    samples_label = r'Emulator / Data [all simulations]'
                    samples_label_real = r'$|\mathrm{Mean - Truth}|$ / Data [all simulations]'
                    violin_colours = {samples_label: colours[2], samples_label_real: colours[3]}
                split_cut_df = ([samples_label,] * errors_list[j][:, z_cut].size) + ([samples_label_real,] * errors_list[j][:, z_cut].size)
            print(i, j, k_bin_df.shape, errors_df.shape, len(split_cut_df))

            data_frames[idx] = pd.DataFrame({'kbin': k_bin_df, 'ErrorSigmas': errors_df, 'Distribution': split_cut_df})
            sb.violinplot(data_frames[idx].kbin, data_frames[idx].ErrorSigmas, data_frames[idx].Distribution,
                          ax=axes[idx], scale='width', bw=kernel_bw, inner=None, split=True, palette=violin_colours, cut=0.,
                          linewidth=2.5, saturation=1.)
            # #cut=0

            axes[idx].set(ylim=ylim)
            axes[idx].axhline(y=0., color='black', ls=':', lw=2.5)
            axes[idx].axvline(x=0., color='black', ls='-', lw=2.5)
            axes[idx].axvline(x=1., color='black', ls='-', lw=2.5)
            axes[idx].axvline(x=2., color='black', ls='-', lw=2.5)
            axes[idx].text(0.9, text_height, r'$z = %.1f$'%redshifts[i], transform=axes[idx].transAxes) #, fontsize=16.)
            axes[idx].get_legend().remove()
            axes[idx].set(ylabel=ylabel)
            #if idx < 5:
            #axes[idx].xaxis.set_ticklabels([])
            if (idx == 0) or (idx == 3):
                axes[idx].legend(loc=legend_loc, frameon=True, facecolor='white', fancybox=False, shadow=False,
                                 framealpha=1., edgecolor='white', fontsize=15.)
            if idx < 5:
                axes[idx].xaxis.set_visible(False)
            else:
                axes[idx].set(xlabel=r'$k [h\,\mathrm{Mpc}^{-1}]$ bin')
            if distribution == 'data':
                axes[idx].axhline(y=1., color='black', ls=':', lw=2.5)
                axes[idx].axhline(y=-1., color='black', ls=':', lw=2.5)

    fig.subplots_adjust(top=0.99, bottom=0.05, right=0.95, hspace=0.1)
    plt.savefig('/Users/keir/Documents/emulator_paper_axions/' + save_file)
    return k, z, p, f, m, s, k_data, s_data, k_max, data_frames

if __name__ == "__main__":
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=18.)

    plt.rc('axes', linewidth=1.5)
    plt.rc('xtick.major', width=1.5)
    plt.rc('xtick.minor', width=1.5)
    plt.rc('ytick.major', width=1.5)
    plt.rc('ytick.minor', width=1.5)

    #plot_transfer_function()
    #k, z, p, f, m, s, k_max = make_error_distribution()
    k, z, p, f, m, s, k_data, s_data, k_max, data_frames = violinplot_error_distribution(distribution='data')
