import os
import json
import h5py
import copy as cp
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import scipy.optimize as spo
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import getdist as gd
import getdist.plots as gdp

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

def ultra_light_axion_numerical_model_inverse(nCDM_parameters, h=0.6686):
    """Inverse of numerical ultra-light axion model. Valid for -22 < log ULA mass [eV] < -18"""
    nCDM_corrected = nCDM_parameters
    nCDM_corrected[0] = np.log10(nCDM_corrected[0] * h_planck / h)
    nCDM_corrected[2] = np.log10(-1. * nCDM_corrected[2])
    log_mass = [None] * nCDM_parameters.shape[0]

    for i, model_parameters in enumerate([alpha_model_parameters, beta_model_parameters, gamma_model_parameters]):
        model_coefficients = cp.deepcopy(model_parameters)
        model_coefficients[-1] -= nCDM_corrected[i]
        model_roots = np.roots(model_coefficients)
        log_mass[i] = model_roots[(model_roots <= -18.) * (model_roots >= -22.)][0].real
        print('log mass =', log_mass[i])
        if i == 1:
            assert ((log_mass[i] - log_mass[0]) / log_mass[0]) <= 1.e-10

    return log_mass[0]

def plot_numerical_convergence():
    """Plot the numerical convergence of nCDM simulations; the effect of imperfect spectrograph modelling; and the
    effect of redshift evolution within sections of forest."""
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6.4*1., 6.4*2.))
    colours = lyc.get_distinct(4)
    colours = ['black',] + colours
    redshifts = [4.95, 4.58, 4.24]
    flux_fnames = [None] * 12
    #flux_fnames[0] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/mf_fixed10_emulator_flux_vectors_512_256.hdf5'
    #flux_fnames[1] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/mf10_emulator_flux_vectors_768_WDM.hdf5'
    #flux_fnames[0] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/mfraw_emulator_flux_vectors_512_256.hdf5'
    #flux_fnames[1] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/mfraw_emulator_4_flux_vectors_768_WDM.hdf5'
    flux_fnames[0] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/cc_emulator_flux_vectors_0_01_512_256.hdf5'
    flux_fnames[1] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/cc_emulator_4_flux_vectors_0_01_768_WDM.hdf5'
    flux_fnames[2] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/mfraw_emulator_flux_vectors_512_256.hdf5'
    flux_fnames[3] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/mfraw_emulator_15_flux_vectors.hdf5'
    flux_fnames[4] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/mfraw_emulator_flux_vectors.hdf5'
    flux_fnames[5] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/mfraw_emulator_flux_vectors_spec_res_6.hdf5'
    flux_fnames[6] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/mfraw_emulator_flux_vectors_spec_res_7_2.hdf5'
    flux_fnames[7] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/mfraw_emulator_flux_vectors_spec_res_6_6.hdf5'
    flux_fnames[8] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/mfraw_emulator_flux_vectors_spec_res_5_4.hdf5'
    flux_fnames[9] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/mfraw_emulator_flux_vectors_spec_res_4_8.hdf5'
    flux_fnames[10] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/cc_emulator_flux_vectors_mfevol.hdf5'
    flux_fnames[11] = '/Users/keir/Documents/emulator_data/emulator_flux_vectors/convergence/mfraw_emulator_flux_vectors_z_evol.hdf5'

    for i in range(4):
        power_arrays = [None] * 3
        if i == 2:
            power_arrays = [None] * 5
            #labels = [r'768', r'512', r'256']
            labels = [r'Exact spec. res.', r'$+ 20\%$ error', r'$+ 10\%$', r'$- 10\%$', r'$- 20\%$']

            flux_file = h5py.File(flux_fnames[6])
            k_log = np.log10(np.array(flux_file['kfkms'])[0]) #3, n_k #18
            print('Parameters =', np.array(flux_file['params'])[0])
            power_arrays[1] = np.array(flux_file['flux_vectors'])[0] #3 x n_k #7.2

            flux_file = h5py.File(flux_fnames[7])
            print('Parameters =', np.array(flux_file['params'])[0]) #19
            power_arrays[2] = np.array(flux_file['flux_vectors'])[0] #6.6

            flux_file = h5py.File(flux_fnames[8])
            print('Parameters =', np.array(flux_file['params'])[0]) #19
            power_arrays[3] = np.array(flux_file['flux_vectors'])[0] #5.4

            flux_file = h5py.File(flux_fnames[9])
            print('Parameters =', np.array(flux_file['params'])[0]) #19
            power_arrays[4] = np.array(flux_file['flux_vectors'])[0] #4.8

            flux_file = h5py.File(flux_fnames[5])
            print('Parameters =', np.array(flux_file['params'])[0]) #39
            power_arrays[0] = np.array(flux_file['flux_vectors'])[0] #3 x n_k #6
        elif i == 1:
            labels = [r'$(17.5\,h^{-1}\,\mathrm{Mpc})^3$ box volume', r'$(15\,h^{-1}\,\mathrm{Mpc})^3$',
                      r'$(10\,h^{-1}\,\mathrm{Mpc})^3$'] #[r'10', r'15', r'17.5']

            flux_file = h5py.File(flux_fnames[2])
            k_log = np.log10(np.array(flux_file['kfkms'])[0]) #3, n_k
            print('Parameters =', np.array(flux_file['params'])[0])
            power_arrays[2] = np.array(flux_file['flux_vectors'])[0] #3 x n_k #10

            flux_file = h5py.File(flux_fnames[3])
            print('Parameters =', np.array(flux_file['params'])[0])
            power_arrays[1] = np.zeros_like(power_arrays[2]).reshape(3, -1)
            power_raw = np.array(flux_file['flux_vectors'])[0].reshape(3, -1) #15
            for a in range(3):
                power_arrays[1][a, :] = 10. ** np.interp(k_log[a], np.log10(np.array(flux_file['kfkms'])[0])[a],
                                                         np.log10(power_raw[a]))
            power_arrays[1] = np.ravel(power_arrays[1])

            flux_file = h5py.File(flux_fnames[4])
            print('Parameters =', np.array(flux_file['params'])[0])
            power_arrays[0] = np.zeros_like(power_arrays[2]).reshape(3, -1)
            power_raw = np.array(flux_file['flux_vectors'])[0].reshape(3, -1) #17.5
            for a in range(3):
                power_arrays[0][a, :] = 10. ** np.interp(k_log[a], np.log10(np.array(flux_file['kfkms'])[0])[a],
                                                         np.log10(power_raw[a]))
                #print(k_log[a], np.log10(np.array(flux_file['kfkms'])[0])[a], power_raw[a])
            power_arrays[0] = np.ravel(power_arrays[0])
            #return power_arrays, k_log
        elif i == 0:
            labels = [r'$2 \times 768^3$ particles', r'$2 \times 512^3$', r'$2 \times 256^3$']

            flux_file = h5py.File(flux_fnames[1])
            k_log = np.log10(np.array(flux_file['kfkms'])[0]) #3, n_k #36-38
            print('Parameters =', np.array(flux_file['params'])[0])
            power_arrays[2] = np.array(flux_file['flux_vectors'])[0] #3 x n_k #256
            print('Parameters =', np.array(flux_file['params'])[1])
            power_arrays[1] = np.array(flux_file['flux_vectors'])[1] #512
            print('Parameters =', np.array(flux_file['params'])[2])
            power_arrays[0] = np.array(flux_file['flux_vectors'])[2] #768
        elif i == 3:
            power_arrays = [None] * 2
            labels = [r'Uniform mean flux', r'Mean flux evolution [uncorrected]']

            flux_file = h5py.File(flux_fnames[10])
            k_log = np.log10(np.array(flux_file['kfkms'])[0]) #3, n_k
            print('Parameters =', np.array(flux_file['params'])[0])
            power_arrays[0] = np.array(flux_file['flux_vectors'])[0] #3 x n_k #Constant

            flux_file = h5py.File(flux_fnames[11])
            print('Parameters =', np.array(flux_file['params'])[0])
            power_arrays[1] = np.array(flux_file['flux_vectors'])[0] #Evolution
            #print('Parameters =', np.array(flux_file['params'])[0])
            #power_arrays[2] = np.array(flux_file['flux_vectors'])[0] #Evolution
        for j in range(1):
            for k in range(0, len(power_arrays)):
                print(i, j, k)
                data = np.genfromtxt(
                    '/Users/keir/Software/lya_emulator/lyaemu/data/Boera_HIRES_UVES_flux_power/flux_power_z_%.1f.dat' % redshifts[j],
                    skip_header=5, skip_footer=1)

                power_ratio = (power_arrays[k] / power_arrays[0])[(j * k_log.shape[1]): ((j + 1) * k_log.shape[1])]
                #if i == 0:
                #    power_ratio = np.ones_like(power_ratio) * 0.
                axes[i].plot(k_log[j], power_ratio, color=colours[k], label=labels[k], lw=2.5)
                #axes[i].fill_between(data[:, 0], y1=(1. + (1. * data[:, 3] / data[:, 2])),
                #                                y2=(1. + (-1. * (data[:, 3] / data[:, 2]))))
            #axes[i].axhline(y=1., color='black', ls='-', lw=2.5)
            #axes[i].axhline(y=0.9, color='black', ls=':', lw=2.5)
            #axes[i].axhline(y=1.1, color='black', ls=':', lw=2.5)
            axes[i].set_xlim([-2.2, -0.7])
            axes[i].set_ylim([0.75, 1.25])
            axes[i].set_ylabel(r'$P_\mathrm{f}^i (k_\mathrm{f}) / P_\mathrm{f}^\mathrm{fiducial} (k_\mathrm{f})$')
            if i < 3:
                axes[i].set_xticklabels([])
        axes[-1].set_xlabel(r'$\mathrm{log} (k_\mathrm{f} [\mathrm{s}\,\mathrm{km}^{-1}])$')
        if i == 2:
            ncol = 2
        else:
            ncol = 1
        axes[i].legend(fontsize=15., frameon=True, facecolor='white', fancybox=False, shadow=False, framealpha=1.,
                       edgecolor='white', ncol=ncol)
        #frameon=True, facecolor='white', fancybox=False, shadow=False, framealpha=1., edgecolor='white'

    fig.subplots_adjust(top=0.99, bottom=0.08, right=0.95, hspace=0.05, left=0.15)
    plt.savefig('/Users/keir/Documents/emulator_paper_axions/systematics.pdf')
    return 0, 0

def plot_transfer_function(y='transfer'):
    """Plot the nCDM transfer function."""
    if y == 'transfer':
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
            if y == 'flux_power':
                flux_file = h5py.File(
                    '/Users/keir/Documents/emulator_data/emulator_flux_vectors/mf_fixed10_emulator_flux_vectors_CDM.hdf5')
                params = np.array(flux_file['params'])[11]
                print('Parameters =', params)
                k_log = np.log10(np.array(flux_file['kfkms'])[11, 0])
                power_CDM = np.array(flux_file['flux_vectors'])[11, :k_log.shape[0]]
                power_ratio = power_CDM / power_CDM
        elif i == 1:
            plot_label = r'WDM (2 keV)' #+ r'$[\alpha, \beta, \gamma] = [%.2f, %.1f, %.1f]$'%(alpha, betas[i], gammas[i])
            plot_colour = 'gray'
            line_styles[i] = '--'
            if y == 'flux_power':
                flux_file = h5py.File(
                    '/Users/keir/Documents/emulator_data/emulator_flux_vectors/mf10_emulator_flux_vectors_WDM.hdf5')
                params = np.array(flux_file['params'])[20]
                print('Parameters =', params)
                npt.assert_array_equal(k_log, np.log10(np.array(flux_file['kfkms'])[20, 0]))
                power_ratio = np.array(flux_file['flux_vectors'])[20, :k_log.shape[0]]
                power_ratio /= power_CDM
        elif i == 2:
            plot_label = r'ULA ($10^{-22}\,\mathrm{eV}$)' #+ plot_labels(i)
            plot_colour = 'gray'
            line_styles[i] = ':'
            if y == 'flux_power':
                flux_file = h5py.File(
                    '/Users/keir/Documents/emulator_data/emulator_flux_vectors/mf10_emulator_flux_vectors_WDM.hdf5')
                params = np.array(flux_file['params'])[20]
                print('Parameters =', params)
                npt.assert_array_equal(k_log, np.log10(np.array(flux_file['kfkms'])[20, 0]))
                power_ratio = np.array(flux_file['flux_vectors'])[20, :k_log.shape[0]]
                power_ratio /= power_CDM
        else:
            plot_label = plot_labels(i)
            plot_colour = colours[i - 3]
            if y == 'flux_power':
                flux_file = h5py.File(
                    '/Users/keir/Documents/emulator_data/emulator_flux_vectors/mf_emulator_flux_vectors_extremal.hdf5')
                params = np.array(flux_file['params'])[53+i]
                print('Parameters =', params)
                npt.assert_array_equal(k_log, np.log10(np.array(flux_file['kfkms'])[53+i, 0]))
                power_ratio = np.array(flux_file['flux_vectors'])[53+i, :k_log.shape[0]]
                power_ratio /= power_CDM
        if y == 'transfer':
            ax.plot(k_log, transfer_function_nCDM(10. ** k_log, alpha, betas[i], gammas[i]), label=plot_label,
                    color=plot_colour, ls=line_styles[i], lw=line_weights[i])
        elif y == 'flux_power':
            ax.plot(k_log, power_ratio, label=plot_label, color=plot_colour, ls=line_styles[i], lw=line_weights[i])

    if y == 'transfer':
        ax.set_xlabel(r'$\mathrm{log} (k [h\,\mathrm{Mpc}^{-1}])$')
        ax.set_ylabel(r'$T(k)$')
        ax.set_xlim([-1.2, 1.6])
        save_file = 'transfer.pdf'
    elif y == 'flux_power':
        ax.set_xlabel(r'$\mathrm{log} (k_\mathrm{f} [\mathrm{s}\,\mathrm{km}^{-1}])$')
        ax.set_ylabel(r'$P_\mathrm{f}^\mathrm{nCDM}(k_\mathrm{f}) / P_\mathrm{f}^\mathrm{CDM}(k_\mathrm{f})$')
        ax.set_xlim([-2.2, -0.7])
        save_file = 'flux_power.pdf'
    ax.set_ylim([-0.1, 1.05])
    ax.legend(fontsize=16., frameon=False) #fontsize=16.)
    fig.subplots_adjust(top=0.95, bottom=0.15, right=0.95)
    plt.savefig('/Users/keir/Documents/emulator_paper_axions/' + save_file)

def plot_convergence():
    """Plot the marginalised posterior summary statistics convergence."""
    convergence_data = np.load('/Users/keir/Software/lya_emulator/plots/convergence.npz')
    posterior_means = convergence_data['arr_0']
    posterior_limits = convergence_data['arr_1']
    sim_num = np.concatenate((np.array([0, 5, 10, 14, 16, 18]), np.arange(19, 44, 2)))
    slice_array = np.concatenate((np.arange(0, 15), np.arange(16, 23, 2)))
    redshifts = [4.95, 4.58, 4.24]

    one_sigma = (posterior_limits[slice_array, :, 2] - posterior_limits[slice_array, :, 1]) / 2.
    two_sigma = (posterior_limits[slice_array, :, 3] - posterior_limits[slice_array, :, 0]) / 2.
    mean_diff = (posterior_means[slice_array[1:]] - posterior_means[slice_array[:-1]]) / (two_sigma[:-1] / 2.)
    one_sigma_diff = (one_sigma[1:] - one_sigma[:-1]) / (two_sigma[:-1] / 2.)
    two_sigma_diff = (two_sigma[1:] - two_sigma[:-1]) / (two_sigma[:-1] / 2.)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6.4*2., 6.4*2.5))
    plot_labels = [r'$\tau_0 (z = %.1f)$'%redshifts[0], r'$\tau_0 (z = %.1f)$'%redshifts[1],
                   r'$\tau_0 (z = %.1f)$'%redshifts[2], r'$n_\mathrm{s}$', r'$A_\mathrm{s}$',
                   r'$T_0 (z = %.1f)$'%redshifts[0], r'$T_0 (z = %.1f)$'%redshifts[1], r'$T_0 (z = %.1f)$'%redshifts[2],
                   r'$\widetilde{\gamma} (z = %.1f)$'%redshifts[0], r'$\widetilde{\gamma} (z = %.1f)$'%redshifts[1],
                   r'$\widetilde{\gamma} (z = %.1f)$'%redshifts[2], r'$u_0 (z = %.1f)$'%redshifts[0],
                   r'$u_0 (z = %.1f)$'%redshifts[1], r'$u_0 (z = %.1f)$'%redshifts[2],
                   r'$\log(m_\mathrm{a} [\mathrm{eV}])$']
    colours = lyc.get_distinct(8)
    colours += colours[:7]
    line_style = '-'

    for i in range(len(plot_labels)):
        if i > 7:
            line_style = '--'
        axes[0].plot(sim_num[1:], mean_diff[:, i], label=plot_labels[i], color=colours[i], lw=2.5, ls=line_style)
        axes[1].plot(sim_num[1:], one_sigma_diff[:, i], color=colours[i], lw=2.5, ls=line_style)
        axes[2].plot(sim_num[1:], two_sigma_diff[:, i], color=colours[i], lw=2.5, ls=line_style)

    axes[0].axhline(y=0.5, color='black', ls=':', lw=2.5)
    axes[0].axhline(y=-0.5, color='black', ls=':', lw=2.5)
    axes[1].axhline(y=0.4, color='black', ls=':', lw=2.5)
    axes[1].axhline(y=-0.4, color='black', ls=':', lw=2.5)
    axes[2].axhline(y=0.75, color='black', ls=':', lw=2.5)
    axes[2].axhline(y=-0.75, color='black', ls=':', lw=2.5)

    axes[2].set_xlabel(r'Optimization simulation number')
    axes[0].set_ylabel(r'Number of sigma shift [posterior means]')
    axes[1].set_ylabel(r'Number of sigma shift [1 sigma]')
    axes[2].set_ylabel(r'Number of sigma shift [2 sigma]')
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[0].set_ylim([-3.9, 3.9])
    axes[0].legend(frameon=False, loc='lower right', ncol=3, fontsize=17.)
    fig.subplots_adjust(top=0.99, bottom=0.05, right=0.95, hspace=0.05)
    plt.savefig('/Users/keir/Documents/emulator_paper_axions/convergenceUS.pdf')

    return posterior_means, posterior_limits

def plot_exploration():
    """Plot the exploration convergence."""
    exploration = np.load('/Users/keir/Software/lya_emulator/plots/exploration.npy')
    sim_num = np.arange(exploration.size) + 1

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 6.4))
    colour = lyc.get_distinct(1)
    ax.plot(sim_num, exploration, color=colour[0], lw=2.5)
    #ax.axhline(y=0., color='black', ls=':', lw=2.5)
    ax.set_xlabel(r'Optimization simulation number')
    ax.set_ylabel(r'Exploration')
    fig.subplots_adjust(top=0.95, bottom=0.15, right=0.95)
    plt.savefig('/Users/keir/Documents/emulator_paper_axions/explorationUS.pdf')

def make_error_distribution():
    """Calculate the emulator error distribution for leave-one-out cross-validation."""
    emudir = '/share/data2/keir/Simulations/nCDM_emulator_512'
    emu_json = 'emulator_params_TDR_u0_original_emu50.json' #'emulator_params_batch18_2_TDR_u0.json'
    flux_power_file = 'emulator_flux_vectors2.hdf5' #'batch18_2_emulator_flux_vectors.hdf5'
    n_sims = 50 #93
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
    n_sims = 50 #93
    n_LH = 50
    n_BO = n_sims - n_LH
    n_mf = 10 #2
    n_k_cut = 45
    n_k_data = 16
    k_bins = np.concatenate((np.repeat(1, 15), np.repeat(2, 15), np.repeat(3, 15)))

    #Load data
    validation_data = np.load('/home/keir/Software/lya_emulator/plots/cross_validation_bDM2.npz')
    k = validation_data['k']
    print('Emulator wavenumbers =', k)
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
        data_all = np.genfromtxt('/home/keir/Software/lya_emulator/lyaemu/data/Boera_HIRES_UVES_flux_power/flux_power_z_%.1f.dat'%z,
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
        kernel_bw = 'scott' #0.5
        colours = lyc.get_distinct(3)
        ylim = [-4.9, 4.9]
        ylabel = r'$(\mathrm{Mean - Truth})\,/\,\sigma$'
        text_height = 0.85
        legend_loc = 'upper left'
        save_file = 'validation_bDM.pdf'
    elif distribution == 'data':
        errors = np.log10(s[:(n_sims * n_mf)] / s_data_expand) #np.log10(
        errors_real = np.log10(np.absolute(m - f_cut)[:(n_sims * n_mf)] / s_data_expand)
        kernel_bw = 'scott' #0.2
        colours = lyc.get_distinct(4)
        ylim = [-5., 2.5]
        ylabel = r'$\mathrm{log}\,(\sigma_\mathrm{Theory}\,/\,\sigma_\mathrm{Data})$'
        text_height = 0.1
        legend_loc = 'lower center'
        save_file = 'data_error_bDM.pdf'
    LH_cut = np.sort([np.arange(i, i+(n_LH + n_BO), 1) for i in range(0, n_sims*n_mf, n_sims)], axis=None)
    BO_cut = np.sort([np.arange(i+n_LH, i+n_sims, 1) for i in range(0, n_sims*n_mf, n_sims)], axis=None)
    errors_LH = errors[LH_cut]
    errors_BO = errors[BO_cut]
    errors_list = [errors_BO, errors_LH]
    if distribution == 'data':
        errors_list_real = [errors_real[BO_cut], errors_real[LH_cut]] #[errors_real[BO_cut], errors_real[BO_cut]]

    #k bins
    #k_bins_input = k[:n_k_cut]
    k_bins_input = cp.deepcopy(k_bins)
    k_bin_LH = np.tile(k_bins_input, ((n_LH + n_BO) * n_mf, redshifts.size))
    k_bin_BO = np.tile(k_bins_input, (n_BO * n_mf, redshifts.size))
    k_bin_list = [k_bin_BO, k_bin_LH]

    #BO
    if distribution == 'data':
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6.4 * 2., 6.4 * 1.25))  # 6.4*1.))
    else:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6.4*2., 6.4*1.25)) #6.4*1.))
    data_frames = [None] * redshifts.size * len(errors_list)
    for i, z in enumerate(redshifts):
        #z cut
        z_cut = np.arange(i*n_k_cut, (i+1)*n_k_cut, 1)
        for j in [1,]: #'data' #range(len(errors_list)):
            idx = (j * redshifts.size) + i
            if distribution == 'validation':
                k_bin_df = np.concatenate((np.ravel(k_bin_list[j-1][:, z_cut]), np.ravel(k_bin_list[j][:, z_cut])))
                #np.tile(np.ravel(k_bin_list[j][:, z_cut]), 2) #1)
                errors_df = np.concatenate((np.ravel(errors_list[j-1][:, z_cut]), np.ravel(errors_list[j][:, z_cut])))
                #npr.normal(size=errors_list[j][:, z_cut].size * 20)
                #if j == 0:
                samples_label0 = r'Validation test [optimization simulations]'
                #else:
                samples_label1 = r'Validation test [all simulations]'
                samples_label_real = 'Unit Gaussian model'
                if j == 0:
                    violin_colours = {samples_label: colours[0], samples_label_real: colours[2]}
                else:
                    violin_colours = {samples_label0: colours[1], samples_label1: colours[2]}
                split_cut_df = ([samples_label0,] * errors_list[j-1][:, z_cut].size) +\
                               ([samples_label1,] * errors_list[j][:, z_cut].size)
                #([samples_label_real,] * errors_list[j][:, z_cut].size * 20)
                axes_idx = idx - 3
                violinplot_split = False
            elif distribution == 'data':
                k_bin_df = np.concatenate((np.ravel(k_bin_list[j][:, z_cut]), np.ravel(k_bin_list[j][:, z_cut]))) #j[2] --> 0 for BO only
                #np.tile(np.ravel(k_bin_list[j][:, z_cut]), 2)
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
                split_cut_df = ([samples_label,] * errors_list[j][:, z_cut].size) + ([samples_label_real,] * errors_list[j][:, z_cut].size) #0 --> j[2]
                axes_idx = idx - 3
                violinplot_split = True
            print(i, j, k_bin_df.shape, errors_df.shape, len(split_cut_df))

            if distribution == 'validation':
                axes[axes_idx].axhspan(ymin=-3., ymax=3., alpha=0.075, color=colours[0])
                axes[axes_idx].axhspan(ymin=-1., ymax=1., alpha=0.15, color=colours[0])

            data_frames[idx] = pd.DataFrame({'kbin': k_bin_df, 'ErrorSigmas': errors_df, 'Distribution': split_cut_df})
            sb.violinplot(data_frames[idx].kbin, data_frames[idx].ErrorSigmas, data_frames[idx].Distribution,
                          ax=axes[axes_idx], scale='width', bw=kernel_bw, inner=None, split=violinplot_split, palette=violin_colours, cut=0.,
                          linewidth=2.5, saturation=1.) #split=True

            axes[axes_idx].set(ylim=ylim)
            axes[axes_idx].axhline(y=0., color='black', ls=':', lw=2.5)
            axes[axes_idx].axvline(x=0., color='black', ls='-', lw=2.5)
            axes[axes_idx].axvline(x=1., color='black', ls='-', lw=2.5)
            axes[axes_idx].axvline(x=2., color='black', ls='-', lw=2.5)
            axes[axes_idx].text(0.9, text_height, r'$z = %.1f$'%redshifts[i], transform=axes[axes_idx].transAxes) #, fontsize=16.)
            axes[axes_idx].get_legend().remove()
            axes[axes_idx].set(ylabel=ylabel)
            #if idx < 5:
            #axes[idx].xaxis.set_ticklabels([])
            if (idx == 0) or (idx == 3):
                axes[axes_idx].legend(loc=legend_loc, frameon=True, facecolor='white', fancybox=False, shadow=False,
                                 framealpha=1., edgecolor='white', fontsize=15.)
            if idx < 5:
                axes[axes_idx].xaxis.set_visible(False)
            else:
                axes[axes_idx].set(xlabel=r'$k [h\,\mathrm{Mpc}^{-1}]$ bin')
            if distribution == 'data':
                axes[axes_idx].axhline(y=1., color='black', ls=':', lw=2.5)
                axes[axes_idx].axhline(y=-1., color='black', ls=':', lw=2.5)

    if distribution == 'data':
        bottom_adjust = 0.1
    else:
        bottom_adjust = 0.1
    fig.subplots_adjust(top=0.99, bottom=bottom_adjust, right=0.95, hspace=0.1)
    plt.savefig('/home/keir/Plots/nCDM/' + save_file) #'/Users/keir/Documents/emulator_paper_axions/' + save_file)
    return k, z, p, f, m, s, k_data, s_data, k_max, data_frames

def plot_posterior(parameters='all'):
    """Make a triangle plot of marginalised 1D and 2D posteriors."""
    if parameters == 'all':
        n_chains = 4
        save_file = 'posterior42_sims2.pdf'
    elif parameters == 'logma':
        n_chains = 6
        save_file = 'posterior_logma2.pdf'
    elif parameters == 'PRL':
        n_chains = 1
        save_file = 'posterior_PRL2.pdf'
    elif parameters == 'mock':
        n_chains = 2
        save_file = 'posterior42_mock_all.pdf'

    chainfiles = [None] * n_chains
    chainfile_root = '/Users/keir/Documents/emulator_data/chains_final'
    if (parameters == 'all') or (parameters == 'logma'):
        chainfiles[
            0] = 'chain_ns0.964As1.83e-09heat_slope0heat_amp1omega_m0.321alpha0beta1gamma-1z_rei8T_rei2e+04_1_emu50_512_data_ULA_diag_emu_TDR_u0_15000.txt'
        chainfiles[
            1] = 'chain_ns0.964As1.83e-09heat_slope0heat_amp1omega_m0.321alpha0beta1gamma-1z_rei8T_rei2e+04_1_batch6_data_diag_emu_TDR_u0_15000_ULA_fit_convex_hull_omega_m_fixed_tau_Planck_T0_tighter_prior_no_jump_Tu0_Tu0CH_0_T012_g08_u012_18.txt'
        chainfiles[
            2] = 'chain_ns0.964As1.83e-09heat_slope0heat_amp1omega_m0.321alpha0beta1gamma-1z_rei8T_rei2e+04_1_batch14_data_diag_emu_TDR_u0_15000_ULA_fit_convex_hull_omega_m_fixed_tau_Planck_T0_tighter_prior_no_jump_Tu0_Tu0CH_0_T012_g08_u012_18.txt'
        chainfiles[
            3] = 'chain_ns0.964As1.83e-09heat_slope0heat_amp1omega_m0.321alpha0beta1gamma-1z_rei8T_rei2e+04_1_batch18_2_data_diag_emu_TDR_u0_30000_ULA_fit_convex_hull_omega_m_fixed_tau_Planck_T0_tighter_prior_no_jump_Tu0_Tu0CH_0_T012_g08_u012_18.txt'
    elif parameters == 'PRL':
        chainfiles[
            0] = 'chain_ns0.964As1.83e-09heat_slope0heat_amp1omega_m0.321alpha0beta1gamma-1z_rei8T_rei2e+04_1_batch18_2_data_diag_emu_TDR_u0_30000_ULA_fit_convex_hull_omega_m_fixed_tau_Planck_T0_tighter_prior_no_jump_Tu0_Tu0CH_0_T012_g08_u012_18.txt'
    if parameters == 'logma':
        chainfiles[4] = 'chain_ns0.964As1.83e-09heat_slope0heat_amp1omega_m0.321alpha0beta1gamma-1z_rei8T_rei2e+04_1_batch9_data_diag_emu_TDR_u0_15000_ULA_fit_convex_hull_omega_m_fixed_tau_Planck_T0_tighter_prior_no_jump_Tu0_Tu0CH_0_T012_g08_u012_18.txt'
        chainfiles[5] = 'chain_ns0.964As1.83e-09heat_slope0heat_amp1omega_m0.321alpha0beta1gamma-1z_rei8T_rei2e+04_1_batch17_1_data_diag_emu_TDR_u0_15000_ULA_fit_convex_hull_omega_m_fixed_tau_Planck_T0_tighter_prior_no_jump_Tu0_Tu0CH_0_T012_g08_u012_18.txt'
    if parameters == 'mock':
        chainfiles[0] = 'chain_ns0.969As1.8e-09heat_slope0.316heat_amp1.02omega_m0.321alpha0.00701beta5.81gamma-1.36z_rei7.25T_rei3.09e+04_87_referee_test_mock87_30000_86.txt'
        chainfiles[1] = 'chain_ns0.969As1.8e-09heat_slope0.316heat_amp1.02omega_m0.321alpha0.00701beta5.81gamma-1.36z_rei7.25T_rei3.09e+04_87_referee_test_mock87_T0p_30000_86.txt'

    for i, chainfile in enumerate(chainfiles):
        chainfiles[i] = os.path.join(chainfile_root, chainfile)

    redshifts = [4.95, 4.58, 4.24]
    parameter_names = ['t5', 't46', 't42', 'ns', 'As', 'T5', 'T46', 'T42', 'g5', 'g46', 'g42', 'u5', 'u46', 'u42',
                       'logma']
    parameter_labels = [r'\tau_0^{5.0}', r'\tau_0^{4.6}',
                        r'\tau_0 (z = 4.2)', r'n_\mathrm{s}', r'A_\mathrm{s}',
                        r'T_0^{5.0}', r'T_0^{4.6}', r'T_0 (z = 4.2)',
                        r'\widetilde{\gamma}^{5.0}', r'\widetilde{\gamma}^{4.6}',
                        r'\widetilde{\gamma} (z = 4.2)', r'u_0^{5.0}',
                        r'u_0^{4.6}', r'u_0 (z = 4.2)',
                        r'\log\,m_\mathrm{a}'] #^\mathrm{eV}
    if parameters == 'PRL':
        parameter_labels[-1] = r'\log(m_\mathrm{a} [\mathrm{eV}])'

    legend_labels = [r'Initial emulator', r'After 19 optimization simulations',
                     r'After 35 optimization simulations',
                     r'After 43 optimization simulations']
    if parameters == 'PRL':
        colours = [lyc.get_distinct(6)[-1],]
    else:
        colours = lyc.get_distinct(len(chainfiles))
    line_widths = [2.5,] * len(chainfiles)
    if parameters == 'logma':
        legend_labels = legend_labels[:2] + [r'After 25 optimization simulations',] + [legend_labels[2],] +\
                        [r'After 40 optimization simulations',] + [legend_labels[3],]
    if parameters == 'mock':
        legend_labels = [r'Mock data', r'Fiducial prior']

    samples = [None] * len(chainfiles)
    for i in range(len(samples)):
        if i < 3:
            samples[i] = np.loadtxt(chainfiles[i]) #, max_rows=450)
        else:
            samples[i] = np.loadtxt(chainfiles[i]) #, max_rows=4500)
        samples[i][:, 4] *= 1.e+9
        if (parameters == 'all') or (parameters == 'logma') or (parameters == 'mock'):
            samples[i][:, 5] /= 1.e+4
            samples[i][:, 6] /= 1.e+4
            samples[i][:, 7] /= 1.e+4
            samples[i] = samples[i][:, np.array([2, 3, 4, 7, 10, 13, 14])]
        if parameters == 'logma':
            samples[i] = samples[i][:, -1].reshape(-1, 1)
    if (parameters == 'all') or (parameters == 'mock'):
        width_inch = 6.4*2.5
        legend_loc = 'upper right'
        tick_label_size = 16.
        parameter_names = parameter_names[2:5] + [parameter_names[7],] + [parameter_names[10],] + parameter_names[13:]
        parameter_labels = parameter_labels[2:5] + [parameter_labels[7],] + [parameter_labels[10],]\
                           + parameter_labels[13:]
    elif parameters == 'logma':
        width_inch = 6.4
        legend_loc = [0.07, 0.2]
        tick_label_size = 14.
        samples = list(np.array(samples)[np.array([0, 1, 4, 2, 5, 3])])
        parameter_names = [parameter_names[-1],]
        parameter_labels = [r'\log(m_\mathrm{a} [\mathrm{eV}])',]
    elif parameters == 'PRL':
        width_inch = 6.4
        tick_label_size = 14.

    posterior_MCsamples = [None] * len(samples)
    for i, samples_single in enumerate(samples):
        posterior_MCsamples[i] = gd.MCSamples(samples=samples_single, names=parameter_names, labels=parameter_labels,
                                              ranges={'logma': [-22., -19.]})

    subplot_instance = gdp.getSubplotPlotter(width_inch=width_inch, rc_sizes=True, scaling=False)
    if parameters == 'logma':
        subplot_instance.settings.legend_fontsize = 12.
        subplot_instance.settings.figure_legend_frame = False
    elif (parameters == 'all') or (parameters == 'mock'):
        subplot_instance.settings.legend_fontsize = 22.
        subplot_instance.settings.figure_legend_frame = False
    if (parameters == 'all') or (parameters == 'logma') or (parameters == 'mock'):
        subplot_instance.triangle_plot(posterior_MCsamples, filled=True, contour_colors=colours, contour_lws=line_widths,
                                       legend_labels=legend_labels, legend_loc=legend_loc)
    elif parameters == 'PRL':
        #fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6.4, 6.4 * 2.5))
        subplot_instance = gdp.getSubplotPlotter(width_inch=6.4*2.5/3., rc_sizes=True, scaling=False)

        subplot_instance.plot_1d(posterior_MCsamples[0], 'logma', colors=colours, lws=[5.,], lims=[-20.5, -19.])
        ax = subplot_instance.subplots[0, 0]
        ax.set_ylabel(r'1D PDF', rotation='horizontal')  # parameter_labels[7 + (i * 3)]
        ax.yaxis.set_label_coords(0.22, 0.81)
        ax.xaxis.set_ticks([-20.5, -19.])
        ax.xaxis.label.set_size(34.)
        ax.yaxis.label.set_size(34.)
        ax.xaxis.set_tick_params(labelsize=30.)
        ax.yaxis.set_tick_params(labelsize=30.)
        subplot_instance.fig.subplots_adjust(hspace=0., wspace=0., bottom=0., top=1., left=0., right=1.)
        plt.savefig('/Users/keir/Documents/emulator_paper_axions/posterior_PRL_1D.pdf')

        subplot_instance = gdp.getSubplotPlotter(width_inch=6.4 * 2.5, rc_sizes=True, scaling=False)
        subplot_instance.plots_2d(param1='logma', params2=['T42', 'g42', 'u42'], roots=[posterior_MCsamples[0],],
                                  filled=True, colors=colours, lws=[5.,], nx=3)
    #subplot_instance.add_legend(legend_labels, frameon=False)
    #subplot_instance.fig.legend(frameon=False)
    subplot_instance.fig.subplots_adjust(hspace=0., wspace=0.)
    #subplot_instance.finish_plot(legend_frame=False)

    with open('/Users/keir/Documents/emulator_data/chains_final/emulator_params_batch18_2_TDR_u0.json',
              'r') as json_file:
        json_dict = json.load(json_file)
    emulator_samples = np.concatenate((np.array(json_dict['sample_params'])[:, :2],
                                       np.array(json_dict['measured_sample_params'])[:, np.array([2, 5, 8])]), axis=1)
    log_mass = np.zeros((emulator_samples.shape[0], 1))
    for i in range(50, emulator_samples.shape[0]):
        log_mass[i, 0] = ultra_light_axion_numerical_model_inverse(np.array(json_dict['sample_params'])[i, 5:8])
    emulator_samples = np.concatenate((emulator_samples, log_mass), axis=1)
    emulator_samples[:, 1] *= 1.e+9
    if (parameters == 'all') or (parameters == 'logma') or (parameters == 'mock'):
        emulator_samples[:, 2] /= 1.e+4
        #emulator_samples[:, 3] /= 1.e+4
        #emulator_samples[:, 4] /= 1.e+4

    if parameters == 'PRL':
        parameter_labels = [r'$T_0^{z = 4.2} [\mathrm{K}]$', r'$\widetilde{\gamma}^{z = 4.2}$',
                            r'$u_0^{z = 4.2} [\frac{\mathrm{eV}}{m_\mathrm{p}}]$']
        for i in range(3):
            ax = subplot_instance.subplots[0, i]
            ax.set_ylabel(parameter_labels[i], rotation='horizontal') #parameter_labels[7 + (i * 3)]
            if i == 1:
                ax.yaxis.set_label_coords(0.16, 0.81)
            else:
                ax.yaxis.set_label_coords(0.24, 0.81)
            ax.set_xlim([-20.5, -19.])
            ax.xaxis.set_ticks([-20.5, -19.])
            ax.xaxis.label.set_size(34.)
            ax.yaxis.label.set_size(34.)
            ax.xaxis.set_tick_params(labelsize=30.)
            ax.yaxis.set_tick_params(labelsize=30.)
        subplot_instance.fig.subplots_adjust(hspace=0., wspace=0., bottom=0., top=1., left=0., right=1.)
    else:
        for p in range(samples[0].shape[1]):
            for q in range(p + 1):
                ax = subplot_instance.subplots[p, q]
                if parameters == 'logma':
                    ax.xaxis.label.set_size(18.)
                    ax.yaxis.label.set_size(18.)
                elif (parameters == 'all') or (parameters == 'mock'):
                    ax.xaxis.label.set_size(22.)
                    ax.yaxis.label.set_size(22.)
                ax.xaxis.set_tick_params(labelsize=tick_label_size)
                ax.yaxis.set_tick_params(labelsize=tick_label_size)

                if (q < p) and (q > 0):
                    #if (p in np.array([3, 4, 14])) or (q in np.array([3, 4, 14])) or (
                    #        (p in np.arange(8, 11)) and (q == (p - 3))) or (
                    #        (p in np.arange(11, 14)) and ((q == (p - 3)) or (q == (p - 6)))):
                    msize = 200
                    if parameters != 'mock':
                        ax.scatter(emulator_samples[:50, q-1], emulator_samples[:50, p-1], color=colours[0], marker='+', s=msize)
                        ax.scatter(emulator_samples[50:69, q-1], emulator_samples[50:69, p-1], color=colours[1], marker='+', s=msize)
                        ax.scatter(emulator_samples[69:85, q-1], emulator_samples[69:85, p-1], color=colours[2], marker='+', s=msize)
                        ax.scatter(emulator_samples[85:, q-1], emulator_samples[85:, p-1], color=colours[3], marker='+', s=msize)
                    else:
                        ax.axvline(x=emulator_samples[86, q-1], color='black', ls='--', lw=line_widths[0])
                        ax.axhline(y=emulator_samples[86, p - 1], color='black', ls='--', lw=line_widths[0])
            if parameters == 'mock':
                ax = subplot_instance.subplots[p, p]
                if p == 0:
                    truth = 1.
                else:
                    truth = emulator_samples[86, p-1]
                ax.axvline(x=truth, color='black', ls='--', lw=line_widths[0])

                ax = subplot_instance.subplots[p, 0]
                ax.axvline(x=1., color='black', ls='--', lw=line_widths[0])
                ax.axhline(y=emulator_samples[86, p - 1], color='black', ls='--', lw=line_widths[0])

    #plt.legend(fontsize=18., frameon=False)
    plt.savefig('/Users/keir/Documents/emulator_paper_axions/' + save_file)

def plot_emulator():
    """Make a plot of the emulator training points."""
    redshifts = [4.95, 4.58, 4.24]
    plot_labels = np.array([r'$n_\mathrm{s}$', r'$A_\mathrm{s}$',
                   r'$T_0^{z = %.1f} [10^4\,\mathrm{K}]$'%redshifts[0],
                   r'$T_0^{z = %.1f} [10^4\,\mathrm{K}]$'%redshifts[1],
                   r'$T_0^{z = %.1f} [10^4\,\mathrm{K}]$'%redshifts[2],
                   r'$\widetilde{\gamma}^{z = %.1f}$'%redshifts[0], r'$\widetilde{\gamma}^{z = %.1f}$'%redshifts[1],
                   r'$\widetilde{\gamma}^{z = %.1f}$'%redshifts[2],
                   r'$u_0^{z = %.1f} \left[\frac{\mathrm{eV}}{m_\mathrm{p}}\right]$'%redshifts[0],
                   r'$u_0^{z = %.1f} \left[\frac{\mathrm{eV}}{m_\mathrm{p}}\right]$'%redshifts[1],
                   r'$u_0^{z = %.1f} \left[\frac{\mathrm{eV}}{m_\mathrm{p}}\right]$'%redshifts[2],
                   r'$\alpha$', r'$\beta$', r'$\gamma$']) #r'$\log(m_\mathrm{a} [\mathrm{eV}])$'])
    plot_labels = plot_labels[np.array([0, 1, 11, 12, 13, 2, 5, 8, 3, 6, 9, 4, 7, 10])]
    plot_labels_x = np.concatenate((plot_labels[:5], np.array([r'$T_0^{z = z_i} [10^4\,\mathrm{K}]$',
                                                               r'$\widetilde{\gamma}^{z = z_i}$'])))

    with open('/Users/keir/Documents/emulator_data/chains_final/emulator_params_batch18_2_TDR_u0.json',
              'r') as json_file:
        json_dict = json.load(json_file)
    emulator_samples = np.concatenate((np.array(json_dict['sample_params'])[:, :2],
                                       np.array(json_dict['sample_params'])[:, 5:8],
                                       np.array(json_dict['measured_sample_params'])), axis=1)
    #log_mass = np.zeros((emulator_samples.shape[0], 1))
    #for i in range(50, emulator_samples.shape[0]):
    #    log_mass[i, 0] = ultra_light_axion_numerical_model_inverse(np.array(json_dict['sample_params'])[i, 5:8])
    #emulator_samples = np.concatenate((emulator_samples[:, :2], log_mass, emulator_samples[:, 2:]), axis=1)
    emulator_samples = emulator_samples[:, np.array([0, 1, 2, 3, 4, 3+2, 6+2, 9+2, 4+2, 7+2, 10+2, 5+2, 8+2, 11+2])]
    #np.concatenate((emulator_samples, log_mass), axis=1)
    emulator_samples[:, 1] *= 1.e+9
    emulator_samples[:, 3+2] /= 1.e+4
    emulator_samples[:, 6+2] /= 1.e+4
    emulator_samples[:, 9+2] /= 1.e+4

    fig, axes = plt.subplots(nrows=11+2, ncols=5+2, figsize=(6.4*2.5*(5.73/11.)*(7./5.), 6.4*2.5)) #5.
    alpha_min = 0.25
    alpha_k = lambda k: (k * (1. - alpha_min) / (emulator_samples.shape[0] - 50)) + alpha_min
    for a in range(axes.shape[0]): #11
        i = a + 1
        for j in range(axes.shape[1]): #6
            if a == (axes.shape[0] - 1):
                axes[a, j].set_xlabel(plot_labels_x[j])
            else:
                axes[a, j].set_xticklabels([])
            if j == 0:
                axes[a, j].set_ylabel(plot_labels[i])
            else:
                axes[a, j].set_yticklabels([])

            if j >= i:
                fig.delaxes(axes[a, j])
                continue
            elif (i > 5+2) and (j >= (i - 3)):
                fig.delaxes(axes[a, j])
                continue
            elif (i > 8+2) and (j >= (i - 6)):
                fig.delaxes(axes[a, j])
                continue

            if j < 3+2:
                x_idx = j
            elif (j > 2+2) and (i > 8+2):
                x_idx = j + 6
            elif (j > 2+2) and (i > 5+2):
                x_idx = j + 3
            elif (j > 2+2) and (i > 2+2):
                x_idx = j

            axes[a, j].set_xlim([np.min(emulator_samples[:, x_idx]), np.max(emulator_samples[:, x_idx])])
            axes[a, j].set_ylim([np.min(emulator_samples[:, i]), np.max(emulator_samples[:, i])])
            if i in np.array([4, 7, 10])+2:
                axes[a, j].set_ylim([0.8, np.max(emulator_samples[:, i])])
            elif i == 5+2:
                axes[a, j].set_ylim([np.min(emulator_samples[:, i]), 18.])
            elif i in np.array([8, 11])+2:
                axes[a, j].set_ylim([np.min(emulator_samples[:, i]), 18.])
            #elif i == 3:
            #    axes[a, j].set_ylim([np.min(emulator_samples[:, i]), 12000.])
            if j == 3+2:
                axes[a, j].set_xlim([np.min(emulator_samples[:, np.arange(j, j+7, 3)]),
                                     np.max(emulator_samples[:, np.arange(j, j+7, 3)])])
            elif j == 4+2:
                axes[a, j].set_xlim([0.8, np.max(emulator_samples[:, np.arange(j, j+7, 3)])])
            #if (i == 2) or (j == 2):
            #    if j == 2:
            #        axes[a, j].set_xlim([np.min(emulator_samples[50:, x_idx]), np.max(emulator_samples[50:, x_idx])])
            #    elif i == 2:
            #        axes[a, j].set_ylim([np.min(emulator_samples[50:, i]), np.max(emulator_samples[50:, i])])
            #    pass
            #else:
            axes[a, j].scatter(emulator_samples[:50, x_idx], emulator_samples[:50, i], color='black', marker='+',
                                   alpha=alpha_min)
            for k in range(50, emulator_samples.shape[0]):
                axes[a, j].scatter(emulator_samples[k, x_idx], emulator_samples[k, i], color='black', marker='+',
                                   alpha=alpha_k(k-49))
            axes[a, j].set_aspect(np.diff(axes[a, j].get_xlim()) / np.diff(axes[a, j].get_ylim()))

    axes[10, 0].set_xticks([0.92, 0.98])
    #axes[10, 2].set_xticks([-21., -20.])
    fig.subplots_adjust(top=0.99, bottom=0.05, right=0.95, hspace=0., wspace=0.)
    #plt.legend(fontsize=18., frameon=False)
    plt.savefig('/Users/keir/Documents/emulator_paper_axions/emulator_full2.pdf')

    return emulator_samples

def plot_data():
    """Make a plot of the data flux power spectra."""
    redshifts = [4.95, 4.58, 4.24]
    n_z = len(redshifts)
    n_k = 16
    theory = np.load('/Users/keir/Software/lya_emulator/plots/max_post.npy') #3 x 3 x 16
    theory2 = np.load('/Users/keir/Software/lya_emulator/plots/max_post_axion_21_3.npy')
    data = np.zeros((n_z, n_k, 4))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 6.4))
    plot_labels = [r'$z = %.1f$'%redshifts[0], r'$z = %.1f$'%redshifts[1], r'$z = %.1f$'%redshifts[2]]
    colours = lyc.get_distinct(n_z) #['red', 'orange', 'yellow'] #['#FF00FF', '#A800A8', '#540054']
    colours[-1] = 'orange'
    for i, z in enumerate(redshifts):
        data[i] = np.genfromtxt('/Users/keir/Software/lya_emulator/lyaemu/data/Boera_HIRES_UVES_flux_power/flux_power_z_%.1f.dat'%z,
                                skip_header=5, skip_footer=1)
        ax.errorbar(data[i, :, 0], theory[0, i] * theory[1, i] / np.pi, yerr=theory[0, i] * theory[2, i] / np.pi,
                    elinewidth=2.5, capsize=4., capthick=4., color=colours[i], lw=2.5, label=plot_labels[i])
        #ax.errorbar(data[i, :, 0], theory[0, i] * theory2[1, i] / np.pi, yerr=theory[0, i] * theory2[2, i] / np.pi,
        #            elinewidth=2.5, capsize=4., capthick=4., color=colours[i], lw=2.5, ls='--')

        ax.errorbar(data[i, :, 0], theory[0, i] * data[i, :, 2] / np.pi, yerr=theory[0, i] * data[i, :, 3] / np.pi,
                    elinewidth=2.5, capsize=4., capthick=4., color=colours[i], lw=2.5, ls='', marker='o', markersize=4.)

    ax.errorbar([], [], yerr=[], color='gray', lw=2.5, label=r'Max posterior', elinewidth=2.5, capsize=4., capthick=4.)
    ax.errorbar([], [], yerr=[], color='gray', lw=2.5, label=r'Data', ls='', elinewidth=2.5, capsize=4., capthick=4.,
                marker='o', markersize=4.)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\mathrm{log} (k_\mathrm{f} [\mathrm{s}\,\mathrm{km}^{-1}])$')
    ax.set_ylabel(r'$k_\mathrm{f} P_\mathrm{f}(k_\mathrm{f})/\pi$')
    #ax.set_xlim([-1.2, 1.6])
    #ax.set_ylim([-0.1, 1.05])
    ax.legend(frameon=False) #fontsize=16.)

    fig.subplots_adjust(top=0.95, bottom=0.15, right=0.96, left=0.14)
    plt.savefig('/Users/keir/Documents/emulator_paper_axions/data_PRL_post2.pdf')

def plot_transfer_ULA():
    """Make a plot of the ULA transfer function and the nCDM fit."""
    fname = '/Users/keir/Software/axionCAMB/axion_%s_matterpower_z_99.dat'
    mass_22 = [0.5, 1., 4., 10., 40., 100., 400., 1000., 2000.]

    linear_power = [None] * (len(mass_22) + 1)
    linear_power[0] = np.loadtxt(fname%'CDM')
    nCDM_parameters = np.zeros((len(mass_22), 3))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 6.4))
    plot_label = r'$%.1f$'
    colours = lyc.get_distinct(len(mass_22))
    for i, mass in enumerate(mass_22):
        linear_power[i+1] = np.loadtxt(fname%str(int(mass)))
        transfer_function_ULA = np.sqrt(linear_power[i+1][:, 1] / linear_power[0][:, 1])
        nCDM_parameters[i], nCDM_covariance = spo.curve_fit(transfer_function_nCDM, linear_power[i+1][:, 0],
                                                            transfer_function_ULA, p0=np.array([0.05, 5., -5.]))
        print(nCDM_parameters[i], nCDM_covariance)

        ax.plot(np.log10(linear_power[i+1][:, 0]), transfer_function_nCDM(linear_power[i+1][:, 0], *nCDM_parameters[i]),
                label=plot_label%np.log10(mass * 1.e-22), color=colours[i], lw=2.5)
        ax.plot(np.log10(linear_power[i+1][:, 0]), transfer_function_ULA, color=colours[i], ls='--', lw=2.5)

    ax.plot([], [], label=r'$[\alpha, \beta, \gamma]$ - fit', color='gray', lw=2.5)
    ax.plot([], [], label=r'axionCAMB', color='gray', ls='--', lw=2.5)
    ax.set_xlabel(r'$\mathrm{log} (k [h\,\mathrm{Mpc}^{-1}])$')
    ax.set_ylabel(r'$T(k)$')
    ax.set_xlim([-0.3, 2.8])
    ax.set_ylim([-0.1, 1.05])
    ax.legend(frameon=False, title=r'$\log(m_\mathrm{a} [\mathrm{eV}])$', fontsize=15., title_fontsize=15.)

    fig.subplots_adjust(top=0.95, bottom=0.15, right=0.95)
    #plt.savefig('/Users/keir/Documents/emulator_paper_axions/transfer_ULA.pdf')

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6.4, 6.4*1.5))
    colours = lyc.get_distinct(2)
    mass_plot = np.linspace(-22.5, -18.5, num=1000)
    for i, axis in enumerate(ax):
        if i == 0:
            y_scatter = np.log10(nCDM_parameters[:, 0])
            y_plot = np.log10(ultra_light_axion_alpha_model(mass_plot, *alpha_model_parameters))
            scatter_label = r'$[\alpha, \beta, \gamma]$'
            plot_label = r'Polynomial model fit'
        elif i == 1:
            y_scatter = cp.deepcopy(nCDM_parameters[:, 1])
            y_plot = ultra_light_axion_beta_model(mass_plot, *beta_model_parameters)
            scatter_label = None
            plot_label = None
        elif i == 2:
            y_scatter = np.log10(-1. * nCDM_parameters[:, 2])
            y_plot = np.log10(-1. * ultra_light_axion_gamma_model(mass_plot, *gamma_model_parameters))
            scatter_label = None
            plot_label = None
        axis.scatter(np.log10(np.array(mass_22) * 1.e-22), y_scatter, label=scatter_label, color=colours[1], marker='+',
                     s=200., lw=2.5)
        axis.plot(mass_plot, y_plot, label=plot_label, color=colours[0], lw=2.5)

    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[2].set_xlabel(r'$\log(m_\mathrm{a} [\mathrm{eV}])$')
    ax[0].set_ylabel(r'$\log(\alpha [h^{-1}\,\mathrm{Mpc}])$')
    ax[1].set_ylabel(r'$\beta$')
    ax[2].set_ylabel(r'$\log(\mbox{-} \gamma)$')
    ax[0].legend(frameon=False)

    fig.subplots_adjust(top=0.96, bottom=0.1, right=0.95, hspace=0.05, left=0.15)
    plt.savefig('/Users/keir/Documents/emulator_paper_axions/transfer_ULA_polynomial.pdf')

def plot_comparison():
    """Make a plot comparing ULA DM mass bounds."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 6.4*0.7)) #0.75
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.xaxis.tick_top()
    ax.xaxis.set_tick_params(width=5.)
    ax.axvline(x=-22., color='black', lw=5., ymin=0.96)
    ax.axvline(x=-21., color='black', lw=5., ymin=0.96)
    ax.axvline(x=-20., color='black', lw=5., ymin=0.96)
    ax.axvline(x=-19., color='black', lw=5., ymin=0.96)
    ax.axvline(x=-18., color='black', lw=5., ymin=0.96)
    #plt.tick_params(axis='x', direction='in')
    ax.xaxis.set_label_position('top')
    ax.axvline()
    ax.arrow(-21., 0.96, 4.98-1.5, 0., color='black', width=0.02, length_includes_head=True, head_length=0.12) #, head_width=15.)
    ax.arrow(-19., 0.96, -1.*(4.98-1.5), 0., color='black', width=0.02, length_includes_head=True, head_length=0.12) #, head_width=15.)
    ax.set_xlim([-22.5, -17.5])
    ax.set_ylim([0., 1.])
    ax.set_title(r'Axion dark matter mass [$\log(\mathrm{eV})$]', pad=20)

    colours = lyc.get_distinct(5)
    a = 0.75

    #ax.axvspan(ymin=0., ymax=0.96, xmin=-22., xmax=-21., facecolor='black', edgecolor=None, fill=True, alpha=0.1)
    #ax.text(-21.5, 0.64, r'\begin{center}\textbf{Canonical}\end{center}', horizontalalignment='center',
    #        verticalalignment='center', color='gray', size=14.)

    #ax.axvspan(ymin=0.66, ymax=0.90, xmin=-22.5, xmax=-22, facecolor=colours[0], edgecolor=None, fill=True, alpha=a)
    ax.arrow(-22.5, 0.79, 0.5, 0., color=colours[0], width=0.015, length_includes_head=True, head_length=0.09)
    #ax.text(-22.25, 0.78, r'\begin{center}\textbf{CMB/rei}\end{center}', horizontalalignment='center',
    #        verticalalignment='center', rotation=90.)
    ax.text(-22., 0.74, r'\begin{center}\textbf{CMB/\\reionization}\end{center}', horizontalalignment='center',
            verticalalignment='top', color=colours[0], size=14.)
    #ax.text(0.5 * (-22.5 - 19.64), 0.19, r'\begin{center}$\mathbf{m_\mathrm{\textbf{a}} > 2 \times 10^{-20}\,\mathrm{\textbf{eV}}}$\end{center}',
    #        horizontalalignment='center', verticalalignment='center', color=colours[4], size=16.)
    ax.text(-22., 0.85, r'\begin{center}\textbf{-22}\end{center}', horizontalalignment='center',
            verticalalignment='center', color=colours[0], size=17.)

    #ax.axvspan(ymin=0.66, ymax=0.90, xmin=-18., xmax=-17.5, facecolor=colours[4], edgecolor=None, fill=True, alpha=a)
    ax.arrow(-17.5, 0.79, -0.5, 0., color=colours[4], width=0.015, length_includes_head=True, head_length=0.09)
    #ax.text(-17.75, 0.78, r'\begin{center}\textbf{BHSR}\end{center}', horizontalalignment='center',
    #        verticalalignment='center', rotation=270.)
    ax.text(-18., 0.73, r'\begin{center}\textbf{BHSR}\end{center}', horizontalalignment='center',
            verticalalignment='center', color=colours[4], size=14.)
    #ax.text(0.5 * (-22.5 - 19.64), 0.19, r'\begin{center}$\mathbf{m_\mathrm{\textbf{a}} > 2 \times 10^{-20}\,\mathrm{\textbf{eV}}}$\end{center}',
    #        horizontalalignment='center', verticalalignment='center', color=colours[4], size=16.)
    ax.text(-18., 0.85, r'\begin{center}\textbf{-18}\end{center}', horizontalalignment='center',
            verticalalignment='center', color=colours[4], size=17.)

    #ax.axvspan(ymin=0.66, ymax=0.90, xmin=-22., xmax=np.log10(2.1e-21), facecolor=colours[1], edgecolor=None, fill=True,
    #           alpha=a)
    ax.arrow(-22.5, 0.55, np.log10(2.1e-21) + 22.5, 0., color=colours[1], width=0.015, length_includes_head=True, head_length=0.09)
    #ax.text(0.5*(-22.+np.log10(2.1e-21)), 0.78, r'\begin{center}\textbf{Sub-halo\\mass\\function}\end{center}', horizontalalignment='center',
    #        verticalalignment='center')
    ax.text(np.log10(2.1e-21), 0.49, r'\begin{center}\textbf{Sub-halos}\end{center}', horizontalalignment='center',
            verticalalignment='center', color=colours[1], size=14.)
    #ax.text(0.5 * (-22.5 - 19.64), 0.19, r'\begin{center}$\mathbf{m_\mathrm{\textbf{a}} > 2 \times 10^{-20}\,\mathrm{\textbf{eV}}}$\end{center}',
    #        horizontalalignment='center', verticalalignment='center', color=colours[4], size=16.)
    ax.text(np.log10(2.1e-21), 0.61, r'\begin{center}\textbf{-20.7}\end{center}', horizontalalignment='center',
            verticalalignment='center', color=colours[1], size=17.)

    #ax.axvspan(ymin=0.32, ymax=0.57, xmin=-22., xmax=np.log10(4.e-21), facecolor=colours[2], edgecolor=None, fill=True,
    #           alpha=a)
    ax.arrow(-22.5, 0.31, np.log10(2.e-21) + 22.5, 0., color='orange', width=0.015, length_includes_head=True,
             head_length=0.09)
    #ax.text(0.5*(-22.+np.log10(4.e-21)), 0.44, r'\begin{center}\textbf{Ly-}$\mathbf{\alpha}$\textbf{ forest\\(previous\\work)}\end{center}',
    #        horizontalalignment='center', verticalalignment='center')
    ax.text(np.log10(2.e-21), 0.25, r'\begin{center}\textbf{Ly-}$\mathbf{\alpha}$\textbf{f (previous work)}\end{center}', horizontalalignment='center',
            verticalalignment='center', color='orange', size=14.)
    #ax.text(0.5 * (-22.5 - 19.64), 0.19, r'\begin{center}$\mathbf{m_\mathrm{\textbf{a}} > 2 \times 10^{-20}\,\mathrm{\textbf{eV}}}$\end{center}',
    #        horizontalalignment='center', verticalalignment='center', color=colours[4], size=16.)
    ax.text(np.log10(2.e-21), 0.37, r'\begin{center}\textbf{-20.7}\end{center}', horizontalalignment='center',
            verticalalignment='center', color='orange', size=17.)

    #ax.axvspan(ymin=0., ymax=0.235, xmin=-22., xmax=-19.64, facecolor=colours[3], edgecolor=None, fill=True, alpha=a)
    ax.arrow(-22.5, 0.07, -19.64 + 22.5, 0., color=colours[2], width=0.015, length_includes_head=True,
             head_length=0.09)
    ax.text(-19.64, 0.01, r'\begin{center}\textbf{Ly-}$\mathbf{\alpha}$\textbf{f (this work)}\end{center}', horizontalalignment='center',
            verticalalignment='center', color=colours[2], size=14.)
    #ax.text(0.5 * (-22.5 - 19.64), 0.19, r'\begin{center}$\mathbf{m_\mathrm{\textbf{a}} > 2 \times 10^{-20}\,\mathrm{\textbf{eV}}}$\end{center}',
    #        horizontalalignment='center', verticalalignment='center', color=colours[4], size=16.)
    ax.text(-19.64, 0.12, r'\begin{center}\textbf{-19.6}\end{center}', horizontalalignment='center',
            verticalalignment='center', color=colours[2], size=17.)

    fig.subplots_adjust(top=0.8, bottom=0.03, right=0.95, left=0.05)
    plt.savefig('/Users/keir/Documents/emulator_paper_axions/comparisonUS.pdf')

if __name__ == "__main__":
    #plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=18.) #18 normally - 16 for posteriors - 17 for comparison

    plt.rc('axes', linewidth=1.5)
    plt.rc('xtick.major', width=1.5)
    plt.rc('xtick.minor', width=1.5)
    plt.rc('ytick.major', width=1.5)
    plt.rc('ytick.minor', width=1.5)

    #plot_transfer_function(y='flux_power')
    #k, z, p, f, m, s, k_max = make_error_distribution()
    #k, z, p, f, m, s, k_data, s_data, k_max, data_frames = violinplot_error_distribution(distribution='data')
    #plot_exploration()
    #posterior_means, posterior_limits = plot_convergence()
    #plot_posterior(parameters='mock')
    #plot_data()
    #plot_transfer_ULA()
    #emulator_samples = plot_emulator()
    #plot_comparison()
    #power_arrays, k_log = plot_numerical_convergence()

    k, z, p, f, m, s, k_max = make_error_distribution()
    np.savez('/home/keir/Software/lya_emulator/plots/cross_validation_bDM2.npz', k=k, z=z, p=p, f=f, m=m, s=s,
             k_max=k_max)
    violinplot_error_distribution(distribution='validation')
    violinplot_error_distribution(distribution='data')
