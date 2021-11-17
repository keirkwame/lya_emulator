import os
import json
import h5py

import copy as cp
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import scipy.interpolate as spi
import scipy.optimize as spo
import getdist as gd
import getdist.plots as gdp
import multiprocessing as mg
import pandas as pd
import seaborn as sb

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

#import lyaemu.distinct_colours_py3 as lyc
import lyaemu.coarse_grid as lyc
import lyaemu.mean_flux as lym
#from lyaemu.likelihood import transfer_function_nCDM


#Define global variables
alpha_model_parameters = np.array([5.54530089e-03, 3.31718138e-01, 6.16422310e+00, 3.31219369e+01])
beta_model_parameters = np.array([-0.02576259, -0.82153762, -0.45096863])
gamma_model_parameters = np.array([-1.29071567e-02, -7.52873377e-01, -1.47076333e+01, -9.60752318e+01])

alpha_model_parameters_bDM = np.array([-7.51459969e-03, -2.30695833e-03, 1.28202462e-01, -1.84144080e-01,
                                       -8.96963581e-01, -4.22663264e+00, -2.41323815e+01])
beta_model_parameters_bDM = np.array([-2.32142859e-04, -8.21964286e-02, 2.42587500e+00])
gamma_model_parameters_bDM = np.array([-4.46])

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

def bDM_alpha_model(log_mass, log_sigma, bmass, bsig, amass, asig, mmass, msig, c):
    """Model for alpha as a function of log DM mass & log sigma"""
    return 10. ** ((bmass * (log_mass ** 3)) + (amass * (log_mass ** 2)) + (mmass * log_mass) +
                   (bsig * (log_sigma ** 3)) + (asig * (log_sigma ** 2)) + (msig * log_sigma) + c)

def bDM_beta_model(log_mass, a, m, c):
    """Model for beta as a function of log DM mass"""
    return (a * (log_mass ** 2)) + (m * log_mass) + c

def bDM_gamma_model(c):
    """Model for gamma in parametric bDM model"""
    return c

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

def bDM_numerical_model(bDM_parameters, nCDM_parameter_limits, h=0.6686):
    """Model to map bDM parameters to nCDM parameters using a fit to a numerical Einstein-Boltzmann solver. Valid for 4
    < log DM mass [eV] < 11; -31 < log sigma [cm^2] < -24"""
    log_mass, log_sigma = bDM_parameters
    alpha = bDM_alpha_model(log_mass, log_sigma, *alpha_model_parameters_bDM)
    beta = bDM_beta_model(log_mass, *beta_model_parameters_bDM)
    gamma = bDM_gamma_model(*gamma_model_parameters_bDM)
    nCDM_parameters = np.array([alpha * h / h_planck, beta, gamma])

    for i in range(3):
        if nCDM_parameters[i] < nCDM_parameter_limits[i, 0]:
            nCDM_parameters[i] = nCDM_parameter_limits[i, 0]
        if nCDM_parameters[i] > nCDM_parameter_limits[i, 1]:
            nCDM_parameters[i] = nCDM_parameter_limits[i, 1]
    return nCDM_parameters

def bDM_numerical_model_inverse(nCDM_parameters, h=0.6686):
    """Inverse of numerical bDM model. Valid for 4 < log DM mass [eV] < 11; -31 < log sigma [cm^2] < -24"""
    nCDM_corrected = nCDM_parameters
    nCDM_corrected[0] = np.log10(nCDM_corrected[0] * h_planck / h)
    nCDM_corrected[2] = np.log10(-1. * nCDM_corrected[2])

    #Mass from beta
    model_coefficients = cp.deepcopy(beta_model_parameters_bDM)
    model_coefficients[-1] -= nCDM_corrected[1]
    model_roots = np.roots(model_coefficients)
    print(model_roots)
    log_mass = model_roots[(model_roots <= 11.1) * (model_roots >= 3.9)][0].real
    print('log mass =', log_mass)

    #Sigma from alpha
    model_coefficients = cp.deepcopy(alpha_model_parameters_bDM[np.array([1, 3, 5, 6])])
    model_coefficients[-1] -= nCDM_corrected[0]
    alpha_model_parameters_bDM_no_const = cp.deepcopy(alpha_model_parameters_bDM[:-1])
    model_coefficients[-1] += np.log10(bDM_alpha_model(log_mass, 0., *alpha_model_parameters_bDM_no_const, 0.)
                                       * h_planck / h)
    model_roots = np.roots(model_coefficients)
    print(model_roots)
    log_sigma = model_roots[(model_roots <= -23.9) * (model_roots >= -32.1)][0].real
    print('log sigma =', log_sigma)

    return np.array([log_mass, log_sigma])


def plot_transfer_function():
    """Make a plot of transfer functions."""
    save_file = 'transfer.pdf'

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4 * 1.5, 6.4))
    k_log = np.linspace(0.2, 2.5, num=1000)
    plot_colours = lyc.get_distinct(3)
    plot_colours[2] = 'orange'

    #bDM
    params_bDM = np.array([7., -27.8])
    nCDM_bDM = bDM_numerical_model(params_bDM, nCDM_parameter_limits, h=h_planck)
    ax.plot(k_log, transfer_function_nCDM(10. ** k_log, *nCDM_bDM) ** 2, label=r'$-27.8$', color=plot_colours[0],
            lw=2.5) #[excluded]')

    params_bDM = np.array([7., -28.8])
    nCDM_bDM = bDM_numerical_model(params_bDM, nCDM_parameter_limits, h=h_planck)
    ax.plot(k_log, transfer_function_nCDM(10. ** k_log, *nCDM_bDM) ** 2, label=r'$-28.8$ [$95 \%$ c.l.]',
            color=plot_colours[1], lw=2.5)

    params_bDM = np.array([7., -29.8])
    nCDM_bDM = bDM_numerical_model(params_bDM, nCDM_parameter_limits, h=h_planck)
    ax.plot(k_log, transfer_function_nCDM(10. ** k_log, *nCDM_bDM) ** 2, label=r'$-29.8$', color=plot_colours[2],
            lw=2.5) #[allowed]')

    #ULA DM
    params_ULA = np.array([-19.7])
    nCDM_ULA = ultra_light_axion_numerical_model(params_ULA, nCDM_parameter_limits, h=h_planck)
    ax.plot(k_log, transfer_function_nCDM(10. ** k_log, *nCDM_ULA) ** 2, label=r'ULA DM [$95 \%$ c.l.]', color='black',
            lw=2.5)

    #ax.axvline(x=np.log10(40.), color='black', ls=':')
    ax.axhline(y=0.75, color='black', ls=':', lw=2.5)
    ax.set_xlabel(r'$\mathrm{log} (k [h\,\mathrm{Mpc}^{-1}])$')
    ax.set_ylabel(r'$T^2(k) = \frac{P_\mathrm{nCDM} (k)}{P_\mathrm{CDM} (k)}$')
    ax.set_xlim([0.2, 2.5])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(frameon=False, title=r'$\mathrm{log} (\sigma [\mathrm{cm}^2])$')
    fig.subplots_adjust(top=0.99, right=0.98, bottom=0.15)

    plt.savefig('/Users/keir/Documents/paper_bDM/' + save_file)

def plot_scale():
    """Make a plot of l_0.75 in mass - sigma space."""
    save_file = 'scale_new4.pdf'
    mass = np.linspace(np.log10(15. * (10. ** 3.)), 11., num=128)
    delta_mass = (mass[-1] - mass[0]) / (mass.shape[0] - 1.) / 2.
    sigma = np.linspace(-31., -24., num=128)
    delta_sigma = (sigma[-1] - sigma[0]) / (sigma.shape[0] - 1.) / 2.

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4 * 1.5, 6.4 * (1.11 + 0.1)), #(0.15 / 0.85)
                             gridspec_kw={'height_ratios': [0.1, 1.11]})

    l = np.zeros((mass.shape[0], sigma.shape[0]))
    for i, m in enumerate(mass):
        for j, s in enumerate(sigma):
            nCDM_parameter_limits_extended = cp.deepcopy(nCDM_parameter_limits)
            nCDM_parameter_limits_extended[0, 1] = 10.
            params_nCDM = bDM_numerical_model(np.array([m, s]), nCDM_parameter_limits_extended, h=h_planck)
            print(m, s, params_nCDM)
            k = (((0.75 ** (1. / (2. * params_nCDM[2]))) - 1.) ** (1. / params_nCDM[1])) / params_nCDM[0]
            l[i, j] = np.log10(2. * np.pi * 1.e+6 / k / h_planck) #log(pc)

    axp = axes[1].imshow(l.T, origin='lower', aspect='auto', #axp =
                extent=(mass[0] - delta_mass, mass[-1] + delta_mass, sigma[0] - delta_sigma, sigma[-1] + delta_sigma),
                cmap='inferno')
    axes[1].contour(l.T, levels=np.array([3., 4., 5., 6., 7., 8.]), origin='lower',
                extent=(mass[0] - delta_mass, mass[-1] + delta_mass, sigma[0] - delta_sigma, sigma[-1] + delta_sigma),
                colors='black')
    cb = plt.colorbar(axp, cax=axes[0], orientation='horizontal') #location='top') #orientation='horizontal', #ax=[ax],

    axes[0].set_xlabel(r'$\mathrm{log}\,[\mathrm{Suppression\,\,scale}\,\,\lambda_{0.75}\,(\mathrm{pc})]$', labelpad=12)
    axes[0].xaxis.tick_top()
    axes[0].xaxis.set_label_position('top')

    axes[1].set_xlim([np.log10(15. * (10. ** 3.)), 11.])
    axes[1].set_ylim([-31., -24.])
    axes[1].set_xlabel(r'$\mathrm{log}\,[\mathrm{Dark\,\,matter\,\,mass}\,\,m\,(\mathrm{eV})]$')
    axes[1].set_ylabel(r'$\mathrm{log}\,[\mathrm{Proton-DM\,\,cross\,\,section}\,\,\sigma\,(\mathrm{cm}^2)]$')

    fig.subplots_adjust(top=0.89, bottom=0.12, right=0.98, hspace=0.05)

    plt.savefig('/Users/keir/Documents/paper_bDM/' + save_file)

def plot_comparison():
    """Make a plot comparing mass - sigma bounds."""
    save_file = 'comparison7.pdf'

    chainfile = '/Users/keir/Data/emulator/bDM/chain_ns0.964As1.83e-09heat_slope0heat_amp1omega_m0.321alpha0beta1gamma-1z_rei8T_rei2e+04_1_vary_mass_40000_batch11.txt'
    param_names = ['logmass', 'logsigma']
    param_labels = [r'm', r'\sigma']
    param_ranges = {param_names[0]: [4., 11.], param_names[1]: [-31., -24.]}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4*1.5, 6.4))
    plot_colours = lyc.get_distinct(4)
    plot_colours[2] = 'orange'
    plot_colours = plot_colours[::-1]

    #Rogers21
    x = np.array([4., 5., 7., 9., 11.])
    y1 = np.array([-29.48, -29.22, -28.78, -27.91, -26.28])
    y2 = np.ones_like(y1) * -22.
    ax.fill_between(x=x, y1=y1, y2=y2, label=r'Ly-$\alpha$f (this work)', color=plot_colours[0])

    #Maamari21
    x = np.array([np.log10(15. * (10. ** 3.)), 5., 7., 9., 11.])
    y1 = np.log10([2.7e-29, 6.9e-29, 2.8e-28, 1.8e-27, 1.3e-25])
    y2 = np.ones_like(y1) * -22.
    ax.fill_between(x=x, y1=y1, y2=y2, label=r'MW satellites', color=plot_colours[1])

    #Boddy18
    x = np.array([np.log10(15. * (10. ** 3.)), 6., 9., 12.])
    y1 = np.log10([8.8e-27, 2.6e-26, 1.5e-25, 1.4e-23]) #Table III/IV, O(1)
    y2 = np.ones_like(y1) * -22.
    ax.fill_between(x=x, y1=y1, y2=y2, label=r'CMB', color=plot_colours[2])

    #Xu18
    x = np.array([7., 9., 10.])
    y1 = np.log10([5.6e-27, 1.2e-26, 5.8e-26])
    y2 = np.ones_like(y1) * -22.
    ax.fill_between(x=x, y1=y1, y2=y2, label=r'Ly-$\alpha$f (SDSS-I)', color=plot_colours[3])

    #XQC
    data_xqc = np.loadtxt('/Users/keir/Data/XQC.dat')
    x = np.log10(data_xqc[:, 0] * 1.e+9)
    y1 = np.log10(data_xqc[:, 1])
    y2 = np.ones_like(y1) * -22.
    ax.fill_between(x=x, y1=y1, y2=y2, label=r'X-ray Quantum Calorimeter', color='#D3D3D3', alpha=0.5)
    #plot_colours[4]) #alpha=0.5,

    #CRESST-surface & EDELWEISS-Migdal/standard
    data_cresst_surface = np.loadtxt('/Users/keir/Data/CRESST_EDELWEISS_combined.dat')
    x = np.log10(data_cresst_surface[:, 0] * 1.e+9)
    y1 = np.log10(data_cresst_surface[:, 1])
    y2 = np.log10(data_cresst_surface[:, 2])
    #x0_idx = np.where(data_cresst_surface[:, 0] == 1.07199)[0][0]
    #x1_idx = np.where(data_cresst_surface[:, 0] == 3.82586)[0][0]
    #y2_interpolator = spi.interp1d(x[np.array([x0_idx, x1_idx])], y2[np.array([x0_idx, x1_idx])])
    #y2[np.argwhere(np.isnan(y2))] = y2_interpolator(x[np.argwhere(np.isnan(y2))])
    ax.fill_between(x=x, y1=y1, y2=y2, label=r'Direct detection', color='#A9A9A9', alpha=0.5)
    #plot_colours[5]) #[CRESST-surface] #alpha=0.5,

    #EDELWEISS-Migdal
    '''data_edelweiss_migdal = np.loadtxt('/Users/keir/Data/EDELWEISS-Migdal.dat')
    x = np.log10(data_edelweiss_migdal[:, 0] * 1.e+9)
    y1 = np.log10(data_edelweiss_migdal[:, 1])
    y2 = np.log10(data_edelweiss_migdal[:, 2])
    ax.fill_between(x=x, y1=y1, y2=y2, color='#A9A9A9', alpha=0.5)
    '''

    ax.set_xlabel(r'$\mathrm{log}\,[\mathrm{Dark\,\,matter\,\,mass}\,\,m\,(\mathrm{eV})]$')
    ax.set_ylabel(r'$\mathrm{log}\,[\mathrm{Proton-DM\,\,cross\,\,section}\,\,\sigma\,(\mathrm{cm}^2)]$')
    ax.legend(frameon=False, loc='lower right', fontsize=19., ncol=2)
    ax.set_xlim([np.log10(15. * (10. ** 3.)), 11.])
    ax.set_ylim([-31., -24.])
    fig.subplots_adjust(top=0.98, right=0.98, bottom=0.15)

    #samples = np.loadtxt(chainfile, usecols=(14, 15), max_rows=10000)
    '''posterior_MCsamples = gd.MCSamples(samples=samples, names=param_names, labels=param_labels, ranges=param_ranges)
    posterior_MCsamples.updateSettings({'contours': [1., 0.95]}) #0.9999999
    subplot_instance = gdp.get_single_plotter() #getSubplotPlotter()
    #subplot_instance.plots_2d(param1='logmass', params2=['logsigma',], roots=[posterior_MCsamples,], filled=False)
    subplot_instance.settings.num_plot_contours = 2
    subplot_instance.plot_2d(posterior_MCsamples, 'logmass', 'logsigma', filled=True, lims=[4., 11., -31., -24.])
    '''
    #ax = sb.kdeplot(samples[:, 0], samples[:, 1], levels=2, thresh=0.95) #, shade=True, shade_lowest=True) #n_levels=1

    plt.savefig('/Users/keir/Documents/paper_bDM/' + save_file)

def plot_convergence():
    """Plot the marginalised posterior summary statistics convergence."""
    sim_num = np.array([0, 3, 6, 12, 18, 21, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
    slice_array = np.concatenate((np.arange(0, 8), np.array([10, 13, 16, 19])))
    sim_num = sim_num[slice_array]
    redshifts = [4.95, 4.58, 4.24]

    convergence_data = np.load('/Users/keir/Software/lya_emulator/plots/convergence_bDM_mass7_batch11.npz')
    posterior_means = convergence_data['arr_0']
    posterior_means = posterior_means[slice_array, :]
    posterior_limits = convergence_data['arr_1']
    posterior_limits = posterior_limits[slice_array, :, :]
    convergence_data = np.load('/Users/keir/Software/lya_emulator/plots/convergence_bDM_mass5_batch11.npz')
    posterior_means = np.insert(posterior_means, -1, convergence_data['arr_0'][:, -1], axis=1)
    posterior_limits = np.insert(posterior_limits, -1, convergence_data['arr_1'][:, -1, :], axis=1)
    convergence_data = np.load('/Users/keir/Software/lya_emulator/plots/convergence_bDM_mass9_batch11.npz')
    posterior_means = np.concatenate((posterior_means, convergence_data['arr_0'][:, -1].reshape(-1, 1)), axis=1)
    posterior_limits = np.concatenate((posterior_limits, convergence_data['arr_1'][:, -1, :].reshape(-1, 1, 4)), axis=1)

    one_sigma = (posterior_limits[:, :, 2] - posterior_limits[:, :, 1]) / 2.
    two_sigma = (posterior_limits[:, :, 3] - posterior_limits[:, :, 0]) / 2.
    mean_diff = (posterior_means[1:] - posterior_means[:-1]) / (two_sigma[:-1] / 2.)
    one_sigma_diff = (one_sigma[1:] - one_sigma[:-1]) / (two_sigma[:-1] / 2.)
    two_sigma_diff = (two_sigma[1:] - two_sigma[:-1]) / (two_sigma[:-1] / 2.)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6.4*2., 6.4*2.5))
    plot_labels = [r'$\tau_0 (z = %.1f)$'%redshifts[0], r'$\tau_0 (z = %.1f)$'%redshifts[1],
                   r'$\tau_0 (z = %.1f)$'%redshifts[2], r'$n_\mathrm{s}$', r'$A_\mathrm{s}$',
                   r'$T_0 (z = %.1f)$'%redshifts[0], r'$T_0 (z = %.1f)$'%redshifts[1], r'$T_0 (z = %.1f)$'%redshifts[2],
                   r'$\widetilde{\gamma} (z = %.1f)$'%redshifts[0], r'$\widetilde{\gamma} (z = %.1f)$'%redshifts[1],
                   r'$\widetilde{\gamma} (z = %.1f)$'%redshifts[2], r'$u_0 (z = %.1f)$'%redshifts[0],
                   r'$u_0 (z = %.1f)$'%redshifts[1], r'$u_0 (z = %.1f)$'%redshifts[2],
                   r'$\log[\sigma (m = 100\,\mathrm{keV})]$', r'$\log[\sigma (m = 10\,\mathrm{MeV})]$',
                   r'$\log[\sigma (m = 1\,\mathrm{GeV})]$'] #[\mathrm{cm}^2]]$']
    colours = lyc.get_distinct(8)
    colours += colours[:8]
    colours += [colours[0],]
    line_style = '-'

    for i in range(len(plot_labels)):
        if i > 7:
            line_style = '--'
        if i > 15:
            line_style = ':'

        if i < 5:
            plot_label = plot_labels[i]
        else:
            plot_label = None
        axes[0].plot(sim_num[1:], mean_diff[:, i], label=plot_label, color=colours[i], lw=2.5, ls=line_style)

        if (i > 4) and (i < 11):
            plot_label = plot_labels[i]
        else:
            plot_label = None
        axes[1].plot(sim_num[1:], one_sigma_diff[:, i], label=plot_label, color=colours[i], lw=2.5, ls=line_style)

        if i > 10:
            plot_label = plot_labels[i]
        else:
            plot_label = None
        axes[2].plot(sim_num[1:], two_sigma_diff[:, i], label=plot_label, color=colours[i], lw=2.5, ls=line_style)

    axes[0].axhline(y=0.2, color='black', ls='--', lw=3.) #2.5)
    axes[0].axhline(y=-0.2, color='black', ls='--', lw=3.) #2.5)
    axes[1].axhline(y=0.2, color='black', ls='--', lw=3.) #2.5)
    axes[1].axhline(y=-0.2, color='black', ls='--', lw=3.) #2.5)
    axes[2].axhline(y=0.2, color='black', ls='--', lw=3.) #2.5)
    axes[2].axhline(y=-0.2, color='black', ls='--', lw=3.) #2.5)

    axes[2].set_xlabel(r'Optimization simulation number')
    axes[0].set_ylabel(r'\textnumero\,\,of sigma shift [mean]')
    axes[1].set_ylabel(r'\textnumero\,\,of sigma shift [1 sigma]')
    axes[2].set_ylabel(r'\textnumero\,\,of sigma shift [2 sigma]')
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[2].set_ylim([-0.8, 1.9])
    axes[0].legend(frameon=False, loc='upper right', ncol=2, fontsize=25.)
    axes[1].legend(frameon=False, loc='upper right', ncol=2, fontsize=25.)
    axes[2].legend(frameon=False, loc='upper right', ncol=2, fontsize=25.)
    fig.subplots_adjust(top=0.99, bottom=0.06, right=0.95, hspace=0.05)
    plt.savefig('/Users/keir/Documents/paper_bDM/convergence3.pdf')

    return posterior_means, posterior_limits

def plot_posterior(parameters='all'):
    """Make a triangle plot of marginalised 1D and 2D posteriors."""
    if parameters == 'all':
        n_chains = 4
        save_file = 'posterior2.pdf'
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
    chainfile_root = '/Users/keir/Data/emulator/bDM'
    if (parameters == 'all') or (parameters == 'logma'):
        chainfiles[
            0] = 'chain_ns0.964As1.83e-09heat_slope0heat_amp1omega_m0.321alpha0beta1gamma-1z_rei8T_rei2e+04_1_vary_mass_40000_batch8.txt'
        chainfiles[
            1] = 'chain_ns0.964As1.83e-09heat_slope0heat_amp1omega_m0.321alpha0beta1gamma-1z_rei8T_rei2e+04_1_vary_mass_40000_batch8.txt'
        chainfiles[
            2] = 'chain_ns0.964As1.83e-09heat_slope0heat_amp1omega_m0.321alpha0beta1gamma-1z_rei8T_rei2e+04_1_vary_mass_40000_batch10_2.txt'
        chainfiles[
            3] = 'chain_ns0.964As1.83e-09heat_slope0heat_amp1omega_m0.321alpha0beta1gamma-1z_rei8T_rei2e+04_1_vary_mass_40000_batch11.txt'
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
                       'logm', 'logs']
    parameter_labels = [r'\tau_0^{5.0}', r'\tau_0 (z = 4.6)',
                        r'\tau_0 (z = 4.2)', r'n_\mathrm{s}', r'A_\mathrm{s}',
                        r'T_0^{5.0}', r'T_0 (z = 4.6)', r'T_0 (z = 4.2)',
                        r'\widetilde{\gamma}^{5.0}', r'\widetilde{\gamma} (z = 4.6)',
                        r'\widetilde{\gamma} (z = 4.2)', r'u_0^{5.0}',
                        r'u_0 (z = 4.6)', r'u_0 (z = 4.2)',
                        r'\log\,m', r'\log\,\sigma'] #^\mathrm{eV}
    if parameters == 'PRL':
        parameter_labels[-1] = r'\log(m_\mathrm{a} [\mathrm{eV}])'

    legend_labels = [r'Initial emulator', r'After 30 optimization simulations',
                     r'After 35 optimization simulations',
                     r'After 39 optimization simulations']
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
            samples[i] = np.loadtxt(chainfiles[i]) #, max_rows=4500)
        else:
            samples[i] = np.loadtxt(chainfiles[i]) #, max_rows=4500)
        samples[i][:, 4] *= 1.e+9
        if (parameters == 'all') or (parameters == 'logma') or (parameters == 'mock'):
            samples[i][:, 5] /= 1.e+4
            samples[i][:, 6] /= 1.e+4
            samples[i][:, 7] /= 1.e+4
            samples[i] = samples[i][:, np.array([1, 3, 4, 6, 9, 12, 14, 15])]
        if parameters == 'logma':
            samples[i] = samples[i][:, -1].reshape(-1, 1)
    if (parameters == 'all') or (parameters == 'mock'):
        width_inch = 6.4*2.5
        legend_loc = 'upper right'
        tick_label_size = 16.
        parameter_names = [parameter_names[1],] + parameter_names[3:5] + [parameter_names[6],] + [parameter_names[9],]\
                          + [parameter_names[12],] + parameter_names[14:]
        parameter_labels = [parameter_labels[1],] + parameter_labels[3:5] + [parameter_labels[6],]\
                           + [parameter_labels[9],] + [parameter_labels[12],] + parameter_labels[14:]
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
                                              ranges={'logm': [4., 11.]})

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

    with open('/Users/keir/Data/emulator/bDM/emulator_params_bDM_batch11_TDR_u0.json', 'r') as json_file:
        json_dict = json.load(json_file)
    emulator_samples = np.concatenate((np.array(json_dict['sample_params'])[:, :2],
                                       np.array(json_dict['measured_sample_params'])[:, np.array([1, 4, 7])]), axis=1)
    log_mass = np.zeros((emulator_samples.shape[0], 1))
    log_sigma = np.zeros((emulator_samples.shape[0], 1))
    for i in range(50, emulator_samples.shape[0]):
        log_mass[i, 0], log_sigma[i, 0] = bDM_numerical_model_inverse(np.array(json_dict['sample_params'])[i, 5:8])
    emulator_samples = np.concatenate((emulator_samples, log_mass, log_sigma), axis=1)
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
                        ax.scatter(emulator_samples[50:80, q-1], emulator_samples[50:80, p-1], color=colours[1], marker='+', s=msize)
                        ax.scatter(emulator_samples[80:85, q-1], emulator_samples[80:85, p-1], color=colours[2], marker='+', s=msize)
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
    plt.savefig('/Users/keir/Documents/paper_bDM/' + save_file)

def _get_emulator(args_list):
    """Get the emulator output for cross-validation test i."""
    i, emudir, emu_json, flux_power_file, training_parameters, n_sims = args_list
    print('Getting data for simulation number', i)
    mf_instance = lym.FreeMeanFlux()
    emu_instance = lyc.nCDMEmulator(emudir, mf=mf_instance, leave_out_validation=np.array([i, ]))
    emu_instance.load(dumpfile=emu_json)
    test_parameters = emu_instance.get_combined_params(use_all=True)[i]
    test_parameters = np.concatenate((np.array([[1., ], ]), test_parameters.reshape(1, -1)), axis=1)

    GP_instance = emu_instance.get_emulator(use_measured_parameters=True, redshift_dependent_parameters=True,
                                            savefile=flux_power_file)
    test_parameters_tau0 = training_parameters[np.arange(i, training_parameters.shape[0], n_sims)]
    GP_mean = [None] * test_parameters_tau0.shape[0]
    GP_std = [None] * test_parameters_tau0.shape[0]
    for j, test_parameters_tau0_single in enumerate(test_parameters_tau0):
        print('Getting data for mean flux sample number', j)
        npt.assert_array_equal(test_parameters_tau0_single[1:], test_parameters[0, 1:])
        tau0 = np.ones(emu_instance.redshifts.size) * test_parameters_tau0_single[0]
        GP_mean_single, GP_std_single = GP_instance.predict(test_parameters, tau0_factors=tau0)
        GP_mean[j] = GP_mean_single[0]
        GP_std[j] = GP_std_single[0]
    return [GP_mean, GP_std]

def make_error_distribution():
    """Calculate the emulator error distribution for leave-one-out cross-validation."""
    emudir = '/share/data2/keir/Simulations/nCDM_emulator_512'
    emu_json = 'emulator_params_bDM_batch11_TDR_u0.json' #'emulator_params_batch18_2_TDR_u0.json'
    flux_power_file = 'bDM_batch11_emulator_flux_vectors.hdf5' #'batch18_2_emulator_flux_vectors.hdf5'
    n_sims = 89 #50 #93
    mf_instance = lym.FreeMeanFlux()

    emu_instance_full = lyc.nCDMEmulator(emudir, mf=mf_instance)
    emu_instance_full.load(dumpfile=emu_json)
    training_parameters, k, training_flux_powers = emu_instance_full.get_flux_vectors(kfunits='mpc',
                                                    redshifts=emu_instance_full.redshifts, pixel_resolution_km_s=1.,
                                                    use_measured_parameters=True, savefile=flux_power_file)

    GP_mean = [None] * training_flux_powers.shape[0] #np.zeros_like(training_flux_powers)
    GP_std = [None] * training_flux_powers.shape[0] #np.zeros_like(GP_mean)

    pool = mg.Pool(60)
    args_list = [(i, emudir, emu_json, flux_power_file, training_parameters, n_sims) for i in range(n_sims)]
    emu_output = pool.map(_get_emulator, args_list)

    for i in range(n_sims):
        for j in range(int(training_flux_powers.shape[0] / n_sims)):
            idx = (j * n_sims) + i
            GP_mean[idx] = emu_output[i][0][j] #GP_mean_single[0]
            GP_std[idx] = emu_output[i][1][j] #GP_std_single[0]

    return k, emu_instance_full.redshifts, training_parameters, training_flux_powers, np.array(GP_mean),\
           np.array(GP_std), emu_instance_full._get_k_max_emulated_h_Mpc()

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
        save_file = 'validationUS2.pdf'
    elif distribution == 'data':
        errors = np.log10(s[:(n_sims * n_mf)] / s_data_expand) #np.log10(
        errors_real = np.log10(np.absolute(m - f_cut)[:(n_sims * n_mf)] / s_data_expand)
        kernel_bw = 'scott' #0.2
        colours = lyc.get_distinct(4)
        ylim = [-5., 2.5]
        ylabel = r'$\mathrm{log}\,(\sigma_\mathrm{Theory}\,/\,\sigma_\mathrm{Data})$'
        text_height = 0.1
        legend_loc = 'lower center'
        save_file = 'data_error_optimise2.pdf'
    LH_cut = np.sort([np.arange(i, i+(n_LH + n_BO), 1) for i in range(0, n_sims*n_mf, n_sims)], axis=None)
    BO_cut = np.sort([np.arange(i+n_LH, i+n_sims, 1) for i in range(0, n_sims*n_mf, n_sims)], axis=None)
    errors_LH = errors[LH_cut]
    errors_BO = errors[BO_cut]
    errors_list = [errors_BO, errors_LH]
    if distribution == 'data':
        errors_list_real = [errors_real[BO_cut], errors_real[BO_cut]] #[errors_real[BO_cut], errors_real[LH_cut]]

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
            elif distribution == 'data':
                k_bin_df = np.concatenate((np.ravel(k_bin_list[j][:, z_cut]), np.ravel(k_bin_list[0][:, z_cut])))
                #np.tile(np.ravel(k_bin_list[j][:, z_cut]), 2)
                errors_df = np.concatenate(
                    (np.ravel(errors_list[j][:, z_cut]), np.ravel(errors_list_real[j][:, z_cut])))
                if j == 0:
                    samples_label = r'Emulator / Data [optimisation simulations]'
                    samples_label_real = r'$|\mathrm{Mean - Truth}|$ / Data [optimisation simulations]'
                    violin_colours = {samples_label: colours[0], samples_label_real: colours[1]}
                else:
                    samples_label = r'Emulator / Data [all simulations]'
                    samples_label_real = r'$|\mathrm{Mean - Truth}|$ / Data [optimization simulations]'
                    violin_colours = {samples_label: colours[2], samples_label_real: colours[3]}
                split_cut_df = ([samples_label,] * errors_list[j][:, z_cut].size) + ([samples_label_real,] * errors_list[0][:, z_cut].size) #0 --> j
                axes_idx = idx - 3
            print(i, j, k_bin_df.shape, errors_df.shape, len(split_cut_df))

            if distribution == 'validation':
                axes[axes_idx].axhspan(ymin=-3., ymax=3., alpha=0.075, color=colours[0])
                axes[axes_idx].axhspan(ymin=-1., ymax=1., alpha=0.15, color=colours[0])

            data_frames[idx] = pd.DataFrame({'kbin': k_bin_df, 'ErrorSigmas': errors_df, 'Distribution': split_cut_df})
            sb.violinplot(data_frames[idx].kbin, data_frames[idx].ErrorSigmas, data_frames[idx].Distribution,
                          ax=axes[axes_idx], scale='width', bw=kernel_bw, inner=None, split=True, palette=violin_colours, cut=0.,
                          linewidth=2.5, saturation=1.)
            # #cut=0

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
    plt.savefig('/Users/keir/Documents/emulator_paper_axions/' + save_file)
    return k, z, p, f, m, s, k_data, s_data, k_max, data_frames


if __name__ == "__main__":
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=24.) #24 normally #32 for convergence

    plt.rc('axes', linewidth=1.5)
    plt.rc('xtick.major', width=1.5)
    plt.rc('xtick.minor', width=1.5)
    plt.rc('ytick.major', width=1.5)
    plt.rc('ytick.minor', width=1.5)

    #plot_comparison()
    #plot_transfer_function()
    #plot_scale()
    #plot_convergence()
    #plot_posterior()

    k, z, p, f, m, s, k_max = make_error_distribution()
    np.savez('/home/keir/Software/lya_emulator/plots/cross_validation_bDM_89.npz', k=k, z=z, p=p, f=f, m=m, s=s,
             k_max=k_max)
