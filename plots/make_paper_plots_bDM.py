import os
import json
import h5py

import copy as cp
import numpy as np
import numpy.random as npr
import numpy.testing as npt
import scipy.optimize as spo
import getdist as gd
import getdist.plots as gdp
import multiprocessing as mg
import pandas as pd
import seaborn as sb

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

import lyaemu.distinct_colours_py3 as lyc
#import lyaemu.coarse_grid as lyc
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
    save_file = 'scale.pdf'
    mass = np.linspace(np.log10(15. * (10. ** 3.)), 11., num=8)
    delta_mass = (mass[-1] - mass[0]) / (mass.shape[0] - 1.) / 2.
    sigma = np.linspace(-31., -24., num=8)
    delta_sigma = (sigma[-1] - sigma[0]) / (sigma.shape[0] - 1.) / 2.

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4 * 1.5, 6.4 * (1. + (0.15 / 0.85))))

    l = np.zeros((mass.shape[0], sigma.shape[0]))
    for i, m in enumerate(mass):
        for j, s in enumerate(sigma):
            params_nCDM = bDM_numerical_model(np.array([m, s]), nCDM_parameter_limits, h=h_planck)
            k = (((0.75 ** (1. / (2. * params_nCDM[2]))) - 1.) ** (1. / params_nCDM[1])) / params_nCDM[0]
            l[i, j] = np.log10(2. * np.pi * 1.e+6 / k) #log(pc / h)

    axp = ax.imshow(l.T, origin='lower', aspect='auto',
                extent=(mass[0] - delta_mass, mass[-1] + delta_mass, sigma[0] - delta_sigma, sigma[-1] + delta_sigma))
    cb = plt.colorbar(axp, ax=[ax], location='top') #orientation='horizontal',

    ax.set_xlabel(r'$\mathrm{log}\,[\mathrm{Dark\,\,matter\,\,mass}\,\,m\,(\mathrm{eV})]$')
    ax.set_ylabel(r'$\mathrm{log}\,[\mathrm{Baryon-DM\,\,cross\,\,section}\,\,\sigma\,(\mathrm{cm}^2)]$')
    #fig.subplots_adjust(right=0.99, bottom=0.15)

    plt.savefig('/Users/keir/Documents/paper_bDM/' + save_file)

def plot_comparison():
    """Make a plot comparing mass - sigma bounds."""
    save_file = 'comparison3.pdf'

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
    ax.fill_between(x=x, y1=y1, y2=y2, label=r'Ly-$\alpha$f (previous work)', color=plot_colours[3])

    #XQC
    data_xqc = np.loadtxt('/Users/keir/Data/XQC.dat')
    x = np.log10(data_xqc[:, 0] * 1.e+9)
    y1 = np.log10(data_xqc[:, 1])
    y2 = np.ones_like(y1) * -22.
    ax.fill_between(x=x, y1=y1, y2=y2, label=r'X-ray Quantum Calorimeter', color='#D3D3D3', alpha=0.5)
    #plot_colours[4]) #alpha=0.5,

    #CRESST-surface
    data_cresst_surface = np.loadtxt('/Users/keir/Data/CRESST-surface.dat')
    x = np.log10(data_cresst_surface[:, 0] * 1.e+9)
    y1 = np.log10(data_cresst_surface[:, 1])
    y2 = np.log10(data_cresst_surface[:, 2])
    ax.fill_between(x=x, y1=y1, y2=y2, label=r'Direct detection', color='#A9A9A9', alpha=0.5)
    #plot_colours[5]) #[CRESST-surface] #alpha=0.5,

    ax.set_xlabel(r'$\mathrm{log}\,[\mathrm{Dark\,\,matter\,\,mass}\,\,m\,(\mathrm{eV})]$')
    ax.set_ylabel(r'$\mathrm{log}\,[\mathrm{Baryon-DM\,\,cross\,\,section}\,\,\sigma\,(\mathrm{cm}^2)]$')
    ax.legend(frameon=False, loc='lower right', fontsize=19., ncol=2)
    ax.set_xlim([np.log10(15. * (10. ** 3.)), 11.])
    ax.set_ylim([-31., -24.])
    fig.subplots_adjust(top=0.98, right=0.99, bottom=0.15)

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

if __name__ == "__main__":
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=24.)

    plt.rc('axes', linewidth=1.5)
    plt.rc('xtick.major', width=1.5)
    plt.rc('xtick.minor', width=1.5)
    plt.rc('ytick.major', width=1.5)
    plt.rc('ytick.minor', width=1.5)

    #plot_comparison()
    #plot_transfer_function()
    plot_scale()