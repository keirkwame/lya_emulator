import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

import lyaemu.distinct_colours_py3 as lyc
import lyaemu.coarse_grid as lyc
import lyaemu.mean_flux as lym
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

def violinplot_error_distribution():
    """Make a violin-plot of the emulator error distribution for leave-one-out cross-validation."""


if __name__ == "__main__":
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=18.)

    plt.rc('axes', linewidth=1.5)
    plt.rc('xtick.major', width=1.5)
    plt.rc('xtick.minor', width=1.5)
    plt.rc('ytick.major', width=1.5)
    plt.rc('ytick.minor', width=1.5)

    #plot_transfer_function()
    k, z, p, f, m, s, k_max = make_error_distribution()

