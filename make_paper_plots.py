"""Make plots for the first emulator paper"""
import os.path as path
import numpy as np
import latin_hypercube
import coarse_grid
import flux_power
from quadratic_emulator import QuadraticEmulator
from mean_flux import ConstMeanFlux
import lyman_data
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
from plot_latin_hypercube import plot_points_hypercube
import coarse_grid_plot
import GPy
import distinct_colours_py3 as dc

plotdir = path.expanduser("~/papers/emulator_paper_1/plots")
#plotdir = '/home/keir/Plots/Emulator'
#plotdir = '/Users/kwame/Papers/emulator_paper_1/plots'

def hypercube_plot():
    """Make a plot of some hypercubes"""
    limits = np.array([[0,1],[0,1]])
    cut = np.linspace(0, 1, 8 + 1)
    # Fill points uniformly in each interval
    a = cut[:8]
    b = cut[1:8 + 1]
    #Get list of central values
    xval = (a + b)/2
    plot_points_hypercube(xval, xval)
    plt.savefig(path.join(plotdir,"latin_hypercube_bad.pdf"))
    plt.clf()
    xval = (a + b)/2
    xval_quad = np.concatenate([xval, np.repeat(xval[3],8)])
    yval_quad = np.concatenate([np.repeat(xval[3],8),xval])
    ndivision = 8
    xticks = np.linspace(0,1,ndivision+1)
    plt.scatter(xval_quad, yval_quad, marker='o', s=300, color="blue")
    plt.grid(b=True, which='major')
    plt.xticks(xticks)
    plt.yticks(xticks)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(path.join(plotdir,"latin_hypercube_quadratic.pdf"))
    plt.clf()
    samples = latin_hypercube.get_hypercube_samples(limits, 8)
    plot_points_hypercube(samples[:,0], samples[:,1])
    plt.savefig(path.join(plotdir,"latin_hypercube_good.pdf"))
    plt.clf()


def dlogPfdt(spec, t1, t2):
    """Computes the change in flux power with optical depth"""
    pf1 = spec.get_flux_power_1D("H",1,1215,mean_flux_desired=np.exp(-t1))
    pf2 = spec.get_flux_power_1D("H",1,1215,mean_flux_desired=np.exp(-t2))
    return (pf1[0], (np.log(pf1[1]) - np.log(pf2[1]))/(t1-t2))

def show_t0_gradient(spec, tmin,tmax,steps=20):
    """Find the mean gradient of the flux power with tau0"""
    tt = np.linspace(tmin,tmax,steps)
    df = [np.mean(dlogPfdlogF(spec, t,t-0.005)[1]) for t in tt]
    return tt, df

def single_parameter_plot():
    """Plot change in each parameter of an emulator from direct simulations."""
    emulatordir = path.expanduser("simulations/hires_s8_quadratic")
    mf = ConstMeanFlux(value=1.)
    emu = coarse_grid.Emulator(emulatordir, mf=mf)
    emu.load()
    par, flux_vectors = emu.get_flux_vectors(max_z=2.4)
    params = emu.param_names
    defpar = par[0,:]
    deffv = flux_vectors[0]
    for (name, index) in params.items():
        ind = np.where(par[:,index] != defpar[index])
        for i in np.ravel(ind):
            tp = par[i,index]
            fp = (flux_vectors[i]/deffv).reshape(-1,len(emu.kf))
            plt.semilogx(emu.kf, fp[0,:], label=name+"="+str(tp)+" (z=2.4)")
        plt.xlim(1e-3,2e-2)
        plt.ylim(ymin=0.6)
        plt.legend(loc=0)
        plt.savefig(path.join(plotdir,"single_param_"+name+".pdf"))
        plt.clf()

def test_s8_plots():
    """Plot emulator test-cases"""
    testdir = path.expanduser("simulations/hires_s8_test")
    quaddir = path.expanduser("simulations/hires_s8_quadratic")
    emudir = path.expanduser("simulations/hires_s8")
    gp_emu = coarse_grid_plot.plot_test_interpolate(emudir, testdir,savedir=path.join(plotdir,"hires_s8"))
    gp_quad = coarse_grid_plot.plot_test_interpolate(quaddir, testdir,savedir=path.join(plotdir,"hires_s8_quadratic"))
    quad_quad = coarse_grid_plot.plot_test_interpolate(quaddir, testdir,savedir=path.join(plotdir,"hires_s8_quad_quad"),emuclass=QuadraticEmulator)
    return (gp_emu, gp_quad, quad_quad)

def test_knot_plots(mf=1, testdir = None, emudir = None, plotdir = None, plotname="", kf_bin_nums=None, data_err=False, max_z=4.2):
    """Plot emulator test-cases"""
    if testdir is None:
        testdir = path.expanduser("~/data/Lya_Boss/hires_knots_test")
    if emudir is None:
        emudir = path.expanduser("~/data/Lya_Boss/hires_knots")
    if plotdir is None:
        plotdir = path.expanduser('~/papers/emulator_paper_1/plots/hires_knots_mf')
    gp_emu = coarse_grid_plot.plot_test_interpolate(emudir, testdir,savedir=plotdir+str(mf),plotname=plotname,mean_flux=mf,max_z=max_z,kf_bin_nums=kf_bin_nums,data_err=data_err)
    return gp_emu

def sample_var_plot():
    """Check the effect of sample variance"""
    mys = flux_power.MySpectra()
    sd = lyman_data.SDSSData()
    kf = sd.get_kf()
    fp0 = mys.get_snapshot_list("simulations/hires_sample/ns1.1As2.1e-09heat_slope0heat_amp1hub0.7/output/")
    fp1 = mys.get_snapshot_list("simulations/hires_sample/ns1.1As2.1e-09heat_slope0heat_amp1hub0.7seed1/output/")
    fp2 = mys.get_snapshot_list("simulations/hires_sample/ns1.1As2.1e-09heat_slope0heat_amp1hub0.7seed2/output/")
    pk0 = fp0.get_power(kf,mean_fluxes=None)
    pk1 = fp1.get_power(kf,mean_fluxes=None)
    pk2 = fp2.get_power(kf,mean_fluxes=None)
    nred = len(mys.zout)
    nk = len(kf)
    assert np.shape(pk0) == (nred*nk,)
    for i in (5,10):
        plt.semilogx(kf,(pk1/pk2)[i*nk:(i+1)*nk],label="Seed 1 z="+str(mys.zout[i]))
        plt.semilogx(kf,(pk0/pk2)[i*nk:(i+1)*nk],label="Seed 2 z="+str(mys.zout[i]))
    plt.xlabel(r"$k_F$ (s/km)")
    plt.ylabel(r"Sample Variance Ratio")
    plt.title("Sample Variance")
    plt.xlim(xmax=0.05)
    plt.legend(loc=0)
    plt.savefig(path.join(plotdir, "sample_var.pdf"))
    plt.clf()

def make_plot_Bayesian_optimisation(f_name):
    n_params = 1 #theta
    (a, b, c, d) = (2., -4., -3., 2.)
    plotting_theta = np.linspace(-0.75, 1.9, num=100) #np.linspace(-2., 10., num=100)
    training_theta = np.linspace(-0.7, 1.85, num=3).reshape((-1, 1)) #np.linspace(-1., 9., num=3).reshape((-1, 1))
    training_theta = np.concatenate((training_theta, np.array([[1.8],])))
    training_theta = np.concatenate((training_theta, np.array([[0.5 * (training_theta[0,0] + training_theta[1,0])],])))
    training_theta[-1, 0] += 0. #-= 0.05
    theta_true = 0.25 #2.
    data_sigma = 2.
    alpha = [2., 40.]

    f_true = lambda theta: a + (b * theta) + (c * ((theta - 0.) ** 2)) + (d * ((theta + 0.) ** 3))
    #f_true = lambda theta: np.sin(a * 2. * np.pi * theta)
    #f_true = lambda theta: 2. - np.exp(-1.*((theta-2)**2)) - np.exp(-0.1*((theta-6)**2)) - (1./((theta**2)+1.))
    training_f = f_true(training_theta)

    figure, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 14))  # , sharex='True')
    plot_instance = Plot(font_size=15.)

    for i in range(2):
        GP_kernel = GPy.kern.Linear(n_params)
        GP_kernel += GPy.kern.RBF(n_params)
        if i == 1:
            #GP_kernel.linear.variances.constrain_bounded(2.4, 2.6)
            #GP_kernel.rbf.variance.constrain_bounded(2.6, 2.8)
            GP_kernel.rbf.lengthscale.constrain_bounded(0.25, 0.35)
        GP_instance = GPy.models.GPRegression(training_theta, training_f, kernel=GP_kernel, noise_var=1.e-10)
        GP_instance.optimize(messages=True)
        print('Trained Gaussian process:', GP_instance)

        f_emulator = lambda theta: GP_instance.predict(plotting_theta.reshape((-1, 1)))[0].flatten()
        f_emulator_variance = lambda theta: GP_instance.predict(plotting_theta.reshape((-1, 1)))[1].flatten()
        log_likelihood = lambda theta, f, emu_variance: -0.5 * ((f_true(theta_true) - f(theta)) ** 2) / (emu_variance(theta) + ((data_sigma * f_true(theta_true)) ** 2))
        log_likelihood_emulator = lambda theta: log_likelihood(theta, f_emulator, f_emulator_variance)
        log_likelihood_true = lambda theta: log_likelihood(theta, f_true, lambda theta: 0.)
        acquisition = lambda theta: log_likelihood_emulator(theta) + (alpha[i] * f_emulator_variance(theta) / ((data_sigma * f_true(theta_true)) ** 2))

        line_labels = [r'$f_\mathrm{true}$', r'$f_\mathrm{emulated}$', r'$\mathcal{L}(f_\mathrm{true})$', r'$\mathcal{L}(f_\mathrm{emulated}, \sigma_\mathrm{emulated})$', None]
        dis_cols = dc.get_distinct(2)
        line_colours = dis_cols
        x_label = [r'', r'$\theta$']
        y_label = [r'$f(\theta)$', r'$\mathcal{L}(f(\theta), \sigma(\theta))$', r'$\mathcal{A}(\mathcal{L}(f_\mathrm{emulated}), \sigma_\mathrm{emulated})$']
        x_log_scale = False
        y_log_scale = False
        line_styles = ['--', '-']

        figure, axis = plot_instance.plot_lines([plotting_theta,]*2, [f_true(plotting_theta), f_emulator(plotting_theta)], line_labels[:2], line_colours, x_label[0], y_label[0], x_log_scale, y_log_scale, line_styles=line_styles, fig=figure, ax=axes[0, i], errorbars=[None, 1. * np.sqrt(f_emulator_variance(plotting_theta))], error_band=True)
        figure, axis = plot_instance.plot_lines([plotting_theta,]*2, [log_likelihood_true(plotting_theta), log_likelihood_emulator(plotting_theta)], line_labels[2:4], line_colours, x_label[0], y_label[1], x_log_scale, y_log_scale, line_styles=line_styles, fig=figure, ax=axes[1, i])
        figure, axis = plot_instance.plot_lines([plotting_theta,], [acquisition(plotting_theta),], [line_labels[4],], [line_colours[1],], x_label[1], y_label[2], x_log_scale, y_log_scale, line_styles=[line_styles[1],], fig=figure, ax=axes[2, i])

        axes[0, i].axvline(x=theta_true, color='black', ls='-')
        axes[1, i].axvline(x=theta_true, color='black', ls='-')
        axes[2, i].axvline(x=theta_true, color='black', ls='-')
        for training_sample in training_theta:
            axes[0, i].axvline(x=training_sample[0], color='black', ls=':')
            axes[1, i].axvline(x=training_sample[0], color='black', ls=':')
            axes[2, i].axvline(x=training_sample[0], color='black', ls=':')
        for axis in axes.flatten():
            axis.legend(frameon=True, edgecolor='white', facecolor='white',
                        framealpha=1.)  # , ncol=2) #, loc='upper right')

        #After Bayesian optimisation
        optimisation_theta = plotting_theta[np.argmax(acquisition(plotting_theta))]
        print('Optimisation theta =', optimisation_theta)
        training_theta = np.concatenate((training_theta, np.array([[optimisation_theta],])))
        training_f = f_true(training_theta)
        axes[2, i].axvline(x = optimisation_theta, color=line_colours[1], ls=':')

    #axis.set_xlim([1.e-2, 1.])
    #axis.set_ylim([-0.00155, 0.00595])
    #axis.legend(frameon=True, edgecolor='white', facecolor='white', framealpha=1.) #, ncol=2) #, loc='upper right')
    #axis.annotate('', xy = (0.35, 0.35), xytext = (0.48, 0.77), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'})
    #axis.annotate('', xy = (0.35, 0.35), xytext = (0.61, 0.91), xycoords = 'axes fraction', textcoords = 'axes fraction', arrowprops = {'arrowstyle': '->'}) #sim_fitted
    #plt.text(0.35, 0.35, r'Decreasing $\mu$', transform=axis.transAxes, horizontalalignment = 'center', verticalalignment = 'top', fontsize = 12.0)
    #plt.text(0.61, 0.91, r'Decreasing $\mu$', transform=axis.transAxes, horizontalalignment = 'left', verticalalignment = 'bottom', fontsize = 12.0) #sim_fitted

    figure.subplots_adjust(right=0.97, left=0.17, bottom=0.13)  # figure.subplots_adjust(hspace=0., right=0.98, top=0.99, bottom=0.04, left=0.08)
    plt.savefig(f_name)


class Plot():
    """Class to make plots"""
    def __init__(self, font_size = 15.0):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=font_size)

        plt.rc('axes', linewidth=1.5)
        plt.rc('xtick.major', width=1.5)
        plt.rc('xtick.minor', width=1.5)
        plt.rc('ytick.major', width=1.5)
        plt.rc('ytick.minor', width=1.5)
        #plt.rc('lines', linewidth=1.0)

    def plot_lines(self, x, y, line_labels, line_colours, x_label, y_label, x_log_scale, y_log_scale, line_styles='default', line_weights='default', marker_styles = None, plot_title='', fig = None, ax = None, errorbars=False, errorbar_widths='default', error_band=False, reverse_legend=False):
        n_lines = len(line_labels)
        if line_styles == 'default':
            line_styles = ['-'] * n_lines
        if line_weights == 'default':
            line_weights = [1.5,] * n_lines
        if marker_styles == None:
            marker_styles = [''] * n_lines
        if fig == None:
            fig, ax = plt.subplots(1) #, figsize=(8, 12))
        if errorbar_widths == 'default':
            errorbar_widths = [1.5,] * n_lines
        if reverse_legend == False:
            line_iterator = range(n_lines)
        elif reverse_legend == True:
            line_iterator = range(n_lines)[::-1]
        else:
            line_iterator = reverse_legend
        for i in line_iterator:
            print(i)
            ax.plot(x[i], y[i], label=line_labels[i], color=line_colours[i], ls=line_styles[i], lw=line_weights[i], marker=marker_styles[i])
            if errorbars is not False:
                if errorbars[i] is not None:
                    if error_band is False:
                        ax.errorbar(x[i], y[i], yerr=errorbars[i], ecolor=line_colours[i], ls='', elinewidth=errorbar_widths[i])
                    else:
                        ax.fill_between(x=x[i], y1=y[i]-errorbars[i], y2=y[i]+errorbars[i], color=line_colours[i], alpha=0.5)
        ax.legend(frameon=False, fontsize=15.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(plot_title)
        if x_log_scale == True:
            ax.set_xscale('log')
        if y_log_scale == True:
            ax.set_yscale('log')
        fig.subplots_adjust(right=0.99)

        return fig, ax

    def plot_histograms(self, arrays, bin_edges, labels, colours, x_label, y_label, x_log_scale, y_log_scale, plot_title='', fig = None, ax = None):
        n_histograms = len(labels)
        if fig == None:
            fig, ax = plt.subplots(1) #, figsize=(8, 12))
        for i in range(n_histograms):
            print('Plotting histogram', str(i+1))
            ax.hist(arrays[i], bins = bin_edges[i], label = labels[i], color = colours[i])
        ax.legend(frameon=False, fontsize=15.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(plot_title)
        if x_log_scale == True:
            ax.set_xscale('log')
        if y_log_scale == True:
            ax.set_yscale('log')
        fig.subplots_adjust(right=0.99)

        return fig, ax


if __name__ == "__main__":
    #sample_var_plot()
    #hypercube_plot()
    #single_parameter_plot()
    #test_s8_plots()
    #test_knot_plots(mf=1)
    #test_knot_plots(mf=2)
    make_plot_Bayesian_optimisation('/Users/kwame/Documents/emulator_paper_1/refinement_paper/Bayesian_optimisation_a8.pdf')
