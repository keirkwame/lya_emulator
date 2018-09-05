import scipy.optimize as spo

from latin_hypercube import *

def optimise_acquisition_function_parallel(arguments):
    """Version of optimise_acquisition_function for multiprocessing. Sits in separate file so we can use interactive Python environments"""
    starting_parameters, likelihood_class_instance, optimisation_bounds, nu, exploitation_weight, integration_bounds = arguments
    acquisition_function = lambda parameters: -1. * likelihood_class_instance.acquisition_function_GP_UCB_marginalised_mean_flux(map_from_unit_cube(parameters, likelihood_class_instance.param_limits[2:]), nu=nu, exploitation_weight=exploitation_weight, integration_bounds=integration_bounds)
    return spo.minimize(acquisition_function, starting_parameters, bounds=optimisation_bounds, options={'disp': True})

