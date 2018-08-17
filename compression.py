import copy as cp
import numpy as np
import numpy.linalg as npl
import numpy.random as npr

import likelihood as li
import mean_flux as mf


class Compression(object):
    """Class to handle data compression"""
    def __init__(self, data_vector, redshift_vector, fixed_covariance_matrix, emulator_object, parameter_vector=None, add_noise=False, random_seed=42):
        """Redshift vector - in reverse order
        Fixed covariance matrix - full (though probably sparse) data_dimensions^2 matrix - should default to zeros"""
        self.redshift_vector = redshift_vector
        self.fixed_covariance_matrix = fixed_covariance_matrix
        self.emulator_object = emulator_object
        self.add_noise = add_noise
        self.random_seed = random_seed

        self.parameter_vector = parameter_vector
        if self.parameter_vector is None:
            self.data_vector = data_vector
            if add_noise:
                self.data_vector += self.get_noise_Gaussian_realisation(self.fixed_covariance_matrix, seed=self.random_seed)
        else:
            self.data_vector, _ = self._get_data_vector_and_covariance_from_emulator(self.parameter_vector, add_noise=self.add_noise)

    def get_noise_Gaussian_realisation(self, covariance_matrix, seed=42):
        """Get a noise realisation as drawn from a multivariate Gaussian distribution with zero mean"""
        npr.seed(seed=seed)
        return npr.multivariate_normal(np.zeros(covariance_matrix.shape[0]), covariance_matrix)

    def _get_data_vector_and_covariance_from_emulator(self, parameter_vector, add_noise=False):
        """Emulate data vector and covariance (and add any fixed covariance)"""
        mean_flux_parameters = mf.mean_flux_slope_to_factor(self.redshift_vector, parameter_vector[0])
        data_vector, std_diagonal = self.emulator_object.predict(parameter_vector[1:].reshape(1, -1), tau0_factors=mean_flux_parameters)
        total_covariance_matrix = self.fixed_covariance_matrix + np.diag(std_diagonal[0] ** 2)
        if add_noise: #Add realisation of noise drawn from Gaussian distribution with fixed covariance matrix
            data_vector += self.get_noise_Gaussian_realisation(self.fixed_covariance_matrix, seed=self.random_seed)
        return data_vector[0], total_covariance_matrix

    def _invert_block_diagonal_covariance(self, full_covariance_matrix):
        """Efficiently invert block diagonal covariance matrix"""
        '''inverse_covariance_matrix = np.zeros_like(full_covariance_matrix)
        nz = self.redshift_vector.shape[0]
        nk = int(full_covariance_matrix.shape[0] / nz)
        for z in range(nz): #Loop over blocks by redshift
            start_index = nk * z
            end_index = nk * (z + 1)
            inverse_covariance_block = npl.inv(full_covariance_matrix[start_index: end_index, start_index: end_index])
            inverse_covariance_matrix[start_index: end_index, start_index: end_index] = inverse_covariance_block'''
        return li.invert_block_diagonal_covariance(full_covariance_matrix, self.redshift_vector.shape[0])

    def _get_gradients_from_emulator(self, parameter_vector, data_vector, covariance):
        """Estimate gradient of data vector and covariance using emulator for perturbed parameter values"""
        data_gradient = np.zeros((parameter_vector.shape[0], data_vector.shape[0]))
        covariance_gradient = np.zeros((parameter_vector.shape[0], covariance.shape[0], covariance.shape[1]))

        for i in range(parameter_vector.shape[0]): #Loop over parameters
            perturbed_parameter_vector = cp.deepcopy(parameter_vector)
            perturbed_parameter_vector[i] *= 1.01
            perturbed_data_vector, perturbed_covariance = self._get_data_vector_and_covariance_from_emulator(perturbed_parameter_vector, add_noise=self.add_noise)
            delta_parameter = perturbed_parameter_vector - parameter_vector
            data_gradient[i, :] = (perturbed_data_vector - data_vector) / delta_parameter[i]
            covariance_gradient[i, :, :] = (perturbed_covariance - covariance) / delta_parameter[i]

        return data_gradient, covariance_gradient

    def _get_gradients_from_emulator_analytical(self, parameter_vector):
        """Get gradient of data vector and covariance using analytical form from emulator model"""
        mean_flux_parameters = mf.mean_flux_slope_to_factor(self.redshift_vector, parameter_vector[0])
        data_gradient, variance_gradient = self.emulator_object.get_predictive_gradients(parameter_vector[1:].reshape(1, -1), tau0_factors=mean_flux_parameters, mean_flux_slope=parameter_vector[0], redshifts=self.redshift_vector, pivot_redshift=2.2)
        covariance_gradient = np.zeros((variance_gradient.shape[1], variance_gradient.shape[2], variance_gradient.shape[2]))
        for i in range(covariance_gradient.shape[0]): #Looping over parameters
            covariance_gradient[i, :, :] = np.diag(variance_gradient[0, i, :]) #Expand variance gradient into covariance
        return data_gradient[0], covariance_gradient

    def get_Fisher_matrix(self, fiducial_parameter_vector):
        """Get the Fisher matrix (for a Gaussian likelihood)"""
        _, fiducial_covariance_matrix = self._get_data_vector_and_covariance_from_emulator(
            fiducial_parameter_vector)
        fiducial_inverse_covariance = self._invert_block_diagonal_covariance(fiducial_covariance_matrix)  # dims: p x p
        fiducial_data_gradient, fiducial_covariance_gradient = self._get_gradients_from_emulator_analytical(
            fiducial_parameter_vector)

        inv_cov_data_grad = np.dot(fiducial_inverse_covariance, fiducial_data_gradient.transpose()) #dims: p x t
        data_gradient_term = np.dot(fiducial_data_gradient, inv_cov_data_grad) #dims: t x t #Don't actually need trace!?
        cov_grad_inv_cov = np.dot(fiducial_covariance_gradient, fiducial_inverse_covariance) #dims: t x p x p
        covariance_gradient_term = 0.5 * np.trace(np.dot(cov_grad_inv_cov, cov_grad_inv_cov), axis1=1, axis2=3)
        #dims: tr[(t x p x p) * (t x p x p)] = tr[t x p x t x p] = t x t

        return data_gradient_term + covariance_gradient_term

    def get_inverse_Fisher_matrix(self, fiducial_parameter_vector):
        """Get the inverse of the Fisher matrix (for a Gaussian likelihood)"""
        return npl.inv(self.get_Fisher_matrix(fiducial_parameter_vector))


class ScoreFunctionCompression(Compression):
    """Sub-class to handle data compression to the score function"""
    def __init__(self, data_vector, redshift_vector, fixed_covariance_matrix, emulator_object, parameter_vector=None, add_noise=False, random_seed=42):
        super(ScoreFunctionCompression, self).__init__(data_vector, redshift_vector, fixed_covariance_matrix, emulator_object, parameter_vector=parameter_vector, add_noise=add_noise, random_seed=random_seed)

    def compress_data_vector(self, fiducial_parameter_vector):
        """Compress data to score function"""
        fiducial_data_vector, fiducial_covariance_matrix = self._get_data_vector_and_covariance_from_emulator(fiducial_parameter_vector, add_noise=self.add_noise)
        fiducial_inverse_covariance = self._invert_block_diagonal_covariance(fiducial_covariance_matrix) #dims: p x p
        fiducial_data_gradient, fiducial_covariance_gradient = self._get_gradients_from_emulator_analytical(fiducial_parameter_vector)
        #, fiducial_data_vector, fiducial_covariance_matrix)

        inv_cov_data_diff = np.dot(fiducial_inverse_covariance, (self.data_vector - fiducial_data_vector)) #dims: p
        data_gradient_term = np.dot(fiducial_data_gradient, inv_cov_data_diff) #dims: (t x p) * p = t
        covariance_gradient_term = 0.5 * np.dot(inv_cov_data_diff, np.dot(fiducial_covariance_gradient, inv_cov_data_diff).transpose())
        #dims: p * [(t x p x p) * p]^T = p * (t x p)^T = t
        trace_term = -0.5 * np.trace(np.dot(fiducial_covariance_gradient, fiducial_inverse_covariance), axis1=1, axis2=2)
        #dims: tr[(t x p x p) * (p x p)] = tr[t x p x p] = t

        return data_gradient_term + covariance_gradient_term + trace_term

    def compress_data_vector_linearised(self, fiducial_parameter_vector):
        """Compress data to linearised form of score function"""
        inv_fish_score = np.dot(self.get_inverse_Fisher_matrix(fiducial_parameter_vector), self.compress_data_vector(fiducial_parameter_vector))
        return fiducial_parameter_vector + inv_fish_score


class IMNNsCompression(Compression):
    """Sub-class to handle data compression by information maximising neural networks (IMNNs)"""
    def __init__(self, data_vector, redshift_vector, fixed_covariance_matrix, emulator_object, parameter_vector=None, add_noise=False, random_seed=42):
        super(IMNNsCompression, self).__init__(data_vector, redshift_vector, fixed_covariance_matrix, emulator_object, parameter_vector=parameter_vector, add_noise=add_noise, random_seed=random_seed)

    def compress_data_vector(self):
        """Compress data by IMNNs"""
        pass


def parameter_vector_to_compressed_data_vector_score_function(parameter_vector, fiducial_parameter_vector, emulator_object, fixed_covar, redshift_vector=np.arange(4.2, 2.1, -0.2), add_noise=False):
    """Helper function for generating compressed data vector (score function) for given parameter vector"""
    compression_instance = ScoreFunctionCompression(None, redshift_vector, fixed_covar, emulator_object, parameter_vector=parameter_vector, add_noise=add_noise)
    return compression_instance.compress_data_vector_linearised(fiducial_parameter_vector)

def data_vector_to_compressed_data_vector_score_function(data_vector, fiducial_parameter_vector, emulator_object, fixed_covar, redshift_vector=np.arange(4.2, 2.1, -0.2), add_noise=False):
    """Helper function for compressing a given data vector (to the score function)"""
    compression_instance = ScoreFunctionCompression(data_vector, redshift_vector, fixed_covar, emulator_object, add_noise=add_noise)
    return compression_instance.compress_data_vector_linearised(fiducial_parameter_vector)
