import copy as cp
import numpy as np
import numpy.linalg as npl

import mean_flux as mf


class Compression(object):
    """Class to handle data compression"""
    def __init__(self, data_vector, redshift_vector, fixed_covariance_matrix, emulator_object, parameter_vector=None):
        """Redshift vector - in reverse order
        Fixed covariance matrix - full (though probably sparse) data_dimensions^2 matrix - should default to zeros"""
        self.redshift_vector = redshift_vector
        self.fixed_covariance_matrix = fixed_covariance_matrix
        self.emulator_object = emulator_object

        self.parameter_vector = parameter_vector
        if self.parameter_vector is None:
            self.data_vector = data_vector
        else:
            self.data_vector, _ = self._get_data_vector_and_covariance_from_emulator(self.parameter_vector)

    def _get_data_vector_and_covariance_from_emulator(self, parameter_vector):
        """Emulate data vector and covariance (and add any fixed covariance)"""
        mean_flux_parameters = mf.mean_flux_slope_to_factor(self.redshift_vector, parameter_vector[0])
        data_vector, std_diagonal = self.emulator_object.predict(parameter_vector[1:].reshape(1, -1), tau0_factors=mean_flux_parameters)
        total_covariance_matrix = self.fixed_covariance_matrix + np.diag(std_diagonal[0] ** 2)
        return data_vector[0], total_covariance_matrix

    def _invert_block_diagonal_covariance(self, full_covariance_matrix):
        """Efficiently invert block diagonal covariance matrix"""
        inverse_covariance_matrix = np.zeros_like(full_covariance_matrix)
        nz = self.redshift_vector.shape[0]
        nk = full_covariance_matrix.shape[0] / nz
        for z in range(nz): #Loop over blocks by redshift
            start_index = nk * z
            end_index = nk * (z + 1)
            inverse_covariance_block = npl.inv(full_covariance_matrix[start_index: end_index, start_index, end_index])
            inverse_covariance_matrix[start_index: end_index, start_index, end_index] = inverse_covariance_block
        return inverse_covariance_matrix

    def _get_gradients_from_emulator(self, parameter_vector, data_vector, covariance):
        """Estimate gradient of data vector and covariance using emulator for perturbed parameter values"""
        data_gradient = np.zeros((parameter_vector.shape[0], data_vector.shape[0]))
        covariance_gradient = np.zeros((parameter_vector.shape[0], covariance.shape[0], covariance.shape[1]))

        for i in range(parameter_vector.shape[0]): #Loop over parameters
            perturbed_parameter_vector = cp.deepcopy(parameter_vector)
            perturbed_parameter_vector[i] *= 1.01
            perturbed_data_vector, perturbed_covariance = self._get_data_vector_and_covariance_from_emulator(perturbed_parameter_vector)
            delta_parameter = perturbed_parameter_vector - parameter_vector
            data_gradient[i, :] = (perturbed_data_vector - data_vector) / delta_parameter[i]
            covariance_gradient[i, :, :] = (perturbed_covariance - covariance) / delta_parameter[i]

        return data_gradient, covariance_gradient


class ScoreFunctionCompression(Compression):
    """Sub-class to handle data compression to the score function"""
    def __init__(self, data_vector, redshift_vector, fixed_covariance_matrix, emulator_object):
        super(Compression, self).__init__(data_vector, redshift_vector, fixed_covariance_matrix, emulator_object)

    def compress_data_vector(self, fiducial_parameter_vector):
        """Compress data to score function"""
        fiducial_data_vector, fiducial_covariance_matrix = self._get_data_vector_and_covariance_from_emulator(fiducial_parameter_vector)
        fiducial_inverse_covariance = self._invert_block_diagonal_covariance(fiducial_covariance_matrix) #dims: p x p
        fiducial_data_gradient, fiducial_covariance_gradient = self._get_gradients_from_emulator(fiducial_parameter_vector, fiducial_data_vector, fiducial_covariance_matrix)

        inv_cov_data_diff = np.dot(fiducial_inverse_covariance, (self.data_vector - fiducial_data_vector)) #dims: p
        data_gradient_term = np.dot(fiducial_data_gradient, inv_cov_data_diff) #dims: (t x p) * p = t
        covariance_gradient_term = np.dot(inv_cov_data_diff, np.dot(fiducial_covariance_gradient, inv_cov_data_diff).transpose())
        #dims: p * [(t x p x p) * p]^T = p * (t x p)^T = t
        return data_gradient_term + covariance_gradient_term


class IMNNsCompression(Compression):
    """Sub-class to handle data compression by information maximising neural networks (IMNNs)"""
    def __init__(self, data_vector, redshift_vector, fixed_covariance_matrix, emulator_object):
        super(Compression, self).__init__(data_vector, redshift_vector, fixed_covariance_matrix, emulator_object)

    def compress_data_vector(self):
        """Compress data by IMNNs"""
        pass


def parameter_vector_to_compressed_data_vector(parameter_vector, redshift_vector=np.arange(4.2, 2.1, -0.2), fixed_covar=None):
    """Helper function for generating compressed data vector for given parameter vector"""
    if fixed_covar is None:

        fixed_covar = np.zeros(())

    compression_instance = Compression(None, redshift_vector, fixed_covar, )
