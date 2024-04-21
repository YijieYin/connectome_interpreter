# Happy Path: Tests that check the function works as expected with typical input.
# Boundary Conditions: Tests that challenge the edges of input domains and operational boundaries.
# Error Conditions: Tests that ensure the function handles errors gracefully, such as bad input values.
# Performance: If necessary, tests that evaluate the function’s performance to ensure it meets required standards.
import unittest
import numpy as np
from scipy.sparse import csr_matrix
# Adjust the import according to your package structure
from connectome_interpreter.compress_paths import compress_paths


class TestCompressPaths(unittest.TestCase):

    def test_simple_matrix_multiplication(self):
        # Test simple matrix multiplication without any thresholding
        matrix = csr_matrix(np.array([[0.1, 0.2], [0.3, 0.4]]))
        result = compress_paths(matrix, 2, threshold=0, output_threshold=0)
        expected_output = np.array([[0.07, 0.1], [0.15, 0.22]])

        # Check the shape and some values
        self.assertEqual(result[-1].shape, (2, 2))
        np.testing.assert_array_almost_equal(
            result[-1].toarray(), expected_output)

    def test_threshold_effect(self):
        # Test the effect of threshold on matrix elements
        matrix = csr_matrix(np.array([[0.5, 0.2], [0.1, 0.8]]))
        result = compress_paths(
            matrix, 1, threshold=0.25, output_threshold=0.01)
        # Expected to zero out all elements below 0.25
        expected_output = np.array([[0.5, 0], [0, 0.8]])

        self.assertEqual(result[0].shape, (2, 2))
        np.testing.assert_array_almost_equal(
            result[0].toarray(), expected_output)

    def test_output_threshold(self):
        # Test output threshold to check final sparsity
        matrix = csr_matrix(np.array([[0.01, 0.02], [0.03, 0.04]]))
        result = compress_paths(matrix, 1, threshold=0, output_threshold=0.03)
        # Expected to drop values below 0.03
        expected_output = np.array([[0, 0], [0.03, 0.04]])

        self.assertEqual(result[0].shape, (2, 2))
        np.testing.assert_array_almost_equal(
            result[0].toarray(), expected_output)

    def test_no_steps(self):
        # Test with zero steps
        matrix = csr_matrix(np.array([[1, 2], [3, 4]]))
        result = compress_paths(matrix, 0, threshold=0, output_threshold=0)

        # Should return an empty list
        self.assertEqual(len(result), 0)

    def test_matmul(self):
        # Test with a matrix multiplication
        matrix = csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float32))
        result = compress_paths(matrix, 2, threshold=0, output_threshold=10)
        expected_output = np.array([[0, 10], [15, 22]], dtype=np.float32)

        self.assertEqual(result[-1].shape, (2, 2))
        np.testing.assert_array_almost_equal(
            result[-1].toarray(), expected_output)


if __name__ == '__main__':
    unittest.main()
