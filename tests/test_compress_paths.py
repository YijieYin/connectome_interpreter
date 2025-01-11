# Happy Path: Tests that check the function works as expected with typical input.
# Boundary Conditions: Tests that challenge the edges of input domains and operational boundaries.
# Error Conditions: Tests that ensure the function handles errors gracefully, such as bad input values.
# Performance: If necessary, tests that evaluate the function’s performance to ensure it meets required standards.
import unittest

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import torch

from connectome_interpreter.compress_paths import compress_paths, compress_paths_not_chunked


class TestCompressPaths(unittest.TestCase):
    def setUp(self):
        """Set up test matrices for use in multiple tests."""
        # Simple 2x2 matrix
        self.simple_matrix = csr_matrix(np.array([[0.5, 0.5],
                                                 [0.3, 0.7]]))

        # Larger sparse matrix
        size = 100
        data = np.random.random(size * 3)  # Create some random data
        rows = np.random.randint(0, size, size * 3)
        cols = np.random.randint(0, size, size * 3)
        self.large_matrix = csr_matrix(
            (data, (rows, cols)), shape=(size, size))

        # Zero matrix
        self.zero_matrix = csr_matrix((2, 2))

    def test_basic_functionality(self):
        """Test basic functionality with a simple matrix."""
        result = compress_paths(self.simple_matrix, step_number=2)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], csc_matrix)
        self.assertEqual(result[0].shape, (2, 2))

    def test_single_step(self):
        """Test with step_number=1 returns input matrix in correct format."""
        result = compress_paths(self.simple_matrix, step_number=1)

        self.assertEqual(len(result), 1)
        self.assertTrue(np.allclose(
            result[0].toarray(), self.simple_matrix.toarray()))

    def test_invalid_step_number_zero(self):
        """Test that step_number=0 raises ValueError."""
        with self.assertRaises(ValueError):
            compress_paths(self.simple_matrix, step_number=0)

    def test_invalid_step_number_negative(self):
        """Test that negative step_number raises ValueError."""
        with self.assertRaises(ValueError):
            compress_paths(self.simple_matrix, step_number=-1)

        with self.assertRaises(ValueError):
            compress_paths(self.simple_matrix, step_number=-100)

    def test_threshold_during_multiplication(self):
        """Test that threshold parameter works during multiplication."""
        result = compress_paths(self.simple_matrix,
                                step_number=2,
                                threshold=0.4)

        # Check that no values below threshold exist in intermediate steps
        self.assertTrue(np.all(result[1].data >= 0.4))

    def test_output_threshold(self):
        """Test that output_threshold parameter works."""
        result = compress_paths(self.simple_matrix,
                                step_number=2,
                                output_threshold=0.3)

        # Check that no values below output_threshold exist in final result
        for matrix in result:
            self.assertTrue(np.all(matrix.data >= 0.3))

    def test_root_option(self):
        """Test that root option correctly takes nth root."""
        result_with_root = compress_paths(self.simple_matrix,
                                          step_number=2,
                                          root=True)
        result_without_root = compress_paths(self.simple_matrix,
                                             step_number=2,
                                             root=False)

        # Values in root version should be larger (as they're nth roots)
        self.assertTrue(
            np.all(result_with_root[1].data >= result_without_root[1].data))

    def test_chunk_size(self):
        """Test different chunk sizes produce same results."""
        result1 = compress_paths(self.large_matrix,
                                 step_number=2,
                                 chunkSize=10)
        result2 = compress_paths(self.large_matrix,
                                 step_number=2,
                                 chunkSize=20)

        # Results should be the same regardless of chunk size
        for m1, m2 in zip(result1, result2):
            self.assertTrue(np.allclose(m1.toarray(), m2.toarray()))

    def test_output_correctness(self):
        input_matrix = []
        if torch.cuda.is_available():
            # Very large sparse matrix
            size = 10000
            data = np.random.random(size * 3)  # Create some random data
            rows = np.random.randint(0, size, size * 3)
            cols = np.random.randint(0, size, size * 3)
            input_matrix = csr_matrix(
                (data, (rows, cols)), shape=(size, size))
        else:
            input_matrix = self.large_matrix
        """Test non chunked and chunked version of path compression produce same results."""
        result1 = compress_paths(input_matrix,
                                 step_number=2,
                                 chunkSize=20)
        result2 = compress_paths_not_chunked(input_matrix,
                                 step_number=2)

        # Results should be the same regardless of chunk size
        for m1, m2 in zip(result1, result2):
            self.assertTrue(np.allclose(m1.toarray(), m2.toarray()))

    def test_zero_matrix(self):
        """Test behavior with zero matrix input."""
        result = compress_paths(self.zero_matrix, step_number=2)

        # Result should be zero matrices
        for matrix in result:
            self.assertTrue(np.all(matrix.toarray() == 0))

    def test_memory_cleanup(self):
        """Test that GPU memory is properly cleaned up after execution."""
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            _ = compress_paths(self.large_matrix, step_number=2)
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()

            # Memory should be cleaned up
            self.assertEqual(initial_memory, final_memory)

    def test_large_matrix_performance(self):
        """Test performance with larger matrices."""
        result = compress_paths(self.large_matrix,
                                step_number=2,
                                chunkSize=50)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (100, 100))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gpu_usage(self):
        """Test that GPU is being used when available."""
        with unittest.mock.patch('torch.cuda.is_available', return_value=True):
            result = compress_paths(self.simple_matrix, step_number=2)
            self.assertEqual(len(result), 2)

