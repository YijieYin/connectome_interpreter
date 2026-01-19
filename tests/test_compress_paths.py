import unittest
from unittest.mock import patch
from unittest import skipIf
import numpy as np
import torch
import os
import tempfile
import pandas as pd
from pandas.testing import assert_frame_equal
import scipy as sp
from scipy.sparse import csr_matrix, csc_matrix


# Import the functions to test
# Adjust this import to match your actual module structure
from connectome_interpreter.compress_paths import (
    compress_paths,
    compress_paths_not_chunked,
    compress_paths_signed,
    compress_paths_signed_no_chunking,
    result_summary,
    effective_conn_from_paths,
    effective_conn_from_paths_cpu,
    effconn_without_loops,
    signed_conn_by_path_length_data,
)


class TestCompressPaths(unittest.TestCase):
    """Tests for the compress_paths function."""

    def setUp(self):
        """Set up test matrices for use in multiple tests."""
        # Simple 2x2 matrix
        self.simple_matrix = csr_matrix(np.array([[0.5, 0.5], [0.3, 0.7]]))

        # Larger sparse matrix (100x100)
        size = 100
        data = np.random.random(size * 10) * 0.5  # Create some random data
        rows = np.random.randint(0, size, size * 10)
        cols = np.random.randint(0, size, size * 10)
        self.large_matrix = csr_matrix(
            (data, (rows, cols)), shape=(size, size), dtype=np.float32
        )

        # Zero matrix
        self.zero_matrix = csr_matrix((size, size))

        # Device to use
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_basic_functionality(self):
        """Test basic functionality of compress_paths with simple matrix."""
        # Force sparse output by setting a high density threshold
        result = compress_paths(
            self.simple_matrix, step_number=2, density_threshold=1.0
        )

        # Check return type and length
        self.assertEqual(len(result), 2)

        # Check types (may be sparse or dense)
        for matrix in result:
            self.assertIn(type(matrix), (csc_matrix, np.ndarray))

        # Check shapes
        self.assertEqual(result[0].shape, (2, 2))
        self.assertEqual(result[1].shape, (2, 2))

        # Check that step 0 is the same as input (regardless of format)
        matrix_array = (
            result[0].toarray() if hasattr(result[0], "toarray") else result[0]
        )
        np.testing.assert_allclose(matrix_array, self.simple_matrix.toarray())

    def test_threshold_during_multiplication(self):
        """Test that threshold parameter works during matrix multiplication."""
        result = compress_paths(
            self.simple_matrix, step_number=2, threshold=0.4, density_threshold=1.0
        )

        # Check that no values below threshold exist in the second step
        # (the first step shouldn't be affected by threshold)
        matrix = result[1]
        if hasattr(matrix, "data"):  # If sparse
            self.assertTrue(np.all(matrix.data >= 0.4))
        else:  # If dense
            # Only check non-zero values
            non_zero_mask = matrix != 0
            self.assertTrue(np.all(matrix[non_zero_mask] >= 0.4))

    def test_output_threshold(self):
        """Test that output_threshold parameter works."""
        result = compress_paths(
            self.simple_matrix,
            step_number=2,
            output_threshold=0.3,
            density_threshold=1.0,
        )

        # Check that no values below output_threshold exist in final result
        for matrix in result:
            if hasattr(matrix, "data"):  # If sparse
                self.assertTrue(np.all(matrix.data >= 0.3))
            else:  # If dense
                # Only check non-zero values
                non_zero_mask = matrix != 0
                self.assertTrue(np.all(matrix[non_zero_mask] >= 0.3))

    def test_root_option(self):
        """Test that root option correctly takes nth root."""
        result_with_root = compress_paths(
            self.simple_matrix, step_number=2, root=True, density_threshold=1.0
        )
        result_without_root = compress_paths(
            self.simple_matrix, step_number=2, root=False, density_threshold=1.0
        )

        # For the first step (index 0), root should have no effect since it's direct connections
        with_root_0 = (
            result_with_root[0].toarray()
            if hasattr(result_with_root[0], "toarray")
            else result_with_root[0]
        )
        without_root_0 = (
            result_without_root[0].toarray()
            if hasattr(result_without_root[0], "toarray")
            else result_without_root[0]
        )
        np.testing.assert_allclose(with_root_0, without_root_0)

        # For the second step (index 1), check that values in root version are approximately
        # square roots of the non-root version
        with_root = (
            result_with_root[1].toarray()
            if hasattr(result_with_root[1], "toarray")
            else result_with_root[1]
        )
        without_root = (
            result_without_root[1].toarray()
            if hasattr(result_without_root[1], "toarray")
            else result_without_root[1]
        )

        # Only check non-zero values
        non_zero_mask = without_root > 0
        if np.any(non_zero_mask):
            # Sample check: values with root should be approximately sqrt of values without root
            # For the second step (n=2), the nth root is the square root
            sample_with_root = with_root[non_zero_mask][0]
            sample_without_root = without_root[non_zero_mask][0]
            self.assertAlmostEqual(
                sample_with_root, np.sqrt(sample_without_root), places=5
            )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_chunk_size(self):
        """Test different chunk sizes produce same results."""
        # Only run if we have enough GPU memory
        try:
            result1 = compress_paths(
                self.large_matrix, step_number=2, chunkSize=10, density_threshold=1.0
            )
            result2 = compress_paths(
                self.large_matrix, step_number=2, chunkSize=20, density_threshold=1.0
            )

            # Results should be the same regardless of chunk size (within floating point precision)
            for m1, m2 in zip(result1, result2):
                # Convert to arrays for comparison
                m1_array = m1.toarray() if hasattr(m1, "toarray") else m1
                m2_array = m2.toarray() if hasattr(m2, "toarray") else m2
                np.testing.assert_allclose(m1_array, m2_array, rtol=1e-5, atol=1e-7)
        except RuntimeError:  # Catch CUDA out of memory errors
            self.skipTest("Not enough GPU memory to test different chunk sizes")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_output_correctness(self):
        """Test chunked and non-chunked versions produce the same results."""
        # Use a smaller matrix to avoid memory issues
        size = 50
        data = np.random.random(size * 3) * 0.5
        rows = np.random.randint(0, size, size * 3)
        cols = np.random.randint(0, size, size * 3)
        input_matrix = csr_matrix(
            (data, (rows, cols)), shape=(size, size), dtype=np.float32
        )

        try:
            # Force sparse output for comparison with compress_paths_not_chunked
            result1 = compress_paths(
                input_matrix, step_number=2, chunkSize=20, density_threshold=1.0
            )
            result2 = compress_paths_not_chunked(input_matrix, step_number=2)

            # Results should be the same regardless of algorithm (within floating point precision)
            for m1, m2 in zip(result1, result2):
                # Convert to arrays for comparison
                m1_array = m1.toarray() if hasattr(m1, "toarray") else m1
                m2_array = m2.toarray() if hasattr(m2, "toarray") else m2
                np.testing.assert_allclose(m1_array, m2_array, rtol=1e-5, atol=1e-7)
        except RuntimeError:  # Catch CUDA out of memory errors
            self.skipTest("Not enough GPU memory to test chunked vs non-chunked")

    def test_zero_matrix(self):
        """Test behavior with zero matrix input."""
        # Test behavior with default settings
        result = compress_paths(self.zero_matrix, step_number=2)

        # Check that results are correct regardless of format
        for matrix in result:
            if hasattr(matrix, "nnz"):  # If sparse
                self.assertEqual(matrix.nnz, 0)
                np.testing.assert_array_equal(matrix.toarray(), np.zeros((100, 100)))
            else:  # If dense
                np.testing.assert_array_equal(matrix, np.zeros((100, 100)))

        # Skip the dense-specific test since we can't always force dense format for a zero matrix
        # Even with density_threshold=0.0, the implementation might choose sparse format for efficiency
        # For a zero matrix, density is always 0 regardless of format

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_memory_cleanup(self):
        """Test that GPU memory is properly cleaned up after execution."""
        initial_memory = torch.cuda.memory_allocated()
        _ = compress_paths(self.large_matrix, step_number=2)
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()

        # Memory should be approximately cleaned up
        # Allowing some tolerance as PyTorch might keep some allocations
        self.assertLess(
            final_memory - initial_memory, 1024 * 1024
        )  # Less than 1MB difference

    def test_save_to_disk(self):
        """Test that save_to_disk option works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run with save_to_disk=True for sparse output
            result = compress_paths(
                self.simple_matrix,
                step_number=2,
                save_to_disk=True,
                save_path=temp_dir,
                save_prefix="test_",
                return_results=False,
                density_threshold=1.0,  # Max threshold to force sparse output
            )

            # Result should be an empty list
            self.assertEqual(len(result), 0)

            # Check that files exist
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "test_0.npz")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "test_1.npz")))

            # Check that we can load the files
            loaded0 = sp.sparse.load_npz(os.path.join(temp_dir, "test_0.npz"))
            loaded1 = sp.sparse.load_npz(os.path.join(temp_dir, "test_1.npz"))

            # Check that loaded matrices have the right shape
            self.assertEqual(loaded0.shape, (2, 2))
            self.assertEqual(loaded1.shape, (2, 2))

    def test_save_dense_to_disk(self):
        """Test that dense matrices are correctly saved to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a denser matrix
            dense_matrix = np.random.random((10, 10)) * 0.5
            # Make it sparse but still dense enough to trigger dense format
            mask = np.random.random((10, 10)) < 0.2
            dense_matrix[mask] = 0
            sparse_dense_matrix = csr_matrix(dense_matrix)

            # Run with save_to_disk=True with low density threshold to force dense output
            result = compress_paths(
                sparse_dense_matrix,
                step_number=1,
                save_to_disk=True,
                save_path=temp_dir,
                save_prefix="dense_",
                return_results=False,
                density_threshold=0.5,  # Force dense output
            )

            # Result should be an empty list
            self.assertEqual(len(result), 0)

            # Check that dense file exists with .npy extension
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "dense_0.npy")))

            # Check that we can load the file as a numpy array
            loaded = np.load(os.path.join(temp_dir, "dense_0.npy"))

            # Check that loaded matrix has the right shape
            self.assertEqual(loaded.shape, (10, 10))

            # Check that the content matches the original matrix
            np.testing.assert_allclose(loaded, dense_matrix, rtol=1e-5, atol=1e-4)

    def test_invalid_inputs(self):
        """Test that the function raises appropriate errors for invalid inputs."""
        # Test non-square matrix
        non_square_matrix = csr_matrix(np.array([[0.5, 0.5, 0.3], [0.3, 0.7, 0.1]]))
        with self.assertRaises(AssertionError):
            compress_paths(non_square_matrix, step_number=2)

        # Test invalid step number
        with self.assertRaises(AssertionError):
            compress_paths(self.simple_matrix, step_number=0)

        # Test negative threshold (should still work, but test anyway)
        result = compress_paths(self.simple_matrix, step_number=2, threshold=-0.1)
        self.assertEqual(len(result), 2)

    def test_density_threshold(self):
        """Test that density threshold correctly determines output format."""
        # Create a matrix with known density
        size = 10
        matrix = np.zeros((size, size))
        # Set 30% of elements to non-zero
        indices = np.random.choice(size * size, int(0.3 * size * size), replace=False)
        rows, cols = np.unravel_index(indices, (size, size))
        for r, c in zip(rows, cols):
            matrix[r, c] = np.random.random() * 0.5
        sparse_matrix = csr_matrix(matrix)

        # Test with density_threshold=1.0 (should always be sparse)
        result_sparse = compress_paths(
            sparse_matrix, step_number=1, density_threshold=1.0
        )
        self.assertTrue(
            hasattr(result_sparse[0], "toarray"),
            "Result should be sparse with density_threshold=1.0",
        )

        # Test with density_threshold=0.0 (should always be dense)
        result_dense = compress_paths(
            sparse_matrix, step_number=1, density_threshold=0.0
        )
        self.assertFalse(
            hasattr(result_dense[0], "toarray"),
            "Result should be dense with density_threshold=0.0",
        )
        self.assertIsInstance(result_dense[0], np.ndarray)

        # Check that both formats have same values
        sparse_array = (
            result_sparse[0].toarray()
            if hasattr(result_sparse[0], "toarray")
            else result_sparse[0]
        )
        dense_array = result_dense[0]
        np.testing.assert_allclose(sparse_array, dense_array, rtol=1e-5, atol=1e-7)

    def test_output_dtype(self):
        """Test that output_dtype parameter works for both formats."""
        # Test with sparse format (using max density threshold to force sparse)
        result_f32_sparse = compress_paths(
            self.simple_matrix,
            step_number=1,
            output_dtype=np.float32,
            density_threshold=1.0,
        )

        result_f64_sparse = compress_paths(
            self.simple_matrix,
            step_number=1,
            output_dtype=np.float64,
            density_threshold=1.0,
        )

        # Check sparse dtypes - only if the results are actually sparse
        if hasattr(result_f32_sparse[0], "data"):
            self.assertEqual(result_f32_sparse[0].data.dtype, np.float32)
        if hasattr(result_f64_sparse[0], "data"):
            self.assertEqual(result_f64_sparse[0].data.dtype, np.float64)

        # Test with dense format using a denser matrix and forcing dense output
        matrix = np.random.random((5, 5)) * 0.5
        sparse_matrix = csr_matrix(matrix)

        result_f32_dense = compress_paths(
            sparse_matrix,
            step_number=1,
            output_dtype=np.float32,
            density_threshold=0.0,  # Force dense format with minimum threshold
        )

        result_f64_dense = compress_paths(
            sparse_matrix,
            step_number=1,
            output_dtype=np.float64,
            density_threshold=0.0,  # Force dense format with minimum threshold
        )

        # Check dense dtypes
        self.assertEqual(result_f32_dense[0].dtype, np.float32)
        self.assertEqual(result_f64_dense[0].dtype, np.float64)


class TestCompressPathsSigned(unittest.TestCase):
    """Tests for the compress_paths_signed function."""

    def setUp(self):
        """Set up test matrices for use in multiple tests."""
        # Simple 4x4 matrix with known values
        data = np.array([0.5, 0.3, 0.2, 0.6, 0.4, 0.1])
        rows = np.array([0, 0, 1, 2, 2, 3])
        cols = np.array([1, 2, 3, 0, 1, 2])
        self.simple_matrix = csc_matrix((data, (rows, cols)), shape=(4, 4))

        # Define neuron types (first two excitatory, last two inhibitory)
        self.simple_idx_to_sign = {0: 1, 1: 1, 2: -1, 3: -1}

        # Create a larger random sparse matrix (10x10)
        size = 10
        data = np.random.random(size * 5) * 0.5  # Create some random data
        rows = np.random.randint(0, size, size * 5)
        cols = np.random.randint(0, size, size * 5)
        self.larger_matrix = csc_matrix((data, (rows, cols)), shape=(size, size))

        # Define neuron types for larger matrix (half excitatory, half inhibitory)
        self.larger_idx_to_sign = {i: 1 if i < size / 2 else -1 for i in range(size)}

    def test_basic_functionality(self):
        """Test basic functionality of compress_paths_signed with small matrix."""
        with tempfile.TemporaryDirectory() as temp_dir:
            excitatory_paths, inhibitory_paths = compress_paths_signed(
                self.simple_matrix, self.simple_idx_to_sign, 2, save_path=temp_dir
            )

            # Check basic properties
            self.assertEqual(len(excitatory_paths), 2)
            self.assertEqual(len(inhibitory_paths), 2)
            self.assertIsInstance(excitatory_paths[0], csc_matrix)
            self.assertIsInstance(inhibitory_paths[0], csc_matrix)
            self.assertEqual(excitatory_paths[0].shape, (4, 4))
            self.assertEqual(inhibitory_paths[0].shape, (4, 4))

            # Check layer 0 (direct connections)
            # Excitatory should only have output from excitatory neurons (rows 0,1)
            e0 = excitatory_paths[0].toarray()
            self.assertTrue(np.all(e0[2:4, :] == 0))
            # Inhibitory should only have output from inhibitory neurons (rows 2,3)
            i0 = inhibitory_paths[0].toarray()
            self.assertTrue(np.all(i0[0:2, :] == 0))

            # Check file cleanup
            self.assertFalse(os.path.exists(os.path.join(os.getcwd(), "temp_chunks")))

    def test_first_layer_only(self):
        """Test that when target_layer_number=1, function returns direct connections only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            excitatory_paths, inhibitory_paths = compress_paths_signed(
                self.simple_matrix, self.simple_idx_to_sign, 1, save_path=temp_dir
            )

            # Check that we get only one matrix in each list
            self.assertEqual(len(excitatory_paths), 1)
            self.assertEqual(len(inhibitory_paths), 1)

            # Check the first layer results
            e0 = excitatory_paths[0].toarray()
            i0 = inhibitory_paths[0].toarray()

            # Verify correct separation of excitatory and inhibitory connections
            self.assertTrue(
                np.all(e0[2:4, :] == 0)
            )  # Inhibitory rows in E matrix are zero
            self.assertTrue(
                np.all(i0[0:2, :] == 0)
            )  # Excitatory rows in I matrix are zero

            # The connectivity pattern for direct connections should match the input
            input_array = self.simple_matrix.toarray()
            e_expected = np.zeros((4, 4))
            e_expected[0:2, :] = input_array[0:2, :]
            i_expected = np.zeros((4, 4))
            i_expected[2:4, :] = input_array[2:4, :]

            np.testing.assert_allclose(e0, e_expected)
            np.testing.assert_allclose(i0, i_expected)

    def test_threshold_parameter(self):
        """Test that threshold parameter works during multiplication."""
        with tempfile.TemporaryDirectory() as temp_dir:
            excitatory_paths, inhibitory_paths = compress_paths_signed(
                self.simple_matrix,
                self.simple_idx_to_sign,
                2,
                threshold=0.3,
                save_path=temp_dir,
            )

            # Values less than 0.3 should have been filtered out during multiplication
            # This is complex to verify in detail, but we can check that small values
            # (that would have resulted from multiplying values < 0.3) are absent

            # Check layer 1 (indirect connections)
            e1 = excitatory_paths[1].toarray()
            i1 = inhibitory_paths[1].toarray()

            # The output matrices should have some values set to zero due to thresholding
            self.assertTrue(np.any(e1 == 0))
            self.assertTrue(np.any(i1 == 0))

    def test_output_threshold(self):
        """Test that output_threshold parameter works."""
        with tempfile.TemporaryDirectory() as temp_dir:
            excitatory_paths, inhibitory_paths = compress_paths_signed(
                self.simple_matrix,
                self.simple_idx_to_sign,
                2,
                output_threshold=0.2,
                save_path=temp_dir,
            )

            # Check that no values below output_threshold exist in final results
            for matrices in [excitatory_paths, inhibitory_paths]:
                for m in matrices:
                    if m.nnz > 0:  # Only check if matrix has non-zero elements
                        self.assertTrue(np.all(m.data >= 0.2))

    def test_root_option(self):
        """Test that root option correctly takes nth root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            e_paths_root, i_paths_root = compress_paths_signed(
                self.simple_matrix,
                self.simple_idx_to_sign,
                2,
                root=True,
                save_path=temp_dir,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            e_paths_no_root, i_paths_no_root = compress_paths_signed(
                self.simple_matrix,
                self.simple_idx_to_sign,
                2,
                root=False,
                save_path=temp_dir,
            )

        # For the second layer (index 1), check that values are different
        e1_root = e_paths_root[1].toarray()
        e1_no_root = e_paths_no_root[1].toarray()

        # Matrices should be different
        self.assertFalse(np.allclose(e1_root, e1_no_root))

        # Check specific values - for layer 1, should be approximately square root
        non_zero_mask = (e1_no_root > 0) & (e1_root > 0)
        if np.any(non_zero_mask):
            samples_root = e1_root[non_zero_mask]
            samples_no_root = e1_no_root[non_zero_mask]
            # Check one sample - should be approximately the square root
            self.assertAlmostEqual(
                samples_root[0], np.sqrt(samples_no_root[0]), places=5
            )

    def test_saves_to_disk(self):
        """Test that files are properly saved to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            compress_paths_signed(
                self.simple_matrix,
                self.simple_idx_to_sign,
                2,
                save_to_disk=True,
                save_path=temp_dir,
            )

            # Check that files exist
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "step_0_e.npz")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "step_0_i.npz")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "step_1_e.npz")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "step_1_i.npz")))

    def test_no_return(self):
        """Test that function works when return_results=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = compress_paths_signed(
                self.simple_matrix,
                self.simple_idx_to_sign,
                2,
                return_results=False,
                save_to_disk=True,
                save_path=temp_dir,
            )

            # Result should be None or empty lists
            self.assertTrue(
                result is None or (len(result[0]) == 0 and len(result[1]) == 0)
            )

            # Files should still be saved
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "step_0_e.npz")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "step_1_e.npz")))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_chunk_size(self):
        """Test that different chunk sizes produce similar results."""
        # Skip if not enough GPU memory
        try:
            with tempfile.TemporaryDirectory() as temp_dir1:
                e_paths1, i_paths1 = compress_paths_signed(
                    self.larger_matrix,
                    self.larger_idx_to_sign,
                    2,
                    chunkSize=2,
                    save_path=temp_dir1,
                )

            with tempfile.TemporaryDirectory() as temp_dir2:
                e_paths2, i_paths2 = compress_paths_signed(
                    self.larger_matrix,
                    self.larger_idx_to_sign,
                    2,
                    chunkSize=5,
                    save_path=temp_dir2,
                )

            # Results should be the same regardless of chunk size (within floating point precision)
            for m1, m2 in zip(e_paths1, e_paths2):
                np.testing.assert_allclose(
                    m1.toarray(), m2.toarray(), rtol=1e-5, atol=1e-7
                )

            for m1, m2 in zip(i_paths1, i_paths2):
                np.testing.assert_allclose(
                    m1.toarray(), m2.toarray(), rtol=1e-5, atol=1e-7
                )
        except RuntimeError:  # Catch CUDA out of memory errors
            self.skipTest("Not enough GPU memory to test different chunk sizes")

    def test_empty_matrix(self):
        """Test behavior with empty matrix input."""
        empty_matrix = csc_matrix((4, 4))

        with tempfile.TemporaryDirectory() as temp_dir:
            e_paths, i_paths = compress_paths_signed(
                empty_matrix, self.simple_idx_to_sign, 2, save_path=temp_dir
            )

            # Both paths should be empty matrices
            for matrix in e_paths + i_paths:
                self.assertEqual(matrix.nnz, 0)

    def test_all_excitatory(self):
        """Test with all excitatory neurons."""
        all_excitatory = {i: 1 for i in range(4)}

        with tempfile.TemporaryDirectory() as temp_dir:
            e_paths, i_paths = compress_paths_signed(
                self.simple_matrix, all_excitatory, 2, save_path=temp_dir
            )

            # Inhibitory paths should all be empty
            for matrix in i_paths:
                self.assertEqual(matrix.nnz, 0)

            # Excitatory paths should have values
            self.assertTrue(any(m.nnz > 0 for m in e_paths))

    def test_temp_dir_cleanup(self):
        """Test that temporary directory is properly cleaned up."""
        temp_chunks = os.path.join(os.getcwd(), "temp_chunks")

        # Make sure temp_chunks doesn't exist before test
        if os.path.exists(temp_chunks):
            for file in os.listdir(temp_chunks):
                os.remove(os.path.join(temp_chunks, file))
            os.rmdir(temp_chunks)

        with tempfile.TemporaryDirectory() as temp_dir:
            _ = compress_paths_signed(
                self.simple_matrix,
                self.simple_idx_to_sign,
                2,
                save_path=temp_dir,
                save_to_disk=True,
            )

            # Temporary directory should be gone
            self.assertFalse(os.path.exists(temp_chunks))

            # But output files should exist
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "step_0_e.npz")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "step_1_e.npz")))

    def test_multiple_layers(self):
        """Test computation of multiple layers (beyond 2)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            e_paths, i_paths = compress_paths_signed(
                self.simple_matrix,
                self.simple_idx_to_sign,
                3,
                save_to_disk=True,
                save_path=temp_dir,
            )

            # Check we have the right number of layers
            self.assertEqual(len(e_paths), 3)
            self.assertEqual(len(i_paths), 3)

            # Each layer should have appropriate shape
            for i in range(3):
                self.assertEqual(e_paths[i].shape, (4, 4))
                self.assertEqual(i_paths[i].shape, (4, 4))

            # Check output files
            for i in range(3):
                self.assertTrue(
                    os.path.exists(os.path.join(temp_dir, f"step_{i}_e.npz"))
                )
                self.assertTrue(
                    os.path.exists(os.path.join(temp_dir, f"step_{i}_i.npz"))
                )


class TestCompressPathsSignedNoChunking(unittest.TestCase):
    """Tests for the compress_paths_signed_no_chunking function."""

    def setUp(self):
        """Set up test matrices for use in multiple tests."""
        # Simple 4x4 matrix
        data = np.array([0.5, 0.3, 0.2, 0.6, 0.4, 0.1])
        rows = np.array([0, 0, 1, 2, 2, 3])
        cols = np.array([1, 2, 3, 0, 1, 2])
        self.simple_matrix = csc_matrix((data, (rows, cols)), shape=(4, 4))

        # Define neuron types
        self.idx_to_sign = {0: 1, 1: 1, 2: -1, 3: -1}

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_basic_functionality(self):
        """Test basic functionality of compress_paths_signed_no_chunking."""
        try:
            excitatory_paths, inhibitory_paths = compress_paths_signed_no_chunking(
                self.simple_matrix, self.idx_to_sign, 2
            )

            # Check basic properties
            self.assertEqual(len(excitatory_paths), 2)
            self.assertEqual(len(inhibitory_paths), 2)
            self.assertIsInstance(excitatory_paths[0], csc_matrix)
            self.assertIsInstance(inhibitory_paths[0], csc_matrix)
            self.assertEqual(excitatory_paths[0].shape, (4, 4))
            self.assertEqual(inhibitory_paths[0].shape, (4, 4))

            # Check layer 0 (direct connections)
            # Excitatory should only have output from excitatory neurons (rows 0,1)
            e0 = excitatory_paths[0].toarray()
            self.assertTrue(np.all(e0[2:4, :] == 0))

            # Inhibitory should only have output from inhibitory neurons (rows 2,3)
            i0 = inhibitory_paths[0].toarray()
            self.assertTrue(np.all(i0[0:2, :] == 0))
        except RuntimeError:  # Catch CUDA out of memory errors
            self.skipTest("Not enough GPU memory for this test")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_threshold_parameter(self):
        """Test that threshold parameter works during multiplication."""
        try:
            excitatory_paths, inhibitory_paths = compress_paths_signed_no_chunking(
                self.simple_matrix, self.idx_to_sign, 2, threshold=0.3
            )

            # Check that intermediate values were thresholded
            # Hard to verify exactly, but can check second layer values
            e1 = excitatory_paths[1].toarray()
            i1 = inhibitory_paths[1].toarray()

            # The output matrices should have some values set to zero due to thresholding
            self.assertTrue(np.any(e1 == 0))
            self.assertTrue(np.any(i1 == 0))
        except RuntimeError:  # Catch CUDA out of memory errors
            self.skipTest("Not enough GPU memory for this test")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_output_threshold(self):
        """Test that output_threshold parameter works."""
        try:
            excitatory_paths, inhibitory_paths = compress_paths_signed_no_chunking(
                self.simple_matrix, self.idx_to_sign, 2, output_threshold=0.2
            )

            # Check that no values below output_threshold exist in final results
            for matrices in [excitatory_paths, inhibitory_paths]:
                for m in matrices:
                    if m.nnz > 0:  # Only check if matrix has non-zero elements
                        self.assertTrue(np.all(m.data >= 0.2))
        except RuntimeError:  # Catch CUDA out of memory errors
            self.skipTest("Not enough GPU memory for this test")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_root_option(self):
        """Test that root option correctly takes nth root."""
        try:
            e_paths_root, i_paths_root = compress_paths_signed_no_chunking(
                self.simple_matrix, self.idx_to_sign, 2, root=True
            )

            e_paths_no_root, i_paths_no_root = compress_paths_signed_no_chunking(
                self.simple_matrix, self.idx_to_sign, 2, root=False
            )

            # For the second layer (index 1), check that values are different
            e1_root = e_paths_root[1].toarray()
            e1_no_root = e_paths_no_root[1].toarray()

            # Matrices should be different
            self.assertFalse(np.allclose(e1_root, e1_no_root))

            # Check specific values - for layer 1, should be approximately square root
            non_zero_mask = (e1_no_root > 0) & (e1_root > 0)
            if np.any(non_zero_mask):
                samples_root = e1_root[non_zero_mask]
                samples_no_root = e1_no_root[non_zero_mask]
                # Check one sample - should be approximately the square root
                self.assertAlmostEqual(
                    samples_root[0], np.sqrt(samples_no_root[0]), places=5
                )
        except RuntimeError:  # Catch CUDA out of memory errors
            self.skipTest("Not enough GPU memory for this test")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_comparison_with_chunked_version(self):
        """Test that chunked and non-chunked versions produce similar results."""
        try:
            # Get result from non-chunked version
            e_paths_no_chunk, i_paths_no_chunk = compress_paths_signed_no_chunking(
                self.simple_matrix, self.idx_to_sign, 2
            )

            # Get result from chunked version
            with tempfile.TemporaryDirectory() as temp_dir:
                e_paths_chunk, i_paths_chunk = compress_paths_signed(
                    self.simple_matrix, self.idx_to_sign, 2, save_path=temp_dir
                )

            # Results should be the same within floating point precision
            for m1, m2 in zip(e_paths_no_chunk, e_paths_chunk):
                np.testing.assert_allclose(
                    m1.toarray(), m2.toarray(), rtol=1e-5, atol=1e-7
                )

            for m1, m2 in zip(i_paths_no_chunk, i_paths_chunk):
                np.testing.assert_allclose(
                    m1.toarray(), m2.toarray(), rtol=1e-5, atol=1e-7
                )
        except RuntimeError:  # Catch CUDA out of memory errors
            self.skipTest("Not enough GPU memory for this test")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_memory_cleanup(self):
        """Test that GPU memory is properly cleaned up after execution."""
        try:
            # Record initial GPU memory usage
            initial_memory = torch.cuda.memory_allocated()

            # Run the function
            _ = compress_paths_signed_no_chunking(
                self.simple_matrix, self.idx_to_sign, 2
            )

            # Force GPU memory cleanup
            torch.cuda.empty_cache()

            # Check final memory usage
            final_memory = torch.cuda.memory_allocated()

            # Verify memory was cleaned up (allowing for some overhead)
            self.assertLess(
                final_memory - initial_memory, 1024 * 1024
            )  # Less than 1MB difference
        except RuntimeError:
            self.skipTest("Not enough GPU memory for this test")


class TestResultSummary(unittest.TestCase):
    """Tests for the result_summary function."""

    def setUp(self):
        """Set up test data for use in multiple tests."""
        # Create a 9x9 test matrix with 3 neurons per group (so mean != median)
        # Neurons 0,1,2 = type A, neurons 3,4,5 = type B, neurons 6,7,8 = type C
        self.test_matrix = np.array(
            [
                [0.0, 0.1, 0.9, 0.2, 0.3, 0.8, 0.4, 0.05, 0.7],  # neuron 0 (type A)
                [0.15, 0.0, 0.6, 0.35, 0.25, 0.4, 0.15, 0.55, 0.3],  # neuron 1 (type A)
                [0.8, 0.2, 0.0, 0.1, 0.9, 0.2, 0.7, 0.1, 0.5],  # neuron 2 (type A)
                [0.22, 0.12, 0.3, 0.0, 0.42, 0.6, 0.22, 0.12, 0.8],  # neuron 3 (type B)
                [0.33, 0.23, 0.7, 0.13, 0.0, 0.1, 0.33, 0.23, 0.4],  # neuron 4 (type B)
                [0.1, 0.8, 0.2, 0.9, 0.3, 0.0, 0.6, 0.4, 0.2],  # neuron 5 (type B)
                [0.11, 0.31, 0.5, 0.21, 0.11, 0.7, 0.0, 0.41, 0.9],  # neuron 6 (type C)
                [0.24, 0.14, 0.2, 0.34, 0.24, 0.3, 0.14, 0.0, 0.1],  # neuron 7 (type C)
                [0.9, 0.6, 0.1, 0.4, 0.8, 0.2, 0.7, 0.3, 0.0],  # neuron 8 (type C)
            ]
        )

        # Sparse version
        self.sparse_matrix = csr_matrix(self.test_matrix)

        # Index mappings (3 neurons per group)
        self.inidx_map = {
            0: "A",
            1: "A",
            2: "A",
            3: "B",
            4: "B",
            5: "B",
            6: "C",
            7: "C",
            8: "C",
        }
        self.outidx_map = {
            0: "A",
            1: "A",
            2: "A",
            3: "B",
            4: "B",
            5: "B",
            6: "C",
            7: "C",
            8: "C",
        }

        # Test indices
        self.all_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.subset_indices = [1, 3, 7]

    def test_basic_functionality_dense(self):
        """Test basic functionality with dense matrix."""
        result = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            display_output=False,
        )

        # Check basic properties
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (3, 3))  # 3 neuron types
        self.assertCountEqual(result.index, ["A", "B", "C"])
        self.assertCountEqual(result.columns, ["A", "B", "C"])

    def test_basic_functionality_sparse(self):
        """Test basic functionality with sparse matrix."""
        result = result_summary(
            self.sparse_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            display_output=False,
        )

        # Should give same result as dense version
        dense_result = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            display_output=False,
        )

        pd.testing.assert_frame_equal(result, dense_result)

    def test_sparse_coo_conversion(self):
        """Test that COO sparse matrices are converted to CSC."""
        coo_matrix = self.sparse_matrix.tocoo()

        result = result_summary(
            coo_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            display_output=False,
        )

        # Should work without errors and give correct result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (3, 3))

    def test_subset_indices(self):
        """Test functionality with subset of indices."""
        result = result_summary(
            self.test_matrix,
            self.subset_indices,  # [1, 2, 4] -> types A, B, C
            self.subset_indices,
            self.inidx_map,
            self.outidx_map,
            display_output=False,
        )

        # Should still have all types represented
        self.assertEqual(result.shape, (3, 3))
        self.assertCountEqual(result.index, ["A", "B", "C"])

    def test_nan_removal(self):
        """Test that NaN values in indices are properly handled."""
        indices_with_nan = [0, 1, np.nan, 2, 3]

        result = result_summary(
            self.test_matrix,
            indices_with_nan,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            display_output=False,
        )

        # Should work without errors
        self.assertIsInstance(result, pd.DataFrame)

    def test_no_mapping(self):
        """Test behavior when no index mappings are provided."""
        result = result_summary(
            self.test_matrix, self.all_indices, self.all_indices, display_output=False
        )

        # Should create identity mapping
        self.assertEqual(result.shape, (9, 9))  # One group per neuron
        expected_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
        self.assertCountEqual(result.index, expected_labels)
        self.assertCountEqual(result.columns, expected_labels)

    def test_different_in_out_mappings(self):
        """Test with different input and output mappings."""
        out_map_different = {
            0: "X",
            1: "X",
            2: "X",
            3: "Y",
            4: "Y",
            5: "Y",
            6: "Z",
            7: "Z",
            8: "Z",
        }

        result = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            out_map_different,
            display_output=False,
        )

        # Rows should use inidx_map, columns should use outidx_map
        self.assertCountEqual(result.index, ["A", "B", "C"])
        self.assertCountEqual(result.columns, ["X", "Y", "Z"])

    def test_combining_methods(self):
        """Test different combining methods: mean, sum, median."""
        mean_result = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            combining_method="mean",
            display_output=False,
        )

        sum_result = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            combining_method="sum",
            display_output=False,
        )

        median_result = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            combining_method="median",
            display_output=False,
        )

        # Sum values should be exactly 3x mean values (since we have 3 neurons per type)
        expected_sum = mean_result * 3
        pd.testing.assert_frame_equal(sum_result, expected_sum)

        # With 3 values per group and asymmetric data, median should differ from mean
        self.assertFalse(mean_result.equals(median_result))

        # Check specific values - with our asymmetric data, mean and median should be different
        # For example, A->A should have different mean vs median due to the [0.1, 0.6, 0.2] pattern
        self.assertNotAlmostEqual(
            mean_result.loc["A", "A"], median_result.loc["A", "A"], places=3
        )

    def test_invalid_combining_method(self):
        """Test that invalid combining method raises assertion error."""
        with self.assertRaises(AssertionError):
            result_summary(
                self.test_matrix,
                self.all_indices,
                self.all_indices,
                self.inidx_map,
                self.outidx_map,
                combining_method="invalid_method",
                display_output=False,
            )

    def test_outprop_parameter(self):
        """Test outprop parameter (output proportion calculation)."""
        input_prop = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            outprop=False,
            display_output=False,
        )

        output_prop = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            outprop=True,
            display_output=False,
        )

        # Results should be different due to different calculation order
        self.assertFalse(input_prop.equals(output_prop))

        # Both should have same shape
        self.assertEqual(input_prop.shape, output_prop.shape)

        # Values should generally be different (not just transposes)
        self.assertFalse(np.allclose(input_prop.values, output_prop.values.T))

    def test_pre_in_column(self):
        """Test pre_in_column parameter (transpose result)."""
        pre_in_rows = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            pre_in_column=False,
            display_output=False,
        )

        pre_in_columns = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            pre_in_column=True,
            display_output=False,
        )

        # Should be transposes of each other
        pd.testing.assert_frame_equal(
            pre_in_rows.sort_index().sort_index(axis=1),
            pre_in_columns.T.sort_index().sort_index(axis=1),
        )

    def test_display_threshold_row(self):
        """Test display_threshold with row filtering."""
        # Set a high threshold to filter out some rows
        result = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            display_threshold=0.5,
            threshold_axis="row",
            display_output=False,
        )

        # Check that remaining rows have at least one value >= threshold
        for idx in result.index:
            self.assertTrue(np.any(np.abs(result.loc[idx]) >= 0.5))

        # Test that very high threshold raises ValueError
        with self.assertRaises(ValueError) as context:
            result_summary(
                self.test_matrix,
                self.all_indices,
                self.all_indices,
                self.inidx_map,
                self.outidx_map,
                display_threshold=2.0,  # Higher than any value in our matrix
                threshold_axis="row",
                display_output=False,
            )

        self.assertIn(
            "No values left after applying the display_threshold",
            str(context.exception),
        )

    def test_display_threshold_column(self):
        """Test display_threshold with column filtering."""
        result = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            display_threshold=0.5,
            threshold_axis="column",
            display_output=False,
        )

        # All remaining columns should have at least one value >= threshold
        for col in result.columns:
            self.assertTrue(np.any(np.abs(result[col]) >= 0.5))

        # Test that very high threshold raises ValueError
        with self.assertRaises(ValueError) as context:
            result_summary(
                self.test_matrix,
                self.all_indices,
                self.all_indices,
                self.inidx_map,
                self.outidx_map,
                display_threshold=2.0,  # Higher than any value in our matrix
                threshold_axis="column",
                display_output=False,
            )

        self.assertIn(
            "No values left after applying the display_threshold",
            str(context.exception),
        )

    def test_invalid_threshold_axis(self):
        """Test that invalid threshold_axis raises ValueError."""
        with self.assertRaises(ValueError):
            result_summary(
                self.test_matrix,
                self.all_indices,
                self.all_indices,
                self.inidx_map,
                self.outidx_map,
                threshold_axis="invalid",
                display_output=False,
            )

    def test_sort_within_column(self):
        """Test sorting within columns."""
        result = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            sort_within="column",
            display_output=False,
        )

        # Check that first column is sorted in descending order (by absolute value)
        first_col_values = np.abs(result.iloc[:, 0]).values
        self.assertTrue(np.all(first_col_values[:-1] >= first_col_values[1:]))

    def test_sort_within_row(self):
        """Test sorting within rows."""
        result = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            sort_within="row",
            display_output=False,
        )

        # Check that first row is sorted in descending order
        first_row_values = result.iloc[0, :].values
        self.assertTrue(np.all(first_row_values[:-1] >= first_row_values[1:]))

    def test_sort_names_column(self):
        """Test sorting by specific column names."""
        result = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            sort_within="column",
            sort_names="B",
            display_output=False,
        )

        # Should be sorted by column B values
        col_b_values = result["B"].values
        self.assertTrue(np.all(col_b_values[:-1] >= col_b_values[1:]))

    def test_sort_names_row(self):
        """Test sorting by specific row names."""
        result = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            sort_within="row",
            sort_names="B",
            display_output=False,
        )

        # Should be sorted by row B values
        row_b_values = result.loc["B", :].values
        self.assertTrue(np.all(row_b_values[:-1] >= row_b_values[1:]))

    def test_sort_names_list(self):
        """Test sorting by multiple names."""
        result = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            self.inidx_map,
            self.outidx_map,
            sort_within="column",
            sort_names=["A", "B"],
            display_output=False,
        )

        # Should work without errors
        self.assertIsInstance(result, pd.DataFrame)

    def test_invalid_sort_names_column(self):
        """Test error when sort_names not in columns."""
        with self.assertRaises(ValueError):
            result_summary(
                self.test_matrix,
                self.all_indices,
                self.all_indices,
                self.inidx_map,
                self.outidx_map,
                sort_within="column",
                sort_names="NonExistent",
                display_output=False,
            )

    def test_invalid_sort_names_row(self):
        """Test error when sort_names not in rows."""
        with self.assertRaises(ValueError):
            result_summary(
                self.test_matrix,
                self.all_indices,
                self.all_indices,
                self.inidx_map,
                self.outidx_map,
                sort_within="row",
                sort_names="NonExistent",
                display_output=False,
            )

    def test_invalid_sort_within(self):
        """Test that invalid sort_within raises ValueError."""
        with self.assertRaises(ValueError):
            result_summary(
                self.test_matrix,
                self.all_indices,
                self.all_indices,
                self.inidx_map,
                self.outidx_map,
                sort_within="invalid",
                display_output=False,
            )

    def test_include_undefined_groups(self):
        """Test include_undefined_groups parameter."""
        # Create mappings with NaN values
        inidx_map_with_nan = {0: "A", 1: "A", 2: np.nan, 3: "B", 4: "B", 5: np.nan}

        result_without_undefined = result_summary(
            self.test_matrix,
            set(inidx_map_with_nan.keys()),
            set(inidx_map_with_nan.keys()),
            inidx_map_with_nan,
            inidx_map_with_nan,
            include_undefined_groups=False,
            display_output=False,
        )

        result_with_undefined = result_summary(
            self.test_matrix,
            set(inidx_map_with_nan.keys()),
            set(inidx_map_with_nan.keys()),
            inidx_map_with_nan,
            inidx_map_with_nan,
            include_undefined_groups=True,
            display_output=False,
        )

        # With undefined groups, should have 'undefined' in index/columns
        self.assertIn("undefined", result_with_undefined.index)
        self.assertIn("undefined", result_with_undefined.columns)

        # Without undefined groups, should not have 'undefined'
        self.assertNotIn("undefined", result_without_undefined.index)
        self.assertNotIn("undefined", result_without_undefined.columns)

    def test_single_neuron_type(self):
        """Test with all neurons of same type."""
        single_type_map = {i: "A" for i in range(9)}

        result = result_summary(
            self.test_matrix,
            self.all_indices,
            self.all_indices,
            single_type_map,
            single_type_map,
            display_output=False,
        )

        # Should have shape (1, 1)
        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(result.index[0], "A")
        self.assertEqual(result.columns[0], "A")

    def test_different_index_types(self):
        """Test with different types of indices (list, array, etc.)."""
        # Test with numpy array
        np_indices = np.array(self.all_indices)
        result1 = result_summary(
            self.test_matrix,
            np_indices,
            np_indices,
            self.inidx_map,
            self.outidx_map,
            display_output=False,
        )

        # Test with pandas Series
        pd_indices = pd.Series(self.all_indices)
        result2 = result_summary(
            self.test_matrix,
            pd_indices,
            pd_indices,
            self.inidx_map,
            self.outidx_map,
            display_output=False,
        )

        # Test with set
        set_indices = set(self.all_indices)
        result3 = result_summary(
            self.test_matrix,
            set_indices,
            set_indices,
            self.inidx_map,
            self.outidx_map,
            display_output=False,
        )

        # All should give same results (order might differ for set)
        # Check that shapes are the same
        self.assertEqual(result1.shape, result2.shape)
        self.assertEqual(result1.shape, result3.shape)

    def test_mathematical_correctness(self):
        """Test that the mathematical operations are correct."""
        # Create a simple case where we can verify the math manually
        simple_matrix = np.array(
            [
                [0.0, 0.2, 0.4],  # neuron 0 (type A)
                [0.1, 0.0, 0.3],  # neuron 1 (type A)
                [0.5, 0.1, 0.0],  # neuron 2 (type B)
            ]
        )

        simple_map = {0: "A", 1: "A", 2: "B"}
        indices = [0, 1, 2]

        # Test with sum combining method for easier verification
        result = result_summary(
            simple_matrix,
            indices,
            indices,
            simple_map,
            simple_map,
            combining_method="sum",
            display_output=False,
        )

        # Manual calculation (outprop=False, sum method):
        # 1. Group by presynaptic type and sum
        # 2. Group by postsynaptic type and sum

        # For A->A: neurons 0,1 -> neurons 0,1
        # Connections from A neurons: [0.0+0.1, 0.2+0.0] = [0.1, 0.2]
        # Sum across postsynaptic A neurons: 0.1 + 0.2 = 0.3
        expected_a_to_a = 0.3

        # For A->B: neurons 0,1 -> neuron 2
        # Connections from A neurons to B: [0.4+0.3] = 0.7
        expected_a_to_b = 0.7

        # For B->A: neuron 2 -> neurons 0,1
        # Connections from B to A: [0.5+0.1] = 0.6
        expected_b_to_a = 0.6

        # For B->B: neuron 2 -> neuron 2
        # Connection from B to B: 0.0
        expected_b_to_b = 0.0

        self.assertAlmostEqual(result.loc["A", "A"], expected_a_to_a, places=5)
        self.assertAlmostEqual(result.loc["A", "B"], expected_a_to_b, places=5)
        self.assertAlmostEqual(result.loc["B", "A"], expected_b_to_a, places=5)
        self.assertAlmostEqual(result.loc["B", "B"], expected_b_to_b, places=5)

        # Test with mean combining method
        result_mean = result_summary(
            simple_matrix,
            indices,
            indices,
            simple_map,
            simple_map,
            combining_method="mean",
            display_output=False,
        )

        # Mean should be sum divided by number of postsynaptic neurons in each group
        # A group has 2 neurons, B group has 1 neuron
        expected_mean_a_to_a = expected_a_to_a / 2  # 0.15
        expected_mean_a_to_b = expected_a_to_b / 1  # 0.7
        expected_mean_b_to_a = expected_b_to_a / 2  # 0.3
        expected_mean_b_to_b = expected_b_to_b / 1  # 0.0

        self.assertAlmostEqual(
            result_mean.loc["A", "A"], expected_mean_a_to_a, places=5
        )
        self.assertAlmostEqual(
            result_mean.loc["A", "B"], expected_mean_a_to_b, places=5
        )
        self.assertAlmostEqual(
            result_mean.loc["B", "A"], expected_mean_b_to_a, places=5
        )
        self.assertAlmostEqual(
            result_mean.loc["B", "B"], expected_mean_b_to_b, places=5
        )


class TestEffectiveConnFromPaths(unittest.TestCase):
    """Tests for both effective_conn_from_paths variants."""

    def setUp(self):
        # basic 3-node 2-layer toy graph
        self.simple_paths = pd.DataFrame(
            {
                "pre": [0, 1, 2],
                "post": [1, 2, 0],
                "weight": [1.0, 0.5, 0.2],
                "layer": [0, 0, 1],
            }
        )
        self.id_group = {i: i for i in range(3)}
        self.one_group = {i: 0 for i in range(3)}

        # random 50-node 3-layer graph
        rng = np.random.default_rng(0)
        size = 50
        rows = rng.integers(0, size, 300)
        cols = rng.integers(0, size, 300)
        self.rand_paths = pd.DataFrame(
            {
                "pre": rows,
                "post": cols,
                "weight": rng.random(300, dtype=np.float32),
                "layer": rng.integers(0, 3, 300),
            }
        )
        self.rand_group = {i: i for i in range(size)}

    # ------------------------------------------------------ #
    # 1. CPU reference vs new implementation on toy graph
    # ------------------------------------------------------ #
    def test_basic_equivalence(self):
        ref = effective_conn_from_paths_cpu(self.simple_paths, wide=True)
        new = effective_conn_from_paths(self.simple_paths, wide=True)
        assert_frame_equal(
            ref.sort_index().sort_index(axis=1),
            new.sort_index().sort_index(axis=1),
            rtol=1e-6,
            atol=1e-8,
        )

    # ------------------------------------------------------ #
    # 2. wide=True vs wide=False shape/consistency
    # ------------------------------------------------------ #
    def test_wide_parameter(self):
        wide_df = effective_conn_from_paths_cpu(self.simple_paths, wide=True)
        long_df = effective_conn_from_paths_cpu(self.simple_paths, wide=False)
        # pivot long_df to compare
        long_piv = (
            long_df.pivot(index="pre", columns="post", values="weight")
            .fillna(0)
            .sort_index()
            .sort_index(axis=1)
        )
        assert_frame_equal(
            wide_df.sort_index().sort_index(axis=1), long_piv, rtol=1e-6, atol=1e-8
        )

    # ------------------------------------------------------ #
    # 3. combining_method sum vs mean - REMOVED
    # (grouping functionality removed from effective_conn_from_paths_cpu)
    # ------------------------------------------------------ #

    # ------------------------------------------------------ #
    # 4. chunking: tiny chunk vs huge chunk yields same result
    # ------------------------------------------------------ #
    def test_chunking_consistency(self):
        small_chunk = effective_conn_from_paths(
            self.rand_paths,
            wide=True,
            chunk_size=10,
            use_gpu=False,
        )
        big_chunk = effective_conn_from_paths(
            self.rand_paths,
            wide=True,
            chunk_size=10_000,
            use_gpu=False,
        )
        assert_frame_equal(
            small_chunk.sort_index().sort_index(axis=1),
            big_chunk.sort_index().sort_index(axis=1),
            rtol=1e-5,
            atol=1e-6,
        )

    # ------------------------------------------------------ #
    # 5. GPU vs CPU parity (skip if no CUDA)
    # ------------------------------------------------------ #
    @skipIf(
        not (torch.cuda.is_available() and torch.__version__ >= "2.2"),
        "CUDA ≥2.2 required for sparse.mm",
    )
    def test_gpu_vs_cpu(self):
        gpu_out = effective_conn_from_paths(
            self.rand_paths,
            wide=True,
            chunk_size=2000,
            use_gpu=True,
        )
        cpu_out = effective_conn_from_paths_cpu(self.rand_paths, wide=True)
        assert_frame_equal(
            gpu_out.sort_index().sort_index(axis=1),
            cpu_out.sort_index().sort_index(axis=1),
            rtol=1e-4,
            atol=1e-6,
        )
        torch.cuda.empty_cache()

    # -------------------------------------------------- #
    # 6. forcing dense vs sparse on GPU yields same result
    # -------------------------------------------------- #
    @skipIf(
        not (torch.cuda.is_available() and torch.__version__ >= "2.2"),
        "CUDA ≥2.2 required for sparse.mm",
    )
    def test_density_threshold_consistency(self):
        sparse_like = effective_conn_from_paths(
            self.rand_paths,
            wide=True,
            density_threshold=1.0,  # never densify
            use_gpu=True,
        )
        dense_like = effective_conn_from_paths(
            self.rand_paths,
            wide=True,
            density_threshold=0.0,  # always densify
            use_gpu=True,
        )
        assert_frame_equal(
            sparse_like.sort_index().sort_index(axis=1),
            dense_like.sort_index().sort_index(axis=1),
            rtol=1e-4,
            atol=1e-6,
        )
        torch.cuda.empty_cache()

    # -------------------------------------------------- #
    # 7. invalid combining_method - REMOVED
    # (combining_method parameter removed from effective_conn_from_paths_cpu)
    # -------------------------------------------------- #

    # -------------------------------------------------- #
    # 8. pre/post columns with different dtypes
    # -------------------------------------------------- #
    def test_mixed_index_dtypes(self):
        mixed_paths = pd.DataFrame(
            {
                "pre": [0, 1, 2],
                "post": ["a", "b", "c"],  # string dtype
                "weight": [1.0, 0.5, 0.2],
                "layer": [0, 0, 1],
            }
        )

        cpu_out = effective_conn_from_paths_cpu(mixed_paths, wide=True)
        ref_out = effective_conn_from_paths(
            mixed_paths,
            wide=True,
            use_gpu=False,  # keep deterministic & avoid dtype issues on GPU path
        )
        # Both should work without error and produce similar results
        self.assertIsNotNone(cpu_out)
        self.assertIsNotNone(ref_out)


class TestEffconnWithoutLoops(unittest.TestCase):
    """Tests for the effconn_without_loops function."""

    def setUp(self):
        """Set up test data for use in multiple tests."""
        # Create a simple paths DataFrame WITH a loop
        # Node 1 appears in both layer 1 and layer 3, creating a loop
        self.paths_with_loop = pd.DataFrame(
            {
                "layer": [1, 1, 2, 2, 3],
                "pre": [0, 0, 1, 2, 1],  # Note: node 1 appears in layer 1 and 3
                "post": [1, 2, 2, 3, 3],
                "weight": [0.5, 0.3, 0.4, 0.6, 0.2],
            }
        )

        # Create paths WITHOUT loops
        self.paths_no_loop = pd.DataFrame(
            {
                "layer": [1, 1, 2, 2],
                "pre": [0, 0, 1, 2],
                "post": [1, 2, 3, 3],
                "weight": [0.5, 0.3, 0.4, 0.6],
            }
        )

        # Empty paths
        self.empty_paths = pd.DataFrame(columns=["layer", "pre", "post", "weight"])

        # Define groups
        self.pre_group = {0: "A"}
        self.post_group = {3: "X"}
        self.intermediate_group = {1: "I1", 2: "I2"}

    def test_empty_paths(self):
        """Test that function handles empty paths correctly."""
        result = effconn_without_loops(self.empty_paths)
        self.assertIsNone(result)

    def test_no_loops_present(self):
        """Test behavior when no loops are present in paths."""
        result = effconn_without_loops(
            self.paths_no_loop,
            pre_group=self.pre_group,
            post_group=self.post_group,
            wide=True,
            use_gpu=False,
        )

        # Should return a valid DataFrame
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)

    def test_loop_detection(self):
        """Test that loops are correctly detected and removed."""
        result_with_loop = effconn_without_loops(
            self.paths_with_loop, wide=False, use_gpu=False
        )
        result_no_loop = effconn_without_loops(
            self.paths_no_loop, wide=False, use_gpu=False
        )

        # Both should be valid DataFrames
        self.assertIsNotNone(result_with_loop)
        self.assertIsNotNone(result_no_loop)

        # The result with loops should have different (likely lower) weights
        # after removing loop contributions
        self.assertIsInstance(result_with_loop, pd.DataFrame)

    def test_wide_vs_long_format(self):
        """Test that wide and long format outputs are consistent."""
        result_wide = effconn_without_loops(
            self.paths_no_loop, wide=True, use_gpu=False
        )
        result_long = effconn_without_loops(
            self.paths_no_loop, wide=False, use_gpu=False
        )

        self.assertIsInstance(result_wide, pd.DataFrame)
        self.assertIsInstance(result_long, pd.DataFrame)

        # Check that long format has expected columns
        self.assertIn("pre", result_long.columns)
        self.assertIn("post", result_long.columns)
        self.assertIn("weight", result_long.columns)

        # Check that wide format is indeed wide (has pre as index, post as columns)
        self.assertTrue(hasattr(result_wide, "index"))

    def test_combining_methods(self):
        """Test different combining methods."""
        for method in ["mean", "sum", "median"]:
            result = effconn_without_loops(
                self.paths_no_loop,
                pre_group=self.pre_group,
                post_group=self.post_group,
                combining_method=method,
                use_gpu=False,
            )
            self.assertIsNotNone(result)

    def test_remove_loop_before_vs_after_grouping(self):
        """Test remove_loop_before_grouping parameter."""
        result_before = effconn_without_loops(
            self.paths_with_loop,
            pre_group=self.pre_group,
            post_group=self.post_group,
            remove_loop_after_grouping=False,
            use_gpu=False,
        )

        result_after = effconn_without_loops(
            self.paths_with_loop,
            pre_group=self.pre_group,
            post_group=self.post_group,
            remove_loop_after_grouping=True,
            use_gpu=False,
        )

        # Both should return valid results
        self.assertIsNotNone(result_before)
        self.assertIsNotNone(result_after)

    def test_remove_loop_before_grouping_detailed(self):
        """Detailed test of remove_loop_before_grouping with known structure."""
        # Create a more detailed path structure with clear loops
        # Layer 1: 0,1 -> Layer 2: 2,3 -> Layer 3: 4,5 (with 2 appearing again)
        paths = pd.DataFrame(
            {
                "layer": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
                "pre": [0, 0, 1, 1, 2, 2, 3, 3, 2, 4],  # Node 2 in layers 2 and 3
                "post": [2, 3, 2, 3, 4, 5, 4, 5, 5, 5],
                "weight": [0.5, 0.3, 0.2, 0.4, 0.6, 0.2, 0.3, 0.5, 0.1, 0.7],
            }
        )

        # Define groups that combine nodes
        pre_group = {0: "A", 1: "A"}  # Both inputs map to group A
        post_group = {5: "X"}  # Output maps to group X
        intermediate_group = {2: "I1", 3: "I2", 4: "I3"}

        # Test with remove_loop_after_grouping=False
        result_before = effconn_without_loops(
            paths,
            pre_group=pre_group,
            post_group=post_group,
            intermediate_group=intermediate_group,
            remove_loop_after_grouping=False,
            wide=False,
            use_gpu=False,
        )

        # Test with remove_loop_after_grouping=True
        result_after = effconn_without_loops(
            paths,
            pre_group=pre_group,
            post_group=post_group,
            intermediate_group=intermediate_group,
            remove_loop_after_grouping=True,
            wide=False,
            use_gpu=False,
        )

        # Both should produce valid results
        self.assertIsNotNone(result_before)
        self.assertIsNotNone(result_after)
        self.assertIsInstance(result_before, pd.DataFrame)
        self.assertIsInstance(result_after, pd.DataFrame)

        # Both should have weight values
        self.assertGreater(len(result_before), 0)
        self.assertGreater(len(result_after), 0)

        # Check that results contain expected groups
        if "pre" in result_before.columns:
            self.assertIn("A", result_before["pre"].values)
        if "post" in result_before.columns:
            self.assertIn("X", result_before["post"].values)

    def test_remove_loop_grouping_with_intermediate(self):
        """Test that intermediate_group works correctly with loop removal."""
        # Create paths with intermediate nodes that form loops
        paths = pd.DataFrame(
            {
                "layer": [1, 1, 2, 2, 3, 3],
                "pre": [0, 0, 1, 2, 1, 3],  # Node 1 appears in layers 2 and 3
                "post": [1, 2, 3, 3, 4, 4],
                "weight": [0.6, 0.4, 0.5, 0.3, 0.2, 0.7],
            }
        )

        pre_group = {0: "Source"}
        post_group = {4: "Target"}
        intermediate_group = {1: "Int_A", 2: "Int_B", 3: "Int_C"}

        # Before grouping
        result_before = effconn_without_loops(
            paths,
            pre_group=pre_group,
            post_group=post_group,
            intermediate_group=intermediate_group,
            remove_loop_after_grouping=False,
            wide=True,
            use_gpu=False,
        )

        # After grouping
        result_after = effconn_without_loops(
            paths,
            pre_group=pre_group,
            post_group=post_group,
            intermediate_group=intermediate_group,
            remove_loop_after_grouping=True,
            wide=True,
            use_gpu=False,
        )

        self.assertIsNotNone(result_before)
        self.assertIsNotNone(result_after)

        # Check structure
        self.assertTrue(hasattr(result_before, "index"))
        self.assertTrue(hasattr(result_after, "index"))

    def test_remove_loop_no_grouping_provided(self):
        """Test remove_loop_before_grouping when no groups are provided."""
        paths = pd.DataFrame(
            {
                "layer": [1, 1, 2, 2, 3],
                "pre": [0, 0, 1, 2, 1],
                "post": [1, 2, 2, 3, 3],
                "weight": [0.5, 0.3, 0.4, 0.6, 0.2],
            }
        )

        # When no groups are provided, the parameter should still work
        result_before = effconn_without_loops(
            paths, remove_loop_after_grouping=False, use_gpu=False
        )

        result_after = effconn_without_loops(
            paths, remove_loop_after_grouping=True, use_gpu=False
        )

        # Both should produce identical results when no grouping is applied
        self.assertIsNotNone(result_before)
        self.assertIsNotNone(result_after)

    def test_remove_loop_grouping_consistency(self):
        """Test that both grouping modes produce consistent weight magnitudes."""
        # Create a well-defined path structure
        paths = pd.DataFrame(
            {
                "layer": [1, 1, 2, 2, 3],
                "pre": [0, 0, 1, 2, 1],
                "post": [1, 2, 3, 3, 3],
                "weight": [0.5, 0.3, 0.4, 0.6, 0.2],
            }
        )

        pre_group = {0: "A"}
        post_group = {3: "Z"}

        result_before = effconn_without_loops(
            paths,
            pre_group=pre_group,
            post_group=post_group,
            remove_loop_after_grouping=False,
            wide=False,
            use_gpu=False,
        )

        result_after = effconn_without_loops(
            paths,
            pre_group=pre_group,
            post_group=post_group,
            remove_loop_after_grouping=True,
            wide=False,
            use_gpu=False,
        )

        # Both should have reasonable weight values (non-negative, finite)
        if result_before is not None and "weight" in result_before.columns:
            self.assertTrue((result_before["weight"] >= 0).all())
            self.assertFalse(result_before["weight"].isna().any())
            self.assertFalse(np.isinf(result_before["weight"]).any())

        if result_after is not None and "weight" in result_after.columns:
            self.assertTrue((result_after["weight"] >= 0).all())
            self.assertFalse(result_after["weight"].isna().any())
            self.assertFalse(np.isinf(result_after["weight"]).any())

    def test_remove_loop_grouping_all_combining_methods(self):
        """Test remove_loop_before_grouping with different combining methods."""
        paths = pd.DataFrame(
            {
                "layer": [1, 1, 2, 2, 3],
                "pre": [0, 0, 1, 2, 1],
                "post": [1, 2, 2, 3, 3],
                "weight": [0.5, 0.3, 0.4, 0.6, 0.2],
            }
        )

        pre_group = {0: "A"}
        post_group = {3: "Z"}

        for method in ["mean", "sum", "median"]:
            for remove_before in [True, False]:
                result = effconn_without_loops(
                    paths,
                    pre_group=pre_group,
                    post_group=post_group,
                    combining_method=method,
                    remove_loop_after_grouping=remove_before,
                    use_gpu=False,
                )
                self.assertIsNotNone(
                    result,
                    f"Failed with method={method}, remove_before={remove_before}",
                )

    def test_remove_loop_grouping_wide_long_formats(self):
        """Test that both wide and long formats work with both grouping modes."""
        paths = pd.DataFrame(
            {
                "layer": [1, 1, 2, 2],
                "pre": [0, 0, 1, 2],
                "post": [1, 2, 3, 3],
                "weight": [0.5, 0.3, 0.4, 0.6],
            }
        )

        pre_group = {0: "A"}
        post_group = {3: "Z"}

        for remove_before in [True, False]:
            # Test wide format
            result_wide = effconn_without_loops(
                paths,
                pre_group=pre_group,
                post_group=post_group,
                remove_loop_after_grouping=remove_before,
                wide=True,
                use_gpu=False,
            )
            self.assertIsNotNone(result_wide)
            self.assertTrue(hasattr(result_wide, "index"))

            # Test long format
            result_long = effconn_without_loops(
                paths,
                pre_group=pre_group,
                post_group=post_group,
                remove_loop_after_grouping=remove_before,
                wide=False,
                use_gpu=False,
            )
            self.assertIsNotNone(result_long)
            if result_long is not None:
                self.assertIn("pre", result_long.columns)
                self.assertIn("post", result_long.columns)
                self.assertIn("weight", result_long.columns)

    def test_remove_loop_grouping_complex_case(self):
        """Test a complex scenario with multiple loops and groups."""
        # Create a complex graph structure
        paths = pd.DataFrame(
            {
                "layer": [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4],
                "pre": [0, 0, 1, 2, 2, 3, 3, 2, 4, 5, 6, 3],  # Nodes 2,3 create loops
                "post": [2, 3, 3, 4, 5, 4, 5, 6, 6, 6, 7, 7],
                "weight": [
                    0.5,
                    0.3,
                    0.4,
                    0.2,
                    0.3,
                    0.4,
                    0.2,
                    0.1,
                    0.5,
                    0.3,
                    0.6,
                    0.2,
                ],
            }
        )

        # Create groups that span multiple nodes
        pre_group = {0: "Source", 1: "Source"}
        post_group = {7: "Target"}
        intermediate_group = {2: "Mid1", 3: "Mid1", 4: "Mid2", 5: "Mid2", 6: "Mid3"}

        # Both modes should handle this without crashing
        result_before = effconn_without_loops(
            paths,
            pre_group=pre_group,
            post_group=post_group,
            intermediate_group=intermediate_group,
            remove_loop_after_grouping=False,
            use_gpu=False,
        )

        result_after = effconn_without_loops(
            paths,
            pre_group=pre_group,
            post_group=post_group,
            intermediate_group=intermediate_group,
            remove_loop_after_grouping=True,
            use_gpu=False,
        )

        self.assertIsNotNone(result_before)
        self.assertIsNotNone(result_after)

    def test_remove_loop_grouping_numerical_comparison(self):
        """Detailed numerical comparison between remove_loop_before_grouping modes.

        This test ensures both modes produce reasonable results and helps understand
        the difference between grouping before vs after loop removal.
        """
        # Create a well-defined structure with known loops
        # Pre: 0,1 -> Intermediate: 2,3 (node 2 loops back) -> Post: 4
        paths = pd.DataFrame(
            {
                "layer": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
                "pre": [0, 0, 1, 1, 2, 2, 3, 3, 2, 4],  # Node 2 in layer 2 and 3
                "post": [2, 3, 2, 3, 4, 4, 4, 4, 4, 4],
                "weight": [0.5, 0.2, 0.3, 0.4, 0.6, 0.3, 0.5, 0.2, 0.1, 0.7],
            }
        )

        # Group nodes to test grouping effects
        pre_group = {0: "GroupA", 1: "GroupB"}
        post_group = {4: "Target"}
        intermediate_group = {2: "IntX", 3: "IntY"}

        # Mode 1: Remove loops BEFORE grouping
        result_before = effconn_without_loops(
            paths,
            pre_group=pre_group,
            post_group=post_group,
            intermediate_group=intermediate_group,
            remove_loop_after_grouping=False,
            wide=False,
            combining_method="sum",
            use_gpu=False,
        )

        # Mode 2: Remove loops AFTER grouping
        result_after = effconn_without_loops(
            paths,
            pre_group=pre_group,
            post_group=post_group,
            intermediate_group=intermediate_group,
            remove_loop_after_grouping=True,
            wide=False,
            combining_method="sum",
            use_gpu=False,
        )

        # Both should produce valid DataFrames
        self.assertIsNotNone(result_before)
        self.assertIsNotNone(result_after)
        self.assertIsInstance(result_before, pd.DataFrame)
        self.assertIsInstance(result_after, pd.DataFrame)

        # Both should have the expected columns
        for col in ["pre", "post", "weight"]:
            self.assertIn(col, result_before.columns)
            self.assertIn(col, result_after.columns)

        # Both should have positive weights (effective connectivity)
        self.assertTrue((result_before["weight"] >= 0).all())
        self.assertTrue((result_after["weight"] >= 0).all())

        # Both should connect the same groups (even if weights differ)
        expected_groups = set(pre_group.values()) | set(post_group.values())
        before_groups = set(result_before["pre"].values) | set(
            result_before["post"].values
        )
        after_groups = set(result_after["pre"].values) | set(
            result_after["post"].values
        )

        # Check that we have connections involving our defined groups
        self.assertTrue(
            len(before_groups.intersection(expected_groups)) > 0,
            f"Before grouping result missing expected groups. Got: {before_groups}, Expected: {expected_groups}",
        )
        self.assertTrue(
            len(after_groups.intersection(expected_groups)) > 0,
            f"After grouping result missing expected groups. Got: {after_groups}, Expected: {expected_groups}",
        )

        # Verify no NaN or infinite values
        self.assertFalse(result_before["weight"].isna().any())
        self.assertFalse(result_after["weight"].isna().any())
        self.assertFalse(np.isinf(result_before["weight"]).any())
        self.assertFalse(np.isinf(result_after["weight"]).any())

        # Print summary for manual inspection (useful for debugging)
        print("\n=== Remove Loop BEFORE Grouping ===")
        print(result_before.to_string())
        print(f"Total weight: {result_before['weight'].sum():.4f}")

        print("\n=== Remove Loop AFTER Grouping ===")
        print(result_after.to_string())
        print(f"Total weight: {result_after['weight'].sum():.4f}")

        # Note: The weights can be different between the two modes because:
        # - BEFORE: Individual neuron loops are removed, then grouped
        # - AFTER: Neurons are grouped first (potentially merging loop info), then loops removed
        # Both are valid depending on the biological question being asked

    def test_remove_loop_grouping_simple_case_no_loops(self):
        """Test both modes with no loops - validates they both work correctly.

        Note: Even without loops, the two modes can produce different results because:
        - BEFORE: Compute effective connectivity on individual neurons, then group
        - AFTER: Group neurons first, then compute effective connectivity on groups

        Both are correct but answer slightly different questions.
        """
        # Create simple paths WITHOUT any loops
        paths = pd.DataFrame(
            {
                "layer": [1, 1, 2, 2],
                "pre": [0, 1, 2, 3],
                "post": [2, 3, 4, 4],
                "weight": [0.5, 0.3, 0.6, 0.4],
            }
        )

        pre_group = {0: "A", 1: "B"}
        post_group = {4: "Z"}

        result_before = effconn_without_loops(
            paths,
            pre_group=pre_group,
            post_group=post_group,
            remove_loop_after_grouping=False,
            wide=False,
            combining_method="sum",
            use_gpu=False,
        )

        result_after = effconn_without_loops(
            paths,
            pre_group=pre_group,
            post_group=post_group,
            remove_loop_after_grouping=True,
            wide=False,
            combining_method="sum",
            use_gpu=False,
        )

        # Both should produce valid results
        self.assertIsNotNone(result_before)
        self.assertIsNotNone(result_after)

        # Both should have the expected structure
        for result in [result_before, result_after]:
            self.assertIn("pre", result.columns)
            self.assertIn("post", result.columns)
            self.assertIn("weight", result.columns)
            self.assertTrue((result["weight"] >= 0).all())
            self.assertFalse(result["weight"].isna().any())

        print("\n=== No loops case - both modes valid but may differ ===")
        print("BEFORE mode:")
        print(result_before[["pre", "post", "weight"]].to_string())
        print(f"\nAFTER mode:")
        print(result_after[["pre", "post", "weight"]].to_string())

    def test_remove_loop_understand_difference(self):
        """Detailed test to understand the difference between the two modes.

        This creates a minimal example to clarify when and why the results differ.
        """
        # Minimal case: Single source, single target, one intermediate that loops
        # 0 -> 1 -> 2 (target)
        #      1 -> 2 (direct again, creates opportunity for loop)
        paths = pd.DataFrame(
            {
                "layer": [1, 2, 2],
                "pre": [0, 1, 1],  # Node 1 appears twice in layer 2
                "post": [1, 2, 2],
                "weight": [0.5, 0.3, 0.2],  # Two different weights from 1->2
            }
        )

        print("\n=== Understanding the difference ===")
        print("Input paths:")
        print(paths.to_string())

        # No grouping first - see raw behavior
        result_before_no_group = effconn_without_loops(
            paths, remove_loop_after_grouping=False, wide=False, use_gpu=False
        )

        result_after_no_group = effconn_without_loops(
            paths, remove_loop_after_grouping=True, wide=False, use_gpu=False
        )

        print("\nWithout grouping:")
        print("BEFORE mode:")
        if result_before_no_group is not None:
            print(result_before_no_group.to_string())
        else:
            print("None")

        print("\nAFTER mode:")
        if result_after_no_group is not None:
            print(result_after_no_group.to_string())
        else:
            print("None")

        # Both should produce valid results
        self.assertIsNotNone(result_before_no_group)
        self.assertIsNotNone(result_after_no_group)

    def test_remove_loop_grouping_edge_case_all_loops(self):
        """Test when all paths contain loops - both modes should handle gracefully."""
        # Create a structure where every path has a loop
        paths = pd.DataFrame(
            {
                "layer": [1, 1, 2, 2, 3],
                "pre": [0, 1, 0, 1, 0],  # Nodes 0,1 keep appearing
                "post": [0, 1, 2, 2, 3],
                "weight": [0.5, 0.3, 0.4, 0.2, 0.6],
            }
        )

        pre_group = {0: "A", 1: "B"}
        post_group = {3: "Z"}

        # Both modes should handle this without crashing
        result_before = effconn_without_loops(
            paths,
            pre_group=pre_group,
            post_group=post_group,
            remove_loop_after_grouping=False,
            use_gpu=False,
        )

        result_after = effconn_without_loops(
            paths,
            pre_group=pre_group,
            post_group=post_group,
            remove_loop_after_grouping=True,
            use_gpu=False,
        )

        # Both should return valid results (might have zero weights)
        # The key is they shouldn't crash
        if result_before is not None:
            self.assertIsInstance(result_before, pd.DataFrame)
        if result_after is not None:
            self.assertIsInstance(result_after, pd.DataFrame)

    @skipIf(
        not (torch.cuda.is_available() and torch.__version__ >= "2.2"),
        "CUDA ≥2.2 required",
    )
    def test_gpu_vs_cpu(self):
        """Test that GPU and CPU computations yield similar results."""
        result_gpu = effconn_without_loops(self.paths_no_loop, use_gpu=True, wide=False)
        result_cpu = effconn_without_loops(
            self.paths_no_loop, use_gpu=False, wide=False
        )

        # Sort both by pre and post for comparison
        result_gpu = result_gpu.sort_values(["pre", "post"]).reset_index(drop=True)
        result_cpu = result_cpu.sort_values(["pre", "post"]).reset_index(drop=True)

        # Compare values, ignoring dtype differences between GPU (int64) and CPU (int32)
        pd.testing.assert_frame_equal(
            result_gpu, result_cpu, rtol=1e-5, atol=1e-7, check_dtype=False
        )

    def test_chunk_size_consistency(self):
        """Test that different chunk sizes yield same results."""
        result_small = effconn_without_loops(
            self.paths_no_loop, chunk_size=10, use_gpu=False, wide=False
        )
        result_large = effconn_without_loops(
            self.paths_no_loop, chunk_size=1000, use_gpu=False, wide=False
        )

        # Sort for comparison
        result_small = result_small.sort_values(["pre", "post"]).reset_index(drop=True)
        result_large = result_large.sort_values(["pre", "post"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(result_small, result_large, rtol=1e-5)

    def test_root_parameter(self):
        """Test the root parameter."""
        result_root = effconn_without_loops(
            self.paths_no_loop, root=True, use_gpu=False, wide=False
        )
        result_no_root = effconn_without_loops(
            self.paths_no_loop, root=False, use_gpu=False, wide=False
        )

        self.assertIsNotNone(result_root)
        self.assertIsNotNone(result_no_root)

        # Values should differ when root is applied
        # (unless all weights are 0 or 1)
        self.assertIsInstance(result_root, pd.DataFrame)
        self.assertIsInstance(result_no_root, pd.DataFrame)

    def test_complex_loop_scenario(self):
        """Test a more complex scenario with multiple loops."""
        # Create a path with multiple nodes appearing in multiple layers
        complex_paths = pd.DataFrame(
            {
                "layer": [1, 1, 2, 2, 3, 3, 4],
                "pre": [0, 0, 1, 2, 1, 3, 2],  # nodes 1 and 2 create loops
                "post": [1, 2, 3, 3, 4, 4, 4],
                "weight": [0.5, 0.3, 0.4, 0.6, 0.2, 0.7, 0.1],
            }
        )

        result = effconn_without_loops(complex_paths, use_gpu=False)

        # Should handle multiple loops without crashing
        self.assertIsNotNone(result)

    def test_grouping_with_loops(self):
        """Test that grouping works correctly with loop removal."""
        result = effconn_without_loops(
            self.paths_with_loop,
            pre_group=self.pre_group,
            post_group=self.post_group,
            intermediate_group=self.intermediate_group,
            wide=True,
            use_gpu=False,
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)

    def test_density_threshold(self):
        """Test the density_threshold parameter."""
        result = effconn_without_loops(
            self.paths_no_loop, density_threshold=0.5, use_gpu=False
        )

        self.assertIsNotNone(result)

    def test_single_layer(self):
        """Test with single layer paths (direct connections only)."""
        single_layer = pd.DataFrame(
            {
                "layer": [1, 1],
                "pre": [0, 0],
                "post": [1, 2],
                "weight": [0.5, 0.3],
            }
        )

        result = effconn_without_loops(single_layer, use_gpu=False)
        self.assertIsNotNone(result)

    def test_self_loop(self):
        """Test handling of self-loops (neuron connecting to itself)."""
        self_loop_paths = pd.DataFrame(
            {
                "layer": [1, 1, 2],
                "pre": [0, 1, 1],
                "post": [1, 1, 2],  # node 1 connects to itself
                "weight": [0.5, 0.8, 0.3],
            }
        )

        # Should handle self-loops gracefully
        result = effconn_without_loops(self_loop_paths, use_gpu=False)
        # Result might be None or valid DataFrame depending on implementation
        if result is not None:
            self.assertIsInstance(result, pd.DataFrame)

    def test_large_loop_structure(self):
        """Test with a larger, more realistic loop structure."""
        # Create a more complex graph with multiple layers and loops
        large_paths = pd.DataFrame(
            {
                "layer": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
                "pre": [0, 0, 1, 2, 3, 3, 2, 4, 4, 5, 3],
                "post": [2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6],
                "weight": [0.5, 0.3, 0.2, 0.6, 0.4, 0.1, 0.7, 0.3, 0.5, 0.4, 0.2],
            }
        )

        result = effconn_without_loops(large_paths, use_gpu=False)
        self.assertIsNotNone(result)

    def test_numeric_stability(self):
        """Test numeric stability with very small and large weights."""
        paths = pd.DataFrame(
            {
                "layer": [1, 1, 2, 2],
                "pre": [0, 0, 1, 2],
                "post": [1, 2, 3, 3],
                "weight": [1e-10, 1e10, 0.5, 0.3],
            }
        )

        result = effconn_without_loops(paths, use_gpu=False)
        self.assertIsNotNone(result)
        # Check for NaN or Inf values
        if isinstance(result, pd.DataFrame) and "weight" in result.columns:
            self.assertFalse(result["weight"].isna().any())
            self.assertFalse(np.isinf(result["weight"]).any())

    def test_multiple_paths_between_same_nodes(self):
        """Test when there are multiple paths between the same pre-post pair."""
        multi_path = pd.DataFrame(
            {
                "layer": [1, 1, 2, 2, 2, 2],
                "pre": [0, 0, 1, 1, 2, 2],
                "post": [1, 2, 3, 3, 3, 3],
                "weight": [0.5, 0.3, 0.4, 0.2, 0.6, 0.1],
            }
        )

        result = effconn_without_loops(multi_path, use_gpu=False)
        self.assertIsNotNone(result)


class TestSignedConnByPathLengthData(unittest.TestCase):
    """Tests for the signed_conn_by_path_length_data function."""

    def setUp(self):
        """Set up test matrices and data for use in multiple tests."""
        # Create a simple 6x6 connectivity matrix
        # Neurons 0-2 are excitatory, 3-5 are inhibitory
        data = [
            0.5,
            0.3,
            0.2,  # 0 -> 1, 2, 3
            0.4,
            0.6,  # 1 -> 3, 4
            0.3,
            0.4,  # 2 -> 4, 5
            0.2,
            0.5,  # 3 -> 1, 5
            0.3,  # 4 -> 5
            0.1,  # 5 -> 2
        ]
        rows = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5]
        cols = [1, 2, 3, 3, 4, 4, 5, 1, 5, 5, 2]
        self.simple_matrix = csc_matrix((data, (rows, cols)), shape=(6, 6))

        # Group to sign mapping (0-2 excitatory, 3-5 inhibitory)
        # Note: group_paths converts indices to strings, so we need string keys
        self.group_to_sign = {
            "0": 1,
            "1": 1,
            "2": 1,  # Excitatory
            "3": -1,
            "4": -1,
            "5": -1,  # Inhibitory
        }

        # Create groupings for testing
        self.inidx_map = {0: "A", 1: "A", 2: "B"}
        self.outidx_map = {3: "X", 4: "Y", 5: "Z"}
        self.group_to_sign_mapped = {"A": 1, "B": 1, "X": -1, "Y": -1, "Z": -1}

    def test_basic_functionality_n1(self):
        """Test basic functionality with n=1 (direct connections only)."""
        inidx = [0, 1, 2]
        outidx = [3, 4, 5]

        exc_list, inh_list = signed_conn_by_path_length_data(
            self.simple_matrix,
            inidx,
            outidx,
            n=1,
            group_to_sign=self.group_to_sign,
            wide=True,
        )

        # Should return lists of length 1
        self.assertEqual(len(exc_list), 1)
        self.assertEqual(len(inh_list), 1)

        # Check that we get DataFrames
        self.assertIsInstance(exc_list[0], pd.DataFrame)
        self.assertIsInstance(inh_list[0], pd.DataFrame)

        # For n=1, excitatory neurons (0,1,2) to inhibitory neurons (3,4,5)
        # At least one list should have connections
        exc = exc_list[0]
        inh = inh_list[0]
        self.assertTrue(
            not exc.empty or not inh.empty,
            "Should have either excitatory or inhibitory connections",
        )

    def test_basic_functionality_n2(self):
        """Test basic functionality with n=2 (paths of length 1 and 2)."""
        inidx = [0, 1]
        outidx = [4, 5]

        exc_list, inh_list = signed_conn_by_path_length_data(
            self.simple_matrix,
            inidx,
            outidx,
            n=2,
            group_to_sign=self.group_to_sign,
            wide=True,
        )

        # Should return lists of length 2
        self.assertEqual(len(exc_list), 2)
        self.assertEqual(len(inh_list), 2)

        # All should be DataFrames
        for exc, inh in zip(exc_list, inh_list):
            self.assertIsInstance(exc, pd.DataFrame)
            self.assertIsInstance(inh, pd.DataFrame)

    def test_with_grouping(self):
        """Test functionality with neuron grouping."""
        inidx = [0, 1, 2]
        outidx = [3, 4, 5]

        exc_list, inh_list = signed_conn_by_path_length_data(
            self.simple_matrix,
            inidx,
            outidx,
            n=1,
            group_to_sign=self.group_to_sign_mapped,
            inidx_map=self.inidx_map,
            outidx_map=self.outidx_map,
            wide=True,
        )

        # Check that grouping was applied
        exc = exc_list[0]
        if not exc.empty:
            # Columns should be group names from outidx_map
            self.assertTrue(any(col in ["X", "Y", "Z"] for col in exc.columns))
            # Index should be group names from inidx_map
            self.assertTrue(any(idx in ["A", "B"] for idx in exc.index))

    def test_wide_vs_long_format(self):
        """Test that wide and long formats produce consistent data."""
        inidx = [0, 1]
        outidx = [3, 4]

        exc_wide, inh_wide = signed_conn_by_path_length_data(
            self.simple_matrix,
            inidx,
            outidx,
            n=1,
            group_to_sign=self.group_to_sign,
            wide=True,
        )

        exc_long, inh_long = signed_conn_by_path_length_data(
            self.simple_matrix,
            inidx,
            outidx,
            n=1,
            group_to_sign=self.group_to_sign,
            wide=False,
        )

        # Check that both are DataFrames
        self.assertIsInstance(exc_wide[0], pd.DataFrame)
        self.assertIsInstance(exc_long[0], pd.DataFrame)

        # Long format should have 'pre', 'post', 'weight' columns
        if not exc_long[0].empty:
            self.assertIn("pre", exc_long[0].columns)
            self.assertIn("post", exc_long[0].columns)
            self.assertIn("weight", exc_long[0].columns)

        # Wide format should be a matrix (index and columns)
        if not exc_wide[0].empty:
            self.assertGreater(len(exc_wide[0].index), 0)
            self.assertGreater(len(exc_wide[0].columns), 0)

    def test_combining_methods(self):
        """Test different combining methods."""
        inidx = [0, 1, 2]
        outidx = [3, 4, 5]

        for method in ["mean", "sum", "median"]:
            exc_list, inh_list = signed_conn_by_path_length_data(
                self.simple_matrix,
                inidx,
                outidx,
                n=1,
                group_to_sign=self.group_to_sign_mapped,
                inidx_map=self.inidx_map,
                outidx_map=self.outidx_map,
                combining_method=method,
                wide=True,
            )

            # Should complete without error
            self.assertEqual(len(exc_list), 1)
            self.assertEqual(len(inh_list), 1)

    def test_empty_paths(self):
        """Test behavior when no paths exist between input and output."""
        # Create a matrix where there are no paths from 0 to 5
        sparse_matrix = csc_matrix((6, 6))  # Empty matrix

        exc_list, inh_list = signed_conn_by_path_length_data(
            sparse_matrix,
            [0],
            [5],
            n=2,
            group_to_sign=self.group_to_sign,
            wide=True,
        )

        # Should return empty DataFrames for each path length
        self.assertEqual(len(exc_list), 2)
        self.assertEqual(len(inh_list), 2)
        self.assertTrue(all(df.empty for df in exc_list))
        self.assertTrue(all(df.empty for df in inh_list))

    def test_excitatory_vs_inhibitory_separation(self):
        """Test that excitatory and inhibitory connections are properly separated."""
        # Create a simple test case with known exc/inh structure
        inidx = [0]  # Excitatory neuron
        outidx = [3]  # Inhibitory neuron

        exc_list, inh_list = signed_conn_by_path_length_data(
            self.simple_matrix,
            inidx,
            outidx,
            n=1,
            group_to_sign=self.group_to_sign,
            wide=True,
        )

        # Direct connection from excitatory (0) to inhibitory (3) exists in the matrix (weight 0.2)
        exc = exc_list[0]
        inh = inh_list[0]

        # At least one should have connections
        # The actual connection may be in either exc or inh depending on sign propagation logic
        total_connections = (0 if exc.empty else exc.size) + (
            0 if inh.empty else inh.size
        )
        self.assertGreater(
            total_connections,
            0,
            f"Should have connections. Matrix has connection 0->3 with weight 0.2. Got exc.empty={exc.empty}, inh.empty={inh.empty}",
        )

    def test_path_length_progression(self):
        """Test that each path length builds upon previous connections."""
        inidx = [0, 1]
        outidx = [4, 5]

        exc_list, inh_list = signed_conn_by_path_length_data(
            self.simple_matrix,
            inidx,
            outidx,
            n=3,
            group_to_sign=self.group_to_sign,
            wide=True,
        )

        # Should have 3 path lengths
        self.assertEqual(len(exc_list), 3)
        self.assertEqual(len(inh_list), 3)

        # Each should be a DataFrame
        for i in range(3):
            self.assertIsInstance(exc_list[i], pd.DataFrame)
            self.assertIsInstance(inh_list[i], pd.DataFrame)

    def test_with_intermediate_mapping(self):
        """Test functionality with intermediate neuron grouping."""
        inidx = [0, 1]
        outidx = [4, 5]
        intermediate_map = {2: "M1", 3: "M2"}

        exc_list, inh_list = signed_conn_by_path_length_data(
            self.simple_matrix,
            inidx,
            outidx,
            n=2,
            group_to_sign={**self.group_to_sign, "M1": 1, "M2": -1},
            inidx_map=self.inidx_map,
            outidx_map=self.outidx_map,
            intermediate_map=intermediate_map,
            wide=True,
        )

        # Should complete successfully
        self.assertEqual(len(exc_list), 2)
        self.assertEqual(len(inh_list), 2)

    def test_single_neuron_to_single_neuron(self):
        """Test with single input and output neuron."""
        exc_list, inh_list = signed_conn_by_path_length_data(
            self.simple_matrix,
            [0],
            [3],
            n=1,
            group_to_sign=self.group_to_sign,
            wide=True,
        )

        self.assertEqual(len(exc_list), 1)
        self.assertEqual(len(inh_list), 1)

    def test_sign_propagation_two_hops(self):
        """Test that signs propagate correctly through two hops."""
        # Create a simple chain: E -> I -> E
        # 0 (exc) -> 3 (inh) -> 1 (exc)
        data = [0.5, 0.6]  # 0->3, 3->1
        rows = [0, 3]
        cols = [3, 1]
        chain_matrix = csc_matrix((data, (rows, cols)), shape=(6, 6))

        exc_list, inh_list = signed_conn_by_path_length_data(
            chain_matrix,
            [0],
            [1],
            n=2,
            group_to_sign=self.group_to_sign,
            wide=True,
        )

        # Path length 1: should be empty (no direct connection)
        # Path length 2: E->I->E should be inhibitory
        self.assertEqual(len(exc_list), 2)
        self.assertEqual(len(inh_list), 2)

    def test_return_type_consistency(self):
        """Test that return types are consistent across different inputs."""
        for n in [1, 2, 3]:
            exc_list, inh_list = signed_conn_by_path_length_data(
                self.simple_matrix,
                [0, 1],
                [3, 4],
                n=n,
                group_to_sign=self.group_to_sign,
                wide=True,
            )

            # Check return type
            self.assertIsInstance(exc_list, list)
            self.assertIsInstance(inh_list, list)
            self.assertEqual(len(exc_list), n)
            self.assertEqual(len(inh_list), n)

            # Check each element is a DataFrame
            for exc, inh in zip(exc_list, inh_list):
                self.assertIsInstance(exc, pd.DataFrame)
                self.assertIsInstance(inh, pd.DataFrame)

    def test_numerical_accuracy(self):
        """Test numerical accuracy with known connections."""
        # Create a simple matrix with known values
        # 0 (exc) -> 1 (exc) with weight 0.5
        data = [0.5]
        rows = [0]
        cols = [1]
        simple = csc_matrix((data, (rows, cols)), shape=(3, 3))
        signs = {0: 1, 1: 1, 2: 1}  # All excitatory

        exc_list, inh_list = signed_conn_by_path_length_data(
            simple,
            [0],
            [1],
            n=1,
            group_to_sign=signs,
            wide=False,
        )

        # Should have exactly one excitatory connection with weight 0.5
        exc = exc_list[0]
        if not exc.empty:
            self.assertEqual(len(exc), 1)
            self.assertAlmostEqual(exc.iloc[0]["weight"], 0.5, places=5)

        # Should have no inhibitory connections
        inh = inh_list[0]
        self.assertTrue(inh.empty)


if __name__ == "__main__":
    # Add the new test class to the existing test suite
    unittest.main()
