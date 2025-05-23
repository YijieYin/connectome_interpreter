import unittest
import numpy as np
import torch
import os
import tempfile
import pytest
import scipy as sp
from scipy.sparse import csr_matrix, csc_matrix

# Import the functions to test
# Adjust this import to match your actual module structure
from connectome_interpreter.compress_paths import (
    compress_paths,
    compress_paths_dense_chunked,
    compress_paths_not_chunked,
    compress_paths_signed,
    compress_paths_signed_no_chunking,
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
            np.testing.assert_allclose(loaded, dense_matrix, rtol=1e-5, atol=1e-7)

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
                self.simple_matrix, self.simple_idx_to_sign, 2, save_path=temp_dir
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
                self.simple_matrix, self.simple_idx_to_sign, 2, save_path=temp_dir
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
                self.simple_matrix, self.simple_idx_to_sign, 3, save_path=temp_dir
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


if __name__ == "__main__":
    unittest.main()
