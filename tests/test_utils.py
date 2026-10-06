import unittest

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix

from connectome_interpreter.utils import modify_coo_matrix, pytorch_sparse_to_scipy


class TestPytorchSparseToScipy(unittest.TestCase):
    def test_round_trip_values_and_formats(self):
        dense = np.array([[0.0, 1.5], [-2.0, 0.0]], dtype=np.float32)
        tensor = torch.tensor(dense).to_sparse()
        for fmt in ("csr", "csc", "coo"):
            out = pytorch_sparse_to_scipy(tensor, scipy_format=fmt)
            self.assertEqual(out.format, fmt)
            np.testing.assert_array_equal(out.toarray(), dense)
        with self.assertRaises(ValueError):
            pytorch_sparse_to_scipy(tensor, scipy_format="lil")

    def test_accepts_tensor_that_requires_grad(self):
        # e.g. a network's effective_weights while its slopes are trainable
        values = torch.tensor([1.5, -2.0], requires_grad=True)
        indices = torch.tensor([[0, 1], [1, 0]])
        tensor = torch.sparse_coo_tensor(indices, values, (2, 2))
        out = pytorch_sparse_to_scipy(tensor)
        np.testing.assert_array_equal(out.toarray(), [[0.0, 1.5], [-2.0, 0.0]])


class TestModifyCooMatrix(unittest.TestCase):

    def setUp(self):
        # Create a small sparse matrix with float data type and values between 0 and 1
        self.example_coo = coo_matrix(
            np.array([[0.1, 0.2, 0], [0, 0, 0.3], [0.4, 0, 0.5]], dtype=np.float32)
        )  # Ensure dtype is float32

        # DataFrame for batch updates with float values
        self.updates_df = pd.DataFrame(
            {
                "input_idx": [0, 2],
                "output_idx": [2, 1],
                "value": [0.25, 0.75],  # Use values between 0 and 1
            }
        )

    def test_single_update(self):
        # Test updating a single value with a float value between 0 and 1
        result = modify_coo_matrix(
            self.example_coo, input_idx=1, output_idx=2, value=0.99, re_normalize=False
        )
        self.assertAlmostEqual(result.toarray()[1, 2], 0.99)

    def test_batch_update(self):
        # Test updating using a DataFrame of updates with float values
        result = modify_coo_matrix(
            self.example_coo, updates_df=self.updates_df, re_normalize=False
        )
        self.assertAlmostEqual(result.toarray()[0, 2], 0.25)
        self.assertAlmostEqual(result.toarray()[2, 1], 0.75)

    def test_renormalize(self):
        # Test the re-normalization functionality with float values
        result = modify_coo_matrix(
            self.example_coo, updates_df=self.updates_df, re_normalize=True
        )
        updated_cols = self.updates_df["output_idx"].unique()
        # Check if columns sum up to 1 or close to it
        column_sums = np.array(result.sum(axis=0))[0]
        for col_sum in column_sums[updated_cols]:
            # Assuming a little margin for floating-point arithmetic
            self.assertTrue(0.99 <= col_sum <= 1.01)

    def test_handle_zero_colsums(self):
        # Test how the function handles columns with zero sums after update with float values
        zero_updates_df = pd.DataFrame(
            {
                "input_idx": [1, 0],
                "output_idx": [2, 1],
                "value": [0.0, 0.0],  # Use float zero to keep consistent data type
            }
        )
        result = modify_coo_matrix(
            self.example_coo, updates_df=zero_updates_df, re_normalize=True
        )
        column_sums = np.array(result.sum(axis=0))[0]
        self.assertAlmostEqual(column_sums[1], 0.0)
