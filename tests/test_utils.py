import pytest
import unittest
from connectome_interpreter.utils import modify_coo_matrix, find_xor, XORCircuit
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
import random
import itertools


class TestModifyCooMatrix(unittest.TestCase):

    def setUp(self):
        # Create a small sparse matrix with float data type and values between 0 and 1
        self.example_coo = coo_matrix(np.array([
            [0.1, 0.2, 0],
            [0, 0, 0.3],
            [0.4, 0, 0.5]
        ], dtype=np.float32))  # Ensure dtype is float32

        # DataFrame for batch updates with float values
        self.updates_df = pd.DataFrame({
            'input_idx': [0, 2],
            'output_idx': [2, 1],
            'value': [0.25, 0.75]  # Use values between 0 and 1
        })

    def test_single_update(self):
        # Test updating a single value with a float value between 0 and 1
        result = modify_coo_matrix(
            self.example_coo, input_idx=1, output_idx=2, value=0.99, re_normalize=False)
        self.assertAlmostEqual(result.toarray()[1, 2], 0.99)

    def test_batch_update(self):
        # Test updating using a DataFrame of updates with float values
        result = modify_coo_matrix(
            self.example_coo, updates_df=self.updates_df, re_normalize=False)
        self.assertAlmostEqual(result.toarray()[0, 2], 0.25)
        self.assertAlmostEqual(result.toarray()[2, 1], 0.75)

    def test_renormalize(self):
        # Test the re-normalization functionality with float values
        result = modify_coo_matrix(
            self.example_coo, updates_df=self.updates_df, re_normalize=True)
        updated_cols = self.updates_df['output_idx'].unique()
        # Check if columns sum up to 1 or close to it
        column_sums = np.array(result.sum(axis=0))[0]
        for col_sum in column_sums[updated_cols]:
            # Assuming a little margin for floating-point arithmetic
            self.assertTrue(0.99 <= col_sum <= 1.01)

    def test_handle_zero_colsums(self):
        # Test how the function handles columns with zero sums after update with float values
        zero_updates_df = pd.DataFrame({
            'input_idx': [1, 0],
            'output_idx': [2, 1],
            'value': [0.0, 0.0]  # Use float zero to keep consistent data type
        })
        result = modify_coo_matrix(
            self.example_coo, updates_df=zero_updates_df, re_normalize=True)
        column_sums = np.array(result.sum(axis=0))[0]
        self.assertAlmostEqual(column_sums[1], 0.0)


class TestFindXOR(unittest.TestCase):

    def create_test_df(self, edges):
        """Helper function to create test DataFrame from edge list"""
        return pd.DataFrame(edges, columns=['pre', 'post', 'sign', 'layer'])

    def test_basic_xor_circuit(self):
        """Test basic XOR circuit with single inputs/outputs"""
        edges = [
            ('a', 'c', 1, 1), ('b', 'd', 1, 1), ('a', 'e', 1, 1), ('b', 'e', 1, 1),
            ('c', 'f', 1, 2), ('d', 'f', 1, 2), ('e', 'f', -1, 2),
        ]
        df = self.create_test_df(edges)
        circuits = find_xor(df)
        self.assertEqual(len(circuits), 1)
        circuit = circuits[0]
        self.assertEqual(circuit.input1, ['a'])
        self.assertEqual(circuit.input2, ['b'])
        self.assertEqual({'c', 'd'}, {circuit.exciter1, circuit.exciter2})
        self.assertEqual(circuit.inhibitor, 'e')
        self.assertEqual(circuit.output, ['f'])

    def test_multiple_inputs_outputs(self):
        """Test XOR circuit with multiple inputs and outputs"""
        edges = [
            ('a1', 'c', 1, 1), ('a2', 'c', 1,
                                1), ('b1', 'd', 1, 1), ('b2', 'd', 1, 1),
            ('a1', 'e', 1, 1), ('a2', 'e', 1,
                                1), ('b1', 'e', 1, 1), ('b2', 'e', 1, 1),
            ('c', 'f1', 1, 2), ('c', 'f2', 1,
                                2), ('d', 'f1', 1, 2), ('d', 'f2', 1, 2),
            ('e', 'f1', -1, 2), ('e', 'f2', -1, 2),
        ]
        df = self.create_test_df(edges)
        circuits = find_xor(df)
        self.assertEqual(len(circuits), 1)
        circuit = circuits[0]
        self.assertEqual(set(circuit.input1), {'a1', 'a2'})
        self.assertEqual(set(circuit.input2), {'b1', 'b2'})
        self.assertEqual({'c', 'd'}, {circuit.exciter1, circuit.exciter2})
        self.assertEqual(circuit.inhibitor, 'e')
        self.assertEqual(set(circuit.output), {'f1', 'f2'})

    def test_no_xor_circuit(self):
        """Test case where no XOR circuit exists"""
        edges = [
            ('a', 'c', 1, 1), ('b', 'd', 1, 1), ('c', 'f', 1, 2), ('d', 'f', 1, 2),
        ]
        df = self.create_test_df(edges)
        circuits = find_xor(df)
        self.assertEqual(len(circuits), 0)

    def test_invalid_layer_numbers(self):
        """Test error handling for invalid layer numbers"""
        edges = [('a', 'b', 1, 1), ('b', 'c', 1, 3)]
        df = self.create_test_df(edges)
        with self.assertRaises(ValueError):
            find_xor(df)

    def test_multiple_xor_circuits(self):
        """Test detection of multiple XOR circuits"""
        edges = [
            ('a1', 'c1', 1, 1), ('b1', 'd1', 1,
                                 1), ('a1', 'e1', 1, 1), ('b1', 'e1', 1, 1),
            ('c1', 'f1', 1, 2), ('d1', 'f1', 1, 2), ('e1', 'f1', -1, 2),
            ('a2', 'c2', 1, 1), ('b2', 'd2', 1,
                                 1), ('a2', 'e2', 1, 1), ('b2', 'e2', 1, 1),
            ('c2', 'f2', 1, 2), ('d2', 'f2', 1, 2), ('e2', 'f2', -1, 2),
        ]
        df = self.create_test_df(edges)
        circuits = find_xor(df)
        self.assertEqual(len(circuits), 2)
        for circuit in circuits:
            self.assertEqual(len(circuit.input1), 1)
            self.assertEqual(len(circuit.input2), 1)
            self.assertIn(circuit.inhibitor, ['e1', 'e2'])
            self.assertEqual(len(circuit.output), 1)

    def test_error_conditions(self):
        """Test various error conditions and edge cases"""
        with self.assertRaises(ValueError):
            find_xor(pd.DataFrame(columns=['pre', 'post', 'sign', 'layer']))
        incomplete_df = pd.DataFrame(
            {'pre': ['a', 'b'], 'post': ['c', 'd'], 'layer': [1, 2]})
        with self.assertRaises(ValueError):
            find_xor(incomplete_df)
        invalid_signs = [('a', 'c', 2, 1), ('b', 'd', 1, 1),
                         ('c', 'f', 1, 2), ('d', 'f', -1, 2)]
        df = self.create_test_df(invalid_signs)
        with self.assertRaises(ValueError):
            find_xor(df)
        disconnected = [('a', 'c', 1, 1), ('b', 'd', 1, 1), ('e', 'f', -1, 2)]
        df = self.create_test_df(disconnected)
        circuits = find_xor(df)
        self.assertEqual(len(circuits), 0)

    def test_overlapping_xor_circuits(self):
        """Test detection of XOR circuits that share nodes"""
        edges = [
            ('a', 'c', 1, 1), ('b1', 'd1', 1, 1), ('b2', 'd2', 1, 1),
            ('a', 'e1', 1, 1), ('b1', 'e1', 1,
                                1), ('a', 'e2', 1, 1), ('b2', 'e2', 1, 1),
            ('c', 'f1', 1, 2), ('c', 'f2', 1,
                                2), ('d1', 'f1', 1, 2), ('d2', 'f2', 1, 2),
            ('e1', 'f1', -1, 2), ('e2', 'f2', -1, 2),
        ]
        df = self.create_test_df(edges)
        circuits = find_xor(df)
        self.assertGreater(len(circuits), 0)
        input1_nodes = set()
        for circuit in circuits:
            input1_nodes.update(circuit.input1)
        self.assertIn('a', input1_nodes)

    def test_specific_topologies(self):
        """Test specific circuit topologies"""
        convergent = [
            ('a1', 'c', 1, 1), ('a2', 'c', 1,
                                1), ('b1', 'd', 1, 1), ('b2', 'd', 1, 1),
            ('a1', 'e', 1, 1), ('a2', 'e', 1,
                                1), ('b1', 'e', 1, 1), ('b2', 'e', 1, 1),
            ('c', 'f', 1, 2), ('d', 'f', 1, 2), ('e', 'f', -1, 2),
        ]
        df = self.create_test_df(convergent)
        circuits = find_xor(df)
        self.assertGreater(len(circuits), 0)
        self.assertGreater(len(circuits[0].input1), 1)

        divergent = [
            ('a', 'c', 1, 1), ('b', 'd', 1, 1), ('a', 'e', 1, 1), ('b', 'e', 1, 1),
            ('c', 'f1', 1, 2), ('c', 'f2', 1,
                                2), ('d', 'f1', 1, 2), ('d', 'f2', 1, 2),
            ('e', 'f1', -1, 2), ('e', 'f2', -1, 2),
        ]
        df = self.create_test_df(divergent)
        circuits = find_xor(df)
        self.assertGreater(len(circuits), 0)
        self.assertGreater(len(circuits[0].output), 1)

    def test_fully_connected_network(self):
        """Test with a fully connected network"""
        edges = []
        inputs = ['a', 'b']
        middles = ['c', 'd', 'e']
        outputs = ['f']
        for inp in inputs:
            for mid in middles:
                edges.append((inp, mid, 1, 1))
        for mid in middles:
            for out in outputs:
                edges.append((mid, out, random.choice([1, -1]), 2))
        df = self.create_test_df(edges)
        circuits = find_xor(df)
        max_possible = len(list(itertools.combinations(middles, 3)))
        self.assertLessEqual(len(circuits), max_possible)


# Add these test cases to your test suite
if __name__ == "__main__":
    pytest.main([__file__])
