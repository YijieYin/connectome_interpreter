import itertools
import random
import unittest

import pandas as pd

from connectome_interpreter.path_finding import find_xor, find_paths_of_length

import numpy as np
import scipy.sparse as sp


class TestFindXOR(unittest.TestCase):

    def create_test_df(self, edges):
        """Helper function to create test DataFrame from edge list"""
        return pd.DataFrame(edges, columns=["pre", "post", "sign", "layer"])

    def test_basic_xor_circuit(self):
        """Test basic XOR circuit with single inputs/outputs"""
        edges = [
            ("a", "c", 1, 1),
            ("b", "d", 1, 1),
            ("a", "e", 1, 1),
            ("b", "e", 1, 1),
            ("c", "f", 1, 2),
            ("d", "f", 1, 2),
            ("e", "f", -1, 2),
        ]
        df = self.create_test_df(edges)
        circuits = find_xor(df)
        self.assertEqual(len(circuits), 1)
        circuit = circuits[0]
        self.assertEqual(circuit.input1, ["a"])
        self.assertEqual(circuit.input2, ["b"])
        self.assertEqual({"c", "d"}, {circuit.exciter1, circuit.exciter2})
        self.assertEqual(circuit.inhibitor, "e")
        self.assertEqual(circuit.output, ["f"])

    def test_multiple_inputs_outputs(self):
        """Test XOR circuit with multiple inputs and outputs"""
        edges = [
            ("a1", "c", 1, 1),
            ("a2", "c", 1, 1),
            ("b1", "d", 1, 1),
            ("b2", "d", 1, 1),
            ("a1", "e", 1, 1),
            ("a2", "e", 1, 1),
            ("b1", "e", 1, 1),
            ("b2", "e", 1, 1),
            ("c", "f1", 1, 2),
            ("c", "f2", 1, 2),
            ("d", "f1", 1, 2),
            ("d", "f2", 1, 2),
            ("e", "f1", -1, 2),
            ("e", "f2", -1, 2),
        ]
        df = self.create_test_df(edges)
        circuits = find_xor(df)
        self.assertEqual(len(circuits), 1)
        circuit = circuits[0]
        self.assertEqual(set(circuit.input1), {"a1", "a2"})
        self.assertEqual(set(circuit.input2), {"b1", "b2"})
        self.assertEqual({"c", "d"}, {circuit.exciter1, circuit.exciter2})
        self.assertEqual(circuit.inhibitor, "e")
        self.assertEqual(set(circuit.output), {"f1", "f2"})

    def test_no_xor_circuit(self):
        """Test case where no XOR circuit exists"""
        edges = [
            ("a", "c", 1, 1),
            ("b", "d", 1, 1),
            ("c", "f", 1, 2),
            ("d", "f", 1, 2),
        ]
        df = self.create_test_df(edges)
        circuits = find_xor(df)
        self.assertEqual(len(circuits), 0)

    def test_invalid_layer_numbers(self):
        """Test error handling for invalid layer numbers"""
        edges = [("a", "b", 1, 1), ("b", "c", 1, 3)]
        df = self.create_test_df(edges)
        with self.assertRaises(ValueError):
            find_xor(df)

    def test_multiple_xor_circuits(self):
        """Test detection of multiple XOR circuits"""
        edges = [
            ("a1", "c1", 1, 1),
            ("b1", "d1", 1, 1),
            ("a1", "e1", 1, 1),
            ("b1", "e1", 1, 1),
            ("c1", "f1", 1, 2),
            ("d1", "f1", 1, 2),
            ("e1", "f1", -1, 2),
            ("a2", "c2", 1, 1),
            ("b2", "d2", 1, 1),
            ("a2", "e2", 1, 1),
            ("b2", "e2", 1, 1),
            ("c2", "f2", 1, 2),
            ("d2", "f2", 1, 2),
            ("e2", "f2", -1, 2),
        ]
        df = self.create_test_df(edges)
        circuits = find_xor(df)
        self.assertEqual(len(circuits), 2)
        for circuit in circuits:
            self.assertEqual(len(circuit.input1), 1)
            self.assertEqual(len(circuit.input2), 1)
            self.assertIn(circuit.inhibitor, ["e1", "e2"])
            self.assertEqual(len(circuit.output), 1)

    def test_error_conditions(self):
        """Test various error conditions and edge cases"""
        with self.assertRaises(ValueError):
            find_xor(pd.DataFrame(columns=["pre", "post", "sign", "layer"]))
        incomplete_df = pd.DataFrame(
            {"pre": ["a", "b"], "post": ["c", "d"], "layer": [1, 2]}
        )
        with self.assertRaises(ValueError):
            find_xor(incomplete_df)
        invalid_signs = [
            ("a", "c", 2, 1),
            ("b", "d", 1, 1),
            ("c", "f", 1, 2),
            ("d", "f", -1, 2),
        ]
        df = self.create_test_df(invalid_signs)
        with self.assertRaises(ValueError):
            find_xor(df)
        disconnected = [("a", "c", 1, 1), ("b", "d", 1, 1), ("e", "f", -1, 2)]
        df = self.create_test_df(disconnected)
        circuits = find_xor(df)
        self.assertEqual(len(circuits), 0)

    def test_overlapping_xor_circuits(self):
        """Test detection of XOR circuits that share nodes"""
        edges = [
            ("a", "c", 1, 1),
            ("b1", "d1", 1, 1),
            ("b2", "d2", 1, 1),
            ("a", "e1", 1, 1),
            ("b1", "e1", 1, 1),
            ("a", "e2", 1, 1),
            ("b2", "e2", 1, 1),
            ("c", "f1", 1, 2),
            ("c", "f2", 1, 2),
            ("d1", "f1", 1, 2),
            ("d2", "f2", 1, 2),
            ("e1", "f1", -1, 2),
            ("e2", "f2", -1, 2),
        ]
        df = self.create_test_df(edges)
        circuits = find_xor(df)
        self.assertGreater(len(circuits), 0)
        input1_nodes = set()
        for circuit in circuits:
            input1_nodes.update(circuit.input1)
        self.assertIn("a", input1_nodes)

    def test_specific_topologies(self):
        """Test specific circuit topologies"""
        convergent = [
            ("a1", "c", 1, 1),
            ("a2", "c", 1, 1),
            ("b1", "d", 1, 1),
            ("b2", "d", 1, 1),
            ("a1", "e", 1, 1),
            ("a2", "e", 1, 1),
            ("b1", "e", 1, 1),
            ("b2", "e", 1, 1),
            ("c", "f", 1, 2),
            ("d", "f", 1, 2),
            ("e", "f", -1, 2),
        ]
        df = self.create_test_df(convergent)
        circuits = find_xor(df)
        self.assertGreater(len(circuits), 0)
        self.assertGreater(len(circuits[0].input1), 1)

        divergent = [
            ("a", "c", 1, 1),
            ("b", "d", 1, 1),
            ("a", "e", 1, 1),
            ("b", "e", 1, 1),
            ("c", "f1", 1, 2),
            ("c", "f2", 1, 2),
            ("d", "f1", 1, 2),
            ("d", "f2", 1, 2),
            ("e", "f1", -1, 2),
            ("e", "f2", -1, 2),
        ]
        df = self.create_test_df(divergent)
        circuits = find_xor(df)
        self.assertGreater(len(circuits), 0)
        self.assertGreater(len(circuits[0].output), 1)

    def test_fully_connected_network(self):
        """Test with a fully connected network"""
        edges = []
        inputs = ["a", "b"]
        middles = ["c", "d", "e"]
        outputs = ["f"]
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


class TestFindPathsOfLength(unittest.TestCase):
    # ------------------------------------------------------------------ helpers
    @staticmethod
    def mk_chain_df(nodes, w=1.0):
        """chain graph → DataFrame"""
        return pd.DataFrame(
            [(nodes[i], nodes[i + 1], w) for i in range(len(nodes) - 1)],
            columns=["pre", "post", "weight"],
        )

    @staticmethod
    def mk_chain_sparse(n):
        """simple 0->1->…->n sparse COO"""
        rows = np.arange(n)
        cols = np.arange(1, n + 1)
        data = np.ones(n)
        return sp.coo_matrix((data, (rows, cols)), shape=(n + 1, n + 1))

    # --------------------------------------------------------- basic correctness
    def test_layer1_direct(self):
        df = self.mk_chain_df(["a", "b"])
        res = find_paths_of_length(df, ["a"], ["b"], 1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["pre"], "a")
        self.assertEqual(res.iloc[0]["post"], "b")
        self.assertEqual(res.iloc[0]["layer"], 1)

    def test_even_length_path(self):
        df = self.mk_chain_df(list(range(5)))  # 0-1-2-3-4
        res = find_paths_of_length(df, [0], [4], 4)
        self.assertEqual(len(res), 4)
        self.assertTrue((res["layer"].values == np.arange(1, 5)).all())
        self.assertListEqual(res["pre"].tolist(), [0, 1, 2, 3])
        self.assertListEqual(res["post"].tolist(), [1, 2, 3, 4])

    def test_odd_length_path(self):
        df = self.mk_chain_df(list("abcd"))  # a-b-c-d
        res = find_paths_of_length(df, ["a"], ["d"], 3)
        self.assertEqual(len(res), 3)
        self.assertListEqual(res["layer"].tolist(), [1, 2, 3])

    # ------------------------------------------------------ input type handling
    def test_sparse_matrix_input(self):
        sparse_graph = self.mk_chain_sparse(3)  # 0-1-2-3
        res = find_paths_of_length(sparse_graph, [0], [3], 3)
        self.assertEqual(len(res), 3)
        self.assertListEqual(res["pre"].tolist(), [0, 1, 2])
        self.assertListEqual(res["post"].tolist(), [1, 2, 3])

    # ------------------------------------------------------------- edge cases
    def test_no_path_returns_none(self):
        df = pd.DataFrame(
            [("a", "b", 1.0), ("c", "d", 1.0)], columns=["pre", "post", "weight"]
        )
        self.assertIsNone(find_paths_of_length(df, ["a"], ["d"], 2))

    def test_invalid_target_layer(self):
        df = self.mk_chain_df(["x", "y"])
        with self.assertRaises(AssertionError):
            find_paths_of_length(df, ["x"], ["y"], 0)

    def test_empty_indices(self):
        df = self.mk_chain_df(["x", "y"])
        with self.assertRaises(AssertionError):
            find_paths_of_length(df, [], ["y"], 1)
        with self.assertRaises(AssertionError):
            find_paths_of_length(df, ["x"], [], 1)

    def test_missing_columns(self):
        bad_df = pd.DataFrame({"pre": ["a"], "post": ["b"]})
        with self.assertRaises(ValueError):
            find_paths_of_length(bad_df, ["a"], ["b"], 1)

    # ------------------------------- algorithm early-exit on unreachable middle
    def test_early_exit_middle_layer(self):
        # graph: 0->1  (no edge onward), so layer 2 unreachable
        df = self.mk_chain_df([0, 1])
        self.assertIsNone(find_paths_of_length(df, [0], [2], 2))
