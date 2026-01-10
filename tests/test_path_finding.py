import itertools
import random
import unittest

import pandas as pd

from connectome_interpreter.path_finding import (
    find_xor,
    find_paths_of_length,
    enumerate_paths,
    count_paths,
)

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


class TestEnumeratePathsBetweenLayers(unittest.TestCase):
    # -------------------------- helpers
    @staticmethod
    def mk_df(edges):
        return pd.DataFrame(edges, columns=["pre", "post", "weight", "layer"])

    @staticmethod
    def to_set(paths):
        return {tuple(path) for path in paths}

    # ------------------------ correctness
    def test_simple_chain(self):
        edges = [
            ("a", "b", 1.0, 1),
            ("b", "c", 2.0, 2),
        ]
        df = self.mk_df(edges)
        paths = enumerate_paths(df, 1, 2)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0], [("a", "b", 1.0), ("b", "c", 2.0)])

    def test_branching_two_paths(self):
        edges = [
            ("a", "b", 1.0, 1),
            ("a", "d", 1.0, 1),
            ("b", "c", 1.0, 2),
            ("d", "c", 1.0, 2),
        ]
        df = self.mk_df(edges)
        paths = enumerate_paths(df, 1, 2)
        expected = {
            (("a", "b", 1.0), ("b", "c", 1.0)),
            (("a", "d", 1.0), ("d", "c", 1.0)),
        }
        self.assertEqual(self.to_set(paths), expected)

    def test_default_end_layer(self):
        edges = [
            ("x", "y", 1, 1),
            ("y", "z", 1, 2),
            ("z", "w", 1, 3),
        ]
        df = self.mk_df(edges)
        paths = enumerate_paths(df)  # end_layer defaults to 3
        self.assertEqual(paths[0][-1], ("z", "w", 1))

    # --------------------------- edge cases
    def test_no_paths_due_to_gap(self):
        edges = [
            ("p", "q", 1, 1),
            ("q", "r", 1, 3),  # layer 2 missing
        ]
        df = self.mk_df(edges)
        self.assertEqual(enumerate_paths(df, 1, 3), [])

    def test_start_equals_end_layer(self):
        edges = [
            ("m", "n", 0.5, 2),
            ("o", "p", 1.0, 2),
        ]
        df = self.mk_df(edges)
        paths = enumerate_paths(df, 2, 2)
        expected = {
            (("m", "n", 0.5),),
            (("o", "p", 1.0),),
        }
        self.assertEqual(self.to_set(paths), expected)

    def test_invalid_layer_order_raises(self):
        df = self.mk_df([("a", "b", 1, 1), ("b", "c", 1, 2)])
        with self.assertRaises(ValueError):
            enumerate_paths(df, 3, 2)

    def test_mixed_node_types_and_weights(self):
        edges = [
            (0, "a", 0.3, 1),
            ("a", 2, 0.7, 2),
        ]
        df = self.mk_df(edges)
        paths = enumerate_paths(df, 1, 2)
        self.assertEqual(paths[0], [(0, "a", 0.3), ("a", 2, 0.7)])

    # ------------------------ generator tests
    def test_generator_simple_chain(self):
        edges = [
            ("a", "b", 1.0, 1),
            ("b", "c", 2.0, 2),
        ]
        df = self.mk_df(edges)
        paths_gen = enumerate_paths(df, 1, 2, return_generator=True)
        paths = list(paths_gen)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0], [("a", "b", 1.0), ("b", "c", 2.0)])

    def test_generator_branching(self):
        edges = [
            ("a", "b", 1.0, 1),
            ("a", "d", 1.0, 1),
            ("b", "c", 1.0, 2),
            ("d", "c", 1.0, 2),
        ]
        df = self.mk_df(edges)
        paths_gen = enumerate_paths(df, 1, 2, return_generator=True)
        paths = list(paths_gen)
        expected = {
            (("a", "b", 1.0), ("b", "c", 1.0)),
            (("a", "d", 1.0), ("d", "c", 1.0)),
        }
        self.assertEqual(self.to_set(paths), expected)

    def test_generator_equals_list_output(self):
        """Ensure generator produces same results as list version"""
        edges = [
            ("a", "b", 1.0, 1),
            ("a", "c", 2.0, 1),
            ("b", "d", 3.0, 2),
            ("c", "d", 4.0, 2),
            ("c", "e", 5.0, 2),
            ("d", "f", 6.0, 3),
            ("e", "f", 7.0, 3),
        ]
        df = self.mk_df(edges)

        list_paths = enumerate_paths(df, 1, 3, return_generator=False)
        gen_paths = list(enumerate_paths(df, 1, 3, return_generator=True))

        self.assertEqual(self.to_set(list_paths), self.to_set(gen_paths))

    def test_generator_empty_result(self):
        """Test generator returns empty when no paths exist"""
        edges = [
            ("p", "q", 1, 1),
            ("q", "r", 1, 3),  # layer 2 missing
        ]
        df = self.mk_df(edges)
        paths_gen = enumerate_paths(df, 1, 3, return_generator=True)
        paths = list(paths_gen)
        self.assertEqual(paths, [])

    def test_generator_is_lazy(self):
        """Test that generator doesn't compute all paths upfront"""
        edges = [
            ("a", "b", 1.0, 1),
            ("b", "c", 2.0, 2),
        ]
        df = self.mk_df(edges)
        paths_gen = enumerate_paths(df, 1, 2, return_generator=True)
        # Generator should be returned without computing paths
        self.assertTrue(hasattr(paths_gen, "__iter__"))
        self.assertTrue(hasattr(paths_gen, "__next__"))


class TestCountPaths(unittest.TestCase):
    # -------------------------- helpers
    @staticmethod
    def mk_df(edges):
        return pd.DataFrame(edges, columns=["pre", "post", "weight", "layer"])

    # ------------------------ correctness
    def test_simple_chain(self):
        edges = [
            ("a", "b", 1.0, 1),
            ("b", "c", 2.0, 2),
        ]
        df = self.mk_df(edges)
        count = count_paths(df, 1, 2)
        self.assertEqual(count, 1)

    def test_branching_two_paths(self):
        edges = [
            ("a", "b", 1.0, 1),
            ("a", "d", 1.0, 1),
            ("b", "c", 1.0, 2),
            ("d", "c", 1.0, 2),
        ]
        df = self.mk_df(edges)
        count = count_paths(df, 1, 2)
        self.assertEqual(count, 2)

    def test_convergent_paths(self):
        """Test diamond pattern: a->b->d, a->c->d (2 paths)"""
        edges = [
            ("a", "b", 1.0, 1),
            ("a", "c", 1.0, 1),
            ("b", "d", 1.0, 2),
            ("c", "d", 1.0, 2),
        ]
        df = self.mk_df(edges)
        count = count_paths(df, 1, 2)
        self.assertEqual(count, 2)

    def test_complex_graph(self):
        """Test more complex graph with multiple paths"""
        edges = [
            ("a", "b", 1.0, 1),
            ("a", "c", 2.0, 1),
            ("b", "d", 3.0, 2),
            ("c", "d", 4.0, 2),
            ("c", "e", 5.0, 2),
            ("d", "f", 6.0, 3),
            ("e", "f", 7.0, 3),
        ]
        df = self.mk_df(edges)
        count = count_paths(df, 1, 3)
        # Paths: a->b->d->f, a->c->d->f, a->c->e->f
        self.assertEqual(count, 3)

    def test_default_end_layer(self):
        edges = [
            ("x", "y", 1, 1),
            ("y", "z", 1, 2),
            ("z", "w", 1, 3),
        ]
        df = self.mk_df(edges)
        count = count_paths(df)  # end_layer defaults to max
        self.assertEqual(count, 1)

    def test_no_paths_due_to_gap(self):
        edges = [
            ("p", "q", 1, 1),
            ("q", "r", 1, 3),  # layer 2 missing
        ]
        df = self.mk_df(edges)
        count = count_paths(df, 1, 3)
        self.assertEqual(count, 0)

    def test_start_equals_end_layer(self):
        edges = [
            ("m", "n", 0.5, 2),
            ("o", "p", 1.0, 2),
        ]
        df = self.mk_df(edges)
        count = count_paths(df, 2, 2)
        self.assertEqual(count, 2)

    def test_invalid_layer_order_raises(self):
        df = self.mk_df([("a", "b", 1, 1), ("b", "c", 1, 2)])
        with self.assertRaises(ValueError):
            count_paths(df, 3, 2)

    def test_count_matches_enumerate(self):
        """Verify count_paths matches the actual number of enumerated paths"""
        edges = [
            ("a", "b", 1.0, 1),
            ("a", "c", 2.0, 1),
            ("b", "d", 3.0, 2),
            ("b", "e", 3.5, 2),
            ("c", "d", 4.0, 2),
            ("c", "e", 4.5, 2),
            ("d", "f", 5.0, 3),
            ("e", "f", 6.0, 3),
        ]
        df = self.mk_df(edges)

        count = count_paths(df, 1, 3)
        paths = enumerate_paths(df, 1, 3)

        self.assertEqual(count, len(paths))

    def test_large_branching_factor(self):
        """Test with many branches to verify efficiency"""
        edges = []
        # Layer 1: single source to 3 nodes
        for i in range(3):
            edges.append(("src", f"m{i}", 1.0, 1))
        # Layer 2: each middle node to 3 targets
        for i in range(3):
            for j in range(3):
                edges.append((f"m{i}", f"t{j}", 1.0, 2))

        df = self.mk_df(edges)
        count = count_paths(df, 1, 2)
        # 3 middle nodes × 3 targets = 9 paths
        self.assertEqual(count, 9)

    def test_empty_graph(self):
        """Test with no edges"""
        df = self.mk_df([])
        count = count_paths(df, 1, 2)
        self.assertEqual(count, 0)

    def test_single_layer_multiple_edges(self):
        """Test counting paths within a single layer"""
        edges = [
            ("a", "b", 1.0, 1),
            ("c", "d", 2.0, 1),
            ("e", "f", 3.0, 1),
        ]
        df = self.mk_df(edges)
        count = count_paths(df, 1, 1)
        self.assertEqual(count, 3)
