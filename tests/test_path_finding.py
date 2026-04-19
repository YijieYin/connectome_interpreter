import itertools
import random
import unittest

import pandas as pd

from connectome_interpreter.path_finding import (
    find_xor,
    find_paths_of_length,
    enumerate_paths,
    find_shortest_paths,
    count_paths,
    group_paths,
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


class TestFindShortestPaths(unittest.TestCase):

    def setUp(self):
        # Create a simple graph with known shortest paths
        # Graph structure:
        #   A -> B (0.5)
        #   A -> C (0.3)
        #   B -> D (0.4)
        #   C -> D (0.6)
        #   C -> E (0.7)
        #   D -> E (0.8)
        # Expected shortest paths (by weight):
        #   A -> D: A -> B -> D (product: 0.5 * 0.4 = 0.2)
        #   A -> E: A -> C -> E (product: 0.3 * 0.7 = 0.21)

        self.simple_paths = pd.DataFrame(
            {
                "pre": ["A", "A", "B", "C", "C", "D"],
                "post": ["B", "C", "D", "D", "E", "E"],
                "weight": [0.5, 0.3, 0.4, 0.6, 0.7, 0.8],
            }
        )

    def test_single_start_single_end(self):
        # Test finding a single shortest path
        result = find_shortest_paths(
            self.simple_paths, start_nodes=["A"], end_nodes=["D"]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ["A", "B", "D"])

    def test_single_start_multiple_ends(self):
        # Test finding multiple shortest paths from one start node
        result = find_shortest_paths(
            self.simple_paths, start_nodes=["A"], end_nodes=["D", "E"]
        )

        self.assertEqual(len(result), 2)
        # Check that both paths start with 'A'
        self.assertTrue(all(path[0] == "A" for path in result))
        # Check that paths end with correct nodes
        end_nodes = {path[-1] for path in result}
        self.assertEqual(end_nodes, {"D", "E"})

    def test_multiple_starts_single_end(self):
        # Test finding paths from multiple start nodes to one end
        result = find_shortest_paths(
            self.simple_paths, start_nodes=["A", "B"], end_nodes=["E"]
        )

        self.assertEqual(len(result), 2)
        # Check that we have paths from both A and B
        start_nodes = {path[0] for path in result}
        self.assertEqual(start_nodes, {"A", "B"})

    def test_multiple_starts_multiple_ends(self):
        # Test finding all combinations of start and end nodes
        result = find_shortest_paths(
            self.simple_paths, start_nodes=["A", "B"], end_nodes=["D", "E"]
        )

        # Should have 4 paths: A->D, A->E, B->D, B->E
        self.assertEqual(len(result), 4)

    def test_no_path_exists(self):
        # Test case where no path exists
        result = find_shortest_paths(
            self.simple_paths,
            start_nodes=["D"],
            end_nodes=["A"],  # No path from D back to A
        )

        self.assertEqual(len(result), 0)

    def test_start_equals_end(self):
        # Test that paths from a node to itself are excluded
        result = find_shortest_paths(
            self.simple_paths, start_nodes=["A", "B"], end_nodes=["A", "B"]
        )

        # Should only have A->B and B->A (if exists, but B->A doesn't)
        # So we should get no self-loops
        self.assertTrue(all(path[0] != path[-1] for path in result))

    def test_nonexistent_start_node(self):
        # Test with a start node that doesn't exist in the graph
        result = find_shortest_paths(
            self.simple_paths, start_nodes=["Z"], end_nodes=["D"]  # Doesn't exist
        )

        self.assertEqual(len(result), 0)

    def test_nonexistent_end_node(self):
        # Test with an end node that doesn't exist in the graph
        result = find_shortest_paths(
            self.simple_paths, start_nodes=["A"], end_nodes=["Z"]  # Doesn't exist
        )

        self.assertEqual(len(result), 0)

    def test_empty_paths_dataframe(self):
        # Test with an empty DataFrame
        empty_paths = pd.DataFrame(columns=["pre", "post", "weight"])
        result = find_shortest_paths(empty_paths, start_nodes=["A"], end_nodes=["B"])

        self.assertEqual(len(result), 0)

    def test_direct_connection(self):
        # Test that direct connections are found correctly
        result = find_shortest_paths(
            self.simple_paths, start_nodes=["A"], end_nodes=["B"]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ["A", "B"])

    def test_complex_graph(self):
        # Create a more complex graph to test robustness
        complex_paths = pd.DataFrame(
            {
                "pre": ["A", "A", "B", "B", "C", "D", "E", "F"],
                "post": ["B", "C", "D", "E", "D", "F", "F", "G"],
                "weight": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            }
        )

        result = find_shortest_paths(complex_paths, start_nodes=["A"], end_nodes=["G"])

        # Should find at least one path from A to G
        self.assertGreater(len(result), 0)
        # All paths should start with A and end with G
        for path in result:
            self.assertEqual(path[0], "A")
            self.assertEqual(path[-1], "G")

    def test_path_ordering(self):
        # Verify that paths are ordered correctly (start to end)
        result = find_shortest_paths(
            self.simple_paths, start_nodes=["A"], end_nodes=["E"]
        )

        self.assertEqual(len(result), 1)
        path = result[0]
        # Check that each consecutive pair exists in the original paths
        for i in range(len(path) - 1):
            edge_exists = any(
                (self.simple_paths["pre"] == path[i])
                & (self.simple_paths["post"] == path[i + 1])
            )
            self.assertTrue(
                edge_exists, f"Edge {path[i]} -> {path[i+1]} not found in graph"
            )


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

    def test_include_loops_true(self):
        """Test that loop_mode='allow' counts all paths including loops"""
        edges = [
            ("a", "b", 1.0, 1),
            ("b", "a", 1.0, 2),  # Creates a loop back to 'a'
            ("a", "c", 1.0, 3),
        ]
        df = self.mk_df(edges)
        count = count_paths(df, 1, 3, loop_mode="allow")
        # Path: a->b->a->c (this has a loop)
        self.assertEqual(count, 1)

    def test_include_loops_false_with_loop(self):
        """Test that loop_mode='exclude' excludes paths with loops in the middle"""
        edges = [
            ("a", "b", 1.0, 1),
            ("b", "a", 1.0, 2),  # Creates a loop back to 'a'
            ("a", "c", 1.0, 3),
        ]
        df = self.mk_df(edges)
        count = count_paths(df, 1, 3, loop_mode="exclude")
        # Path a->b->a->c should be excluded (a appears at start and in middle)
        self.assertEqual(count, 0)

    def test_include_loops_false_start_equals_end_allowed(self):
        """Test that a node appearing at start and end is allowed (not a loop)"""
        edges = [
            ("a", "b", 1.0, 1),
            ("b", "c", 1.0, 2),
            ("c", "a", 1.0, 3),  # Returns to 'a' at the end
        ]
        df = self.mk_df(edges)
        count = count_paths(df, 1, 3, loop_mode="exclude")
        # Path: a->b->c->a is allowed (a only at start and end)
        self.assertEqual(count, 1)

    def test_include_loops_false_multiple_same_node_at_end(self):
        """Test A-B-C-A-A case: loop because A appears in middle then again"""
        edges = [
            ("a", "b", 1.0, 1),
            ("b", "c", 1.0, 2),
            ("c", "a", 1.0, 3),
            ("a", "a", 1.0, 4),  # A appears again after returning
        ]
        df = self.mk_df(edges)
        count = count_paths(df, 1, 4, loop_mode="exclude")
        # Path a->b->c->a->a has a loop (a appears at positions 0, 3, 4)
        self.assertEqual(count, 0)

    def test_include_loops_false_no_loop(self):
        """Test that loop_mode='exclude' allows paths without loops"""
        edges = [
            ("a", "b", 1.0, 1),
            ("b", "c", 1.0, 2),
            ("c", "d", 1.0, 3),
        ]
        df = self.mk_df(edges)
        count_with_loops = count_paths(df, 1, 3, loop_mode="allow")
        count_without_loops = count_paths(df, 1, 3, loop_mode="exclude")
        # No loops exist, so counts should be the same
        self.assertEqual(count_with_loops, count_without_loops)
        self.assertEqual(count_without_loops, 1)

    def test_include_loops_false_diamond_pattern(self):
        """Test diamond pattern without loops is counted correctly"""
        edges = [
            ("a", "b", 1.0, 1),
            ("a", "c", 1.0, 1),
            ("b", "d", 1.0, 2),
            ("c", "d", 1.0, 2),
        ]
        df = self.mk_df(edges)
        count = count_paths(df, 1, 2, loop_mode="exclude")
        # Two paths: a->b->d and a->c->d, neither has loops
        self.assertEqual(count, 2)

    def test_include_loops_complex_with_some_loops(self):
        """Test complex graph where some paths have loops and some don't"""
        edges = [
            ("a", "b", 1.0, 1),
            ("a", "c", 1.0, 1),
            ("b", "d", 1.0, 2),
            ("c", "d", 1.0, 2),
            ("d", "b", 1.0, 3),  # Creates loop back to b
            ("d", "e", 1.0, 3),  # Non-loop path
        ]
        df = self.mk_df(edges)
        count_with_loops = count_paths(df, 1, 3, loop_mode="allow")
        count_without_loops = count_paths(df, 1, 3, loop_mode="exclude")
        # With loops: a->b->d->b, a->b->d->e, a->c->d->b, a->c->d->e = 4
        # Without loops:
        #   - a->b->d->b has loop (b appears at position 1 and 3)
        #   - a->b->d->e is OK
        #   - a->c->d->b is OK (b only at end)
        #   - a->c->d->e is OK
        # So 3 paths without loops
        self.assertEqual(count_with_loops, 4)
        self.assertEqual(count_without_loops, 3)

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

    def test_return_both_simple(self):
        """Test loop_mode='both' with a simple graph"""
        edges = [
            ("a", "b", 1.0, 1),
            ("b", "c", 2.0, 2),
        ]
        df = self.mk_df(edges)
        count_with, count_without = count_paths(df, 1, 2, loop_mode="both")
        self.assertEqual(count_with, 1)
        self.assertEqual(count_without, 1)

    def test_return_both_with_loops(self):
        """Test loop_mode='both' returns different counts when loops exist"""
        edges = [
            ("a", "b", 1.0, 1),
            ("a", "c", 1.0, 1),
            ("b", "d", 1.0, 2),
            ("c", "d", 1.0, 2),
            ("d", "b", 1.0, 3),  # Creates loop back to b
            ("d", "e", 1.0, 3),  # Non-loop path
        ]
        df = self.mk_df(edges)
        count_with, count_without = count_paths(df, 1, 3, loop_mode="both")
        # Verify it matches individual calls
        count_with_only = count_paths(df, 1, 3, loop_mode="allow")
        count_without_only = count_paths(df, 1, 3, loop_mode="exclude")
        self.assertEqual(count_with, count_with_only)
        self.assertEqual(count_without, count_without_only)
        self.assertEqual(count_with, 4)
        self.assertEqual(count_without, 3)

    def test_return_both_matches_separate_calls(self):
        """Verify loop_mode='both' gives same results as two separate calls"""
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

        # Get both counts in one call
        count_with, count_without = count_paths(df, 1, 3, loop_mode="both")

        # Get counts separately
        count_with_separate = count_paths(df, 1, 3, loop_mode="allow")
        count_without_separate = count_paths(df, 1, 3, loop_mode="exclude")

        self.assertEqual(count_with, count_with_separate)
        self.assertEqual(count_without, count_without_separate)

    def test_return_both_default_false(self):
        """Test that loop_mode defaults to 'allow' and returns int"""
        edges = [
            ("a", "b", 1.0, 1),
            ("b", "c", 2.0, 2),
        ]
        df = self.mk_df(edges)
        count = count_paths(df, 1, 2)
        # Should return int, not tuple
        self.assertIsInstance(count, int)
        self.assertEqual(count, 1)

    def test_invalid_loop_mode(self):
        """Test that invalid loop_mode raises ValueError"""
        edges = [
            ("a", "b", 1.0, 1),
            ("b", "c", 2.0, 2),
        ]
        df = self.mk_df(edges)
        with self.assertRaises(ValueError):
            count_paths(df, 1, 2, loop_mode="invalid")


class TestGroupPaths(unittest.TestCase):
    # ------------------------------------------------------------------ helpers
    @staticmethod
    def mk_df(edges, cols=None):
        """Build a paths DataFrame from a list of tuples."""
        if cols is None:
            cols = ["pre", "post", "weight"]
        return pd.DataFrame(edges, columns=cols)

    @staticmethod
    def weight_of(df, pre, post, layer=None):
        """Fetch the weight for a given (pre, post[, layer]) row, or None."""
        mask = (df["pre"] == pre) & (df["post"] == post)
        if layer is not None:
            mask &= df["layer"] == layer
        row = df[mask]
        return None if len(row) == 0 else row["weight"].iloc[0]

    # --------------------------------------------------------------- edge cases
    def test_empty_paths_returns_unchanged(self):
        empty = pd.DataFrame(columns=["pre", "post", "weight"])
        out = group_paths(empty)
        self.assertEqual(out.shape[0], 0)

    def test_none_paths_returns_none(self):
        self.assertIsNone(group_paths(None))

    def test_invalid_combining_method_raises(self):
        df = self.mk_df([("a", "b", 1.0)])
        with self.assertRaises(AssertionError):
            group_paths(df, combining_method="max")

    def test_valid_combining_methods_accepted(self):
        df = self.mk_df([("a", "b", 1.0)])
        for method in ("sum", "mean", "median"):
            group_paths(df, combining_method=method)  # should not raise

    def test_none_groups_default_to_identity(self):
        """With pre_group=post_group=None, nodes map to themselves."""
        df = self.mk_df([("a", "b", 1.0), ("c", "d", 2.0)])
        out = group_paths(df)
        self.assertAlmostEqual(self.weight_of(out, "a", "b"), 1.0)
        self.assertAlmostEqual(self.weight_of(out, "c", "d"), 2.0)

    # -------------------------------------------- basic grouping (outprop=False)
    def test_outprop_false_sums_within_pre_type(self):
        """Weights from different pre of same type → same post are summed first."""
        df = self.mk_df([("a1", "b1", 1.0), ("a2", "b1", 2.0), ("a1", "b2", 3.0)])
        out = group_paths(
            df,
            pre_group={"a1": "A", "a2": "A"},
            post_group={"b1": "B", "b2": "B"},
            combining_method="sum",
            avg_within_connected=True,
        )
        # step 1 (sum within pre_type per post): b1=3, b2=3
        # step 2 (sum across posts of same type): 3+3 = 6
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(self.weight_of(out, "A", "B"), 6.0)

    def test_outprop_false_mean_across_post_type(self):
        df = self.mk_df([("a1", "b1", 2.0), ("a1", "b2", 4.0)])
        out = group_paths(
            df,
            pre_group={"a1": "A"},
            post_group={"b1": "B", "b2": "B"},
            combining_method="mean",
            avg_within_connected=True,
        )
        # mean(2, 4) = 3
        self.assertAlmostEqual(self.weight_of(out, "A", "B"), 3.0)

    def test_outprop_false_median_across_post_type(self):
        df = self.mk_df([("a1", "b1", 1.0), ("a1", "b2", 3.0), ("a1", "b3", 100.0)])
        out = group_paths(
            df,
            pre_group={"a1": "A"},
            post_group={"b1": "B", "b2": "B", "b3": "B"},
            combining_method="median",
            avg_within_connected=True,
        )
        # median(1, 3, 100) = 3
        self.assertAlmostEqual(self.weight_of(out, "A", "B"), 3.0)

    # --------------------------------------------- basic grouping (outprop=True)
    def test_outprop_true_sum(self):
        """outprop=True: sum over posts of same type per pre, then sum across pre."""
        df = self.mk_df([("a1", "b1", 1.0), ("a1", "b2", 3.0), ("a2", "b1", 2.0)])
        out = group_paths(
            df,
            pre_group={"a1": "A", "a2": "A"},
            post_group={"b1": "B", "b2": "B"},
            outprop=True,
            combining_method="sum",
            avg_within_connected=True,
        )
        # per pre summed over posts of type B: a1=4, a2=2; total sum = 6
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(self.weight_of(out, "A", "B"), 6.0)

    def test_outprop_true_mean(self):
        df = self.mk_df([("a1", "b1", 1.0), ("a1", "b2", 3.0), ("a2", "b1", 2.0)])
        out = group_paths(
            df,
            pre_group={"a1": "A", "a2": "A"},
            post_group={"b1": "B", "b2": "B"},
            outprop=True,
            combining_method="mean",
            avg_within_connected=True,
        )
        # per pre: a1=4, a2=2; mean across pre = 3
        self.assertAlmostEqual(self.weight_of(out, "A", "B"), 3.0)

    def test_outprop_true_median(self):
        df = self.mk_df([("a1", "b1", 1.0), ("a2", "b1", 3.0), ("a3", "b1", 100.0)])
        out = group_paths(
            df,
            pre_group={"a1": "A", "a2": "A", "a3": "A"},
            post_group={"b1": "B"},
            outprop=True,
            combining_method="median",
            avg_within_connected=True,
        )
        # median(1, 3, 100) = 3
        self.assertAlmostEqual(self.weight_of(out, "A", "B"), 3.0)

    # ------------------------------------------- avg_within_connected distinction
    def test_avg_within_connected_true_ignores_disconnected(self):
        """avg_within_connected=True averages only over actually-connected neurons."""
        # b1 connected to A; b2 connected only to C (different pre_type)
        df = self.mk_df([("a1", "b1", 2.0), ("c1", "b2", 4.0)])
        out = group_paths(
            df,
            pre_group={"a1": "A", "c1": "C"},
            post_group={"b1": "B", "b2": "B"},
            combining_method="mean",
            avg_within_connected=True,
        )
        # (A, B): only b1 connected to any A neuron → mean = 2
        # (C, B): only b2 connected to any C neuron → mean = 4
        self.assertAlmostEqual(self.weight_of(out, "A", "B"), 2.0)
        self.assertAlmostEqual(self.weight_of(out, "C", "B"), 4.0)

    def test_avg_within_connected_false_fills_zero_for_disconnected(self):
        """avg_within_connected=False: every post of the type counts, missing=0.

        Per docstring: 'combined across all postsynaptic neurons of the same group
        (even if some postsynaptic neurons are not in paths)'.
        """
        df = self.mk_df([("a1", "b1", 2.0), ("c1", "b2", 4.0)])
        out = group_paths(
            df,
            pre_group={"a1": "A", "c1": "C"},
            post_group={"b1": "B", "b2": "B"},
            combining_method="mean",
            avg_within_connected=False,
        )
        # (A, B): b1 contributes 2, b2 contributes 0 → (2+0)/2 = 1
        # (C, B): b1 contributes 0, b2 contributes 4 → (0+4)/2 = 2
        self.assertAlmostEqual(self.weight_of(out, "A", "B"), 1.0)
        self.assertAlmostEqual(self.weight_of(out, "C", "B"), 2.0)

    def test_avg_within_connected_false_outprop_true(self):
        """Mirror of above for outprop=True (average across all pre of type)."""
        df = self.mk_df([("a1", "b1", 2.0), ("a2", "c1", 4.0)])
        out = group_paths(
            df,
            pre_group={"a1": "A", "a2": "A"},
            post_group={"b1": "B", "c1": "C"},
            outprop=True,
            combining_method="mean",
            avg_within_connected=False,
        )
        # (A, B): a1→b1=2, a2→(B)=0 → mean = 1
        # (A, C): a1→(C)=0, a2→c1=4 → mean = 2
        self.assertAlmostEqual(self.weight_of(out, "A", "B"), 1.0)
        self.assertAlmostEqual(self.weight_of(out, "A", "C"), 2.0)

    def test_avg_within_connected_false_includes_group_member_absent_from_paths(self):
        """A neuron in the group but never appearing in paths must still be counted."""
        # b2 is in post_group B but has no edges at all
        df = self.mk_df([("a1", "b1", 4.0)])
        out = group_paths(
            df,
            pre_group={"a1": "A"},
            post_group={"b1": "B", "b2": "B"},
            combining_method="mean",
            avg_within_connected=False,
        )
        # Per docstring: average over b1 and b2 even though b2 absent → (4+0)/2 = 2
        self.assertAlmostEqual(self.weight_of(out, "A", "B"), 2.0)

    # ------------------------------------------------------------ layer handling
    def test_layer_first_layer_uses_pre_group(self):
        """At the minimum layer, pre is grouped by pre_group (not intermediate)."""
        df = self.mk_df(
            [("a1", "m", 1.0, 1), ("a2", "m", 2.0, 1), ("m", "b", 3.0, 2)],
            cols=["pre", "post", "weight", "layer"],
        )
        out = group_paths(
            df,
            pre_group={"a1": "A", "a2": "A"},
            post_group={"b": "B"},
            intermediate_group={"a1": "X", "a2": "X", "m": "M", "b": "Y"},
            combining_method="sum",
            avg_within_connected=True,
        )
        layer1 = out[out["layer"] == 1]
        self.assertTrue((layer1["pre"] == "A").all())  # pre_group, not "X"
        self.assertTrue((layer1["post"] == "M").all())  # intermediate_group

    def test_layer_last_layer_uses_post_group(self):
        """At the maximum layer, post is grouped by post_group (not intermediate)."""
        df = self.mk_df(
            [("a", "m", 1.0, 1), ("m", "b1", 2.0, 2), ("m", "b2", 3.0, 2)],
            cols=["pre", "post", "weight", "layer"],
        )
        out = group_paths(
            df,
            pre_group={"a": "A"},
            post_group={"b1": "B", "b2": "B"},
            intermediate_group={"a": "X", "m": "M", "b1": "Y", "b2": "Y"},
            combining_method="sum",
            avg_within_connected=True,
        )
        layer2 = out[out["layer"] == 2]
        self.assertTrue((layer2["post"] == "B").all())  # post_group, not "Y"
        self.assertTrue((layer2["pre"] == "M").all())  # intermediate_group

    def test_layer_middle_uses_intermediate_group(self):
        """Non-first, non-last layer nodes are grouped by intermediate_group."""
        df = self.mk_df(
            [
                ("a", "m1", 1.0, 1),
                ("m1", "m2", 1.0, 2),  # middle layer
                ("m2", "b", 1.0, 3),
            ],
            cols=["pre", "post", "weight", "layer"],
        )
        out = group_paths(
            df,
            pre_group={"a": "A"},
            post_group={"b": "B"},
            intermediate_group={"m1": "M", "m2": "M", "a": "A", "b": "B"},
            combining_method="sum",
            avg_within_connected=True,
        )
        middle = out[out["layer"] == 2]
        self.assertTrue((middle["pre"] == "M").all())
        self.assertTrue((middle["post"] == "M").all())

    def test_intermediate_group_defaults_to_pre_group(self):
        """If intermediate_group is None, it should fall back to pre_group."""
        df = self.mk_df(
            [("a", "m", 1.0, 1), ("m", "b", 2.0, 2)],
            cols=["pre", "post", "weight", "layer"],
        )
        out = group_paths(
            df,
            pre_group={"a": "A", "m": "M"},
            post_group={"b": "B"},
            combining_method="sum",
            avg_within_connected=True,
        )
        # 'm' at layer-1-post is intermediate → mapped via pre_group → "M"
        self.assertEqual(self.weight_of(out, "A", "M", layer=1), 1.0)
        # 'm' at layer-2-pre is intermediate → mapped via pre_group → "M"
        self.assertEqual(self.weight_of(out, "M", "B", layer=2), 2.0)

    # ----------------------------------------------- activation column handling
    def test_activation_columns_averaged_within_groups(self):
        """pre_activation / post_activation should be mean-aggregated per group."""
        df = self.mk_df(
            [
                ("a1", "b1", 1.0, 0.2, 0.8),
                ("a2", "b1", 2.0, 0.4, 0.8),
            ],
            cols=["pre", "post", "weight", "pre_activation", "post_activation"],
        )
        out = group_paths(
            df,
            pre_group={"a1": "A", "a2": "A"},
            post_group={"b1": "B"},
            combining_method="sum",
            avg_within_connected=True,
        )
        row = out[(out["pre"] == "A") & (out["post"] == "B")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(row["pre_activation"].iloc[0], 0.3)  # mean(0.2, 0.4)
        self.assertAlmostEqual(row["post_activation"].iloc[0], 0.8)

    # -------------------------------------------------------------- output shape
    def test_output_columns_renamed(self):
        """Final output uses 'pre'/'post', not 'pre_type'/'post_type'."""
        df = self.mk_df([("a", "b", 1.0)])
        out = group_paths(df, pre_group={"a": "A"}, post_group={"b": "B"})
        self.assertIn("pre", out.columns)
        self.assertIn("post", out.columns)
        self.assertNotIn("pre_type", out.columns)
        self.assertNotIn("post_type", out.columns)

    def test_output_has_no_duplicate_columns(self):
        """Grouped output should have a single 'post' column, not a leftover one."""
        df = self.mk_df([("a1", "b1", 1.0), ("a1", "b2", 3.0)])
        out = group_paths(
            df,
            pre_group={"a1": "A"},
            post_group={"b1": "B", "b2": "B"},
            combining_method="mean",
        )
        self.assertEqual(list(out.columns).count("pre"), 1)
        self.assertEqual(list(out.columns).count("post"), 1)

    def test_output_one_row_per_group_pair(self):
        """After grouping, each (pre_type, post_type) combo appears exactly once."""
        df = self.mk_df([("a1", "b1", 1.0), ("a1", "b2", 2.0), ("a2", "b1", 3.0)])
        out = group_paths(
            df,
            pre_group={"a1": "A", "a2": "A"},
            post_group={"b1": "B", "b2": "B"},
            combining_method="sum",
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["pre"], "A")
        self.assertEqual(out.iloc[0]["post"], "B")

    def test_avg_within_connected_false_zero_fills_per_layer(self):
        """With a layer column, zero-filling must happen per (layer, types) group."""
        # 'a' connects to b1 at layer 1 and to b2 at layer 2 only.
        df = self.mk_df(
            [("a", "b1", 2.0, 1), ("a", "b2", 3.0, 2)],
            cols=["pre", "post", "weight", "layer"],
        )
        out = group_paths(
            df,
            pre_group={"a": "A"},
            post_group={"b1": "B", "b2": "B"},
            intermediate_group={"a": "A", "b1": "B", "b2": "B"},  # <-- added
            combining_method="mean",
            avg_within_connected=False,
        )
        # layer 1: b1 contributes 2, b2 absent → 0; mean = 1.0
        # layer 2: b1 absent → 0, b2 contributes 3; mean = 1.5
        self.assertAlmostEqual(self.weight_of(out, "A", "B", layer=1), 1.0)
        self.assertAlmostEqual(self.weight_of(out, "A", "B", layer=2), 1.5)

    def test_missing_post_group_key_becomes_own_group(self):
        """Post neurons absent from post_group fall back to their own id as group."""
        df = self.mk_df([("a1", "b1", 2.0), ("a1", "b2", 4.0)])
        out = group_paths(
            df,
            pre_group={"a1": "A"},
            post_group={"b1": "B"},  # 'b2' missing
            combining_method="sum",
            avg_within_connected=True,
        )
        self.assertAlmostEqual(self.weight_of(out, "A", "B"), 2.0)
        self.assertAlmostEqual(self.weight_of(out, "A", "b2"), 4.0)

    def test_missing_pre_group_key_outprop(self):
        """Pre neurons absent from pre_group fall back to their own id under outprop."""
        df = self.mk_df([("a1", "b", 2.0), ("a2", "b", 4.0)])
        out = group_paths(
            df,
            pre_group={"a1": "A"},  # 'a2' missing
            post_group={"b": "B"},
            outprop=True,
            combining_method="sum",
            avg_within_connected=True,
        )
        self.assertAlmostEqual(self.weight_of(out, "A", "B"), 2.0)
        self.assertAlmostEqual(self.weight_of(out, "a2", "B"), 4.0)

    def test_missing_intermediate_group_key(self):
        """Intermediate nodes absent from intermediate_group fall back to their own id."""
        df = self.mk_df(
            [("a", "m", 1.0, 1), ("m", "n", 2.0, 2), ("n", "b", 3.0, 3)],
            cols=["pre", "post", "weight", "layer"],
        )
        out = group_paths(
            df,
            pre_group={"a": "A"},
            post_group={"b": "B"},
            intermediate_group={"n": "N"},  # 'm' missing
            combining_method="sum",
            avg_within_connected=True,
        )
        self.assertEqual(self.weight_of(out, "A", "m", layer=1), 1.0)
        self.assertEqual(self.weight_of(out, "m", "N", layer=2), 2.0)
        self.assertEqual(self.weight_of(out, "N", "B", layer=3), 3.0)

    def test_missing_key_still_zero_fills_known_group(self):
        """A missing post key shouldn't perturb zero-filling of known groups."""
        df = self.mk_df([("a1", "b1", 2.0), ("a2", "b1", 3.0), ("a1", "b2", 4.0)])
        out = group_paths(
            df,
            pre_group={"a1": "A", "a2": "A"},
            post_group={"b1": "B"},  # 'b2' missing → own group
            combining_method="mean",
            avg_within_connected=False,
        )
        # Group B only contains b1; a1+a2 sum to 5 on b1 → mean(5) = 5
        self.assertAlmostEqual(self.weight_of(out, "A", "B"), 5.0)
        # Own-group b2 holds weight 4
        self.assertAlmostEqual(self.weight_of(out, "A", "b2"), 4.0)
