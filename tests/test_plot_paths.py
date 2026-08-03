import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from connectome_interpreter.utils import (
    _EDGE_LABELS_FOLLOW_CURVES,
    _partition_edges_for_curvature,
    plot_paths,
)


class FakeNetwork:
    """Small PyVis stand-in for testing per-edge options."""

    last_instance = None

    def __init__(self, **kwargs):
        self.nodes = []
        self.edges = []
        self.options = None
        FakeNetwork.last_instance = self

    def from_nx(self, graph):
        self.nodes = [{"id": node} for node in graph.nodes()]
        self.edges = []
        for source, target, data in graph.edges(data=True):
            edge = {"from": source, "to": target}
            edge.update(data)
            self.edges.append(edge)

    def set_options(self, options):
        self.options = options

    def show(self, *args, **kwargs):
        return None


class TestPartitionEdgesForCurvature(unittest.TestCase):
    """A and B share a layer; B and C are reciprocally connected; D self-loops."""

    edges = [("A", "B"), ("B", "C"), ("C", "B"), ("C", "D"), ("D", "D")]
    pos = {"A": (1.0, 0.2), "B": (1.0, 0.8), "C": (2.0, 0.5), "D": (3.0, 0.5)}

    def assert_curved(self, curve_edges, expected):
        straight, curved = _partition_edges_for_curvature(
            self.edges, self.pos, curve_edges
        )
        self.assertEqual(curved, expected)
        self.assertEqual(straight, [e for e in self.edges if e not in expected])

    def test_same_layer_mode(self):
        self.assert_curved("same_layer", [("A", "B")])

    def test_reciprocal_mode(self):
        self.assert_curved("reciprocal", [("B", "C"), ("C", "B")])

    def test_overlapping_mode_covers_both(self):
        self.assert_curved("overlapping", [("A", "B"), ("B", "C"), ("C", "B")])

    def test_all_mode_keeps_self_loops_straight(self):
        self.assert_curved("all", [("A", "B"), ("B", "C"), ("C", "B"), ("C", "D")])

    def test_none_mode(self):
        self.assert_curved("none", [])


class TestPlotPathsCurvedEdges(unittest.TestCase):
    def setUp(self):
        # A and B sit in the same layer, so A -> B would be drawn vertically.
        self.paths = pd.DataFrame(
            {
                "pre": ["A", "B"],
                "post": ["B", "C"],
                "weight": [1.0, 0.5],
                "pre_layer": [1.0, 1.0],
                "post_layer": [1.0, 2.0],
            }
        )
        # B -> C and C -> B would be drawn on top of each other.
        self.reciprocal_paths = pd.DataFrame(
            {
                "pre": ["A", "B", "C"],
                "post": ["B", "C", "B"],
                "weight": [1.0, 0.5, 0.5],
                "pre_layer": [1.0, 2.0, 3.0],
                "post_layer": [2.0, 3.0, 2.0],
            }
        )

    def tearDown(self):
        plt.close("all")

    def fake_pyvis(self):
        pyvis = ModuleType("pyvis")
        pyvis_network = ModuleType("pyvis.network")
        pyvis_network.Network = FakeNetwork
        pyvis.network = pyvis_network
        return patch.dict(
            sys.modules,
            {"pyvis": pyvis, "pyvis.network": pyvis_network},
        )

    @patch("connectome_interpreter.utils.nx.draw")
    @patch("connectome_interpreter.utils.nx.draw_networkx_edges")
    def test_static_same_layer_edges_use_requested_curvature(self, draw_edges, draw):
        plot_paths(
            self.paths,
            show=False,
            edge_text=False,
            node_text=False,
            edge_curvature=0.4,
            seed=0,
        )

        self.assertEqual(draw.call_args.kwargs["edgelist"], [("B", "C")])
        draw_edges.assert_called_once()
        self.assertEqual(draw_edges.call_args.kwargs["edgelist"], [("A", "B")])
        self.assertEqual(draw_edges.call_args.kwargs["connectionstyle"], "arc3,rad=0.4")

    @patch("connectome_interpreter.utils.nx.draw")
    @patch("connectome_interpreter.utils.nx.draw_networkx_edges")
    def test_static_reciprocal_edges_are_curved_by_default_mode(self, draw_edges, draw):
        plot_paths(
            self.reciprocal_paths,
            show=False,
            edge_text=False,
            node_text=False,
            edge_curvature=0.4,
            seed=0,
        )

        self.assertEqual(draw.call_args.kwargs["edgelist"], [("A", "B")])
        self.assertEqual(
            sorted(draw_edges.call_args.kwargs["edgelist"]),
            [("B", "C"), ("C", "B")],
        )

    @patch("connectome_interpreter.utils.nx.draw")
    @patch("connectome_interpreter.utils.nx.draw_networkx_edges")
    def test_static_curve_edges_mode_restricts_selection(self, draw_edges, draw):
        plot_paths(
            self.reciprocal_paths,
            show=False,
            edge_text=False,
            node_text=False,
            edge_curvature=0.4,
            curve_edges="same_layer",
            seed=0,
        )

        draw_edges.assert_not_called()
        self.assertEqual(len(draw.call_args.kwargs["edgelist"]), 3)

    @patch("connectome_interpreter.utils.nx.draw")
    @patch("connectome_interpreter.utils.nx.draw_networkx_edges")
    @patch("connectome_interpreter.utils.nx.draw_networkx_edge_labels")
    def test_static_curved_labels_follow_curved_edges(
        self, draw_edge_labels, draw_edges, draw
    ):
        plot_paths(
            self.paths,
            show=False,
            edge_text=True,
            node_text=False,
            edge_curvature=0.4,
            seed=0,
        )

        curved_call = next(
            call
            for call in draw_edge_labels.call_args_list
            if ("A", "B") in call.kwargs["edge_labels"]
        )
        if _EDGE_LABELS_FOLLOW_CURVES:
            self.assertEqual(curved_call.kwargs["connectionstyle"], "arc3,rad=0.4")
        else:
            # older networkx cannot route labels along arcs, so the kwarg is omitted
            self.assertNotIn("connectionstyle", curved_call.kwargs)

    @patch("connectome_interpreter.utils.nx.draw")
    @patch("connectome_interpreter.utils.nx.draw_networkx_edges")
    def test_default_curvature_uses_straight_rendering(self, draw_edges, draw):
        plot_paths(
            self.paths,
            show=False,
            edge_text=False,
            node_text=False,
            seed=0,
        )

        draw_edges.assert_not_called()
        self.assertEqual(draw.call_args.kwargs["edgelist"], [("A", "B"), ("B", "C")])

    def test_interactive_curved_edges_use_requested_curvature(self):
        with self.fake_pyvis():
            plot_paths(
                self.paths,
                interactive=True,
                edge_text=False,
                node_text=False,
                edge_curvature=0.4,
                seed=0,
            )

        edges = {
            (edge["from"], edge["to"]): edge for edge in FakeNetwork.last_instance.edges
        }
        self.assertEqual(
            edges[("A", "B")]["smooth"],
            {"enabled": True, "type": "curvedCW", "roundness": 0.4},
        )
        self.assertNotIn("smooth", edges[("B", "C")])

    def test_negative_curvature_bends_interactive_edges_counterclockwise(self):
        with self.fake_pyvis():
            plot_paths(
                self.paths,
                interactive=True,
                edge_text=False,
                node_text=False,
                edge_curvature=-0.4,
                seed=0,
            )

        curved_edge = next(
            edge
            for edge in FakeNetwork.last_instance.edges
            if (edge["from"], edge["to"]) == ("A", "B")
        )
        self.assertEqual(
            curved_edge["smooth"],
            {"enabled": True, "type": "curvedCCW", "roundness": 0.4},
        )

    def test_curvature_must_be_between_negative_and_positive_one(self):
        for curvature in (-1.1, 1.1):
            with self.subTest(curvature=curvature), self.assertRaises(ValueError):
                plot_paths(
                    self.paths,
                    edge_curvature=curvature,
                )

    def test_unknown_curve_edges_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            plot_paths(
                self.paths,
                edge_curvature=0.4,
                curve_edges="between_layers",
            )


if __name__ == "__main__":
    unittest.main()
