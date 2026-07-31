import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from connectome_interpreter.utils import (
    _partition_edges_by_x_position,
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


class TestPlotPathsSameLayerEdges(unittest.TestCase):
    def setUp(self):
        self.paths = pd.DataFrame(
            {
                "pre": ["A", "B"],
                "post": ["B", "C"],
                "weight": [1.0, 0.5],
                "pre_layer": [1.0, 1.0],
                "post_layer": [1.0, 2.0],
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

    def test_partition_keeps_self_loops_with_ordinary_edges(self):
        edges = [("A", "B"), ("B", "C"), ("C", "C")]
        pos = {"A": (1.0, 0.2), "B": (1.0, 0.8), "C": (2.0, 0.5)}

        ordinary, same_layer = _partition_edges_by_x_position(edges, pos)

        self.assertEqual(ordinary, [("B", "C"), ("C", "C")])
        self.assertEqual(same_layer, [("A", "B")])

    @patch("connectome_interpreter.utils.nx.draw")
    @patch("connectome_interpreter.utils.nx.draw_networkx_edges")
    def test_static_same_layer_edges_use_requested_curvature(self, draw_edges, draw):
        plot_paths(
            self.paths,
            show=False,
            edge_text=False,
            node_text=False,
            same_layer_curvature=0.4,
            seed=0,
        )

        self.assertEqual(draw.call_args.kwargs["edgelist"], [("B", "C")])
        draw_edges.assert_called_once()
        self.assertEqual(draw_edges.call_args.kwargs["edgelist"], [("A", "B")])
        self.assertEqual(draw_edges.call_args.kwargs["connectionstyle"], "arc3,rad=0.4")

    @patch("connectome_interpreter.utils.nx.draw")
    @patch("connectome_interpreter.utils.nx.draw_networkx_edges")
    @patch("connectome_interpreter.utils.nx.draw_networkx_edge_labels")
    def test_static_same_layer_labels_follow_curved_edges(
        self, draw_edge_labels, draw_edges, draw
    ):
        plot_paths(
            self.paths,
            show=False,
            edge_text=True,
            node_text=False,
            same_layer_curvature=0.4,
            seed=0,
        )

        same_layer_call = next(
            call
            for call in draw_edge_labels.call_args_list
            if ("A", "B") in call.kwargs["edge_labels"]
        )
        self.assertEqual(same_layer_call.kwargs["connectionstyle"], "arc3,rad=0.4")

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

    def test_interactive_same_layer_edges_use_requested_curvature(self):
        with self.fake_pyvis():
            plot_paths(
                self.paths,
                interactive=True,
                edge_text=False,
                node_text=False,
                same_layer_curvature=0.4,
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
                same_layer_curvature=-0.4,
                seed=0,
            )

        same_layer_edge = next(
            edge
            for edge in FakeNetwork.last_instance.edges
            if (edge["from"], edge["to"]) == ("A", "B")
        )
        self.assertEqual(
            same_layer_edge["smooth"],
            {"enabled": True, "type": "curvedCCW", "roundness": 0.4},
        )

    def test_curvature_must_be_between_negative_and_positive_one(self):
        for curvature in (-1.1, 1.1):
            with self.subTest(curvature=curvature), self.assertRaises(ValueError):
                plot_paths(
                    self.paths,
                    same_layer_curvature=curvature,
                )


if __name__ == "__main__":
    unittest.main()
