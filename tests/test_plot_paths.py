import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from connectome_interpreter.utils import (
    _partition_edges_by_x_position,
    plot_paths,
)


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

    def test_partition_keeps_self_loops_with_ordinary_edges(self):
        edges = [("A", "B"), ("B", "C"), ("C", "C")]
        pos = {"A": (1.0, 0.2), "B": (1.0, 0.8), "C": (2.0, 0.5)}

        ordinary, same_layer = _partition_edges_by_x_position(edges, pos)

        self.assertEqual(ordinary, [("B", "C"), ("C", "C")])
        self.assertEqual(same_layer, [("A", "B")])

    def test_same_layer_edges_use_requested_curvature(self):
        with patch(
            "connectome_interpreter.utils.nx.draw_networkx_edges",
            wraps=nx.draw_networkx_edges,
        ) as draw_edges:
            plot_paths(
                self.paths,
                show=False,
                edge_text=False,
                node_text=False,
                same_layer_edge_curvature=0.4,
                seed=0,
            )

        draw_edges.assert_called_once()
        self.assertEqual(draw_edges.call_args.kwargs["edgelist"], [("A", "B")])
        self.assertEqual(
            draw_edges.call_args.kwargs["connectionstyle"], "arc3,rad=0.4"
        )

    def test_zero_curvature_restores_straight_rendering(self):
        with patch(
            "connectome_interpreter.utils.nx.draw_networkx_edges",
            wraps=nx.draw_networkx_edges,
        ) as draw_edges:
            plot_paths(
                self.paths,
                show=False,
                edge_text=False,
                node_text=False,
                same_layer_edge_curvature=0,
                seed=0,
            )

        draw_edges.assert_not_called()


if __name__ == "__main__":
    unittest.main()
