import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import plotly.graph_objects as go

from connectome_interpreter.external_map import hex_heatmap, plot_mollweide_projection

EYEMAP_ROWS = [
    {"p": 0, "q": 0, "x": 0.0, "y": 0.0, "z": 1.0},
    {"p": 1, "q": 0, "x": 0.1, "y": 0.0, "z": 0.99},
    {"p": 0, "q": 1, "x": 0.0, "y": 0.1, "z": 0.99},
    {"p": -1, "q": 0, "x": -0.1, "y": 0.0, "z": 0.99},
    {"p": 0, "q": -1, "x": 0.0, "y": -0.1, "z": 0.99},
]


class TestEyemapPath(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.eyemap_path = Path(self.tmpdir.name) / "eyemap.csv"
        pd.DataFrame(EYEMAP_ROWS).to_csv(self.eyemap_path, index=False)

        # data index format matches "x,y" = (hex2_id - hex1_id, hex2_id + hex1_id),
        # i.e. (p - q, p + q), so it aligns with the rows above.
        self.data = pd.Series(
            [1.0, 2.0, 3.0],
            index=["0,0", "1,1", "-1,-1"],
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_hex_heatmap_with_eyemap_path(self):
        fig = hex_heatmap(self.data, eyemap_path=str(self.eyemap_path))
        self.assertIsInstance(fig, go.Figure)

    def test_plot_mollweide_projection_with_eyemap_path(self):
        fig = plot_mollweide_projection(self.data, eyemap_path=str(self.eyemap_path))
        self.assertIsInstance(fig, go.Figure)

    def test_hex_heatmap_missing_columns_raises(self):
        bad_path = Path(self.tmpdir.name) / "bad_eyemap.csv"
        pd.DataFrame([{"p": 0, "x": 0.0}]).to_csv(bad_path, index=False)

        with self.assertRaises(ValueError):
            hex_heatmap(self.data, eyemap_path=str(bad_path))

    def test_plot_mollweide_projection_missing_columns_raises(self):
        bad_path = Path(self.tmpdir.name) / "bad_eyemap.csv"
        pd.DataFrame([{"p": 0, "q": 0}]).to_csv(bad_path, index=False)

        with self.assertRaises(ValueError):
            plot_mollweide_projection(self.data, eyemap_path=str(bad_path))

    def test_hex_heatmap_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            hex_heatmap(self.data, eyemap_path=str(Path(self.tmpdir.name) / "nope.csv"))
