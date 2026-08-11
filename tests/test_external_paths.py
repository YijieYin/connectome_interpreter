import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from connectome_interpreter.external_paths import layered_el


class TestLayeredEl(unittest.TestCase):
    def test_percentile_is_forwarded_to_group_paths(self):
        matrix = csr_matrix(
            (
                np.array([1.0, 2.0, 3.0, 4.0]),
                (np.zeros(4, dtype=int), np.arange(1, 5)),
            ),
            shape=(5, 5),
        )
        idx_to_group = {0: "A", 1: "B", 2: "B", 3: "B", 4: "B"}
        flow = pd.DataFrame(
            {"cell_group": ["A", "B"], "hitting_time": [0.0, 1.0]}
        )

        result, _, _, _ = layered_el(
            matrix,
            inidx=[0],
            outidx=[1, 2, 3, 4],
            n=1,
            idx_to_group=idx_to_group,
            combining_method="percentile",
            percentile=75,
            flow=flow,
        )

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["weight"], 3.25)
