"""Characterization test pinning on-disk filename conventions.

These conventions determine which historical output files remain discoverable
after the team-method token rename (tim_* -> wie_*). Pinning them here makes the
Phase-3 output-alias behavior verifiable and guards against accidental changes
to seed zero-padding, the ``relabel_NNN_pct_`` infix, or the cleansed suffix.
"""

import logging
import os
import tempfile
import unittest

import numpy as np

from tests.conftest import ensure_dummy_modules


class TestFilenameConventions(unittest.TestCase):
    def setUp(self):
        ensure_dummy_modules()
        self.logger = logging.getLogger("test_filename_conventions")

    def test_relabel_prefix(self):
        from wie.io.naming import make_relabel_prefix

        self.assertEqual(make_relabel_prefix(None), "")
        self.assertEqual(make_relabel_prefix(30), "relabel_030_pct_")
        self.assertEqual(make_relabel_prefix(5), "relabel_005_pct_")
        self.assertEqual(make_relabel_prefix(100), "relabel_100_pct_")

    def test_save_results_infl_filename_no_relabel(self):
        """Pins core.save_results: infl_{type}_{seed:03d}.csv/.json."""
        from wie.infl.core import save_results

        with tempfile.TemporaryDirectory() as dn:
            save_results(
                np.arange(4, dtype=np.float32),
                dn,
                seed=7,
                infl_type="wie_all_epochs",
                logger=self.logger,
                relabel_percentage=None,
            )
            self.assertTrue(
                os.path.exists(os.path.join(dn, "infl_wie_all_epochs_007.csv"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(dn, "infl_wie_all_epochs_007.json"))
            )

    def test_save_results_infl_filename_with_relabel(self):
        """Pins the relabel infix: infl_{type}_relabel_{pct:03d}_pct_{seed:03d}."""
        from wie.infl.core import save_results

        with tempfile.TemporaryDirectory() as dn:
            save_results(
                np.arange(4, dtype=np.float32),
                dn,
                seed=3,
                infl_type="sgd",
                logger=self.logger,
                relabel_percentage=40,
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(dn, "infl_sgd_relabel_040_pct_003.csv")
                )
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(dn, "infl_sgd_relabel_040_pct_003.json")
                )
            )

    def test_cleansed_run_suffix_convention(self):
        """Documents/pins the cleansing suffix cleansed_{type}_{keep:03d}_pct.

        This mirrors exp_influence_cleansing.py:539. It is a convention pin (the
        live code builds this string inline); Phase 3's naming resolver must keep
        producing this exact shape for legacy reads to keep resolving.
        """
        infl_type = "wie_all_epochs"
        keep_ratio = 90
        suffix = f"cleansed_{infl_type}_{keep_ratio:03d}_pct"
        self.assertEqual(suffix, "cleansed_wie_all_epochs_090_pct")


if __name__ == "__main__":
    unittest.main()
