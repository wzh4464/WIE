import os
import json
import unittest
import tempfile
import logging
import sys
import types
import numpy as np


def _ensure_dummy_emnist_module():
    if "emnist" not in sys.modules:
        dummy = types.ModuleType("emnist")

        def _extract_training_samples(*args, **kwargs):
            return np.zeros((2, 28, 28), dtype=np.uint8), np.zeros((2,), dtype=np.int64)

        dummy.extract_training_samples = _extract_training_samples
        sys.modules["emnist"] = dummy


class TestInflCoreExportsAndIO(unittest.TestCase):
    def setUp(self):
        _ensure_dummy_emnist_module()

    def test_icml_constants_exported(self):
        from wie.infl.core import (
            BATCH_SIZE_ICML,
            LR_ICML,
            MOMENTUM_ICML,
            NUM_EPOCHS_ICML,
        )

        self.assertIsInstance(BATCH_SIZE_ICML, int)
        self.assertGreater(BATCH_SIZE_ICML, 0)
        self.assertIsInstance(LR_ICML, float)
        self.assertGreater(LR_ICML, 0)
        self.assertIsInstance(MOMENTUM_ICML, float)
        self.assertGreaterEqual(MOMENTUM_ICML, 0)
        self.assertIsInstance(NUM_EPOCHS_ICML, int)
        self.assertGreater(NUM_EPOCHS_ICML, 0)

    def test_save_results_single_and_list(self):
        from wie.infl.core import save_results

        logger = logging.getLogger("test")
        with tempfile.TemporaryDirectory() as dn:
            # Single array case
            arr = np.array([0.1, -0.2, 0.3])
            save_results(arr, dn, seed=7, infl_type="sgd", logger=logger)
            csv_path = os.path.join(dn, "infl_sgd_007.csv")
            json_path = os.path.join(dn, "infl_sgd_007.json")
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(json_path))
            with open(json_path) as f:
                meta = json.load(f)
            self.assertEqual(meta["seed"], 7)
            self.assertEqual(meta["type"], "sgd")

            # List (wie_all_epochs style)
            arr_list = [np.zeros(3), np.ones(3)]
            save_results(
                arr_list, dn, seed=3, infl_type="wie_all_epochs", logger=logger
            )
            csv_path2 = os.path.join(dn, "infl_wie_all_epochs_003.csv")
            json_path2 = os.path.join(dn, "infl_wie_all_epochs_003.json")
            self.assertTrue(os.path.exists(csv_path2))
            self.assertTrue(os.path.exists(json_path2))


if __name__ == "__main__":
    unittest.main()
