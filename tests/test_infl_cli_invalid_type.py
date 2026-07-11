import sys
import unittest
from unittest.mock import patch
import types
import numpy as np


def _ensure_dummy_emnist_module():
    if "emnist" not in sys.modules:
        dummy = types.ModuleType("emnist")

        def _extract_training_samples(*args, **kwargs):
            return np.zeros((2, 28, 28), dtype=np.uint8), np.zeros((2,), dtype=np.int64)

        dummy.extract_training_samples = _extract_training_samples
        sys.modules["emnist"] = dummy


class TestInflCliInvalidType(unittest.TestCase):
    def test_unknown_type_exits_nonzero(self):
        # Unknown type writes no scores -> CLI logs the error and exits NONZERO
        # (so a subprocess caller with check=True sees the failure) rather than
        # swallowing it and returning 0.
        argv = [
            "prog",
            "--target",
            "adult",
            "--model",
            "logreg",
            "--type",
            "not_registered",
            "--seed",
            "0",
            "--gpu",
            "0",
        ]
        _ensure_dummy_emnist_module()
        with patch.object(sys, "argv", argv):
            from wie.infl.cli import main

            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
