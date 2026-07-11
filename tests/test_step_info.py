# test infl.load_step_data

import unittest
import os
import sys
import types
import numpy as np
import importlib.machinery


def _ensure_dummy_modules():
    if "emnist" not in sys.modules:
        dummy = types.ModuleType("emnist")

        def _extract_training_samples(*args, **kwargs):
            return np.zeros((2, 28, 28), dtype=np.uint8), np.zeros((2,), dtype=np.int64)

        dummy.extract_training_samples = _extract_training_samples
        sys.modules["emnist"] = dummy

    if "matplotlib" not in sys.modules:
        mpl = types.ModuleType("matplotlib")
        mpl.__spec__ = importlib.machinery.ModuleSpec("matplotlib", loader=None)
        pyplot = types.ModuleType("pyplot")
        pyplot.__spec__ = importlib.machinery.ModuleSpec(
            "matplotlib.pyplot", loader=None
        )

        def _noop(*args, **kwargs):
            return None

        pyplot.imshow = _noop
        pyplot.axis = _noop
        pyplot.show = _noop
        mpl.pyplot = pyplot
        sys.modules["matplotlib"] = mpl
        sys.modules["matplotlib.pyplot"] = pyplot


_ensure_dummy_modules()
from wie.infl import load_step_data


class TestStepInfo(unittest.TestCase):
    def test_load_step_data(self):
        save_dir = "test"  # Use only the test directory
        dir_name_base = "experiment"
        step = 1
        seed = 0
        relabel_percentage = None
        dir_name = os.path.join(dir_name_base, save_dir)
        try:
            data = load_step_data(dir_name, step, seed, relabel_percentage)
        except FileNotFoundError as e:
            self.skipTest(f"Test data not found: {e}")
        self.assertIsInstance(data, dict)
        self.assertIn("model_state", data)
        self.assertIn("idx", data)
        self.assertIn("lr", data)
        self.assertIsInstance(data["model_state"], dict)


if __name__ == "__main__":
    unittest.main()
