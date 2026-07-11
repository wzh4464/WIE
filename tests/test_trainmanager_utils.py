import unittest
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
from wie.training.train import TrainManager


class TestTrainManagerUtils(unittest.TestCase):
    def test_generate_relabel_indices(self):
        tm = TrainManager(target="adult", model="logreg", seed=42)
        n = 100
        pct = 30
        idx = tm.generate_relabel_indices(n, pct, seed=42)
        self.assertEqual(len(idx), int(n * pct / 100))
        self.assertTrue((idx >= 0).all() and (idx < n).all())


if __name__ == "__main__":
    unittest.main()
