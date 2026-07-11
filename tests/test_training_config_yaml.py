import os
import unittest
import sys
import types
import importlib.machinery


def _ensure_dummy_modules():
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
from wie.training.config import TrainingConfig


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestTrainingConfigYAML(unittest.TestCase):
    def test_load_yaml_config(self):
        cfg_path = os.path.join(REPO_ROOT, "configs", "imdb_bert_base.yaml")
        self.assertTrue(os.path.exists(cfg_path))
        cfg = TrainingConfig.from_yaml(cfg_path)
        # Sanity checks for a few keys
        self.assertEqual(cfg.model_name_or_path, "bert-base-uncased")
        self.assertEqual(cfg.num_labels, 2)
        self.assertEqual(cfg.max_length, 128)


if __name__ == "__main__":
    unittest.main()
