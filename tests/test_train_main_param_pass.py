import os
import sys
import unittest
from unittest.mock import patch


class TestTrainMainParamPass(unittest.TestCase):
    def test_cli_label_smoothing_and_dropout_env(self):
        # Prepare argv for a light-weight config (adult + logreg avoids transformers)
        argv = [
            "prog",
            "--target",
            "adult",
            "--model",
            "logreg",
            "--save_dir",
            "tmp_train_param_pass",
            "--seed",
            "0",
            "--gpu",
            "0",
            "--log_level",
            "INFO",
            "--label_smoothing",
            "0.123",
            "--dropout",
            "0.4",
        ]

        captured = {}

        # Patch TrainManager.__init__ to capture label_smoothing without heavy setup
        import wie.training.train as train_mod

        def fake_init(self, **kwargs):
            captured["label_smoothing"] = kwargs.get("label_smoothing")
            # No-op to skip heavy initialization
            return None

        with (
            patch.object(sys, "argv", argv),
            patch.object(train_mod.TrainManager, "__init__", fake_init),
            patch.object(train_mod.TrainManager, "train_and_save", return_value=None),
            patch.dict(os.environ, {}, clear=False),
        ):
            # Act
            train_mod.TrainManager.main()

            # Assert label smoothing wired through CLI to TrainManager
            self.assertIn("label_smoothing", captured)
            self.assertAlmostEqual(float(captured["label_smoothing"]), 0.123, places=6)

            # Assert dropout CLI set the env var used by Bert head
            self.assertEqual(os.environ.get("HF_TEXT_DROPOUT"), "0.4")


if __name__ == "__main__":
    unittest.main()
