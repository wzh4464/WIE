import sys
import unittest
from unittest.mock import patch


class TestEpochWiseKeepRatioOrchestration(unittest.TestCase):
    def test_builds_and_runs_three_subprocesses(self):
        # Arrange: build argv to simulate the provided minimal command
        argv = [
            "prog",
            "--target",
            "sentiment",
            "--model",
            "bert",
            "--save_dir",
            "sentiment_bert_test",
            "--relabel",
            "30",
            "--seed",
            "42",
            "--type",
            "wie_all_epochs",
            "--log_level",
            "INFO",
            "--n_tr",
            "16",
            "--n_val",
            "16",
            "--num_epoch",
            "2",
            "--save_recording",
            "--gpu",
            "0",
        ]

        with patch.object(sys, "argv", argv):
            # Patch subprocess.run to avoid actually running heavy workloads
            with patch("scripts.epoch_wise_keep_ratio.subprocess.run") as mock_run:
                from scripts.epoch_wise_keep_ratio import main

                mock_run.return_value = None
                # Act
                main()

                # Assert: called 3 times (train, infl, cleansing)
                self.assertEqual(mock_run.call_count, 3)

                # Extract the command lists
                calls = [args[0][0] for args in mock_run.call_args_list]
                # Each call gets a list like ["python","-m","wie.training.train", ...]
                called_modules = [c[2] if len(c) > 2 else None for c in calls]
                self.assertIn("wie.training.train", called_modules)
                self.assertIn("wie.infl", called_modules)
                self.assertIn("wie.training.exp_influence_cleansing", called_modules)

    def test_accepts_lava_type_and_runs(self):
        argv = [
            "prog",
            "--target",
            "sentiment",
            "--model",
            "bert",
            "--save_dir",
            "sentiment_bert_test_lava",
            "--relabel",
            "30",
            "--seed",
            "42",
            "--type",
            "lava",
            "--log_level",
            "INFO",
            "--n_tr",
            "8",
            "--n_val",
            "8",
            "--num_epoch",
            "1",
            "--gpu",
            "0",
            "--dry-run",
        ]

        with patch.object(sys, "argv", argv):
            # Patch subprocess.run to avoid executing heavy workloads
            with patch("scripts.epoch_wise_keep_ratio.subprocess.run") as mock_run:
                from scripts.epoch_wise_keep_ratio import main

                mock_run.return_value = None
                # Act
                main()

                # Assert (dry-run prints, but we still build 3 commands)
                # In dry-run we don't call subprocess.run; ensure no calls
                self.assertEqual(mock_run.call_count, 0)


if __name__ == "__main__":
    unittest.main()
