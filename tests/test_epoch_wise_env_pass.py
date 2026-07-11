import sys
import unittest
from unittest.mock import patch


class TestEpochWiseEnvPropagation(unittest.TestCase):
    def test_dropout_and_label_smoothing_passed_to_train_env(self):
        argv = [
            "prog",
            "--target",
            "sentiment",
            "--model",
            "bert",
            "--save_dir",
            "sentiment_bert_envtest",
            "--relabel",
            "30",
            "--seed",
            "42",
            "--type",
            "wie_all_epochs",
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
            "--dropout",
            "0.4",
            "--label_smoothing",
            "0.1",
        ]

        with patch.object(sys, "argv", argv):
            with patch("scripts.epoch_wise_keep_ratio.subprocess.run") as mock_run:
                from scripts.epoch_wise_keep_ratio import main

                mock_run.return_value = None

                main()

                # Three subprocesses should be invoked
                self.assertEqual(mock_run.call_count, 3)

                # First call is training; kwargs should contain env with our vars
                first_call_args, first_call_kwargs = mock_run.call_args_list[0]
                self.assertIn("env", first_call_kwargs)
                env = first_call_kwargs["env"]
                self.assertEqual(env.get("HF_TEXT_DROPOUT"), "0.4")
                self.assertEqual(env.get("LABEL_SMOOTHING"), "0.1")

                # Subsequent calls should either not override env or be default
                # We only assert they don't accidentally pass the training env keys
                for i in [1, 2]:
                    _, kw = mock_run.call_args_list[i]
                    if "env" in kw and kw["env"] is not None:
                        self.assertNotIn(
                            "HF_TEXT_DROPOUT", kw["env"]
                        )  # only for training
                        self.assertNotIn(
                            "LABEL_SMOOTHING", kw["env"]
                        )  # only for training


if __name__ == "__main__":
    unittest.main()
