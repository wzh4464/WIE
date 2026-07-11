import sys
import unittest
from unittest.mock import patch, MagicMock


class TestInflCliFactory(unittest.TestCase):
    def test_factory_invocation(self):
        argv = [
            "prog",
            "--target",
            "adult",
            "--model",
            "logreg",
            "--type",
            "sgd",
            "--seed",
            "0",
            "--gpu",
            "0",
        ]

        with patch.object(sys, "argv", argv):
            # Mock the factory to avoid heavy compute
            with patch(
                "wie.infl.cli.InfluenceCalculatorFactory.create"
            ) as mock_create:
                dummy = MagicMock()
                dummy.run.return_value = None
                mock_create.return_value = dummy

                # Import and run main
                from wie.infl.cli import main

                main()

                mock_create.assert_called_once()
                args, kwargs = mock_create.call_args
                # First positional arg is type key
                self.assertEqual(args[0], "sgd")
                # Ensure essential kwargs are passed through
                self.assertEqual(kwargs["key"], "adult")
                self.assertEqual(kwargs["model_type"], "logreg")
                self.assertEqual(kwargs["seed"], 0)


if __name__ == "__main__":
    unittest.main()
