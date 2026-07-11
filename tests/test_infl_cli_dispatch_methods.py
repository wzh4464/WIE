import sys
import unittest
from unittest.mock import patch, MagicMock

from tests.conftest import ensure_dummy_modules


# All 21 registered influence types. The CLI must dispatch every one of them to
# the factory with the type as the first positional arg and the shared kwargs.
ALL_METHOD_TYPES = [
    "sgd",
    "icml",
    "tracin",
    "lava",
    "lava_all_epochs",
    "loo_all_epochs",
    "dve",
    "dve_all_epochs",
    "icml_all_epochs",
    "td_influence",
    "true",
    "lie",
    "nohess",
    "segment_true_full",
    "wie_all_epochs",
    "wie_last",
    "wie_first",
    "wie_middle",
    "true_first",
    "true_middle",
    "true_last",
]


class TestInflCliDispatchPerMethod(unittest.TestCase):
    def test_cli_dispatch_for_all_methods(self):
        for m in ALL_METHOD_TYPES:
            with self.subTest(m=m):
                argv = [
                    "prog",
                    "--target",
                    "adult",
                    "--model",
                    "logreg",
                    "--type",
                    m,
                    "--seed",
                    "0",
                    "--gpu",
                    "0",
                ]

                # Patch argv and the factory so we don't run heavy compute
                with patch.object(sys, "argv", argv):
                    ensure_dummy_modules()
                    with patch(
                        "wie.infl.cli.InfluenceCalculatorFactory.create"
                    ) as mock_create:
                        dummy = MagicMock()
                        dummy.run.return_value = None
                        mock_create.return_value = dummy

                        from wie.infl.cli import main

                        main()

                        self.assertTrue(mock_create.called)
                        args, kwargs = mock_create.call_args
                        # First positional argument is the type key
                        self.assertEqual(args[0], m)
                        # Ensure essential kwargs are propagated
                        self.assertEqual(kwargs["key"], "adult")
                        self.assertEqual(kwargs["model_type"], "logreg")


class TestQueryCliValidation(unittest.TestCase):
    """--query (prediction/saliency) is only valid for WIE window calculators."""

    def _run(self, type_):
        argv = [
            "prog", "--target", "adult", "--model", "logreg",
            "--type", type_, "--query", "prediction", "--seed", "0", "--gpu", "0",
        ]
        with patch.object(sys, "argv", argv):
            ensure_dummy_modules()
            with patch("wie.infl.cli.InfluenceCalculatorFactory.create") as mock_create:
                mock_create.return_value = MagicMock(run=MagicMock(return_value=None))
                from wie.infl.cli import main

                code = None
                try:
                    main()
                except SystemExit as e:
                    code = e.code
                return code, mock_create

    def test_rejects_query_for_non_window_type(self):
        code, mock_create = self._run("sgd")
        self.assertEqual(code, 1)
        mock_create.assert_not_called()

    def test_allows_query_for_window_type(self):
        _, mock_create = self._run("wie_last")
        self.assertTrue(mock_create.called)
        _, kwargs = mock_create.call_args
        self.assertEqual(kwargs.get("query_type"), "prediction")


if __name__ == "__main__":
    unittest.main()
