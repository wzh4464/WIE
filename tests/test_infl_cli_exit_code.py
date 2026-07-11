"""The influence CLI must exit NONZERO when a calculation fails / writes no
scores, so a subprocess caller (run_pipeline check=True) sees the failure
instead of proceeding on a stale/absent CSV. The success path must stay exit 0.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock

from tests.conftest import ensure_dummy_modules


def _argv(type_key="wie_last", seed="0", extra=None):
    return [
        "prog",
        "--target", "adult",
        "--model", "logreg",
        "--type", type_key,
        "--seed", seed,
        "--gpu", "0",
    ] + (extra or [])


class TestInflCliExitCode(unittest.TestCase):
    def setUp(self):
        ensure_dummy_modules()

    def test_calculation_failure_exits_nonzero(self):
        # A calculator whose run() raises (e.g. the empty-window / length<1
        # guard, or any run() error) must make main() exit(1).
        with patch.object(sys, "argv", _argv()):
            with patch("wie.infl.cli.InfluenceCalculatorFactory.create") as create:
                calc = MagicMock()
                calc.run.side_effect = ValueError("window is empty after clamping")
                create.return_value = calc
                from wie.infl.cli import main

                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)
                calc.run.assert_called_once()

    def test_success_exits_zero(self):
        # run() returns normally -> main() returns without SystemExit (exit 0).
        with patch.object(sys, "argv", _argv()):
            with patch("wie.infl.cli.InfluenceCalculatorFactory.create") as create:
                calc = MagicMock()
                calc.run.return_value = None
                create.return_value = calc
                from wie.infl.cli import main

                self.assertIsNone(main())
                calc.run.assert_called_once()

    def test_multiseed_partial_failure_exits_nonzero(self):
        # seed < 0 loops 0..99; if ANY seed fails, exit nonzero (partial success
        # must be visible), but every seed is still attempted.
        with patch.object(sys, "argv", _argv(seed="-1")):
            with patch("wie.infl.cli.InfluenceCalculatorFactory.create") as create:
                calls = {"n": 0}

                def _make(*a, **k):
                    calls["n"] += 1
                    calc = MagicMock()
                    # Fail exactly one seed (the 4th call) to prove partial
                    # failure still yields a nonzero exit.
                    if calls["n"] == 4:
                        calc.run.side_effect = RuntimeError("boom")
                    else:
                        calc.run.return_value = None
                    return calc

                create.side_effect = _make
                from wie.infl.cli import main

                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)
                # All 100 seeds attempted despite the mid-loop failure.
                self.assertEqual(calls["n"], 100)


if __name__ == "__main__":
    unittest.main()
