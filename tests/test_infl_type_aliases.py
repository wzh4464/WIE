"""Phase-3 alias tests: legacy tim_* keys/outputs remain usable after the rename.

The team method was renamed tim_* -> wie_*. Pre-existing configs/commands (keys)
and on-disk outputs (filenames) still say tim_*; these must keep resolving while
new runs write wie_*. Nothing under outputs/ is renamed.
"""

import os
import tempfile
import unittest

from tests.conftest import ensure_dummy_modules


# All team-method aliases whose canonical target is registered. wie_first/
# wie_middle are now implemented, so tim_first/tim_middle are restored.
LEGACY_TO_CANONICAL = {
    "tim_all_epochs": "wie_all_epochs",
    "tim_last": "wie_last",
    "tim_first": "wie_first",
    "tim_middle": "wie_middle",
}


class TestInflTypeAliases(unittest.TestCase):
    def setUp(self):
        ensure_dummy_modules()
        import wie.infl  # noqa: F401  (register all calculators)

    def test_factory_resolves_legacy_keys(self):
        from wie.infl.core import InfluenceCalculatorFactory as F

        for legacy, canonical in LEGACY_TO_CANONICAL.items():
            self.assertEqual(F._resolve(legacy), canonical)
        # Non-team / already-canonical keys are returned unchanged.
        self.assertEqual(F._resolve("wie_all_epochs"), "wie_all_epochs")
        self.assertEqual(F._resolve("sgd"), "sgd")

    def test_legacy_key_maps_to_wie_calculator(self):
        from wie.infl.core import InfluenceCalculatorFactory as F

        expected = {
            "tim_all_epochs": "WieAllEpochsInfluenceCalculator",
            "tim_last": "WieLastInfluenceCalculator",
            "tim_first": "WieFirstInfluenceCalculator",
            "tim_middle": "WieMiddleInfluenceCalculator",
        }
        for legacy, class_name in expected.items():
            cls = F._calculators[F._resolve(legacy)]
            self.assertEqual(cls.__name__, class_name)

    def test_glob_patterns_include_legacy_prefix(self):
        from wie.io.naming import infl_glob_patterns, infl_type_read_aliases

        self.assertEqual(
            infl_type_read_aliases("wie_all_epochs"),
            ["wie_all_epochs", "tim_all_epochs"],
        )
        # Bidirectional: passing the LEGACY key must still normalize to the
        # canonical wie_* name first (new outputs are written wie_*).
        self.assertEqual(
            infl_type_read_aliases("tim_all_epochs"),
            ["wie_all_epochs", "tim_all_epochs"],
        )
        for legacy, canonical in LEGACY_TO_CANONICAL.items():
            self.assertEqual(
                infl_type_read_aliases(legacy),
                [canonical, legacy],
            )
        # Non-team keys are returned unchanged (no alias).
        self.assertEqual(infl_type_read_aliases("sgd"), ["sgd"])
        self.assertEqual(
            infl_glob_patterns("wie_all_epochs"),
            ["infl_wie_all_epochs_*.csv", "infl_tim_all_epochs_*.csv"],
        )

    def test_resolve_infl_csv_falls_back_to_legacy(self):
        from wie.io.naming import resolve_infl_csv

        with tempfile.TemporaryDirectory() as dn:
            # Only a legacy tim_* file exists -> resolver must find it.
            legacy = os.path.join(dn, "infl_tim_all_epochs_003.csv")
            open(legacy, "w").close()
            self.assertEqual(
                resolve_infl_csv(dn, "wie_all_epochs", seed=3), legacy
            )

            # When the canonical wie_* file also exists, it wins.
            canonical = os.path.join(dn, "infl_wie_all_epochs_003.csv")
            open(canonical, "w").close()
            self.assertEqual(
                resolve_infl_csv(dn, "wie_all_epochs", seed=3), canonical
            )

    def test_glob_infl_csvs_finds_both(self):
        from wie.io.naming import glob_infl_csvs

        with tempfile.TemporaryDirectory() as dn:
            open(os.path.join(dn, "infl_tim_all_epochs_000.csv"), "w").close()
            open(os.path.join(dn, "infl_wie_all_epochs_001.csv"), "w").close()
            found = {os.path.basename(p) for p in glob_infl_csvs(dn, "wie_all_epochs")}
            self.assertEqual(
                found,
                {"infl_tim_all_epochs_000.csv", "infl_wie_all_epochs_001.csv"},
            )


if __name__ == "__main__":
    unittest.main()
