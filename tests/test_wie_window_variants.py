"""Unit-level correctness tests for the WIE window variants.

These test the WINDOW-BOUNDS math only -- no training trajectory is required.
We instantiate each calculator via ``__new__`` (bypassing the data-loading
``__init__``) and set the handful of attributes the bounds computation reads
(``total_steps``/``steps_per_epoch``/``num_epoch``/``length``), then assert
``_window_step_bounds()`` and the query-init wiring.

Key semantic check (test_full_window_invariant): with ``length == num_epoch``
all three of wie_first/wie_middle/wie_last cover the FULL window
``[0, total_steps]``. This pins the degeneracy that the first/last/middle
windows all reduce to the full trajectory when the window spans every epoch --
wie_first's ``w_end`` reaches ``total_steps`` and wie_last's ``w_start`` reaches
``0``.
"""

import unittest

from tests.conftest import ensure_dummy_modules


def _make(cls, num_epoch, steps_per_epoch, length, total_steps=None):
    """Build a calculator without running the data-loading __init__."""
    obj = cls.__new__(cls)
    obj.num_epoch = num_epoch
    obj.steps_per_epoch = steps_per_epoch
    obj.length = length
    obj.total_steps = (
        total_steps if total_steps is not None else num_epoch * steps_per_epoch
    )
    return obj


class TestWieWindowVariants(unittest.TestCase):
    def setUp(self):
        ensure_dummy_modules()
        import wie.infl  # noqa: F401  (registers all calculators)

    # -----------------------------
    # Per-variant window bounds
    # -----------------------------
    def test_wie_first_bounds(self):
        from wie.infl.wie_first import WieFirstInfluenceCalculator

        # num_epoch=10, spe=5, length=3 -> [0, min(3,10)*5] = (0, 15)
        obj = _make(WieFirstInfluenceCalculator, num_epoch=10, steps_per_epoch=5, length=3)
        self.assertEqual(obj._window_step_bounds(), (0, 15))

    def test_wie_first_length_clamped_to_num_epoch(self):
        from wie.infl.wie_first import WieFirstInfluenceCalculator

        # length > num_epoch is clamped by min(length, num_epoch).
        obj = _make(WieFirstInfluenceCalculator, num_epoch=4, steps_per_epoch=5, length=9)
        self.assertEqual(obj._window_step_bounds(), (0, 20))

    def test_wie_last_bounds(self):
        from wie.infl.wie_last import WieLastInfluenceCalculator

        # total_steps=50, last 3 epochs -> (50 - 3*5, 50) = (35, 50).
        # Matches the pre-refactor start_step_incl/end_step_excl exactly.
        obj = _make(WieLastInfluenceCalculator, num_epoch=10, steps_per_epoch=5, length=3)
        self.assertEqual(obj._window_step_bounds(), (35, 50))

    def test_wie_last_bounds_clamped_at_zero(self):
        from wie.infl.wie_last import WieLastInfluenceCalculator

        # length > num_epoch -> lower bound clamped to 0.
        obj = _make(WieLastInfluenceCalculator, num_epoch=4, steps_per_epoch=5, length=9)
        self.assertEqual(obj._window_step_bounds(), (0, 20))

    def test_wie_middle_bounds(self):
        from wie.infl.wie_middle import WieMiddleInfluenceCalculator

        # num_epoch=10, length=3 -> start_epoch=(10-3)//2=3;
        # w_start=3*5=15; w_end=min(3+3,10)*5=30 -> (15, 30) centered.
        obj = _make(WieMiddleInfluenceCalculator, num_epoch=10, steps_per_epoch=5, length=3)
        self.assertEqual(obj._window_step_bounds(), (15, 30))

    def test_wie_middle_bounds_odd_offset(self):
        from wie.infl.wie_middle import WieMiddleInfluenceCalculator

        # num_epoch=9, length=4 -> start_epoch=(9-4)//2=2;
        # w_start=2*3=6; w_end=min(2+4,9)*3=18 -> (6, 18).
        obj = _make(WieMiddleInfluenceCalculator, num_epoch=9, steps_per_epoch=3, length=4)
        self.assertEqual(obj._window_step_bounds(), (6, 18))

    # -----------------------------
    # Degeneracy invariant
    # -----------------------------
    def test_full_window_invariant(self):
        """length == num_epoch => first/middle/last all cover [0, total_steps]."""
        from wie.infl.wie_first import WieFirstInfluenceCalculator
        from wie.infl.wie_middle import WieMiddleInfluenceCalculator
        from wie.infl.wie_last import WieLastInfluenceCalculator

        E, spe = 8, 4
        total = E * spe  # 32
        first = _make(WieFirstInfluenceCalculator, E, spe, E)
        middle = _make(WieMiddleInfluenceCalculator, E, spe, E)
        last = _make(WieLastInfluenceCalculator, E, spe, E)

        self.assertEqual(first._window_step_bounds(), (0, total))
        self.assertEqual(middle._window_step_bounds(), (0, total))
        self.assertEqual(last._window_step_bounds(), (0, total))

        # The three windows are identical (all == the full trajectory).
        self.assertEqual(
            first._window_step_bounds(),
            last._window_step_bounds(),
        )
        self.assertEqual(
            first._window_step_bounds(),
            middle._window_step_bounds(),
        )

    # -----------------------------
    # Registration / factory wiring
    # -----------------------------
    def test_variants_registered(self):
        from wie.infl.core import InfluenceCalculatorFactory as F
        from wie.infl.wie_first import WieFirstInfluenceCalculator
        from wie.infl.wie_middle import WieMiddleInfluenceCalculator

        self.assertIn("wie_first", F._calculators)
        self.assertIn("wie_middle", F._calculators)
        self.assertIs(F._calculators["wie_first"], WieFirstInfluenceCalculator)
        self.assertIs(F._calculators["wie_middle"], WieMiddleInfluenceCalculator)

    def test_variants_share_window_base(self):
        from wie.infl.wie_window_base import _WieWindowInfluenceCalculator
        from wie.infl.wie_first import WieFirstInfluenceCalculator
        from wie.infl.wie_middle import WieMiddleInfluenceCalculator
        from wie.infl.wie_last import WieLastInfluenceCalculator

        # All three variants reuse (not duplicate) the shared sweep helpers.
        for cls in (
            WieFirstInfluenceCalculator,
            WieMiddleInfluenceCalculator,
            WieLastInfluenceCalculator,
        ):
            self.assertTrue(issubclass(cls, _WieWindowInfluenceCalculator))

    def test_legacy_aliases_resolve_to_variants(self):
        from wie.infl.core import InfluenceCalculatorFactory as F
        from wie.infl.wie_first import WieFirstInfluenceCalculator
        from wie.infl.wie_middle import WieMiddleInfluenceCalculator

        self.assertEqual(F._resolve("tim_first"), "wie_first")
        self.assertEqual(F._resolve("tim_middle"), "wie_middle")
        self.assertIs(
            F._calculators[F._resolve("tim_first")], WieFirstInfluenceCalculator
        )
        self.assertIs(
            F._calculators[F._resolve("tim_middle")], WieMiddleInfluenceCalculator
        )


class TestWindowBoundsClamping(unittest.TestCase):
    """Partial/early-stopped runs: bounds must clamp to the RECORDED trajectory.

    The critical case is that ``total_steps`` is the *nominal*
    ``num_epoch * steps_per_epoch`` even for a partial run, so clamping to
    ``total_steps`` alone is a no-op exactly when it's needed. These tests drive
    the recorded-endpoint path: they create a temp ``records`` dir with
    checkpoints only up to N < nominal and assert ``w_end`` clamps to N.
    """

    def setUp(self):
        ensure_dummy_modules()
        import wie.infl  # noqa: F401

    def _with_records(self, cls, num_epoch, spe, length, seed, filenames):
        """Build a variant whose self.dn/records holds the given checkpoint files."""
        import os
        import tempfile

        obj = _make(
            cls, num_epoch=num_epoch, steps_per_epoch=spe, length=length,
            total_steps=num_epoch * spe,  # nominal (records are shorter)
        )
        obj.seed = seed
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        records = os.path.join(tmp, "records")
        os.makedirs(records)
        for fn in filenames:
            open(os.path.join(records, fn), "w").close()
        obj.dn = tmp
        return obj

    def test_clamps_to_last_recorded_step_file(self):
        from wie.infl.wie_first import WieFirstInfluenceCalculator
        from wie.infl.wie_middle import WieMiddleInfluenceCalculator

        # nominal w_end would be 15 (=3*5); only step files 1..8 exist for seed 7.
        seed, N = 7, 8
        step_files = [f"step_{k}_{seed:03d}.pt" for k in range(1, N + 1)]
        for cls in (WieFirstInfluenceCalculator, WieMiddleInfluenceCalculator):
            obj = self._with_records(cls, 3, 5, 3, seed, step_files)
            # Raw (nominal) bound points past the recorded trajectory.
            self.assertGreater(obj._window_step_bounds()[1], N)
            w_start, w_end = obj._clamped_window_step_bounds()
            self.assertEqual(w_end, N, "w_end must clamp to the last recorded step")
            self.assertGreaterEqual(w_start, 0)
            self.assertLessEqual(w_start, w_end)

    def test_endpoint_is_max_across_step_and_epoch_formats(self):
        from wie.infl.wie_first import WieFirstInfluenceCalculator

        # Mixed/reused run: step files 1..8 AND epoch files 0..3. load_step_data
        # serves steps > 8 via the epoch fallback, so the endpoint must be the
        # FURTHER of the two -- epoch endpoint (3+1)*spe=20, not the step max 8.
        seed, spe = 4, 5
        steps = [f"step_{k}_{seed:03d}.pt" for k in range(1, 9)]
        epochs = [f"epoch_{e}_{seed:03d}.pt" for e in range(0, 4)]
        obj = self._with_records(
            WieFirstInfluenceCalculator, 5, spe, 5, seed, steps + epochs
        )
        self.assertEqual(obj._last_recorded_step(), 20)
        _, w_end = obj._clamped_window_step_bounds()
        self.assertEqual(
            w_end, 20, "endpoint must be the furthest loader-serviceable step"
        )

    def test_clamps_to_last_recorded_epoch_file(self):
        from wie.infl.wie_first import WieFirstInfluenceCalculator

        # No step files; epoch files 0,1 exist -> last usable step = (1+1)*spe = 10.
        seed, spe = 3, 5
        epoch_files = [f"epoch_{e}_{seed:03d}.pt" for e in (0, 1)]
        # A relabel-prefixed final file and unseeded steps-only dumps must be ignored.
        noise = ["epoch_final_003.pt", "step_000012.pt", "init_003.pt"]
        obj = self._with_records(
            WieFirstInfluenceCalculator, 3, spe, 3, seed, epoch_files + noise
        )
        _, w_end = obj._clamped_window_step_bounds()
        self.assertEqual(w_end, (1 + 1) * spe)

    def test_no_records_falls_back_to_total_steps(self):
        from wie.infl.wie_first import WieFirstInfluenceCalculator

        # Full run / undetectable records dir -> strict no-op (clamp == nominal).
        obj = _make(WieFirstInfluenceCalculator, num_epoch=4, steps_per_epoch=5, length=4)
        obj.seed = 0
        obj.dn = "/nonexistent-records-dir-xyz"
        w_start, w_end = obj._clamped_window_step_bounds()
        self.assertEqual((w_start, w_end), (0, 20))

    def test_recorded_endpoint_capped_by_total_steps(self):
        from wie.infl.wie_first import WieFirstInfluenceCalculator

        # All 3 epoch files exist, but the final epoch was PARTIAL so the real
        # trajectory is total_steps=13 (< 3*5=15). The epoch fallback rounds up
        # to (2+1)*5=15; without the total_steps cap that would request missing
        # steps. Cap must pull w_end back to 13.
        seed, spe = 5, 5
        epoch_files = [f"epoch_{e}_{seed:03d}.pt" for e in (0, 1, 2)]
        obj = self._with_records(
            WieFirstInfluenceCalculator, 3, spe, 3, seed, epoch_files
        )
        obj.total_steps = 13  # partial final epoch; below the nominal 15
        # Sanity: the raw recorded endpoint overshoots total_steps.
        self.assertEqual(obj._last_recorded_step(), (2 + 1) * spe)
        _, w_end = obj._clamped_window_step_bounds()
        self.assertEqual(w_end, 13)

    def test_scopes_to_active_relabel_prefix(self):
        from wie.infl.wie_first import WieFirstInfluenceCalculator

        # The active relabel-30 trajectory is recorded up to step 6; a stale
        # unprefixed trajectory (up to step 12) sharing the records dir must NOT
        # set the endpoint (else the loader would mix in the wrong states).
        seed = 7
        active = [f"relabel_030_pct_step_{k}_{seed:03d}.pt" for k in range(1, 7)]
        stale = [f"step_{k}_{seed:03d}.pt" for k in range(1, 13)]
        obj = self._with_records(
            WieFirstInfluenceCalculator, 3, 5, 3, seed, active + stale
        )
        obj.relabel_percentage = 30
        self.assertEqual(obj._last_recorded_step(), 6)
        _, w_end = obj._clamped_window_step_bounds()
        self.assertEqual(w_end, 6)

    def test_relabel_falls_back_to_unprefixed_when_no_prefixed_records(self):
        from wie.infl.wie_first import WieFirstInfluenceCalculator

        # No records carry this run's relabel prefix -> fall back to unprefixed
        # (mirrors resolve_step_file's prefixed-then-unprefixed lookup).
        seed = 7
        unpref = [f"step_{k}_{seed:03d}.pt" for k in range(1, 5)]  # up to 4
        obj = self._with_records(WieFirstInfluenceCalculator, 3, 5, 3, seed, unpref)
        obj.relabel_percentage = 30
        self.assertEqual(obj._last_recorded_step(), 4)

    def test_non_relabel_run_ignores_relabel_prefixed_records(self):
        from wie.infl.wie_first import WieFirstInfluenceCalculator

        # A non-relabel run must ignore relabel-prefixed trajectories entirely.
        seed = 7
        unpref = [f"step_{k}_{seed:03d}.pt" for k in range(1, 9)]  # up to 8
        relab = [
            f"relabel_030_pct_step_{k}_{seed:03d}.pt" for k in range(1, 13)
        ]  # up to 12
        obj = self._with_records(
            WieFirstInfluenceCalculator, 3, 5, 3, seed, unpref + relab
        )
        obj.relabel_percentage = None
        self.assertEqual(obj._last_recorded_step(), 8)

    def test_endpoint_stays_within_one_namespace(self):
        from wie.infl.wie_first import WieFirstInfluenceCalculator

        # Active relabel-30 namespace has ONLY epoch checkpoints (0,1 -> endpoint
        # (1+1)*5 = 10). A stale UNprefixed *step* trajectory (up to 12) shares
        # the dir. The endpoint must come entirely from the prefixed namespace
        # (10), NOT combine the prefixed epoch with the unprefixed step (12).
        seed, spe = 7, 5
        prefixed_epoch = [f"relabel_030_pct_epoch_{e}_{seed:03d}.pt" for e in (0, 1)]
        stale_unpref_step = [f"step_{k}_{seed:03d}.pt" for k in range(1, 13)]
        obj = self._with_records(
            WieFirstInfluenceCalculator, 3, spe, 3, seed,
            prefixed_epoch + stale_unpref_step,
        )
        obj.relabel_percentage = 30
        self.assertEqual(obj._last_recorded_step(), (1 + 1) * spe)

    def test_combines_step_and_epoch_within_same_namespace(self):
        from wie.infl.wie_first import WieFirstInfluenceCalculator

        # Within ONE namespace, step and epoch endpoints combine by max: seeded
        # steps up to 12 and epoch 1 (-> (1+1)*5=10) both prefixed -> max = 12.
        seed, spe = 7, 5
        files = [f"relabel_030_pct_step_{k}_{seed:03d}.pt" for k in range(1, 13)]
        files += [f"relabel_030_pct_epoch_{e}_{seed:03d}.pt" for e in (0, 1)]
        obj = self._with_records(WieFirstInfluenceCalculator, 3, spe, 3, seed, files)
        obj.relabel_percentage = 30
        self.assertEqual(obj._last_recorded_step(), 12)


class TestWindowLengthValidation(unittest.TestCase):
    """A non-positive window length must be rejected, not silently all-zeroed."""

    def setUp(self):
        ensure_dummy_modules()
        import wie.infl  # noqa: F401

    def test_calculate_rejects_nonpositive_length(self):
        from wie.infl.wie_first import WieFirstInfluenceCalculator
        from wie.infl.wie_middle import WieMiddleInfluenceCalculator
        from wie.infl.wie_last import WieLastInfluenceCalculator

        for cls in (
            WieFirstInfluenceCalculator,
            WieMiddleInfluenceCalculator,
            WieLastInfluenceCalculator,
        ):
            for bad in (0, -1, -5):
                obj = cls.__new__(cls)
                obj.length = bad
                with self.assertRaises(ValueError):
                    obj.calculate()

    def test_pipeline_parser_rejects_nonpositive_length(self):
        from scripts.epoch_wise_keep_ratio import build_parser

        base = [
            "--target", "adult", "--model", "logreg", "--save_dir", "t",
            "--relabel", "30", "--seed", "0", "--type", "wie_first", "--gpu", "0",
        ]
        for bad in ("0", "-1"):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(base + ["--length", bad])
        # A positive length still parses.
        args = build_parser().parse_args(base + ["--length", "5"])
        self.assertEqual(args.length, 5)


class TestEmptyClampedWindow(unittest.TestCase):
    """A late/middle window that clamps to empty on a partial run must raise."""

    def setUp(self):
        ensure_dummy_modules()
        import wie.infl  # noqa: F401

    def _with_records(self, cls, num_epoch, spe, length, seed, filenames):
        """Build a variant whose self.dn/records holds the given checkpoint files."""
        import os
        import tempfile

        obj = _make(
            cls, num_epoch=num_epoch, steps_per_epoch=spe, length=length,
            total_steps=num_epoch * spe,  # nominal (records are shorter)
        )
        obj.seed = seed
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        records = os.path.join(tmp, "records")
        os.makedirs(records)
        for fn in filenames:
            open(os.path.join(records, fn), "w").close()
        obj.dn = tmp
        return obj

    def test_late_window_empty_after_clamp_raises(self):
        from wie.infl.wie_last import WieLastInfluenceCalculator
        from wie.infl.wie_middle import WieMiddleInfluenceCalculator

        # 50 planned steps (num_epoch=10, spe=5) but only 10 recorded. wie_last's
        # nominal window [35, 50] and wie_middle's [15, 30] both clamp to [10, 10].
        seed = 7
        step_files = [f"step_{k}_{seed:03d}.pt" for k in range(1, 11)]  # recorded=10
        for cls in (WieLastInfluenceCalculator, WieMiddleInfluenceCalculator):
            obj = self._with_records(cls, 10, 5, 3, seed, step_files)
            w_start, w_end = obj._clamped_window_step_bounds()
            self.assertEqual(w_start, w_end, "window should be empty after clamp")
            with self.assertRaises(ValueError):
                obj.calculate()

    def test_wie_first_partial_run_window_stays_nonempty(self):
        from wie.infl.wie_first import WieFirstInfluenceCalculator

        # Same partial run: wie_first's window is [0, recorded] = [0, 10],
        # non-empty, so it must NOT hit the empty-window guard.
        seed = 7
        step_files = [f"step_{k}_{seed:03d}.pt" for k in range(1, 11)]
        obj = self._with_records(WieFirstInfluenceCalculator, 10, 5, 3, seed, step_files)
        w_start, w_end = obj._clamped_window_step_bounds()
        self.assertEqual((w_start, w_end), (0, 10))
        self.assertLess(w_start, w_end)


if __name__ == "__main__":
    unittest.main()
