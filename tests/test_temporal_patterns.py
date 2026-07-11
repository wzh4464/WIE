"""Tests for the RQ2 temporal influence-pattern classifier.

Constructs synthetic per-epoch influence matrices with known trajectories
(monotone-decreasing, monotone-increasing, flat, sign-alternating) plus a stable
background population, and checks each is labeled as the paper's corresponding
pattern. Also covers standardization, the distribution helper, CSV loading, and
the CLI script end-to-end.
"""

import importlib.util
import logging
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

# Stub optional heavy deps before importing wie (lean-env discovery safety).
from tests.conftest import ensure_dummy_modules

ensure_dummy_modules()

from wie.analysis.temporal_patterns import (  # noqa: E402
    PATTERN_LABELS,
    STABLE,
    EARLY,
    LATE,
    FLUCTUATING,
    standardize_per_epoch,
    classify_patterns,
    pattern_distribution,
    load_influence_matrix,
    mean_trajectories,
)


def _matrix_with_known_patterns(E=10, n_stable=30, seed=0):
    rng = np.random.RandomState(seed)
    e = np.arange(E, dtype=np.float64)
    # Stable background: each sample constant across epochs (spread of constants).
    stable_bg = np.repeat(rng.uniform(-1, 1, size=n_stable)[:, None], E, axis=1)
    early = 5.0 - 1.0 * e  # monotone decreasing -> Early Influencer
    late = -5.0 + 1.0 * e  # monotone increasing -> Late Bloomer
    flat = np.full(E, 0.3)  # constant -> Stable
    fluc = 3.0 * ((-1.0) ** e)  # alternating sign -> Highly Fluctuating
    M = np.vstack([stable_bg, early, late, flat, fluc])
    idx = {
        "early": n_stable,
        "late": n_stable + 1,
        "flat": n_stable + 2,
        "fluc": n_stable + 3,
    }
    return M, idx


class TestClassify(unittest.TestCase):
    def test_four_known_patterns(self):
        M, idx = _matrix_with_known_patterns()
        labels, stats = classify_patterns(M)
        self.assertEqual(labels[idx["early"]], EARLY)
        self.assertEqual(labels[idx["late"]], LATE)
        self.assertEqual(labels[idx["flat"]], STABLE)
        self.assertEqual(labels[idx["fluc"]], FLUCTUATING)
        # stats aligns with labels and has the documented columns
        self.assertEqual(len(stats), M.shape[0])
        self.assertLess(stats.loc[idx["early"], "slope"], 0)
        self.assertGreater(stats.loc[idx["late"], "slope"], 0)
        self.assertGreaterEqual(stats.loc[idx["fluc"], "flip_ratio"], 0.5)

    def test_all_stable(self):
        # Every sample constant across epochs -> all Stable.
        rng = np.random.RandomState(1)
        M = np.repeat(rng.uniform(-2, 2, size=25)[:, None], 8, axis=1)
        labels, _ = classify_patterns(M)
        self.assertTrue(all(lab == STABLE for lab in labels))

    def test_distribution_sums_to_100_and_covers_all_labels(self):
        M, _ = _matrix_with_known_patterns()
        labels, _ = classify_patterns(M)
        dist = pattern_distribution(labels, as_percent=True)
        self.assertEqual(set(dist.keys()), set(PATTERN_LABELS))
        self.assertAlmostEqual(sum(dist.values()), 100.0, places=6)
        counts = pattern_distribution(labels, as_percent=False)
        self.assertEqual(sum(counts.values()), M.shape[0])

    def test_short_trajectory_is_stable(self):
        # Fewer than 3 epochs: slope test undefined -> Stable, never crashes.
        M = np.array([[1.0, 2.0], [3.0, 1.0], [0.0, 0.0]])
        labels, _ = classify_patterns(M)
        self.assertTrue(all(lab == STABLE for lab in labels))

    def test_no_standardize_option_runs(self):
        M, idx = _matrix_with_known_patterns()
        labels, _ = classify_patterns(M, standardize=False)
        # Raw trends are still monotone, so early/late remain trend classes.
        self.assertIn(labels[idx["early"]], (EARLY, LATE, STABLE, FLUCTUATING))
        self.assertEqual(labels[idx["fluc"]], FLUCTUATING)

    def test_r2_effect_size_gate_moves_weak_trends_to_stable(self):
        # A faint-but-significant linear trend (p<0.05, low R^2) is a trend
        # without the effect-size gate, but falls back to Stable once min_r2 is
        # raised; a clean strong trend (R^2=1) survives the gate. This is the
        # fix for the pure-p classifier being degenerate on long trajectories.
        from wie.analysis.temporal_patterns import _ols_trend

        E = 40
        e = np.arange(E, dtype=np.float64)
        rng = np.random.RandomState(3)
        weak = 0.06 * e + rng.normal(0, 1.2, E)  # significant but weakly explained
        strong = 0.5 * e  # perfect line -> R^2 = 1
        stable_bg = np.repeat(rng.uniform(-1, 1, size=20)[:, None], E, axis=1)
        M = np.vstack([stable_bg, weak[None, :], strong[None, :]])
        wi, si = 20, 21

        _, p_w, r2_w = _ols_trend(weak)
        self.assertLess(p_w, 0.05)  # weak trend IS significant
        self.assertLess(r2_w, 0.7)  # but weakly explained

        # flip_ratio_threshold>1 disables the HF branch so it can't intercept
        # the noisy weak row; standardize=False keeps R^2 on the raw values.
        lab0, st0 = classify_patterns(
            M, standardize=False, flip_ratio_threshold=2.0, min_r2=0.0)
        labg, _ = classify_patterns(
            M, standardize=False, flip_ratio_threshold=2.0, min_r2=0.7)
        self.assertIn("r_squared", st0.columns)
        self.assertAlmostEqual(float(st0.loc[si, "r_squared"]), 1.0, places=6)
        self.assertIn(lab0[wi], (EARLY, LATE))  # trend without the gate
        self.assertEqual(labg[wi], STABLE)  # gated out -> Stable
        self.assertIn(labg[si], (EARLY, LATE))  # strong trend survives


class TestStandardize(unittest.TestCase):
    def test_zero_variance_column_maps_to_zero(self):
        M = np.array([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]])  # col0 constant
        Z = standardize_per_epoch(M)
        np.testing.assert_allclose(Z[:, 0], 0.0)
        # col1 standardized to zero mean, unit std
        self.assertAlmostEqual(float(Z[:, 1].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(Z[:, 1].std()), 1.0, places=6)

    def test_mean_trajectories_shape(self):
        M, _ = _matrix_with_known_patterns()
        labels, _ = classify_patterns(M)
        traj = mean_trajectories(M, labels)
        self.assertEqual(traj.shape[0], M.shape[1])  # one row per epoch
        self.assertTrue(set(traj.columns).issubset(set(PATTERN_LABELS)))


class TestIO(unittest.TestCase):
    def _write_wie_all_epochs_csv(self, path, M):
        df = pd.DataFrame({"sample_idx": np.arange(M.shape[0])})
        for e in range(M.shape[1]):
            df[f"influence_epoch_{e}"] = M[:, e]
        df.to_csv(path, index=False)

    def test_load_influence_matrix_roundtrip(self):
        M = np.arange(12, dtype=np.float64).reshape(3, 4)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "infl_wie_all_epochs_000.csv")
            self._write_wie_all_epochs_csv(path, M)
            loaded = load_influence_matrix(path)
            np.testing.assert_allclose(loaded, M)

    def test_cli_script_end_to_end(self):
        # Import the script module by path and run main() on a synthetic CSV.
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "classify_temporal_patterns.py",
        )
        spec = importlib.util.spec_from_file_location(
            "classify_temporal_patterns", script_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        M, idx = _matrix_with_known_patterns()
        with tempfile.TemporaryDirectory() as d:
            csv = os.path.join(d, "infl_wie_all_epochs_000.csv")
            self._write_wie_all_epochs_csv(csv, M)
            out = os.path.join(d, "patterns")
            rc = mod.main(["--input", csv, "--output-dir", out])
            self.assertEqual(rc, 0)
            labels_csv = os.path.join(out, "temporal_pattern_labels.csv")
            dist_csv = os.path.join(out, "temporal_pattern_distribution.csv")
            self.assertTrue(os.path.isfile(labels_csv))
            self.assertTrue(os.path.isfile(dist_csv))
            lab_df = pd.read_csv(labels_csv)
            self.assertEqual(lab_df.loc[idx["fluc"], "label"], FLUCTUATING)
            dist_df = pd.read_csv(dist_csv)
            self.assertEqual(set(dist_df["pattern"]), set(PATTERN_LABELS))
            self.assertEqual(int(dist_df["count"].sum()), M.shape[0])


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main()
