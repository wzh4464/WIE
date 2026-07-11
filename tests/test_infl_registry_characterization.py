"""Characterization test pinning the full influence-calculator registry.

This is the critical missing safety net for the package refactor. It:

1. Imports *every* implemented calculator module so all registrations fire,
   including the formerly CLI-unreachable orphans.
2. Asserts the factory holds EXACTLY the expected 21 keys (catches a dropped
   orphan during a file move). The window variants wie_first/wie_middle and the
   window-level LOO ground-truth oracles true_first/true_middle/true_last (now
   implemented via BaseDifferenceCalculator differencing of segment_true_full)
   are all registered.
3. Pins ``{key -> registered class name}`` as a golden map. This is what
   catches the ``sgd``/``icml``/``tracin`` last-writer-wins swap: the wired
   adapter classes (``SgdAdapterCalculator`` ...) have different names from the
   legacy implementations (``SgdInfluenceCalculator`` ...), so a silent swap of
   which module wins the key changes the pinned class name and fails here.

We pin the class *name* (``__name__``) rather than the fully-qualified
``module.__name__``: the class names are stable across the intended module
moves (Phase 1 relocates files without renaming classes), while still uniquely
identifying which implementation serves each key. Update the golden
deliberately when the dedup decision or a rename is made.
"""

import importlib
import unittest

from tests.conftest import ensure_dummy_modules


# Importing the package registers ALL implemented calculators (its __init__
# imports every implemented calculator module, including the formerly
# CLI-unreachable orphans and the true_* window LOO oracles).
_CALCULATOR_MODULES = [
    "wie.infl",
]

# Golden {infl_type key -> registered class __name__}. Exactly 21 entries.
GOLDEN_REGISTRY = {
    # baseline methods (named after their papers -- never renamed)
    "sgd": "SgdAdapterCalculator",
    "icml": "IcmlAdapterCalculator",
    "tracin": "TracinAdapterCalculator",
    "lava": "LavaInfluenceCalculator",
    "lava_all_epochs": "LavaAllEpochsInfluenceCalculator",
    "loo_all_epochs": "LOOAllEpochsInfluenceCalculator",
    "dve": "DVEInfluenceCalculator",
    "dve_all_epochs": "DVEAllEpochsInfluenceCalculator",
    "icml_all_epochs": "ICMLAllEpochsInfluenceCalculator",
    "td_influence": "TDInfluenceCalculator",
    "true": "TrueInfluenceCalculator",
    "lie": "LieInfluenceCalculator",
    "nohess": "NoHessInfluenceCalculator",
    "segment_true_full": "SegmentTrueFullInfluenceCalculator",
    # team method window variants (all share the reverse-SGD window base)
    "wie_all_epochs": "WieAllEpochsInfluenceCalculator",
    "wie_last": "WieLastInfluenceCalculator",
    "wie_first": "WieFirstInfluenceCalculator",
    "wie_middle": "WieMiddleInfluenceCalculator",
    # window-level LOO ground-truth oracles (segment_true_full differencing)
    "true_first": "TrueFirstInfluenceCalculator",
    "true_middle": "TrueMiddleInfluenceCalculator",
    "true_last": "TrueLastInfluenceCalculator",
}


def _load_factory():
    ensure_dummy_modules()
    for mod in _CALCULATOR_MODULES:
        importlib.import_module(mod)
    from wie.infl.core import InfluenceCalculatorFactory

    return InfluenceCalculatorFactory


class TestInfluenceRegistryCharacterization(unittest.TestCase):
    def test_exactly_21_keys(self):
        factory = _load_factory()
        self.assertEqual(
            set(factory._calculators.keys()),
            set(GOLDEN_REGISTRY.keys()),
            "Registry key set drifted from the golden 21-key set",
        )
        self.assertEqual(len(factory._calculators), 21)

    def test_each_key_maps_to_expected_class(self):
        # Post-dedup: each key resolves to exactly one canonical class. The
        # sgd/icml/tracin adapters are the single, deduplicated implementations,
        # so the golden is asserted exactly (no import-order ambiguity remains).
        factory = _load_factory()
        actual = {k: v.__name__ for k, v in factory._calculators.items()}
        self.assertEqual(
            actual,
            GOLDEN_REGISTRY,
            "A registered class name changed (possible impl swap or rename)",
        )


if __name__ == "__main__":
    unittest.main()
