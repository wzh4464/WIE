"""Shared test fixtures/helpers centralizing the dummy-module injection that
several tests previously hand-rolled via local ``_ensure_dummy_modules()``.

Under ``python -m unittest discover`` there is no automatic conftest loading,
so tests import :func:`ensure_dummy_modules` explicitly::

    from tests.conftest import ensure_dummy_modules

The repository root is on ``sys.path`` (tests already rely on this to import
``experiment.*`` / ``wie.*``), so ``tests.conftest`` resolves regardless of how
the discovering runner names the individual test module.
"""

import importlib.machinery
import sys
import types

import numpy as np


def ensure_dummy_modules() -> None:
    """Inject lightweight stand-ins for optional heavy deps (emnist, matplotlib).

    Idempotent: only injects a module if it is not already importable in
    ``sys.modules``. Kept behaviourally identical to the per-test helpers it
    replaces so it can be adopted incrementally without changing outcomes.
    """
    if "emnist" not in sys.modules:
        dummy = types.ModuleType("emnist")

        def _extract_training_samples(*args, **kwargs):
            # Return tiny dummy arrays to satisfy signatures if ever called
            return (
                np.zeros((2, 28, 28), dtype=np.uint8),
                np.zeros((2,), dtype=np.int64),
            )

        dummy.extract_training_samples = _extract_training_samples
        sys.modules["emnist"] = dummy

    if "matplotlib" not in sys.modules:
        mpl = types.ModuleType("matplotlib")
        mpl.__spec__ = importlib.machinery.ModuleSpec("matplotlib", loader=None)
        pyplot = types.ModuleType("pyplot")
        pyplot.__spec__ = importlib.machinery.ModuleSpec(
            "matplotlib.pyplot", loader=None
        )

        def _noop(*args, **kwargs):
            return None

        pyplot.imshow = _noop
        pyplot.axis = _noop
        pyplot.show = _noop
        mpl.pyplot = pyplot
        sys.modules["matplotlib"] = mpl
        sys.modules["matplotlib.pyplot"] = pyplot


# Optional pytest fixture form (harmless under unittest; useful if the suite is
# ever run under pytest).
try:  # pragma: no cover - pytest is not the primary runner
    import pytest

    @pytest.fixture(autouse=True)
    def _auto_dummy_modules():
        ensure_dummy_modules()
        yield
except Exception:  # pragma: no cover
    pass
