"""WIE: Window-level Influence Estimator.

Single installable package consolidating the per-epoch SGD-influence /
label-flip-detection code (formerly split across the colliding ``experiment``
and ``src.experiment`` trees).

Compatibility shim:
- SymPy >=1.13 removed ``sympy.sets.conditionset``; some upstream deps still
  import it. We expose ``ConditionSet`` under the legacy path so imports like
  ``from sympy.sets.conditionset import ConditionSet`` keep working. This lives
  at the package root so it loads before any sympy-touching import.
"""

# --- SymPy compatibility shim for 1.13+ (legacy import path) ---
try:
    import importlib
    import sys
    import types
    import sympy  # noqa: F401

    try:
        # If this succeeds, nothing to do
        importlib.import_module("sympy.sets.conditionset")
    except Exception:
        # Provide a tiny shim module exposing ConditionSet from the new location
        try:
            from sympy.sets.sets import ConditionSet  # type: ignore

            m = types.ModuleType("sympy.sets.conditionset")
            setattr(m, "ConditionSet", ConditionSet)
            # Register under fully-qualified module name
            sys.modules["sympy.sets.conditionset"] = m
        except Exception:
            # Best-effort only; if sympy not present or structure unexpected, ignore
            pass
except Exception:
    # SymPy not installed; leave as-is (non-text/image flows may not require it)
    pass

__all__: list[str] = []
