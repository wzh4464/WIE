"""Influence-calculation package.

Importing this package registers every *implemented* influence calculator with
the factory so all ``infl_type`` keys are reachable from the CLI (this includes
calculators that were previously CLI-unreachable orphans: wie_last, true, lie,
nohess, segment_true_full).

The paper's window-level variants ``wie_first`` and ``wie_middle`` are now real
implementations (see ``wie_window_base``): they share ``wie_last``'s reverse-SGD
sweep and numeric helpers verbatim, differing ONLY in the epoch window and the
query-init point (window-end model state via ``_u_at_step``). They are imported
and registered here.

The ground-truth LOO window variants ``true_first``/``true_middle``/
``true_last`` are also registered now. They subclass ``BaseDifferenceCalculator``
and compute the window leave-one-out oracle by differencing the per-epoch
full-LOO validation-loss trajectory (``segment_true_full``) across the window's
epoch endpoints: ``seg[t2] - seg[t1]``, which equals the paper's Test-Loss
window quantity. This needs only the full-LOO per-epoch counterfactual models
the trainer already writes under ``--compute_counterfactual`` -- no
window-restricted retraining -- so they are the LOO oracle the ``wie_*`` window
estimators are validated against (Table 1 RQ1-Local / Table 2 First-Mid-Last).
"""

from .core import (  # noqa: F401
    get_device,
    get_file_paths,
    load_global_info,
    load_data,
    get_input_dim,
    compute_gradient,
    load_epoch_data,
    load_step_data,
    load_model_state_from_fallback,
    load_initial_model,
    load_final_model,
    save_results,
    compute_adaptive_lambda,
    compute_hvp_with_finite_diff,
    infl_diff_helper,
    InfluenceCalculator,
    InfluenceCalculatorFactory,
    BaseDifferenceCalculator,
)

# --- Import every calculator module to trigger @register decorators ---
# Canonical baseline adapters (sgd/icml/tracin): the single, deduplicated impls.
from .adapters import (  # noqa: F401
    SgdAdapterCalculator,
    IcmlAdapterCalculator,
    TracinAdapterCalculator,
)

# Baseline "all epochs" / standalone calculators
from .lava import LavaInfluenceCalculator  # noqa: F401
from .lava_all_epochs import LavaAllEpochsInfluenceCalculator  # noqa: F401
from .dve import DVEInfluenceCalculator  # noqa: F401
from .dve_all_epochs import DVEAllEpochsInfluenceCalculator  # noqa: F401
from .loo_all_epochs import LOOAllEpochsInfluenceCalculator  # noqa: F401
from .icml_all_epochs import ICMLAllEpochsInfluenceCalculator  # noqa: F401
from .td_influence import TDInfluenceCalculator  # noqa: F401

# Team method (window-level WIE variants). wie_last/wie_first/wie_middle all
# subclass the shared reverse-SGD base (wie_window_base) and differ only in the
# epoch window + query-init point; wie_all_epochs is the per-epoch calculator.
from .wie_all_epochs import WieAllEpochsInfluenceCalculator  # noqa: F401
from .wie_last import WieLastInfluenceCalculator  # noqa: F401
from .wie_first import WieFirstInfluenceCalculator  # noqa: F401
from .wie_middle import WieMiddleInfluenceCalculator  # noqa: F401

# Previously CLI-unreachable orphan baselines -- now registered on import.
from .true_influence import TrueInfluenceCalculator  # noqa: F401
from .lie import LieInfluenceCalculator  # noqa: F401
from .nohess import NoHessInfluenceCalculator  # noqa: F401
from .segment_true import SegmentTrueFullInfluenceCalculator  # noqa: F401

# Window-level LOO ground-truth oracles (segment_true_full differencing).
from .true_first import TrueFirstInfluenceCalculator  # noqa: F401
from .true_middle import TrueMiddleInfluenceCalculator  # noqa: F401
from .true_last import TrueLastInfluenceCalculator  # noqa: F401

# CLI entrypoint
from .cli import main  # noqa: F401

__all__ = [
    # Core helpers
    "get_device",
    "get_file_paths",
    "load_global_info",
    "load_data",
    "get_input_dim",
    "compute_gradient",
    "load_epoch_data",
    "load_step_data",
    "load_model_state_from_fallback",
    "load_initial_model",
    "load_final_model",
    "save_results",
    "compute_adaptive_lambda",
    "compute_hvp_with_finite_diff",
    "infl_diff_helper",
    # Factory / base
    "InfluenceCalculator",
    "InfluenceCalculatorFactory",
    "BaseDifferenceCalculator",
    # CLI
    "main",
]
