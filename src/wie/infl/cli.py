import argparse
import logging
import gc
import sys
import torch

from wie.data.modules import DATA_MODULE_REGISTRY  # type: ignore
from wie.models.networks import NETWORK_REGISTRY  # type: ignore
from .core import InfluenceCalculatorFactory

# Calculators that honor the extended query toolkit (--query prediction/saliency):
# the WIE window variants sharing _WieWindowInfluenceCalculator. tim_* aliases
# resolve to these via InfluenceCalculatorFactory._resolve.
_WINDOW_QUERY_TYPES = {"wie_first", "wie_middle", "wie_last"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Influence Functions (Factory Pattern)"
    )
    parser.add_argument(
        "--target", default="adult", type=str, help="Target dataset key"
    )
    parser.add_argument("--model", default="logreg", type=str, help="Model type key")
    parser.add_argument(
        "--type", default="sgd", type=str, help="Influence calculation type"
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="Random seed or negative to run 0..99"
    )
    parser.add_argument(
        "--gpu", default=0, type=int, help="GPU index (or MPS if available)"
    )
    parser.add_argument(
        "--save_dir", default=None, type=str, help="Override save directory"
    )
    parser.add_argument(
        "--relabel", default=None, type=float, help="Relabel percentage (0-100)"
    )
    # TD-Influence options
    parser.add_argument(
        "--use_projection",
        action="store_true",
        default=True,
        help="Enable random projection for TD-Influence (default: enabled)",
    )
    parser.add_argument(
        "--proj_dim",
        default=None,
        type=int,
        help="Projection dimension for TD-Influence",
    )
    parser.add_argument(
        "--proj_type",
        default="gaussian",
        type=str,
        choices=["gaussian", "achlioptas"],
        help="Projection type for TD-Influence",
    )
    parser.add_argument(
        "--use_last_layer_only",
        action="store_true",
        help="Use only the last linear layer gradients/factors (TD-Influence)",
    )
    parser.add_argument(
        "--length",
        default=3,
        type=int,
        help="Length for methods that use it (e.g., WIE)",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        type=str,
        help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    parser.add_argument(
        "--use_tensorboard", action="store_true", help="Enable TensorBoard logging"
    )
    parser.add_argument(
        "--alpha", default=None, type=float, help="L2 regularization parameter"
    )
    parser.add_argument(
        "--effective-lr",
        dest="effective_lr",
        action="store_true",
        help=(
            "WIE window variants: use the Appendix-E checkpoint effective-lr "
            "heuristic (scalar ||dtheta||/||g_bar|| per step) instead of the "
            "recorded eta_t, for non-SGD (Adam/AdamW) trajectories. Requires "
            "seeded per-step checkpoints ('step_{t}_{seed:03d}.pt'); on epoch-"
            "only or unseeded-step runs it logs a warning and falls back to the "
            "nominal lr (no effect)."
        ),
    )
    parser.add_argument(
        "--query",
        dest="query_type",
        default="loss",
        choices=["loss", "prediction", "saliency"],
        help=(
            "WIE window variants: functional query q(t) (Def. 2 / Extended "
            "Toolkit). 'loss' = val-loss gradient (default); 'prediction' = "
            "grad_theta of a class logit; 'saliency' = grad_theta of one "
            "input-gradient component."
        ),
    )
    parser.add_argument(
        "--query-index",
        dest="query_index",
        default=0,
        type=int,
        help="Validation-example index used as x_test for prediction/saliency queries.",
    )
    parser.add_argument(
        "--query-class",
        dest="query_class",
        default=None,
        type=int,
        help="Class/logit index for the prediction query (default: argmax class).",
    )
    parser.add_argument(
        "--query-coord",
        dest="query_coord",
        default=0,
        type=int,
        help="Input-gradient coordinate for the saliency query (default: 0).",
    )
    return parser.parse_args()


def _setup_logging(level: str) -> logging.Logger:
    try:
        logging.getLogger().setLevel(getattr(logging, level.upper()))
    except AttributeError:
        logging.warning(f"Invalid log level '{level}'. Defaulting to INFO.")
        logging.getLogger().setLevel(logging.INFO)
    return logging.getLogger("main")


def _build_calculator_args(args: argparse.Namespace) -> dict:
    calc_args = {
        "key": args.target,
        "model_type": args.model,
        "seed": args.seed,
        "gpu": args.gpu,
        "save_dir": args.save_dir,
        "relabel_percentage": args.relabel,
        "use_tensorboard": args.use_tensorboard,
        "length": args.length,
    }
    if args.use_projection:
        calc_args["use_projection"] = True
    if args.proj_dim is not None:
        calc_args["proj_dim"] = args.proj_dim
    if args.proj_type is not None:
        calc_args["proj_type"] = args.proj_type
    if args.use_last_layer_only:
        calc_args["use_last_layer_only"] = True
    if args.alpha is not None:
        calc_args["alpha"] = args.alpha
    if getattr(args, "effective_lr", False):
        calc_args["use_effective_lr"] = True
    if getattr(args, "query_type", "loss") != "loss":
        calc_args["query_type"] = args.query_type
        calc_args["query_index"] = args.query_index
        calc_args["query_class"] = args.query_class
        calc_args["query_coord"] = args.query_coord
    return calc_args


def _run_once(
    method: str, base_args: dict, seed_val: int, logger: logging.Logger
) -> bool:
    current_args = dict(base_args)
    current_args["seed"] = seed_val
    try:
        calculator = InfluenceCalculatorFactory.create(method, **current_args)
        calculator.run()
        return True
    except ValueError as e:
        logger.error(f"[Seed {seed_val}] Configuration error: {e}")
        return False
    except NotImplementedError as e:
        logger.error(
            f"[Seed {seed_val}] Calculation type '{method}' not fully implemented: {e}"
        )
        return False
    except Exception as e:
        logger.error(f"[Seed {seed_val}] Calculation failed: {e}", exc_info=True)
        return False
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    logger_main = _setup_logging(args.log_level)

    # Invalid target/model are error paths: exit nonzero so a subprocess caller
    # (check=True) sees the failure instead of proceeding on a stale/absent CSV.
    if args.target not in DATA_MODULE_REGISTRY:
        logger_main.error(
            f"Invalid target data '{args.target}'. Choose from {list(DATA_MODULE_REGISTRY.keys())}."
        )
        sys.exit(1)
    if args.model not in NETWORK_REGISTRY:
        logger_main.error(
            f"Invalid model type '{args.model}'. Choose from {list(NETWORK_REGISTRY.keys())}."
        )
        sys.exit(1)

    # The extended query toolkit is only honored by the WIE window calculators
    # (wie_first/middle/last); passing --query to any other type would compute
    # default scores yet save them under a query-suffixed name. Reject it.
    if getattr(args, "query_type", "loss") != "loss":
        resolved = InfluenceCalculatorFactory._resolve(args.type)
        if resolved not in _WINDOW_QUERY_TYPES:
            logger_main.error(
                f"--query {args.query_type} is only supported for the WIE window "
                f"calculators {sorted(_WINDOW_QUERY_TYPES)}; got --type "
                f"'{args.type}'. Re-run with a window type or --query loss."
            )
            sys.exit(1)

    calc_args = _build_calculator_args(args)

    if args.seed >= 0:
        logger_main.info(
            f"Type: {args.type}, Dataset: {args.target}, Model: {args.model}, Seed: {args.seed}"
        )
        # A failed calculation (unknown type, empty-window / length<1 guard, or
        # any run() error) must exit NONZERO. Otherwise the subprocess returns 0
        # and callers (run_pipeline check=True, --skip_train reruns) cleanse a
        # stale/absent CSV and "succeed", even deleting checkpoints.
        if not _run_once(args.type, calc_args, args.seed, logger_main):
            logger_main.error(
                f"--- Calculation FAILED for seed {args.seed} (no scores written); "
                "exiting nonzero ---"
            )
            sys.exit(1)
        logger_main.info(f"--- Calculation successful for seed {args.seed} ---")
        return

    logger_main.info(
        f"Seed < 0 detected. Running for seeds 0 to 99 for type '{args.type}'..."
    )
    successful, failed = 0, []
    for s in range(100):
        logger_main.info(f"--- Running for Seed {s} ---")
        if _run_once(args.type, calc_args, s, logger_main):
            successful += 1
        else:
            failed.append(s)
    logger_main.info("--- Seed Loop Summary ---")
    logger_main.info(f"Successfully completed seeds: {successful}/100")
    # Multi-seed: run every seed, but surface partial success as a nonzero exit
    # so a failed seed's missing output isn't mistaken for success downstream.
    if failed:
        logger_main.warning(f"Failed seeds: {failed}")
        sys.exit(1)


__all__ = ["main", "InfluenceCalculatorFactory"]
