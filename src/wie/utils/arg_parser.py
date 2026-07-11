import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the training script.

    Groups related arguments to improve readability while preserving
    the original flags and defaults for backward compatibility.
    """
    parser = argparse.ArgumentParser(description="Train Models & Save")

    # Dataset/model
    parser.add_argument("--target", default="adult", type=str, help="target data")
    parser.add_argument("--model", default="logreg", type=str, help="model type")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--gpu", default=0, type=int, help="gpu index")

    # Training sizes and hyperparameters
    parser.add_argument("--n_tr", type=int, help="number of training samples")
    parser.add_argument("--n_val", type=int, help="number of validation samples")
    parser.add_argument("--n_test", type=int, help="number of test samples")
    parser.add_argument("--num_epoch", type=int, help="number of epochs")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--lr", type=float, help="initial learning rate")
    parser.add_argument(
        "--decay",
        type=_bool_like,
        default=False,
        help="Enable learning rate decay: True/False/Yes/No/1/0 (default: False)",
    )
    # Regularization knobs
    parser.add_argument(
        "--label_smoothing",
        type=float,
        help="Label smoothing epsilon for BCE [0,0.5)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="Classifier head dropout probability [0,1) (BERT)",
    )

    # Recording and IO
    parser.add_argument(
        "--save_dir", type=str, help="directory to save models and results"
    )
    parser.add_argument(
        "--no-loo",
        action="store_false",
        dest="compute_counterfactual",
        help="Disable computation of counterfactual models (leave-one-out)",
    )
    parser.add_argument(
        "--no-recording",
        action="store_false",
        dest="save_recording",
        help="Disable saving full recordings (.dat). Only save metrics CSV",
    )
    parser.add_argument(
        "--steps-only",
        action="store_true",
        dest="steps_only",
        help="Only record steps (no epochs/overall model)",
    )
    parser.add_argument(
        "--epochs-only",
        action="store_true",
        dest="epochs_only",
        help="Only record epochs (disable step-level checkpoints)",
    )
    parser.add_argument(
        "--init_model",
        type=str,
        help="Path to initialization model (uses last model in provided list)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )

    # Relabeling
    parser.add_argument(
        "--relabel", type=int, help="percentage of training data to relabel"
    )
    parser.add_argument(
        "--relabel_csv", type=str, help="CSV with indices of samples to relabel"
    )

    # DVE (Data Valuation for Evaluation) options
    parser.add_argument(
        "--dve-enable",
        action="store_true",
        dest="dve_enable",
        help="Enable DVE (Data Valuation for Evaluation) during training",
    )
    parser.add_argument(
        "--dve-proj-dim",
        type=int,
        default=128,
        dest="dve_proj_dim",
        help="DVE projection dimension (default: 128)",
    )
    parser.add_argument(
        "--dve-proj-type",
        type=str,
        default="gaussian",
        dest="dve_proj_type",
        choices=["gaussian", "achlioptas"],
        help="DVE projection type: gaussian or achlioptas (default: gaussian)",
    )
    parser.add_argument(
        "--dve-granularity",
        type=str,
        default="step",
        dest="dve_granularity",
        choices=["step", "epoch"],
        help="DVE recording granularity: step or epoch (default: step)",
    )
    parser.add_argument(
        "--dve-last-layer-only",
        action="store_true",
        dest="dve_last_layer_only",
        default=True,
        help="DVE only for last layer (minimal viable DVE, default: True)",
    )
    parser.add_argument(
        "--dve-fp16",
        action="store_true",
        dest="dve_fp16",
        default=True,
        help="Save DVE shards in FP16 format (default: True)",
    )
    parser.add_argument(
        "--dve-sample-rate",
        type=float,
        default=1.0,
        dest="dve_sample_rate",
        help="DVE sample rate for epoch-mode training data subsampling (0<r<=1, default: 1.0)",
    )

    # Defaults
    parser.set_defaults(
        compute_counterfactual=True,
        save_recording=True,
        steps_only=False,
        epochs_only=False,
        dve_enable=False,
        dve_last_layer_only=True,
        dve_fp16=True,
    )

    return parser


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    return build_parser().parse_args()


def _bool_like(x: str) -> bool:
    """Parse common truthy/falsey strings into a boolean.

    Accepts: yes/true/t/y/1 or no/false/f/n/0 (case-insensitive).
    """
    s = str(x).strip().lower()
    if s in {"yes", "true", "t", "y", "1"}:
        return True
    if s in {"no", "false", "f", "n", "0"}:
        return False
    # Fallback: keep False to preserve the previous behavior
    return False
