import os
import numpy as np
import pandas as pd
import logging


def generate_relabel_indices(
    n_samples: int, relabel_percentage: float, seed: int
) -> np.ndarray:
    """
    Generate indices of samples to be relabeled.

    Args:
        n_samples: Total number of samples
        relabel_percentage: Percentage of samples to relabel
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Indices of samples to be relabeled
    """
    np.random.seed(seed)
    num_to_relabel = int(n_samples * relabel_percentage / 100)
    return np.random.choice(n_samples, num_to_relabel, replace=False)


def save_relabel_indices(
    indices: np.ndarray, save_dir: str, seed: int, relabel_percentage: int | None = None
) -> str:
    """
    Save relabel indices to a CSV file.

    Args:
        indices: Indices of samples to be relabeled
        save_dir: Directory to save the file
        seed: Random seed for the filename
        relabel_percentage: Percentage of samples that were relabeled

    Returns:
        str: Path to the saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    if relabel_percentage is None:
        filename = os.path.join(save_dir, f"indices_{seed:03d}.csv")
    else:
        filename = os.path.join(
            save_dir, f"relabel_{relabel_percentage:03d}_pct_indices_{seed:03d}.csv"
        )

    pd.DataFrame({"relabel_indices": indices}).to_csv(filename, index=False)
    return filename


def load_relabel_indices(filename: str) -> np.ndarray:
    """
    Load relabel indices from a CSV file.

    Args:
        filename: Path to the CSV file

    Returns:
        np.ndarray: Indices of samples to be relabeled
    """
    df = pd.read_csv(filename)
    return df["relabel_indices"].values


def handle_relabeling(
    args,
    save_dir: str,
    logger: logging.Logger,
    initialize_data_and_params,
    current_dir: str,
):
    """
    Handle relabeling based on command line arguments and return relabel CSV, prefix, and percentage.

    Args:
        args: Command line arguments
        save_dir: Directory to save relabel indices
        logger: Logger instance
        initialize_data_and_params: Function to initialize data and parameters
        current_dir: Current directory path

    Returns:
        tuple: (relabel_csv, relabel_prefix, relabel_percentage)
    """
    relabel_csv = None
    relabel_percentage = None
    relabel_prefix = ""

    if args.relabel is not None and args.relabel_csv is None:
        # We need to know n_tr to generate relabel indices
        _, data_sizes, _ = initialize_data_and_params(
            args.target,
            args.model,
            os.path.join(current_dir, "data"),
            logger,
            args.seed,
        )
        n_tr = args.n_tr or data_sizes["n_tr"]

        relabel_indices = generate_relabel_indices(n_tr, args.relabel, args.seed)
        relabel_csv = save_relabel_indices(
            relabel_indices, save_dir, args.seed, args.relabel
        )
        logger.info(
            f"Generated and saved relabel indices for {args.relabel}% of data to {relabel_csv}"
        )
    elif args.relabel_csv is not None:
        relabel_csv = args.relabel_csv
        logger.info(f"Using provided relabel CSV file: {args.relabel_csv}")
    else:
        logger.info("No relabeling will be applied")
        return None, "", None

    # Parse relabel_prefix and relabel_percentage
    if relabel_csv:
        relabel_filename = os.path.basename(relabel_csv)
        logger.debug(f"relabel_filename: {relabel_filename}")
        relabel_prefix = os.path.splitext(relabel_filename)[0] + "_"
        relabel_prefix = relabel_prefix.split("_")[:-2]
        relabel_prefix = "_".join(relabel_prefix) + "_"  # relabel_10_pct_
        logger.debug(f"relabel_prefix: {relabel_prefix}")
        relabel_percentage = None
        if "pct_" in relabel_filename:
            try:
                pct_part = relabel_filename.split("_pct_")[0].split("_")[-1]
                relabel_percentage = int(pct_part)
                relabel_prefix = f"relabel_{relabel_percentage:03d}_pct_"
            except (ValueError, IndexError):
                logger.warning(
                    f"Could not extract relabel percentage from filename: {relabel_filename}"
                )
        logger.debug(f"relabel_prefix: {relabel_prefix}")
    return relabel_csv, relabel_prefix, relabel_percentage
