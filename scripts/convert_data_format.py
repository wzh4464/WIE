#!/usr/bin/env python3
"""
Utility to convert old data format (.dat files) to new unified format.
- Converts .dat model lists to individual .pt checkpoints
- Extracts metrics to CSV files
- Preserves all important information
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from typing import Dict
import argparse
import logging
from datetime import datetime


def setup_logger(name: str = "converter") -> logging.Logger:
    """Setup logger for conversion process."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def convert_dat_to_checkpoints(
    dat_file: str, output_dir: str, logger: logging.Logger
) -> Dict:
    """Convert .dat file to individual checkpoint files."""
    logger.info(f"Converting {dat_file}...")

    # Load .dat file
    try:
        data = torch.load(dat_file, map_location="cpu")
    except Exception as e:
        logger.error(f"Failed to load {dat_file}: {e}")
        return {}

    # Create checkpoints directory
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    conversion_info = {
        "source_file": dat_file,
        "timestamp": datetime.now().isoformat(),
        "converted_files": [],
    }

    # Extract seed from filename
    import re

    seed_match = re.search(r"_(\d{3})\.dat$", dat_file)
    seed = seed_match.group(1) if seed_match else "000"

    # Handle different .dat file structures
    if isinstance(data, dict):
        # Extract models if present
        if "models" in data:
            models = data["models"]
            # Handle NetList wrapper if present
            if hasattr(models, "models"):
                models = models.models

            # Save each model as a checkpoint
            for i, model in enumerate(models):
                checkpoint_file = os.path.join(
                    checkpoints_dir, f"checkpoint_epoch_{i:03d}_{seed}.pt"
                )

                checkpoint = {
                    "epoch": i,
                    "model_state_dict": model.state_dict()
                    if hasattr(model, "state_dict")
                    else model,
                    "converted_from": dat_file,
                    "timestamp": datetime.now().isoformat(),
                }

                # Add metrics if available
                if "main_losses" in data and i < len(data["main_losses"]):
                    checkpoint["metrics"] = {
                        "val_loss": data["main_losses"][i]
                        if i < len(data["main_losses"])
                        else None,
                        "val_accuracy": data.get("test_accuracies", [])[i]
                        if i < len(data.get("test_accuracies", []))
                        else None,
                        "train_loss": data.get("train_losses", [])[i]
                        if i < len(data.get("train_losses", []))
                        else None,
                    }

                torch.save(checkpoint, checkpoint_file)
                conversion_info["converted_files"].append(checkpoint_file)
                logger.info(f"  Created checkpoint: {checkpoint_file}")

        # Extract metrics to CSV
        if any(k in data for k in ["main_losses", "test_accuracies", "train_losses"]):
            metrics_file = os.path.join(output_dir, f"training_metrics_{seed}.csv")

            # Determine number of epochs
            num_epochs = max(
                len(data.get("main_losses", [])),
                len(data.get("test_accuracies", [])),
                len(data.get("train_losses", [])),
            )

            metrics_df = pd.DataFrame(
                {
                    "epoch": range(num_epochs),
                    "val_loss": data.get("main_losses", [None] * num_epochs),
                    "val_accuracy": data.get("test_accuracies", [None] * num_epochs),
                    "train_loss": data.get("train_losses", [None] * num_epochs),
                }
            )

            metrics_df.to_csv(metrics_file, index=False)
            conversion_info["metrics_file"] = metrics_file
            logger.info(f"  Created metrics CSV: {metrics_file}")

    elif isinstance(data, list):
        # Handle list of models
        for i, model in enumerate(data):
            checkpoint_file = os.path.join(
                checkpoints_dir, f"checkpoint_epoch_{i:03d}_{seed}.pt"
            )

            checkpoint = {
                "epoch": i,
                "model_state_dict": model.state_dict()
                if hasattr(model, "state_dict")
                else model,
                "converted_from": dat_file,
                "timestamp": datetime.now().isoformat(),
            }

            torch.save(checkpoint, checkpoint_file)
            conversion_info["converted_files"].append(checkpoint_file)
            logger.info(f"  Created checkpoint: {checkpoint_file}")

    return conversion_info


def convert_step_files(
    records_dir: str, output_dir: str, logger: logging.Logger
) -> Dict:
    """Convert step_*.pt files to unified format."""
    logger.info(f"Converting step files from {records_dir}...")

    conversion_info = {"source_dir": records_dir, "converted_count": 0}

    # Find all step files
    step_files = [
        f
        for f in os.listdir(records_dir)
        if f.startswith("step_") and f.endswith(".pt")
    ]

    if not step_files:
        logger.info("  No step files found")
        return conversion_info

    # Group by epoch (if we can determine it)
    # For now, we'll skip step files as they're redundant with epoch checkpoints
    logger.info(f"  Found {len(step_files)} step files")
    logger.info("  Step files are redundant with epoch checkpoints in new format")
    logger.info("  Skipping step file conversion to avoid duplication")

    conversion_info["skipped_files"] = len(step_files)

    return conversion_info


def convert_influence_dat(
    dat_file: str, output_dir: str, logger: logging.Logger
) -> str:
    """Convert influence .dat file to CSV."""
    logger.info(f"Converting influence file {dat_file}...")

    # Load .dat file
    try:
        data = torch.load(dat_file, map_location="cpu")
    except Exception as e:
        logger.error(f"Failed to load {dat_file}: {e}")
        return None

    # Extract seed and type from filename
    import re

    filename = os.path.basename(dat_file)

    # Pattern: infl_<type>_<seed>.dat or infl_<type>_relabel_<pct>_pct_<seed>.dat
    match = re.search(r"infl_([^_]+)_.*_(\d{3})\.dat$", filename)
    if match:
        infl_type = match.group(1)
        seed = match.group(2)
    else:
        infl_type = "unknown"
        seed = "000"

    # Convert to numpy if tensor
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Handle different data structures
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            # Single influence array
            csv_file = os.path.join(output_dir, f"infl_{infl_type}_{seed}.csv")
            df = pd.DataFrame({"sample_idx": np.arange(len(data)), "influence": data})
            df.to_csv(csv_file, index=False)
            logger.info(f"  Created influence CSV: {csv_file}")
            return csv_file

    elif isinstance(data, list):
        # Multiple epochs/segments
        csv_file = os.path.join(output_dir, f"infl_{infl_type}_{seed}.csv")
        df = pd.DataFrame({"sample_idx": np.arange(len(data[0]))})

        for i, epoch_data in enumerate(data):
            if isinstance(epoch_data, torch.Tensor):
                epoch_data = epoch_data.cpu().numpy()
            df[f"influence_epoch_{i}"] = epoch_data

        df.to_csv(csv_file, index=False)
        logger.info(f"  Created multi-epoch influence CSV: {csv_file}")
        return csv_file

    logger.warning(f"  Unknown data structure in {dat_file}")
    return None


def convert_directory(
    input_dir: str, output_dir: str = None, logger: logging.Logger = None
) -> Dict:
    """Convert all data files in a directory to new format."""
    if logger is None:
        logger = setup_logger()

    if output_dir is None:
        output_dir = input_dir + "_converted"

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Converting directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    conversion_summary = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "timestamp": datetime.now().isoformat(),
        "converted_files": [],
    }

    # Convert model .dat files
    for file in os.listdir(input_dir):
        if file.endswith(".dat"):
            filepath = os.path.join(input_dir, file)

            # Check if it's an influence file
            if "infl_" in file:
                result = convert_influence_dat(filepath, output_dir, logger)
                if result:
                    conversion_summary["converted_files"].append(result)
            # Check if it's a model list file
            elif "model_list" in file or "sgd_models" in file:
                result = convert_dat_to_checkpoints(filepath, output_dir, logger)
                conversion_summary["converted_files"].extend(
                    result.get("converted_files", [])
                )

    # Convert step files if records directory exists
    records_dir = os.path.join(input_dir, "records")
    if os.path.exists(records_dir):
        convert_step_files(records_dir, output_dir, logger)

    # Copy other important files
    for file in ["global_info.json", "training_metrics.csv"]:
        for pattern in [file, f"*{file}"]:
            import glob

            matches = glob.glob(os.path.join(input_dir, pattern))
            for src in matches:
                if os.path.exists(src):
                    import shutil

                    dst = os.path.join(output_dir, os.path.basename(src))
                    shutil.copy2(src, dst)
                    logger.info(f"Copied {os.path.basename(src)}")

    # Save conversion summary
    summary_file = os.path.join(output_dir, "conversion_summary.json")
    with open(summary_file, "w") as f:
        json.dump(conversion_summary, f, indent=2, default=str)

    logger.info(f"Conversion complete! Summary saved to {summary_file}")

    return conversion_summary


def main():
    """Main entry point for conversion utility."""
    parser = argparse.ArgumentParser(
        description="Convert old data format (.dat files) to new unified format"
    )
    parser.add_argument("input_path", help="Input file or directory to convert")
    parser.add_argument(
        "--output", help="Output directory (default: input_path_converted)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Check input path
    if not os.path.exists(args.input_path):
        logger.error(f"Input path does not exist: {args.input_path}")
        sys.exit(1)

    # Convert based on input type
    if os.path.isdir(args.input_path):
        # Convert entire directory
        convert_directory(args.input_path, args.output, logger)
    elif args.input_path.endswith(".dat"):
        # Convert single file
        output_dir = args.output or os.path.dirname(args.input_path) + "_converted"
        os.makedirs(output_dir, exist_ok=True)

        if "infl_" in args.input_path:
            convert_influence_dat(args.input_path, output_dir, logger)
        else:
            convert_dat_to_checkpoints(args.input_path, output_dir, logger)
    else:
        logger.error("Input must be a .dat file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
