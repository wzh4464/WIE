#!/usr/bin/env python
"""
Test script for DVEInfluenceCalculator.

Usage:
    python test_dve_calculator.py --save_dir <path_to_training_output>
"""

import argparse
import logging
import os

from wie.infl import InfluenceCalculatorFactory


def main():
    parser = argparse.ArgumentParser(description="Test DVE Influence Calculator")
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to training output directory (e.g., outputs/myrun)",
    )
    parser.add_argument("--target", type=str, default="mnist", help="Dataset name")
    parser.add_argument("--model", type=str, default="dnn", help="Model type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index (-1 for CPU)")
    parser.add_argument(
        "--relabel", type=float, default=None, help="Relabel percentage"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Creating DVE Influence Calculator...")

    try:
        # Create DVE calculator
        calculator = InfluenceCalculatorFactory.create(
            infl_type="dve",
            key=args.target,
            model_type=args.model,
            seed=args.seed,
            gpu=args.gpu,
            save_dir=args.save_dir,
            relabel_percentage=args.relabel,
        )

        logger.info("Running DVE calculation...")
        calculator.run()

        logger.info("DVE calculation completed successfully!")

    except Exception as e:
        logger.error(f"Failed to run DVE calculator: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
