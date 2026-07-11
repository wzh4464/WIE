#!/usr/bin/env python3
"""
Helper script to create custom grid configuration files for influence cleansing experiments.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def create_grid_config(
    methods: List[str],
    keep_ratios: List[int],
    relabel_percentages: List[int],
    seeds: List[int],
) -> Dict[str, Any]:
    """Create a grid configuration dictionary."""
    config = {
        "methods": methods,
        "keep_ratios": keep_ratios,
        "relabel_percentages": relabel_percentages,
        "seeds": seeds
    }
    return config


def save_config(config: Dict[str, Any], output_file: str) -> None:
    """Save configuration to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Grid configuration saved to: {output_file}")
    print("📄 Configuration contents:")
    print(json.dumps(config, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Create custom grid configuration for cleansing experiments")

    # Methods
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["wie_all_epochs", "icml_all_epochs", "lava_all_epochs", "dve_all_epochs"],
        help="Influence methods to include in grid"
    )

    # Keep ratios
    parser.add_argument(
        "--keep-ratios",
        nargs="+",
        type=int,
        default=[70, 80, 90, 95],
        help="Keep ratio percentages to include"
    )

    # Relabel percentages
    parser.add_argument(
        "--relabel-percentages",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
        help="Relabel percentages to include"
    )

    # Seeds
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds to include"
    )

    # Output file
    parser.add_argument(
        "-o", "--output",
        default="configs/custom_grid.json",
        help="Output file path"
    )

    # Predefined configurations
    parser.add_argument(
        "--preset",
        choices=["full", "small", "wie-only", "fast", "precision-only"],
        help="Use a predefined configuration preset"
    )

    args = parser.parse_args()

    # Handle presets
    if args.preset == "full":
        config = create_grid_config(
            methods=["wie_all_epochs", "icml_all_epochs", "lava_all_epochs", "dve_all_epochs"],
            keep_ratios=[70, 80, 90, 95],
            relabel_percentages=[5, 10, 15, 20],
            seeds=[0, 1, 2]
        )
    elif args.preset == "small":
        config = create_grid_config(
            methods=["wie_all_epochs", "icml_all_epochs"],
            keep_ratios=[80, 90],
            relabel_percentages=[10, 20],
            seeds=[0, 1]
        )
    elif args.preset == "wie-only":
        config = create_grid_config(
            methods=["wie_all_epochs"],
            keep_ratios=[70, 80, 90, 95],
            relabel_percentages=[5, 10, 15, 20, 25],
            seeds=[0, 1, 2, 3, 4]
        )
    elif args.preset == "fast":
        config = create_grid_config(
            methods=["wie_all_epochs", "icml_all_epochs"],
            keep_ratios=[90],
            relabel_percentages=[10, 20],
            seeds=[0]
        )
    elif args.preset == "precision-only":
        config = create_grid_config(
            methods=["wie_all_epochs", "icml_all_epochs", "lava_all_epochs", "dve_all_epochs"],
            keep_ratios=[70, 80, 90, 95],
            relabel_percentages=[5, 10, 15, 20, 25, 30],
            seeds=[0, 1, 2, 3]
        )
    else:
        # Use command line arguments
        config = create_grid_config(
            methods=args.methods,
            keep_ratios=args.keep_ratios,
            relabel_percentages=args.relabel_percentages,
            seeds=args.seeds
        )

    # Calculate total experiments
    total = len(config["methods"]) * len(config["keep_ratios"]) * len(config["relabel_percentages"]) * len(config["seeds"])
    print(f"📊 Total experiments: {total}")
    print(f"   Methods: {len(config['methods'])}")
    print(f"   Keep ratios: {len(config['keep_ratios'])}")
    print(f"   Relabel percentages: {len(config['relabel_percentages'])}")
    print(f"   Seeds: {len(config['seeds'])}")
    print()

    # Save configuration
    save_config(config, args.output)

    print()
    print("🚀 To run this configuration:")
    print(f"   bash scripts/run_cleansing_grid.sh -g {args.output}")
    print()
    print("🔍 To do a dry run first:")
    print(f"   bash scripts/run_cleansing_grid.sh -g {args.output} --dry-run")


if __name__ == "__main__":
    main()