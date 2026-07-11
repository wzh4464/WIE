#!/usr/bin/env python3
"""
Multi-seed LOO (Leave-One-Out) experiments runner with GPU distribution.

This script runs the complete LOO experimental pipeline across multiple seeds
and distributes the workload across available GPUs.

Pipeline steps for each seed:
1. Train model with DVE enabled
2. Copy and create LOO valuation matrix
3. Run influence calculations: DVE, WIE, LAVA, ICML, LOO
"""

import argparse
import subprocess
import sys
import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

# Make the src/ layout importable when this script is run by path
# (python scripts/foo.py) without the editable install (pixi run install).
try:
    import wie  # noqa: F401
except ModuleNotFoundError:
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from wie.utils.paths import resolve_output_dir


def run_command(cmd: str, dry_run: bool = False) -> Tuple[bool, str]:
    """Execute a command and return success status and output."""
    if dry_run:
        print(f"[DRY-RUN] {cmd}")
        return True, ""

    print(f"[RUNNING] {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"[SUCCESS] {cmd}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Command failed: {cmd}\nReturn code: {e.returncode}\nStderr: {e.stderr}"
        )
        print(f"[ERROR] {error_msg}")
        return False, error_msg


def copy_directory(src: str, dst: str, dry_run: bool = False) -> Tuple[bool, str]:
    """Copy directory from src to dst.

    Paths are resolved using centralized resolve_output_dir for consistency.
    """
    src_resolved = str(resolve_output_dir(src))
    dst_resolved = str(resolve_output_dir(dst))

    if dry_run:
        print(f"[DRY-RUN] cp -r {src_resolved} {dst_resolved}")
        return True, ""

    print(f"[RUNNING] cp -r {src_resolved} {dst_resolved}")
    try:
        if os.path.exists(dst_resolved):
            shutil.rmtree(dst_resolved)
        shutil.copytree(src_resolved, dst_resolved)
        print(f"[SUCCESS] Copied {src_resolved} to {dst_resolved}")
        return True, ""
    except Exception as e:
        error_msg = f"Failed to copy {src_resolved} to {dst_resolved}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return False, error_msg


def run_seed_experiment(
    seed: int, gpu: int, base_dir: str, dry_run: bool = False
) -> bool:
    """Run complete experiment pipeline for a single seed."""
    print(f"\n{'=' * 60}")
    print(f"Starting experiment for seed {seed} on GPU {gpu}")
    print(f"{'=' * 60}")

    # Directory names with seed suffix
    base_name = f"loo_similar_256_seed_{seed}"
    dirs = {
        "with_dve": f"{base_name}_with_dve",
        "base": f"{base_name}",
        "dve": f"{base_name}_dve",
        "wie": f"{base_name}_wie",
        "lava": f"{base_name}_lava",
        "icml": f"{base_name}_icml",
        "loo": f"{base_name}_loo",
    }

    steps = [
        # Step 1: Train model with DVE
        (
            "Training with DVE",
            lambda: run_command(
                f"pixi run python -m wie.training.train --target mnist --model dnn "
                f"--num_epoch 21 --n_tr 256 --seed {seed} --dve-enable "
                f"--save_dir {dirs['with_dve']} --gpu {gpu}",
                dry_run,
            ),
        ),
        # Step 2: Copy to base directory
        (
            "Copying to base directory",
            lambda: copy_directory(
                dirs['with_dve'], dirs['base'], dry_run
            ),
        ),
        # # Step 3: Create LOO valuation matrix
        # ("Creating LOO valuation matrix", lambda: run_command(
        #     f"pixi run python create_loo_valuation_matrix.py "
        #     f"--exp_dir outputs/{dirs['base']} --seed {seed}", dry_run
        # )),
        # Step 4: Copy to DVE directory and run DVE influence
        (
            "Copying to DVE directory",
            lambda: copy_directory(
                dirs['base'], dirs['dve'], dry_run
            ),
        ),
        (
            "Running DVE influence",
            lambda: run_command(
                f"pixi run python -m wie.infl --target mnist --model dnn "
                f"--save_dir {dirs['dve']} --relabel 0 --type dve_all_epochs "
                f"--use_projection --seed {seed} --gpu {gpu}",
                dry_run,
            ),
        ),
        # Step 5: Copy to WIE directory and run WIE influence
        (
            "Copying to WIE directory",
            lambda: copy_directory(
                dirs['dve'], dirs['wie'], dry_run
            ),
        ),
        (
            "Running WIE influence",
            lambda: run_command(
                f"pixi run python -m wie.infl --target mnist --model dnn "
                f"--save_dir {dirs['wie']} --relabel 0 --type wie_all_epochs "
                f"--seed {seed} --gpu {gpu}",
                dry_run,
            ),
        ),
        # Step 6: Copy to LAVA directory and run LAVA influence
        (
            "Copying to LAVA directory",
            lambda: copy_directory(
                dirs['wie'], dirs['lava'], dry_run
            ),
        ),
        (
            "Running LAVA influence",
            lambda: run_command(
                f"pixi run python -m wie.infl --target mnist --model dnn "
                f"--save_dir {dirs['lava']} --relabel 0 --type lava_all_epochs "
                f"--seed {seed} --gpu {gpu}",
                dry_run,
            ),
        ),
        # Step 7: Copy to ICML directory and run ICML influence
        (
            "Copying to ICML directory",
            lambda: copy_directory(
                dirs['lava'], dirs['icml'], dry_run
            ),
        ),
        (
            "Running ICML influence",
            lambda: run_command(
                f"pixi run python -m wie.infl --target mnist --model dnn "
                f"--save_dir {dirs['icml']} --relabel 0 --type icml_all_epochs "
                f"--seed {seed} --gpu {gpu}",
                dry_run,
            ),
        ),
        # Step 8: Copy to LOO directory and run LOO influence
        (
            "Copying to LOO directory",
            lambda: copy_directory(
                dirs['icml'], dirs['loo'], dry_run
            ),
        ),
        (
            "Running LOO influence",
            lambda: run_command(
                f"pixi run python -m wie.infl --target mnist --model dnn "
                f"--save_dir {dirs['loo']} --relabel 0 --type loo_all_epochs "
                f"--seed {seed} --gpu {gpu}",
                dry_run,
            ),
        ),
    ]

    # Execute all steps
    for step_name, step_func in steps:
        print(f"\n[SEED {seed}] {step_name}...")
        success, output = step_func()
        if not success:
            print(f"[SEED {seed}] FAILED at step: {step_name}")
            print(f"[SEED {seed}] Error: {output}")
            return False

    print(f"\n[SEED {seed}] ✓ All steps completed successfully!")
    return True


def run_gpu_queue(
    gpu: int, seeds: List[int], base_dir: str, dry_run: bool = False
) -> int:
    """Run experiments for all seeds assigned to a specific GPU sequentially."""
    success_count = 0

    print(f"\n[GPU {gpu}] Starting to process {len(seeds)} seeds: {seeds}")

    for i, seed in enumerate(seeds):
        print(f"\n[GPU {gpu}] Processing seed {seed} ({i + 1}/{len(seeds)})")

        success = run_seed_experiment(seed, gpu, base_dir, dry_run)
        if success:
            success_count += 1
            print(f"[GPU {gpu}] ✓ Seed {seed} completed successfully")
        else:
            print(f"[GPU {gpu}] ✗ Seed {seed} failed")

    print(
        f"\n[GPU {gpu}] Finished processing all seeds. Success: {success_count}/{len(seeds)}"
    )
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-seed LOO experiments with GPU distribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/run_multi_seed_loo_experiments.py --parallel
  python scripts/run_multi_seed_loo_experiments.py --full-parallel
  python scripts/run_multi_seed_loo_experiments.py --seeds 42 43 44 45 --dry-run
  python scripts/run_multi_seed_loo_experiments.py --seeds 1 2 3 4 --gpus 0 1 --parallel
        """,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(1, 17)),
        help="Seeds to run experiments for (default: [1, 2, ..., 16])",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="GPUs to use (default: [0, 1, 2, 3])",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel across GPUs",
    )
    parser.add_argument(
        "--full-parallel",
        action="store_true",
        help="Run all seeds in full parallel (no queueing within GPU)",
    )
    parser.add_argument(
        "--base-dir",
        default="outputs",
        help="Base directory for outputs (default: outputs)",
    )

    args = parser.parse_args()

    print("Multi-seed LOO Experiments")
    print(f"Seeds: {args.seeds}")
    print(f"GPUs: {args.gpus}")
    print(f"Parallel: {args.parallel}")
    print(f"Full parallel: {args.full_parallel}")
    print(f"Dry run: {args.dry_run}")
    print(f"Base directory: {args.base_dir}")

    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN MODE - Commands will be printed but not executed")
        print("=" * 60)

    # Create outputs directory if it doesn't exist
    if not args.dry_run:
        os.makedirs(args.base_dir, exist_ok=True)

    success_count = 0
    total_experiments = len(args.seeds)

    if args.full_parallel:
        # Full parallel mode: all seeds run simultaneously
        print(f"\nRunning {total_experiments} experiments in full parallel mode...")
        print(f"Available GPUs: {args.gpus}")

        with ThreadPoolExecutor(max_workers=total_experiments) as executor:
            future_to_seed = {}

            # Submit all jobs with round-robin GPU assignment
            for i, seed in enumerate(args.seeds):
                gpu = args.gpus[i % len(args.gpus)]
                future = executor.submit(
                    run_seed_experiment, seed, gpu, args.base_dir, args.dry_run
                )
                future_to_seed[future] = seed

            # Wait for completion and collect results
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                        print(f"\n✓ Seed {seed} completed successfully")
                    else:
                        print(f"\n✗ Seed {seed} failed")
                except Exception as e:
                    print(f"\n✗ Seed {seed} failed with exception: {e}")

    elif args.parallel:
        # Smart GPU scheduling: parallel across different GPUs, queued on same GPU
        from collections import defaultdict

        print(f"\nRunning {total_experiments} experiments with smart GPU scheduling...")
        print(f"Available GPUs: {args.gpus}")

        # Group seeds by GPU assignment (round-robin)
        gpu_queues = defaultdict(list)
        for i, seed in enumerate(args.seeds):
            gpu = args.gpus[i % len(args.gpus)]
            gpu_queues[gpu].append(seed)

        print("GPU assignment:")
        for gpu, seeds in gpu_queues.items():
            print(f"  GPU {gpu}: seeds {seeds}")

        # Create thread pool with one worker per GPU
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            future_to_info = {}

            # Submit one job per GPU that processes its queue of seeds
            for gpu, seeds in gpu_queues.items():
                future = executor.submit(
                    run_gpu_queue, gpu, seeds, args.base_dir, args.dry_run
                )
                future_to_info[future] = {"gpu": gpu, "seeds": seeds}

            # Wait for completion and collect results
            for future in as_completed(future_to_info):
                info = future_to_info[future]
                gpu = info["gpu"]
                seeds = info["seeds"]
                try:
                    gpu_success_count = future.result()
                    success_count += gpu_success_count
                    print(
                        f"\n✓ GPU {gpu} completed {gpu_success_count}/{len(seeds)} seeds successfully"
                    )
                except Exception as e:
                    print(f"\n✗ GPU {gpu} failed with exception: {e}")

    else:
        # Sequential execution
        print(f"\nRunning {total_experiments} experiments sequentially...")

        for i, seed in enumerate(args.seeds):
            gpu = args.gpus[i % len(args.gpus)]
            print(f"\nExperiment {i + 1}/{total_experiments}")

            success = run_seed_experiment(seed, gpu, args.base_dir, args.dry_run)
            if success:
                success_count += 1
            else:
                print(
                    f"\nExperiment for seed {seed} failed. Continuing with next seed..."
                )

    # Summary
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_experiments - success_count}")

    if success_count == total_experiments:
        print("🎉 All experiments completed successfully!")
        sys.exit(0)
    else:
        print("⚠️  Some experiments failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
