#!/usr/bin/env python3
"""
Multi-seed experiment runner for epoch-wise keep ratio experiments.

This script runs the epoch_wise_keep_ratio.py script across multiple seeds,
distributing the workload across available GPUs from the GPU list configuration.

Usage:
    pixi run python -m scripts.run_multi_seed_experiment --target sentiment --model bert --save_dir sentiment_bert_dve --relabel 30 --gpu 2 --type dve_all_epochs --log_level INFO --n_tr 16384 --n_val 2048 --num_epoch 10 --lr 1e-6 --seeds 16
"""

import argparse
import os
import sys
import subprocess
import yaml
import time
import threading
from typing import List, Dict, Any
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def load_gpu_config(config_path: str) -> List[int]:
    """Load GPU list from YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("gpu_list", [0]) if isinstance(config, dict) else [0]
    except FileNotFoundError:
        print(
            f"Warning: GPU config file not found at {config_path}, using default GPU 0"
        )
        return [0]
    except Exception as e:
        print(f"Warning: Error loading GPU config: {e}, using default GPU 0")
        return [0]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for multi-seed experiments."""
    parser = argparse.ArgumentParser(
        description="Multi-seed epoch-wise cleansing experiment"
    )

    # Required parameters (same as epoch_wise_keep_ratio.py)
    parser.add_argument("--target", type=str, required=True, help="Target dataset")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--save_dir", type=str, required=True, help="Save directory")
    parser.add_argument("--relabel", type=int, required=True, help="Relabel parameter")
    parser.add_argument("--type", type=str, required=True, help="Influence method")

    # Multi-seed specific parameters
    parser.add_argument(
        "--seeds", type=int, default=16, help="Number of seeds to run (default: 16)"
    )
    parser.add_argument(
        "--start_seed", type=int, default=0, help="Starting seed number (default: 0)"
    )
    parser.add_argument(
        "--gpu_config",
        type=str,
        default="configs/gpu_list.yaml",
        help="Path to GPU configuration file",
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=None,
        help="Maximum parallel jobs (default: number of GPUs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would run, then exit",
    )

    # Pass-through parameters for epoch_wise_keep_ratio.py
    parser.add_argument(
        "--decay", type=str, default="False", help="LR decay (True/False)"
    )
    parser.add_argument(
        "--keep_ratio", type=int, default=90, help="Percent to keep (default 90)"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--n_tr", type=int, help="Number of training samples")
    parser.add_argument("--n_val", type=int, help="Number of validation samples")
    parser.add_argument("--num_epoch", type=int, default=5, help="Training epochs")
    parser.add_argument(
        "--dropout", type=float, default=None, help="Head dropout prob for BERT [0,1)"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=None,
        help="Label smoothing epsilon for BCE [0,0.5)",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--max_length", type=int, default=128, help="Max sequence length"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=500, help="Warmup steps"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Pretrained model name",
    )
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Grad accumulation steps",
    )
    parser.add_argument(
        "--validation_split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision")
    parser.add_argument(
        "--use_tensorboard", action="store_true", help="Use TensorBoard logging"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases logging"
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps", "no"],
        help="Evaluation strategy",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="best",
        choices=["best", "epoch", "steps", "no"],
        help="Save strategy",
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        default=True,
        help="Load best model at end",
    )
    parser.add_argument(
        "--save_recording", action="store_true", default=True, help="Save recording"
    )
    parser.add_argument("--steps_only", action="store_true", help="Steps only")
    parser.add_argument("--epochs_only", action="store_true", help="Epochs only")
    parser.add_argument(
        "--compute_counterfactual", action="store_true", help="Compute counterfactual"
    )
    parser.add_argument(
        "--init_model", type=str, default=None, help="Initial model path"
    )
    parser.add_argument(
        "--relabel_csv", type=str, default=None, help="Relabel CSV file"
    )
    parser.add_argument(
        "--use_projection",
        action="store_true",
        help="Enable random projection for TD-Influence",
    )
    parser.add_argument(
        "--proj_dim", type=int, default=None, help="Projection dimension"
    )
    parser.add_argument(
        "--proj_type",
        type=str,
        default=None,
        choices=["gaussian", "achlioptas"],
        help="Projection type",
    )
    parser.add_argument(
        "--use_last_layer_only",
        action="store_true",
        help="Last layer only (TD-Influence)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training step and use existing training data",
    )
    parser.add_argument(
        "--existing_train_dir",
        type=str,
        help="Path to existing training data directory (required when --skip_train is used)",
    )

    return parser


def build_epoch_wise_cmd(args: argparse.Namespace, seed: int, gpu: int) -> List[str]:
    """Build command for epoch_wise_keep_ratio.py with specific seed and GPU."""
    cmd = [
        sys.executable,
        "-m",
        "scripts.epoch_wise_keep_ratio",
        "--target",
        args.target,
        "--model",
        args.model,
        "--save_dir",
        f"{args.save_dir}_seed_{seed:03d}",
        "--relabel",
        str(args.relabel),
        "--seed",
        str(seed),
        "--gpu",
        str(gpu),
        "--type",
        args.type,
        "--log_level",
        args.log_level,
    ]

    # Add all pass-through parameters
    if args.decay != "False":
        cmd.extend(["--decay", args.decay])
    if args.keep_ratio != 90:
        cmd.extend(["--keep_ratio", str(args.keep_ratio)])
    if args.lr != 2e-5:
        cmd.extend(["--lr", str(args.lr)])
    if args.n_tr is not None:
        cmd.extend(["--n_tr", str(args.n_tr)])
    if args.n_val is not None:
        cmd.extend(["--n_val", str(args.n_val)])
    if args.num_epoch != 5:
        cmd.extend(["--num_epoch", str(args.num_epoch)])
    if args.dropout is not None:
        cmd.extend(["--dropout", str(args.dropout)])
    if args.label_smoothing is not None:
        cmd.extend(["--label_smoothing", str(args.label_smoothing)])
    if args.batch_size != 16:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.max_length != 128:
        cmd.extend(["--max_length", str(args.max_length)])
    if args.weight_decay != 0.01:
        cmd.extend(["--weight_decay", str(args.weight_decay)])
    if args.num_warmup_steps != 500:
        cmd.extend(["--num_warmup_steps", str(args.num_warmup_steps)])
    if args.model_name_or_path != "bert-base-uncased":
        cmd.extend(["--model_name_or_path", args.model_name_or_path])
    if args.num_labels != 2:
        cmd.extend(["--num_labels", str(args.num_labels)])
    if args.gradient_accumulation_steps != 1:
        cmd.extend(
            ["--gradient_accumulation_steps", str(args.gradient_accumulation_steps)]
        )
    if args.validation_split != 0.1:
        cmd.extend(["--validation_split", str(args.validation_split)])
    if args.fp16:
        cmd.append("--fp16")
    if args.bf16:
        cmd.append("--bf16")
    if args.use_tensorboard:
        cmd.append("--use_tensorboard")
    if args.use_wandb:
        cmd.append("--use_wandb")
    if args.eval_strategy != "epoch":
        cmd.extend(["--eval_strategy", args.eval_strategy])
    if args.save_strategy != "best":
        cmd.extend(["--save_strategy", args.save_strategy])
    if not args.load_best_model_at_end:
        cmd.append("--no-load_best_model_at_end")
    if not args.save_recording:
        cmd.append("--no-save_recording")
    if args.steps_only:
        cmd.append("--steps_only")
    if args.epochs_only:
        cmd.append("--epochs_only")
    if args.compute_counterfactual:
        cmd.append("--compute_counterfactual")
    if args.init_model is not None:
        cmd.extend(["--init_model", args.init_model])
    if args.relabel_csv is not None:
        cmd.extend(["--relabel_csv", args.relabel_csv])
    if args.use_projection:
        cmd.append("--use_projection")
    if args.proj_dim is not None:
        cmd.extend(["--proj_dim", str(args.proj_dim)])
    if args.proj_type is not None:
        cmd.extend(["--proj_type", args.proj_type])
    if args.use_last_layer_only:
        cmd.append("--use_last_layer_only")
    if args.skip_train:
        cmd.append("--skip_train")
        if args.existing_train_dir is not None:
            cmd.extend(["--existing_train_dir", args.existing_train_dir])

    return cmd


def run_single_experiment(
    args: argparse.Namespace, seed: int, gpu: int, logger: logging.Logger
) -> Dict[str, Any]:
    """Run a single experiment with given seed and GPU."""
    cmd = build_epoch_wise_cmd(args, seed, gpu)
    start_time = time.time()

    logger.info(f"Starting experiment: seed={seed}, gpu={gpu}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=project_root
        )

        duration = time.time() - start_time
        logger.info(
            f"Completed experiment: seed={seed}, gpu={gpu}, duration={duration:.2f}s"
        )

        return {
            "seed": seed,
            "gpu": gpu,
            "status": "success",
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        logger.error(
            f"Failed experiment: seed={seed}, gpu={gpu}, duration={duration:.2f}s"
        )
        logger.error(f"Error: {e.stderr}")

        return {
            "seed": seed,
            "gpu": gpu,
            "status": "failed",
            "duration": duration,
            "stdout": e.stdout,
            "stderr": e.stderr,
            "returncode": e.returncode,
        }


def setup_logging(log_level: str) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("multi_seed_experiment.log"),
        ],
    )
    return logging.getLogger(__name__)


def run_multi_seed_experiments(args: argparse.Namespace) -> None:
    """Run experiments across multiple seeds and GPUs."""
    logger = setup_logging(args.log_level)

    # Load GPU configuration
    gpu_list = load_gpu_config(args.gpu_config)
    logger.info(f"Available GPUs: {gpu_list}")

    # Generate seed list
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    logger.info(f"Running experiments for seeds: {seeds}")

    if args.dry_run:
        logger.info("DRY RUN - Commands that would be executed:")
        for i, seed in enumerate(seeds):
            gpu = gpu_list[i % len(gpu_list)]
            cmd = build_epoch_wise_cmd(args, seed, gpu)
            logger.info(f"Seed {seed} (GPU {gpu}): {' '.join(cmd)}")
        return

    # Group tasks by GPU to ensure GPU exclusivity
    gpu_tasks = {gpu: [] for gpu in gpu_list}
    for i, seed in enumerate(seeds):
        gpu = gpu_list[i % len(gpu_list)]
        gpu_tasks[gpu].append((args, seed, gpu, logger))

    logger.info(
        f"Task distribution: {[(gpu, len(tasks)) for gpu, tasks in gpu_tasks.items()]}"
    )

    # Run experiments with GPU exclusivity - sequential execution per GPU
    results = []
    start_time = time.time()

    # Create threads for each GPU to run tasks sequentially
    def run_gpu_tasks(gpu, tasks):
        """Run all tasks for a specific GPU sequentially."""
        gpu_results = []
        logger.info(f"GPU {gpu}: Starting {len(tasks)} tasks")

        for task in tasks:
            try:
                result = run_single_experiment(*task)
                gpu_results.append(result)
                logger.info(f"GPU {gpu}: Completed seed {task[1]}")
            except Exception as e:
                logger.error(f"GPU {gpu}: Task {task} generated an exception: {e}")
                gpu_results.append(
                    {
                        "seed": task[1],
                        "gpu": task[2],
                        "status": "exception",
                        "error": str(e),
                    }
                )

        logger.info(f"GPU {gpu}: Completed all {len(gpu_results)} tasks")
        return gpu_results

    # Run tasks for each GPU in parallel, but tasks within each GPU run sequentially
    threads = []
    for gpu, tasks in gpu_tasks.items():
        if tasks:  # Only create thread if there are tasks for this GPU
            thread = threading.Thread(
                target=lambda g=gpu, t=tasks: results.extend(run_gpu_tasks(g, t))
            )
            threads.append(thread)
            thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Sort results by seed for consistent output
    results.sort(key=lambda x: x["seed"])

    # Summary
    total_duration = time.time() - start_time
    successful = sum(r["status"] == "success" for r in results)
    failed = len(results) - successful

    logger.info("Experiment Summary:")
    logger.info(f"  Total experiments: {len(results)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total duration: {total_duration:.2f}s")
    logger.info(
        f"  Average duration per experiment: {total_duration / len(results):.2f}s"
    )

    # Log failed experiments
    if failed > 0:
        logger.warning("Failed experiments:")
        for result in results:
            if result["status"] != "success":
                logger.warning(
                    f"  Seed {result['seed']} (GPU {result['gpu']}): {result['status']}"
                )


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Validate arguments
    if args.seeds <= 0:
        raise ValueError("Number of seeds must be positive")

    if args.start_seed < 0:
        raise ValueError("Start seed must be non-negative")

    run_multi_seed_experiments(args)


if __name__ == "__main__":
    main()
