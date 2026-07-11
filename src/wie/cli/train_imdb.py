#!/usr/bin/env python
"""
CLI for training BERT on IMDB dataset
Inspired by TextAttack's CLI ergonomics (re-implemented)
"""

import argparse
import logging
import os
import random

import numpy as np
import torch
from transformers import set_seed

from wie.data.imdb import IMDBDataModule
from wie.models.bert_sentiment import create_model_and_tokenizer
from wie.training.config import TrainingConfig
from wie.training.trainer import Trainer


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train BERT model on IMDB dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="bert-base-uncased",
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--num-labels", type=int, default=2, help="Number of labels for classification"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=16,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=32,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--max-length", type=int, default=128, help="Maximum sequence length"
    )

    # Learning rate scheduler
    parser.add_argument(
        "--num-warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="linear",
        choices=["linear", "cosine", "constant"],
        help="Type of learning rate scheduler",
    )

    # Training strategy
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--eval-strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps", "no"],
        help="Evaluation strategy",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Evaluation steps (when eval_strategy='steps')",
    )
    parser.add_argument(
        "--save-strategy",
        type=str,
        default="best",
        choices=["best", "epoch", "steps", "no"],
        help="Model saving strategy",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Save steps (when save_strategy='steps')",
    )

    # Logging
    parser.add_argument("--logging-steps", type=int, default=100, help="Logging steps")
    parser.add_argument(
        "--logging-first-step", action="store_true", help="Log first step"
    )

    # Checkpointing
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--load-best-model-at-end",
        action="store_true",
        default=True,
        help="Load best model at the end of training",
    )
    parser.add_argument(
        "--metric-for-best-model",
        type=str,
        default="eval_accuracy",
        help="Metric to use for selecting best model",
    )
    parser.add_argument(
        "--greater-is-better",
        action="store_true",
        default=True,
        help="Whether higher metric value is better",
    )

    # Early stopping
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=None,
        help="Early stopping threshold",
    )

    # Output directories
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/imdb_bert",
        help="Output directory for model and logs",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for logs (default: output_dir/logs)",
    )

    # Mixed precision
    parser.add_argument(
        "--fp16", action="store_true", help="Use FP16 mixed precision training"
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use BF16 mixed precision training"
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Data loading
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--dataloader-pin-memory",
        action="store_true",
        default=True,
        help="Pin memory for data loading",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of original training data to use (default: 1.0 for full dataset)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=1.0,
        help="Fraction of original test data to use (default: 1.0 for full dataset)",
    )

    # Optional features
    parser.add_argument(
        "--use-tensorboard", action="store_true", help="Use TensorBoard for logging"
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="Weights & Biases project name"
    )

    # Device
    parser.add_argument(
        "--no-cuda", action="store_true", help="Disable CUDA even if available"
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (YAML or JSON)",
    )

    # Cache directory
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for models and datasets",
    )

    return parser.parse_args()


def setup_logging(output_dir: str):
    """Setup logging configuration."""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(),
        ],
    )


def set_reproducibility(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Create training configuration
    if args.config:
        # Load from config file
        if args.config.endswith(".yaml"):
            config = TrainingConfig.from_yaml(args.config)
        elif args.config.endswith(".json"):
            config = TrainingConfig.from_json(args.config)
        else:
            raise ValueError(f"Unknown config file format: {args.config}")

        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None and hasattr(config, key.replace("-", "_")):
                setattr(config, key.replace("-", "_"), value)
    else:
        # Create config from arguments
        config_dict = {}
        for key, value in vars(args).items():
            if value is not None:
                # Convert argument names to config field names
                config_key = key.replace("-", "_")
                if hasattr(TrainingConfig, config_key):
                    config_dict[config_key] = value

        # Handle special mappings
        if "epochs" in vars(args) and args.epochs is not None:
            config_dict["num_epochs"] = args.epochs

        config = TrainingConfig(**config_dict)

    # Setup logging
    setup_logging(config.output_dir)

    # Log configuration
    logger.info("Training configuration:")
    logger.info(config)

    # Set reproducibility
    set_reproducibility(config.seed)

    # Setup data module
    logger.info("Setting up data module...")
    data_module = IMDBDataModule(
        model_name=config.model_name_or_path,
        max_length=config.max_length,
        train_batch_size=config.per_device_train_batch_size,
        eval_batch_size=config.per_device_eval_batch_size,
        validation_split=args.validation_split,
        train_fraction=args.train_fraction,
        test_fraction=args.test_fraction,
        cache_dir=args.cache_dir,
        num_workers=config.dataloader_num_workers,
        seed=config.seed,
    )
    data_module.setup("fit")

    # Create model and tokenizer
    logger.info(f"Loading model: {config.model_name_or_path}")
    model, tokenizer = create_model_and_tokenizer(
        model_name=config.model_name_or_path,
        num_labels=config.num_labels,
        cache_dir=args.cache_dir,
    )

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data_module.train_dataset,
        eval_dataset=data_module.val_dataset,
        config=config,
    )

    # Train model
    logger.info("Starting training...")
    results = trainer.train()

    # Log final results
    logger.info("Training completed!")
    logger.info(f"Best eval accuracy: {results.get('best_metric', 'N/A')}")
    logger.info(f"Total training steps: {results.get('global_step', 0)}")

    # Test on test set if available
    if hasattr(data_module, "test_dataset"):
        logger.info("Evaluating on test set...")
        data_module.setup("test")  # Setup test dataset
        trainer.eval_dataset = data_module.test_dataset
        trainer.eval_dataloader = data_module.test_dataloader()
        test_metrics = trainer.evaluate()
        logger.info(f"Test accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}")
        logger.info(f"Test F1: {test_metrics.get('f1', 'N/A'):.4f}")

        # Save test metrics
        import json

        test_metrics_path = os.path.join(config.output_dir, "test_metrics.json")
        with open(test_metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Test metrics saved to {test_metrics_path}")

    logger.info(f"All outputs saved to {config.output_dir}")


if __name__ == "__main__":
    main()
