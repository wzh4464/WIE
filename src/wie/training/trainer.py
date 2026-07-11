"""
Trainer class for BERT models
Inspired by TextAttack's trainer patterns (re-implemented)
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Sized

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from .config import TrainingConfig


logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for BERT-based models following TextAttack-style patterns.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[TrainingConfig] = None,
        collate_fn: Optional[Any] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            config: Training configuration
            collate_fn: Custom collate function for DataLoader
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or TrainingConfig()
        self.collate_fn = collate_fn

        # Setup device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.config.no_cuda else "cpu"
        )
        self.model.to(self.device)

        # Initialize tracking variables
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.best_model_path = None

        # Setup directories
        self._setup_directories()

        # Setup logging
        self._setup_logging()

        # Setup dataloaders
        self._setup_dataloaders()

        # Setup optimizer and scheduler
        self._setup_optimizer_scheduler()

        # Setup mixed precision
        self._setup_mixed_precision()

        # Initialize metrics tracking
        self.train_history = []
        self.eval_history = []

    def _setup_directories(self):
        """Create necessary directories for outputs."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        if self.config.log_dir:
            os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "checkpoints"), exist_ok=True)

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=[
                (
                    logging.FileHandler(
                        os.path.join(self.config.log_dir, "training.log")
                    )
                    if self.config.log_dir
                    else logging.NullHandler()
                ),
                logging.StreamHandler(),
            ],
        )

        # Optional: Setup TensorBoard
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard.writer import SummaryWriter

                self.tb_writer = SummaryWriter(log_dir=self.config.log_dir)
            except ImportError:
                logger.warning(
                    "TensorBoard not available. Install tensorboard to enable logging."
                )
                self.tb_writer = None
        else:
            self.tb_writer = None

        # Optional: Setup Weights & Biases
        if self.config.use_wandb:
            try:
                import wandb

                run_name = (
                    f"{self.config.model_name_or_path}_"
                    f"{self.config.output_dir.split('/')[-1]}"
                )
                wandb.init(
                    project=self.config.wandb_project or "bert-training",
                    config=self.config.to_dict(),
                    name=run_name,
                )
                self.wandb = wandb
            except ImportError:
                logger.warning(
                    "Weights & Biases not available. Install wandb to enable logging."
                )
                self.wandb = None
        else:
            self.wandb = None

    def _setup_dataloaders(self):
        """Create DataLoaders for training and evaluation."""
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            collate_fn=self.collate_fn,
        )

        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.per_device_eval_batch_size,
                shuffle=False,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=self.config.dataloader_pin_memory,
                collate_fn=self.collate_fn,
            )
        else:
            self.eval_dataloader = None

    def _setup_optimizer_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # Prepare optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.config.learning_rate, eps=1e-8
        )

        # Calculate total training steps
        num_training_steps = len(self.train_dataloader) * self.config.num_epochs
        num_training_steps = (
            num_training_steps // self.config.gradient_accumulation_steps
        )

        # Setup scheduler based on type
        if self.config.lr_scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.config.lr_scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.config.lr_scheduler_type == "constant":
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.config.num_warmup_steps
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.lr_scheduler_type}")

        self.num_training_steps = num_training_steps
        logger.info(f"Total optimization steps: {num_training_steps}")

    def _setup_mixed_precision(self):
        """Setup mixed precision training if enabled."""
        self.scaler = None
        if self.config.fp16:
            try:
                from torch.cuda.amp import GradScaler

                self.scaler = GradScaler()
                logger.info("Using FP16 mixed precision training")
            except ImportError:
                logger.warning("FP16 training requested but CUDA AMP not available")
        elif self.config.bf16:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                logger.info("Using BF16 mixed precision training")
            else:
                logger.warning(
                    "BF16 training requested but not supported on this device"
                )

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Dictionary containing training metrics and history
        """
        logger.info("***** Running training *****")
        num_examples = (
            len(self.train_dataset)
            if isinstance(self.train_dataset, Sized)
            else "Unknown"
        )
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {self.config.num_epochs}")
        logger.info(f"  Batch size = {self.config.per_device_train_batch_size}")
        logger.info(
            f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {self.num_training_steps}")

        self.model.train()
        self.model.zero_grad()

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()

            # Train for one epoch
            train_metrics = self._train_epoch()

            # Evaluate if needed
            eval_metrics = {}
            if self._should_evaluate():
                eval_metrics = self.evaluate()

                # Check for best model
                if (
                    self.config.save_strategy == "best"
                    and self.config.metric_for_best_model in eval_metrics
                ):
                    current_metric = eval_metrics[self.config.metric_for_best_model]
                    if self._is_better_metric(current_metric):
                        self.best_metric = current_metric
                        self._save_checkpoint(is_best=True)

            # Save checkpoint if needed
            if self._should_save_checkpoint():
                self._save_checkpoint(is_best=False)

            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} completed in {epoch_time:.2f}s"
            )
            logger.info(f"  Train loss: {train_metrics.get('loss', 0):.4f}")
            if eval_metrics:
                logger.info(f"  Eval accuracy: {eval_metrics.get('accuracy', 0):.4f}")

            # Check early stopping
            if self._should_stop_early(eval_metrics):
                logger.info("Early stopping triggered")
                break

        # Load best model if configured
        if self.config.load_best_model_at_end and self.best_model_path:
            self._load_checkpoint(self.best_model_path)
            logger.info(f"Loaded best model from {self.best_model_path}")

        # Save final model
        self._save_final_model()

        # Save training history
        self._save_training_history()

        # Cleanup
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb:
            self.wandb.finish()

        return {
            "train_history": self.train_history,
            "eval_history": self.eval_history,
            "best_metric": self.best_metric,
            "global_step": self.global_step,
        }

    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary containing epoch metrics
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch + 1}")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass with mixed precision if enabled
            if self.config.fp16 and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            elif self.config.bf16:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
            else:
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()

            total_loss += loss.item() * self.config.gradient_accumulation_steps

            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.fp16 and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step
                if self.config.fp16 and self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.model.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / (num_batches + 1)
                    current_lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix(
                        {"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"}
                    )

                    if self.tb_writer:
                        self.tb_writer.add_scalar(
                            "train/loss", avg_loss, self.global_step
                        )
                        self.tb_writer.add_scalar(
                            "train/learning_rate", current_lr, self.global_step
                        )

                    if self.wandb:
                        self.wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/learning_rate": current_lr,
                                "train/global_step": self.global_step,
                            }
                        )

            num_batches += 1

        avg_loss = total_loss / num_batches
        self.train_history.append(
            {"epoch": self.epoch + 1, "loss": avg_loss, "global_step": self.global_step}
        )

        return {"loss": avg_loss}

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.eval_dataloader is None:
            logger.warning("No evaluation dataset provided")
            return {}

        logger.info("***** Running evaluation *****")
        num_eval = (
            len(self.eval_dataset)
            if (self.eval_dataset is not None and isinstance(self.eval_dataset, Sized))
            else "Unknown"
        )
        logger.info(f"  Num examples = {num_eval}")

        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                if self.config.bf16:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)

                total_loss += outputs.loss.item()

                # Get predictions
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(self.eval_dataloader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

        # Calculate additional metrics
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="binary"
        )

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "eval_accuracy": accuracy,  # For compatibility
        }

        # Log metrics
        logger.info(f"  Eval loss: {avg_loss:.4f}")
        logger.info(f"  Eval accuracy: {accuracy:.4f}")
        logger.info(f"  Eval F1: {f1:.4f}")

        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"eval/{key}", value, self.global_step)

        if self.wandb:
            wandb_metrics = {f"eval/{k}": v for k, v in metrics.items()}
            wandb_metrics["eval/epoch"] = self.epoch + 1
            self.wandb.log(wandb_metrics)

        self.eval_history.append(
            {"epoch": self.epoch + 1, "global_step": self.global_step, **metrics}
        )

        self.model.train()
        return metrics

    def _should_evaluate(self) -> bool:
        """Check if evaluation should be performed."""
        if self.eval_dataloader is None:
            return False

        if self.config.eval_strategy == "no":
            return False
        elif self.config.eval_strategy == "epoch":
            return True
        elif self.config.eval_strategy == "steps":
            return (
                self.config.eval_steps is not None
                and self.config.eval_steps > 0
                and self.global_step % self.config.eval_steps == 0
            )

        return False

    def _should_save_checkpoint(self) -> bool:
        """Check if checkpoint should be saved."""
        if self.config.save_strategy == "no":
            return False
        elif self.config.save_strategy == "epoch":
            return True
        elif self.config.save_strategy == "steps":
            return (
                self.config.save_steps is not None
                and self.config.save_steps > 0
                and self.global_step % self.config.save_steps == 0
            )

        return False

    def _is_better_metric(self, current_metric: float) -> bool:
        """Check if current metric is better than best metric."""
        if self.best_metric is None:
            return True

        if self.config.greater_is_better:
            return current_metric > self.best_metric
        else:
            return current_metric < self.best_metric

    def _should_stop_early(self, eval_metrics: Dict[str, float]) -> bool:
        """Check if training should stop early."""
        if self.config.early_stopping_patience is None:
            return False

        # Implement early stopping logic here if needed
        # This is a simplified version
        return False

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        if is_best:
            checkpoint_dir = os.path.join(self.config.output_dir, "best_model")
        else:
            checkpoint_dir = os.path.join(
                self.config.output_dir, "checkpoints", f"checkpoint-{self.global_step}"
            )

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "config": self.config.to_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(state, os.path.join(checkpoint_dir, "trainer_state.pt"))

        # Save config
        self.config.save_to_json(os.path.join(checkpoint_dir, "training_config.json"))

        if is_best:
            self.best_model_path = checkpoint_dir
            logger.info(f"Saved best model checkpoint to {checkpoint_dir}")
        else:
            logger.info(f"Saved checkpoint to {checkpoint_dir}")

        # Manage checkpoint limit
        if self.config.save_total_limit and not is_best:
            self._rotate_checkpoints()

    def _rotate_checkpoints(self):
        """Keep only the most recent checkpoints."""
        checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            return

        checkpoints = sorted(
            [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )

        if (
            self.config.save_total_limit is not None
            and len(checkpoints) > self.config.save_total_limit
        ):
            for checkpoint in checkpoints[: -self.config.save_total_limit]:
                import shutil

                shutil.rmtree(os.path.join(checkpoint_dir, checkpoint))
                logger.info(f"Deleted old checkpoint: {checkpoint}")

    def _load_checkpoint(self, checkpoint_dir: str):
        """Load model from checkpoint."""
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        self.model.to(self.device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        # Load training state if exists
        state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_metric = state["best_metric"]
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_dir}")

    def _save_final_model(self):
        """Save the final model."""
        final_dir = os.path.join(self.config.output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)

        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        self.config.save_to_json(os.path.join(final_dir, "training_config.json"))

        logger.info(f"Saved final model to {final_dir}")

    def _save_training_history(self):
        """Save training history to JSON."""
        history = {
            "train_history": self.train_history,
            "eval_history": self.eval_history,
            "best_metric": self.best_metric,
            "config": self.config.to_dict(),
        }

        history_path = os.path.join(self.config.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Also save metrics in a simpler format
        metrics_path = os.path.join(self.config.output_dir, "metrics.json")
        final_metrics = {
            "final_train_loss": (
                self.train_history[-1]["loss"] if self.train_history else None
            ),
            "best_eval_accuracy": self.best_metric,
            "total_steps": self.global_step,
            "num_epochs_trained": self.epoch + 1,
        }

        if self.eval_history:
            final_metrics.update(
                {
                    "final_eval_accuracy": self.eval_history[-1].get("accuracy"),
                    "final_eval_f1": self.eval_history[-1].get("f1"),
                    "final_eval_loss": self.eval_history[-1].get("loss"),
                }
            )

        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)

        logger.info(f"Saved training history to {history_path}")
        logger.info(f"Saved metrics to {metrics_path}")
