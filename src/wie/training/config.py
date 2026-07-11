"""
Training configuration dataclass
Inspired by TextAttack's training patterns (re-implemented)
"""

from dataclasses import dataclass, asdict
from typing import Optional
import os
import yaml
import json


@dataclass
class TrainingConfig:
    """
    Configuration for training BERT models.
    Follows TextAttack-style training argument patterns.
    """

    # Model configuration
    model_name_or_path: str = "bert-base-uncased"
    num_labels: int = 2

    # Training hyperparameters
    num_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    max_length: int = 128

    # Learning rate scheduler
    num_warmup_steps: int = 500
    lr_scheduler_type: str = "linear"

    # Training strategy
    gradient_accumulation_steps: int = 1
    eval_strategy: str = "epoch"  # "epoch" or "steps"
    eval_steps: Optional[int] = None
    save_strategy: str = "best"  # "best", "epoch", "steps"
    save_steps: Optional[int] = None

    # Logging
    logging_steps: int = 100
    logging_first_step: bool = True

    # Checkpointing
    save_total_limit: Optional[int] = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"
    greater_is_better: bool = True

    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: Optional[float] = None

    # Output directories
    output_dir: str = "outputs/imdb_bert"
    log_dir: Optional[str] = None

    # Mixed precision training
    bf16: bool = False
    fp16: bool = False

    # Reproducibility
    seed: int = 42

    # Data loading
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True

    # Optional features
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: Optional[str] = None

    # Evaluation
    eval_accumulation_steps: Optional[int] = None

    # Device
    no_cuda: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Set log directory if not specified
        if self.log_dir is None:
            self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Validate eval strategy
        if self.eval_strategy not in ["epoch", "steps", "no"]:
            raise ValueError(
                f"eval_strategy must be 'epoch', 'steps', or 'no', got {self.eval_strategy}"
            )

        if self.eval_strategy == "steps" and self.eval_steps is None:
            raise ValueError("eval_steps must be specified when eval_strategy='steps'")

        # Validate save strategy
        if self.save_strategy not in ["best", "epoch", "steps", "no"]:
            raise ValueError(
                f"save_strategy must be 'best', 'epoch', 'steps', or 'no', got {self.save_strategy}"
            )

        if self.save_strategy == "steps" and self.save_steps is None:
            raise ValueError("save_steps must be specified when save_strategy='steps'")

        # Validate mixed precision
        if self.bf16 and self.fp16:
            raise ValueError("Cannot use both bf16 and fp16 training")

    @classmethod
    def from_yaml(cls, yaml_path: str, **kwargs) -> "TrainingConfig":
        """
        Load configuration from YAML file with optional overrides.

        Args:
            yaml_path: Path to YAML configuration file
            **kwargs: Override parameters

        Returns:
            TrainingConfig instance
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Apply overrides
        config_dict.update(kwargs)

        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str, **kwargs) -> "TrainingConfig":
        """
        Load configuration from JSON file with optional overrides.

        Args:
            json_path: Path to JSON configuration file
            **kwargs: Override parameters

        Returns:
            TrainingConfig instance
        """
        with open(json_path, "r") as f:
            config_dict = json.load(f)

        # Apply overrides
        config_dict.update(kwargs)

        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save_to_json(self, json_path: str):
        """Save configuration to JSON file."""
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size including gradient accumulation."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def __str__(self) -> str:
        """Pretty print configuration."""
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2)
