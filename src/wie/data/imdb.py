"""
IMDB dataset loader for BERT training
Inspired by TextAttack's dataset handling (re-implemented)
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


@dataclass
class IMDBExample:
    """Single IMDB example."""

    text: str
    label: int
    idx: Optional[int] = None


class IMDBDataset(Dataset):
    """
    IMDB dataset for PyTorch training.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        split: str = "train",
        max_length: int = 128,
        cache_dir: Optional[str] = None,
        num_samples: Optional[int] = None,
        validation_split: float = 0.1,
        train_fraction: float = 1.0,
        test_fraction: float = 1.0,
        seed: int = 42,
    ):
        """
        Initialize IMDB dataset.

        Args:
            tokenizer: Tokenizer to use for encoding
            split: Dataset split ("train", "test", or "validation")
            max_length: Maximum sequence length
            cache_dir: Directory to cache the dataset
            num_samples: Limit number of samples (for debugging)
            validation_split: Fraction of training data to use for validation
            train_fraction: Fraction of original training data to use
            test_fraction: Fraction of original test data to use
            seed: Random seed for validation split
        """
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.num_samples = num_samples
        self.validation_split = validation_split
        self.train_fraction = train_fraction
        self.test_fraction = test_fraction
        self.seed = seed

        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load and prepare the IMDB dataset."""
        logger.info(f"Loading IMDB dataset (split: {self.split})")

        # Load from Hugging Face datasets
        try:
            dataset = load_dataset("imdb", cache_dir=self.cache_dir)
        except Exception as e:
            logger.error(f"Failed to load IMDB dataset: {e}")
            raise

        # Handle train/validation split
        if self.split == "validation":
            # Create validation set from training data
            if "train" in dataset:
                train_dataset = dataset["train"]

                # Apply train_fraction first if needed
                if self.train_fraction < 1.0:
                    train_size = int(len(train_dataset) * self.train_fraction)
                    train_dataset = train_dataset.select(range(train_size))
                    msg = "Using %d examples from original training data (%s)" % (
                        train_size,
                        f"{self.train_fraction:.1%}",
                    )
                    logger.info(msg)

                # Split training data for validation
                split_dataset = train_dataset.train_test_split(
                    test_size=self.validation_split, seed=self.seed
                )
                self.examples = split_dataset["test"]
                logger.info(
                    "Created validation set with %d examples", len(self.examples)
                )
            else:
                raise ValueError(
                    "No training data available to create validation split"
                )
        elif self.split == "train":
            if "train" in dataset:
                train_dataset = dataset["train"]

                # Apply train_fraction first if needed
                if self.train_fraction < 1.0:
                    train_size = int(len(train_dataset) * self.train_fraction)
                    train_dataset = train_dataset.select(range(train_size))
                    msg = "Using %d examples from original training data (%s)" % (
                        train_size,
                        f"{self.train_fraction:.1%}",
                    )
                    logger.info(msg)

                if self.validation_split > 0:
                    # Use remaining training data after validation split
                    split_dataset = train_dataset.train_test_split(
                        test_size=self.validation_split, seed=self.seed
                    )
                    self.examples = split_dataset["train"]
                    logger.info(
                        "Using %d training examples after validation split",
                        len(self.examples),
                    )
                else:
                    self.examples = train_dataset
            else:
                raise ValueError("No training data available")
        elif self.split == "test":
            if "test" in dataset:
                test_dataset = dataset["test"]

                # Apply test_fraction if needed
                if self.test_fraction < 1.0:
                    test_size = int(len(test_dataset) * self.test_fraction)
                    test_dataset = test_dataset.select(range(test_size))
                    msg = "Using %d examples from original test data (%s)" % (
                        test_size,
                        f"{self.test_fraction:.1%}",
                    )
                    logger.info(msg)

                self.examples = test_dataset
            else:
                raise ValueError("No test data available")
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Limit samples if specified
        if self.num_samples is not None and self.num_samples < len(self.examples):
            self.examples = self.examples.select(range(self.num_samples))
            logger.info("Limited to %d samples", self.num_samples)

        logger.info("Loaded %d examples for %s split", len(self.examples), self.split)

        # Pre-tokenize all examples for efficiency
        self._tokenize_examples()

    def _tokenize_examples(self):
        """Tokenize all examples."""
        logger.info("Tokenizing examples...")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,  # Return lists for dataset mapping
            )

        # Tokenize in batches
        self.examples = self.examples.map(
            tokenize_function, batched=True, desc=f"Tokenizing {self.split} dataset"
        )

        # Set format for PyTorch
        self.examples.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.

        Args:
            idx: Index of the example

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        item = self.examples[idx]

        # Rename 'label' to 'labels' for Hugging Face models
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["label"],
        }


class IMDBDataModule:
    """
    Data module for IMDB dataset.
    Handles train/validation/test splits and data loading.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        train_batch_size: int = 16,
        eval_batch_size: int = 32,
        validation_split: float = 0.1,
        train_fraction: float = 1.0,
        test_fraction: float = 1.0,
        cache_dir: Optional[str] = None,
        num_workers: int = 0,
        seed: int = 42,
    ):
        """
        Initialize data module.

        Args:
            model_name: Name of the model for tokenizer
            max_length: Maximum sequence length
            train_batch_size: Training batch size
            eval_batch_size: Evaluation batch size
            validation_split: Fraction of training data for validation
            train_fraction: Fraction of original training data to use
            test_fraction: Fraction of original test data to use
            cache_dir: Cache directory for datasets
            num_workers: Number of workers for data loading
            seed: Random seed
        """
        self.model_name = model_name
        self.max_length = max_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.validation_split = validation_split
        self.train_fraction = train_fraction
        self.test_fraction = test_fraction
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "imdb"
        )
        self.num_workers = num_workers
        self.seed = seed

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for different stages.

        Args:
            stage: Stage of training ("fit", "validate", "test")
        """
        if stage == "fit" or stage is None:
            # Setup training and validation datasets
            self.train_dataset = IMDBDataset(
                tokenizer=self.tokenizer,
                split="train",
                max_length=self.max_length,
                cache_dir=self.cache_dir,
                validation_split=self.validation_split,
                train_fraction=self.train_fraction,
                test_fraction=self.test_fraction,
                seed=self.seed,
            )

            self.val_dataset = IMDBDataset(
                tokenizer=self.tokenizer,
                split="validation",
                max_length=self.max_length,
                cache_dir=self.cache_dir,
                validation_split=self.validation_split,
                train_fraction=self.train_fraction,
                test_fraction=self.test_fraction,
                seed=self.seed,
            )

        if stage == "test" or stage is None:
            # Setup test dataset
            self.test_dataset = IMDBDataset(
                tokenizer=self.tokenizer,
                split="test",
                max_length=self.max_length,
                cache_dir=self.cache_dir,
                validation_split=0,  # No validation split for test
                train_fraction=self.train_fraction,
                test_fraction=self.test_fraction,
                seed=self.seed,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            self.setup("fit")

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get validation dataloader."""
        if self.val_dataset is None:
            self.setup("fit")

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get test dataloader."""
        if self.test_dataset is None:
            self.setup("test")

        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_num_labels(self) -> int:
        """Get number of labels in the dataset."""
        return 2  # Binary classification for IMDB

    def get_label_names(self) -> List[str]:
        """Get label names."""
        return ["negative", "positive"]

    def get_id2label(self) -> Dict[int, str]:
        """Get id to label mapping."""
        return {0: "negative", 1: "positive"}

    def get_label2id(self) -> Dict[str, int]:
        """Get label to id mapping."""
        return {"negative": 0, "positive": 1}
