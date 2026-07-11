"""
BERT model wrapper for sentiment classification
Inspired by TextAttack's model handling (re-implemented)
"""

import os
import logging
from typing import Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
)


logger = logging.getLogger(__name__)


class BertForSentiment:
    """
    Wrapper for BERT-based sentiment classification model.
    """

    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        num_labels: int = 2,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize BERT model for sentiment classification.

        Args:
            model_name_or_path: Model name or path
            num_labels: Number of labels for classification
            cache_dir: Cache directory for model files
            device: Device to use (cuda/cpu)
        """
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.cache_dir = cache_dir

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model and tokenizer
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        logger.info(f"Loading model: {self.model_name_or_path}")

        # Load configuration
        try:
            self.config = AutoConfig.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels,
                cache_dir=self.cache_dir,
            )
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Create default config
            from transformers import BertConfig

            self.config = BertConfig(
                num_labels=self.num_labels,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
            )

        # Load model
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path, config=self.config, cache_dir=self.cache_dir
            )
        except Exception as e:
            logger.warning(f"Failed to load pretrained model: {e}")
            logger.info("Initializing model with random weights")
            self.model = AutoModelForSequenceClassification.from_config(self.config)

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, cache_dir=self.cache_dir
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

        # Move model to device
        self.model.to(self.device)

        logger.info(f"Model loaded successfully on {self.device}")

    def get_model(self) -> PreTrainedModel:
        """Get the underlying model."""
        return self.model

    def get_tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer."""
        return self.tokenizer

    def save_pretrained(self, save_directory: str):
        """
        Save model and tokenizer to directory.

        Args:
            save_directory: Directory to save model files
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save model
        self.model.save_pretrained(save_directory)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

        # Save label mappings
        import json

        label_info = {
            "num_labels": self.num_labels,
            "id2label": {0: "negative", 1: "positive"},
            "label2id": {"negative": 0, "positive": 1},
        }
        with open(os.path.join(save_directory, "label_info.json"), "w") as f:
            json.dump(label_info, f, indent=2)

        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path: str, device: Optional[str] = None):
        """
        Load model from saved directory.

        Args:
            model_path: Path to saved model directory
            device: Device to use

        Returns:
            BertForSentiment instance
        """
        # Load label info if exists
        label_info_path = os.path.join(model_path, "label_info.json")
        if os.path.exists(label_info_path):
            import json

            with open(label_info_path, "r") as f:
                label_info = json.load(f)
            num_labels = label_info.get("num_labels", 2)
        else:
            num_labels = 2

        # Create instance
        instance = cls(
            model_name_or_path=model_path, num_labels=num_labels, device=device
        )

        return instance

    def predict(self, texts, batch_size: int = 32):
        """
        Predict sentiment for given texts.

        Args:
            texts: List of texts or single text
            batch_size: Batch size for prediction

        Returns:
            Predictions (labels and probabilities)
        """
        if isinstance(texts, str):
            texts = [texts]

        self.model.eval()
        predictions = []
        probabilities = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)

                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Get predictions
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                predictions.extend(preds.cpu().numpy().tolist())
                probabilities.extend(probs.cpu().numpy().tolist())

        return {
            "labels": predictions,
            "probabilities": probabilities,
            "label_names": ["negative" if p == 0 else "positive" for p in predictions],
        }


def create_model_and_tokenizer(
    model_name: str = "bert-base-uncased",
    num_labels: int = 2,
    cache_dir: Optional[str] = None,
) -> tuple:
    """
    Create model and tokenizer for training.

    Args:
        model_name: Model name or path
        num_labels: Number of labels
        cache_dir: Cache directory

    Returns:
        Tuple of (model, tokenizer)
    """
    wrapper = BertForSentiment(
        model_name_or_path=model_name, num_labels=num_labels, cache_dir=cache_dir
    )

    return wrapper.get_model(), wrapper.get_tokenizer()
