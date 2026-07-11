#!/usr/bin/env python
"""
CLI for evaluating trained BERT model on IMDB dataset
"""

import argparse
import json
import logging
import os
from typing import Dict, List

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from wie.data.imdb import IMDBDataModule
from wie.models.bert_sentiment import BertForSentiment


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained BERT model on IMDB dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max-length", type=int, default=128, help="Maximum sequence length"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for predictions (JSON format)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None, help="Cache directory for datasets"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (for debugging)",
    )

    return parser.parse_args()


def evaluate_model(model_wrapper, dataloader, device) -> Dict:
    """Evaluate model and return metrics and raw predictions."""
    model = model_wrapper.get_model()
    model.eval()
    model.to(device)

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[List[float]] = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_to_device(batch, device)
            outputs = model(**batch)
            if hasattr(outputs, "loss"):
                total_loss += float(outputs.loss.item())
                num_batches += 1
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return _compute_metrics(all_labels, all_preds, all_probs, total_loss, num_batches)


def _move_to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }


def _compute_metrics(
    labels: List[int],
    preds: List[int],
    probs: List[List[float]],
    total_loss: float,
    num_batches: int,
) -> Dict:
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    p_pc, r_pc, f_pc, s_pc = precision_recall_fscore_support(
        labels, preds, average=None
    )
    cm = confusion_matrix(labels, preds)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "loss": float(avg_loss),
        "num_samples": len(labels),
        "per_class_metrics": {
            "negative": {
                "precision": float(p_pc[0]),
                "recall": float(r_pc[0]),
                "f1": float(f_pc[0]),
                "support": int(s_pc[0]),
            },
            "positive": {
                "precision": float(p_pc[1]),
                "recall": float(r_pc[1]),
                "f1": float(f_pc[1]),
                "support": int(s_pc[1]),
            },
        },
        "confusion_matrix": cm.tolist(),
        "predictions": preds,
        "labels": labels,
        "probabilities": probs,
    }


def main():
    """Main evaluation entrypoint."""
    args = parse_args()
    _setup_logging()
    _validate_model_path(args.model_path)

    logger.info(f"Loading model from {args.model_path}")
    model_wrapper = BertForSentiment.from_pretrained(
        args.model_path, device=args.device
    )

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    dataloader, dataset = _build_dataloader(args)

    if args.num_samples:
        logger.info(f"Evaluating on {min(args.num_samples, len(dataset))} samples")
    else:
        logger.info(f"Evaluating on {len(dataset)} samples")

    logger.info("Starting evaluation...")
    results = evaluate_model(model_wrapper, dataloader, device)
    _print_results(results, args.split)
    _maybe_save_results(args, results)


def _setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def _validate_model_path(path: str) -> None:
    if not os.path.exists(path):
        raise ValueError(f"Model path does not exist: {path}")


def _build_dataloader(args: argparse.Namespace):
    logger.info(f"Loading {args.split} dataset...")
    dm = IMDBDataModule(
        model_name=args.model_path,
        max_length=args.max_length,
        eval_batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )
    if args.split == "train":
        dm.setup("fit")
        dataloader = dm.train_dataloader()
        dataset = dm.train_dataset
    elif args.split == "validation":
        dm.setup("fit")
        dataloader = dm.val_dataloader()
        dataset = dm.val_dataset
    else:
        dm.setup("test")
        dataloader = dm.test_dataloader()
        dataset = dm.test_dataset

    if args.num_samples:
        from torch.utils.data import DataLoader, Subset

        indices = list(range(min(args.num_samples, len(dataset))))
        subset = Subset(dataset, indices)
        dataloader = DataLoader(
            subset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
    return dataloader, dataset


def _print_results(results: Dict, split: str) -> None:
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Dataset Split: {split}")
    print(f"Number of samples: {results['num_samples']}")
    print("-" * 50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"Loss:      {results['loss']:.4f}")
    print("-" * 50)
    print("Per-class metrics:")
    for class_name, metrics in results["per_class_metrics"].items():
        print(f"\n{class_name.upper()}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Support:   {metrics['support']}")
    print("-" * 50)
    print("Confusion Matrix:")
    print("               Predicted")
    print("           Negative  Positive")
    cm = results["confusion_matrix"]
    print(f"Actual Negative  {cm[0][0]:5d}    {cm[0][1]:5d}")
    print(f"      Positive  {cm[1][0]:5d}    {cm[1][1]:5d}")
    print("=" * 50)


def _maybe_save_results(args: argparse.Namespace, results: Dict) -> None:
    if not args.output_file:
        return
    output_results = {
        k: v
        for k, v in results.items()
        if k not in ["predictions", "labels", "probabilities"]
    }
    with open(args.output_file, "w") as f:
        json.dump(output_results, f, indent=2)
    logger.info(f"Results saved to {args.output_file}")

    predictions_file = args.output_file.replace(".json", "_predictions.json")
    predictions_data = {
        "predictions": results["predictions"],
        "labels": results["labels"],
        "probabilities": results["probabilities"],
    }
    with open(predictions_file, "w") as f:
        json.dump(predictions_data, f)
    logger.info(f"Predictions saved to {predictions_file}")


if __name__ == "__main__":
    main()
