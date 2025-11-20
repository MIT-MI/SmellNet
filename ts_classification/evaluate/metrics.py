"""Metrics computation for classification tasks."""

from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    average: str = "weighted",
    include_per_class: bool = False,
    class_names: Optional[list] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        predictions: Predicted class indices, shape (N,)
        targets: Ground truth class indices, shape (N,)
        average: Averaging strategy for multi-class metrics ('weighted', 'macro', 'micro')
        include_per_class: If True, include per-class metrics
        class_names: Optional list of class names for per-class metrics

    Returns:
        Dictionary containing:
        - accuracy: Overall accuracy
        - f1_weighted, f1_macro, f1_micro: F1 scores with different averaging
        - precision_weighted, precision_macro, precision_micro: Precision scores
        - recall_weighted, recall_macro, recall_micro: Recall scores
        - per_class_f1: Per-class F1 scores (if include_per_class=True)
        - per_class_precision: Per-class precision (if include_per_class=True)
        - per_class_recall: Per-class recall (if include_per_class=True)
    """
    metrics = {}

    # Overall accuracy
    metrics["accuracy"] = float(accuracy_score(targets, predictions))

    # F1 scores with different averaging strategies
    metrics["f1_weighted"] = float(f1_score(targets, predictions, average="weighted", zero_division=0))
    metrics["f1_macro"] = float(f1_score(targets, predictions, average="macro", zero_division=0))
    metrics["f1_micro"] = float(f1_score(targets, predictions, average="micro", zero_division=0))

    # Precision scores
    metrics["precision_weighted"] = float(precision_score(targets, predictions, average="weighted", zero_division=0))
    metrics["precision_macro"] = float(precision_score(targets, predictions, average="macro", zero_division=0))
    metrics["precision_micro"] = float(precision_score(targets, predictions, average="micro", zero_division=0))

    # Recall scores
    metrics["recall_weighted"] = float(recall_score(targets, predictions, average="weighted", zero_division=0))
    metrics["recall_macro"] = float(recall_score(targets, predictions, average="macro", zero_division=0))
    metrics["recall_micro"] = float(recall_score(targets, predictions, average="micro", zero_division=0))

    # Per-class metrics if requested
    if include_per_class:
        # Get per-class F1, precision, recall (returns array with one value per class)
        per_class_f1 = f1_score(targets, predictions, average=None, zero_division=0)
        per_class_precision = precision_score(targets, predictions, average=None, zero_division=0)
        per_class_recall = recall_score(targets, predictions, average=None, zero_division=0)

        # If class names provided, create a dictionary mapping class names to metrics
        if class_names is not None:
            for idx, class_name in enumerate(class_names):
                if idx < len(per_class_f1):
                    metrics[f"f1_class_{class_name}"] = float(per_class_f1[idx])
                    metrics[f"precision_class_{class_name}"] = float(per_class_precision[idx])
                    metrics[f"recall_class_{class_name}"] = float(per_class_recall[idx])
        else:
            # Otherwise, use class indices
            for idx in range(len(per_class_f1)):
                metrics[f"f1_class_{idx}"] = float(per_class_f1[idx])
                metrics[f"precision_class_{idx}"] = float(per_class_precision[idx])
                metrics[f"recall_class_{idx}"] = float(per_class_recall[idx])

    return metrics


def print_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[list] = None,
) -> str:
    """
    Generate a detailed classification report.

    Args:
        predictions: Predicted class indices, shape (N,)
        targets: Ground truth class indices, shape (N,)
        class_names: Optional list of class names

    Returns:
        String containing the classification report
    """
    return classification_report(
        targets,
        predictions,
        target_names=class_names,
        zero_division=0
    )
