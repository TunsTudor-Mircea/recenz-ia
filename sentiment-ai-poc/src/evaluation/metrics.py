"""
Metrics calculation module for model evaluation.

This module provides comprehensive metrics calculation including
accuracy, precision, recall, F1-score, confusion matrix, and
classification reports.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True labels (1D array of class indices).
        y_pred: Predicted labels (1D array of class indices).
        y_pred_proba: Predicted probabilities (2D array of shape [n_samples, n_classes]).
        class_names: Names of classes. If None, uses class indices.

    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - precision_macro: Macro-averaged precision
            - precision_weighted: Weighted-averaged precision
            - recall_macro: Macro-averaged recall
            - recall_weighted: Weighted-averaged recall
            - f1_macro: Macro-averaged F1 score
            - f1_weighted: Weighted-averaged F1 score
            - per_class_precision: Precision per class
            - per_class_recall: Recall per class
            - per_class_f1: F1 score per class
            - confusion_matrix: Confusion matrix
            - classification_report: Text classification report
            - classification_report_dict: Classification report as dictionary
            - roc_auc_ovr: ROC AUC one-vs-rest (if y_pred_proba provided)
            - roc_auc_ovo: ROC AUC one-vs-one (if y_pred_proba provided)
            - class_names: List of class names used
    """
    logger.info("Calculating evaluation metrics")

    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have same length. Got {len(y_true)} and {len(y_pred)}")

    if y_pred_proba is not None and len(y_true) != len(y_pred_proba):
        raise ValueError(f"y_true and y_pred_proba must have same length. Got {len(y_true)} and {len(y_pred_proba)}")

    # Determine number of classes
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))

    # Set class names if not provided
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]
    elif len(class_names) != n_classes:
        logger.warning(f"Number of class names ({len(class_names)}) doesn't match number of classes ({n_classes}). Using indices.")
        class_names = [f"class_{i}" for i in range(n_classes)]

    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate precision, recall, and F1 scores
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)

    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Generate classification report
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Build metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'precision_weighted': float(precision_weighted),
        'recall_macro': float(recall_macro),
        'recall_weighted': float(recall_weighted),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'per_class_precision': {class_names[i]: float(per_class_precision[i]) for i in range(len(class_names))},
        'per_class_recall': {class_names[i]: float(per_class_recall[i]) for i in range(len(class_names))},
        'per_class_f1': {class_names[i]: float(per_class_f1[i]) for i in range(len(class_names))},
        'confusion_matrix': cm.tolist(),
        'classification_report': report_text,
        'classification_report_dict': report_dict,
        'class_names': class_names,
    }

    # Calculate ROC AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            # One-vs-Rest ROC AUC
            roc_auc_ovr = roc_auc_score(
                y_true,
                y_pred_proba,
                multi_class='ovr',
                average='macro'
            )
            metrics['roc_auc_ovr'] = float(roc_auc_ovr)

            # One-vs-One ROC AUC
            roc_auc_ovo = roc_auc_score(
                y_true,
                y_pred_proba,
                multi_class='ovo',
                average='macro'
            )
            metrics['roc_auc_ovo'] = float(roc_auc_ovo)

            # Per-class ROC AUC (OvR)
            per_class_roc_auc = {}
            for i, class_name in enumerate(class_names):
                try:
                    if i < y_pred_proba.shape[1]:
                        y_true_binary = (y_true == i).astype(int)
                        if len(np.unique(y_true_binary)) > 1:  # Only if both classes present
                            auc = roc_auc_score(y_true_binary, y_pred_proba[:, i])
                            per_class_roc_auc[class_name] = float(auc)
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC for class {class_name}: {e}")

            if per_class_roc_auc:
                metrics['per_class_roc_auc'] = per_class_roc_auc

        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC scores: {e}")

    logger.info(f"Metrics calculated. Accuracy: {accuracy:.4f}, F1 (macro): {f1_macro:.4f}")

    return metrics
