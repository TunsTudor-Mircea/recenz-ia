"""
Evaluation module for model performance assessment.

This module provides metrics calculation and visualization
for sentiment analysis models.
"""

from evaluation.metrics import calculate_metrics
from evaluation.visualizer import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_feature_importance,
    plot_convergence_curve,
)

__all__ = [
    'calculate_metrics',
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_precision_recall_curves',
    'plot_feature_importance',
    'plot_convergence_curve',
]
