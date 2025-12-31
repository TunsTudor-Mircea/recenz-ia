"""
Evaluation module for model performance assessment.

This module provides metrics calculation, visualization, and reporting
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
from evaluation.reporter import generate_report

__all__ = [
    'calculate_metrics',
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_precision_recall_curves',
    'plot_feature_importance',
    'plot_convergence_curve',
    'generate_report',
]
