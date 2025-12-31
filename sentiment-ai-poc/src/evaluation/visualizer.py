"""
Visualization module for model evaluation.

This module provides functions to create and save various plots
for model performance analysis including training curves, confusion matrices,
ROC curves, precision-recall curves, and feature importance.
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import logging

logger = logging.getLogger(__name__)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Union[str, Path],
    title: str = "Training History"
) -> None:
    """
    Plot training and validation loss and accuracy curves.

    Args:
        history: Dictionary with keys 'loss', 'accuracy', 'val_loss', 'val_accuracy'
                (or similar) containing training history.
        save_path: Path to save the plot.
        title: Title for the plot.
    """
    logger.info(f"Plotting training curves to {save_path}")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine which metrics are available
    has_loss = 'loss' in history
    has_accuracy = 'accuracy' in history or 'acc' in history
    has_val_loss = 'val_loss' in history
    has_val_accuracy = 'val_accuracy' in history or 'val_acc' in history

    # Normalize key names
    acc_key = 'accuracy' if 'accuracy' in history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'

    # Create subplots
    n_plots = int(has_loss) + int(has_accuracy)
    if n_plots == 0:
        logger.warning("No metrics found in history to plot")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot loss
    if has_loss:
        ax = axes[plot_idx]
        epochs = range(1, len(history['loss']) + 1)
        ax.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        if has_val_loss:
            ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot accuracy
    if has_accuracy:
        ax = axes[plot_idx]
        epochs = range(1, len(history[acc_key]) + 1)
        ax.plot(epochs, history[acc_key], 'b-', label='Training Accuracy', linewidth=2)
        if has_val_accuracy:
            ax.plot(epochs, history[val_acc_key], 'r-', label='Validation Accuracy', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Training curves saved to {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path],
    normalize: bool = False,
    title: str = "Confusion Matrix"
) -> None:
    """
    Plot confusion matrix as a heatmap with annotations.

    Args:
        cm: Confusion matrix array.
        class_names: Names of classes.
        save_path: Path to save the plot.
        normalize: If True, normalize the confusion matrix.
        title: Title for the plot.
    """
    logger.info(f"Plotting confusion matrix to {save_path}")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Replace NaN with 0

    plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))

    # Create heatmap
    fmt = '.2f' if normalize else 'd'
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path],
    title: str = "ROC Curves"
) -> None:
    """
    Plot ROC curves for each class (one-vs-rest).

    Args:
        y_true: True labels (1D array of class indices).
        y_pred_proba: Predicted probabilities (2D array of shape [n_samples, n_classes]).
        class_names: Names of classes.
        save_path: Path to save the plot.
        title: Title for the plot.
    """
    logger.info(f"Plotting ROC curves to {save_path}")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    plt.figure(figsize=(10, 8))

    # Compute ROC curve and ROC area for each class
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        if i >= y_pred_proba.shape[1]:
            logger.warning(f"Class {class_name} index {i} exceeds probability matrix columns")
            continue

        # Binarize the output
        y_true_binary = (y_true == i).astype(int)

        # Check if we have both classes
        if len(np.unique(y_true_binary)) < 2:
            logger.warning(f"Skipping ROC curve for class {class_name}: only one class present")
            continue

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot
        plt.plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=f'{class_name} (AUC = {roc_auc:.3f})'
        )

    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"ROC curves saved to {save_path}")


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path],
    title: str = "Precision-Recall Curves"
) -> None:
    """
    Plot precision-recall curves for each class (one-vs-rest).

    Args:
        y_true: True labels (1D array of class indices).
        y_pred_proba: Predicted probabilities (2D array of shape [n_samples, n_classes]).
        class_names: Names of classes.
        save_path: Path to save the plot.
        title: Title for the plot.
    """
    logger.info(f"Plotting precision-recall curves to {save_path}")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    plt.figure(figsize=(10, 8))

    # Compute precision-recall curve for each class
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        if i >= y_pred_proba.shape[1]:
            logger.warning(f"Class {class_name} index {i} exceeds probability matrix columns")
            continue

        # Binarize the output
        y_true_binary = (y_true == i).astype(int)

        # Check if we have both classes
        if len(np.unique(y_true_binary)) < 2:
            logger.warning(f"Skipping PR curve for class {class_name}: only one class present")
            continue

        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_proba[:, i])
        avg_precision = average_precision_score(y_true_binary, y_pred_proba[:, i])

        # Plot
        plt.plot(
            recall,
            precision,
            color=color,
            lw=2,
            label=f'{class_name} (AP = {avg_precision:.3f})'
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Precision-recall curves saved to {save_path}")


def plot_feature_importance(
    features: List[str],
    weights: np.ndarray,
    save_path: Union[str, Path],
    top_n: int = 50,
    title: str = "Feature Importance"
) -> None:
    """
    Plot feature importance as a horizontal bar chart.

    Args:
        features: List of feature names.
        weights: Feature weights/importance scores.
        save_path: Path to save the plot.
        top_n: Number of top features to display.
        title: Title for the plot.
    """
    logger.info(f"Plotting feature importance to {save_path}")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(features) != len(weights):
        raise ValueError(f"Number of features ({len(features)}) must match number of weights ({len(weights)})")

    # Get absolute values for sorting
    abs_weights = np.abs(weights)

    # Sort by importance
    sorted_indices = np.argsort(abs_weights)[::-1][:top_n]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))

    # Create color map (positive = green, negative = red)
    colors = ['green' if w >= 0 else 'red' for w in sorted_weights]

    # Plot horizontal bars
    y_pos = np.arange(len(sorted_features))
    ax.barh(y_pos, sorted_weights, color=colors, alpha=0.7)

    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, fontsize=9)
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel('Weight', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Feature importance plot saved to {save_path}")


def plot_convergence_curve(
    fitness_values: List[float],
    save_path: Union[str, Path],
    title: str = "Convergence Curve",
    xlabel: str = "Iteration",
    ylabel: str = "Fitness Value"
) -> None:
    """
    Plot convergence curve for optimization algorithms.

    Args:
        fitness_values: List of fitness values over iterations.
        save_path: Path to save the plot.
        title: Title for the plot.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
    """
    logger.info(f"Plotting convergence curve to {save_path}")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    iterations = range(1, len(fitness_values) + 1)
    plt.plot(iterations, fitness_values, 'b-', linewidth=2, marker='o', markersize=4, markevery=max(1, len(fitness_values) // 20))

    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add best value annotation
    best_idx = np.argmax(fitness_values) if 'accuracy' in ylabel.lower() else np.argmin(fitness_values)
    best_value = fitness_values[best_idx]
    plt.axhline(y=best_value, color='r', linestyle='--', linewidth=1, alpha=0.7, label=f'Best: {best_value:.4f}')
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Convergence curve saved to {save_path}")
