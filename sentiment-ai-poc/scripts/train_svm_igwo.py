#!/usr/bin/env python
"""
SVM training script for sentiment classification.

This script trains a Support Vector Machine classifier using the same
preprocessing and feature selection pipeline as XGBoost for direct comparison.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import argparse
import sys
import logging
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import load_config
from utils.logger import setup_logger
from utils.reproducibility import set_seed
from preprocessing.pipeline import PreprocessingPipeline
from features.selector import FeatureSelector
from evaluation.metrics import calculate_metrics
from evaluation.visualizer import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves
)

# Set up logging
logger = setup_logger(
    name='train_svm',
    log_dir=Path('logs'),
    log_filename=f'train_svm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO
)


def main():
    """Main training function for SVM."""
    parser = argparse.ArgumentParser(description='Train SVM sentiment model')
    parser.add_argument('--experiment-name', type=str, default=f'svm_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('--C', type=float, default=1.0, help='SVM regularization parameter')
    parser.add_argument('--grid-search', action='store_true', help='Perform grid search over C values')
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("SVM SENTIMENT ANALYSIS TRAINING")
    logger.info("="*80)
    logger.info(f"Experiment name: {args.experiment_name}")

    # Set seed
    set_seed(42)

    # Load configs
    logger.info("Loading configurations...")
    preprocessing_config = load_config('configs/preprocessing_config.yaml')
    feature_config = load_config('configs/feature_config.yaml')

    # Create experiment directory
    experiment_dir = Path(f"results/experiments/{args.experiment_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset...")
    from huggingface_hub import hf_hub_download
    import pandas as pd

    train_file = hf_hub_download(
        repo_id="universityofbucharest/laroseda",
        filename="laroseda/train/0000.parquet",
        repo_type="dataset",
        revision="refs/convert/parquet"
    )
    test_file = hf_hub_download(
        repo_id="universityofbucharest/laroseda",
        filename="laroseda/test/0000.parquet",
        repo_type="dataset",
        revision="refs/convert/parquet"
    )

    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)

    train_texts = [f"{row['title']} {row['content']}" for _, row in train_df.iterrows()]
    test_texts = [f"{row['title']} {row['content']}" for _, row in test_df.iterrows()]
    train_labels = np.array([0 if row['starRating'] in [1, 2] else 1 for _, row in train_df.iterrows()])
    test_labels = np.array([0 if row['starRating'] in [1, 2] else 1 for _, row in test_df.iterrows()])

    logger.info(f"Loaded {len(train_texts)} train and {len(test_texts)} test samples")

    # Preprocess
    logger.info("Preprocessing texts...")
    preprocessor = PreprocessingPipeline(preprocessing_config)
    train_preprocessed = preprocessor.preprocess_batch(train_texts, batch_size=1000)
    test_preprocessed = preprocessor.preprocess_batch(test_texts, batch_size=1000)

    # Save preprocessor
    preprocessor.save(experiment_dir / 'preprocessor.joblib')

    # Feature extraction and selection using config
    logger.info("Extracting and selecting features with IGWO from config...")
    lf_micf_config = feature_config.get('lf_micf', {})
    igwo_config = feature_config.get('igwo', {})
    
    selector = FeatureSelector(
        extractor_params={
            'min_df': lf_micf_config.get('min_df', 5),
            'max_df': lf_micf_config.get('max_df', 0.8),
        },
        selector_params={
            'n_wolves': igwo_config.get('n_wolves', 10),
            'n_iterations': igwo_config.get('n_iterations', 20),
            'inertia_weight': igwo_config.get('inertia_weight', 0.9),
            'target_features': igwo_config.get('target_features', 800),
            'cv_folds': igwo_config.get('fitness_cv_folds', 3),
            'random_state': igwo_config.get('random_state', 42),
        }
    )

    # Use sample size from config
    sample_size = igwo_config.get('sample_size', 1000)
    logger.info(f"Using sample_size={sample_size} for IGWO feature selection")
    indices = np.random.RandomState(42).choice(len(train_preprocessed), min(sample_size, len(train_preprocessed)), replace=False)
    sample_texts = [train_preprocessed[i] for i in indices]
    sample_labels = train_labels[indices]

    selector.fit(sample_texts, sample_labels, verbose=True)

    # Transform datasets
    X_train = selector.transform(train_preprocessed)
    X_test = selector.transform(test_preprocessed)

    logger.info(f"Selected {X_train.shape[1]} features")

    # Save selector
    selector.save(experiment_dir / 'feature_selector.joblib')

    # Split train into train/val
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    # Grid search or single C value
    if args.grid_search:
        logger.info("Performing grid search over C values...")
        C_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
        best_C = None
        best_val_acc = 0.0
        results = {}
        
        for C in C_values:
            logger.info(f"  Testing C={C}...")
            svm = LinearSVC(C=C, max_iter=1000, random_state=42, dual=False)
            svm.fit(X_train_split, y_train_split)
            
            # Quick validation accuracy check
            val_acc = svm.score(X_val, y_val)
            results[C] = val_acc
            logger.info(f"    Validation accuracy: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_C = C
        
        logger.info(f"\nGrid search results:")
        for C, acc in results.items():
            logger.info(f"  C={C}: {acc:.4f}")
        logger.info(f"\nBest C: {best_C} with validation accuracy: {best_val_acc:.4f}")
        
        # Save grid search results
        with open(experiment_dir / 'grid_search_results.json', 'w') as f:
            json.dump({'results': {str(k): v for k, v in results.items()}, 'best_C': best_C, 'best_val_acc': best_val_acc}, f, indent=2)
        
        final_C = best_C
        logger.info(f"\nTraining final model with C={final_C}...")
    else:
        final_C = args.C
        logger.info(f"Training LinearSVC with C={final_C}...")

    # Train final SVM
    svm = LinearSVC(
        C=final_C,
        max_iter=1000,
        random_state=42,
        dual=False  # Faster for n_samples > n_features
    )
    svm.fit(X_train_split, y_train_split)

    # Wrap with CalibratedClassifierCV for probability estimates
    logger.info("Calibrating SVM for probability estimates...")
    calibrated_svm = CalibratedClassifierCV(svm, cv=3)
    calibrated_svm.fit(X_train_split, y_train_split)

    # Evaluate on validation set
    val_acc = calibrated_svm.score(X_val, y_val)
    logger.info(f"Final validation accuracy: {val_acc:.4f}")

    # Evaluate on test set
    logger.info("Evaluating model on test set...")
    y_pred = calibrated_svm.predict(X_test)
    y_pred_proba = calibrated_svm.predict_proba(X_test)

    metrics = calculate_metrics(test_labels, y_pred, y_pred_proba)

    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall: {metrics['recall_macro']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_macro']:.4f}")
    if 'roc_auc_macro' in metrics and metrics['roc_auc_macro'] is not None:
        logger.info(f"  ROC AUC: {metrics['roc_auc_macro']:.4f}")

    # Save results
    with open(experiment_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(experiment_dir / 'hyperparameters.json', 'w') as f:
        json.dump({'C': final_C}, f, indent=2)

    joblib.dump(calibrated_svm, experiment_dir / 'svm_model.joblib')

    # Generate plots
    plots_dir = experiment_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Get class names from metrics
    class_names = metrics.get('class_names', ['Negative', 'Positive'])
    cm = np.array(metrics['confusion_matrix'])

    plot_confusion_matrix(cm, class_names, plots_dir / 'confusion_matrix.png')
    plot_roc_curves(test_labels, y_pred_proba, class_names, plots_dir / 'roc_curves.png')
    plot_precision_recall_curves(test_labels, y_pred_proba, class_names, plots_dir / 'precision_recall.png')

    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Results saved to: {experiment_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
