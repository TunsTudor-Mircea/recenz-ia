#!/usr/bin/env python
"""
XGBoost training script - faster and better for TF-IDF features.

This script uses the same preprocessing and feature selection (IGWO),
but replaces LSTM with XGBoost for much better results.
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import load_config
from utils.logger import setup_logger
from utils.reproducibility import set_seed
from preprocessing.pipeline import PreprocessingPipeline
from features.selector import FeatureSelector
from models.xgboost_model import XGBoostModel
from evaluation.metrics import calculate_metrics
from evaluation.visualizer import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves
)

# Set up logging
logger = setup_logger(
    name='train_xgboost',
    log_dir=Path('logs'),
    log_filename=f'train_xgboost_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO
)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train XGBoost sentiment model')
    parser.add_argument('--experiment-name', type=str, default=f'xgb_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("XGBOOST SENTIMENT ANALYSIS TRAINING")
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
    import numpy as np

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

    # Feature extraction and selection
    logger.info("Extracting and selecting features with IGWO...")
    selector = FeatureSelector(
        extractor_params={
            'min_df': feature_config['lf_micf']['min_df'],
            'max_df': feature_config['lf_micf']['max_df'],
        },
        selector_params={
            'n_wolves': feature_config['igwo']['n_wolves'],
            'n_iterations': feature_config['igwo']['n_iterations'],
            'inertia_weight': feature_config['igwo']['inertia_weight'],
            'target_features': feature_config['igwo']['target_features'],
            'cv_folds': feature_config['igwo']['fitness_cv_folds'],
            'random_state': feature_config['igwo']['random_state'],
        }
    )

    # Use sample for IGWO
    sample_size = feature_config['igwo'].get('sample_size', 1000)
    if len(train_preprocessed) > sample_size:
        import numpy as np
        indices = np.random.RandomState(42).choice(len(train_preprocessed), sample_size, replace=False)
        sample_texts = [train_preprocessed[i] for i in indices]
        sample_labels = train_labels[indices]
    else:
        sample_texts = train_preprocessed
        sample_labels = train_labels

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

    # Train XGBoost
    logger.info("Training XGBoost model...")
    model = XGBoostModel(
        hyperparams={
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        n_classes=2
    )

    model.fit(X_train_split, y_train_split, X_val, y_val, verbose=True)

    # Evaluate
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = calculate_metrics(test_labels, y_pred, y_pred_proba)

    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall: {metrics['recall_macro']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_macro']:.4f}")

    # Save results
    with open(experiment_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    model.save(experiment_dir / 'xgboost_model.joblib')

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
