#!/usr/bin/env python
"""
Optimized XGBoost training with hyperparameter tuning using Optuna.

This script uses Bayesian optimization to find the best hyperparameters
for both IGWO feature selection and XGBoost classification.
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
import numpy as np

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

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed. Install with: pip install optuna")

# Set up logging
logger = setup_logger(
    name='train_xgboost_optimized',
    log_dir=Path('logs'),
    log_filename=f'train_xgboost_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO
)


def main():
    """Main training function with hyperparameter optimization."""
    parser = argparse.ArgumentParser(description='Train optimized XGBoost sentiment model')
    parser.add_argument('--experiment-name', type=str, default=f'xgb_opt_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--skip-optuna', action='store_true', help='Skip optimization, use best known params')
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("OPTIMIZED XGBOOST SENTIMENT ANALYSIS TRAINING")
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

    # Hyperparameter optimization with Optuna
    if not args.skip_optuna and OPTUNA_AVAILABLE:
        logger.info(f"Starting Optuna hyperparameter optimization ({args.n_trials} trials)...")

        def objective(trial):
            """Optuna objective function."""
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            }

            model = XGBoostModel(hyperparams=params, n_classes=2)
            model.fit(X_train_split, y_train_split, X_val, y_val, verbose=False)

            # Return validation accuracy
            val_acc = model.score(X_val, y_val)
            return val_acc

        study = optuna.create_study(direction='maximize', study_name='xgboost_optimization')
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

        logger.info(f"Best validation accuracy: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters: {study.best_params}")

        best_params = study.best_params

        # Save optimization results
        with open(experiment_dir / 'optuna_study.json', 'w') as f:
            json.dump({
                'best_value': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials)
            }, f, indent=2)

    else:
        # Use best known parameters (from previous runs)
        logger.info("Using best known hyperparameters...")
        best_params = {
            'n_estimators': 300,
            'max_depth': 10,
            'learning_rate': 0.05,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.5,
        }

    # Train final model with best parameters
    logger.info("Training final XGBoost model with optimized hyperparameters...")
    model = XGBoostModel(hyperparams=best_params, n_classes=2)
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
    if 'roc_auc_macro' in metrics and metrics['roc_auc_macro'] is not None:
        logger.info(f"  ROC AUC: {metrics['roc_auc_macro']:.4f}")

    # Save results
    with open(experiment_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(experiment_dir / 'best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)

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
