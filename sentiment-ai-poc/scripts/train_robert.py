#!/usr/bin/env python
"""
RoBERT (Romanian BERT) training script for sentiment analysis.

This script trains RoBERT on the LaRoSeDa dataset for binary sentiment
classification (negative/positive).
"""

# Disable TensorFlow backend in transformers to avoid Keras conflicts
import os
os.environ['USE_TF'] = 'NO'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

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
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import load_config
from utils.logger import setup_logger
from utils.reproducibility import set_seed
from preprocessing.basic_cleaner import BasicTextCleaner
from models.robert_model import RoBERTModel
from evaluation.metrics import calculate_metrics
from evaluation.visualizer import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves
)

# Set up logging
logger = setup_logger(
    name='train_robert',
    log_dir=Path('logs'),
    log_filename=f'train_robert_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO
)


def load_laroseda_dataset():
    """
    Load LaRoSeDa dataset from Hugging Face.

    Returns:
        Tuple of (train_df, test_df).
    """
    logger.info("Loading LaRoSeDa dataset from Hugging Face...")

    from huggingface_hub import hf_hub_download

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

    logger.info(f"Loaded {len(train_df)} train and {len(test_df)} test samples")

    return train_df, test_df


def prepare_data(train_df, test_df, config):
    """
    Prepare data for training.

    Args:
        train_df: Training dataframe.
        test_df: Test dataframe.
        config: Configuration dictionary.

    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels).
    """
    logger.info("Preparing data...")

    # Initialize cleaner
    preprocessing_config = config.get('preprocessing', {})
    cleaner = BasicTextCleaner(
        remove_urls=preprocessing_config.get('remove_urls', True),
        remove_html=preprocessing_config.get('remove_html', True),
        normalize_whitespace=preprocessing_config.get('normalize_whitespace', True),
        lowercase=preprocessing_config.get('lowercase', False)
    )

    # Combine title and content
    logger.info("Combining title and content fields...")
    train_texts_raw = [
        f"{row['title']} {row['content']}"
        for _, row in train_df.iterrows()
    ]
    test_texts_raw = [
        f"{row['title']} {row['content']}"
        for _, row in test_df.iterrows()
    ]

    # Clean texts
    logger.info("Cleaning texts...")
    train_texts = cleaner.clean_batch(train_texts_raw)
    test_texts = cleaner.clean_batch(test_texts_raw)

    # Create binary labels (1-2 stars = 0 negative, 3-5 stars = 1 positive)
    data_config = config.get('data', {})
    negative_stars = data_config.get('negative_stars', [1, 2])
    positive_stars = data_config.get('positive_stars', [3, 4, 5])

    logger.info(f"Creating binary labels: {negative_stars} → 0 (negative), {positive_stars} → 1 (positive)")

    train_labels = [
        0 if row['starRating'] in negative_stars else 1
        for _, row in train_df.iterrows()
    ]
    test_labels = [
        0 if row['starRating'] in negative_stars else 1
        for _, row in test_df.iterrows()
    ]

    # Log class distribution
    train_labels_array = np.array(train_labels)
    test_labels_array = np.array(test_labels)

    logger.info(f"Train class distribution: Negative={np.sum(train_labels_array == 0)}, Positive={np.sum(train_labels_array == 1)}")
    logger.info(f"Test class distribution: Negative={np.sum(test_labels_array == 0)}, Positive={np.sum(test_labels_array == 1)}")

    return train_texts, train_labels, test_texts, test_labels


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train RoBERT sentiment model')
    parser.add_argument('--experiment-name', type=str, default=f'robert_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('--config', type=str, default='configs/robert_config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--learning-rate', type=float, default=None, help='Override learning rate')
    parser.add_argument('--max-length', type=int, default=None, help='Override max sequence length')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio')
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("ROBERT (ROMANIAN BERT) SENTIMENT ANALYSIS TRAINING")
    logger.info("="*80)
    logger.info(f"Experiment name: {args.experiment_name}")

    # Set seed
    set_seed(42)

    # Load config
    logger.info(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Override config with command-line arguments
    training_config = config.get('training', {})
    model_config = config.get('model', {})

    if args.epochs is not None:
        training_config['num_train_epochs'] = args.epochs
    if args.batch_size is not None:
        training_config['per_device_train_batch_size'] = args.batch_size
    if args.learning_rate is not None:
        training_config['learning_rate'] = args.learning_rate
    if args.max_length is not None:
        model_config['max_length'] = args.max_length

    # Create experiment directory
    experiment_dir = Path(f"results/experiments/{args.experiment_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train_df, test_df = load_laroseda_dataset()

    # Prepare data
    train_texts, train_labels, test_texts, test_labels = prepare_data(
        train_df, test_df, config
    )

    # Create validation split from training data
    logger.info(f"Creating validation split ({args.validation_split:.1%})...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=args.validation_split,
        random_state=42,
        stratify=train_labels
    )

    logger.info(f"Final dataset sizes:")
    logger.info(f"  Train: {len(train_texts)} samples")
    logger.info(f"  Validation: {len(val_texts)} samples")
    logger.info(f"  Test: {len(test_texts)} samples")

    # Initialize RoBERT model
    logger.info("Initializing RoBERT model...")
    model = RoBERTModel(
        model_name=model_config.get('name', 'dumitrescustefan/bert-base-romanian-cased-v1'),
        num_labels=model_config.get('num_labels', 2),
        max_length=model_config.get('max_length', 512)
    )

    # Train model
    logger.info("Training RoBERT model...")
    eval_metrics = model.fit(
        train_texts=train_texts,
        train_labels=train_labels,
        eval_texts=val_texts,
        eval_labels=val_labels,
        output_dir=experiment_dir / "checkpoints",
        num_train_epochs=training_config.get('num_train_epochs', 5),
        learning_rate=training_config.get('learning_rate', 2e-5),
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 16),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 32),
        warmup_ratio=training_config.get('warmup_ratio', 0.1),
        weight_decay=training_config.get('weight_decay', 0.01),
        logging_steps=training_config.get('logging_steps', 50),
        eval_strategy=training_config.get('eval_strategy', 'epoch'),
        save_strategy=training_config.get('save_strategy', 'epoch'),
        save_total_limit=training_config.get('save_total_limit', 2),
        load_best_model_at_end=training_config.get('load_best_model_at_end', True),
        metric_for_best_model=training_config.get('metric_for_best_model', 'accuracy'),
        early_stopping_patience=training_config.get('early_stopping_patience', 2),
        seed=training_config.get('seed', 42)
    )

    # Save final model
    logger.info("Saving final model...")
    model.save(experiment_dir / "robert_model")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    y_pred, y_pred_proba = model.predict(test_texts, return_probs=True)
    y_pred = np.array(y_pred)
    test_labels_array = np.array(test_labels)

    # Calculate metrics
    metrics = calculate_metrics(test_labels_array, y_pred, y_pred_proba)

    logger.info("\n" + "="*80)
    logger.info("TEST RESULTS:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall:    {metrics['recall_macro']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1_macro']:.4f}")
    if 'roc_auc_macro' in metrics and metrics['roc_auc_macro'] is not None:
        logger.info(f"  ROC AUC:   {metrics['roc_auc_macro']:.4f}")
    logger.info("="*80)

    # Save metrics
    with open(experiment_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    predictions_df = pd.DataFrame({
        'text': test_texts,
        'true_label': test_labels_array,
        'predicted_label': y_pred,
        'prob_negative': y_pred_proba[:, 0],
        'prob_positive': y_pred_proba[:, 1]
    })
    predictions_df.to_csv(experiment_dir / 'predictions.csv', index=False)

    # Generate plots
    logger.info("Generating visualizations...")
    plots_dir = experiment_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    class_names = metrics.get('class_names', ['Negative', 'Positive'])
    cm = np.array(metrics['confusion_matrix'])

    plot_confusion_matrix(cm, class_names, plots_dir / 'confusion_matrix.png')
    plot_roc_curves(test_labels_array, y_pred_proba, class_names, plots_dir / 'roc_curves.png')
    plot_precision_recall_curves(test_labels_array, y_pred_proba, class_names, plots_dir / 'precision_recall.png')

    # Save config used
    with open(experiment_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Results saved to: {experiment_dir}")
    logger.info("="*80)

    # Print comparison with XGBoost if available
    logger.info("\nTo compare with XGBoost results, check:")
    logger.info("  - XGBoost results: results/experiments/xgb_opt_*/metrics.json")
    logger.info(f"  - RoBERT results: {experiment_dir}/metrics.json")


if __name__ == '__main__':
    main()
