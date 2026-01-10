"""
Train Logistic Regression model for sentiment analysis feature selection.
Uses full TF-IDF features with grid search for hyperparameter optimization.
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing.pipeline import PreprocessingPipeline
from utils.logger import setup_logger
from utils.config import load_config
from utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train Logistic Regression without IGWO')
    parser.add_argument('--experiment-name', type=str, required=True,
                        help='Name for this experiment (creates folder in results/experiments/)')
    parser.add_argument('--grid-search', action='store_true',
                        help='Perform grid search for C and penalty')
    return parser.parse_args()


def create_experiment_dir(experiment_name: str) -> Path:
    """Create directory for experiment results."""
    exp_dir = Path(__file__).parent.parent / 'results' / 'experiments' / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def grid_search_logistic_regression(X_train, y_train, X_val, y_val, logger):
    """Perform grid search to find best C and penalty parameters."""
    logger.info("Starting grid search for Logistic Regression hyperparameters")
    
    # Define parameter grid
    c_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    penalties = ['l2']  # Start with l2 only (works with all solvers)
    
    best_score = 0
    best_params = {}
    results = []
    
    total_combinations = len(c_values) * len(penalties)
    logger.info(f"Testing {total_combinations} parameter combinations")
    
    for c in c_values:
        for penalty in penalties:
            logger.info(f"\nTesting C={c}, penalty={penalty}")
            
            # Train model
            model = LogisticRegression(
                C=c,
                penalty=penalty,
                solver='saga',  # saga supports l1 and l2
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            try:
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_pred = model.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                val_f1 = f1_score(y_val, val_pred, average='macro')
                
                logger.info(f"  Validation Accuracy: {val_acc:.4f}")
                logger.info(f"  Validation F1 (macro): {val_f1:.4f}")
                
                results.append({
                    'C': c,
                    'penalty': penalty,
                    'val_accuracy': val_acc,
                    'val_f1': val_f1
                })
                
                # Track best model
                if val_acc > best_score:
                    best_score = val_acc
                    best_params = {'C': c, 'penalty': penalty}
                    logger.info(f"  ✓ New best validation accuracy: {val_acc:.4f}")
                    
            except Exception as e:
                logger.warning(f"  ✗ Failed with C={c}, penalty={penalty}: {e}")
                continue
    
    logger.info(f"\n{'='*80}")
    logger.info("GRID SEARCH RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Best parameters: C={best_params['C']}, penalty={best_params['penalty']}")
    logger.info(f"Best validation accuracy: {best_score:.4f}")
    logger.info(f"{'='*80}\n")
    
    return best_params, results


def main():
    args = parse_args()
    
    # Setup
    exp_dir = create_experiment_dir(args.experiment_name)
    logger = setup_logger('train_logistic_regression', exp_dir / 'training.log')
    
    logger.info("="*80)
    logger.info("LOGISTIC REGRESSION TRAINING (NO IGWO)")
    logger.info("="*80)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Grid Search: {args.grid_search}")
    logger.info(f"Results directory: {exp_dir}")
    
    # Set seed
    set_seed(42)
    
    # Load data
    logger.info("\n" + "="*80)
    logger.info("LOADING DATA")
    logger.info("="*80)
    
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
    
    # Combine title and content
    train_texts = [f"{row['title']} {row['content']}" for _, row in train_df.iterrows()]
    test_texts = [f"{row['title']} {row['content']}" for _, row in test_df.iterrows()]
    
    # Map star ratings to binary sentiment (1-2 stars = 0, 3-5 stars = 1)
    train_labels = np.array([0 if row['starRating'] in [1, 2] else 1 for _, row in train_df.iterrows()])
    test_labels = np.array([0 if row['starRating'] in [1, 2] else 1 for _, row in test_df.iterrows()])
    
    # Create validation split
    train_texts_split, val_texts, train_labels_split, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )
    
    logger.info(f"Training samples: {len(train_texts_split)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    logger.info(f"Test samples: {len(test_texts)}")
    logger.info(f"Class distribution (train): Negative={np.sum(train_labels_split==0)}, Positive={np.sum(train_labels_split==1)}")
    
    # Initialize preprocessing pipeline
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING")
    logger.info("="*80)
    
    # Load preprocessing config
    preprocessing_config = load_config('configs/preprocessing_config.yaml')
    
    pipeline = PreprocessingPipeline(preprocessing_config)
    
    logger.info("Preprocessing training data...")
    start_time = datetime.now()
    X_train_processed = pipeline.preprocess_batch(train_texts_split, batch_size=1000)
    preprocessing_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training preprocessing completed in {preprocessing_time:.2f}s")
    
    logger.info("Preprocessing validation data...")
    X_val_processed = pipeline.preprocess_batch(val_texts, batch_size=1000)
    
    logger.info("Preprocessing test data...")
    X_test_processed = pipeline.preprocess_batch(test_texts, batch_size=1000)
    
    y_train = train_labels_split
    y_val = val_labels
    y_test = test_labels
    
    # TF-IDF Vectorization (NO IGWO - use all features)
    logger.info("\n" + "="*80)
    logger.info("TF-IDF VECTORIZATION (NO IGWO)")
    logger.info("="*80)
    
    vectorizer = TfidfVectorizer(
        max_features=None,  # No feature limit
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        ngram_range=(1, 2)
    )
    
    logger.info("Fitting TF-IDF vectorizer on training data...")
    X_train_tfidf = vectorizer.fit_transform(X_train_processed)
    X_val_tfidf = vectorizer.transform(X_val_processed)
    X_test_tfidf = vectorizer.transform(X_test_processed)
    
    logger.info(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    logger.info(f"Training matrix shape: {X_train_tfidf.shape}")
    logger.info(f"Validation matrix shape: {X_val_tfidf.shape}")
    logger.info(f"Test matrix shape: {X_test_tfidf.shape}")
    
    # Grid search or single training
    if args.grid_search:
        best_params, grid_results = grid_search_logistic_regression(
            X_train_tfidf, y_train,
            X_val_tfidf, y_val,
            logger
        )
        
        # Save grid search results
        with open(exp_dir / 'grid_search_results.json', 'w') as f:
            json.dump(grid_results, f, indent=2)
        logger.info(f"Grid search results saved to {exp_dir / 'grid_search_results.json'}")
    else:
        best_params = {'C': 1.0, 'penalty': 'l2'}
        logger.info(f"Using default parameters: {best_params}")
    
    # Train final model with best parameters
    logger.info("\n" + "="*80)
    logger.info("TRAINING FINAL MODEL")
    logger.info("="*80)
    logger.info(f"Parameters: C={best_params['C']}, penalty={best_params['penalty']}")
    
    model = LogisticRegression(
        C=best_params['C'],
        penalty=best_params['penalty'],
        solver='saga',
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    logger.info("Training Logistic Regression model...")
    train_start = datetime.now()
    model.fit(X_train_tfidf, y_train)
    training_time = (datetime.now() - train_start).total_seconds()
    logger.info(f"Training completed in {training_time:.2f}s")
    
    # Evaluate on validation set
    logger.info("\n" + "="*80)
    logger.info("VALIDATION EVALUATION")
    logger.info("="*80)
    
    val_pred = model.predict(X_val_tfidf)
    val_metrics = {
        'accuracy': accuracy_score(y_val, val_pred),
        'precision_macro': precision_score(y_val, val_pred, average='macro'),
        'recall_macro': recall_score(y_val, val_pred, average='macro'),
        'f1_macro': f1_score(y_val, val_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_val, val_pred).tolist()
    }
    
    logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"Validation Precision (macro): {val_metrics['precision_macro']:.4f}")
    logger.info(f"Validation Recall (macro): {val_metrics['recall_macro']:.4f}")
    logger.info(f"Validation F1 (macro): {val_metrics['f1_macro']:.4f}")
    
    # Evaluate on test set
    logger.info("\n" + "="*80)
    logger.info("TEST EVALUATION")
    logger.info("="*80)
    
    test_pred = model.predict(X_test_tfidf)
    test_metrics = {
        'accuracy': accuracy_score(y_test, test_pred),
        'precision_macro': precision_score(y_test, test_pred, average='macro'),
        'recall_macro': recall_score(y_test, test_pred, average='macro'),
        'f1_macro': f1_score(y_test, test_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
    }
    
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision (macro): {test_metrics['precision_macro']:.4f}")
    logger.info(f"Test Recall (macro): {test_metrics['recall_macro']:.4f}")
    logger.info(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    
    # Classification report
    logger.info("\n" + "="*80)
    logger.info("CLASSIFICATION REPORT (TEST SET)")
    logger.info("="*80)
    logger.info("\n" + classification_report(
        y_test, test_pred,
        target_names=['Negative', 'Positive'],
        digits=4
    ))
    
    # Save models and artifacts
    logger.info("\n" + "="*80)
    logger.info("SAVING ARTIFACTS")
    logger.info("="*80)
    
    model_path = exp_dir / 'logistic_regression_model.joblib'
    preprocessor_path = exp_dir / 'preprocessor.joblib'
    vectorizer_path = exp_dir / 'tfidf_vectorizer.joblib'
    
    joblib.dump(model, model_path)
    logger.info(f"✓ Model saved to {model_path}")
    
    joblib.dump(pipeline, preprocessor_path)
    logger.info(f"✓ Preprocessor saved to {preprocessor_path}")
    
    joblib.dump(vectorizer, vectorizer_path)
    logger.info(f"✓ Vectorizer saved to {vectorizer_path}")
    
    # Save metrics
    metrics = {
        'experiment_name': args.experiment_name,
        'timestamp': datetime.now().isoformat(),
        'model_type': 'Logistic Regression',
        'feature_selection': 'None (Full TF-IDF)',
        'preprocessing': {
            'stemming': False,
            'lemmatization': True,
            'language': 'ro'
        },
        'hyperparameters': {
            'C': best_params['C'],
            'penalty': best_params['penalty'],
            'solver': 'saga',
            'max_iter': 1000
        },
        'vocabulary_size': len(vectorizer.vocabulary_),
        'training_time_seconds': training_time,
        'preprocessing_time_seconds': preprocessing_time,
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'accuracy': test_metrics['accuracy'],
        'precision_macro': test_metrics['precision_macro'],
        'recall_macro': test_metrics['recall_macro'],
        'f1_macro': test_metrics['f1_macro'],
        'confusion_matrix': test_metrics['confusion_matrix']
    }
    
    metrics_path = exp_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✓ Metrics saved to {metrics_path}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Final Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    logger.info(f"Results saved to: {exp_dir}")


if __name__ == '__main__':
    main()
