#!/usr/bin/env python
"""
RoBERT inference script for sentiment prediction.

This script loads a trained RoBERT model and makes predictions
on Romanian text inputs.
"""

# Disable TensorFlow backend in transformers to avoid Keras conflicts
import os
os.environ['USE_TF'] = 'NO'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import argparse
import sys
from pathlib import Path
import pandas as pd
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.robert_model import RoBERTModel
from preprocessing.basic_cleaner import BasicTextCleaner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def predict_single(model: RoBERTModel, text: str, cleaner: BasicTextCleaner) -> dict:
    """
    Predict sentiment for a single text.

    Args:
        model: Trained RoBERT model.
        text: Input text.
        cleaner: Text cleaner.

    Returns:
        Dictionary with prediction results.
    """
    # Clean text
    cleaned_text = cleaner.clean(text)

    # Predict
    pred, probs = model.predict(cleaned_text, return_probs=True)

    return {
        'text': text,
        'cleaned_text': cleaned_text,
        'label': int(pred),
        'sentiment': 'positive' if pred == 1 else 'negative',
        'confidence': float(probs[pred]),
        'prob_negative': float(probs[0]),
        'prob_positive': float(probs[1])
    }


def predict_batch(
    model: RoBERTModel,
    texts: list,
    cleaner: BasicTextCleaner,
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Predict sentiment for multiple texts.

    Args:
        model: Trained RoBERT model.
        texts: List of input texts.
        cleaner: Text cleaner.
        batch_size: Batch size for prediction.

    Returns:
        DataFrame with prediction results.
    """
    # Clean texts
    logger.info(f"Cleaning {len(texts)} texts...")
    cleaned_texts = cleaner.clean_batch(texts)

    # Predict
    logger.info(f"Making predictions (batch_size={batch_size})...")
    predictions, probs = model.predict(cleaned_texts, return_probs=True)

    # Create results dataframe
    results = pd.DataFrame({
        'text': texts,
        'cleaned_text': cleaned_texts,
        'label': predictions,
        'sentiment': ['positive' if p == 1 else 'negative' for p in predictions],
        'confidence': [float(probs[i][predictions[i]]) for i in range(len(predictions))],
        'prob_negative': probs[:, 0],
        'prob_positive': probs[:, 1]
    })

    return results


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict sentiment using trained RoBERT model')

    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained RoBERT model directory'
    )

    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--text',
        type=str,
        help='Single text to predict'
    )
    input_group.add_argument(
        '--input',
        type=str,
        help='Path to input file (one text per line or CSV with "text" column)'
    )

    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output CSV file (required for batch prediction)'
    )

    # Processing arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for prediction (default: 32)'
    )
    parser.add_argument(
        '--no-clean',
        action='store_true',
        help='Skip text cleaning (not recommended)'
    )

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading RoBERT model from {args.model}...")
    model = RoBERTModel.load(args.model)
    logger.info("Model loaded successfully")

    # Initialize cleaner
    cleaner = BasicTextCleaner(
        remove_urls=not args.no_clean,
        remove_html=not args.no_clean,
        normalize_whitespace=not args.no_clean,
        lowercase=False
    )

    # Single text prediction
    if args.text:
        logger.info("Predicting single text...")
        result = predict_single(model, args.text, cleaner)

        # Print results
        print("\n" + "="*80)
        print("PREDICTION RESULTS")
        print("="*80)
        print(f"Text: {result['text']}")
        if args.text != result['cleaned_text']:
            print(f"Cleaned: {result['cleaned_text']}")
        print(f"\nSentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nProbabilities:")
        print(f"  Negative: {result['prob_negative']:.2%}")
        print(f"  Positive: {result['prob_positive']:.2%}")
        print("="*80)

    # Batch prediction
    else:
        logger.info(f"Loading texts from {args.input}...")

        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)

        # Load texts
        if input_path.suffix == '.csv':
            # Read CSV file
            df = pd.read_csv(input_path)
            if 'text' not in df.columns:
                logger.error("CSV file must have a 'text' column")
                sys.exit(1)
            texts = df['text'].tolist()
        else:
            # Read text file (one text per line)
            with open(input_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(texts)} texts")

        # Predict
        results = predict_batch(model, texts, cleaner, args.batch_size)

        # Display summary
        print("\n" + "="*80)
        print("PREDICTION SUMMARY")
        print("="*80)
        print(f"Total texts: {len(results)}")
        print(f"Negative: {(results['sentiment'] == 'negative').sum()} ({(results['sentiment'] == 'negative').sum() / len(results):.1%})")
        print(f"Positive: {(results['sentiment'] == 'positive').sum()} ({(results['sentiment'] == 'positive').sum() / len(results):.1%})")
        print(f"Average confidence: {results['confidence'].mean():.2%}")
        print("="*80)

        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        else:
            # Print first few results
            print("\nFirst 10 predictions:")
            print(results[['text', 'sentiment', 'confidence']].head(10).to_string(index=False))
            print("\nUse --output to save all results to CSV")

        # Ask if user wants to see all results
        if not args.output and len(results) > 10:
            logger.warning(f"Showing only first 10 of {len(results)} predictions. Use --output to save all results.")


if __name__ == '__main__':
    main()
