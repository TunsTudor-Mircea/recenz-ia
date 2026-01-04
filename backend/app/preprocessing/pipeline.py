"""
Preprocessing pipeline module for Romanian text preprocessing.

This module provides a configurable pipeline that orchestrates all
preprocessing steps.
"""

from typing import List, Dict, Any, Union
import logging
from pathlib import Path
import joblib
from tqdm import tqdm

from preprocessing.cleaner import TextCleaner
from preprocessing.tokenizer import WhitespaceTokenizer
from preprocessing.lemmatizer import RomanianLemmatizer
from preprocessing.stemmer import RomanianStemmer

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for Romanian text.

    Orchestrates text cleaning, tokenization, lemmatization, and stemming
    with configurable options and batch processing support.

    Attributes:
        config: Configuration dictionary for preprocessing steps.
        cleaner: TextCleaner instance.
        tokenizer: WhitespaceTokenizer instance.
        lemmatizer: RomanianLemmatizer instance or None.
        stemmer: RomanianStemmer instance or None.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PreprocessingPipeline.

        Args:
            config: Configuration dictionary with keys:
                - lowercase: bool
                - remove_stopwords: bool
                - lemmatize: bool
                - stem: bool
                - min_token_length: int
                - max_token_length: int
        """
        self.config = config

        # Initialize components
        self.cleaner = TextCleaner(
            lowercase=config.get('lowercase', True)
        )

        self.tokenizer = WhitespaceTokenizer(
            min_token_length=config.get('min_token_length', 2),
            max_token_length=config.get('max_token_length', 50)
        )

        # Initialize lemmatizer if enabled
        self.lemmatizer = None
        if config.get('lemmatize', True):
            self.lemmatizer = RomanianLemmatizer(
                remove_stopwords=config.get('remove_stopwords', True)
            )

        # Initialize stemmer if enabled
        self.stemmer = None
        if config.get('stem', True):
            self.stemmer = RomanianStemmer()

        logger.info("PreprocessingPipeline initialized with config: %s", config)

    def preprocess(self, text: str) -> str:
        """
        Preprocess a single text string.

        Args:
            text: Input text to preprocess.

        Returns:
            Preprocessed text string (tokens joined by spaces).
        """
        # Clean text
        cleaned = self.cleaner.clean(text)

        # Tokenize
        tokens = self.tokenizer.tokenize(cleaned)

        # Lemmatize if enabled
        if self.lemmatizer is not None:
            tokens = self.lemmatizer.lemmatize(tokens)

        # Stem if enabled
        if self.stemmer is not None:
            tokens = self.stemmer.stem(tokens)

        # Join tokens back to string
        return ' '.join(tokens)

    def preprocess_batch(
        self,
        texts: List[str],
        batch_size: int = 1000,
        show_progress: bool = True
    ) -> List[str]:
        """
        Preprocess a batch of texts with progress tracking.

        Args:
            texts: List of input texts to preprocess.
            batch_size: Number of texts to process at once.
            show_progress: If True, show progress bar.

        Returns:
            List of preprocessed text strings.
        """
        results = []
        n_batches = (len(texts) + batch_size - 1) // batch_size

        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Preprocessing texts")

        for i in iterator:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch = texts[start_idx:end_idx]

            # Process batch
            batch_results = [self.preprocess(text) for text in batch]
            results.extend(batch_results)

        return results

    def fit(self, texts: List[str]) -> 'PreprocessingPipeline':
        """
        Fit the pipeline (placeholder for sklearn compatibility).

        Currently no fitting is needed, but this method exists for
        sklearn-compatible API.

        Args:
            texts: Training texts (not used).

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting preprocessing pipeline (no-op)")
        return self

    def transform(self, texts: List[str]) -> List[str]:
        """
        Transform texts (sklearn-compatible API).

        Args:
            texts: Texts to transform.

        Returns:
            Preprocessed texts.
        """
        return self.preprocess_batch(texts)

    def fit_transform(self, texts: List[str]) -> List[str]:
        """
        Fit and transform texts (sklearn-compatible API).

        Args:
            texts: Texts to fit and transform.

        Returns:
            Preprocessed texts.
        """
        return self.fit(texts).transform(texts)

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save preprocessor state to disk.

        Args:
            filepath: Path to save the preprocessor.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'config': self.config,
            'cleaner': self.cleaner,
            'tokenizer': self.tokenizer,
            'lemmatizer': self.lemmatizer,
            'stemmer': self.stemmer,
        }

        joblib.dump(save_data, filepath)
        logger.info(f"Saved preprocessing pipeline to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'PreprocessingPipeline':
        """
        Load preprocessor state from disk.

        Args:
            filepath: Path to load the preprocessor from.

        Returns:
            Loaded PreprocessingPipeline instance.
        """
        save_data = joblib.load(filepath)

        # Create instance with saved config
        instance = cls(save_data['config'])

        # Restore saved components
        instance.cleaner = save_data['cleaner']
        instance.tokenizer = save_data['tokenizer']
        instance.lemmatizer = save_data['lemmatizer']
        instance.stemmer = save_data['stemmer']

        logger.info(f"Loaded preprocessing pipeline from {filepath}")
        return instance

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.

        Returns:
            Configuration dictionary.
        """
        return self.config.copy()
