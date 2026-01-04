"""
Stemming module for Romanian text preprocessing.

This module provides Romanian stemming using NLTK's Snowball stemmer.
"""

from typing import List
import logging

logger = logging.getLogger(__name__)


class RomanianStemmer:
    """
    Romanian stemmer using NLTK's Snowball stemmer.

    The Snowball stemmer reduces words to their root form.
    """

    def __init__(self):
        """Initialize RomanianStemmer."""
        self._stemmer = None

    def _ensure_stemmer(self) -> None:
        """Lazy load NLTK Snowball stemmer."""
        if self._stemmer is None:
            try:
                from nltk.stem.snowball import SnowballStemmer
                self._stemmer = SnowballStemmer('romanian')
            except ImportError as e:
                logger.error("NLTK not installed. Please install with: pip install nltk")
                raise ImportError("NLTK is required for stemming") from e
            except Exception as e:
                logger.error(f"Failed to initialize Romanian Snowball stemmer: {e}")
                raise

    def stem(self, tokens: List[str]) -> List[str]:
        """
        Stem a list of tokens.

        Args:
            tokens: List of tokens to stem.

        Returns:
            List of stemmed tokens.
        """
        if not tokens:
            return []

        self._ensure_stemmer()

        try:
            stemmed_tokens = [self._stemmer.stem(token) for token in tokens]
            return stemmed_tokens
        except Exception as e:
            logger.warning(f"Stemming failed: {e}. Returning original tokens.")
            return tokens

    def stem_batch(self, token_lists: List[List[str]]) -> List[List[str]]:
        """
        Stem a batch of token lists.

        Args:
            token_lists: List of token lists to stem.

        Returns:
            List of stemmed token lists.
        """
        return [self.stem(tokens) for tokens in token_lists]
