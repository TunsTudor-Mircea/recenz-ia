"""
Tokenization module for Romanian text preprocessing.

This module provides whitespace-based tokenization with filtering capabilities.
"""

from typing import List


class WhitespaceTokenizer:
    """
    Whitespace-based tokenizer with token length filtering.

    Attributes:
        min_token_length: Minimum length for tokens to keep.
        max_token_length: Maximum length for tokens to keep.
    """

    def __init__(
        self,
        min_token_length: int = 2,
        max_token_length: int = 50
    ):
        """
        Initialize WhitespaceTokenizer.

        Args:
            min_token_length: Minimum token length (inclusive).
            max_token_length: Maximum token length (inclusive).
        """
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using whitespace splitting with length filtering.

        Args:
            text: Input text to tokenize.

        Returns:
            List of tokens.
        """
        if not text or not isinstance(text, str):
            return []

        # Split on whitespace
        tokens = text.split()

        # Filter by length
        tokens = [
            token for token in tokens
            if self.min_token_length <= len(token) <= self.max_token_length
        ]

        return tokens

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize a batch of texts.

        Args:
            texts: List of input texts to tokenize.

        Returns:
            List of token lists.
        """
        return [self.tokenize(text) for text in texts]
