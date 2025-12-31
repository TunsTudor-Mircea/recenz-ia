"""
Text cleaning module for Romanian text preprocessing.

This module provides functionality to clean text by removing HTML tags,
URLs, special characters, and normalizing whitespace.
"""

import re
from typing import List, Optional
import unicodedata


class TextCleaner:
    """
    Clean Romanian text data for sentiment analysis.

    Handles removal of HTML tags, URLs, special characters, and normalizes
    whitespace while preserving Romanian diacritics and sentiment-relevant
    punctuation.

    Attributes:
        lowercase: Whether to convert text to lowercase.
        preserve_punctuation: List of punctuation marks to preserve.
    """

    def __init__(
        self,
        lowercase: bool = True,
        preserve_punctuation: Optional[List[str]] = None
    ):
        """
        Initialize TextCleaner.

        Args:
            lowercase: If True, convert text to lowercase.
            preserve_punctuation: List of punctuation to preserve (e.g., ['!', '?']).
                                If None, preserves common sentiment punctuation.
        """
        self.lowercase = lowercase
        self.preserve_punctuation = preserve_punctuation or ['!', '?', '.']

        # Compile regex patterns for efficiency
        self._html_pattern = re.compile(r'<[^>]+>')
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._email_pattern = re.compile(r'\S+@\S+')
        self._whitespace_pattern = re.compile(r'\s+')

    def clean(self, text: str) -> str:
        """
        Clean a single text string.

        Args:
            text: Input text to clean.

        Returns:
            Cleaned text string.
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove HTML tags
        text = self._html_pattern.sub(' ', text)

        # Remove URLs
        text = self._url_pattern.sub(' ', text)

        # Remove email addresses
        text = self._email_pattern.sub(' ', text)

        # Normalize unicode characters (preserve Romanian diacritics)
        text = unicodedata.normalize('NFC', text)

        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()

        # Remove special characters but preserve Romanian diacritics and specified punctuation
        # Romanian diacritics: ă, â, î, ș, ț (and uppercase)
        romanian_chars = r'a-zA-ZăâîșțĂÂÎȘȚ'
        preserved_punct = ''.join(re.escape(p) for p in self.preserve_punctuation)
        pattern = f'[^{romanian_chars}0-9{preserved_punct}\\s]'
        text = re.sub(pattern, ' ', text)

        # Normalize whitespace
        text = self._whitespace_pattern.sub(' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of text strings.

        Args:
            texts: List of input texts to clean.

        Returns:
            List of cleaned text strings.
        """
        return [self.clean(text) for text in texts]
