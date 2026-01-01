"""
Basic text cleaning module for RoBERT preprocessing.

This module provides minimal preprocessing for BERT-based models.
Unlike traditional ML models, BERT models need:
- Original word forms (no lemmatization/stemming)
- All words including stopwords (for context)
- Minimal cleaning (URLs, HTML, whitespace normalization)
"""

import re
from typing import List
import logging

logger = logging.getLogger(__name__)


class BasicTextCleaner:
    """
    Basic text cleaner for RoBERT preprocessing.

    This cleaner applies minimal preprocessing to preserve the text
    structure needed for contextual embeddings in transformer models.

    Attributes:
        remove_urls: Whether to remove URLs.
        remove_html: Whether to remove HTML tags.
        normalize_whitespace: Whether to normalize whitespace.
        lowercase: Whether to convert to lowercase (not recommended for cased BERT).
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_html: bool = True,
        normalize_whitespace: bool = True,
        lowercase: bool = False
    ):
        """
        Initialize BasicTextCleaner.

        Args:
            remove_urls: If True, remove HTTP/HTTPS/WWW URLs.
            remove_html: If True, remove HTML tags.
            normalize_whitespace: If True, normalize excessive whitespace.
            lowercase: If True, convert text to lowercase.
        """
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase

        # Compile regex patterns for efficiency
        if self.remove_urls:
            # Match http(s):// or www. URLs
            self._url_pattern = re.compile(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                r'|www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                re.IGNORECASE
            )

        if self.remove_html:
            # Match HTML tags
            self._html_pattern = re.compile(r'<[^>]+>')

        if self.normalize_whitespace:
            # Match multiple whitespace characters
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

        # Remove URLs
        if self.remove_urls:
            text = self._url_pattern.sub('', text)

        # Remove HTML tags
        if self.remove_html:
            text = self._html_pattern.sub('', text)

        # Normalize whitespace
        if self.normalize_whitespace:
            # Replace multiple spaces/tabs/newlines with single space
            text = self._whitespace_pattern.sub(' ', text)
            # Strip leading/trailing whitespace
            text = text.strip()

        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()

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

    def __repr__(self) -> str:
        """String representation of cleaner configuration."""
        return (
            f"BasicTextCleaner("
            f"remove_urls={self.remove_urls}, "
            f"remove_html={self.remove_html}, "
            f"normalize_whitespace={self.normalize_whitespace}, "
            f"lowercase={self.lowercase})"
        )


def basic_preprocess(text: str) -> str:
    """
    Quick preprocessing function with default settings.

    This is a convenience function for one-off cleaning without
    creating a BasicTextCleaner instance.

    Args:
        text: Input text to clean.

    Returns:
        Cleaned text string.

    Example:
        >>> text = "Check out http://example.com for more! <br> Great product!"
        >>> basic_preprocess(text)
        'Check out for more! Great product!'
    """
    cleaner = BasicTextCleaner(
        remove_urls=True,
        remove_html=True,
        normalize_whitespace=True,
        lowercase=False
    )
    return cleaner.clean(text)


if __name__ == '__main__':
    # Test the cleaner
    test_texts = [
        "Produsul este excelent! Recomand cu caldura. http://example.com",
        "   Foarte   bun   <br>   dar    scump   ",
        "Nu mi-a placut deloc. Calitate proasta. www.site.ro",
        "<p>Perfect pentru <b>bucatarie</b>! https://link.com/produs</p>"
    ]

    cleaner = BasicTextCleaner()
    print("Testing BasicTextCleaner:")
    print("-" * 80)

    for i, text in enumerate(test_texts, 1):
        cleaned = cleaner.clean(text)
        print(f"Original {i}: {text}")
        print(f"Cleaned {i}:  {cleaned}")
        print()
