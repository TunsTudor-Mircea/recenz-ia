"""
Input validation and sanitization utilities.
"""
import re
import html
from typing import Optional
from fastapi import HTTPException, status


class InputSanitizer:
    """Utilities for sanitizing and validating user inputs."""

    # Dangerous patterns that could indicate XSS or injection attacks
    DANGEROUS_PATTERNS = [
        r'<script[\s\S]*?>[\s\S]*?</script>',  # Script tags
        r'javascript:',  # JavaScript protocol
        r'on\w+\s*=',  # Event handlers (onclick, onerror, etc)
        r'<iframe',  # Iframes
        r'<object',  # Objects
        r'<embed',  # Embeds
        r'eval\(',  # Eval function
    ]

    # SQL injection patterns
    SQL_PATTERNS = [
        r'(\bUNION\b.*\bSELECT\b)',
        r'(\bDROP\b.*\bTABLE\b)',
        r'(\bINSERT\b.*\bINTO\b)',
        r'(\bDELETE\b.*\bFROM\b)',
        r'(;.*--)',
        r'(\'.*OR.*\'.*=.*\')',
    ]

    @staticmethod
    def sanitize_html(text: str) -> str:
        """
        Sanitize HTML content by escaping special characters.

        Args:
            text: Input text that may contain HTML

        Returns:
            HTML-escaped text
        """
        if not text:
            return text
        return html.escape(text)

    @staticmethod
    def validate_no_xss(text: str, field_name: str = "input") -> None:
        """
        Validate that text doesn't contain XSS patterns.

        Args:
            text: Text to validate
            field_name: Name of the field for error messages

        Raises:
            HTTPException: If dangerous patterns are detected
        """
        if not text:
            return

        for pattern in InputSanitizer.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid {field_name}: potentially dangerous content detected"
                )

    @staticmethod
    def validate_no_sql_injection(text: str, field_name: str = "input") -> None:
        """
        Validate that text doesn't contain SQL injection patterns.

        Args:
            text: Text to validate
            field_name: Name of the field for error messages

        Raises:
            HTTPException: If SQL injection patterns are detected
        """
        if not text:
            return

        for pattern in InputSanitizer.SQL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid {field_name}: potentially dangerous SQL pattern detected"
                )

    @staticmethod
    def validate_url(url: str, allowed_domains: Optional[list] = None) -> None:
        """
        Validate URL format and optionally check domain whitelist.

        Args:
            url: URL to validate
            allowed_domains: Optional list of allowed domains

        Raises:
            HTTPException: If URL is invalid or not in allowed domains
        """
        if not url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URL cannot be empty"
            )

        # Basic URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )

        if not url_pattern.match(url):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid URL format"
            )

        # Check domain whitelist if provided
        if allowed_domains:
            domain_found = False
            for domain in allowed_domains:
                if domain in url.lower():
                    domain_found = True
                    break

            if not domain_found:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"URL domain not allowed. Allowed domains: {', '.join(allowed_domains)}"
                )

    @staticmethod
    def sanitize_search_query(query: str, max_length: int = 200) -> str:
        """
        Sanitize search query by removing special characters and limiting length.

        Args:
            query: Search query
            max_length: Maximum allowed length

        Returns:
            Sanitized query
        """
        if not query:
            return query

        # Remove special SQL characters
        query = query.replace('%', '').replace('_', '').replace('\\', '')

        # Limit length
        query = query[:max_length]

        # Remove excessive whitespace
        query = ' '.join(query.split())

        return query

    @staticmethod
    def validate_rating(rating: int) -> None:
        """
        Validate rating is in valid range.

        Args:
            rating: Rating value to validate

        Raises:
            HTTPException: If rating is out of range
        """
        if not 1 <= rating <= 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Rating must be between 1 and 5"
            )

    @staticmethod
    def validate_text_length(
        text: str,
        field_name: str,
        min_length: int = 0,
        max_length: int = 10000
    ) -> None:
        """
        Validate text length is within acceptable range.

        Args:
            text: Text to validate
            field_name: Name of the field for error messages
            min_length: Minimum allowed length
            max_length: Maximum allowed length

        Raises:
            HTTPException: If text length is out of range
        """
        if not text:
            if min_length > 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"{field_name} cannot be empty"
                )
            return

        text_len = len(text)

        if text_len < min_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must be at least {min_length} characters"
            )

        if text_len > max_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must not exceed {max_length} characters"
            )

    @staticmethod
    def validate_and_sanitize_review_text(text: str) -> str:
        """
        Comprehensive validation and sanitization for review text.

        Args:
            text: Review text

        Returns:
            Sanitized review text

        Raises:
            HTTPException: If validation fails
        """
        # Validate length
        InputSanitizer.validate_text_length(
            text,
            "Review text",
            min_length=10,
            max_length=5000
        )

        # Check for XSS patterns
        InputSanitizer.validate_no_xss(text, "Review text")

        # Check for SQL injection patterns
        InputSanitizer.validate_no_sql_injection(text, "Review text")

        # Sanitize HTML
        sanitized = InputSanitizer.sanitize_html(text)

        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())

        return sanitized

    @staticmethod
    def validate_and_sanitize_product_name(name: str) -> str:
        """
        Comprehensive validation and sanitization for product names.

        Args:
            name: Product name

        Returns:
            Sanitized product name

        Raises:
            HTTPException: If validation fails
        """
        # Validate length
        InputSanitizer.validate_text_length(
            name,
            "Product name",
            min_length=2,
            max_length=500
        )

        # Check for XSS patterns
        InputSanitizer.validate_no_xss(name, "Product name")

        # Check for SQL injection patterns
        InputSanitizer.validate_no_sql_injection(name, "Product name")

        # Sanitize HTML
        sanitized = InputSanitizer.sanitize_html(name)

        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())

        return sanitized
