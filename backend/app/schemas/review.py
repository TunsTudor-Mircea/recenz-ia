"""
Review schemas for request/response validation.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class ReviewBase(BaseModel):
    """Base review schema."""

    product_name: str = Field(..., min_length=2, max_length=500, description="Product name")
    review_text: str = Field(..., min_length=10, max_length=5000, description="Review text content")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5 stars")
    review_date: Optional[datetime] = Field(None, description="Original review date from source")

    @field_validator('product_name')
    @classmethod
    def validate_product_name(cls, v: str) -> str:
        """Validate and sanitize product name."""
        from app.core.validation import InputSanitizer
        return InputSanitizer.validate_and_sanitize_product_name(v)

    @field_validator('review_text')
    @classmethod
    def validate_review_text(cls, v: str) -> str:
        """Validate and sanitize review text."""
        from app.core.validation import InputSanitizer
        return InputSanitizer.validate_and_sanitize_review_text(v)

    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v: int) -> int:
        """Validate rating is in valid range."""
        from app.core.validation import InputSanitizer
        InputSanitizer.validate_rating(v)
        return v


class ReviewCreate(ReviewBase):
    """Schema for creating a review."""

    pass


class ReviewUpdate(BaseModel):
    """Schema for updating a review."""

    product_name: Optional[str] = Field(None, min_length=2, max_length=500, description="Product name")
    review_text: Optional[str] = Field(None, min_length=10, max_length=5000, description="Review text content")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1 to 5 stars")

    @field_validator('product_name')
    @classmethod
    def validate_product_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate and sanitize product name."""
        if v is None:
            return v
        from app.core.validation import InputSanitizer
        return InputSanitizer.validate_and_sanitize_product_name(v)

    @field_validator('review_text')
    @classmethod
    def validate_review_text(cls, v: Optional[str]) -> Optional[str]:
        """Validate and sanitize review text."""
        if v is None:
            return v
        from app.core.validation import InputSanitizer
        return InputSanitizer.validate_and_sanitize_review_text(v)

    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v: Optional[int]) -> Optional[int]:
        """Validate rating is in valid range."""
        if v is None:
            return v
        from app.core.validation import InputSanitizer
        InputSanitizer.validate_rating(v)
        return v


class ReviewResponse(ReviewBase):
    """Schema for review response."""

    id: UUID
    user_id: UUID
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None
    model_used: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ReviewListResponse(BaseModel):
    """Schema for paginated review list."""

    reviews: list[ReviewResponse]
    total: int
    page: int
    page_size: int
