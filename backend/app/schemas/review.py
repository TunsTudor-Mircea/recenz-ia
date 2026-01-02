"""
Review schemas for request/response validation.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


class ReviewBase(BaseModel):
    """Base review schema."""

    product_name: str = Field(..., min_length=1, max_length=255)
    review_text: str = Field(..., min_length=1)
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5 stars")


class ReviewCreate(ReviewBase):
    """Schema for creating a review."""

    pass


class ReviewUpdate(BaseModel):
    """Schema for updating a review."""

    product_name: Optional[str] = Field(None, min_length=1, max_length=255)
    review_text: Optional[str] = Field(None, min_length=1)
    rating: Optional[int] = Field(None, ge=1, le=5)


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
