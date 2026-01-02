"""
Analytics schemas for review aggregation and insights.
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class SentimentDistribution(BaseModel):
    """Sentiment distribution statistics."""

    positive: int = Field(..., description="Count of positive reviews")
    neutral: int = Field(..., description="Count of neutral reviews")
    negative: int = Field(..., description="Count of negative reviews")
    total: int = Field(..., description="Total number of reviews")


class RatingDistribution(BaseModel):
    """Rating distribution statistics."""

    rating_1: int = Field(..., description="Count of 1-star reviews")
    rating_2: int = Field(..., description="Count of 2-star reviews")
    rating_3: int = Field(..., description="Count of 3-star reviews")
    rating_4: int = Field(..., description="Count of 4-star reviews")
    rating_5: int = Field(..., description="Count of 5-star reviews")
    average_rating: float = Field(..., description="Average rating")
    total: int = Field(..., description="Total number of reviews")


class ProductAnalytics(BaseModel):
    """Analytics for a specific product."""

    product_name: str
    total_reviews: int
    average_rating: float
    average_sentiment_score: float
    sentiment_distribution: SentimentDistribution
    rating_distribution: RatingDistribution
    oldest_review_date: Optional[datetime] = None
    newest_review_date: Optional[datetime] = None


class TrendDataPoint(BaseModel):
    """Single data point in a time series."""

    date: datetime
    count: int
    average_rating: Optional[float] = None
    average_sentiment: Optional[float] = None


class ProductTrend(BaseModel):
    """Trend data for a product over time."""

    product_name: str
    data_points: List[TrendDataPoint]
    period: str = Field(..., description="Time period (day, week, month)")


class TopReview(BaseModel):
    """Top review based on criteria."""

    id: str
    product_name: str
    review_text: str
    rating: int
    sentiment_label: str
    sentiment_score: float
    review_date: Optional[datetime]
    created_at: datetime

    model_config = {
        "from_attributes": True
    }


class AnalyticsSummary(BaseModel):
    """Overall analytics summary."""

    total_reviews: int
    total_products: int
    average_rating: float
    sentiment_distribution: SentimentDistribution
    top_rated_products: List[str]
    most_reviewed_products: List[str]
