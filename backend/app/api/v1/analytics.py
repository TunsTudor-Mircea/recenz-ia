"""
Analytics API endpoints for review insights.
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.v1.auth import get_current_user
from app.models.user import User
from app.schemas.analytics import (
    ProductAnalytics,
    AnalyticsSummary,
    ProductTrend,
    TopReview
)
from app.services.analytics import analytics_service

router = APIRouter()


@router.get("/summary", response_model=AnalyticsSummary)
def get_analytics_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get overall analytics summary across all products.

    Returns:
        Overall analytics including total reviews, sentiment distribution, top products
    """
    return analytics_service.get_overall_summary(db)


@router.get("/products/{product_name}/trend", response_model=ProductTrend)
def get_product_trend(
    product_name: str,
    period: str = Query(default="day", regex="^(day|week|month)$"),
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get trend data for a product over time.

    Args:
        product_name: Name of the product
        period: Grouping period (day, week, month)
        days: Number of days to include in the trend (1-365)

    Returns:
        Trend data showing review counts and sentiment over time
    """
    trend = analytics_service.get_product_trend(db, product_name, period, days, str(current_user.id))

    if not trend:
        raise HTTPException(status_code=404, detail=f"No reviews found for product: {product_name}")

    return trend


@router.get("/products/{product_name:path}", response_model=ProductAnalytics)
def get_product_analytics(
    product_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive analytics for a specific product.

    Args:
        product_name: Name of the product

    Returns:
        Product analytics including ratings, sentiment, and trends
    """
    print(f"[ANALYTICS ENDPOINT] Getting analytics for product: {product_name}, user_id: {current_user.id}")
    analytics = analytics_service.get_product_analytics(db, product_name, str(current_user.id))
    print(f"[ANALYTICS ENDPOINT] Analytics result: {analytics}")

    if not analytics:
        raise HTTPException(status_code=404, detail=f"No reviews found for product: {product_name}")

    return analytics


@router.get("/top-reviews", response_model=List[TopReview])
def get_top_reviews(
    limit: int = Query(default=10, ge=1, le=100),
    sentiment: Optional[str] = Query(default=None, regex="^(positive|neutral|negative)$"),
    min_rating: Optional[int] = Query(default=None, ge=1, le=5),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get top reviews based on criteria.

    Args:
        limit: Number of reviews to return (1-100)
        sentiment: Filter by sentiment label (positive, neutral, negative)
        min_rating: Minimum rating filter (1-5)

    Returns:
        List of top reviews sorted by sentiment score and rating
    """
    return analytics_service.get_top_reviews(db, limit, sentiment, min_rating)
