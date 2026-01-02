"""
Analytics service for review data aggregation and insights.
"""
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy import func, desc
from sqlalchemy.orm import Session
from loguru import logger

from app.models.review import Review
from app.schemas.analytics import (
    ProductAnalytics,
    SentimentDistribution,
    RatingDistribution,
    AnalyticsSummary,
    ProductTrend,
    TrendDataPoint,
    TopReview
)


class AnalyticsService:
    """Service for generating review analytics and insights."""

    @staticmethod
    def get_product_analytics(db: Session, product_name: str) -> Optional[ProductAnalytics]:
        """
        Get comprehensive analytics for a specific product.

        Args:
            db: Database session
            product_name: Name of the product

        Returns:
            ProductAnalytics object or None if product not found
        """
        # Get all reviews for the product
        reviews = db.query(Review).filter(Review.product_name == product_name).all()

        if not reviews:
            return None

        # Calculate sentiment distribution
        sentiment_counts = db.query(
            Review.sentiment_label,
            func.count(Review.id)
        ).filter(
            Review.product_name == product_name
        ).group_by(Review.sentiment_label).all()

        sentiment_dict = {label: count for label, count in sentiment_counts}
        sentiment_distribution = SentimentDistribution(
            positive=sentiment_dict.get('positive', 0),
            neutral=sentiment_dict.get('neutral', 0),
            negative=sentiment_dict.get('negative', 0),
            total=len(reviews)
        )

        # Calculate rating distribution
        rating_counts = db.query(
            Review.rating,
            func.count(Review.id)
        ).filter(
            Review.product_name == product_name
        ).group_by(Review.rating).all()

        rating_dict = {rating: count for rating, count in rating_counts}
        avg_rating = db.query(func.avg(Review.rating)).filter(
            Review.product_name == product_name
        ).scalar() or 0.0

        rating_distribution = RatingDistribution(
            rating_1=rating_dict.get(1, 0),
            rating_2=rating_dict.get(2, 0),
            rating_3=rating_dict.get(3, 0),
            rating_4=rating_dict.get(4, 0),
            rating_5=rating_dict.get(5, 0),
            average_rating=float(avg_rating),
            total=len(reviews)
        )

        # Calculate average sentiment score
        avg_sentiment = db.query(func.avg(Review.sentiment_score)).filter(
            Review.product_name == product_name
        ).scalar() or 0.0

        # Get date range
        date_stats = db.query(
            func.min(Review.review_date),
            func.max(Review.review_date)
        ).filter(
            Review.product_name == product_name
        ).first()

        return ProductAnalytics(
            product_name=product_name,
            total_reviews=len(reviews),
            average_rating=float(avg_rating),
            average_sentiment_score=float(avg_sentiment),
            sentiment_distribution=sentiment_distribution,
            rating_distribution=rating_distribution,
            oldest_review_date=date_stats[0] if date_stats else None,
            newest_review_date=date_stats[1] if date_stats else None
        )

    @staticmethod
    def get_overall_summary(db: Session) -> AnalyticsSummary:
        """
        Get overall analytics summary across all products.

        Args:
            db: Database session

        Returns:
            AnalyticsSummary object
        """
        # Total reviews and products
        total_reviews = db.query(func.count(Review.id)).scalar() or 0
        total_products = db.query(func.count(func.distinct(Review.product_name))).scalar() or 0

        # Average rating across all reviews
        avg_rating = db.query(func.avg(Review.rating)).scalar() or 0.0

        # Sentiment distribution
        sentiment_counts = db.query(
            Review.sentiment_label,
            func.count(Review.id)
        ).group_by(Review.sentiment_label).all()

        sentiment_dict = {label: count for label, count in sentiment_counts}
        sentiment_distribution = SentimentDistribution(
            positive=sentiment_dict.get('positive', 0),
            neutral=sentiment_dict.get('neutral', 0),
            negative=sentiment_dict.get('negative', 0),
            total=total_reviews
        )

        # Top rated products (by average rating)
        top_rated = db.query(
            Review.product_name,
            func.avg(Review.rating).label('avg_rating')
        ).group_by(
            Review.product_name
        ).order_by(
            desc('avg_rating')
        ).limit(5).all()

        top_rated_products = [product[0] for product in top_rated]

        # Most reviewed products (by count)
        most_reviewed = db.query(
            Review.product_name,
            func.count(Review.id).label('review_count')
        ).group_by(
            Review.product_name
        ).order_by(
            desc('review_count')
        ).limit(5).all()

        most_reviewed_products = [product[0] for product in most_reviewed]

        return AnalyticsSummary(
            total_reviews=total_reviews,
            total_products=total_products,
            average_rating=float(avg_rating),
            sentiment_distribution=sentiment_distribution,
            top_rated_products=top_rated_products,
            most_reviewed_products=most_reviewed_products
        )

    @staticmethod
    def get_product_trend(
        db: Session,
        product_name: str,
        period: str = 'day',
        days: int = 30
    ) -> Optional[ProductTrend]:
        """
        Get trend data for a product over time.

        Args:
            db: Database session
            product_name: Name of the product
            period: Grouping period ('day', 'week', 'month')
            days: Number of days to include in the trend

        Returns:
            ProductTrend object or None if product not found
        """
        # Check if product exists
        product_exists = db.query(Review).filter(Review.product_name == product_name).first()
        if not product_exists:
            return None

        # Calculate start date
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Query reviews within date range
        reviews = db.query(Review).filter(
            Review.product_name == product_name,
            Review.review_date >= start_date,
            Review.review_date <= end_date
        ).all()

        # Group reviews by period
        data_points = []
        if period == 'day':
            # Group by day
            for i in range(days + 1):
                current_date = start_date + timedelta(days=i)
                day_reviews = [r for r in reviews if r.review_date and r.review_date.date() == current_date.date()]

                if day_reviews:
                    avg_rating = sum(r.rating for r in day_reviews) / len(day_reviews)
                    avg_sentiment = sum(r.sentiment_score for r in day_reviews) / len(day_reviews)

                    data_points.append(TrendDataPoint(
                        date=current_date,
                        count=len(day_reviews),
                        average_rating=avg_rating,
                        average_sentiment=avg_sentiment
                    ))

        return ProductTrend(
            product_name=product_name,
            data_points=data_points,
            period=period
        )

    @staticmethod
    def get_top_reviews(
        db: Session,
        limit: int = 10,
        sentiment: Optional[str] = None,
        min_rating: Optional[int] = None
    ) -> List[TopReview]:
        """
        Get top reviews based on criteria.

        Args:
            db: Database session
            limit: Number of reviews to return
            sentiment: Filter by sentiment label
            min_rating: Minimum rating filter

        Returns:
            List of TopReview objects
        """
        query = db.query(Review)

        # Apply filters
        if sentiment:
            query = query.filter(Review.sentiment_label == sentiment)
        if min_rating:
            query = query.filter(Review.rating >= min_rating)

        # Order by sentiment score (high confidence) and rating
        query = query.order_by(
            desc(Review.sentiment_score),
            desc(Review.rating)
        ).limit(limit)

        reviews = query.all()

        return [
            TopReview(
                id=str(review.id),
                product_name=review.product_name,
                review_text=review.review_text,
                rating=review.rating,
                sentiment_label=review.sentiment_label,
                sentiment_score=review.sentiment_score,
                review_date=review.review_date,
                created_at=review.created_at
            )
            for review in reviews
        ]


# Create singleton instance
analytics_service = AnalyticsService()
