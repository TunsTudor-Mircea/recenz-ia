"""
Review API endpoints.
"""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.v1.auth import get_current_user
from app.models.user import User
from app.models.review import Review
from app.schemas.review import ReviewCreate, ReviewResponse, ReviewUpdate, ReviewListResponse
from app.services.sentiment import sentiment_analyzer

router = APIRouter()


@router.post("/", response_model=ReviewResponse, status_code=status.HTTP_201_CREATED)
def create_review(
    review_data: ReviewCreate,
    model: str = Query("robert", regex="^(robert|xgboost)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new review with sentiment analysis.

    Args:
        review_data: Review data
        model: ML model to use for sentiment analysis (robert or xgboost)
        db: Database session
        current_user: Current authenticated user

    Returns:
        Created review with sentiment analysis results
    """
    # Analyze sentiment
    try:
        sentiment_result = sentiment_analyzer.analyze(
            text=review_data.review_text,
            model=model
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}"
        )

    # Create review
    review = Review(
        user_id=current_user.id,
        product_name=review_data.product_name,
        review_text=review_data.review_text,
        rating=review_data.rating,
        sentiment_label=sentiment_result["sentiment_label"],
        sentiment_score=sentiment_result["sentiment_score"],
        model_used=sentiment_result["model_used"]
    )

    db.add(review)
    db.commit()
    db.refresh(review)

    return review


@router.get("/", response_model=ReviewListResponse)
def list_reviews(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of records to return"),
    product_name: Optional[str] = Query(None, description="Filter by product name (partial match)"),
    sentiment_label: Optional[str] = Query(None, regex="^(positive|neutral|negative)$", description="Filter by sentiment"),
    min_rating: Optional[int] = Query(None, ge=1, le=5, description="Minimum rating (1-5)"),
    max_rating: Optional[int] = Query(None, ge=1, le=5, description="Maximum rating (1-5)"),
    search: Optional[str] = Query(None, min_length=3, description="Search in review text"),
    sort_by: str = Query("created_at", regex="^(created_at|rating|sentiment_score|review_date)$", description="Sort field"),
    order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List reviews with advanced filtering and search.

    Args:
        skip: Number of reviews to skip (pagination)
        limit: Maximum number of reviews to return (1-100)
        product_name: Filter by product name (case-insensitive partial match)
        sentiment_label: Filter by sentiment (positive, neutral, negative)
        min_rating: Minimum rating filter (1-5 stars)
        max_rating: Maximum rating filter (1-5 stars)
        search: Full-text search in review text (minimum 3 characters)
        sort_by: Sort by field (created_at, rating, sentiment_score, review_date)
        order: Sort order (asc or desc)
        db: Database session
        current_user: Current authenticated user

    Returns:
        Paginated list of reviews with filtering applied
    """
    # Build query
    query = db.query(Review).filter(Review.user_id == current_user.id)

    # Apply filters
    if product_name:
        query = query.filter(Review.product_name.ilike(f"%{product_name}%"))

    if sentiment_label:
        query = query.filter(Review.sentiment_label == sentiment_label)

    if min_rating:
        query = query.filter(Review.rating >= min_rating)

    if max_rating:
        query = query.filter(Review.rating <= max_rating)

    if search:
        # Sanitize search query to prevent SQL injection
        from app.core.validation import InputSanitizer
        sanitized_search = InputSanitizer.sanitize_search_query(search)
        # Full-text search in review text
        query = query.filter(Review.review_text.ilike(f"%{sanitized_search}%"))

    # Get total count before pagination
    total = query.count()

    # Apply sorting
    sort_column = getattr(Review, sort_by)
    if order == "desc":
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())

    # Apply pagination
    reviews = query.offset(skip).limit(limit).all()

    return ReviewListResponse(
        reviews=reviews,
        total=total,
        page=skip // limit + 1 if limit > 0 else 1,
        page_size=limit
    )


@router.get("/{review_id}", response_model=ReviewResponse)
def get_review(
    review_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific review by ID.

    Args:
        review_id: Review ID
        db: Database session
        current_user: Current authenticated user

    Returns:
        Review details
    """
    review = db.query(Review).filter(
        Review.id == review_id,
        Review.user_id == current_user.id
    ).first()

    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found"
        )

    return review


@router.put("/{review_id}", response_model=ReviewResponse)
def update_review(
    review_id: UUID,
    review_data: ReviewUpdate,
    model: str = Query("robert", regex="^(robert|xgboost)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update a review and re-analyze sentiment.

    Args:
        review_id: Review ID
        review_data: Updated review data
        model: ML model to use for sentiment analysis
        db: Database session
        current_user: Current authenticated user

    Returns:
        Updated review
    """
    review = db.query(Review).filter(
        Review.id == review_id,
        Review.user_id == current_user.id
    ).first()

    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found"
        )

    # Update review fields
    update_data = review_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(review, field, value)

    # Re-analyze sentiment if review text was updated
    if review_data.review_text:
        try:
            sentiment_result = sentiment_analyzer.analyze(
                text=review.review_text,
                model=model
            )
            review.sentiment_label = sentiment_result["sentiment_label"]
            review.sentiment_score = sentiment_result["sentiment_score"]
            review.model_used = sentiment_result["model_used"]
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Sentiment analysis failed: {str(e)}"
            )

    db.commit()
    db.refresh(review)

    return review


@router.delete("/{review_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_review(
    review_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a review.

    Args:
        review_id: Review ID
        db: Database session
        current_user: Current authenticated user
    """
    review = db.query(Review).filter(
        Review.id == review_id,
        Review.user_id == current_user.id
    ).first()

    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found"
        )

    db.delete(review)
    db.commit()
