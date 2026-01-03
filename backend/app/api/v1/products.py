"""
Products API endpoints for aggregated product data.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, asc
from loguru import logger

from app.core.database import get_db
from app.api.v1.auth import get_current_user
from app.models.user import User
from app.models.review import Review
from app.schemas.analytics import ProductSummary, PaginatedProducts

router = APIRouter()


@router.get("/", response_model=PaginatedProducts)
def get_products(
    skip: int = Query(default=0, ge=0, description="Number of products to skip"),
    limit: int = Query(default=6, ge=1, le=100, description="Number of products to return"),
    sort_by: str = Query(
        default="updated_at",
        regex="^(name|rating|reviews|updated_at)$",
        description="Field to sort by (name, rating, reviews, updated_at)"
    ),
    sort_order: str = Query(
        default="desc",
        regex="^(asc|desc)$",
        description="Sort order (asc, desc)"
    ),
    search: Optional[str] = Query(default=None, description="Search query for product names"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get paginated list of products with aggregated statistics.

    Args:
        skip: Number of products to skip (for pagination)
        limit: Number of products to return per page
        sort_by: Field to sort by (name, rating, reviews, updated_at)
        sort_order: Sort order (asc or desc)
        search: Optional search query to filter product names

    Returns:
        Paginated list of products with total count
    """
    # Base query - group reviews by product_name for the current user
    from sqlalchemy import case

    query = db.query(
        Review.product_name.label('name'),
        func.count(Review.id).label('total_reviews'),
        func.avg(Review.rating).label('average_rating'),
        func.max(Review.created_at).label('last_updated'),  # Use created_at instead of review_date
        func.sum(case((Review.sentiment_label == 'positive', 1), else_=0)).label('positive'),
        func.sum(case((Review.sentiment_label == 'neutral', 1), else_=0)).label('neutral'),
        func.sum(case((Review.sentiment_label == 'negative', 1), else_=0)).label('negative'),
    ).filter(
        Review.user_id == current_user.id
    ).group_by(
        Review.product_name
    )

    # Apply search filter if provided
    if search:
        query = query.filter(Review.product_name.ilike(f"%{search}%"))

    # Get total count before pagination
    total = query.count()

    # Apply sorting
    sort_column_map = {
        'name': Review.product_name,
        'rating': func.avg(Review.rating),
        'reviews': func.count(Review.id),
        'updated_at': func.max(Review.created_at)  # Use created_at instead of review_date
    }

    sort_column = sort_column_map.get(sort_by, func.max(Review.created_at))
    if sort_order == 'desc':
        query = query.order_by(desc(sort_column))
    else:
        query = query.order_by(asc(sort_column))

    # Apply pagination
    query = query.offset(skip).limit(limit)

    # Execute query
    results = query.all()

    # Format results
    products = []
    for row in results:
        products.append(ProductSummary(
            name=row.name,
            total_reviews=row.total_reviews,
            average_rating=float(row.average_rating) if row.average_rating else 0.0,
            sentiment_distribution={
                'positive': row.positive or 0,
                'neutral': row.neutral or 0,
                'negative': row.negative or 0,
                'total': row.total_reviews
            },
            last_updated=row.last_updated.isoformat() if row.last_updated else None
        ))

    return PaginatedProducts(
        products=products,
        total=total,
        skip=skip,
        limit=limit
    )


@router.delete("/{product_name}")
def delete_product(
    product_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete all reviews for a specific product.

    Args:
        product_name: Name of the product to delete

    Returns:
        Success message with count of deleted reviews
    """
    # Check if product exists for this user
    reviews_count = db.query(Review).filter(
        Review.user_id == current_user.id,
        Review.product_name == product_name
    ).count()

    if reviews_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_name}' not found or you don't have permission to delete it"
        )

    # Delete all reviews for this product
    deleted_count = db.query(Review).filter(
        Review.user_id == current_user.id,
        Review.product_name == product_name
    ).delete()

    db.commit()

    logger.info(f"User {current_user.email} deleted product '{product_name}' ({deleted_count} reviews)")

    return {
        "message": f"Product '{product_name}' deleted successfully",
        "deleted_reviews": deleted_count
    }
