"""
Review model for storing product reviews.
"""
from datetime import datetime
from sqlalchemy import Column, String, Text, Float, Integer, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
import hashlib

from app.core.database import Base


class Review(Base):
    """Review model."""

    __tablename__ = "reviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Content hash for deduplication
    content_hash = Column(String(64), unique=True, index=True, nullable=False)

    # Review content
    product_name = Column(String(255), nullable=False, index=True)
    review_text = Column(Text, nullable=False)
    rating = Column(Integer, nullable=False)  # 1-5 stars
    review_date = Column(DateTime(timezone=True), nullable=True)  # Original review date from source

    # Sentiment analysis results
    sentiment_label = Column(String(50))  # positive, negative, neutral
    sentiment_score = Column(Float)  # confidence score
    model_used = Column(String(50))  # robert or xgboost

    # Metadata
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="reviews")

    @staticmethod
    def generate_content_hash(product_name: str, review_text: str, review_date: datetime = None) -> str:
        """
        Generate a unique hash for review deduplication.
        Uses product name, review text, and date to identify duplicates.
        """
        # Normalize text: lowercase, strip whitespace
        normalized_product = product_name.lower().strip()
        normalized_review = review_text.lower().strip()

        # Create hash input
        hash_input = f"{normalized_product}|{normalized_review}"

        # Include date if available for better uniqueness
        if review_date:
            hash_input += f"|{review_date.strftime('%Y-%m-%d')}"

        # Generate SHA-256 hash
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
