"""
Scraping job model for tracking web scraping tasks.
"""
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid
import enum
from app.core.database import Base


class JobStatus(str, enum.Enum):
    """Job status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScrapingJob(Base):
    """Scraping job model."""
    __tablename__ = "scraping_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Job details
    url = Column(String(2048), nullable=False)
    site_type = Column(String(50))  # emag, cel, altex, etc.
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True)

    # Results
    reviews_scraped = Column(Integer, default=0)
    reviews_created = Column(Integer, default=0)
    error_message = Column(Text)
    job_metadata = Column(JSONB)  # Store additional info like product name, etc.

    # Celery task tracking
    celery_task_id = Column(String(255), index=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="scraping_jobs")
