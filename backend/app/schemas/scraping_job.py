"""
Scraping job schemas for request/response validation.
"""
from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from app.models.scraping_job import JobStatus


class ScrapingJobCreate(BaseModel):
    """Schema for creating a scraping job."""
    url: str = Field(..., description="URL of the product page to scrape")
    site_type: Optional[str] = Field(None, description="Type of site (emag, cel, altex, etc.)")
    model_type: str = Field(default="robert", description="Sentiment analysis model to use (robert or xgboost)")

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format and allowed domains."""
        from app.core.validation import InputSanitizer
        # For now, only allow eMAG
        allowed_domains = ['emag.ro']
        InputSanitizer.validate_url(v, allowed_domains=allowed_domains)
        return v

    @field_validator('site_type')
    @classmethod
    def validate_site_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate site type."""
        if v is None:
            return v
        allowed_types = ['emag', 'cel', 'altex']
        if v.lower() not in allowed_types:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid site_type. Allowed values: {', '.join(allowed_types)}"
            )
        return v.lower()

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Validate model type."""
        allowed_models = ['robert', 'xgboost']
        if v.lower() not in allowed_models:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model_type. Allowed values: {', '.join(allowed_models)}"
            )
        return v.lower()


class ScrapingJobResponse(BaseModel):
    """Schema for scraping job response."""
    id: UUID
    user_id: UUID
    url: str
    site_type: Optional[str] = None
    status: JobStatus
    reviews_scraped: int = 0
    reviews_created: int = 0
    error_message: Optional[str] = None
    job_metadata: Optional[Dict[str, Any]] = None
    celery_task_id: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {
        "from_attributes": True
    }


class ScrapingJobListResponse(BaseModel):
    """Schema for listing scraping jobs."""
    jobs: list[ScrapingJobResponse]
    total: int
    page: int
    page_size: int
