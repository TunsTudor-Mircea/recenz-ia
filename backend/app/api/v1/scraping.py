"""
Scraping API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID

from app.core.database import get_db
from app.models.user import User
from app.models.scraping_job import ScrapingJob, JobStatus
from app.schemas.scraping_job import (
    ScrapingJobCreate,
    ScrapingJobResponse,
    ScrapingJobListResponse
)
from app.api.v1.auth import get_current_user
from app.tasks.scraping import scrape_product_reviews

router = APIRouter()


@router.post("/", response_model=ScrapingJobResponse, status_code=status.HTTP_201_CREATED)
def create_scraping_job(
    job_data: ScrapingJobCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new scraping job to extract reviews from a product page.

    This endpoint creates a background job that will:
    1. Scrape reviews from the specified URL
    2. Analyze sentiment for each review using RoBERT
    3. Save reviews to the database
    """
    # Create scraping job
    job = ScrapingJob(
        user_id=current_user.id,
        url=job_data.url,
        site_type=job_data.site_type,
        status=JobStatus.PENDING
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    # Start background task
    scrape_product_reviews.delay(
        job_id=str(job.id),
        url=job_data.url,
        site_type=job_data.site_type,
        user_id=str(current_user.id),
        model_type=job_data.model_type
    )

    return job


@router.get("/", response_model=ScrapingJobListResponse)
def list_scraping_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    status_filter: Optional[JobStatus] = Query(None, alias="status"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List scraping jobs for the current user with pagination and filtering.

    Query parameters:
    - skip: Number of jobs to skip (for pagination)
    - limit: Maximum number of jobs to return
    - status: Filter by job status (pending, in_progress, completed, failed, cancelled)
    """
    query = db.query(ScrapingJob).filter(ScrapingJob.user_id == current_user.id)

    # Apply status filter
    if status_filter:
        query = query.filter(ScrapingJob.status == status_filter)

    # Get total count
    total = query.count()

    # Get paginated results
    jobs = query.order_by(ScrapingJob.created_at.desc()).offset(skip).limit(limit).all()

    return ScrapingJobListResponse(
        jobs=jobs,
        total=total,
        page=skip // limit + 1,
        page_size=limit
    )


@router.get("/{job_id}", response_model=ScrapingJobResponse)
def get_scraping_job(
    job_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get details of a specific scraping job.
    """
    job = db.query(ScrapingJob).filter(
        ScrapingJob.id == job_id,
        ScrapingJob.user_id == current_user.id
    ).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scraping job not found"
        )

    return job


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def cancel_scraping_job(
    job_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Cancel a pending or in-progress scraping job.
    """
    job = db.query(ScrapingJob).filter(
        ScrapingJob.id == job_id,
        ScrapingJob.user_id == current_user.id
    ).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scraping job not found"
        )

    # Only allow cancellation of pending or in-progress jobs
    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job.status}"
        )

    # Update status to cancelled
    job.status = JobStatus.CANCELLED
    db.commit()

    # TODO: Also revoke the Celery task if it's still running
    # if job.celery_task_id:
    #     celery_app.control.revoke(job.celery_task_id, terminate=True)
