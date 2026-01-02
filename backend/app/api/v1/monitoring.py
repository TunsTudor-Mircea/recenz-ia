"""
Health check and monitoring API endpoints.
"""
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.orm import Session
from celery.result import AsyncResult

from app.core.database import get_db
from app.tasks.celery_app import celery_app
from app.config import settings

router = APIRouter()


@router.get("/health")
def health_check():
    """
    Basic health check endpoint for container orchestration.

    Returns:
        Simple status message
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health/detailed")
def detailed_health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Detailed health check with dependency status.

    Checks:
    - API status
    - Database connectivity
    - Celery worker status
    - Redis connectivity (via Celery)

    Returns:
        Detailed health status of all components
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": {}
    }

    # Check database
    try:
        db.execute(text("SELECT 1"))
        health_status["components"]["database"] = {
            "status": "healthy",
            "message": "Database connection successful"
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}"
        }

    # Check Celery/Redis
    try:
        # Ping Celery workers
        inspect = celery_app.control.inspect()
        active_workers = inspect.active()

        if active_workers:
            worker_count = len(active_workers)
            health_status["components"]["celery"] = {
                "status": "healthy",
                "message": f"{worker_count} worker(s) active",
                "workers": list(active_workers.keys())
            }
        else:
            health_status["status"] = "degraded"
            health_status["components"]["celery"] = {
                "status": "degraded",
                "message": "No active workers found"
            }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["celery"] = {
            "status": "unhealthy",
            "message": f"Celery check failed: {str(e)}"
        }

    # Check Redis (broker)
    try:
        # Test Redis connection via Celery ping
        celery_app.broker_connection().ensure_connection(max_retries=3)
        health_status["components"]["redis"] = {
            "status": "healthy",
            "message": "Redis broker connection successful"
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["components"]["redis"] = {
            "status": "unhealthy",
            "message": f"Redis connection failed: {str(e)}"
        }

    return health_status


@router.get("/metrics")
def get_metrics(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get application metrics for monitoring.

    Returns:
        Application metrics including database statistics
    """
    from app.models.user import User
    from app.models.review import Review
    from app.models.scraping_job import ScrapingJob, JobStatus
    from sqlalchemy import func

    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "database": {},
        "celery": {}
    }

    try:
        # User metrics
        total_users = db.query(func.count(User.id)).scalar() or 0
        active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar() or 0

        metrics["database"]["users"] = {
            "total": total_users,
            "active": active_users
        }

        # Review metrics
        total_reviews = db.query(func.count(Review.id)).scalar() or 0
        avg_rating = db.query(func.avg(Review.rating)).scalar() or 0.0
        avg_sentiment = db.query(func.avg(Review.sentiment_score)).scalar() or 0.0

        # Sentiment distribution
        sentiment_counts = db.query(
            Review.sentiment_label,
            func.count(Review.id)
        ).group_by(Review.sentiment_label).all()

        sentiment_dist = {label: count for label, count in sentiment_counts}

        metrics["database"]["reviews"] = {
            "total": total_reviews,
            "average_rating": float(avg_rating),
            "average_sentiment_score": float(avg_sentiment),
            "sentiment_distribution": sentiment_dist
        }

        # Scraping job metrics
        job_status_counts = db.query(
            ScrapingJob.status,
            func.count(ScrapingJob.id)
        ).group_by(ScrapingJob.status).all()

        job_stats = {status.value: count for status, count in job_status_counts}

        metrics["database"]["scraping_jobs"] = {
            "total": db.query(func.count(ScrapingJob.id)).scalar() or 0,
            "by_status": job_stats
        }

    except Exception as e:
        metrics["database"]["error"] = str(e)

    # Celery metrics
    try:
        inspect = celery_app.control.inspect()

        # Active tasks
        active_tasks = inspect.active()
        active_count = sum(len(tasks) for tasks in active_tasks.values()) if active_tasks else 0

        # Scheduled tasks
        scheduled_tasks = inspect.scheduled()
        scheduled_count = sum(len(tasks) for tasks in scheduled_tasks.values()) if scheduled_tasks else 0

        # Reserved tasks
        reserved_tasks = inspect.reserved()
        reserved_count = sum(len(tasks) for tasks in reserved_tasks.values()) if reserved_tasks else 0

        metrics["celery"] = {
            "active_tasks": active_count,
            "scheduled_tasks": scheduled_count,
            "reserved_tasks": reserved_count,
            "total_pending": active_count + scheduled_count + reserved_count
        }

        # Worker stats
        stats = inspect.stats()
        if stats:
            metrics["celery"]["workers"] = {
                worker: {
                    "pool": worker_stats.get("pool", {}).get("max-concurrency", "unknown")
                }
                for worker, worker_stats in stats.items()
            }

    except Exception as e:
        metrics["celery"]["error"] = str(e)

    return metrics


@router.get("/readiness")
def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness probe for Kubernetes/container orchestration.

    Checks if the application is ready to serve traffic.

    Returns:
        200 if ready, 503 if not ready
    """
    try:
        # Check database
        db.execute(text("SELECT 1"))

        # Check if critical services are initialized
        # (ML models are loaded lazily, so we don't check them here)

        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )


@router.get("/liveness")
def liveness_check():
    """
    Liveness probe for Kubernetes/container orchestration.

    Checks if the application is alive and running.

    Returns:
        200 if alive
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }
