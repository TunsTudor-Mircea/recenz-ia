"""
Redis pub/sub for broadcasting job updates from Celery to WebSocket clients.

This module provides synchronous Redis pub/sub for Celery workers (sync)
to communicate with WebSocket endpoints (async).
"""
import redis
import json
from app.config import settings
from loguru import logger


def get_sync_redis_client() -> redis.Redis:
    """
    Get a new synchronous Redis client for Celery workers.

    Returns:
        Synchronous Redis client
    """
    return redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True,
        health_check_interval=30
    )


def publish_job_update(
    job_id: str,
    status: str,
    reviews_scraped: int = 0,
    reviews_created: int = 0,
    error_message: str | None = None,
    timestamp: str | None = None
) -> bool:
    """
    Publish a job update to Redis pub/sub channel (synchronous for Celery).

    This allows Celery workers to notify WebSocket clients of job progress.

    Args:
        job_id: The job ID
        status: Job status (pending, in_progress, completed, failed)
        reviews_scraped: Number of reviews scraped
        reviews_created: Number of reviews created
        error_message: Error message if failed
        timestamp: ISO timestamp

    Returns:
        True if published successfully, False otherwise
    """
    redis_client = None
    try:
        redis_client = get_sync_redis_client()

        channel = f"job_updates:{job_id}"
        message = {
            "job_id": job_id,
            "status": status,
            "reviews_scraped": reviews_scraped,
            "reviews_created": reviews_created,
            "error_message": error_message,
            "timestamp": timestamp
        }

        subscribers = redis_client.publish(channel, json.dumps(message))
        logger.debug(
            f"Published update to {channel}: {status} "
            f"({subscribers} subscribers)"
        )

        return True

    except Exception as e:
        logger.error(f"Error publishing to Redis for job {job_id}: {e}")
        return False

    finally:
        if redis_client:
            try:
                redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis client: {e}")
