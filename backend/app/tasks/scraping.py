"""
Celery tasks for web scraping.
"""
from datetime import datetime
from typing import Optional
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.tasks.celery_app import celery_app
from app.core.database import SessionLocal
from app.core.redis_pubsub import publish_job_update
from app.models.scraping_job import ScrapingJob, JobStatus
from app.models.review import Review
from app.services.scraper import scrape_reviews
from app.services.sentiment import sentiment_analyzer


@celery_app.task(
    bind=True,
    name='app.tasks.scraping.scrape_product_reviews',
    max_retries=3,
    default_retry_delay=60,  # Retry after 60 seconds
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,  # Max 10 minutes between retries
    retry_jitter=True  # Add random jitter to prevent thundering herd
)
def scrape_product_reviews(self, job_id: str, url: str, site_type: Optional[str], user_id: str):
    """
    Scrape reviews from a product page and save them to the database.

    Args:
        job_id: UUID of the scraping job
        url: URL of the product page
        site_type: Optional site type (emag, cel, etc.)
        user_id: UUID of the user who created the job
    """
    db: Session = SessionLocal()

    try:
        # Update job status to in_progress
        job = db.query(ScrapingJob).filter(ScrapingJob.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        job.status = JobStatus.IN_PROGRESS
        job.started_at = datetime.utcnow()
        job.celery_task_id = self.request.id
        timestamp_started = job.started_at.isoformat()
        db.commit()

        # Broadcast status update via WebSocket (after commit to avoid race condition)
        try:
            publish_job_update(
                job_id=job_id,
                status="in_progress",
                reviews_scraped=0,
                reviews_created=0,
                timestamp=timestamp_started
            )
        except Exception as e:
            logger.warning(f"Failed to publish job start update: {e}")

        logger.info(f"Starting scraping job {job_id} for URL: {url}")

        # Scrape reviews
        scraped_reviews = scrape_reviews(url, site_type)

        if not scraped_reviews:
            logger.warning(f"No reviews found for {url}")
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.reviews_scraped = 0
            job.reviews_created = 0
            job.job_metadata = {"message": "No reviews found on this page"}
            db.commit()
            return

        # Process and save reviews with sentiment analysis
        reviews_created = 0
        reviews_skipped_duplicates = 0
        product_name = scraped_reviews[0].product_name if scraped_reviews else "Unknown"

        for review_data in scraped_reviews:
            try:
                # Generate content hash for deduplication
                content_hash = Review.generate_content_hash(
                    product_name=review_data.product_name,
                    review_text=review_data.review_text,
                    review_date=review_data.review_date
                )

                # Check if review already exists
                existing_review = db.query(Review).filter(Review.content_hash == content_hash).first()
                if existing_review:
                    logger.info(f"Skipping duplicate review (hash: {content_hash[:16]}...)")
                    reviews_skipped_duplicates += 1
                    continue

                # Analyze sentiment
                sentiment_result = sentiment_analyzer.analyze(
                    text=review_data.review_text,
                    model='robert'  # Use RoBERT by default
                )

                # Create review with content hash
                review = Review(
                    user_id=user_id,
                    product_name=review_data.product_name,
                    review_title=review_data.review_title,
                    review_text=review_data.review_text,
                    rating=review_data.rating,
                    review_date=review_data.review_date,
                    content_hash=content_hash,
                    sentiment_label=sentiment_result['sentiment_label'],
                    sentiment_score=sentiment_result['sentiment_score'],
                    model_used=sentiment_result['model_used']
                )

                db.add(review)
                reviews_created += 1

            except IntegrityError as e:
                # Handle race condition where duplicate was created between check and insert
                logger.warning(f"Duplicate review detected during insert: {e}")
                db.rollback()
                reviews_skipped_duplicates += 1
                continue
            except Exception as e:
                logger.error(f"Error processing review: {e}")
                db.rollback()
                continue

        # Commit all reviews
        try:
            db.commit()
            logger.info(f"Saved {reviews_created} reviews, skipped {reviews_skipped_duplicates} duplicates")
        except Exception as e:
            logger.error(f"Error committing reviews: {e}")
            db.rollback()
            raise

        # Update job status to completed
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.reviews_scraped = len(scraped_reviews)
        job.reviews_created = reviews_created
        job.job_metadata = {
            "product_name": product_name,
            "source": scraped_reviews[0].metadata.get('source', 'unknown') if scraped_reviews else 'unknown',
            "duplicates_skipped": reviews_skipped_duplicates
        }
        timestamp_completed = job.completed_at.isoformat()
        db.commit()

        # Broadcast completion via WebSocket (after commit to ensure consistency)
        try:
            success = publish_job_update(
                job_id=job_id,
                status="completed",
                reviews_scraped=len(scraped_reviews),
                reviews_created=reviews_created,
                timestamp=timestamp_completed
            )
            if not success:
                logger.error(f"Failed to publish completion update for job {job_id}")
        except Exception as e:
            logger.error(f"Exception publishing completion update for job {job_id}: {e}")

        logger.info(f"Completed scraping job {job_id}: {reviews_created}/{len(scraped_reviews)} reviews saved, {reviews_skipped_duplicates} duplicates skipped")

    except Exception as e:
        logger.error(f"Error in scraping job {job_id}: {e}")

        # Update job status to failed
        job = db.query(ScrapingJob).filter(ScrapingJob.id == job_id).first()
        if job:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
            timestamp_failed = job.completed_at.isoformat()
            db.commit()

            # Broadcast failure via WebSocket (after commit)
            try:
                success = publish_job_update(
                    job_id=job_id,
                    status="failed",
                    error_message=str(e),
                    timestamp=timestamp_failed
                )
                if not success:
                    logger.error(f"Failed to publish failure update for job {job_id}")
            except Exception as e:
                logger.error(f"Exception publishing failure update for job {job_id}: {e}")

        raise

    finally:
        db.close()
