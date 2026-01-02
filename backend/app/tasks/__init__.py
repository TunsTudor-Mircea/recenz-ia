from app.tasks.celery_app import celery_app
from app.tasks.scraping import scrape_product_reviews

__all__ = ['celery_app', 'scrape_product_reviews']
