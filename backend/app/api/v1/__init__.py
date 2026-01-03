from fastapi import APIRouter
from app.api.v1 import auth, reviews, scraping, analytics, monitoring, websocket, products

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(reviews.router, prefix="/reviews", tags=["reviews"])
api_router.include_router(scraping.router, prefix="/scraping", tags=["scraping"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
api_router.include_router(websocket.router, prefix="/ws", tags=["websocket"])
api_router.include_router(products.router, prefix="/products", tags=["products"])
