from fastapi import APIRouter
from app.api.v1 import auth, reviews

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(reviews.router, prefix="/reviews", tags=["reviews"])
