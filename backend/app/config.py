"""
Application configuration using Pydantic Settings.
"""
from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    APP_NAME: str = "RecenzIA"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str

    # Redis
    REDIS_URL: str

    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days (7 * 24 * 60)

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    # ML Models — binary sentiment (legacy)
    ROBERT_MODEL_PATH: str
    XGBOOST_MODEL_PATH: str
    XGBOOST_PREPROCESSOR_PATH: str = ""
    XGBOOST_SELECTOR_PATH: str = ""
    SVM_MODEL_PATH: str = ""
    SVM_PREPROCESSOR_PATH: str = ""
    SVM_VECTORIZER_PATH: str = ""
    LR_MODEL_PATH: str = ""
    LR_PREPROCESSOR_PATH: str = ""
    LR_VECTORIZER_PATH: str = ""

    # ABSA models — aspect-based sentiment analysis
    # ABSA_CHECKPOINT_DIR should point to absa/models/checkpoints/ (the directory
    # that contains bert-base-romanian-cased-v1/, xlm-roberta-base/, and
    # bert-base-multilingual-cased/ subdirectories, each with a best_model/ sub-dir).
    ABSA_CHECKPOINT_DIR: str = ""
    # ABSA_BASELINES_DIR should point to absa/models/baselines/ (contains
    # lr_models.pkl and svm_models.pkl produced by train_baselines.py).
    ABSA_BASELINES_DIR: str = ""
    # Which ABSA transformer to load by default when ABSA_DEFAULT_MODEL is not
    # specified per-request.  Must be one of: xlmr, robert, mbert.
    ABSA_DEFAULT_TRANSFORMER: str = "xlmr"

    # Celery
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
