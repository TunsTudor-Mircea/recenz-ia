"""
Models module for sentiment classification.

This module provides XGBoost classifier and RoBERT (Romanian BERT)
for binary sentiment analysis.
"""

from models.xgboost_model import XGBoostModel

try:
    from models.robert_model import RoBERTModel
    ROBERT_AVAILABLE = True
except ImportError:
    ROBERT_AVAILABLE = False
    RoBERTModel = None

__all__ = [
    'XGBoostModel',
    'RoBERTModel',
]
