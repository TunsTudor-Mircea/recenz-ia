"""
XGBoost model for sentiment classification.

A gradient boosting classifier that works well with TF-IDF features.
Much faster and often more accurate than LSTM for this use case.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import logging
from pathlib import Path
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost classifier for sentiment analysis.

    Works excellently with TF-IDF features from LF-MICF + IGWO.

    Attributes:
        hyperparams: Dictionary of hyperparameters.
        n_classes: Number of output classes.
        model: XGBoost classifier instance.
    """

    def __init__(
        self,
        hyperparams: Optional[Dict[str, Any]] = None,
        n_classes: int = 2
    ):
        """
        Initialize XGBoost model.

        Args:
            hyperparams: Dictionary with keys:
                - n_estimators: Number of boosting rounds
                - max_depth: Maximum tree depth
                - learning_rate: Learning rate
                - subsample: Subsample ratio
                - colsample_bytree: Feature subsample ratio
            n_classes: Number of output classes.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        self.n_classes = n_classes
        self.hyperparams = hyperparams or self._default_hyperparams()
        self.model: Optional[xgb.XGBClassifier] = None
        self._build_model()

    def _default_hyperparams(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

    def _build_model(self) -> None:
        """Build the XGBoost classifier."""
        params = {
            'n_estimators': int(self.hyperparams.get('n_estimators', 100)),
            'max_depth': int(self.hyperparams.get('max_depth', 6)),
            'learning_rate': float(self.hyperparams.get('learning_rate', 0.1)),
            'subsample': float(self.hyperparams.get('subsample', 0.8)),
            'colsample_bytree': float(self.hyperparams.get('colsample_bytree', 0.8)),
            'objective': 'binary:logistic' if self.n_classes == 2 else 'multi:softprob',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42,
            'n_jobs': -1  # Use all CPU cores
        }

        if self.n_classes > 2:
            params['num_class'] = self.n_classes

        self.model = xgb.XGBClassifier(**params)
        logger.info(f"Built XGBoost model with hyperparameters: {self.hyperparams}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> 'XGBoostModel':
        """
        Train the XGBoost model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
            verbose: Whether to print progress.

        Returns:
            Self for chaining.
        """
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=verbose
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features to predict.

        Returns:
            Predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features to predict.

        Returns:
            Predicted class probabilities.
        """
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.

        Args:
            X: Features.
            y: True labels.

        Returns:
            Accuracy score.
        """
        return self.model.score(X, y)

    def save(self, filepath: Path) -> None:
        """
        Save model to disk.

        Args:
            filepath: Path to save the model.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'model': self.model,
            'hyperparams': self.hyperparams,
            'n_classes': self.n_classes
        }

        joblib.dump(save_data, filepath)
        logger.info(f"Saved XGBoost model to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'XGBoostModel':
        """
        Load model from disk.

        Args:
            filepath: Path to load the model from.

        Returns:
            Loaded XGBoostModel instance.
        """
        save_data = joblib.load(filepath)

        instance = cls(
            hyperparams=save_data['hyperparams'],
            n_classes=save_data['n_classes']
        )
        instance.model = save_data['model']

        logger.info(f"Loaded XGBoost model from {filepath}")
        return instance
