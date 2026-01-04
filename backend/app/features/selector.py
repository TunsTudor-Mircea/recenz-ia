"""
Feature selector module combining LF-MICF and IGWO.

This module provides a high-level interface for feature extraction
and selection.
"""

from typing import Optional, Union, List
from pathlib import Path
import numpy as np
import logging
import joblib

from features.lf_micf import LF_MICF
from features.igwo import IGWO

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Combined feature extraction and selection pipeline.

    Uses LF-MICF for feature extraction followed by IGWO for
    feature selection.

    Attributes:
        extractor: LF_MICF instance.
        selector: IGWO instance.
        feature_names_: Names of selected features.
    """

    def __init__(
        self,
        extractor_params: Optional[dict] = None,
        selector_params: Optional[dict] = None
    ):
        """
        Initialize FeatureSelector.

        Args:
            extractor_params: Parameters for LF_MICF.
            selector_params: Parameters for IGWO.
        """
        extractor_params = extractor_params or {}
        selector_params = selector_params or {}

        self.extractor = LF_MICF(**extractor_params)
        self.selector = IGWO(**selector_params)
        self.feature_names_: Optional[List[str]] = None

    def fit(
        self,
        documents: List[str],
        labels: np.ndarray,
        verbose: bool = True
    ) -> 'FeatureSelector':
        """
        Fit feature extraction and selection.

        Args:
            documents: Preprocessed text documents.
            labels: Class labels.
            verbose: If True, show progress.

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting feature extractor (LF-MICF)")
        self.extractor.fit(documents, labels)

        logger.info("Transforming documents to features")
        X = self.extractor.transform(documents, sparse=False)

        logger.info("Running feature selection (IGWO)")
        self.selector.fit(X, labels, verbose=verbose)

        # Store selected feature names
        all_features = self.extractor.get_feature_names()
        selected_indices = self.selector.get_selected_features()
        self.feature_names_ = [all_features[i] for i in selected_indices]

        logger.info(f"Feature selection complete. {len(self.feature_names_)} features selected.")
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to selected features.

        Args:
            documents: Preprocessed text documents.

        Returns:
            Feature matrix with selected features.
        """
        X = self.extractor.transform(documents, sparse=False)
        return self.selector.transform(X)

    def fit_transform(
        self,
        documents: List[str],
        labels: np.ndarray,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            documents: Preprocessed text documents.
            labels: Class labels.
            verbose: If True, show progress.

        Returns:
            Feature matrix with selected features.
        """
        return self.fit(documents, labels, verbose).transform(documents)

    def get_feature_names(self) -> List[str]:
        """
        Get names of selected features.

        Returns:
            List of selected feature names.
        """
        return self.feature_names_ or []

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save feature selector to disk.

        Args:
            filepath: Path to save the selector.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'extractor': self.extractor,
            'selector': self.selector,
            'feature_names_': self.feature_names_,
        }

        joblib.dump(save_data, filepath)
        logger.info(f"Saved FeatureSelector to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'FeatureSelector':
        """
        Load feature selector from disk.

        Args:
            filepath: Path to load from.

        Returns:
            Loaded FeatureSelector instance.
        """
        save_data = joblib.load(filepath)

        instance = cls()
        instance.extractor = save_data['extractor']
        instance.selector = save_data['selector']
        instance.feature_names_ = save_data['feature_names_']

        logger.info(f"Loaded FeatureSelector from {filepath}")
        return instance
