"""
Feature extraction and selection module.

This module provides LF-MICF feature extraction and IGWO feature selection
for sentiment analysis.
"""

from features.lf_micf import LF_MICF
from features.igwo import IGWO
from features.selector import FeatureSelector

__all__ = [
    'LF_MICF',
    'IGWO',
    'FeatureSelector',
]
