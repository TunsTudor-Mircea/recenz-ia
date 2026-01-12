"""
Bias Mitigation Techniques

This module implements bias mitigation strategies for machine learning models:

1. Pre-processing: Class Reweighting (simplified Fair-SMOTE approach)
2. Post-processing: Threshold Calibration

References:
- Chakraborty et al. (2021): Fair-SMOTE
- Chen et al. (2024): Fairness mitigation strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_curve
from sklearn.base import BaseEstimator


class FairReweighting:
    """
    Pre-processing mitigation: Compute fair sample weights to balance demographic groups.
    
    This is a simplified version of Fair-SMOTE that uses class weights instead of
    synthetic sample generation for easier integration with existing models.
    """
    
    def __init__(self, protected_attr_name: str = 'protected_attr'):
        """
        Initialize Fair Reweighting.
        
        Args:
            protected_attr_name: Name of the protected attribute column
        """
        self.protected_attr_name = protected_attr_name
        self.weights_ = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'FairReweighting':
        """
        Compute fair sample weights.
        
        Args:
            X: Feature matrix (must include protected attribute column)
            y: Target labels
            
        Returns:
            Self
        """
        if self.protected_attr_name not in X.columns:
            raise ValueError(f"Protected attribute '{self.protected_attr_name}' not found in X")
        
        protected_attr = X[self.protected_attr_name].values
        
        # Create combined grouping: (class_label, protected_attr)
        groups = np.array([f"{label}_{attr}" for label, attr in zip(y, protected_attr)])
        
        # Compute weights to balance all (class, protected_attr) combinations
        self.weights_ = compute_sample_weight('balanced', groups)
        
        # Normalize weights
        self.weights_ = self.weights_ / np.mean(self.weights_)
        
        return self
    
    def get_weights(self) -> np.ndarray:
        """
        Get computed sample weights.
        
        Returns:
            Array of sample weights
        """
        if self.weights_ is None:
            raise ValueError("Must call fit() before get_weights()")
        return self.weights_
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """
        Fit and return sample weights.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Sample weights
        """
        self.fit(X, y)
        return self.weights_


class ThresholdCalibrator:
    """
    Post-processing mitigation: Calibrate decision thresholds per demographic group.
    
    This technique adjusts classification thresholds to equalize true positive rates
    (Equal Opportunity) across demographic groups without retraining.
    """
    
    def __init__(self, target_metric: str = 'equal_opportunity'):
        """
        Initialize Threshold Calibrator.
        
        Args:
            target_metric: Fairness objective ('equal_opportunity' or 'equalized_odds')
        """
        self.target_metric = target_metric
        self.thresholds_ = {}
        self.base_threshold_ = 0.5
        
    def fit(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        protected_attr: np.ndarray
    ) -> 'ThresholdCalibrator':
        """
        Compute optimal thresholds for each demographic group.
        
        Args:
            y_true: True labels
            y_score: Predicted probabilities (or decision scores)
            protected_attr: Protected attribute values
            
        Returns:
            Self
        """
        unique_groups = np.unique(protected_attr)
        
        if self.target_metric == 'equal_opportunity':
            # Equalize TPR across groups
            target_tpr = self._compute_overall_tpr(y_true, y_score)
            
            for group in unique_groups:
                mask = protected_attr == group
                threshold = self._find_threshold_for_tpr(
                    y_true[mask],
                    y_score[mask],
                    target_tpr
                )
                self.thresholds_[group] = threshold
                
        elif self.target_metric == 'equalized_odds':
            # Equalize both TPR and FPR
            target_tpr = self._compute_overall_tpr(y_true, y_score)
            target_fpr = self._compute_overall_fpr(y_true, y_score)
            
            for group in unique_groups:
                mask = protected_attr == group
                # Find threshold balancing both TPR and FPR
                threshold = self._find_threshold_for_tpr_fpr(
                    y_true[mask],
                    y_score[mask],
                    target_tpr,
                    target_fpr
                )
                self.thresholds_[group] = threshold
        else:
            raise ValueError(f"Unknown target_metric: {self.target_metric}")
        
        return self
    
    def predict(self, y_score: np.ndarray, protected_attr: np.ndarray) -> np.ndarray:
        """
        Make predictions using group-specific thresholds.
        
        Args:
            y_score: Predicted probabilities
            protected_attr: Protected attribute values
            
        Returns:
            Binary predictions
        """
        y_pred = np.zeros(len(y_score), dtype=int)
        
        for group, threshold in self.thresholds_.items():
            mask = protected_attr == group
            y_pred[mask] = (y_score[mask] >= threshold).astype(int)
        
        return y_pred
    
    def _compute_overall_tpr(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute overall TPR at default threshold."""
        y_pred = (y_score >= self.base_threshold_).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _compute_overall_fpr(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute overall FPR at default threshold."""
        y_pred = (y_score >= self.base_threshold_).astype(int)
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    def _find_threshold_for_tpr(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        target_tpr: float
    ) -> float:
        """
        Find threshold that achieves target TPR.
        
        Args:
            y_true: True labels for group
            y_score: Predicted scores for group
            target_tpr: Desired TPR
            
        Returns:
            Optimal threshold
        """
        if len(y_true) == 0 or np.sum(y_true == 1) == 0:
            return self.base_threshold_
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # Find threshold closest to target TPR
        idx = np.argmin(np.abs(tpr - target_tpr))
        
        return thresholds[idx] if idx < len(thresholds) else self.base_threshold_
    
    def _find_threshold_for_tpr_fpr(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        target_tpr: float,
        target_fpr: float
    ) -> float:
        """
        Find threshold that balances TPR and FPR targets.
        
        Args:
            y_true: True labels
            y_score: Predicted scores
            target_tpr: Desired TPR
            target_fpr: Desired FPR
            
        Returns:
            Optimal threshold
        """
        if len(y_true) == 0:
            return self.base_threshold_
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # Find threshold minimizing distance to both targets
        distances = np.sqrt((tpr - target_tpr)**2 + (fpr - target_fpr)**2)
        idx = np.argmin(distances)
        
        return thresholds[idx] if idx < len(thresholds) else self.base_threshold_


class FairnessAwareTraining:
    """
    In-processing mitigation: Add fairness penalty to training objective.
    
    This is a wrapper that can be used with scikit-learn estimators to incorporate
    fairness constraints during training.
    """
    
    def __init__(
        self,
        base_estimator: BaseEstimator,
        fairness_penalty: float = 0.1,
        protected_attr_name: str = 'protected_attr'
    ):
        """
        Initialize fairness-aware training wrapper.
        
        Args:
            base_estimator: Scikit-learn compatible estimator
            fairness_penalty: Weight for fairness penalty (lambda)
            protected_attr_name: Name of protected attribute column
        """
        self.base_estimator = base_estimator
        self.fairness_penalty = fairness_penalty
        self.protected_attr_name = protected_attr_name
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'FairnessAwareTraining':
        """
        Train model with fairness penalty.
        
        Note: This is a simplified implementation using sample reweighting.
        For true fairness-aware training, custom loss functions are needed.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Self
        """
        # Use reweighting as proxy for fairness penalty
        reweighter = FairReweighting(self.protected_attr_name)
        sample_weights = reweighter.fit_transform(X, y)
        
        # Apply additional penalty scaling
        sample_weights = sample_weights ** (1 + self.fairness_penalty)
        
        # Remove protected attribute before training (fairness through unawareness)
        X_train = X.drop(columns=[self.protected_attr_name])
        
        # Train with weighted samples
        if hasattr(self.base_estimator, 'fit') and 'sample_weight' in self.base_estimator.fit.__code__.co_varnames:
            self.base_estimator.fit(X_train, y, sample_weight=sample_weights)
        else:
            self.base_estimator.fit(X_train, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_pred = X.drop(columns=[self.protected_attr_name], errors='ignore')
        return self.base_estimator.predict(X_pred)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        X_pred = X.drop(columns=[self.protected_attr_name], errors='ignore')
        return self.base_estimator.predict_proba(X_pred)


def evaluate_mitigation(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_mitigated: np.ndarray,
    protected_attr: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate the effectiveness of bias mitigation.
    
    Args:
        y_true: True labels
        y_pred_baseline: Predictions from baseline model
        y_pred_mitigated: Predictions from mitigated model
        protected_attr: Protected attribute values
        
    Returns:
        Dictionary with comparison metrics
    """
    from fairness_metrics import FairnessMetrics
    from sklearn.metrics import accuracy_score, f1_score
    
    # Baseline metrics
    fm_baseline = FairnessMetrics(y_true, y_pred_baseline, protected_attr)
    baseline_fairness = fm_baseline.compute_all_metrics()
    baseline_perf = {
        'accuracy': accuracy_score(y_true, y_pred_baseline),
        'f1': f1_score(y_true, y_pred_baseline)
    }
    
    # Mitigated metrics
    fm_mitigated = FairnessMetrics(y_true, y_pred_mitigated, protected_attr)
    mitigated_fairness = fm_mitigated.compute_all_metrics()
    mitigated_perf = {
        'accuracy': accuracy_score(y_true, y_pred_mitigated),
        'f1': f1_score(y_true, y_pred_mitigated)
    }
    
    # Compute improvements
    improvements = {}
    for metric in ['spd', 'eod', 'aod']:
        baseline_abs = abs(baseline_fairness[metric])
        mitigated_abs = abs(mitigated_fairness[metric])
        if baseline_abs > 0:
            improvements[f'{metric}_reduction_pct'] = (
                (baseline_abs - mitigated_abs) / baseline_abs * 100
            )
        else:
            improvements[f'{metric}_reduction_pct'] = 0.0
    
    # DI improvement (closer to 1.0)
    baseline_di_dist = abs(baseline_fairness['di'] - 1.0)
    mitigated_di_dist = abs(mitigated_fairness['di'] - 1.0)
    if baseline_di_dist > 0:
        improvements['di_improvement_pct'] = (
            (baseline_di_dist - mitigated_di_dist) / baseline_di_dist * 100
        )
    else:
        improvements['di_improvement_pct'] = 0.0
    
    return {
        'baseline': {
            'fairness': baseline_fairness,
            'performance': baseline_perf
        },
        'mitigated': {
            'fairness': mitigated_fairness,
            'performance': mitigated_perf
        },
        'improvements': improvements
    }


if __name__ == "__main__":
    print("Bias Mitigation Module - Example Usage\n")
    
    # Example: Fair Reweighting
    print("=" * 60)
    print("1. Fair Reweighting (Pre-processing)")
    print("=" * 60)
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic dataset with bias
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'protected_attr': np.random.binomial(1, 0.3, n_samples)  # Imbalanced
    })
    y = np.random.binomial(1, 0.6, n_samples)
    
    reweighter = FairReweighting()
    weights = reweighter.fit_transform(X, y)
    
    print(f"Original group distribution:")
    print(f"  Group 0: {np.sum(X['protected_attr'] == 0)} samples")
    print(f"  Group 1: {np.sum(X['protected_attr'] == 1)} samples")
    print(f"\nAverage weights:")
    print(f"  Group 0: {np.mean(weights[X['protected_attr'] == 0]):.3f}")
    print(f"  Group 1: {np.mean(weights[X['protected_attr'] == 1]):.3f}")
    
    # Example: Threshold Calibration
    print("\n" + "=" * 60)
    print("2. Threshold Calibration (Post-processing)")
    print("=" * 60)
    
    y_true = np.random.binomial(1, 0.6, n_samples)
    y_score = np.random.rand(n_samples)
    protected_attr = np.random.binomial(1, 0.5, n_samples)
    
    calibrator = ThresholdCalibrator(target_metric='equal_opportunity')
    calibrator.fit(y_true, y_score, protected_attr)
    
    print(f"Calibrated thresholds:")
    for group, threshold in calibrator.thresholds_.items():
        print(f"  Group {group}: {threshold:.4f}")
    
    y_pred_calibrated = calibrator.predict(y_score, protected_attr)
    print(f"\nPredictions made with calibrated thresholds: {len(y_pred_calibrated)} samples")
