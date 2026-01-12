"""
Fairness Metrics Module

This module provides implementations of common fairness metrics for evaluating
bias in machine learning models. Metrics implemented:

- Statistical Parity Difference (SPD)
- Equal Opportunity Difference (EOD)
- Average Odds Difference (AOD)
- Disparate Impact (DI)

References:
- Chakraborty et al. (2021): Bias in Machine Learning Software
- Chen et al. (2024): Fairness Improvement with Multiple Protected Attributes
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import confusion_matrix


class FairnessMetrics:
    """Calculate fairness metrics for binary classification models."""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, protected_attr: np.ndarray):
        """
        Initialize fairness metrics calculator.
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            protected_attr: Protected attribute values (0 = unprivileged, 1 = privileged)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.protected_attr = np.array(protected_attr)
        
        # Validate inputs
        assert len(self.y_true) == len(self.y_pred) == len(self.protected_attr), \
            "All arrays must have the same length"
        assert set(self.protected_attr).issubset({0, 1}), \
            "Protected attribute must be binary (0 or 1)"
    
    def _get_group_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get boolean masks for unprivileged and privileged groups."""
        unprivileged_mask = self.protected_attr == 0
        privileged_mask = self.protected_attr == 1
        return unprivileged_mask, privileged_mask
    
    def _compute_rates(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Compute prediction rates for a specific group.
        
        Args:
            mask: Boolean mask for the group
            
        Returns:
            Dictionary with TPR, FPR, TNR, FNR, and positive prediction rate
        """
        y_true_group = self.y_true[mask]
        y_pred_group = self.y_pred[mask]
        
        if len(y_true_group) == 0:
            return {
                'tpr': 0.0, 'fpr': 0.0, 'tnr': 0.0, 'fnr': 0.0,
                'positive_rate': 0.0, 'n_samples': 0
            }
        
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
        
        # Calculate rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate (Specificity)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
        positive_rate = np.mean(y_pred_group)  # P(Y_pred = 1)
        
        return {
            'tpr': tpr,
            'fpr': fpr,
            'tnr': tnr,
            'fnr': fnr,
            'positive_rate': positive_rate,
            'n_samples': len(y_true_group),
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }
    
    def statistical_parity_difference(self) -> float:
        """
        Calculate Statistical Parity Difference (SPD).
        
        SPD = P(Y_pred=1 | A=0) - P(Y_pred=1 | A=1)
        
        Interpretation:
        - SPD = 0: Perfect parity (no disparity)
        - SPD > 0: Unprivileged group receives more positive predictions
        - SPD < 0: Privileged group receives more positive predictions
        
        Returns:
            SPD value
        """
        unprivileged_mask, privileged_mask = self._get_group_masks()
        
        rates_unprivileged = self._compute_rates(unprivileged_mask)
        rates_privileged = self._compute_rates(privileged_mask)
        
        spd = rates_unprivileged['positive_rate'] - rates_privileged['positive_rate']
        
        return spd
    
    def equal_opportunity_difference(self) -> float:
        """
        Calculate Equal Opportunity Difference (EOD).
        
        EOD = P(Y_pred=1 | A=0, Y=1) - P(Y_pred=1 | A=1, Y=1)
            = TPR_unprivileged - TPR_privileged
        
        Interpretation:
        - EOD = 0: Equal opportunity (equal TPR)
        - EOD > 0: Unprivileged group has higher TPR
        - EOD < 0: Privileged group has higher TPR (bias favoring privileged)
        
        Returns:
            EOD value
        """
        unprivileged_mask, privileged_mask = self._get_group_masks()
        
        rates_unprivileged = self._compute_rates(unprivileged_mask)
        rates_privileged = self._compute_rates(privileged_mask)
        
        eod = rates_unprivileged['tpr'] - rates_privileged['tpr']
        
        return eod
    
    def average_odds_difference(self) -> float:
        """
        Calculate Average Odds Difference (AOD).
        
        AOD = 0.5 * [(FPR_unprivileged - FPR_privileged) + (TPR_unprivileged - TPR_privileged)]
        
        Interpretation:
        - AOD = 0: Equal odds (equal FPR and TPR)
        - AOD > 0: Unprivileged group has higher average rates
        - AOD < 0: Privileged group has higher average rates
        
        Returns:
            AOD value
        """
        unprivileged_mask, privileged_mask = self._get_group_masks()
        
        rates_unprivileged = self._compute_rates(unprivileged_mask)
        rates_privileged = self._compute_rates(privileged_mask)
        
        fpr_diff = rates_unprivileged['fpr'] - rates_privileged['fpr']
        tpr_diff = rates_unprivileged['tpr'] - rates_privileged['tpr']
        
        aod = 0.5 * (fpr_diff + tpr_diff)
        
        return aod
    
    def disparate_impact(self) -> float:
        """
        Calculate Disparate Impact (DI).
        
        DI = P(Y_pred=1 | A=0) / P(Y_pred=1 | A=1)
        
        Interpretation:
        - DI = 1.0: Perfect parity
        - DI < 0.8: Fails the "80% rule" (significant adverse impact)
        - DI > 1.25: Reverse discrimination threshold
        
        Returns:
            DI value
        """
        unprivileged_mask, privileged_mask = self._get_group_masks()
        
        rates_unprivileged = self._compute_rates(unprivileged_mask)
        rates_privileged = self._compute_rates(privileged_mask)
        
        if rates_privileged['positive_rate'] == 0:
            return float('inf') if rates_unprivileged['positive_rate'] > 0 else 1.0
        
        di = rates_unprivileged['positive_rate'] / rates_privileged['positive_rate']
        
        return di
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all fairness metrics at once.
        
        Returns:
            Dictionary containing all fairness metrics
        """
        return {
            'spd': self.statistical_parity_difference(),
            'eod': self.equal_opportunity_difference(),
            'aod': self.average_odds_difference(),
            'di': self.disparate_impact()
        }
    
    def get_group_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get detailed statistics for each group.
        
        Returns:
            Dictionary with statistics for unprivileged and privileged groups
        """
        unprivileged_mask, privileged_mask = self._get_group_masks()
        
        return {
            'unprivileged': self._compute_rates(unprivileged_mask),
            'privileged': self._compute_rates(privileged_mask)
        }
    
    def fairness_report(self) -> str:
        """
        Generate a comprehensive fairness report.
        
        Returns:
            Formatted string report
        """
        metrics = self.compute_all_metrics()
        group_stats = self.get_group_statistics()
        
        report = []
        report.append("=" * 70)
        report.append("FAIRNESS EVALUATION REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Group statistics
        report.append("GROUP STATISTICS")
        report.append("-" * 70)
        for group_name, stats in group_stats.items():
            report.append(f"\n{group_name.upper()} GROUP (n={stats['n_samples']}):")
            report.append(f"  True Positive Rate (TPR):  {stats['tpr']:.4f}")
            report.append(f"  False Positive Rate (FPR): {stats['fpr']:.4f}")
            report.append(f"  Positive Prediction Rate:  {stats['positive_rate']:.4f}")
            report.append(f"  Confusion Matrix: TP={stats['tp']}, FP={stats['fp']}, TN={stats['tn']}, FN={stats['fn']}")
        
        report.append("\n")
        report.append("FAIRNESS METRICS")
        report.append("-" * 70)
        report.append(f"Statistical Parity Difference (SPD): {metrics['spd']:+.4f}")
        report.append(f"  → Ideal: 0.0 | Current: {'✓ Fair' if abs(metrics['spd']) < 0.1 else '✗ Biased'}")
        report.append("")
        report.append(f"Equal Opportunity Difference (EOD):  {metrics['eod']:+.4f}")
        report.append(f"  → Ideal: 0.0 | Current: {'✓ Fair' if abs(metrics['eod']) < 0.1 else '✗ Biased'}")
        report.append("")
        report.append(f"Average Odds Difference (AOD):       {metrics['aod']:+.4f}")
        report.append(f"  → Ideal: 0.0 | Current: {'✓ Fair' if abs(metrics['aod']) < 0.1 else '✗ Biased'}")
        report.append("")
        report.append(f"Disparate Impact (DI):               {metrics['di']:.4f}")
        report.append(f"  → Legal Threshold: 0.8-1.25 | Current: {'✓ Pass' if 0.8 <= metrics['di'] <= 1.25 else '✗ Fail'}")
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


def compare_fairness_metrics(
    baseline_metrics: Dict[str, float],
    mitigated_metrics: Dict[str, float]
) -> pd.DataFrame:
    """
    Compare baseline and mitigated fairness metrics.
    
    Args:
        baseline_metrics: Fairness metrics before mitigation
        mitigated_metrics: Fairness metrics after mitigation
        
    Returns:
        DataFrame with comparison
    """
    comparison_data = []
    
    for metric_name in ['spd', 'eod', 'aod', 'di']:
        baseline_val = baseline_metrics[metric_name]
        mitigated_val = mitigated_metrics[metric_name]
        
        # Calculate improvement
        if metric_name == 'di':
            # For DI, improvement is how close to 1.0
            baseline_dist = abs(baseline_val - 1.0)
            mitigated_dist = abs(mitigated_val - 1.0)
            improvement_pct = ((baseline_dist - mitigated_dist) / baseline_dist * 100) if baseline_dist > 0 else 0
        else:
            # For other metrics, improvement is reduction in absolute value
            baseline_abs = abs(baseline_val)
            mitigated_abs = abs(mitigated_val)
            improvement_pct = ((baseline_abs - mitigated_abs) / baseline_abs * 100) if baseline_abs > 0 else 0
        
        comparison_data.append({
            'Metric': metric_name.upper(),
            'Baseline': baseline_val,
            'Mitigated': mitigated_val,
            'Improvement (%)': improvement_pct
        })
    
    return pd.DataFrame(comparison_data)


if __name__ == "__main__":
    # Example usage
    print("Fairness Metrics Module - Example Usage\n")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate biased model: higher accuracy for privileged group
    protected_attr = np.random.binomial(1, 0.5, n_samples)
    y_true = np.random.binomial(1, 0.6, n_samples)
    
    # Introduce bias: lower TPR for unprivileged group
    y_pred = y_true.copy()
    unprivileged_mask = protected_attr == 0
    # Add more errors for unprivileged group
    error_indices = np.where(unprivileged_mask & (y_true == 1))[0]
    y_pred[np.random.choice(error_indices, size=int(0.2 * len(error_indices)), replace=False)] = 0
    
    # Calculate fairness metrics
    fm = FairnessMetrics(y_true, y_pred, protected_attr)
    
    # Print report
    print(fm.fairness_report())
