"""
Fairness Evaluation Script

End-to-end pipeline for evaluating fairness of trained models and applying
bias mitigation techniques.

Usage:
    python evaluate_fairness.py --model_path path/to/model.pkl --data_path path/to/test.csv
"""

import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from fairness_metrics import FairnessMetrics, compare_fairness_metrics
from bias_mitigation import FairReweighting, ThresholdCalibrator, evaluate_mitigation


def load_model(model_path: str):
    """Load a trained model from pickle file."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_data_with_demographics(data_path: str) -> pd.DataFrame:
    """Load test data with demographic attributes."""
    df = pd.DataFrame(pd.read_csv(data_path))
    
    # Check if demographic columns exist
    required_cols = ['text', 'label', 'gender', 'age_group']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Generating synthetic demographics...")
        df = generate_synthetic_demographics(df)
    
    return df


def generate_synthetic_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic demographic attributes for evaluation."""
    np.random.seed(42)
    n_samples = len(df)
    
    # Gender: Male (52%), Female (48%)
    df['gender'] = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.52, 0.48])
    
    # Age groups: Young (40%), Middle (35%), Senior (25%)
    df['age_group'] = np.random.choice(
        ['Young_18-35', 'Middle_36-55', 'Senior_56+'],
        size=n_samples,
        p=[0.40, 0.35, 0.25]
    )
    
    # Ensure stratification preserves class balance
    for label in df['label'].unique():
        label_mask = df['label'] == label
        n_label = label_mask.sum()
        
        df.loc[label_mask, 'gender'] = np.random.choice(
            ['Male', 'Female'], size=n_label, p=[0.52, 0.48]
        )
        df.loc[label_mask, 'age_group'] = np.random.choice(
            ['Young_18-35', 'Middle_36-55', 'Senior_56+'],
            size=n_label,
            p=[0.40, 0.35, 0.25]
        )
    
    return df


def evaluate_baseline_fairness(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    protected_attr: np.ndarray,
    model_name: str = "Model"
) -> dict:
    """Evaluate fairness metrics for baseline model."""
    
    print(f"\n{'='*70}")
    print(f"BASELINE FAIRNESS EVALUATION: {model_name}")
    print(f"{'='*70}\n")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Performance Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print()
    
    # Fairness metrics
    fm = FairnessMetrics(y_test, y_pred, protected_attr)
    print(fm.fairness_report())
    
    metrics = fm.compute_all_metrics()
    
    return {
        'predictions': y_pred,
        'performance': {'accuracy': accuracy, 'f1': f1},
        'fairness': metrics,
        'group_stats': fm.get_group_statistics()
    }


def apply_threshold_calibration(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    protected_attr: np.ndarray,
    model_name: str = "Model"
) -> dict:
    """Apply threshold calibration mitigation."""
    
    print(f"\n{'='*70}")
    print(f"THRESHOLD CALIBRATION MITIGATION: {model_name}")
    print(f"{'='*70}\n")
    
    # Get prediction probabilities
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_score = model.decision_function(X_test)
    else:
        print("Warning: Model doesn't support probability predictions. Skipping calibration.")
        return None
    
    # Calibrate thresholds
    calibrator = ThresholdCalibrator(target_metric='equal_opportunity')
    calibrator.fit(y_test, y_score, protected_attr)
    
    print("Calibrated Thresholds:")
    for group, threshold in calibrator.thresholds_.items():
        group_name = "Female" if group == 0 else "Male"
        print(f"  {group_name} (Group {group}): {threshold:.4f}")
    print()
    
    # Get calibrated predictions
    y_pred_calibrated = calibrator.predict(y_score, protected_attr)
    
    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred_calibrated)
    f1 = f1_score(y_test, y_pred_calibrated, average='weighted')
    
    print(f"Performance Metrics (After Calibration):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print()
    
    # Fairness metrics
    fm = FairnessMetrics(y_test, y_pred_calibrated, protected_attr)
    print(fm.fairness_report())
    
    metrics = fm.compute_all_metrics()
    
    return {
        'predictions': y_pred_calibrated,
        'performance': {'accuracy': accuracy, 'f1': f1},
        'fairness': metrics,
        'calibrator': calibrator
    }


def plot_fairness_comparison(
    baseline_metrics: dict,
    mitigated_metrics: dict,
    save_path: str = None
):
    """Plot comparison of fairness metrics before and after mitigation."""
    
    metrics_to_plot = ['SPD', 'EOD', 'AOD']
    baseline_vals = [
        abs(baseline_metrics['spd']),
        abs(baseline_metrics['eod']),
        abs(baseline_metrics['aod'])
    ]
    mitigated_vals = [
        abs(mitigated_metrics['spd']),
        abs(mitigated_metrics['eod']),
        abs(mitigated_metrics['aod'])
    ]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Absolute metric values
    ax1.bar(x - width/2, baseline_vals, width, label='Baseline', color='#e74c3c')
    ax1.bar(x + width/2, mitigated_vals, width, label='Mitigated', color='#2ecc71')
    ax1.set_ylabel('Absolute Value (Lower is Better)')
    ax1.set_title('Fairness Metrics: Baseline vs. Mitigated')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_to_plot)
    ax1.legend()
    ax1.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Fairness Threshold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Disparate Impact
    di_data = {
        'Baseline': baseline_metrics['di'],
        'Mitigated': mitigated_metrics['di']
    }
    colors = ['#e74c3c' if abs(v - 1.0) > 0.2 else '#2ecc71' for v in di_data.values()]
    ax2.bar(di_data.keys(), di_data.values(), color=colors, width=0.5)
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Legal Threshold (0.8)')
    ax2.axhline(y=1.0, color='green', linestyle='-', alpha=0.5, label='Perfect Parity (1.0)')
    ax2.axhline(y=1.25, color='red', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Disparate Impact Ratio')
    ax2.set_title('Disparate Impact: Baseline vs. Mitigated')
    ax2.set_ylim(0.5, 1.5)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_group_performance(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    title: str = "Group Performance",
    save_path: str = None
):
    """Plot performance metrics for each demographic group."""
    
    from sklearn.metrics import confusion_matrix
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Compute metrics for each group
    groups = [0, 1]
    group_names = ['Female', 'Male']
    metrics_by_group = []
    
    for group in groups:
        mask = protected_attr == group
        y_true_group = y_test[mask]
        y_pred_group = y_pred[mask]
        
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        metrics_by_group.append({
            'Group': group_names[group],
            'TPR': tpr,
            'FPR': fpr,
            'Accuracy': accuracy,
            'Samples': len(y_true_group)
        })
    
    df_metrics = pd.DataFrame(metrics_by_group)
    
    # Plot TPR and FPR
    x = np.arange(len(group_names))
    width = 0.35
    
    axes[0].bar(x - width/2, df_metrics['TPR'], width, label='TPR', color='#3498db')
    axes[0].bar(x + width/2, df_metrics['FPR'], width, label='FPR', color='#e74c3c')
    axes[0].set_ylabel('Rate')
    axes[0].set_title('True Positive Rate & False Positive Rate by Group')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(group_names)
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot accuracy
    axes[1].bar(group_names, df_metrics['Accuracy'], color='#2ecc71', width=0.5)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy by Group')
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add sample counts
    for i, (name, count) in enumerate(zip(group_names, df_metrics['Samples'])):
        axes[1].text(i, df_metrics['Accuracy'].iloc[i] + 0.02, f'n={count}',
                     ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Evaluate fairness of trained models')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument('--data_path', type=str, help='Path to test data CSV')
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='Output directory for results')
    parser.add_argument('--protected_attr', type=str, default='gender',
                        choices=['gender', 'age_group'],
                        help='Protected attribute to evaluate')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("FAIRNESS EVALUATION PIPELINE")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Protected Attribute: {args.protected_attr}")
    print("="*70)
    
    # Load model and data
    if args.model_path:
        model = load_model(args.model_path)
        print(f"✓ Model loaded: {type(model).__name__}")
    else:
        print("No model path provided. Using example mode.")
        return
    
    if args.data_path:
        df = load_data_with_demographics(args.data_path)
        print(f"✓ Data loaded: {len(df)} samples")
    else:
        print("No data path provided. Using example mode.")
        return
    
    # Prepare data
    # (This assumes your data has 'text' and 'label' columns)
    # You'll need to adjust based on your actual data structure
    X_test = df['text'].values  # May need vectorization
    y_test = df['label'].values
    
    # Encode protected attribute
    if args.protected_attr == 'gender':
        protected_attr = (df['gender'] == 'Male').astype(int).values
    else:
        protected_attr = df['age_group'].map({
            'Young_18-35': 0,
            'Middle_36-55': 1,
            'Senior_56+': 1
        }).values
    
    # Evaluate baseline
    baseline_results = evaluate_baseline_fairness(
        model, X_test, y_test, protected_attr,
        model_name=type(model).__name__
    )
    
    # Apply threshold calibration
    mitigated_results = apply_threshold_calibration(
        model, X_test, y_test, protected_attr,
        model_name=type(model).__name__
    )
    
    if mitigated_results:
        # Plot comparisons
        plot_fairness_comparison(
            baseline_results['fairness'],
            mitigated_results['fairness'],
            save_path=output_dir / 'fairness_comparison.png'
        )
        
        plot_group_performance(
            y_test,
            baseline_results['predictions'],
            protected_attr,
            title="Baseline Model: Group Performance",
            save_path=output_dir / 'baseline_group_performance.png'
        )
        
        plot_group_performance(
            y_test,
            mitigated_results['predictions'],
            protected_attr,
            title="Mitigated Model: Group Performance",
            save_path=output_dir / 'mitigated_group_performance.png'
        )
        
        # Save results
        results_summary = {
            'baseline_fairness': baseline_results['fairness'],
            'baseline_performance': baseline_results['performance'],
            'mitigated_fairness': mitigated_results['fairness'],
            'mitigated_performance': mitigated_results['performance']
        }
        
        results_df = pd.DataFrame([results_summary])
        results_df.to_csv(output_dir / 'fairness_evaluation_results.csv', index=False)
        print(f"\n✓ Results saved to: {output_dir}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
