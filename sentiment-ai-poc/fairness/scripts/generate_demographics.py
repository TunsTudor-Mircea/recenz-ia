"""
Generate Synthetic Demographics for LaRoSeDa Dataset

This script generates synthetic demographic attributes (gender, age group) for
the LaRoSeDa test set while preserving sentiment class balance.

Usage:
    python generate_demographics.py --input_path ../../data/raw/test.csv --output_path ../data/test_with_demographics.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_demographics(
    df: pd.DataFrame,
    gender_dist: dict = None,
    age_dist: dict = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic demographic attributes for dataset.
    
    Args:
        df: Input dataframe with 'label' column
        gender_dist: Distribution of gender {'Male': 0.52, 'Female': 0.48}
        age_dist: Distribution of age groups
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with added demographic columns
    """
    np.random.seed(seed)
    
    # Default distributions from research
    if gender_dist is None:
        gender_dist = {'Male': 0.52, 'Female': 0.48}
    
    if age_dist is None:
        age_dist = {
            'Young_18-35': 0.40,
            'Middle_36-55': 0.35,
            'Senior_56+': 0.25
        }
    
    df = df.copy()
    n_samples = len(df)
    
    # Initialize columns
    df['gender'] = ''
    df['age_group'] = ''
    
    # Generate demographics while preserving class balance
    for label in df['label'].unique():
        label_mask = df['label'] == label
        n_label = label_mask.sum()
        
        # Generate gender
        gender_labels = list(gender_dist.keys())
        gender_probs = list(gender_dist.values())
        df.loc[label_mask, 'gender'] = np.random.choice(
            gender_labels,
            size=n_label,
            p=gender_probs
        )
        
        # Generate age groups
        age_labels = list(age_dist.keys())
        age_probs = list(age_dist.values())
        df.loc[label_mask, 'age_group'] = np.random.choice(
            age_labels,
            size=n_label,
            p=age_probs
        )
    
    # Add binary protected attribute for easier processing
    df['protected_attr_gender'] = (df['gender'] == 'Female').astype(int)  # 0=Female (unprivileged), 1=Male (privileged)
    
    # Age groups: combine middle and senior as "older"
    df['protected_attr_age'] = df['age_group'].map({
        'Young_18-35': 1,      # Privileged (1)
        'Middle_36-55': 0,     # Unprivileged (0)
        'Senior_56+': 0        # Unprivileged (0)
    })
    
    return df


def print_demographic_statistics(df: pd.DataFrame):
    """Print statistics about generated demographics."""
    
    print("\n" + "="*70)
    print("DEMOGRAPHIC STATISTICS")
    print("="*70)
    
    print(f"\nTotal samples: {len(df)}")
    
    # Gender distribution
    print("\nGender Distribution:")
    gender_counts = df['gender'].value_counts()
    for gender, count in gender_counts.items():
        pct = count / len(df) * 100
        print(f"  {gender}: {count} ({pct:.1f}%)")
    
    # Age distribution
    print("\nAge Group Distribution:")
    age_counts = df['age_group'].value_counts()
    for age, count in age_counts.items():
        pct = count / len(df) * 100
        print(f"  {age}: {count} ({pct:.1f}%)")
    
    # Class balance within demographics
    if 'label' in df.columns:
        print("\nClass Balance by Gender:")
        for gender in df['gender'].unique():
            gender_df = df[df['gender'] == gender]
            label_dist = gender_df['label'].value_counts()
            print(f"\n  {gender} (n={len(gender_df)}):")
            for label, count in label_dist.items():
                pct = count / len(gender_df) * 100
                print(f"    Label {label}: {count} ({pct:.1f}%)")
        
        print("\nClass Balance by Age Group:")
        for age in df['age_group'].unique():
            age_df = df[df['age_group'] == age]
            label_dist = age_df['label'].value_counts()
            print(f"\n  {age} (n={len(age_df)}):")
            for label, count in label_dist.items():
                pct = count / len(age_df) * 100
                print(f"    Label {label}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*70)


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic demographics for fairness evaluation'
    )
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save output CSV with demographics'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='Subsample dataset to this size (optional)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("SYNTHETIC DEMOGRAPHICS GENERATOR")
    print("="*70)
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Seed: {args.seed}")
    
    # Load data
    try:
        df = pd.read_csv(args.input_path)
        print(f"✓ Loaded {len(df)} samples")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Subsample if requested
    if args.sample_size and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=args.seed)
        print(f"✓ Subsampled to {len(df)} samples")
    
    # Check required columns
    if 'label' not in df.columns:
        print("✗ Error: 'label' column not found in data")
        return
    
    # Generate demographics
    df_with_demographics = generate_synthetic_demographics(df, seed=args.seed)
    print("✓ Generated synthetic demographics")
    
    # Print statistics
    print_demographic_statistics(df_with_demographics)
    
    # Save output
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_demographics.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
