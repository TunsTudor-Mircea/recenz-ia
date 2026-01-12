"""
Fairness Analysis Notebook - Initialization Module

This module contains helper functions and imports for the fairness analysis notebook.
"""

import sys
from pathlib import Path

# Add scripts directory to Python path
scripts_dir = Path('../scripts').resolve()
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Custom fairness modules
from fairness_metrics import FairnessMetrics, compare_fairness_metrics
from bias_mitigation import (
    FairReweighting,
    ThresholdCalibrator,
    FairnessAwareTraining,
    evaluate_mitigation
)

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("✓ Fairness analysis modules loaded successfully")
