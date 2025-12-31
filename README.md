# Recenz-IA

AI-powered sentiment analysis system for Romanian e-commerce product reviews.

## Overview

High-performance sentiment analysis for Romanian language reviews, achieving **91.6% accuracy** using XGBoost with advanced feature engineering (LF-MICF + IGWO).

## Quick Start

```bash
cd sentiment-ai-poc

# Setup
python -m venv venv
venv\Scripts\activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train model
python scripts/train_xgboost.py --experiment-name my_experiment

# With optimization (91.6% accuracy)
python scripts/train_xgboost_optimized.py --experiment-name optimized --n-trials 50
```

**Full documentation:** See [sentiment-ai-poc/README.md](sentiment-ai-poc/README.md)

## Features

- **91.6% accuracy** on LaRoSeDa Romanian reviews (15k samples)
- **Fast training**: ~7 minutes (standard) or ~1 hour (optimized)
- **Romanian NLP**: Stanza lemmatization with GPU acceleration
- **Production-ready**: Model serialization, metrics, visualization
- **Jupyter notebooks**: Data exploration and model analysis

## Technology

- **Feature Engineering**: LF-MICF extraction + IGWO selection
- **Classification**: XGBoost with Optuna optimization
- **NLP**: Stanza (Romanian), NLTK, custom preprocessing
- **Analysis**: Jupyter notebooks with matplotlib/seaborn

## Project Structure

```
recenz-ia/
└── sentiment-ai-poc/          # Main project
    ├── configs/               # YAML configurations
    ├── notebooks/             # Jupyter analysis notebooks
    ├── scripts/               # Training scripts
    ├── src/                   # Source modules
    │   ├── preprocessing/     # Romanian text processing
    │   ├── features/          # LF-MICF + IGWO
    │   ├── models/            # XGBoost classifier
    │   ├── evaluation/        # Metrics & visualization
    │   └── utils/             # Utilities
    └── results/               # Trained models (gitignored)
```

## Usage Example

```python
from pathlib import Path
from models.xgboost_model import XGBoostModel
from features.selector import FeatureSelector
from preprocessing.pipeline import PreprocessingPipeline

# Load trained model
exp_dir = Path('sentiment-ai-poc/results/experiments/my_experiment')
preprocessor = PreprocessingPipeline.load(exp_dir / 'preprocessor.joblib')
selector = FeatureSelector.load(exp_dir / 'feature_selector.joblib')
model = XGBoostModel.load(exp_dir / 'xgboost_model.joblib')

# Predict
review = "Produsul este excelent! Foarte mulțumit."
processed = preprocessor.preprocess_batch([review])
features = selector.transform(processed)
prediction = model.predict(features)
confidence = model.predict_proba(features)[0][prediction[0]]

print(f"Sentiment: {'Positive' if prediction[0] == 1 else 'Negative'}")
print(f"Confidence: {confidence:.2%}")
```

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 91.60% |
| Precision | 91.63% |
| Recall | 91.60% |
| F1 Score | 91.60% |
| Training Time | ~7 min / ~60 min (optimized) |

## Requirements

- Python 3.8+
- 8 GB RAM (16 GB recommended)
- Optional: NVIDIA GPU for faster preprocessing

## License

MIT License

## Citation

```bibtex
@software{recenz_ia,
  title={Recenz-IA: Romanian Sentiment Analysis for e-commerce},
  author={Tudor Tuns},
  year={2025},
  url={https://github.com/TunsTudor-Mircea/recenz-ia}
}
```
