# Recenz-IA

AI-powered sentiment analysis system for e-commerce product reviews in Romanian.

## Overview

This repository contains a high-performance sentiment analysis solution designed specifically for Romanian language e-commerce reviews. The system achieves **91.6% accuracy** using advanced machine learning techniques and Romanian-specific NLP preprocessing.

## Project Structure

```
recenz-ia/
└── sentiment-ai-poc/     # Romanian Sentiment Analysis POC
    ├── configs/          # Configuration files
    ├── data/             # Dataset storage (gitignored)
    ├── notebooks/        # Jupyter notebooks for analysis
    ├── results/          # Trained models and results (gitignored)
    ├── scripts/          # Training scripts
    └── src/              # Source code modules
```

## Components

### Sentiment Analysis POC (`sentiment-ai-poc/`)

A production-ready proof-of-concept for Romanian sentiment analysis using:

- **Advanced Feature Engineering**: LF-MICF (Log-Frequency Modified Inverse Class Frequency) for feature extraction
- **Bio-Inspired Optimization**: IGWO (Improved Grey Wolf Optimizer) for feature selection
- **XGBoost Classification**: Gradient boosting with Optuna hyperparameter optimization
- **Romanian NLP**: Stanza neural lemmatization with GPU acceleration

**Key Features:**
- ✅ **91.6% accuracy** on LaRoSeDa Romanian e-commerce reviews
- ✅ **Fast training** (~7 minutes standard, ~1 hour with optimization)
- ✅ **Production-ready** with model serialization and comprehensive evaluation
- ✅ **GPU-accelerated** preprocessing with Stanza
- ✅ **Jupyter notebooks** for data exploration and model analysis

**Quick Start:**
```bash
cd sentiment-ai-poc

# Install dependencies
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Train model
python scripts/train_xgboost.py --experiment-name my_experiment

# Or with hyperparameter optimization
python scripts/train_xgboost_optimized.py --experiment-name optimized --n-trials 50
```

**Full Documentation:**
See [sentiment-ai-poc/README.md](sentiment-ai-poc/README.md) for complete documentation, API reference, and usage examples.

## Dataset

**LaRoSeDa** (Large Romanian Sentiment Data)
- 15,000 Romanian e-commerce product reviews
- Binary classification (Negative: 1-2 stars, Positive: 3-5 stars)
- Automatically downloaded from HuggingFace
- Source: University of Bucharest

## Technology Stack

**Machine Learning:**
- XGBoost for classification
- Optuna for hyperparameter optimization
- Scikit-learn for utilities

**NLP & Preprocessing:**
- Stanza (Stanford NLP) for Romanian lemmatization
- NLTK for stopwords and tokenization
- Custom LF-MICF feature extraction
- IGWO bio-inspired feature selection

**Visualization & Analysis:**
- Matplotlib & Seaborn for plotting
- Jupyter notebooks for interactive analysis
- Comprehensive metrics and confusion matrices

## Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 91.60% |
| **Precision** | 91.63% |
| **Recall** | 91.60% |
| **F1 Score** | 91.60% |
| **Training Time** | ~7 min (standard) / ~60 min (optimized) |

## System Requirements

**Minimum:**
- Python 3.8+
- 8 GB RAM
- 2 GB disk space

**Recommended:**
- Python 3.10+
- 16 GB RAM
- NVIDIA GPU with CUDA (for faster preprocessing)
- 5 GB disk space

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/recenz-ia.git
cd recenz-ia/sentiment-ai-poc
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLP Resources
```bash
# NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

# Stanza Romanian model (downloads on first run)
python -c "import stanza; stanza.download('ro')"
```

## Usage

### Training a Model

**Standard training:**
```bash
python scripts/train_xgboost.py --experiment-name my_experiment
```

**With hyperparameter optimization:**
```bash
python scripts/train_xgboost_optimized.py --experiment-name optimized --n-trials 50
```

### Using Trained Models

```python
from pathlib import Path
from models.xgboost_model import XGBoostModel
from features.selector import FeatureSelector
from preprocessing.pipeline import PreprocessingPipeline

# Load components
experiment_dir = Path('results/experiments/my_experiment')
preprocessor = PreprocessingPipeline.load(experiment_dir / 'preprocessor.joblib')
selector = FeatureSelector.load(experiment_dir / 'feature_selector.joblib')
model = XGBoostModel.load(experiment_dir / 'xgboost_model.joblib')

# Predict
review = "Produsul este excelent! Foarte mulțumit."
processed = preprocessor.preprocess_batch([review])
features = selector.transform(processed)
prediction = model.predict(features)
probability = model.predict_proba(features)

print(f"Sentiment: {'Positive' if prediction[0] == 1 else 'Negative'}")
print(f"Confidence: {probability[0][prediction[0]]:.2%}")
```

### Jupyter Notebooks

Explore data and analyze models interactively:

```bash
cd notebooks
jupyter notebook
```

**Available notebooks:**
- `01_data_exploration.ipynb` - Dataset analysis and visualization
- `02_model_analysis.ipynb` - Model performance and predictions

## Project Architecture

### Pipeline Flow

```
Romanian Text Reviews
        ↓
1. Preprocessing Pipeline
   - Text cleaning
   - Tokenization
   - Lemmatization (Stanza)
   - Stemming
   - Stopwords removal
        ↓
2. Feature Extraction (LF-MICF)
   - Term frequency calculation
   - Class-specific weighting
   - Log-frequency scaling
        ↓
3. Feature Selection (IGWO)
   - Bio-inspired optimization
   - Cross-validation fitness
   - Binary feature selection
        ↓
4. Classification (XGBoost)
   - Gradient boosting
   - Optuna optimization
   - Binary prediction
        ↓
Sentiment (Positive/Negative)
+ Confidence Score
```

## Results & Outputs

Each training run saves to `results/experiments/<experiment-name>/`:

- `xgboost_model.joblib` - Trained classifier
- `preprocessor.joblib` - Preprocessing pipeline
- `feature_selector.joblib` - Feature extraction & selection
- `metrics.json` - Complete evaluation metrics
- `best_hyperparameters.json` - Optimized parameters
- `plots/` - Confusion matrix, ROC curves, PR curves

## Future Enhancements

Potential areas for expansion:

1. **Multi-class Classification** - Positive/Neutral/Negative
2. **Transformer Models** - BERT, RoBERT-Romanian
3. **Real-time API** - FastAPI/Flask deployment
4. **Web Interface** - User-friendly prediction UI
5. **Aspect-Based Analysis** - Product feature sentiment
6. **Multi-language Support** - Extend to other languages
7. **Docker Deployment** - Containerized production setup

## Contributing

Contributions are welcome! Areas of interest:

- Model improvements and optimizations
- Additional Romanian language features
- Web API and deployment solutions
- Extended documentation and tutorials
- Unit tests and integration tests

Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{recenz_ia,
  title={Recenz-IA: High-Performance Romanian Sentiment Analysis},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/recenz-ia}
}
```

## Acknowledgments

- **LaRoSeDa Dataset** - University of Bucharest
- **HuggingFace** - Dataset hosting and infrastructure
- **Stanza (Stanford NLP)** - Romanian language models
- **Research Community** - LF-MICF, IGWO, and XGBoost methodologies

## Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Note**: This is a proof-of-concept implementation optimized for research and development. For production deployment, consider additional security, monitoring, and scalability measures.
