# Romanian Sentiment Analysis POC

A high-performance sentiment analysis system for Romanian e-commerce product reviews, achieving **91.6% accuracy** using XGBoost with advanced feature engineering.

## Key Features

- **High Accuracy**: 91.6% on LaRoSeDa Romanian e-commerce reviews dataset
- **Fast Training**: Complete training in ~7 minutes (or ~1 hour with hyperparameter optimization)
- **Advanced Feature Engineering**: LF-MICF (Log-Frequency Modified Inverse Class Frequency) + IGWO (Improved Grey Wolf Optimizer) feature selection
- **Production-Ready**: Includes model serialization, comprehensive metrics, and visualization
- **Romanian Language Support**: Specialized preprocessing with Stanza lemmatization and Romanian stopwords

## Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sentiment-ai-poc.git
cd sentiment-ai-poc
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```bash
python -c "import nltk; nltk.download('stopwords')"
```

5. **Download Stanza Romanian model** (first run downloads automatically)
```python
python -c "import stanza; stanza.download('ro')"
```

### Training

#### Standard Training (~7 minutes)
```bash
python scripts/train_xgboost.py --experiment-name my_experiment
```

#### Optimized Training with Hyperparameter Tuning (~1 hour)
```bash
python scripts/train_xgboost_optimized.py --experiment-name optimized --n-trials 50
```

Results are saved to `results/experiments/<experiment-name>/`

### What Gets Saved

Each experiment saves:
- `xgboost_model.joblib` - Trained XGBoost classifier
- `preprocessor.joblib` - Text preprocessing pipeline
- `feature_selector.joblib` - LF-MICF + IGWO feature selector
- `metrics.json` - Complete evaluation metrics
- `best_hyperparameters.json` - Optimized hyperparameters (if using optimized script)
- `plots/` - Confusion matrix, ROC curves, precision-recall curves

### Loading and Using a Trained Model

```python
from pathlib import Path
from models.xgboost_model import XGBoostModel
from features.selector import FeatureSelector
from preprocessing.pipeline import PreprocessingPipeline

# Load saved components
experiment_dir = Path('results/experiments/my_experiment')
preprocessor = PreprocessingPipeline.load(experiment_dir / 'preprocessor.joblib')
selector = FeatureSelector.load(experiment_dir / 'feature_selector.joblib')
model = XGBoostModel.load(experiment_dir / 'xgboost_model.joblib')

# Predict sentiment for new reviews
reviews = [
    "Produsul este excelent! Foarte mulțumit de calitate.",
    "Dezamăgitor. Nu recomand."
]

# Preprocess
processed = preprocessor.preprocess_batch(reviews)

# Extract features
features = selector.transform(processed)

# Predict
predictions = model.predict(features)
probabilities = model.predict_proba(features)

# Print results
class_names = ['Negative', 'Positive']
for review, pred, prob in zip(reviews, predictions, probabilities):
    sentiment = class_names[pred]
    confidence = prob[pred]
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2%})\n")
```

## Architecture

### 1. Text Preprocessing Pipeline

**Romanian-specific preprocessing:**
- HTML/URL/special character removal
- Whitespace tokenization
- Romanian stopwords removal (300+ words)
- **Stanza neural lemmatization** (GPU-accelerated)
- Snowball stemming for Romanian

```python
from preprocessing.pipeline import PreprocessingPipeline

preprocessor = PreprocessingPipeline(config)
cleaned = preprocessor.preprocess_batch(texts, batch_size=1000)
```

### 2. Feature Extraction - LF-MICF

**Log-Frequency Modified Inverse Class Frequency:**

Combines term frequency with class-specific weighting:

```
LF-MICF(t) = LTF(t) × MICF(t)
LTF(t) = log(1 + TF(t))
MICF(t) = Σ(w_q × log(1 + N_q / DF_q(t)))
```

Where:
- `LTF(t)` = Log term frequency (reduces impact of very frequent terms)
- `MICF(t)` = Modified inverse class frequency (class-specific weighting)
- `N_q` = Total documents in class q
- `DF_q(t)` = Document frequency of term t in class q

**Advantages over standard TF-IDF:**
- Better handles class imbalance
- Emphasizes discriminative features per class
- Log scaling reduces dominance of high-frequency terms

```python
from features.lf_micf import LF_MICF

extractor = LF_MICF(min_df=5, max_df=0.8)
X = extractor.fit_transform(documents, labels)
```

### 3. Feature Selection - IGWO

**Improved Grey Wolf Optimizer:**

Bio-inspired optimization algorithm for binary feature selection:

- **Population**: Pack of wolves (default: 10-20 wolves)
- **Hierarchy**: Alpha (best), Beta (2nd), Delta (3rd), Omega (rest)
- **Hunt**: Wolves encircle and hunt optimal feature subset
- **Fitness**: 5-fold cross-validation accuracy with RandomForest
- **Improvements**: Velocity-based updates, adaptive weights

**Process:**
1. Initialize random binary feature vectors (wolves)
2. Evaluate fitness (CV accuracy) for each wolf
3. Update positions based on alpha, beta, delta wolves
4. Apply sigmoid for binary conversion
5. Repeat until convergence or max iterations

```python
from features.igwo import IGWO

selector = IGWO(
    n_wolves=20,
    n_iterations=30,
    target_features=1000,
    cv_folds=5
)
X_selected = selector.fit_transform(X, y)
```

### 4. Classification - XGBoost

**Gradient Boosting Classifier:**

Fast and accurate classifier for the selected features:

**Default hyperparameters:**
- `n_estimators`: 200 (300-500 with optimization)
- `max_depth`: 8 (3-12 with optimization)
- `learning_rate`: 0.1 (0.01-0.3 with optimization)
- `subsample`: 0.8 (0.6-1.0 with optimization)
- `colsample_bytree`: 0.8 (0.6-1.0 with optimization)

**With Optuna optimization (9 hyperparameters tuned):**
- Bayesian optimization over 50 trials
- Validation-based early stopping
- Automatic best parameter selection

```python
from models.xgboost_model import XGBoostModel

model = XGBoostModel(hyperparams={...}, n_classes=2)
model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
```

## Dataset

**LaRoSeDa** (Large Romanian Sentiment Data)
- Source: HuggingFace - `universityofbucharest/laroseda`
- Reviews: 15,000 Romanian e-commerce product reviews
- Classes: Binary (Negative: 1-2 stars, Positive: 3-5 stars)
- Train/Test: 12,000 / 3,000 samples

The dataset is automatically downloaded from HuggingFace on first run and cached locally.

## Configuration

### Preprocessing Config (`configs/preprocessing_config.yaml`)

```yaml
cleaner:
  lowercase: true
  remove_html: true
  remove_urls: true
  remove_special_chars: true

tokenizer:
  method: whitespace

lemmatizer:
  use_lemmatization: true
  language: ro

stemmer:
  use_stemming: true
  language: romanian

stopwords:
  remove: true
  language: romanian
```

### Feature Config (`configs/feature_config.yaml`)

```yaml
lf_micf:
  min_df: 5        # Minimum document frequency
  max_df: 0.8      # Maximum document frequency

igwo:
  n_wolves: 10     # Number of wolves in pack
  n_iterations: 20 # Maximum iterations
  inertia_weight: 0.9
  target_features: 800  # Target number of features
  fitness_cv_folds: 3   # CV folds for fitness
  sample_size: 1000     # Subsample for faster IGWO
  random_state: 42
```

**Optimization tips:**
- Increase `n_wolves` and `n_iterations` for better feature selection (slower)
- Increase `target_features` for potentially higher accuracy (larger models)
- Increase `sample_size` for more robust selection (slower)
- Increase `fitness_cv_folds` for better validation (slower)

## Performance

### Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 91.60% |
| **Precision** | 91.63% |
| **Recall** | 91.60% |
| **F1 Score** | 91.60% |
| **Training Time** | ~7 min (standard) / ~60 min (optimized) |

### System Requirements

**Minimum:**
- CPU: Dual-core processor
- RAM: 8 GB
- Disk: 2 GB free space

**Recommended:**
- CPU: Quad-core or better (Intel Core i5/AMD Ryzen 5+)
- RAM: 16 GB
- GPU: NVIDIA GPU with CUDA support (for Stanza lemmatization)
- Disk: 5 GB free space

**GPU Acceleration:**
- Stanza lemmatization automatically uses GPU if available
- Speeds up preprocessing by ~3-5x
- XGBoost uses all CPU cores (`n_jobs=-1`)

## Project Structure

```
sentiment-ai-poc/
├── configs/                      # Configuration files
│   ├── preprocessing_config.yaml # Text preprocessing settings
│   └── feature_config.yaml       # LF-MICF + IGWO settings
│
├── data/
│   └── raw/                      # HuggingFace dataset cache (gitignored)
│
├── logs/                         # Training logs (gitignored)
│
├── notebooks/                    # Jupyter notebooks (for analysis)
│
├── results/
│   └── experiments/              # Experiment results (gitignored)
│       ├── xgb_20250101_120000/ # Example experiment
│       │   ├── xgboost_model.joblib
│       │   ├── preprocessor.joblib
│       │   ├── feature_selector.joblib
│       │   ├── metrics.json
│       │   ├── best_hyperparameters.json
│       │   └── plots/
│       │       ├── confusion_matrix.png
│       │       ├── roc_curves.png
│       │       └── precision_recall.png
│       └── ...
│
├── scripts/                      # Training scripts
│   ├── train_xgboost.py         # Standard training
│   └── train_xgboost_optimized.py # With Optuna optimization
│
└── src/                          # Source code modules
    ├── preprocessing/            # Text preprocessing
    │   ├── cleaner.py
    │   ├── tokenizer.py
    │   ├── lemmatizer.py        # Stanza-based Romanian lemmatization
    │   ├── stemmer.py
    │   └── pipeline.py
    │
    ├── features/                 # Feature engineering
    │   ├── lf_micf.py           # LF-MICF extraction
    │   ├── igwo.py              # IGWO feature selection
    │   └── selector.py           # Feature selector pipeline
    │
    ├── models/                   # ML models
    │   └── xgboost_model.py     # XGBoost classifier
    │
    ├── evaluation/               # Evaluation & visualization
    │   ├── metrics.py
    │   └── visualizer.py
    │
    └── utils/                    # Utilities
        ├── config.py
        ├── logger.py
        └── reproducibility.py
```

## API Reference

### Core Classes

#### PreprocessingPipeline
```python
from preprocessing.pipeline import PreprocessingPipeline

preprocessor = PreprocessingPipeline(config)
preprocessed = preprocessor.preprocess_batch(texts, batch_size=1000)
preprocessor.save('preprocessor.joblib')
preprocessor = PreprocessingPipeline.load('preprocessor.joblib')
```

#### FeatureSelector
```python
from features.selector import FeatureSelector

selector = FeatureSelector(
    extractor_params={'min_df': 5, 'max_df': 0.8},
    selector_params={'n_wolves': 20, 'n_iterations': 30, 'target_features': 1000}
)
selector.fit(texts, labels, verbose=True)
X = selector.transform(texts)
selector.save('feature_selector.joblib')
selector = FeatureSelector.load('feature_selector.joblib')
```

#### XGBoostModel
```python
from models.xgboost_model import XGBoostModel

model = XGBoostModel(
    hyperparams={'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1},
    n_classes=2
)
model.fit(X_train, y_train, X_val, y_val, verbose=True)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
accuracy = model.score(X_test, y_test)
model.save('model.joblib')
model = XGBoostModel.load('model.joblib')
```

## Advanced Usage

### Custom Feature Selection Parameters

```python
from features.selector import FeatureSelector

# More aggressive feature selection
selector = FeatureSelector(
    extractor_params={
        'min_df': 3,   # Lower threshold - more features
        'max_df': 0.9  # Higher threshold - keep common terms
    },
    selector_params={
        'n_wolves': 30,        # More wolves - better exploration
        'n_iterations': 50,    # More iterations - better convergence
        'target_features': 1500, # More features - potentially higher accuracy
        'cv_folds': 5,         # More folds - better validation
        'sample_size': 3000    # Larger sample - more robust
    }
)
```

### Hyperparameter Optimization with Optuna

The optimized training script explores these parameters:
- `n_estimators`: [100, 500]
- `max_depth`: [3, 12]
- `learning_rate`: [0.01, 0.3] (log scale)
- `subsample`: [0.6, 1.0]
- `colsample_bytree`: [0.6, 1.0]
- `min_child_weight`: [1, 10]
- `gamma`: [0, 0.5]
- `reg_alpha`: [0, 1.0]
- `reg_lambda`: [0, 1.0]

```bash
# Quick optimization (20 trials, ~30 min)
python scripts/train_xgboost_optimized.py --n-trials 20

# Thorough optimization (100 trials, ~2 hours)
python scripts/train_xgboost_optimized.py --n-trials 100

# Skip optimization, use best known params
python scripts/train_xgboost_optimized.py --skip-optuna
```

## Evaluation Metrics

The system provides comprehensive evaluation:

**Classification Metrics:**
- Accuracy, Precision, Recall, F1 Score (macro & weighted)
- Per-class metrics
- Confusion matrix

**Probability Metrics:**
- ROC curves (one-vs-rest)
- AUC scores per class
- Precision-Recall curves
- Average Precision scores

**Visualizations:**
- Confusion matrix heatmap
- ROC curves with AUC
- Precision-Recall curves with AP

All metrics and plots are automatically saved to `results/experiments/<name>/`

## Troubleshooting

### Common Issues

**1. Out of memory during IGWO**
```yaml
# Reduce in configs/feature_config.yaml
igwo:
  sample_size: 500  # Reduce from 1000
  target_features: 500  # Reduce from 800
```

**2. Stanza GPU not detected**
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**3. Slow preprocessing**
- Ensure Stanza is using GPU (if available)
- Increase `batch_size` parameter
- Reduce sample size for faster testing

**4. Matplotlib backend errors**
- The scripts use `Agg` backend (non-interactive)
- If issues persist, set: `export MPLBACKEND=Agg`

## Development

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests (when implemented)
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality
```bash
# Format code
black src/ scripts/

# Type checking
mypy src/
```

## Contributing

Contributions are welcome! Areas for improvement:

1. **Multi-class classification** (positive/neutral/negative)
2. **Additional models** (BERT, RoBERT-Romanian, etc.)
3. **Real-time inference API** (FastAPI/Flask)
4. **Docker containerization**
5. **Unit tests** for all modules
6. **Example notebooks** for analysis
7. **Web interface** for predictions

Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{romanian_sentiment_poc,
  title={Romanian Sentiment Analysis POC: High-Performance E-commerce Review Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/sentiment-ai-poc}
}
```

## Acknowledgments

- **LaRoSeDa Dataset**: University of Bucharest
- **HuggingFace**: Dataset hosting and easy access
- **Stanza**: Stanford NLP Romanian language model
- **Research Papers**: LF-MICF, IGWO, and optimization algorithm concepts

## References

1. **LF-MICF**: Log-Frequency Modified Inverse Class Frequency for text classification
2. **IGWO**: Improved Grey Wolf Optimizer for feature selection
3. **XGBoost**: Gradient boosting framework for classification tasks
4. **LaRoSeDa**: Large Romanian Sentiment Dataset (University of Bucharest)

---

**Note**: This is a proof-of-concept implementation optimized for accuracy and speed. For production deployment, consider adding API wrappers, monitoring, and proper error handling.
