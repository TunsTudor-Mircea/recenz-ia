"""
Classical ML baselines for ABSA sentence-pair classification.

Trains TF-IDF + Logistic Regression and TF-IDF + Linear SVM classifiers,
one per aspect category, as lower-bound baselines for the transformer models.

Approach
--------
For each aspect, a 4-class classifier is trained:
    label ∈ {none (0), positive (1), negative (2), neutral (3)}

The feature vector is TF-IDF over the concatenated review text and auxiliary
sentence:  text_a + " " + text_b.

Class imbalance ("none" typically ~87% of pairs) is handled with
class_weight='balanced', which reweights each sample by the inverse class
frequency during training.

Models trained
--------------
    LR   : LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
    SVM  : LinearSVC(C=0.1, max_iter=5000, class_weight='balanced')

TF-IDF configuration
--------------------
    analyzer='word', ngram_range=(1, 2), min_df=2, max_features=80_000,
    sublinear_tf=True

Both models are trained and evaluated. Best checkpoint (by macro-F1 on val set,
excluding the "none" class) is saved as the baseline model.

Outputs
-------
    absa/models/baselines/{lr,svm}_model.pkl      sklearn Pipeline objects
    absa/models/baselines/baseline_metrics.json   per-aspect + macro metrics
    absa/models/baselines/baseline_metrics.csv    same in tabular form

Usage
-----
    # Train both models on the prepared splits (run from repo root)
    python absa/scripts/train_baselines.py

    # Train only LR
    python absa/scripts/train_baselines.py --model lr

    # Evaluate only (load saved models, re-run evaluation)
    python absa/scripts/train_baselines.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

SPLITS_DIR  = Path('absa/data/splits')
OUT_DIR     = Path('absa/models/baselines')

ASPECT_LABELS = [
    'BATERIE', 'ECRAN', 'SUNET', 'PERFORMANTA', 'CONECTIVITATE',
    'DESIGN', 'CALITATE_CONSTRUCTIE', 'PRET', 'LIVRARE', 'GENERAL',
]

LABEL2ID: dict[str, int] = {'none': 0, 'positive': 1, 'negative': 2, 'neutral': 3}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}
POLARITY_IDS = [1, 2, 3]  # exclude 'none' from macro-F1


# ── model factories ───────────────────────────────────────────────────────────

def make_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        min_df=2,
        max_features=80_000,
        sublinear_tf=True,
    )


def make_lr_pipeline() -> Pipeline:
    return Pipeline([
        ('tfidf', make_tfidf()),
        ('clf', LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            solver='lbfgs',
            multi_class='multinomial',
        )),
    ])


def make_svm_pipeline() -> Pipeline:
    return Pipeline([
        ('tfidf', make_tfidf()),
        ('clf', LinearSVC(
            C=0.1,
            max_iter=5000,
            class_weight='balanced',
            random_state=RANDOM_SEED,
        )),
    ])


# ── evaluation helpers ────────────────────────────────────────────────────────

def aspect_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute per-aspect ABSA metrics.

    Detection metrics: treat label_id > 0 as "aspect mentioned" (binary).
    Polarity metrics: among instances where both true and pred are non-zero,
                      compute 3-class (positive/negative/neutral) F1.
    Combined F1: exact (aspect, polarity) match — proportion of non-none true
                 labels where pred == true.
    """
    det_true = (y_true > 0).astype(int)
    det_pred = (y_pred > 0).astype(int)

    det_p  = precision_score(det_true, det_pred, zero_division=0)
    det_r  = recall_score(det_true, det_pred, zero_division=0)
    det_f1 = f1_score(det_true, det_pred, zero_division=0)

    # Polarity: among true non-none instances
    true_detected = y_true > 0
    pol_p = pol_r = pol_f1 = None
    pol_n = int(true_detected.sum())
    if pol_n >= 2:
        pol_p  = precision_score(y_true[true_detected], y_pred[true_detected],
                                 average='macro', labels=POLARITY_IDS, zero_division=0)
        pol_r  = recall_score(y_true[true_detected], y_pred[true_detected],
                               average='macro', labels=POLARITY_IDS, zero_division=0)
        pol_f1 = f1_score(y_true[true_detected], y_pred[true_detected],
                          average='macro', labels=POLARITY_IDS, zero_division=0)

    # Combined: among all 4 classes (including none)
    combined_f1 = f1_score(y_true, y_pred, average='macro',
                           labels=list(LABEL2ID.values()), zero_division=0)

    return {
        'detection_precision': round(det_p, 4),
        'detection_recall':    round(det_r, 4),
        'detection_f1':        round(det_f1, 4),
        'polarity_macro_f1':   round(pol_f1, 4) if pol_f1 is not None else None,
        'polarity_macro_p':    round(pol_p, 4)  if pol_p  is not None else None,
        'polarity_macro_r':    round(pol_r, 4)  if pol_r  is not None else None,
        'polarity_n':          pol_n,
        'combined_macro_f1':   round(combined_f1, 4),
        'n_true_pos': int((y_true > 0).sum()),
        'n_pred_pos': int((y_pred > 0).sum()),
    }


def macro_over_aspects(per_aspect: dict[str, dict], key: str) -> float | None:
    vals = [v[key] for v in per_aspect.values() if v.get(key) is not None]
    return round(float(np.mean(vals)), 4) if vals else None


# ── training and evaluation ───────────────────────────────────────────────────

def train_and_eval(
    model_name: Literal['lr', 'svm'],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, Pipeline], dict]:
    """
    Train one model per aspect, evaluate on val and test, return models + metrics.
    """
    make_pipeline = make_lr_pipeline if model_name == 'lr' else make_svm_pipeline
    models: dict[str, Pipeline] = {}
    metrics = {'val': {}, 'test': {}}

    for asp in ASPECT_LABELS:
        log.info('[%s] Training %s …', asp, model_name.upper())
        tr = train_df[train_df['aspect'] == asp].reset_index(drop=True)
        vl = val_df[val_df['aspect'] == asp].reset_index(drop=True)
        te = test_df[test_df['aspect'] == asp].reset_index(drop=True)

        X_tr = tr['text_a'] + ' ' + tr['text_b']
        y_tr = tr['label_id'].values

        X_vl = vl['text_a'] + ' ' + vl['text_b']
        y_vl = vl['label_id'].values

        X_te = te['text_a'] + ' ' + te['text_b']
        y_te = te['label_id'].values

        pipe = make_pipeline()
        pipe.fit(X_tr, y_tr)

        y_vl_pred = pipe.predict(X_vl)
        y_te_pred = pipe.predict(X_te)

        metrics['val'][asp]  = aspect_metrics(y_vl, y_vl_pred)
        metrics['test'][asp] = aspect_metrics(y_te, y_te_pred)

        models[asp] = pipe
        log.info('  val  det_f1=%.4f  pol_f1=%s',
                 metrics['val'][asp]['detection_f1'],
                 metrics['val'][asp]['polarity_macro_f1'])
        log.info('  test det_f1=%.4f  pol_f1=%s',
                 metrics['test'][asp]['detection_f1'],
                 metrics['test'][asp]['polarity_macro_f1'])

    # Macro averages
    for split in ('val', 'test'):
        metrics[split]['macro'] = {
            'detection_f1':     macro_over_aspects(metrics[split], 'detection_f1'),
            'polarity_f1':      macro_over_aspects(metrics[split], 'polarity_macro_f1'),
            'combined_f1':      macro_over_aspects(metrics[split], 'combined_macro_f1'),
        }
        log.info('[%s] Macro  det_f1=%.4f  pol_f1=%s  comb_f1=%.4f',
                 split.upper(),
                 metrics[split]['macro']['detection_f1'],
                 metrics[split]['macro']['polarity_f1'],
                 metrics[split]['macro']['combined_f1'])

    return models, metrics


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train classical ML baselines for ABSA.')
    p.add_argument('--model', choices=['lr', 'svm', 'both'], default='both',
                   help='Which model(s) to train (default: both).')
    p.add_argument('--eval-only', action='store_true',
                   help='Load saved models and re-evaluate without retraining.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_SEED)

    for path in (SPLITS_DIR / 'train.csv', SPLITS_DIR / 'val.csv', SPLITS_DIR / 'test_gold.csv'):
        if not path.exists():
            log.error('%s not found. Run prepare_training_data.py first.', path)
            raise SystemExit(1)

    log.info('Loading data splits …')
    train_df = pd.read_csv(SPLITS_DIR / 'train.csv', dtype={'review_id': str})
    val_df   = pd.read_csv(SPLITS_DIR / 'val.csv',   dtype={'review_id': str})
    test_df  = pd.read_csv(SPLITS_DIR / 'test_gold.csv', dtype={'review_id': str})
    log.info('Train: %d  Val: %d  Test: %d', len(train_df), len(val_df), len(test_df))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    models_to_run = ['lr', 'svm'] if args.model == 'both' else [args.model]

    for model_name in models_to_run:
        if args.eval_only:
            model_path = OUT_DIR / f'{model_name}_models.pkl'
            if not model_path.exists():
                log.error('Saved model not found at %s. Train first.', model_path)
                continue
            with model_path.open('rb') as f:
                models = pickle.load(f)
        else:
            models, metrics = train_and_eval(model_name, train_df, val_df, test_df)
            all_metrics[model_name] = metrics
            # Save model
            with (OUT_DIR / f'{model_name}_models.pkl').open('wb') as f:
                pickle.dump(models, f)
            log.info('Saved %s models → %s', model_name.upper(), OUT_DIR / f'{model_name}_models.pkl')

    if not args.eval_only and all_metrics:
        # Save combined metrics
        with (OUT_DIR / 'baseline_metrics.json').open('w') as f:
            json.dump(all_metrics, f, indent=2)

        # Flat CSV for easy comparison
        rows = []
        for model_name, m in all_metrics.items():
            for split in ('val', 'test'):
                for asp in ASPECT_LABELS + ['macro']:
                    d = m[split].get(asp, {})
                    rows.append({
                        'model': model_name.upper(),
                        'split': split,
                        'aspect': asp,
                        **{k: v for k, v in d.items() if not isinstance(v, dict)},
                    })
        pd.DataFrame(rows).to_csv(OUT_DIR / 'baseline_metrics.csv', index=False)
        log.info('Metrics saved → %s', OUT_DIR)


if __name__ == '__main__':
    main()
