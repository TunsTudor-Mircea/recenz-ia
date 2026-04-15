"""
Unified evaluation script for ABSA models on the gold test set.

Evaluates any saved model (transformer checkpoint or sklearn pickle) against
the human-validated gold set, and produces a comparison table that includes:
    - Baseline models    (LR, SVM)
    - Transformer models (mBERT, RoBERT, XLM-R)
    - NLI zero-shot      (from cached nli_gold_annotations.jsonl)

Metrics reported (per aspect + macro average)
---------------------------------------------
    detection_precision   P for binary "aspect mentioned" decision
    detection_recall      R for binary "aspect mentioned" decision
    detection_f1          F1 for binary "aspect mentioned" decision
    polarity_macro_f1     Macro-F1 over {positive, negative, neutral}
                          computed only on instances where both human and
                          model agree the aspect is present
    combined_macro_f1     4-class macro-F1 over {none, positive, negative, neutral}

The combined_macro_f1 is the most directly comparable metric across all model
types and corresponds to the "sentiment accuracy" reported in SemEval ABSA.

Usage
-----
    # Evaluate a transformer checkpoint
    python absa/scripts/evaluate.py \\
        --model-type transformer \\
        --model-path absa/models/checkpoints/xlm-roberta-base/best_model \\
        --run-name xlmr

    # Evaluate LR/SVM baselines (loads from baselines/ directory)
    python absa/scripts/evaluate.py --model-type baseline

    # Evaluate NLI zero-shot only
    python absa/scripts/evaluate.py --model-type nli

    # Build the full comparison table from all saved results
    python absa/scripts/evaluate.py --comparison-table
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

SPLITS_DIR   = Path('absa/data/splits')
BASELINES_DIR = Path('absa/models/baselines')
CKPT_DIR     = Path('absa/models/checkpoints')
NLI_CACHE    = Path('absa/data/nli_gold_annotations.jsonl')
GOLD_CSV     = Path('absa/data/gold_annotated.csv')
RESULTS_DIR  = Path('absa/models/results')

ASPECT_LABELS = [
    'BATERIE', 'ECRAN', 'SUNET', 'PERFORMANTA', 'CONECTIVITATE',
    'DESIGN', 'CALITATE_CONSTRUCTIE', 'PRET', 'LIVRARE', 'GENERAL',
]

LABEL2ID = {'none': 0, 'positive': 1, 'negative': 2, 'neutral': 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
POLARITY_IDS = [1, 2, 3]

ASPECT_AUX: dict[str, str] = {
    'BATERIE':              'baterie și autonomie',
    'ECRAN':                'ecran și afișaj',
    'SUNET':                'sunet și audio',
    'PERFORMANTA':          'performanță și viteză',
    'CONECTIVITATE':        'conectivitate și semnal',
    'DESIGN':               'design și aspect fizic',
    'CALITATE_CONSTRUCTIE': 'calitatea construcției și durabilitate',
    'PRET':                 'preț și raport calitate-preț',
    'LIVRARE':              'livrare și ambalaj',
    'GENERAL':              'impresie generală',
}


# ── metrics ───────────────────────────────────────────────────────────────────

def aspect_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-aspect metrics for a single aspect column."""
    det_true = (y_true > 0).astype(int)
    det_pred = (y_pred > 0).astype(int)

    det_p  = float(precision_score(det_true, det_pred, zero_division=0))
    det_r  = float(recall_score(det_true, det_pred, zero_division=0))
    det_f1 = float(f1_score(det_true, det_pred, zero_division=0))

    true_pos = y_true > 0
    pol_p = pol_r = pol_f1 = None
    if true_pos.sum() >= 2:
        pol_p  = float(precision_score(y_true[true_pos], y_pred[true_pos],
                                       average='macro', labels=POLARITY_IDS, zero_division=0))
        pol_r  = float(recall_score(y_true[true_pos], y_pred[true_pos],
                                     average='macro', labels=POLARITY_IDS, zero_division=0))
        pol_f1 = float(f1_score(y_true[true_pos], y_pred[true_pos],
                                 average='macro', labels=POLARITY_IDS, zero_division=0))

    comb_f1 = float(f1_score(y_true, y_pred, average='macro',
                              labels=list(LABEL2ID.values()), zero_division=0))

    return {
        'n_human':           int(true_pos.sum()),
        'n_pred':            int((y_pred > 0).sum()),
        'detection_p':       round(det_p, 4),
        'detection_r':       round(det_r, 4),
        'detection_f1':      round(det_f1, 4),
        'polarity_macro_f1': round(pol_f1, 4) if pol_f1 is not None else None,
        'polarity_macro_p':  round(pol_p, 4)  if pol_p  is not None else None,
        'polarity_macro_r':  round(pol_r, 4)  if pol_r  is not None else None,
        'combined_macro_f1': round(comb_f1, 4),
    }


def compute_all_metrics(
    test_df: pd.DataFrame,
    pred_label_ids: np.ndarray,
) -> dict:
    """
    Compute per-aspect and macro metrics given gold test pairs and predictions.

    Parameters
    ----------
    test_df        : gold test DataFrame with 'aspect' and 'label_id' columns
    pred_label_ids : predicted label IDs (same order as test_df rows)

    Returns
    -------
    dict with per-aspect results and macro averages
    """
    test_df = test_df.copy()
    test_df['pred_label_id'] = pred_label_ids

    per_aspect = {}
    for asp in ASPECT_LABELS:
        mask = test_df['aspect'] == asp
        y_true = test_df.loc[mask, 'label_id'].values
        y_pred = test_df.loc[mask, 'pred_label_id'].values
        per_aspect[asp] = aspect_metrics(y_true, y_pred)

    def macro(key: str) -> float | None:
        vals = [v[key] for v in per_aspect.values() if v.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    per_aspect['macro'] = {
        'detection_f1':      macro('detection_f1'),
        'polarity_macro_f1': macro('polarity_macro_f1'),
        'combined_macro_f1': macro('combined_macro_f1'),
    }
    return per_aspect


# ── prediction functions ──────────────────────────────────────────────────────

def predict_nli(test_df: pd.DataFrame) -> np.ndarray:
    """Load cached NLI gold annotations and convert to label IDs."""
    if not NLI_CACHE.exists():
        log.error('NLI gold cache not found at %s', NLI_CACHE)
        log.error('Run compute_iaa.py first to generate NLI gold annotations.')
        raise SystemExit(1)

    cache: dict[str, dict] = {}
    with NLI_CACHE.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                cache[str(rec['id'])] = rec['aspects']

    preds = []
    for _, row in test_df.iterrows():
        rid = str(row['review_id'])
        asp = row['aspect']
        ann = cache.get(rid, {})
        polarity = ann.get(asp, 'none')
        preds.append(LABEL2ID.get(polarity, 0))
    return np.array(preds)


def predict_baseline(
    test_df: pd.DataFrame,
    model_name: Literal['lr', 'svm'],
) -> np.ndarray:
    """Load saved sklearn per-aspect pipelines and predict."""
    import pickle
    model_path = BASELINES_DIR / f'{model_name}_models.pkl'
    if not model_path.exists():
        log.error('Baseline model not found at %s', model_path)
        raise SystemExit(1)

    with model_path.open('rb') as f:
        models = pickle.load(f)

    preds = []
    for _, row in test_df.iterrows():
        asp = row['aspect']
        pipe = models[asp]
        feat = row['text_a'] + ' ' + row['text_b']
        pred = pipe.predict([feat])[0]
        preds.append(int(pred))
    return np.array(preds)


def predict_transformer(
    test_df: pd.DataFrame,
    model_path: str,
    batch_size: int = 64,
    max_length: int = 128,
) -> np.ndarray:
    """Load a saved transformer checkpoint and predict on test set."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        log.error('transformers + torch required.')
        raise SystemExit(1)

    log.info('Loading transformer from %s …', model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    all_preds = []
    rows = test_df.to_dict('records')
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        texts_a = [r['text_a'] for r in batch]
        texts_b = [r['text_b'] for r in batch]
        enc = tokenizer(
            texts_a, texts_b,
            padding=True, truncation=True,
            max_length=max_length, return_tensors='pt',
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        all_preds.extend(preds)
        if i % (batch_size * 10) == 0:
            log.info('  Predicted %d / %d', i + len(batch), len(rows))

    return np.array(all_preds)


# ── results table builder ─────────────────────────────────────────────────────

def build_comparison_table() -> pd.DataFrame:
    """Load all saved per-model metric JSONs and build a comparison DataFrame."""
    rows = []
    result_files = list(RESULTS_DIR.glob('*.json'))
    if not result_files:
        log.warning('No result files found in %s', RESULTS_DIR)
        return pd.DataFrame()

    for path in sorted(result_files):
        with path.open() as f:
            data = json.load(f)
        model_name = data.get('model_name', path.stem)
        metrics = data.get('metrics', {})
        macro = metrics.get('macro', {})
        rows.append({
            'model': model_name,
            'macro_detection_f1':  macro.get('detection_f1'),
            'macro_polarity_f1':   macro.get('polarity_macro_f1'),
            'macro_combined_f1':   macro.get('combined_macro_f1'),
        })
        for asp in ASPECT_LABELS:
            asp_metrics = metrics.get(asp, {})
            rows[-1][f'{asp}_det_f1']  = asp_metrics.get('detection_f1')
            rows[-1][f'{asp}_pol_f1']  = asp_metrics.get('polarity_macro_f1')

    return pd.DataFrame(rows)


def print_metrics_table(model_name: str, metrics: dict) -> None:
    """Pretty-print per-aspect metrics to the log."""
    log.info('\n%s — Gold Test Set Results', model_name.upper())
    log.info('%-22s  %6s  %6s  %6s  %6s  %6s', 'Aspect', 'n_h', 'det_F1', 'pol_F1', 'comb_F1', 'n_pred')
    log.info('-' * 70)
    for asp in ASPECT_LABELS:
        m = metrics.get(asp, {})
        log.info('%-22s  %6d  %6s  %6s  %6s  %6d',
                 asp,
                 m.get('n_human', 0),
                 f"{m['detection_f1']:.4f}"      if m.get('detection_f1')      is not None else '  N/A',
                 f"{m['polarity_macro_f1']:.4f}"  if m.get('polarity_macro_f1') is not None else '  N/A',
                 f"{m['combined_macro_f1']:.4f}"  if m.get('combined_macro_f1') is not None else '  N/A',
                 m.get('n_pred', 0))
    macro = metrics.get('macro', {})
    log.info('%-22s  %6s  %6s  %6s  %6s', 'MACRO',
             '',
             f"{macro['detection_f1']:.4f}"      if macro.get('detection_f1')      else '  N/A',
             f"{macro['polarity_macro_f1']:.4f}"  if macro.get('polarity_macro_f1') else '  N/A',
             f"{macro['combined_macro_f1']:.4f}"  if macro.get('combined_macro_f1') else '  N/A')


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Evaluate ABSA models on the gold test set.')
    p.add_argument('--model-type', choices=['transformer', 'baseline', 'nli', 'all'],
                   default='nli', help='Type of model to evaluate.')
    p.add_argument('--model-path', default=None,
                   help='Path to transformer checkpoint (required for --model-type transformer).')
    p.add_argument('--run-name', default=None,
                   help='Label for this run in results files.')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--comparison-table', action='store_true',
                   help='Print comparison table from all saved results and exit.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.comparison_table:
        table = build_comparison_table()
        if table.empty:
            log.info('No results yet. Run individual evaluations first.')
        else:
            log.info('\n%s', table.to_string(index=False))
            table.to_csv(RESULTS_DIR / 'comparison_table.csv', index=False)
            log.info('Comparison table saved → %s', RESULTS_DIR / 'comparison_table.csv')
        return

    # ── load gold test set ─────────────────────────────────────────────────────
    test_path = SPLITS_DIR / 'test_gold.csv'
    if not test_path.exists():
        log.error('%s not found. Run prepare_training_data.py first.', test_path)
        raise SystemExit(1)
    test_df = pd.read_csv(test_path, dtype={'review_id': str})
    log.info('Gold test set: %d pairs from %d reviews',
             len(test_df), test_df['review_id'].nunique())

    models_to_eval = []
    if args.model_type == 'nli':
        models_to_eval = [('NLI-zero-shot', 'nli', None)]
    elif args.model_type == 'baseline':
        models_to_eval = [('LR-baseline', 'baseline_lr', None),
                          ('SVM-baseline', 'baseline_svm', None)]
    elif args.model_type == 'transformer':
        if not args.model_path:
            log.error('--model-path required for --model-type transformer')
            raise SystemExit(1)
        run_name = args.run_name or Path(args.model_path).parent.name
        models_to_eval = [(run_name, 'transformer', args.model_path)]
    elif args.model_type == 'all':
        models_to_eval = [
            ('NLI-zero-shot', 'nli', None),
            ('LR-baseline', 'baseline_lr', None),
            ('SVM-baseline', 'baseline_svm', None),
        ]
        # Add any saved transformer checkpoints
        for ckpt_path in CKPT_DIR.glob('*/best_model'):
            rn = ckpt_path.parent.name
            models_to_eval.append((rn, 'transformer', str(ckpt_path)))

    for model_name, mtype, mpath in models_to_eval:
        log.info('\nEvaluating: %s', model_name)

        if mtype == 'nli':
            preds = predict_nli(test_df)
        elif mtype == 'baseline_lr':
            preds = predict_baseline(test_df, 'lr')
        elif mtype == 'baseline_svm':
            preds = predict_baseline(test_df, 'svm')
        elif mtype == 'transformer':
            preds = predict_transformer(test_df, mpath, args.batch_size, args.max_length)
        else:
            log.error('Unknown model type: %s', mtype)
            continue

        metrics = compute_all_metrics(test_df, preds)
        print_metrics_table(model_name, metrics)

        result = {
            'model_name': model_name,
            'model_type': mtype,
            'model_path': mpath,
            'metrics': metrics,
        }
        out_path = RESULTS_DIR / f'{model_name.lower().replace(" ", "_")}.json'
        with out_path.open('w') as f:
            json.dump(result, f, indent=2)
        log.info('Results saved → %s', out_path)


if __name__ == '__main__':
    main()
