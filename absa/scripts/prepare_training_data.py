"""
Prepare sentence-pair training data for ABSA model training.

Reformats silver_labels.csv and gold_annotated.csv into a sentence-pair
classification format aligned with the approach of Sun et al. (2019):
"Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing
Auxiliary Sentence" (NAACL 2019, arXiv:1903.09588).

Task formulation
----------------
For each (review, aspect_category) pair, produce one training instance:
    text_a : review title + " " + review content
    text_b : Romanian aspect description (auxiliary sentence)
    label  : "none" | "positive" | "negative" | "neutral"

This gives 10 rows per review (one per aspect category).  The "none" label
is assigned whenever the aspect is not present in the annotation.

Data splits
-----------
    Silver set (13,863 reviews)
        → train (90%, stratified on polarity_binary)
        → val   (10%, stratified on polarity_binary)

    Gold set (400 reviews, human-validated)
        → test only — never used for training or hyperparameter tuning

All splits are generated with RANDOM_SEED = 42 for reproducibility.

Outputs
-------
    absa/data/splits/train.csv      silver train set (sentence pairs)
    absa/data/splits/val.csv        silver val set (sentence pairs)
    absa/data/splits/test_gold.csv  gold test set (sentence pairs)
    absa/data/splits/label2id.json  label → integer mapping
    absa/data/splits/split_stats.json  per-split statistics

Usage
-----
    # Generate all splits (run from repo root)
    python absa/scripts/prepare_training_data.py

    # Dry run — print statistics without writing files
    python absa/scripts/prepare_training_data.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

SILVER_CSV = Path('absa/data/silver_labels.csv')
GOLD_CSV   = Path('absa/data/gold_annotated.csv')
OUT_DIR    = Path('absa/data/splits')

ASPECT_LABELS = [
    'BATERIE', 'ECRAN', 'SUNET', 'PERFORMANTA', 'CONECTIVITATE',
    'DESIGN', 'CALITATE_CONSTRUCTIE', 'PRET', 'LIVRARE', 'GENERAL',
]

# Romanian auxiliary sentences — one per aspect, used as text_b in BERT input.
# Phrased as natural descriptions to maximise semantic grounding.
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

# Label schema: 0 = none (aspect not present), 1-3 = polarity if present.
LABEL2ID: dict[str, int] = {
    'none':     0,
    'positive': 1,
    'negative': 2,
    'neutral':  3,
}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}

# ── helpers ───────────────────────────────────────────────────────────────────

def build_text(title: str, content: str) -> str:
    """Concatenate title and content into a single review text."""
    t = str(title).strip() if not pd.isna(title) else ''
    c = str(content).strip() if not pd.isna(content) else ''
    return (t + ' ' + c).strip()


def parse_aspects(aspects_str: str | float) -> dict[str, str]:
    """Parse JSON aspects string to {aspect: polarity} dict."""
    if pd.isna(aspects_str) or aspects_str in ('[]', ''):
        return {}
    try:
        pairs = json.loads(aspects_str)
        return {p['aspect']: p['polarity'] for p in pairs}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def expand_to_pairs(
    df: pd.DataFrame,
    aspect_col: str = 'aspects',
) -> pd.DataFrame:
    """
    Expand a review DataFrame into (review, aspect) sentence-pair rows.

    Parameters
    ----------
    df         : DataFrame with columns id, title, content, polarity_binary, <aspect_col>
    aspect_col : column name holding the JSON aspect-polarity list

    Returns
    -------
    DataFrame with columns:
        review_id, text_a, aspect, text_b, label, label_id,
        polarity_binary, split (if present in df)
    """
    rows = []
    for _, row in df.iterrows():
        text_a    = build_text(row.get('title', ''), row.get('content', ''))
        ann       = parse_aspects(row[aspect_col])
        pol_bin   = row.get('polarity_binary', 'positive')
        split_tag = row.get('split', 'train')

        for asp in ASPECT_LABELS:
            label = ann.get(asp, 'none')
            # Normalise any unexpected label to 'none'
            if label not in LABEL2ID:
                label = 'none'
            rows.append({
                'review_id':      str(row['id']),
                'text_a':         text_a,
                'aspect':         asp,
                'text_b':         ASPECT_AUX[asp],
                'label':          label,
                'label_id':       LABEL2ID[label],
                'polarity_binary': pol_bin,
                'split':          split_tag,
            })

    return pd.DataFrame(rows)


def log_label_dist(name: str, df: pd.DataFrame) -> None:
    """Log label and aspect distribution for a split."""
    total = len(df)
    log.info('%s: %d sentence pairs from %d reviews', name, total, df['review_id'].nunique())
    label_counts = df['label'].value_counts()
    for lbl, cnt in label_counts.items():
        log.info('  %-10s  %6d  (%.1f%%)', lbl, cnt, 100 * cnt / total)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Prepare sentence-pair training data for ABSA.')
    p.add_argument('--dry-run', action='store_true', help='Print stats without writing files.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_SEED)

    # ── load silver candidates ─────────────────────────────────────────────────
    if not SILVER_CSV.exists():
        log.error('silver_labels.csv not found at %s', SILVER_CSV)
        raise SystemExit(1)
    silver = pd.read_csv(SILVER_CSV, dtype={'id': str})
    log.info('Silver set loaded: %d reviews', len(silver))

    # ── stratified train / val split on review level ───────────────────────────
    # Stratify by polarity_binary to preserve sentiment balance.
    # The split is performed at the review level (not pair level) to prevent
    # the same review from appearing in both train and val.
    review_ids  = silver['id'].values
    strat_labels = silver['polarity_binary'].values

    train_ids, val_ids = train_test_split(
        review_ids,
        test_size=0.10,
        random_state=RANDOM_SEED,
        stratify=strat_labels,
    )
    train_ids_set = set(train_ids)
    val_ids_set   = set(val_ids)

    silver_train = silver[silver['id'].isin(train_ids_set)].reset_index(drop=True)
    silver_val   = silver[silver['id'].isin(val_ids_set)].reset_index(drop=True)
    log.info('Train reviews: %d  |  Val reviews: %d', len(silver_train), len(silver_val))

    # ── expand to sentence pairs ───────────────────────────────────────────────
    train_pairs = expand_to_pairs(silver_train, aspect_col='aspects')
    val_pairs   = expand_to_pairs(silver_val,   aspect_col='aspects')

    log_label_dist('Train', train_pairs)
    log_label_dist('Val',   val_pairs)

    # ── gold test set ──────────────────────────────────────────────────────────
    if not GOLD_CSV.exists():
        log.error('gold_annotated.csv not found at %s', GOLD_CSV)
        raise SystemExit(1)
    gold = pd.read_csv(GOLD_CSV, dtype={'id': str})
    log.info('Gold set loaded: %d reviews', len(gold))

    # Use human-validated labels (aspects column), not LLM proposals (llm_aspects)
    gold_pairs = expand_to_pairs(gold, aspect_col='aspects')
    log_label_dist('Gold test', gold_pairs)

    # ── class balance warning ──────────────────────────────────────────────────
    none_frac = (train_pairs['label'] == 'none').mean()
    if none_frac > 0.80:
        log.warning(
            'Class imbalance: "none" accounts for %.1f%% of train pairs. '
            'Use class_weight="balanced" for baselines and weighted CrossEntropyLoss '
            'for transformer fine-tuning.', none_frac * 100
        )

    neutral_frac = (train_pairs['label'] == 'neutral').mean()
    if neutral_frac < 0.02:
        log.warning(
            '"neutral" accounts for only %.2f%% of train pairs. '
            'Models may learn to never predict neutral — '
            'consider this when interpreting per-class F1.', neutral_frac * 100
        )

    # ── statistics JSON ────────────────────────────────────────────────────────
    stats = {
        'random_seed': RANDOM_SEED,
        'silver_reviews': int(len(silver)),
        'train_reviews':  int(len(silver_train)),
        'val_reviews':    int(len(silver_val)),
        'gold_reviews':   int(len(gold)),
        'aspects':        ASPECT_LABELS,
        'label2id':       LABEL2ID,
        'train': {
            'pairs': int(len(train_pairs)),
            'label_dist': train_pairs['label'].value_counts().to_dict(),
        },
        'val': {
            'pairs': int(len(val_pairs)),
            'label_dist': val_pairs['label'].value_counts().to_dict(),
        },
        'gold_test': {
            'pairs': int(len(gold_pairs)),
            'label_dist': gold_pairs['label'].value_counts().to_dict(),
        },
    }

    if args.dry_run:
        log.info('--- DRY RUN: no files written ---')
        log.info('Would write: %s/{train,val,test_gold}.csv', OUT_DIR)
        log.info('Stats: %s', json.dumps(stats, indent=2))
        return

    # ── write outputs ──────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_pairs.to_csv(OUT_DIR / 'train.csv', index=False)
    val_pairs.to_csv(OUT_DIR / 'val.csv',     index=False)
    gold_pairs.to_csv(OUT_DIR / 'test_gold.csv', index=False)

    with (OUT_DIR / 'label2id.json').open('w') as f:
        json.dump(LABEL2ID, f, indent=2)

    with (OUT_DIR / 'split_stats.json').open('w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    log.info('Written to %s/', OUT_DIR)
    log.info('  train.csv    (%d rows)', len(train_pairs))
    log.info('  val.csv      (%d rows)', len(val_pairs))
    log.info('  test_gold.csv (%d rows)', len(gold_pairs))
    log.info('  label2id.json')
    log.info('  split_stats.json')


if __name__ == '__main__':
    main()
