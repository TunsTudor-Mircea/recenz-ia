"""
NLI-based ABSA annotation for the LaRoSeDa electronics subset.

Uses mDeBERTa-v3-base-xnli (multilingual Natural Language Inference) as a
zero-shot classifier to detect aspect mentions and assign sentiment polarity
for each review — with no task-specific fine-tuning.

Pipeline role
-------------
This script is Step 2 of the three-annotator pipeline:
    1. LLM annotator   (annotate_llm.py)
    2. NLI annotator  ← this script
    3. BERTopic        (annotate_bertopic.py)
    4. Majority vote   (aggregate.py)

How it works
------------
For each review (premise = title + content):

  1. **Aspect detection** — for each of the 9 substantive aspects, run NLI
     with a Romanian hypothesis:
         "Recenzia menționează [aspect description]."
     If the entailment score exceeds ASPECT_THRESHOLD (default 0.5), the
     aspect is flagged as mentioned.

  2. **Polarity classification** — for each flagged aspect, run three NLI
     hypotheses (positive / negative / neutral) and take the argmax:
         "Recenzia exprimă o opinie [pozitivă|negativă|neutră] despre [aspect]."

  3. **GENERAL fallback** — if no aspect clears the threshold, annotate with
     GENERAL + polarity inferred from the global star rating.

Outputs
-------
    absa/data/nli_annotations.jsonl   checkpoint (one JSON line per review)
    absa/data/nli_silver.csv          compiled silver labels

Determinism
-----------
    RANDOM_SEED = 42 is set for numpy and torch. The model itself is
    deterministic given fixed input batches; batch size does not affect
    classification results, only throughput.

Usage
-----
    # Dry run — prints scores for first 5 reviews, no file writes
    python absa/scripts/annotate_nli.py --dry-run

    # Full run on silver candidates (GPU auto-detected)
    python absa/scripts/annotate_nli.py

    # Also annotate the gold set
    python absa/scripts/annotate_nli.py --include-gold

    # Compile an already-completed checkpoint without re-running inference
    python absa/scripts/annotate_nli.py --compile-only

    # Custom thresholds / batch size
    python absa/scripts/annotate_nli.py --threshold 0.45 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

LAROSEDA_TRAIN = Path(
    'sentiment-ai-poc/data/raw/datasets--universityofbucharest--laroseda'
    '/snapshots/16d94969db622b13e2e9b5ede774397f1bfad3ca/laroseda/train/0000.parquet'
)
LAROSEDA_TEST = Path(
    'sentiment-ai-poc/data/raw/datasets--universityofbucharest--laroseda'
    '/snapshots/16d94969db622b13e2e9b5ede774397f1bfad3ca/laroseda/test/0000.parquet'
)

GOLD_CSV        = Path('absa/data/gold_set.csv')
SILVER_CAND_CSV = Path('absa/data/silver_candidates.csv')
CHECKPOINT_JSONL = Path('absa/data/nli_annotations.jsonl')
OUTPUT_CSV       = Path('absa/data/nli_silver.csv')

NLI_MODEL = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'

ASPECT_LABELS = [
    'BATERIE',
    'ECRAN',
    'SUNET',
    'PERFORMANTA',
    'CONECTIVITATE',
    'DESIGN',
    'CALITATE_CONSTRUCTIE',
    'PRET',
    'LIVRARE',
    'GENERAL',
]

POLARITY_LABELS = ['positive', 'negative', 'neutral']

# ── NLI hypothesis templates ──────────────────────────────────────────────────
#
# Each substantive aspect gets one detection hypothesis (Romanian) that
# captures the semantic field of that aspect.  GENERAL has no detection
# hypothesis — it is used only as a fallback.

ASPECT_HYPOTHESES: dict[str, str] = {
    'BATERIE': (
        'Recenzia menționează bateria, autonomia sau viteza de încărcare a produsului.'
    ),
    'ECRAN': (
        'Recenzia menționează ecranul, afișajul, rezoluția sau luminozitatea produsului.'
    ),
    'SUNET': (
        'Recenzia menționează sunetul, calitatea audio, difuzoarele sau microfonul produsului.'
    ),
    'PERFORMANTA': (
        'Recenzia menționează performanța, viteza, procesorul sau funcționarea aplicațiilor.'
    ),
    'CONECTIVITATE': (
        'Recenzia menționează conectivitatea, Bluetooth-ul, Wi-Fi-ul sau semnalul produsului.'
    ),
    'DESIGN': (
        'Recenzia menționează designul, aspectul, materialele, culoarea sau ergonomia produsului.'
    ),
    'CALITATE_CONSTRUCTIE': (
        'Recenzia menționează calitatea construcției, durabilitatea sau soliditatea produsului.'
    ),
    'PRET': (
        'Recenzia exprimă o judecată despre prețul sau raportul calitate-preț al produsului.'
    ),
    'LIVRARE': (
        'Recenzia menționează livrarea, ambalajul, curierii sau serviciul clienți.'
    ),
}

# Polarity hypotheses — {aspect_ro} will be replaced with the Romanian aspect name
POLARITY_HYPOTHESIS_TEMPLATES = {
    'positive': 'Recenzia exprimă o opinie pozitivă despre {aspect_ro}.',
    'negative': 'Recenzia exprimă o opinie negativă despre {aspect_ro}.',
    'neutral':  'Recenzia exprimă o opinie neutră despre {aspect_ro}.',
}

# Human-readable Romanian names for each aspect (used in polarity hypotheses)
ASPECT_RO: dict[str, str] = {
    'BATERIE':              'baterie și autonomie',
    'ECRAN':                'ecran și afișaj',
    'SUNET':                'sunet și audio',
    'PERFORMANTA':          'performanță și viteză',
    'CONECTIVITATE':        'conectivitate și semnal',
    'DESIGN':               'design și aspect',
    'CALITATE_CONSTRUCTIE': 'calitatea construcției',
    'PRET':                 'preț și raport calitate-preț',
    'LIVRARE':              'livrare și ambalaj',
    'GENERAL':              'produs în general',
}

# ── checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> dict[str, dict]:
    """Return {id: record} for all completed NLI annotations."""
    done: dict[str, dict] = {}
    if not CHECKPOINT_JSONL.exists():
        return done
    with CHECKPOINT_JSONL.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec['id']] = rec
            except json.JSONDecodeError:
                log.warning('Skipping malformed checkpoint line: %.80s', line)
    return done


def append_checkpoint(record: dict) -> None:
    """Append one completed annotation record to the JSONL checkpoint."""
    CHECKPOINT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_JSONL.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ── data loading ──────────────────────────────────────────────────────────────

def load_silver() -> pd.DataFrame:
    """Load silver candidates CSV (created by annotate_llm.py --gold-only / save_split)."""
    if not SILVER_CAND_CSV.exists():
        log.error('silver_candidates.csv not found at %s', SILVER_CAND_CSV)
        log.error('Run annotate_llm.py first to generate the data split.')
        sys.exit(1)
    df = pd.read_csv(SILVER_CAND_CSV, dtype={'id': str})
    log.info('Loaded silver candidates: %d rows', len(df))
    return df


def load_gold() -> pd.DataFrame:
    """Load gold set CSV."""
    if not GOLD_CSV.exists():
        log.error('gold_set.csv not found at %s', GOLD_CSV)
        sys.exit(1)
    df = pd.read_csv(GOLD_CSV, dtype={'id': str})
    log.info('Loaded gold set: %d rows', len(df))
    return df


# ── NLI inference helpers ─────────────────────────────────────────────────────

def softmax(arr: np.ndarray) -> np.ndarray:
    e = np.exp(arr - arr.max())
    return e / e.sum()


def nli_probs(
    pipe: Any,
    premise: str,
    hypotheses: list[str],
) -> np.ndarray:
    """
    Run zero-shot NLI for a single premise against multiple hypotheses.

    Returns an array of shape (len(hypotheses),) with entailment probabilities
    in the same order as the input hypotheses list.

    Uses multi_label=True so each hypothesis is scored independently (sigmoid
    over [entailment, contradiction]) rather than softmax across all candidates.

    hypothesis_template="{}" tells the pipeline to use the hypothesis text
    as-is instead of wrapping it in the default English template
    ("This example is {}."), which would corrupt our Romanian sentences.
    """
    if not hypotheses:
        return np.array([])

    result = pipe(
        premise,
        hypotheses,
        hypothesis_template="{}",
        multi_label=True,
    )
    # result: {'sequence': ..., 'labels': [...], 'scores': [...]}
    # Labels are sorted by score descending — map back to original order.
    label_to_score: dict[str, float] = dict(zip(result['labels'], result['scores']))
    return np.array([label_to_score.get(h, 0.0) for h in hypotheses])


def annotate_review(
    pipe: Any,
    premise: str,
    aspect_threshold: float,
    polarity_binary: int,
) -> list[dict[str, str]]:
    """
    Classify one review into a list of {aspect, polarity} dicts.

    Steps:
      1. Score all 9 substantive aspect hypotheses.
      2. Keep aspects whose entailment score ≥ aspect_threshold.
      3. For each kept aspect, score 3 polarity hypotheses and take argmax.
      4. If no aspect passes threshold → GENERAL + star-rating polarity fallback.
    """
    substantive = [a for a in ASPECT_LABELS if a != 'GENERAL']

    # Step 1: detect aspects
    aspect_hyps = [ASPECT_HYPOTHESES[a] for a in substantive]
    aspect_scores = nli_probs(pipe, premise, aspect_hyps)

    detected = [
        substantive[i]
        for i, score in enumerate(aspect_scores)
        if score >= aspect_threshold
    ]

    # Step 4 fallback
    if not detected:
        fallback_polarity = 'positive' if polarity_binary == 1 else 'negative'
        return [{'aspect': 'GENERAL', 'polarity': fallback_polarity}]

    # Step 3: polarity per detected aspect
    annotations = []
    for aspect in detected:
        pol_hyps = [
            POLARITY_HYPOTHESIS_TEMPLATES[p].format(aspect_ro=ASPECT_RO[aspect])
            for p in POLARITY_LABELS
        ]
        pol_scores = nli_probs(pipe, premise, pol_hyps)
        best_polarity = POLARITY_LABELS[int(np.argmax(pol_scores))]
        annotations.append({'aspect': aspect, 'polarity': best_polarity})

    return annotations


# ── main annotation loop ──────────────────────────────────────────────────────

def run_annotation(
    df: pd.DataFrame,
    aspect_threshold: float,
    batch_size: int,
    dry_run: bool,
    dry_run_n: int,
) -> None:
    """Iterate over reviews and write NLI annotations to the checkpoint file."""
    try:
        import torch
        from transformers import pipeline as hf_pipeline
    except ImportError:
        log.error('transformers and torch are required. Install with:')
        log.error('  pip install transformers torch --break-system-packages')
        sys.exit(1)

    # Fix seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    done = load_checkpoint()
    todo = df[~df['id'].isin(done)].reset_index(drop=True)
    log.info('Already done: %d  |  Remaining: %d', len(done), len(todo))

    if todo.empty:
        log.info('All reviews already annotated.')
        return

    if dry_run:
        todo = todo.head(dry_run_n)
        log.info('Dry run — processing %d reviews only', len(todo))

    # ── load model ────────────────────────────────────────────────────────────
    device = 0 if torch.cuda.is_available() else -1
    device_name = 'CUDA' if device == 0 else 'CPU'
    log.info('Loading NLI model: %s  (device: %s)', NLI_MODEL, device_name)

    pipe = hf_pipeline(
        task='zero-shot-classification',
        model=NLI_MODEL,
        device=device,
        # Batch size for the underlying forward passes (not review batching)
        batch_size=batch_size,
    )
    log.info('Model loaded. Starting annotation…')

    # ── annotation loop ───────────────────────────────────────────────────────
    total = len(todo)
    errors = 0

    for i, row in todo.iterrows():
        rev_id    = str(row['id'])
        title     = str(row.get('title', '') or '')
        content   = str(row.get('content', '') or '')
        # polarity_binary may be stored as int (1/0) or string ('positive'/'negative')
        raw_pol   = row.get('polarity_binary', 1)
        if isinstance(raw_pol, str):
            pol_bin = 1 if raw_pol.strip().lower() == 'positive' else 0
        else:
            pol_bin = int(raw_pol)
        premise   = f"Titlu: {title}\nRecenzie: {content}".strip()

        try:
            aspects = annotate_review(pipe, premise, aspect_threshold, pol_bin)
        except Exception as exc:
            log.error('id=%s  error=%s', rev_id, exc)
            aspects = [{'aspect': 'GENERAL', 'polarity': 'neutral'}]
            errors += 1

        aspects_json = json.dumps(aspects, ensure_ascii=False)

        record: dict[str, Any] = {
            'id':           rev_id,
            'aspects':      aspects,
            'aspects_json': aspects_json,
            'model':        NLI_MODEL,
            'threshold':    aspect_threshold,
        }

        if dry_run:
            log.info('DRY RUN id=%s  aspects=%s', rev_id, aspects_json)
        else:
            append_checkpoint(record)

        completed = i - todo.index[0] + 1
        if completed % 200 == 0 or completed == total:
            log.info(
                'Progress: %d/%d (%.1f%%)  errors=%d',
                completed, total, completed / total * 100, errors,
            )

    log.info('Annotation complete. Total=%d  errors=%d', total, errors)


# ── compile output CSV ────────────────────────────────────────────────────────

def compile_output(silver: pd.DataFrame) -> None:
    """Merge checkpoint records back into the silver DataFrame and save."""
    done = load_checkpoint()
    if not done:
        log.warning('No checkpoint records found — nothing to compile.')
        return

    records_df = pd.DataFrame([
        {
            'id':      r['id'],
            'aspects': r['aspects_json'],
            'model':   r.get('model', NLI_MODEL),
        }
        for r in done.values()
    ])

    # Drop the placeholder 'aspects' column from silver to avoid pandas
    # column-name collision (aspects_x / aspects_y) after merge.
    silver_clean = silver.drop(columns=['aspects'], errors='ignore')
    result = silver_clean.merge(records_df, on='id', how='left')

    missing_mask = result['aspects'].isna()
    if missing_mask.any():
        log.warning(
            '%d reviews have no NLI annotation — defaulting to GENERAL neutral',
            missing_mask.sum(),
        )
        result.loc[missing_mask, 'aspects'] = '[{"aspect": "GENERAL", "polarity": "neutral"}]'

    out_cols = [
        'id', 'split', 'index', 'title', 'content',
        'starRating', 'polarity_binary', 'aspects', 'model',
    ]
    # Keep only columns that actually exist (gold set has fewer columns)
    out_cols = [c for c in out_cols if c in result.columns]
    result[out_cols].to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    log.info('Saved NLI silver labels → %s  (%d rows)', OUTPUT_CSV, len(result))

    total = len(result)
    log.info('Stats: total=%d  annotated=%d  missing=%d',
             total, total - missing_mask.sum(), missing_mask.sum())


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='NLI-based ABSA annotation for LaRoSeDa electronics reviews.'
    )
    p.add_argument(
        '--include-gold', action='store_true',
        help='Also annotate the 400-review gold set (in addition to silver candidates)',
    )
    p.add_argument(
        '--threshold', type=float, default=0.5,
        help='Entailment probability threshold for aspect detection (default: 0.5)',
    )
    p.add_argument(
        '--batch-size', type=int, default=16,
        help='Tokenizer batch size for the NLI forward pass (default: 16)',
    )
    p.add_argument(
        '--compile-only', action='store_true',
        help='Skip inference; just compile checkpoint into the output CSV',
    )
    p.add_argument(
        '--dry-run', action='store_true',
        help='Run inference on a small sample without writing checkpoint',
    )
    p.add_argument(
        '--dry-run-n', type=int, default=5,
        help='Number of reviews to process in dry-run mode (default: 5)',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    silver = load_silver()

    if args.include_gold:
        gold = load_gold()
        corpus = pd.concat([silver, gold], ignore_index=True)
        log.info('Processing silver + gold: %d reviews total', len(corpus))
    else:
        corpus = silver

    if not args.compile_only:
        run_annotation(
            df=corpus,
            aspect_threshold=args.threshold,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            dry_run_n=args.dry_run_n,
        )

    if not args.dry_run:
        compile_output(silver)


if __name__ == '__main__':
    main()
