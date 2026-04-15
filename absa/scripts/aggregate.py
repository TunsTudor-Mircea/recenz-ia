"""
Majority vote aggregation of the three ABSA annotators.

Combines the outputs of:
    1. LLM annotator     → absa/data/llm_silver.csv
    2. NLI annotator     → absa/data/nli_silver.csv
    3. BERTopic          → absa/data/bertopic_silver.csv

into a single silver label set:
    → absa/data/silver_labels.csv

Pipeline role
-------------
This is the final step of the annotation pipeline:
    1. annotate_llm.py
    2. annotate_nli.py
    3. annotate_bertopic.py
    4. aggregate.py  ← this script

Majority vote rule
------------------
An (aspect, polarity) pair is included in the final silver label for a
review if and only if AT LEAST TWO of the three annotators produce that
exact (aspect, polarity) pair.

If no pair reaches majority agreement:
  - Fall back to the LLM annotation (highest-quality single annotator).
  - Flag the review as disagreement=True in the output.

Disagreement logging
--------------------
Per-review and per-aspect disagreement statistics are written to:
    absa/data/aggregation_report.csv

This is used to:
  a) Identify reviews that need priority human review.
  b) Identify aspect categories with systematically high disagreement
     (which may indicate schema ambiguity).

Outputs
-------
    absa/data/silver_labels.csv        final silver annotations
    absa/data/aggregation_report.csv   per-review disagreement stats

Usage
-----
    python absa/scripts/aggregate.py

    # Inspect disagreement report
    python absa/scripts/aggregate.py --report-only

    # Adjust minimum agreement threshold (default: 2 out of 3)
    python absa/scripts/aggregate.py --min-votes 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────

LLM_CSV      = Path('absa/data/llm_silver.csv')
NLI_CSV      = Path('absa/data/nli_silver.csv')
BERTOPIC_CSV = Path('absa/data/bertopic_silver.csv')

OUTPUT_CSV   = Path('absa/data/silver_labels.csv')
REPORT_CSV   = Path('absa/data/aggregation_report.csv')

ASPECT_LABELS = [
    'BATERIE', 'ECRAN', 'SUNET', 'PERFORMANTA', 'CONECTIVITATE',
    'DESIGN', 'CALITATE_CONSTRUCTIE', 'PRET', 'LIVRARE', 'GENERAL',
]
POLARITY_LABELS = ['positive', 'negative', 'neutral']

ANNOTATORS = ['llm', 'nli', 'bertopic']


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_aspects(raw: Any) -> list[dict[str, str]]:
    """
    Parse an aspects cell from any annotator CSV.

    Handles:
      - JSON string:  '[{"aspect": "BATERIE", "polarity": "negative"}]'
      - Already list: passed through
      - NaN / empty:  returns []
    """
    if pd.isna(raw) or raw == '':
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def aspects_to_pairs(aspects: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Convert aspect list to sorted list of (aspect, polarity) tuples."""
    pairs = []
    for item in aspects:
        a = item.get('aspect', '')
        p = item.get('polarity', '')
        if a in ASPECT_LABELS and p in POLARITY_LABELS:
            pairs.append((a, p))
    return pairs


def majority_vote(
    all_pairs: list[list[tuple[str, str]]],
    min_votes: int,
) -> tuple[list[dict[str, str]], bool]:
    """
    Apply majority vote across annotator outputs.

    Parameters
    ----------
    all_pairs   list of per-annotator (aspect, polarity) pair lists
    min_votes   minimum number of annotators that must agree on a pair

    Returns
    -------
    voted_aspects   list of agreed (aspect, polarity) dicts
    disagreement    True if no pair reached min_votes (fallback applied)
    """
    # Count votes per (aspect, polarity) pair across all annotators
    vote_counter: Counter = Counter()
    for pairs in all_pairs:
        # deduplicate within same annotator (shouldn't happen but be safe)
        for pair in set(pairs):
            vote_counter[pair] += 1

    agreed = [
        {'aspect': a, 'polarity': p}
        for (a, p), count in vote_counter.items()
        if count >= min_votes
    ]

    if agreed:
        # Sort for deterministic output: by aspect label order, then polarity
        order = {a: i for i, a in enumerate(ASPECT_LABELS)}
        agreed.sort(key=lambda x: (order.get(x['aspect'], 99), x['polarity']))
        return agreed, False

    return [], True  # caller will apply fallback


# ── data loading ──────────────────────────────────────────────────────────────

def load_annotator(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        log.error('Missing annotator output: %s', path)
        log.error('Run %s first.', f'annotate_{name}.py')
        sys.exit(1)
    df = pd.read_csv(path, dtype={'id': str})
    log.info('Loaded %s: %d rows', name, len(df))
    return df[['id', 'aspects']].rename(columns={'aspects': f'aspects_{name}'})


# ── main aggregation ──────────────────────────────────────────────────────────

def aggregate(min_votes: int) -> pd.DataFrame:
    """
    Merge all three annotator outputs and apply majority vote.

    Returns the full silver DataFrame with final labels and diagnostics.
    """
    # Load base metadata from LLM output (has full columns)
    if not LLM_CSV.exists():
        log.error('llm_silver.csv not found. Run annotate_llm.py first.')
        sys.exit(1)
    base = pd.read_csv(LLM_CSV, dtype={'id': str})

    llm_df      = load_annotator(LLM_CSV,      'llm')
    nli_df      = load_annotator(NLI_CSV,       'nli')
    bertopic_df = load_annotator(BERTOPIC_CSV,  'bertopic')

    # Merge all three onto base
    merged = (
        base
        .merge(llm_df,      on='id', how='left', suffixes=('', '_llm_dup'))
        .merge(nli_df,      on='id', how='left')
        .merge(bertopic_df, on='id', how='left')
    )
    # Drop duplicate aspects column from base (it's already in aspects_llm)
    if 'aspects' in merged.columns:
        merged = merged.drop(columns=['aspects'])

    log.info('Merged annotator outputs: %d rows', len(merged))

    # ── per-review majority vote ───────────────────────────────────────────────
    final_aspects   = []
    disagreements   = []
    fallback_used   = []
    vote_details    = []

    for _, row in merged.iterrows():
        llm_pairs      = aspects_to_pairs(parse_aspects(row.get('aspects_llm', '[]')))
        nli_pairs      = aspects_to_pairs(parse_aspects(row.get('aspects_nli', '[]')))
        bertopic_pairs = aspects_to_pairs(parse_aspects(row.get('aspects_bertopic', '[]')))

        voted, disagreed = majority_vote(
            [llm_pairs, nli_pairs, bertopic_pairs],
            min_votes=min_votes,
        )

        if disagreed:
            # Fallback: use LLM annotation (primary, highest quality)
            fallback = [{'aspect': a, 'polarity': p} for a, p in llm_pairs] \
                       or [{'aspect': 'GENERAL', 'polarity': 'neutral'}]
            final_aspects.append(json.dumps(fallback, ensure_ascii=False))
            fallback_used.append(True)
        else:
            final_aspects.append(json.dumps(voted, ensure_ascii=False))
            fallback_used.append(False)

        disagreements.append(disagreed)
        vote_details.append({
            'llm':      json.dumps([list(p) for p in llm_pairs]),
            'nli':      json.dumps([list(p) for p in nli_pairs]),
            'bertopic': json.dumps([list(p) for p in bertopic_pairs]),
        })

    merged['aspects']     = final_aspects
    merged['disagreement'] = disagreements
    merged['llm_fallback'] = fallback_used

    return merged, vote_details


# ── report ────────────────────────────────────────────────────────────────────

def build_report(
    result: pd.DataFrame,
    vote_details: list[dict],
    min_votes: int,
) -> pd.DataFrame:
    """
    Build per-review and per-aspect disagreement report.
    """
    total   = len(result)
    n_dis   = result['disagreement'].sum()
    n_agree = total - n_dis

    log.info('─' * 60)
    log.info('Aggregation complete')
    log.info('  Total reviews      : %d', total)
    log.info('  Full agreement     : %d (%.1f%%)', n_agree, n_agree / total * 100)
    log.info('  Disagreement (LLM fallback): %d (%.1f%%)', n_dis, n_dis / total * 100)

    # Per-aspect agreement stats
    log.info('')
    log.info('Per-aspect pair frequency in final labels:')
    all_aspects = []
    for raw in result['aspects']:
        for item in parse_aspects(raw):
            all_aspects.append(item.get('aspect'))
    aspect_counts = pd.Series(all_aspects).value_counts()
    for aspect in ASPECT_LABELS:
        cnt = aspect_counts.get(aspect, 0)
        log.info('  %-25s %5d  (%.1f%%)', aspect, cnt, cnt / total * 100)

    # Per-aspect disagreement: which aspects had the most fallback annotations?
    log.info('')
    log.info('Aspects in fallback (LLM-only) annotations:')
    fallback_mask = result['llm_fallback']
    fallback_aspects = []
    for raw in result.loc[fallback_mask, 'aspects']:
        for item in parse_aspects(raw):
            fallback_aspects.append(item.get('aspect'))
    fallback_counts = pd.Series(fallback_aspects).value_counts()
    for aspect in ASPECT_LABELS:
        cnt = fallback_counts.get(aspect, 0)
        if cnt > 0:
            log.info('  %-25s %5d', aspect, cnt)

    # Build detailed report DataFrame
    report_rows = []
    for i, (_, row) in enumerate(result.iterrows()):
        report_rows.append({
            'id':          row['id'],
            'disagreement': row['disagreement'],
            'llm_fallback': row['llm_fallback'],
            'llm_aspects': vote_details[i]['llm'],
            'nli_aspects': vote_details[i]['nli'],
            'bertopic_aspects': vote_details[i]['bertopic'],
            'final_aspects': row['aspects'],
        })

    return pd.DataFrame(report_rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Majority vote aggregation of LLM + NLI + BERTopic annotations.'
    )
    p.add_argument(
        '--min-votes', type=int, default=2,
        help='Minimum annotators that must agree on a pair (default: 2 of 3)',
    )
    p.add_argument(
        '--report-only', action='store_true',
        help='Print report from existing silver_labels.csv without re-aggregating',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.report_only:
        if not OUTPUT_CSV.exists():
            log.error('silver_labels.csv not found. Run aggregate.py first.')
            sys.exit(1)
        result = pd.read_csv(OUTPUT_CSV, dtype={'id': str})
        total   = len(result)
        n_dis   = result['disagreement'].sum()
        log.info('Total: %d  |  Disagreement: %d (%.1f%%)',
                 total, n_dis, n_dis / total * 100)
        all_aspects = []
        for raw in result['aspects']:
            for item in parse_aspects(raw):
                all_aspects.append(item.get('aspect'))
        print(pd.Series(all_aspects).value_counts().to_string())
        return

    result, vote_details = aggregate(min_votes=args.min_votes)

    report_df = build_report(result, vote_details, min_votes=args.min_votes)

    # Save outputs
    out_cols = [c for c in [
        'id', 'split', 'index', 'title', 'content',
        'starRating', 'polarity_binary', 'aspects',
        'disagreement', 'llm_fallback',
    ] if c in result.columns]

    result[out_cols].to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    log.info('Saved silver labels → %s  (%d rows)', OUTPUT_CSV, len(result))

    report_df.to_csv(REPORT_CSV, index=False, encoding='utf-8')
    log.info('Saved aggregation report → %s', REPORT_CSV)


if __name__ == '__main__':
    main()
