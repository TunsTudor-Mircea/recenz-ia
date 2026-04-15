"""
LLM-based ABSA annotation for the LaRoSeDa electronics subset.

Processes 14,238 Romanian e-commerce reviews through an LLM annotator,
producing structured (aspect, polarity) labels for each review.

Pipeline role
-------------
This script is Step 1 of the three-annotator pipeline:
    1. LLM annotator  ← this script
    2. NLI annotator  (annotate_nli.py)
    3. BERTopic       (annotate_bertopic.py)
    4. Majority vote  (aggregate.py)

Outputs
-------
    absa/data/gold_set.csv           ~400 reviews held out for human validation
    absa/data/silver_candidates.csv  remaining ~13,800 reviews (unlabeled)
    absa/data/llm_annotations.jsonl  checkpoint: one JSON line per completed review
    absa/data/llm_silver.csv         final compiled silver labels (after all reviews done)

Determinism
-----------
    Gold set selection uses RANDOM_SEED = 42.
    Re-running produces the same gold/silver split and the same API calls
    (identical prompts), but API outputs are not deterministic unless
    temperature=0 is set (which this script does).

Usage
-----
    # Dry run — prints prompt for first 3 reviews, no API calls
    python absa/scripts/annotate_llm.py --dry-run

    # Full run with OpenAI (GPT-4o-mini, bulk silver labels)
    python absa/scripts/annotate_llm.py --provider openai --model gpt-4o-mini

    # Gold set only, higher-quality model
    python absa/scripts/annotate_llm.py --provider openai --model gpt-4o --gold-only

    # Local Ollama backend (e.g. llama3.1 or mistral)
    python absa/scripts/annotate_llm.py --provider ollama --model llama3.1

    # Resume interrupted run (skips already-checkpointed reviews)
    python absa/scripts/annotate_llm.py --provider openai  # just re-run same command

Environment variables
---------------------
    OPENAI_API_KEY   required for --provider openai
    OLLAMA_BASE_URL  optional, defaults to http://localhost:11434/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Optional imports — checked at runtime
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
GOLD_SIZE   = 400          # reviews held out for human validation

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
CHECKPOINT_JSONL= Path('absa/data/llm_annotations.jsonl')
OUTPUT_CSV      = Path('absa/data/llm_silver.csv')

# ── non-tech exclusion ───────────────────────────────────────────────────────

NONTEC_KEYWORDS: dict[str, list[str]] = {
    'carte':        ['carte ', 'roman ', 'autor ', 'editura', 'pagini', 'lectura', 'citit', 'volum '],
    'cosmetice':    ['crema ', 'parfum ', 'sampon', 'lotiune', 'serum', 'ruj ', 'fond de ten', 'gel de dus'],
    'imbracaminte': ['haina', 'tricou', 'geaca', 'bluza', 'rochie', 'pantalon', 'adidas', 'nike', 'ghete', 'tenisi'],
    'mobilier':     ['canapea', 'fotoliu', 'pat ', 'dulap', 'raft', 'biblioteca', 'birou '],
    'jucarii':      ['jucarie', 'lego', 'puzzle', 'papusa'],
}

# ── aspect schema ─────────────────────────────────────────────────────────────

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

# ── prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert annotator for Aspect-Based Sentiment Analysis (ABSA) of Romanian \
e-commerce product reviews. Your task is to identify all aspects mentioned in a review \
and assign a sentiment polarity to each one.

ASPECT LABELS (use exactly as written):
  BATERIE              – battery life, charging speed, power capacity
  ECRAN                – display quality, resolution, brightness, touch response
  SUNET                – audio quality, bass, volume, microphone, noise cancellation
  PERFORMANTA          – processing speed, app responsiveness, firmware, GPS/sensor accuracy
  CONECTIVITATE        – Bluetooth, Wi-Fi, NFC, signal quality, connection drops
  DESIGN               – appearance, materials, color, weight, ergonomics, water resistance
  CALITATE_CONSTRUCTIE – build durability, robustness, failure rate, product lifespan
  PRET                 – value for money (only when reviewer explicitly judges value)
  LIVRARE              – shipping speed, packaging condition, courier, customer service
  GENERAL              – overall impression with no identifiable specific aspect

POLARITY VALUES: positive | negative | neutral

RULES:
1. Return at least one annotation. Never return an empty array.
2. Use GENERAL only if NO specific aspect can be identified.
3. Annotate implicit aspects (e.g. "se descarcă în 3 ore" → BATERIE negative).
4. A review may have multiple (aspect, polarity) pairs.
5. For mixed polarity on the same aspect, choose the dominant sentiment.
6. PRET requires an explicit value judgment — do not annotate price facts.
7. LIVRARE covers shipping and packaging, not the product itself.
"""

def build_user_prompt(title: str, content: str) -> str:
    return f"Title: {title}\nReview: {content}"

# ── JSON schema for structured output ────────────────────────────────────────

RESPONSE_SCHEMA = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'absa_annotation',
        'strict': True,
        'schema': {
            'type': 'object',
            'properties': {
                'aspects': {
                    'type': 'array',
                    'description': 'List of aspect-polarity pairs found in the review.',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'aspect':   {'type': 'string', 'enum': ASPECT_LABELS},
                            'polarity': {'type': 'string', 'enum': POLARITY_LABELS},
                        },
                        'required': ['aspect', 'polarity'],
                        'additionalProperties': False,
                    },
                    'minItems': 1,
                },
            },
            'required': ['aspects'],
            'additionalProperties': False,
        },
    },
}

# Ollama models may not support strict JSON schema — fall back to json_object mode
OLLAMA_RESPONSE_FORMAT = {'type': 'json_object'}

OLLAMA_SUFFIX = (
    '\n\nRespond ONLY with a JSON object in this exact format:\n'
    '{"aspects": [{"aspect": "<LABEL>", "polarity": "<positive|negative|neutral>"}, ...]}\n'
    'Use only the aspect labels listed above.'
)

# ── data loading ─────────────────────────────────────────────────────────────

def is_nontec(row: pd.Series) -> bool:
    text = (str(row['title']) + ' ' + str(row['content'])).lower()
    return any(
        kw in text
        for kws in NONTEC_KEYWORDS.values()
        for kw in kws
    )


def load_corpus() -> pd.DataFrame:
    """Load and filter the full LaRoSeDa corpus."""
    train = pd.read_parquet(LAROSEDA_TRAIN)
    test  = pd.read_parquet(LAROSEDA_TEST)
    df = pd.concat([train, test], ignore_index=True)

    df['split']          = ['train'] * len(train) + ['test'] * len(test)
    df['polarity_binary'] = df['starRating'].map(
        lambda r: 'negative' if r in (1, 2) else 'positive'
    )
    df['id'] = df['index'].astype(str) + '_' + df['split']

    before = len(df)
    df = df[~df.apply(is_nontec, axis=1)].reset_index(drop=True)
    log.info('Excluded %d non-tech reviews → %d remaining', before - len(df), len(df))
    return df


# ── gold / silver split ───────────────────────────────────────────────────────

def build_gold_silver_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified sample of GOLD_SIZE reviews as the held-out gold test set.
    Stratifies by polarity_binary to ensure balance.
    Deterministic given RANDOM_SEED.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    neg = df[df['polarity_binary'] == 'negative']
    pos = df[df['polarity_binary'] == 'positive']

    per_class = GOLD_SIZE // 2
    # Safety cap in case one class is smaller
    per_class = min(per_class, len(neg), len(pos))

    gold_neg = neg.sample(per_class, random_state=int(rng.integers(1 << 31)))
    gold_pos = pos.sample(per_class, random_state=int(rng.integers(1 << 31)))
    gold     = pd.concat([gold_neg, gold_pos]).copy()

    silver   = df[~df['id'].isin(gold['id'])].copy()

    log.info('Gold set:   %d reviews (neg=%d pos=%d)', len(gold), per_class, per_class)
    log.info('Silver set: %d reviews', len(silver))
    return gold, silver


def save_split(gold: pd.DataFrame, silver: pd.DataFrame) -> None:
    GOLD_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ['id', 'split', 'index', 'title', 'content', 'starRating', 'polarity_binary']

    gold_out   = gold[cols].copy()
    silver_out = silver[cols].copy()

    gold_out['aspects'] = '[]'
    gold_out['notes']   = ''
    silver_out['aspects'] = '[]'

    gold_out.to_csv(GOLD_CSV, index=False, encoding='utf-8')
    silver_out.to_csv(SILVER_CAND_CSV, index=False, encoding='utf-8')
    log.info('Saved gold set   → %s', GOLD_CSV)
    log.info('Saved silver set → %s', SILVER_CAND_CSV)


# ── checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> dict[str, Any]:
    """Load already-annotated review IDs from checkpoint file."""
    done: dict[str, Any] = {}
    if not CHECKPOINT_JSONL.exists():
        return done
    with CHECKPOINT_JSONL.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                done[record['id']] = record
    log.info('Checkpoint: %d reviews already annotated', len(done))
    return done


def append_checkpoint(record: dict[str, Any]) -> None:
    """Append a single annotation record to the checkpoint file."""
    CHECKPOINT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_JSONL.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ── LLM client ───────────────────────────────────────────────────────────────

def build_client(provider: str) -> AsyncOpenAI:
    if not OPENAI_AVAILABLE:
        log.error('openai package not installed. Run: pip install openai')
        sys.exit(1)

    if provider == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            log.error('OPENAI_API_KEY environment variable not set.')
            sys.exit(1)
        return AsyncOpenAI(api_key=api_key)

    if provider == 'ollama':
        base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
        return AsyncOpenAI(base_url=base_url, api_key='ollama')

    raise ValueError(f'Unknown provider: {provider}')


async def call_llm(
    client: AsyncOpenAI,
    model: str,
    provider: str,
    review_id: str,
    title: str,
    content: str,
    max_retries: int = 5,
) -> dict[str, Any]:
    """
    Call the LLM for a single review. Returns a record dict.
    Retries with exponential backoff on transient errors.
    """
    user_prompt = build_user_prompt(title, content)

    if provider == 'ollama':
        user_prompt = user_prompt + OLLAMA_SUFFIX
        response_format = OLLAMA_RESPONSE_FORMAT
    else:
        response_format = RESPONSE_SCHEMA

    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user',   'content': user_prompt},
                ],
                response_format=response_format,
            )

            raw_json = resp.choices[0].message.content
            parsed   = json.loads(raw_json)
            aspects  = parsed.get('aspects', [])

            # Validate and sanitise
            valid = [
                a for a in aspects
                if a.get('aspect')   in ASPECT_LABELS
                and a.get('polarity') in POLARITY_LABELS
            ]
            if not valid:
                valid = [{'aspect': 'GENERAL', 'polarity': 'neutral'}]

            return {
                'id':             review_id,
                'aspects':        valid,
                'aspects_json':   json.dumps(valid, ensure_ascii=False),
                'raw_response':   raw_json,
                'model':          model,
                'finish_reason':  resp.choices[0].finish_reason,
                'prompt_tokens':  resp.usage.prompt_tokens,
                'completion_tokens': resp.usage.completion_tokens,
                'timestamp':      time.time(),
            }

        except Exception as exc:
            wait = 2 ** attempt
            log.warning(
                'Review %s attempt %d/%d failed: %s — retrying in %ds',
                review_id, attempt + 1, max_retries, exc, wait
            )
            if attempt == max_retries - 1:
                log.error('Review %s permanently failed after %d attempts', review_id, max_retries)
                return {
                    'id':           review_id,
                    'aspects':      [{'aspect': 'GENERAL', 'polarity': 'neutral'}],
                    'aspects_json': '[{"aspect": "GENERAL", "polarity": "neutral"}]',
                    'raw_response': f'ERROR: {exc}',
                    'model':        model,
                    'error':        str(exc),
                    'timestamp':    time.time(),
                }
            await asyncio.sleep(wait)

    # Should never reach here
    raise RuntimeError(f'Exhausted retries for review {review_id}')


# ── main annotation loop ──────────────────────────────────────────────────────

async def annotate(
    silver: pd.DataFrame,
    client: AsyncOpenAI,
    model: str,
    provider: str,
    concurrency: int,
    dry_run: bool,
    dry_run_n: int = 3,
) -> None:
    """Annotate all silver reviews, skipping already-checkpointed ones."""
    done = load_checkpoint()

    todo = silver[~silver['id'].isin(done)].reset_index(drop=True)
    log.info('%d reviews to annotate (%d already done)', len(todo), len(done))

    if dry_run:
        log.info('── DRY RUN: printing prompts for first %d reviews, no API calls ──', dry_run_n)
        for _, row in todo.head(dry_run_n).iterrows():
            print(f'\n{"="*70}')
            print(f'ID: {row["id"]}  |  stars: {row["starRating"]}')
            print(f'SYSTEM:\n{SYSTEM_PROMPT}')
            print(f'USER:\n{build_user_prompt(row["title"], row["content"])}')
        return

    sem = asyncio.Semaphore(concurrency)
    total     = len(todo)
    completed = 0
    errors    = 0

    async def bounded_call(row: pd.Series) -> None:
        nonlocal completed, errors
        async with sem:
            record = await call_llm(
                client, model, provider,
                row['id'], str(row['title']), str(row['content'])
            )
            append_checkpoint(record)
            completed += 1
            if 'error' in record:
                errors += 1
            if completed % 100 == 0 or completed == total:
                log.info(
                    'Progress: %d/%d (%.1f%%)  errors=%d',
                    completed, total, completed / total * 100, errors
                )

    tasks = [bounded_call(row) for _, row in todo.iterrows()]
    await asyncio.gather(*tasks)

    log.info('Annotation complete. Total=%d  errors=%d', completed, errors)


# ── compile output CSV ────────────────────────────────────────────────────────

def compile_output(silver: pd.DataFrame) -> None:
    """Merge checkpoint records back into the silver DataFrame and save."""
    done = load_checkpoint()
    if not done:
        log.warning('No checkpoint records found — nothing to compile.')
        return

    records_df = pd.DataFrame([
        {'id': r['id'], 'aspects': r['aspects_json'], 'model': r.get('model', ''), 'error': r.get('error', '')}
        for r in done.values()
    ])

    # Drop the placeholder 'aspects' column before merging to avoid pandas
    # column-name collision (aspects_x / aspects_y) caused by silver already
    # having an empty 'aspects' column set at CSV creation time.
    silver_clean = silver.drop(columns=['aspects'], errors='ignore')
    result = silver_clean.merge(records_df, on='id', how='left')

    # Fill any missing annotations (shouldn't happen if run completed)
    missing_mask = result['aspects'].isna()
    if missing_mask.any():
        log.warning('%d reviews have no annotation — defaulting to GENERAL neutral', missing_mask.sum())
        result.loc[missing_mask, 'aspects'] = '[{"aspect": "GENERAL", "polarity": "neutral"}]'

    out_cols = ['id', 'split', 'index', 'title', 'content', 'starRating', 'polarity_binary', 'aspects', 'model', 'error']
    result[out_cols].to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    log.info('Saved compiled silver labels → %s  (%d rows)', OUTPUT_CSV, len(result))

    # Quick stats
    total = len(result)
    errored = (result['error'].notna() & (result['error'] != '')).sum()
    log.info('Stats: total=%d  successfully_annotated=%d  errors=%d', total, total - errored, errored)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='LLM-based ABSA annotation for LaRoSeDa electronics reviews.'
    )
    p.add_argument('--provider',     choices=['openai', 'ollama'], default='openai',
                   help='LLM backend provider (default: openai)')
    p.add_argument('--model',        default='gpt-4o-mini',
                   help='Model name (default: gpt-4o-mini). For Ollama try: llama3.1, mistral')
    p.add_argument('--gold-only',    action='store_true',
                   help='Annotate only the gold set (for human validation pass)')
    p.add_argument('--concurrency',  type=int, default=20,
                   help='Max concurrent API requests (default: 20)')
    p.add_argument('--dry-run',      action='store_true',
                   help='Print prompts for a few reviews without making API calls')
    p.add_argument('--dry-run-n',    type=int, default=3,
                   help='Number of reviews to show in dry-run mode (default: 3)')
    p.add_argument('--compile-only', action='store_true',
                   help='Skip annotation, just compile checkpoint into output CSV')
    p.add_argument('--reset-split',  action='store_true',
                   help='Force re-sampling the gold/silver split (deletes existing split files)')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── load data ──
    log.info('Loading LaRoSeDa corpus…')
    df = load_corpus()

    # ── gold / silver split ──
    if args.reset_split:
        for f in [GOLD_CSV, SILVER_CAND_CSV]:
            if f.exists():
                f.unlink()
                log.info('Deleted %s', f)

    if GOLD_CSV.exists() and SILVER_CAND_CSV.exists():
        log.info('Loading existing gold/silver split…')
        gold   = pd.read_csv(GOLD_CSV,        encoding='utf-8')
        silver = pd.read_csv(SILVER_CAND_CSV, encoding='utf-8')
        # Re-attach full content columns if they were truncated by CSV
        for split_df, name in [(gold, 'gold'), (silver, 'silver')]:
            log.info('  %s: %d rows', name, len(split_df))
    else:
        log.info('Building gold/silver split (seed=%d)…', RANDOM_SEED)
        gold, silver = build_gold_silver_split(df)
        save_split(gold, silver)
        # Reload to ensure consistent dtypes
        gold   = pd.read_csv(GOLD_CSV,        encoding='utf-8')
        silver = pd.read_csv(SILVER_CAND_CSV, encoding='utf-8')

    # ── compile-only mode ──
    if args.compile_only:
        compile_output(silver)
        return

    # ── determine what to annotate ──
    target = gold if args.gold_only else silver
    label  = 'gold' if args.gold_only else 'silver'
    log.info('Target: %s set (%d reviews)', label, len(target))

    # ── dry run ──
    if args.dry_run:
        asyncio.run(annotate(
            target, None, args.model, args.provider,
            concurrency=1, dry_run=True, dry_run_n=args.dry_run_n
        ))
        return

    # ── build client ──
    client = build_client(args.provider)

    log.info(
        'Starting annotation: provider=%s  model=%s  concurrency=%d',
        args.provider, args.model, args.concurrency
    )

    # ── run ──
    asyncio.run(annotate(
        target, client, args.model, args.provider,
        concurrency=args.concurrency,
        dry_run=False,
    ))

    # ── compile silver output if full run ──
    if not args.gold_only:
        compile_output(silver)


if __name__ == '__main__':
    main()
