"""
BERTopic-based ABSA annotation for the LaRoSeDa electronics subset.

Uses BERTopic (Grootendorst, 2022) to discover latent topics in the silver
corpus, maps each discovered topic to one of the predefined aspect categories
via keyword scoring, and assigns each review the aspect label of its dominant
topic.  Polarity is derived from the existing binary star-rating label.

This annotator is entirely unsupervised with respect to the aspect schema: it
never receives the aspect labels as input during topic modelling.  Its role is
to provide an independent, data-driven annotation signal that corroborates (or
challenges) the LLM and NLI annotators, and to validate that the schema
categories are empirically grounded in the corpus vocabulary.

Pipeline role
-------------
This script is Step 3 of the three-annotator pipeline:
    1. LLM annotator     (annotate_llm.py)
    2. NLI annotator     (annotate_nli.py)
    3. BERTopic          ← this script
    4. Majority vote     (aggregate.py)

How it works
------------
1. **Embedding** — each review (title + content) is encoded with
   paraphrase-multilingual-mpnet-base-v2, a Sentence Transformers model
   trained on 50+ languages.

2. **Dimensionality reduction** — UMAP reduces the 768-dimensional embeddings
   to 5 dimensions (preserving local and global structure).

3. **Clustering** — HDBSCAN groups the reduced embeddings into density-based
   clusters.  Each cluster becomes a topic.

4. **Topic representation** — c-TF-IDF extracts the most representative words
   for each topic from the concatenated documents in that cluster.

5. **Topic → aspect mapping** — each topic's top-20 words are scored against
   per-aspect keyword sets (Romanian vocabulary).  The aspect with the most
   keyword matches wins.  Topics that match no aspect cleanly are assigned
   GENERAL.  The mapping is saved to a JSON file for inspection and
   reproducibility.

6. **Annotation** — each review is assigned the aspect of its dominant topic.
   Polarity is set from the binary star-rating label (positive for 4–5 stars,
   negative for 1–2 stars).  Reviews assigned to the BERTopic noise cluster
   (topic -1) receive GENERAL.

Outputs
-------
    absa/data/bertopic_topic_map.json      topic-id → aspect mapping (inspect this)
    absa/data/bertopic_topic_words.csv     top-20 words per topic (for thesis)
    absa/data/bertopic_annotations.jsonl   checkpoint (one JSON line per review)
    absa/data/bertopic_silver.csv          compiled silver labels

Determinism
-----------
    RANDOM_SEED = 42 is passed to UMAP and HDBSCAN.  BERTopic itself is
    stochastic in UMAP's neighbour graph; fixing the seed makes runs
    reproducible on the same machine and library version.

Usage
-----
    # Full run (downloads sentence-transformer on first run, ~420 MB)
    python absa/scripts/annotate_bertopic.py

    # Inspect discovered topics without compiling annotation CSV
    python absa/scripts/annotate_bertopic.py --topics-only

    # Override automatic topic→aspect mapping with a hand-edited JSON file
    python absa/scripts/annotate_bertopic.py --topic-map absa/data/bertopic_topic_map.json

    # Skip re-fitting; recompile from an existing checkpoint
    python absa/scripts/annotate_bertopic.py --compile-only

    # Dry run: fit on a small sample and print topic representations
    python absa/scripts/annotate_bertopic.py --dry-run --dry-run-n 2000
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

# ── constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

GOLD_CSV         = Path('absa/data/gold_set.csv')
SILVER_CAND_CSV  = Path('absa/data/silver_candidates.csv')
CHECKPOINT_JSONL = Path('absa/data/bertopic_annotations.jsonl')
TOPIC_MAP_JSON   = Path('absa/data/bertopic_topic_map.json')
TOPIC_WORDS_CSV  = Path('absa/data/bertopic_topic_words.csv')
OUTPUT_CSV       = Path('absa/data/bertopic_silver.csv')

EMBEDDING_MODEL = 'paraphrase-multilingual-mpnet-base-v2'

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

# ── per-aspect Romanian keyword sets ─────────────────────────────────────────
#
# Used to map each BERTopic cluster to the closest aspect label.
# Keywords are lowercase; matching is substring-based on the topic's top words.
# Multiple forms of the same root are included to handle Romanian inflection.

ASPECT_KEYWORDS: dict[str, list[str]] = {
    'BATERIE': [
        'baterie', 'bateria', 'autonomie', 'incarcare', 'încărcare', 'incarca',
        'încarcă', 'descarcare', 'descărcare', 'descarca', 'descarcă',
        'mah', 'fast charge', 'wireless charging', 'powerbank', 'power bank',
        'autonomia', 'tine', 'ține', 'ora', 'oră', 'ore', 'zi', 'zile',
    ],
    'ECRAN': [
        'ecran', 'display', 'rezolutie', 'rezoluție', 'luminozitate',
        'luminozitatii', 'tactil', 'touch', 'imagine', 'pixeli', 'oled',
        'amoled', 'ips', 'lcd', 'hdmi', 'inch', 'diagonal', 'diagonala',
        'diagonală', 'contrast', 'culori', 'refresh', 'fps',
    ],
    'SUNET': [
        'sunet', 'audio', 'bass', 'volum', 'claritate', 'microfon', 'casca',
        'căști', 'casti', 'difuzor', 'speaker', 'zgomot', 'anc', 'noise',
        'muzica', 'muzică', 'stereo', 'dolby', 'treble', 'frecventa',
        'frecvență', 'earphone', 'earbuds', 'inear',
    ],
    'PERFORMANTA': [
        'procesor', 'performanta', 'performanță', 'rapid', 'lent', 'ingheata',
        'îngheață', 'lag', 'sistem', 'aplicatie', 'aplicație', 'update',
        'firmware', 'gps', 'senzor', 'camera', 'foto', 'poza', 'poze',
        'video', 'procesor', 'ram', 'snapdragon', 'mediatek', 'helio',
        'benchmark', 'gaming', 'joc', 'jocuri', 'temperatura', 'overheat',
    ],
    'CONECTIVITATE': [
        'bluetooth', 'wifi', 'wi-fi', 'nfc', 'semnal', '4g', '5g', 'lte',
        'conexiune', 'conectare', 'paired', 'pereche', 'sync', 'internet',
        'retea', 'rețea', 'router', 'hotspot', 'usb', 'cablu', 'port',
    ],
    'DESIGN': [
        'design', 'aspect', 'culoare', 'forma', 'formă', 'grosime', 'greutate',
        'material', 'arata', 'arată', 'mic', 'mare', 'slim', 'subtire',
        'subțire', 'elegant', 'plastic', 'metal', 'sticla', 'sticlă',
        'ergonomic', 'dimensiune', 'rezistent', 'apa', 'apă', 'praf',
    ],
    'CALITATE_CONSTRUCTIE': [
        'calitate', 'rezistent', 'stricat', 'rupt', 'fragil', 'solid',
        'durabil', 'defect', 'defectiune', 'defecțiune', 's-a', 'sa',
        'picat', 'spart', 'crapat', 'crăpat', 'crapaturi', 'zgariat',
        'zgâriat', 'fabricatie', 'fabricație', 'garantie', 'garanție',
        'retunat', 'retur', 'schimbat', 'inlocuit', 'înlocuit',
    ],
    'PRET': [
        'pret', 'preț', 'cost', 'calitate-pret', 'calitate-preț', 'merit',
        'merita', 'merită', 'scump', 'ieftin', 'oferta', 'ofertă', 'promotie',
        'promoție', 'discount', 'reducere', 'bani', 'lei', 'euro',
        'accesibil', 'buget', 'achizitie', 'achiziție',
    ],
    'LIVRARE': [
        'livrare', 'transport', 'curier', 'ambalaj', 'cutie', 'venit',
        'ajuns', 'expediat', 'retur', 'returnare', 'customer', 'service',
        'suport', 'emag', 'amazon', 'fan courier', 'dpd', 'sameday',
        'colet', 'impachetat', 'împachetat', 'deteriorat',
    ],
}


# ── checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> dict[str, dict]:
    """Return {id: record} for all completed BERTopic annotations."""
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
    CHECKPOINT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_JSONL.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ── data loading ──────────────────────────────────────────────────────────────

def load_silver() -> pd.DataFrame:
    if not SILVER_CAND_CSV.exists():
        log.error('silver_candidates.csv not found. Run annotate_llm.py first.')
        sys.exit(1)
    df = pd.read_csv(SILVER_CAND_CSV, dtype={'id': str})
    log.info('Loaded silver candidates: %d rows', len(df))
    return df


def build_texts(df: pd.DataFrame) -> list[str]:
    """Concatenate title and content for each review."""
    texts = []
    for _, row in df.iterrows():
        title   = str(row.get('title',   '') or '')
        content = str(row.get('content', '') or '')
        texts.append(f"{title}. {content}".strip())
    return texts


# ── topic → aspect mapping ────────────────────────────────────────────────────

def score_topic_words(words: list[str]) -> str:
    """
    Given a topic's top words, return the best-matching aspect label.

    Scoring: count how many of the topic's words are substrings of (or equal to)
    any keyword in each aspect's keyword list.  The aspect with the highest
    score wins.  Ties are broken in ASPECT_LABELS order.  If no aspect scores
    above zero, return GENERAL.
    """
    scores: dict[str, int] = {a: 0 for a in ASPECT_LABELS if a != 'GENERAL'}
    words_lower = [w.lower() for w in words]

    for aspect, keywords in ASPECT_KEYWORDS.items():
        for word in words_lower:
            for kw in keywords:
                if kw in word or word in kw:
                    scores[aspect] += 1
                    break  # count each topic word at most once per aspect

    best_aspect = max(scores, key=lambda a: scores[a])
    if scores[best_aspect] == 0:
        return 'GENERAL'
    return best_aspect


def build_topic_map(
    topic_model: Any,
    nr_top_words: int = 20,
) -> tuple[dict[int, str], pd.DataFrame]:
    """
    Map each BERTopic topic ID to an aspect label.

    Returns:
        topic_map   — {topic_id: aspect_label}  (-1 → GENERAL)
        words_df    — DataFrame with columns [topic_id, words, aspect]
    """
    topic_info = topic_model.get_topic_info()
    rows = []
    topic_map: dict[int, str] = {-1: 'GENERAL'}  # noise cluster → GENERAL

    for _, row in topic_info.iterrows():
        topic_id = int(row['Topic'])
        if topic_id == -1:
            continue

        # get_topic returns list of (word, score) tuples
        word_score_pairs = topic_model.get_topic(topic_id)
        if not word_score_pairs:
            topic_map[topic_id] = 'GENERAL'
            rows.append({'topic_id': topic_id, 'words': '', 'aspect': 'GENERAL'})
            continue

        words   = [w for w, _ in word_score_pairs[:nr_top_words]]
        aspect  = score_topic_words(words)
        topic_map[topic_id] = aspect

        rows.append({
            'topic_id': topic_id,
            'count':    int(row.get('Count', 0)),
            'words':    ', '.join(words),
            'aspect':   aspect,
        })
        log.info('  Topic %3d (%4d docs) → %-22s  words: %s',
                 topic_id, int(row.get('Count', 0)), aspect, ', '.join(words[:8]))

    words_df = pd.DataFrame(rows)
    return topic_map, words_df


# ── main annotation ───────────────────────────────────────────────────────────

def run_annotation(
    df: pd.DataFrame,
    topic_map_override: Path | None,
    dry_run: bool,
    dry_run_n: int,
    topics_only: bool,
) -> None:
    """Fit BERTopic, map topics to aspects, write per-review annotations."""
    try:
        import torch
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
        from hdbscan import HDBSCAN
    except ImportError as exc:
        log.error('Required packages missing: %s', exc)
        log.error('Install with:')
        log.error('  pip install bertopic sentence-transformers umap-learn hdbscan --break-system-packages')
        sys.exit(1)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    corpus = df.head(dry_run_n) if dry_run else df
    log.info('Building texts for %d reviews…', len(corpus))
    texts = build_texts(corpus)

    # ── 1. embeddings ─────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info('Loading embedding model: %s  (device: %s)', EMBEDDING_MODEL, device)
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    log.info('Encoding %d documents…', len(texts))
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    log.info('Embeddings shape: %s', embeddings.shape)

    # ── 2. BERTopic pipeline ──────────────────────────────────────────────────
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=RANDOM_SEED,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=50,
        min_samples=10,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
    )

    log.info('Fitting BERTopic…')
    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics='auto',         # reduce to meaningful topics automatically
        top_n_words=20,
        language='multilingual',
        calculate_probabilities=False,
        verbose=True,
        seed_topic_list=None,     # unsupervised — no schema hints
    )

    topics, _ = topic_model.fit_transform(texts, embeddings)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    noise_frac = topics.count(-1) / len(topics) * 100
    log.info('Topics discovered: %d  |  Noise (-1): %.1f%%', n_topics, noise_frac)

    # ── 3. topic → aspect mapping ─────────────────────────────────────────────
    if topic_map_override and topic_map_override.exists():
        log.info('Loading topic map override from %s', topic_map_override)
        with topic_map_override.open('r', encoding='utf-8') as f:
            raw_map = json.load(f)
        topic_map: dict[int, str] = {int(k): v for k, v in raw_map.items()}
        words_df = pd.DataFrame()
    else:
        log.info('Auto-mapping topics to aspects…')
        topic_map, words_df = build_topic_map(topic_model, nr_top_words=20)

    # save topic map and word table
    TOPIC_MAP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with TOPIC_MAP_JSON.open('w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in topic_map.items()}, f,
                  ensure_ascii=False, indent=2)
    log.info('Topic map saved → %s', TOPIC_MAP_JSON)

    if not words_df.empty:
        words_df.to_csv(TOPIC_WORDS_CSV, index=False, encoding='utf-8')
        log.info('Topic word table saved → %s', TOPIC_WORDS_CSV)

    # ── print aspect coverage summary ─────────────────────────────────────────
    aspect_counts: dict[str, int] = {a: 0 for a in ASPECT_LABELS}
    for tid, aspect in topic_map.items():
        if tid == -1:
            continue
        topic_info = topic_model.get_topic_info()
        row = topic_info[topic_info['Topic'] == tid]
        cnt = int(row['Count'].values[0]) if not row.empty else 0
        aspect_counts[aspect] += cnt
    log.info('Aspect coverage from topic mapping:')
    for aspect, cnt in sorted(aspect_counts.items(), key=lambda x: -x[1]):
        log.info('  %-25s %d docs', aspect, cnt)

    if topics_only:
        log.info('--topics-only: skipping annotation write.')
        return

    # ── 4. write per-review annotations ──────────────────────────────────────
    done = load_checkpoint()
    log.info('Already checkpointed: %d', len(done))

    errors = 0
    for idx, (_, row) in enumerate(corpus.iterrows()):
        rev_id = str(row['id'])
        if rev_id in done:
            continue

        topic_id = topics[idx]
        aspect   = topic_map.get(topic_id, 'GENERAL')

        # polarity from binary star-rating label
        raw_pol = row.get('polarity_binary', 'positive')
        if isinstance(raw_pol, str):
            polarity = 'positive' if raw_pol.strip().lower() == 'positive' else 'negative'
        else:
            polarity = 'positive' if int(raw_pol) == 1 else 'negative'

        annotation = [{'aspect': aspect, 'polarity': polarity}]
        aspects_json = json.dumps(annotation, ensure_ascii=False)

        record: dict[str, Any] = {
            'id':           rev_id,
            'aspects':      annotation,
            'aspects_json': aspects_json,
            'topic_id':     topic_id,
            'model':        EMBEDDING_MODEL,
        }

        if dry_run:
            if idx < 10:
                log.info('DRY RUN id=%s  topic=%d  aspect=%s  polarity=%s',
                         rev_id, topic_id, aspect, polarity)
        else:
            append_checkpoint(record)

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(corpus):
            log.info('Progress: %d/%d (%.1f%%)  errors=%d',
                     idx + 1, len(corpus), (idx + 1) / len(corpus) * 100, errors)

    log.info('Annotation complete. errors=%d', errors)


# ── compile output CSV ────────────────────────────────────────────────────────

def compile_output(silver: pd.DataFrame) -> None:
    """Merge checkpoint records back into the silver DataFrame and save."""
    done = load_checkpoint()
    if not done:
        log.warning('No checkpoint records — nothing to compile.')
        return

    records_df = pd.DataFrame([
        {
            'id':       r['id'],
            'aspects':  r['aspects_json'],
            'topic_id': r.get('topic_id', -1),
            'model':    r.get('model', EMBEDDING_MODEL),
        }
        for r in done.values()
    ])

    silver_clean = silver.drop(columns=['aspects'], errors='ignore')
    result = silver_clean.merge(records_df, on='id', how='left')

    missing_mask = result['aspects'].isna()
    if missing_mask.any():
        log.warning('%d reviews missing BERTopic annotation — defaulting to GENERAL neutral',
                    missing_mask.sum())
        result.loc[missing_mask, 'aspects'] = '[{"aspect": "GENERAL", "polarity": "neutral"}]'

    out_cols = [c for c in [
        'id', 'split', 'index', 'title', 'content',
        'starRating', 'polarity_binary', 'aspects', 'topic_id', 'model',
    ] if c in result.columns]

    result[out_cols].to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    log.info('Saved BERTopic silver labels → %s  (%d rows)', OUTPUT_CSV, len(result))
    log.info('Stats: total=%d  annotated=%d  missing=%d',
             len(result), len(result) - missing_mask.sum(), missing_mask.sum())


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='BERTopic-based ABSA annotation for LaRoSeDa electronics reviews.'
    )
    p.add_argument(
        '--topic-map', type=Path, default=None,
        help='Path to a hand-edited topic→aspect JSON file. '
             'Overrides automatic keyword mapping.',
    )
    p.add_argument(
        '--topics-only', action='store_true',
        help='Fit BERTopic and print topic representations, but do not write annotations.',
    )
    p.add_argument(
        '--compile-only', action='store_true',
        help='Skip BERTopic fitting; compile existing checkpoint into output CSV.',
    )
    p.add_argument(
        '--dry-run', action='store_true',
        help='Fit on a small sample; print annotations without writing checkpoint.',
    )
    p.add_argument(
        '--dry-run-n', type=int, default=2000,
        help='Number of reviews for dry run (default: 2000).',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    silver = load_silver()

    if not args.compile_only:
        run_annotation(
            df=silver,
            topic_map_override=args.topic_map,
            dry_run=args.dry_run,
            dry_run_n=args.dry_run_n,
            topics_only=args.topics_only,
        )

    if not args.dry_run and not args.topics_only:
        compile_output(silver)


if __name__ == '__main__':
    main()
