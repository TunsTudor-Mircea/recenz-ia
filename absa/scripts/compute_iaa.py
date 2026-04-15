"""
Inter-Annotator Agreement (IAA) computation for RecenzIA.

Computes Cohen's κ between the human-validated gold set and each automated
annotator (LLM, NLI).  BERTopic is excluded from gold-set IAA because the
fitted model was not persisted to disk; its agreement with the majority vote
on the silver set is documented separately in aggregation_report.csv.

Metrics
-------
For each annotator pair (human vs. X):
  - **Aspect detection κ** — for each aspect category A, treat the binary
    decision "does review R mention aspect A?" as a label.  κ is computed
    over all 400 × 10 = 4,000 (review, aspect) binary cells.

  - **Per-aspect detection κ** — same as above, but computed per-aspect
    across the 400 reviews.

  - **Polarity κ** — for reviews where *both* annotators agree the aspect is
    present, compute κ on the polarity label (positive / negative / neutral).

  - **Exact-match rate** — percentage of reviews where the full set of
    (aspect, polarity) pairs is identical between human and annotator.

Outputs
-------
  absa/data/iaa_report.csv    per-aspect κ for every annotator pair
  absa/data/iaa_summary.json  macro-averaged κ and exact-match rates

Usage
-----
  # Run from the repo root
  python absa/scripts/compute_iaa.py

  # Skip NLI inference (if you only want LLM IAA)
  python absa/scripts/compute_iaa.py --skip-nli

  # Force NLI to re-run even if gold NLI labels are already cached
  python absa/scripts/compute_iaa.py --force-nli
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

GOLD_CSV         = Path('absa/data/gold_annotated.csv')
NLI_GOLD_CACHE   = Path('absa/data/nli_gold_annotations.jsonl')
IAA_REPORT_CSV   = Path('absa/data/iaa_report.csv')
IAA_SUMMARY_JSON = Path('absa/data/iaa_summary.json')

NLI_MODEL       = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
NLI_THRESHOLD   = 0.5
ASPECT_LABELS   = [
    'BATERIE', 'ECRAN', 'SUNET', 'PERFORMANTA', 'CONECTIVITATE',
    'DESIGN', 'CALITATE_CONSTRUCTIE', 'PRET', 'LIVRARE', 'GENERAL',
]
POLARITY_LABELS = ['positive', 'negative', 'neutral']

# ── NLI hypotheses (same as annotate_nli.py) ─────────────────────────────────

ASPECT_HYPOTHESES: dict[str, str] = {
    'BATERIE': 'Recenzia menționează bateria, autonomia sau viteza de încărcare a produsului.',
    'ECRAN': 'Recenzia menționează ecranul, afișajul, rezoluția sau luminozitatea produsului.',
    'SUNET': 'Recenzia menționează sunetul, calitatea audio, difuzoarele sau microfonul produsului.',
    'PERFORMANTA': 'Recenzia menționează performanța, viteza, procesorul sau funcționarea aplicațiilor.',
    'CONECTIVITATE': 'Recenzia menționează conectivitatea, Bluetooth-ul, Wi-Fi-ul sau semnalul produsului.',
    'DESIGN': 'Recenzia menționează designul, aspectul, materialele, culoarea sau ergonomia produsului.',
    'CALITATE_CONSTRUCTIE': 'Recenzia menționează calitatea construcției, durabilitatea sau soliditatea produsului.',
    'PRET': 'Recenzia exprimă o judecată despre prețul sau raportul calitate-preț al produsului.',
    'LIVRARE': 'Recenzia menționează livrarea, ambalajul, curierii sau serviciul clienți.',
}

POLARITY_HYPOTHESIS_TEMPLATES = {
    'positive': 'Recenzia exprimă o opinie pozitivă despre {aspect_ro}.',
    'negative': 'Recenzia exprimă o opinie negativă despre {aspect_ro}.',
    'neutral':  'Recenzia exprimă o opinie neutră despre {aspect_ro}.',
}

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

# ── data helpers ──────────────────────────────────────────────────────────────

def parse_aspects(aspects_str: str | float) -> dict[str, str]:
    """Parse a JSON aspects string to {aspect: polarity} dict."""
    if pd.isna(aspects_str) or aspects_str == '[]' or aspects_str == '':
        return {}
    try:
        pairs = json.loads(aspects_str)
        return {p['aspect']: p['polarity'] for p in pairs}
    except (json.JSONDecodeError, KeyError, TypeError):
        log.warning('Could not parse aspects: %s', aspects_str)
        return {}


def to_binary_matrix(
    annotations: list[dict[str, str]],
    aspects: list[str] = ASPECT_LABELS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of {aspect: polarity} dicts to binary detection + polarity matrices.

    Returns
    -------
    detection : int array, shape (n, len(aspects))  — 1 if aspect mentioned
    polarity  : object array, shape (n, len(aspects)) — polarity label or None
    """
    n = len(annotations)
    detection = np.zeros((n, len(aspects)), dtype=int)
    polarity  = np.full((n, len(aspects)), None, dtype=object)
    for i, ann in enumerate(annotations):
        for j, asp in enumerate(aspects):
            if asp in ann:
                detection[i, j] = 1
                polarity[i, j]  = ann[asp]
    return detection, polarity


# ── κ computation ─────────────────────────────────────────────────────────────

def compute_kappas(
    det1: np.ndarray,
    det2: np.ndarray,
    pol1: np.ndarray,
    pol2: np.ndarray,
    aspects: list[str] = ASPECT_LABELS,
) -> list[dict]:
    """
    Compute per-aspect and aggregate Cohen's κ for detection and polarity.

    Parameters
    ----------
    det1, det2  : binary detection matrices (n × A)
    pol1, pol2  : polarity matrices (n × A, dtype=object, None where absent)
    aspects     : list of aspect labels

    Returns
    -------
    List of per-aspect dicts with keys:
        aspect, n_human, n_auto, n_both,
        detection_kappa, polarity_kappa, polarity_n
    """
    rows = []
    for j, asp in enumerate(aspects):
        n_h    = int(det1[:, j].sum())
        n_a    = int(det2[:, j].sum())
        both   = (det1[:, j] == 1) & (det2[:, j] == 1)
        n_both = int(both.sum())

        # Detection κ (binary, over all 400 reviews)
        try:
            det_k = cohen_kappa_score(det1[:, j], det2[:, j])
        except ValueError:
            det_k = None  # happens when one rater uses only one class

        # Polarity κ: only reviews where BOTH agree the aspect is present
        pol_k = None
        pol_n = 0
        if n_both >= 2:
            p1 = pol1[both, j]
            p2 = pol2[both, j]
            pol_n = int(len(p1))
            try:
                pol_k = cohen_kappa_score(p1, p2)
            except ValueError:
                pol_k = None  # only one polarity class present

        rows.append({
            'aspect':          asp,
            'n_human':         n_h,
            'n_auto':          n_a,
            'n_both_detect':   n_both,
            'detection_kappa': round(det_k, 4) if det_k is not None else None,
            'polarity_kappa':  round(pol_k, 4) if pol_k is not None else None,
            'polarity_n':      pol_n,
        })
    return rows


def exact_match_rate(ann1: list[dict], ann2: list[dict]) -> float:
    """
    Fraction of reviews where the full set of (aspect, polarity) pairs is
    identical between the two annotators.
    """
    matches = sum(1 for a, b in zip(ann1, ann2) if a == b)
    return matches / len(ann1)


def macro_kappa(rows: list[dict], key: str) -> float:
    """Mean of non-None κ values across aspects."""
    vals = [r[key] for r in rows if r[key] is not None]
    return round(float(np.mean(vals)), 4) if vals else None


# ── NLI inference on gold set ─────────────────────────────────────────────────

def nli_probs(pipe, premise: str, hypotheses: list[str]) -> np.ndarray:
    """Run NLI for one premise vs. multiple hypotheses; return entailment probs."""
    if not hypotheses:
        return np.array([])
    result = pipe(premise, hypotheses, hypothesis_template='{}', multi_label=True)
    label_to_score: dict[str, float] = dict(zip(result['labels'], result['scores']))
    return np.array([label_to_score.get(h, 0.0) for h in hypotheses])


def nli_annotate_review(pipe, premise: str, polarity_binary: int) -> dict[str, str]:
    """Annotate one review with NLI; returns {aspect: polarity} dict."""
    substantive = [a for a in ASPECT_LABELS if a != 'GENERAL']
    aspect_hyps  = [ASPECT_HYPOTHESES[a] for a in substantive]
    aspect_scores = nli_probs(pipe, premise, aspect_hyps)

    detected = [substantive[i] for i, s in enumerate(aspect_scores) if s >= NLI_THRESHOLD]

    if not detected:
        fallback_pol = 'positive' if polarity_binary == 1 else 'negative'
        return {'GENERAL': fallback_pol}

    ann = {}
    for aspect in detected:
        pol_hyps = [
            POLARITY_HYPOTHESIS_TEMPLATES[p].format(aspect_ro=ASPECT_RO[aspect])
            for p in POLARITY_LABELS
        ]
        pol_scores = nli_probs(pipe, premise, pol_hyps)
        ann[aspect] = POLARITY_LABELS[int(np.argmax(pol_scores))]
    return ann


def run_nli_on_gold(gold: pd.DataFrame, force: bool = False) -> list[dict[str, str]]:
    """
    Run NLI inference on the 400 gold reviews.

    Results are cached to NLI_GOLD_CACHE (JSONL) so subsequent runs are instant.
    Returns list of {aspect: polarity} dicts, one per row in `gold`.
    """
    # Load cache
    cache: dict[str, dict[str, str]] = {}
    if NLI_GOLD_CACHE.exists() and not force:
        with NLI_GOLD_CACHE.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    cache[rec['id']] = rec['aspects']
        log.info('NLI gold cache loaded: %d / %d reviews', len(cache), len(gold))

    missing = gold[~gold['id'].isin(cache)]
    if missing.empty:
        log.info('All gold reviews already annotated by NLI (from cache).')
    else:
        log.info('Running NLI inference on %d gold reviews …', len(missing))
        try:
            import torch
            from transformers import pipeline as hf_pipeline
        except ImportError:
            log.error('transformers + torch required. Install with:')
            log.error('  pip install transformers torch --break-system-packages')
            sys.exit(1)

        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        device = 0 if torch.cuda.is_available() else -1
        log.info('Loading NLI model: %s  (device=%s)', NLI_MODEL, 'cuda' if device == 0 else 'cpu')
        pipe = hf_pipeline(
            'zero-shot-classification',
            model=NLI_MODEL,
            device=device,
        )

        NLI_GOLD_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with NLI_GOLD_CACHE.open('a', encoding='utf-8') as f:
            for idx, row in missing.iterrows():
                premise = str(row.get('title', '') or '') + ' ' + str(row.get('content', '') or '')
                premise = premise.strip()

                raw_pol = row['polarity_binary']
                if isinstance(raw_pol, str):
                    pol_bin = 1 if raw_pol.strip().lower() == 'positive' else 0
                else:
                    pol_bin = int(raw_pol)

                ann = nli_annotate_review(pipe, premise, pol_bin)
                rec = {'id': row['id'], 'aspects': ann}
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                cache[row['id']] = ann

                if (idx % 50) == 0 or idx == len(gold) - 1:
                    log.info('  NLI gold progress: %d / %d', list(gold['id']).index(row['id']) + 1 if row['id'] in list(gold['id']) else idx + 1, len(gold))

        log.info('NLI gold inference complete.')

    # Return in gold DataFrame order
    return [cache[rid] for rid in gold['id']]


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Compute inter-annotator agreement (Cohen\'s κ) for RecenzIA.')
    p.add_argument(
        '--skip-nli', action='store_true',
        help='Skip NLI inference; report only Human vs. LLM.',
    )
    p.add_argument(
        '--force-nli', action='store_true',
        help='Force NLI re-inference even if gold cache exists.',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_SEED)

    # ── load gold set ──────────────────────────────────────────────────────────
    if not GOLD_CSV.exists():
        log.error('gold_annotated.csv not found at %s', GOLD_CSV)
        sys.exit(1)
    gold = pd.read_csv(GOLD_CSV, dtype={'id': str})
    log.info('Loaded gold set: %d rows', len(gold))

    # ── parse human annotations ────────────────────────────────────────────────
    human_anns = [parse_aspects(a) for a in gold['aspects']]
    llm_anns   = [parse_aspects(a) for a in gold['llm_aspects']]

    # ── NLI on gold ────────────────────────────────────────────────────────────
    nli_anns: list[dict] | None = None
    if not args.skip_nli:
        nli_anns = run_nli_on_gold(gold, force=args.force_nli)

    # ── exact-match rates ──────────────────────────────────────────────────────
    em_llm = exact_match_rate(human_anns, llm_anns)
    em_nli = exact_match_rate(human_anns, nli_anns) if nli_anns else None
    log.info('Exact-match rate  Human vs. LLM: %.1f%%', em_llm * 100)
    if em_nli is not None:
        log.info('Exact-match rate  Human vs. NLI: %.1f%%', em_nli * 100)

    # ── build binary matrices ─────────────────────────────────────────────────
    h_det,   h_pol   = to_binary_matrix(human_anns)
    llm_det, llm_pol = to_binary_matrix(llm_anns)
    if nli_anns:
        nli_det, nli_pol = to_binary_matrix(nli_anns)

    # ── compute κ ─────────────────────────────────────────────────────────────
    rows_llm = compute_kappas(h_det, llm_det, h_pol, llm_pol)
    rows_nli = compute_kappas(h_det, nli_det, h_pol, nli_pol) if nli_anns else None

    # ── aggregate detection κ (all cells) ─────────────────────────────────────
    agg_det_llm = cohen_kappa_score(h_det.ravel(), llm_det.ravel())
    agg_det_nli = cohen_kappa_score(h_det.ravel(), nli_det.ravel()) if nli_anns else None

    # ── log summary ───────────────────────────────────────────────────────────
    log.info('\n--- IAA Summary ---')
    log.info('Aggregate detection κ  Human vs. LLM: %.4f', agg_det_llm)
    if agg_det_nli is not None:
        log.info('Aggregate detection κ  Human vs. NLI: %.4f', agg_det_nli)

    header = f"{'Aspect':<22} | {'H_vs_LLM det.κ':>14} {'H_vs_LLM pol.κ':>14}"
    if rows_nli:
        header += f" | {'H_vs_NLI det.κ':>14} {'H_vs_NLI pol.κ':>14}"
    log.info(header)
    log.info('-' * len(header))

    for i, asp in enumerate(ASPECT_LABELS):
        r_l = rows_llm[i]
        det_l = f"{r_l['detection_kappa']:.4f}" if r_l['detection_kappa'] is not None else '  N/A  '
        pol_l = f"{r_l['polarity_kappa']:.4f}" if r_l['polarity_kappa'] is not None else '  N/A  '
        line = f"{asp:<22} | {det_l:>14} {pol_l:>14}"
        if rows_nli:
            r_n = rows_nli[i]
            det_n = f"{r_n['detection_kappa']:.4f}" if r_n['detection_kappa'] is not None else '  N/A  '
            pol_n = f"{r_n['polarity_kappa']:.4f}" if r_n['polarity_kappa'] is not None else '  N/A  '
            line += f" | {det_n:>14} {pol_n:>14}"
        log.info(line)

    # ── build combined report DataFrame ───────────────────────────────────────
    report_rows = []
    for i, r_l in enumerate(rows_llm):
        row = {
            'aspect': r_l['aspect'],
            'n_human_gold': r_l['n_human'],
            # Human vs. LLM
            'llm_n_auto':       r_l['n_auto'],
            'llm_n_both':       r_l['n_both_detect'],
            'llm_detection_k':  r_l['detection_kappa'],
            'llm_polarity_k':   r_l['polarity_kappa'],
            'llm_polarity_n':   r_l['polarity_n'],
        }
        if rows_nli:
            r_n = rows_nli[i]
            row.update({
                'nli_n_auto':       r_n['n_auto'],
                'nli_n_both':       r_n['n_both_detect'],
                'nli_detection_k':  r_n['detection_kappa'],
                'nli_polarity_k':   r_n['polarity_kappa'],
                'nli_polarity_n':   r_n['polarity_n'],
            })
        report_rows.append(row)

    report_df = pd.DataFrame(report_rows)
    IAA_REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(IAA_REPORT_CSV, index=False)
    log.info('IAA report saved → %s', IAA_REPORT_CSV)

    # ── summary JSON ──────────────────────────────────────────────────────────
    summary = {
        'n_gold_reviews': len(gold),
        'human_vs_llm': {
            'exact_match_rate':     round(em_llm, 4),
            'aggregate_detection_kappa': round(agg_det_llm, 4),
            'macro_detection_kappa': macro_kappa(rows_llm, 'detection_kappa'),
            'macro_polarity_kappa':  macro_kappa(rows_llm, 'polarity_kappa'),
        },
    }
    if nli_anns and rows_nli:
        summary['human_vs_nli'] = {
            'exact_match_rate':          round(em_nli, 4),
            'aggregate_detection_kappa': round(agg_det_nli, 4),
            'macro_detection_kappa':     macro_kappa(rows_nli, 'detection_kappa'),
            'macro_polarity_kappa':      macro_kappa(rows_nli, 'polarity_kappa'),
        }

    with IAA_SUMMARY_JSON.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info('IAA summary saved → %s', IAA_SUMMARY_JSON)


if __name__ == '__main__':
    main()
