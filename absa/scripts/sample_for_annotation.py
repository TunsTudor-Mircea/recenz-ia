"""
Stratified sampling of LaRoSeDa reviews for ABSA annotation.

Produces a deterministic CSV of ~2,808 reviews across three consumer-electronics
domains (telefon, ceas_smart, audio_video), balanced by sentiment polarity within
each domain.  The output is ready for import into Label Studio.

Output columns:
    id                  unique identifier  (domain_<index>)
    domain              telefon | ceas_smart | audio_video
    original_index      row index from LaRoSeDa
    title               review title (Romanian)
    content             review body  (Romanian)
    star_rating         1 | 2 | 4 | 5
    sentiment_binary    negative (1-2 stars) | positive (4-5 stars)

Annotation columns (empty — filled by annotators):
    aspects             JSON list of {aspect, polarity} objects
    notes               free-text annotator notes

Usage:
    python absa/scripts/sample_for_annotation.py

Determinism guarantee:
    All random operations use RANDOM_SEED = 42.  Re-running produces the
    same CSV byte-for-byte (given the same source parquet).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

LAROSEDA_TRAIN = Path(
    "sentiment-ai-poc/data/raw/datasets--universityofbucharest--laroseda"
    "/snapshots/16d94969db622b13e2e9b5ede774397f1bfad3ca/laroseda/train/0000.parquet"
)

OUTPUT_CSV = Path("absa/data/annotation_sample.csv")
OUTPUT_STATS = Path("absa/data/annotation_sample_stats.json")

# Target counts per domain.  ceas_smart is capped at 808 (max balanced).
DOMAIN_TARGETS: dict[str, int] = {
    "telefon":    1000,
    "ceas_smart":  808,
    "audio_video": 1000,
}

# Keywords used to assign a domain label.
# First matching domain wins — order matters (more specific first).
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "ceas_smart": [
        "smartwatch", "ceas", "fitbit", "garmin", "amazfit",
        "apple watch", "bratara fitness", "bratara smart",
    ],
    "audio_video": [
        "casti", "boxe", "boxa", "televizor", " tv ", "speaker",
        "sunet", "audio", "wireless", "bluetooth", "difuzor",
        "headphone", "earphone",
    ],
    "telefon": [
        "telefon", "smartphone", "iphone", "samsung", "xiaomi",
        "huawei", "android", "ios", "galaxy",
    ],
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assign_domain(row: pd.Series) -> str:
    """Return the first matching domain label, or 'other'."""
    text = (str(row["title"]) + " " + str(row["content"])).lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return domain
    return "other"


def sentiment_label(star_rating: int) -> str:
    """Convert star rating to binary sentiment label."""
    if star_rating in (1, 2):
        return "negative"
    if star_rating in (4, 5):
        return "positive"
    raise ValueError(f"Unexpected star rating: {star_rating}")


def stratified_sample(
    df: pd.DataFrame,
    domain: str,
    target: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Sample `target` reviews from `df`, balanced between negative and positive.

    Caps at max_balanced = 2 * min(neg_count, pos_count) if target exceeds it.
    """
    neg = df[df["sentiment_binary"] == "negative"]
    pos = df[df["sentiment_binary"] == "positive"]

    per_class = target // 2
    per_class = min(per_class, len(neg), len(pos))

    neg_sample = neg.sample(per_class, random_state=int(rng.integers(1 << 31)))
    pos_sample = pos.sample(per_class, random_state=int(rng.integers(1 << 31)))

    sample = pd.concat([neg_sample, pos_sample]).copy()
    sample["domain"] = domain

    actual = len(sample)
    log.info(
        "  %s  →  %d reviews  (neg=%d  pos=%d)",
        domain, actual, per_class, per_class,
    )
    return sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    # -- Load ----------------------------------------------------------------
    log.info("Loading LaRoSeDa train split from %s", LAROSEDA_TRAIN)
    df = pd.read_parquet(LAROSEDA_TRAIN)
    log.info("Loaded %d reviews", len(df))

    # -- Domain assignment ---------------------------------------------------
    log.info("Assigning domain labels …")
    df["domain"] = df.apply(assign_domain, axis=1)
    df["sentiment_binary"] = df["starRating"].apply(sentiment_label)

    domain_counts = df["domain"].value_counts()
    log.info("Domain distribution:\n%s", domain_counts.to_string())

    # -- Stratified sampling -------------------------------------------------
    log.info("Sampling …")
    parts: list[pd.DataFrame] = []
    for domain, target in DOMAIN_TARGETS.items():
        domain_df = df[df["domain"] == domain].copy()
        if domain_df.empty:
            log.warning("No reviews found for domain '%s' — skipping", domain)
            continue
        sample = stratified_sample(domain_df, domain, target, rng)
        parts.append(sample)

    result = pd.concat(parts, ignore_index=True)

    # -- Build output --------------------------------------------------------
    result["id"] = result.apply(
        lambda r: f"{r['domain']}_{r['index']}", axis=1
    )
    result = result.rename(columns={"index": "original_index"})

    # Empty annotation columns
    result["aspects"] = "[]"   # annotators fill with JSON list
    result["notes"] = ""

    output_cols = [
        "id", "domain", "original_index",
        "title", "content",
        "star_rating", "sentiment_binary",
        "aspects", "notes",
    ]
    result = result.rename(columns={"starRating": "star_rating"})
    result = result[output_cols].sort_values(["domain", "id"]).reset_index(drop=True)

    # -- Write ---------------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info("Saved %d reviews to %s", len(result), OUTPUT_CSV)

    # -- Stats ---------------------------------------------------------------
    stats = {
        "random_seed": RANDOM_SEED,
        "total_reviews": len(result),
        "by_domain": {},
    }
    for domain in DOMAIN_TARGETS:
        sub = result[result["domain"] == domain]
        stats["by_domain"][domain] = {
            "total": len(sub),
            "negative": int((sub["sentiment_binary"] == "negative").sum()),
            "positive": int((sub["sentiment_binary"] == "positive").sum()),
            "mean_content_length": round(sub["content"].str.len().mean(), 1),
            "median_content_length": round(sub["content"].str.len().median(), 1),
        }

    OUTPUT_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    log.info("Stats written to %s", OUTPUT_STATS)

    # -- Summary -------------------------------------------------------------
    print("\n=== Sampling summary ===")
    print(f"Total reviews selected : {len(result)}")
    for domain, info in stats["by_domain"].items():
        print(
            f"  {domain:<15}  {info['total']:>4}  "
            f"(neg={info['negative']}  pos={info['positive']}  "
            f"median_len={info['median_content_length']:.0f} chars)"
        )
    print(f"\nOutput : {OUTPUT_CSV}")
    print(f"Stats  : {OUTPUT_STATS}")


if __name__ == "__main__":
    main()
