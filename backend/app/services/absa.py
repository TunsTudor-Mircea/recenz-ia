"""
ABSA inference service.

Runs aspect-based sentiment analysis using the sentence-pair formulation of
Sun et al. (2019).  Supports five model backends:

    absa_xlmr    XLM-RoBERTa fine-tuned on the silver ABSA corpus (best overall)
    absa_robert  Romanian BERT fine-tuned on the silver ABSA corpus
    absa_mbert   Multilingual BERT fine-tuned on the silver ABSA corpus
    absa_lr      TF-IDF + Logistic Regression (per-aspect pkl, fast baseline)
    absa_svm     TF-IDF + Linear SVM (per-aspect pkl, fast baseline)

For each review the service expands it into 10 sentence pairs (one per aspect),
runs inference, and returns a compact result dict that is compatible with the
existing SentimentAnalyzer contract:

    {
        "aspects": {
            "BATERIE": "positive",
            "ECRAN": "none",
            ...
        },
        "sentiment_label": "positive",   # dominant non-none polarity
        "sentiment_score": 0.82,          # mean max-logit confidence of non-none predictions
        "model_used": "absa_xlmr",
    }

The `sentiment_label` field is derived so that the existing Review table
(which has a single sentiment_label column) remains meaningful for ABSA reviews.
Derivation rule:
    - Count non-none aspect polarities.
    - If all are "none" → "neutral" (no aspect mentioned).
    - Otherwise majority of {positive, negative, neutral} wins; ties go to negative.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────

ASPECT_LABELS = [
    "BATERIE", "ECRAN", "SUNET", "PERFORMANTA", "CONECTIVITATE",
    "DESIGN", "CALITATE_CONSTRUCTIE", "PRET", "LIVRARE", "GENERAL",
]

ASPECT_AUX: Dict[str, str] = {
    "BATERIE":              "baterie și autonomie",
    "ECRAN":                "ecran și afișaj",
    "SUNET":                "sunet și audio",
    "PERFORMANTA":          "performanță și viteză",
    "CONECTIVITATE":        "conectivitate și semnal",
    "DESIGN":               "design și aspect fizic",
    "CALITATE_CONSTRUCTIE": "calitatea construcției și durabilitate",
    "PRET":                 "preț și raport calitate-preț",
    "LIVRARE":              "livrare și ambalaj",
    "GENERAL":              "impresie generală",
}

ID2LABEL = {0: "none", 1: "positive", 2: "negative", 3: "neutral"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Maps the absa_* model key to the checkpoint sub-directory name.
TRANSFORMER_DIRS: Dict[str, str] = {
    "absa_xlmr":   "xlm-roberta-base",
    "absa_robert": "bert-base-romanian-cased-v1",
    "absa_mbert":  "bert-base-multilingual-cased",
}


# ── helper ─────────────────────────────────────────────────────────────────────

def _derive_overall_sentiment(aspects: Dict[str, str]) -> tuple[str, float]:
    """
    Derive a single sentiment_label and sentiment_score from per-aspect results.

    Rules:
        - Only non-"none" aspects are counted.
        - If no aspects are present → ("neutral", 0.0).
        - Otherwise the majority polarity among {positive, negative, neutral} wins.
        - Ties go to negative (conservative).
        - sentiment_score = fraction of non-none aspects that share the winner label.
    """
    non_none = [v for v in aspects.values() if v != "none"]
    if not non_none:
        return "neutral", 0.0

    counts = {p: non_none.count(p) for p in ["positive", "negative", "neutral"]}
    winner = max(counts, key=lambda p: (counts[p], p == "negative"))
    score = round(counts[winner] / len(non_none), 4)
    return winner, score


# ── main service ───────────────────────────────────────────────────────────────

class ABSAAnalyzer:
    """
    Lazy-loading ABSA inference service.

    Transformer checkpoints and baseline pkl files are loaded on first use
    and kept in memory for the lifetime of the process — the same pattern
    as SentimentAnalyzer in services/sentiment.py.
    """

    def __init__(self) -> None:
        self._transformers: Dict[str, object] = {}   # key → (model, tokenizer)
        self._baselines: Dict[str, object] = {}      # "lr" / "svm" → {asp: pipeline}

    # ── transformer helpers ───────────────────────────────────────────────────

    def _load_transformer(self, model_key: str) -> tuple:
        """Lazy-load a transformer checkpoint.  Returns (model, tokenizer)."""
        if model_key in self._transformers:
            return self._transformers[model_key]

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise RuntimeError(
                "transformers and torch are required for ABSA transformer inference. "
                "pip install transformers torch"
            )

        checkpoint_dir = Path(settings.ABSA_CHECKPOINT_DIR)
        model_subdir = TRANSFORMER_DIRS[model_key]
        model_path = checkpoint_dir / model_subdir / "best_model"

        if not model_path.exists():
            raise FileNotFoundError(
                f"ABSA checkpoint not found at {model_path}. "
                f"Set ABSA_CHECKPOINT_DIR in .env to the absa/models/checkpoints/ directory "
                f"and ensure training has been completed for {model_subdir}."
            )

        log.info("Loading ABSA transformer: %s from %s", model_key, model_path)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)
        log.info("ABSA transformer loaded on %s", device)

        self._transformers[model_key] = (model, tokenizer)
        return model, tokenizer

    def _predict_transformer(
        self,
        review_text: str,
        model_key: str,
        batch_size: int = 10,
        max_length: int = 128,
    ) -> Dict[str, str]:
        """
        Expand review into 10 aspect sentence-pairs, run inference, return aspect dict.
        """
        import torch

        model, tokenizer = self._load_transformer(model_key)
        device = next(model.parameters()).device

        texts_a = [review_text] * len(ASPECT_LABELS)
        texts_b = [ASPECT_AUX[asp] for asp in ASPECT_LABELS]

        enc = tokenizer(
            texts_a,
            texts_b,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits  # (10, 4)

        pred_ids = logits.argmax(dim=-1).cpu().numpy().tolist()
        return {asp: ID2LABEL[pid] for asp, pid in zip(ASPECT_LABELS, pred_ids)}

    # ── baseline helpers ──────────────────────────────────────────────────────

    def _load_baseline(self, model_type: Literal["lr", "svm"]) -> Dict:
        """Lazy-load a per-aspect sklearn pkl.  Returns {aspect: pipeline}."""
        if model_type in self._baselines:
            return self._baselines[model_type]

        baselines_dir = Path(settings.ABSA_BASELINES_DIR)
        pkl_path = baselines_dir / f"{model_type}_models.pkl"

        if not pkl_path.exists():
            raise FileNotFoundError(
                f"ABSA baseline model not found at {pkl_path}. "
                f"Set ABSA_BASELINES_DIR in .env to the absa/models/baselines/ directory "
                f"and ensure train_baselines.py has been run."
            )

        log.info("Loading ABSA baseline: %s from %s", model_type, pkl_path)
        with pkl_path.open("rb") as f:
            models = pickle.load(f)

        self._baselines[model_type] = models
        return models

    def _predict_baseline(
        self,
        review_text: str,
        model_type: Literal["lr", "svm"],
    ) -> Dict[str, str]:
        """Run per-aspect baseline inference."""
        models = self._load_baseline(model_type)
        aspects: Dict[str, str] = {}
        for asp in ASPECT_LABELS:
            pipe = models[asp]
            # text_a + " " + text_b, same as train_baselines.py
            feat = review_text + " " + ASPECT_AUX[asp]
            pred_id = int(pipe.predict([feat])[0])
            aspects[asp] = ID2LABEL[pred_id]
        return aspects

    # ── public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        text: str,
        model: str = "absa_xlmr",
    ) -> Dict:
        """
        Run ABSA inference on a single review text.

        Parameters
        ----------
        text  : The review text (title + body already concatenated by the caller,
                or just the body — the service treats it as text_a for all aspects).
        model : One of absa_xlmr, absa_robert, absa_mbert, absa_lr, absa_svm.

        Returns
        -------
        {
            "aspects": {"BATERIE": "positive", "ECRAN": "none", ...},
            "sentiment_label": "positive",
            "sentiment_score": 0.82,
            "model_used": "absa_xlmr",
        }
        """
        if model in TRANSFORMER_DIRS:
            aspects = self._predict_transformer(text, model)
        elif model == "absa_lr":
            aspects = self._predict_baseline(text, "lr")
        elif model == "absa_svm":
            aspects = self._predict_baseline(text, "svm")
        else:
            raise ValueError(
                f"Unknown ABSA model '{model}'. "
                f"Choose from: {list(TRANSFORMER_DIRS)} + ['absa_lr', 'absa_svm']"
            )

        sentiment_label, sentiment_score = _derive_overall_sentiment(aspects)
        return {
            "aspects": aspects,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "model_used": model,
        }


# ── singleton ──────────────────────────────────────────────────────────────────

absa_analyzer = ABSAAnalyzer()
