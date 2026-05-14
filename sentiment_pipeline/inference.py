"""
Sentiment inference.

Two backends are exposed behind a common interface:

  * `ParsBertScorer`  — HooshvareLab ParsBERT fine-tuned for Persian
                        sentiment. Primary classifier.
  * `TfidfLogRegScorer` — TF-IDF over Hazm lemmas + L2-regularised
                        logistic regression, trained on weak labels
                        derived from the seed lexica. Baseline used
                        for sensitivity analysis.

Both return a Polars DataFrame with columns:

  * `sent_score`  ∈ [-1, 1]   signed sentiment intensity
  * `sent_label`  ∈ {positive, negative, neutral}
  * `sent_conf`   ∈ [0, 1]    classifier confidence in the predicted class

The signed score is constructed as  P(pos) − P(neg)  so the two backends
are directly comparable. Neutral is assigned when |score| falls below
`neutral_band`, a hyperparameter exposed on the scorer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import torch
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import (
    ClassifierBackend,
    NEGATIVE_SEEDS,
    POSITIVE_SEEDS,
    SentimentLabel,
    settings,
)


# --------------------------------------------------------------------------- #
# Device resolution                                                           #
# --------------------------------------------------------------------------- #

def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Base class                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class ScorerConfig:
    neutral_band: float = 0.15      # |score| ≤ band → neutral
    batch_size: int = settings.inference_batch_size


class Scorer(ABC):
    """Abstract scorer. Concrete backends implement `_score_batch`."""

    def __init__(self, cfg: ScorerConfig | None = None) -> None:
        self.cfg = cfg or ScorerConfig()

    @abstractmethod
    def _score_batch(self, texts: list[str]) -> np.ndarray:
        """Return an (n, 2) array of [P(neg), P(pos)] for each input."""

    def score(self, df: pl.DataFrame, text_col: str = "text_clean") -> pl.DataFrame:
        if text_col not in df.columns:
            raise KeyError(f"Column '{text_col}' missing from input.")
        texts = df[text_col].to_list()
        probs = self._batched_predict(texts)
        signed = probs[:, 1] - probs[:, 0]
        conf = probs.max(axis=1)
        labels = np.where(
            np.abs(signed) <= self.cfg.neutral_band,
            SentimentLabel.NEUTRAL.value,
            np.where(signed > 0,
                     SentimentLabel.POSITIVE.value,
                     SentimentLabel.NEGATIVE.value),
        )
        return df.with_columns(
            pl.Series("sent_score", signed.astype(np.float32)),
            pl.Series("sent_label", labels),
            pl.Series("sent_conf", conf.astype(np.float32)),
        )

    def _batched_predict(self, texts: list[str]) -> np.ndarray:
        bs = self.cfg.batch_size
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), bs):
            chunks.append(self._score_batch(texts[i:i + bs]))
        return np.vstack(chunks) if chunks else np.empty((0, 2), dtype=np.float32)


# --------------------------------------------------------------------------- #
# ParsBERT backend                                                            #
# --------------------------------------------------------------------------- #

class ParsBertScorer(Scorer):
    """
    Transformer scorer. The HooshvareLab Digikala model emits three classes
    (negative / neutral / positive in their canonical label order); we
    collapse neutral mass into the signed-score construction by treating
    P(neg) and P(pos) as the two relevant probabilities and renormalising.
    """

    def __init__(self, cfg: ScorerConfig | None = None,
                 model_name: str | None = None) -> None:
        super().__init__(cfg)
        self.device = _resolve_device(settings.device)
        model_id = model_name or settings.parsbert_model
        logger.info("Loading ParsBERT model '{}' on {}.", model_id, self.device)

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_id, use_fast=True,
        )
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            model_id
        ).to(self.device)
        self.model.eval()

        # Resolve the index of each polarity from the model's id2label map.
        # We match on substring to tolerate variants ("LABEL_0" etc.).
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self._pos_idx = self._find_label(id2label, ("pos", "مثبت", "happy"))
        self._neg_idx = self._find_label(id2label, ("neg", "منفی", "sad"))
        logger.debug("Model labels resolved: pos={}, neg={}",
                     self._pos_idx, self._neg_idx)

    @staticmethod
    def _find_label(id2label: dict[int, str], needles: Iterable[str]) -> int:
        for idx, lbl in id2label.items():
            if any(n in lbl for n in needles):
                return idx
        raise ValueError(f"Could not resolve a label among {needles} "
                         f"in {id2label}.")

    @torch.inference_mode()
    def _score_batch(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.max_seq_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # Renormalise over (neg, pos); neutral mass is shed.
        p_neg = probs[:, self._neg_idx]
        p_pos = probs[:, self._pos_idx]
        denom = np.clip(p_neg + p_pos, 1e-9, None)
        return np.stack([p_neg / denom, p_pos / denom], axis=1)


# --------------------------------------------------------------------------- #
# TF-IDF + Logistic Regression baseline                                       #
# --------------------------------------------------------------------------- #

class TfidfLogRegScorer(Scorer):
    """
    Linear baseline. Trained on weak labels obtained from the seed lexica:
    a message is provisionally positive if it contains any positive seed
    and no negative seed, negative under the symmetric condition, and
    discarded for training otherwise. This is standard distant supervision
    (Mintz et al., 2009; Read, 2005) and serves only as a sanity check.

    The trained pipeline is persisted to `interim_dir` and reused if found.
    """

    MODEL_FILENAME = "tfidf_logreg.joblib"

    def __init__(self, cfg: ScorerConfig | None = None) -> None:
        super().__init__(cfg)
        self.pipeline: Pipeline | None = None
        self._model_path: Path = settings.interim_dir / self.MODEL_FILENAME

    # --- training utilities ------------------------------------------------- #
    @staticmethod
    def _weak_label(tokens: list[str]) -> int | None:
        """Return 1 (pos), 0 (neg), or None (drop)."""
        toks = {t.split("_NEG")[0] for t in tokens}  # strip negation tags
        has_pos = bool(toks & POSITIVE_SEEDS)
        has_neg = bool(toks & NEGATIVE_SEEDS)
        if has_pos and not has_neg:
            return 1
        if has_neg and not has_pos:
            return 0
        return None

    def fit(self, df: pl.DataFrame, token_col: str = "tokens") -> "TfidfLogRegScorer":
        logger.info("Training TF-IDF/LogReg baseline on weak labels.")
        token_lists = df[token_col].to_list()
        labels = [self._weak_label(t) for t in token_lists]

        X_text: list[str] = []
        y: list[int] = []
        for toks, lab in zip(token_lists, labels):
            if lab is None:
                continue
            X_text.append(" ".join(toks))
            y.append(lab)

        if len(set(y)) < 2:
            raise RuntimeError(
                "Weak labelling produced a single class; "
                "the seed lexica are too narrow for this corpus."
            )

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.95,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=settings.random_seed,
            )),
        ])
        self.pipeline.fit(X_text, y)
        logger.info("Baseline trained on {} weakly-labelled messages "
                    "({} positive, {} negative).",
                    len(y), sum(y), len(y) - sum(y))

        import joblib
        joblib.dump(self.pipeline, self._model_path)
        return self

    def load_or_fit(self, df: pl.DataFrame, token_col: str = "tokens") -> "TfidfLogRegScorer":
        if self._model_path.exists():
            import joblib
            self.pipeline = joblib.load(self._model_path)
            logger.info("Loaded cached baseline from {}.", self._model_path)
            return self
        return self.fit(df, token_col=token_col)

    # --- scoring ----------------------------------------------------------- #
    def _score_batch(self, texts: list[str]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("TfidfLogRegScorer is not fitted; "
                               "call `fit` or `load_or_fit` first.")
        proba = self.pipeline.predict_proba(texts)
        # sklearn orders columns by sorted class labels: [0=neg, 1=pos].
        return proba.astype(np.float32)


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #

def build_scorer(df_for_baseline: pl.DataFrame | None = None) -> Scorer:
    """
    Return a configured scorer per `settings.classifier_backend`.

    For the TF-IDF baseline, `df_for_baseline` is required on first call to
    train the model from weak labels. Subsequent calls load from disk.
    """
    backend = settings.classifier_backend
    if backend is ClassifierBackend.PARSBERT:
        return ParsBertScorer()
    if backend is ClassifierBackend.TFIDF_LOGREG:
        s = TfidfLogRegScorer()
        if df_for_baseline is None and not s._model_path.exists():
            raise ValueError("Baseline backend requires a training DataFrame "
                             "on first invocation.")
        return s.load_or_fit(df_for_baseline) if df_for_baseline is not None \
            else s.load_or_fit(pl.DataFrame())   # cached path
    raise ValueError(f"Unknown backend: {backend}")


def write_scored(df: pl.DataFrame,
                 name: str = "messages_scored.parquet") -> Path:
    out = settings.output_dir / name
    df.write_parquet(out, compression="zstd", statistics=True)
    logger.info("Wrote scored corpus to {} ({} rows).", out, df.height)
    return out