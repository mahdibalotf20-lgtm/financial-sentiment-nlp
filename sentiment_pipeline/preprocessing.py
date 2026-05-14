"""
Persian text preprocessing for financial message data.

The module produces three artefacts per message:

  1. `text_clean`   — normalised surface form (character unification,
                      ZWNJ handling, punctuation stripping).
  2. `tokens`       — Hazm-tokenised, lemmatised token list with
                      negation-scope tags appended to affected tokens.
  3. `n_tokens`     — token count, used for quality filtering.

All transformations are expressed as Polars expressions where possible.
Hazm operations are wrapped in `map_elements` because Hazm is not
vectorised; this is the standard compromise for Persian NLP in Polars.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import polars as pl
from hazm import Lemmatizer, Normalizer, WordTokenizer, stopwords_list
from loguru import logger

from .config import (
    NEGATION_CUES,
    NEGATION_WINDOW,
    TICKERS,
    settings,
)

# --------------------------------------------------------------------------- #
# Hazm singletons                                                             #
# --------------------------------------------------------------------------- #
# Hazm objects are stateful and moderately expensive to construct. We build
# them once per process. `lru_cache` keeps them out of module import time so
# that downstream imports of `config` do not pay the cost.

@lru_cache(maxsize=1)
def _normalizer() -> Normalizer:
    return Normalizer(
        correct_spacing=True,
        remove_diacritics=True,
        remove_specials_chars=True,
        decrease_repeated_chars=True,
        persian_style=True,
        persian_numbers=True,
        unicodedata_normalize=True,
    )


@lru_cache(maxsize=1)
def _tokenizer() -> WordTokenizer:
    return WordTokenizer(join_verb_parts=True, replace_links=True,
                         replace_ids=True, replace_emails=True,
                         replace_numbers=False, replace_hashtags=False)


@lru_cache(maxsize=1)
def _lemmatizer() -> Lemmatizer:
    return Lemmatizer()


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    # Tickers must never be dropped as stopwords.
    base = set(stopwords_list())
    base.difference_update(TICKERS)
    return frozenset(base)


# --------------------------------------------------------------------------- #
# Low-level string cleaners                                                   #
# --------------------------------------------------------------------------- #

# Unicode ranges and characters we strip outright. Emoji and pictographs are
# stripped because the downstream sentiment model was trained on text only;
# preserving them inflates the unknown-token rate.
_EMOJI_RE = re.compile(
    "["                       # noqa: E501
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002500-\U00002BEF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE,
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\S+")
_MULTI_WS_RE = re.compile(r"\s+")

# Arabic→Persian character mapping that Hazm's normaliser already covers,
# kept here only as a defensive second pass for malformed inputs.
_ARABIC_TO_PERSIAN = str.maketrans({
    "ي": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه",
    "إ": "ا", "أ": "ا", "ؤ": "و", "ئ": "ی",
    "ٱ": "ا",
})

# Punctuation we drop. We deliberately keep ZWNJ (U+200C) because Hazm
# relies on it for compound-verb segmentation.
_PUNCT_TABLE = str.maketrans(
    "",
    "",
    "".join(c for c in map(chr, range(0x10FFFF))
            if unicodedata.category(c).startswith("P") and c != "\u200c"),
)


def _clean_surface(text: str | None) -> str:
    """Deterministic surface cleaning. Returns '' for null inputs."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = _URL_RE.sub(" ", s)
    s = _MENTION_RE.sub(" ", s)
    s = _HASHTAG_RE.sub(" ", s)
    s = _EMOJI_RE.sub(" ", s)
    s = s.translate(_ARABIC_TO_PERSIAN)
    s = s.translate(_PUNCT_TABLE)
    if settings.lowercase_latin:
        s = "".join(c.lower() if c.isascii() else c for c in s)
    s = _normalizer().normalize(s)
    s = _MULTI_WS_RE.sub(" ", s).strip()
    return s


# --------------------------------------------------------------------------- #
# Tokenisation, lemmatisation, negation tagging                               #
# --------------------------------------------------------------------------- #

def _tokenize_and_lemmatize(text: str) -> list[str]:
    if not text:
        return []
    tok = _tokenizer().tokenize(text)
    lem = _lemmatizer()
    stops = _stopwords()
    out: list[str] = []
    for t in tok:
        # Hazm lemmas for verbs come as "present#past"; keep the past stem.
        lemma = lem.lemmatize(t)
        if "#" in lemma:
            lemma = lemma.split("#", 1)[0]
        if lemma and lemma not in stops and len(lemma) > 1:
            out.append(lemma)
    return out


def _tag_negation(tokens: list[str]) -> list[str]:
    """
    Append `_NEG` to tokens that fall inside the scope of a negation cue.
    Scope ends at the configured window or at a sentence-boundary token.
    This is the standard Pang & Lee (2002) treatment, adapted for Persian.
    """
    if not tokens:
        return tokens
    tagged: list[str] = []
    remaining = 0
    for tok in tokens:
        if tok in NEGATION_CUES:
            remaining = NEGATION_WINDOW
            tagged.append(tok)
            continue
        if remaining > 0:
            tagged.append(f"{tok}_NEG")
            remaining -= 1
        else:
            tagged.append(tok)
    return tagged


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def load_raw(path: Path | None = None) -> pl.DataFrame:
    """
    Read the source CSV with explicit schema and robust encoding fallback.
    The legacy file is headerless with three columns: timestamp, author,
    text. We coerce to that schema regardless of input header presence.
    """
    src = path or settings.raw_data_path
    logger.info("Reading raw messages from {}", src)

    # First attempt: assume header present.
    try:
        df = pl.read_csv(
            src,
            encoding="utf8",
            infer_schema_length=10_000,
            ignore_errors=True,
            try_parse_dates=False,
        )
    except Exception as exc:
        logger.warning("Header read failed ({}); retrying as headerless.", exc)
        df = pl.read_csv(
            src, encoding="utf8", has_header=False, ignore_errors=True,
            new_columns=[settings.timestamp_column,
                         settings.author_column,
                         settings.text_column],
        )

    # Rename positional columns if the header was generic ("0","1","2" or
    # "column_1" etc.). We match by position, not by name.
    if df.width < 3:
        raise ValueError(f"Expected ≥3 columns, got {df.width}.")
    rename_map = dict(zip(
        df.columns[:3],
        [settings.timestamp_column, settings.author_column, settings.text_column],
    ))
    df = df.rename(rename_map).select(
        settings.timestamp_column, settings.author_column, settings.text_column,
    )

    # Type coercion. Timestamps are parsed permissively; failures become null
    # and are dropped downstream.
    df = df.with_columns(
        pl.col(settings.timestamp_column)
          .str.strptime(pl.Datetime, strict=False, exact=False)
          .alias(settings.timestamp_column),
        pl.col(settings.author_column).cast(pl.Utf8),
        pl.col(settings.text_column).cast(pl.Utf8),
    )

    n0 = df.height
    df = df.drop_nulls(subset=[settings.text_column])
    logger.info("Loaded {} rows; {} retained after null-text drop.",
                n0, df.height)
    return df


def attach_ticker_flags(df: pl.DataFrame,
                        tickers: Iterable[str] = TICKERS) -> pl.DataFrame:
    """
    Add one boolean column per ticker indicating substring presence in the
    cleaned text. A message may reference multiple tickers; we keep them in
    wide form here and reshape long in `aggregation.py`.
    """
    exprs = [
        pl.col("text_clean").str.contains(re.escape(t), literal=False)
          .fill_null(False).alias(f"has_{t}")
        for t in tickers
    ]
    return df.with_columns(exprs)


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    """
    Full preprocessing pass. Returns a DataFrame with the original columns
    plus `text_clean`, `tokens`, `n_tokens`, and one `has_<ticker>` flag
    per ticker.
    """
    logger.info("Preprocessing {} messages.", df.height)

    df = df.with_columns(
        pl.col(settings.text_column)
          .map_elements(_clean_surface, return_dtype=pl.Utf8)
          .alias("text_clean")
    )

    df = df.with_columns(
        pl.col("text_clean")
          .map_elements(_tokenize_and_lemmatize,
                        return_dtype=pl.List(pl.Utf8))
          .alias("tokens_raw")
    ).with_columns(
        pl.col("tokens_raw")
          .map_elements(_tag_negation, return_dtype=pl.List(pl.Utf8))
          .alias("tokens"),
        pl.col("tokens_raw").list.len().alias("n_tokens"),
    ).drop("tokens_raw")

    # Quality filters.
    n0 = df.height
    df = df.filter(pl.col("n_tokens") >= settings.min_token_count)
    if settings.drop_duplicates:
        df = df.unique(subset=["text_clean"], keep="first")
    logger.info("Filtering removed {} rows ({} → {}).",
                n0 - df.height, n0, df.height)

    df = attach_ticker_flags(df)

    # At least one ticker must be referenced; otherwise the message is
    # outside our analytic frame.
    ticker_cols = [f"has_{t}" for t in TICKERS]
    df = df.filter(pl.any_horizontal([pl.col(c) for c in ticker_cols]))
    logger.info("Final preprocessed corpus: {} messages.", df.height)

    return df


def write_interim(df: pl.DataFrame, name: str = "messages_preprocessed.parquet") -> Path:
    """Persist intermediate output in Parquet (zstd) for downstream modules."""
    out = settings.interim_dir / name
    df.write_parquet(out, compression="zstd", statistics=True)
    logger.info("Wrote interim corpus to {} ({} rows).", out, df.height)
    return out