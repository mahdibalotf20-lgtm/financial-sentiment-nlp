"""
Configuration module for the Persian financial sentiment pipeline.

All runtime parameters are declared here and validated at import time.
Downstream modules import the singleton `settings` and never read paths,
constants, or model identifiers from elsewhere.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Final

from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# --------------------------------------------------------------------------- #
# Enumerations                                                                #
# --------------------------------------------------------------------------- #

class SentimentLabel(str, Enum):
    """Three-way sentiment classes used across the pipeline."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ClassifierBackend(str, Enum):
    """Selectable inference backends. See `inference.py` for implementations."""
    PARSBERT = "parsbert"           # transformer, primary
    TFIDF_LOGREG = "tfidf_logreg"   # linear baseline, sensitivity analysis


# --------------------------------------------------------------------------- #
# Static resources                                                            #
# --------------------------------------------------------------------------- #

# Tehran Stock Exchange tickers analysed in this study. The Persian glyphs
# are the canonical short names used in retail message boards.
TICKERS: Final[tuple[str, ...]] = (
    "فولاد",   # Mobarakeh Steel
    "خودرو",  # Iran Khodro
    "خساپا",  # Saipa
    "شستا",   # Social Security Investment Co.
)

# Persian negation cues. Used by the rule-based negation tagger in
# `preprocessing.py`. Scope is a small window of tokens to the right.
NEGATION_CUES: Final[frozenset[str]] = frozenset({
    "نه", "نیست", "نبود", "نمی", "نخواهد", "هرگز", "بدون", "بی",
})

NEGATION_WINDOW: Final[int] = 4

# Legacy lexica are retained only as weak supervision for the TF-IDF
# baseline. They are NOT used as the primary classifier.
POSITIVE_SEEDS: Final[frozenset[str]] = frozenset({
    "رشد", "افزایش", "خرید", "موفقیت", "صف خرید", "سبز", "قوی",
    "فرصت خرید", "خبر عالی", "خبر خوب", "مثبت", "سقف", "سقف تاریخی",
})

NEGATIVE_SEEDS: Final[frozenset[str]] = frozenset({
    "افت", "کاهش", "فروش", "صف فروش", "شکست", "قرمز", "ضعیف",
    "فرصت فروش", "خبر بد", "منفی", "ضرر", "ریزش", "اصلاح",
})


# --------------------------------------------------------------------------- #
# Settings                                                                    #
# --------------------------------------------------------------------------- #

class Settings(BaseSettings):
    """
    Pipeline configuration. Values may be overridden via environment variables
    prefixed with `SENT_` or a local `.env` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="SENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- I/O ------------------------------------------------------------- #
    raw_data_path: Path = Field(
        default=Path("data/raw/messages.csv"),
        description="Input CSV. Expected columns: timestamp, author, text.",
    )
    interim_dir: Path = Field(default=Path("data/interim"))
    output_dir: Path = Field(default=Path("data/processed"))
    figure_dir: Path = Field(default=Path("reports/figures"))

    # ---- Schema --------------------------------------------------------- #
    # Position-based to remain compatible with the legacy headerless CSV.
    text_column: str = "text"
    timestamp_column: str = "timestamp"
    author_column: str = "author"

    # ---- NLP ------------------------------------------------------------ #
    classifier_backend: ClassifierBackend = ClassifierBackend.PARSBERT
    parsbert_model: str = "HooshvareLab/bert-fa-base-uncased-sentiment-digikala"
    max_seq_length: int = 128
    inference_batch_size: int = 32
    device: str = "auto"   # "cpu", "cuda", "mps", or "auto"

    # ---- Preprocessing -------------------------------------------------- #
    min_token_count: int = 2          # drop messages shorter than this
    drop_duplicates: bool = True
    lowercase_latin: bool = True

    # ---- Reproducibility ----------------------------------------------- #
    random_seed: int = 20250101

    # --------------------------------------------------------------------- #
    @field_validator("raw_data_path")
    @classmethod
    def _check_input_exists(cls, v: Path) -> Path:
        if not v.exists():
            logger.warning(
                "Configured input file '{}' does not exist yet. "
                "This is acceptable if the pipeline will create it.", v,
            )
        return v

    def ensure_directories(self) -> None:
        """Create output directories if missing. Called by the runner."""
        for d in (self.interim_dir, self.output_dir, self.figure_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor. Cached to guarantee one instance per process."""
    s = Settings()
    s.ensure_directories()
    logger.debug("Loaded settings: {}", s.model_dump())
    return s


settings: Final[Settings] = get_settings()