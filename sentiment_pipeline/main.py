"""
End-to-end execution.

    python -m sentiment_pipeline.run

Reads `settings.raw_data_path`, writes interim and processed artefacts,
produces figures, and prints summary diagnostics.
"""

from __future__ import annotations

import sys

import polars as pl
from loguru import logger

from .aggregation import build_panel, summarise, write_panel
from .config import ClassifierBackend, settings
from .inference import ParsBertScorer, TfidfLogRegScorer, write_scored
from .preprocessing import load_raw, preprocess, write_interim
from .visualization import (
    plot_daily_sentiment,
    plot_message_counts,
    plot_polarity_heatmap,
    plot_sentiment_distribution,
)


def _configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> "
                      "<level>{level:<7}</level> {message}")


def main() -> int:
    _configure_logging()
    logger.info("Pipeline start. Backend = {}.", settings.classifier_backend.value)

    # 1. Preprocess.
    raw = load_raw()
    pre = preprocess(raw)
    write_interim(pre)

    # 2. Score.
    if settings.classifier_backend is ClassifierBackend.PARSBERT:
        scorer = ParsBertScorer()
    else:
        scorer = TfidfLogRegScorer().load_or_fit(pre)
    scored = scorer.score(pre, text_col="text_clean")
    write_scored(scored)

    # 3. Aggregate.
    panel = build_panel(scored)
    write_panel(panel)

    # 4. Diagnostics.
    desc = summarise(panel)
    with pl.Config(tbl_rows=20, tbl_cols=12):
        logger.info("Per-ticker summary:\n{}", desc)

    # 5. Figures.
    plot_message_counts(panel)
    plot_sentiment_distribution(scored)
    plot_daily_sentiment(panel)
    plot_polarity_heatmap(panel)

    logger.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())