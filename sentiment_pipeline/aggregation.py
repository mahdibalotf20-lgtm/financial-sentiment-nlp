"""
Econometric aggregation.

The scored message corpus is reshaped from wide (one row per message, one
boolean flag per ticker) to long (one row per message × ticker reference)
and then collapsed into a balanced daily panel:

    (ticker, date) → {
        sent_mean, sent_median, sent_std,
        share_pos, share_neg, share_neu,
        n_messages, n_authors,
        sent_mean_wls,            # confidence-weighted mean
        polarity_index,           # (n_pos - n_neg) / n_total  ∈ [-1, 1]
    }

Two conventions matter for downstream use:

  * The panel is balanced on the calendar grid spanned by the data. Days
    with zero messages for a ticker appear with `n_messages = 0` and all
    sentiment statistics set to null — not zero. This matters for fixed-
    effects estimation and for distinguishing "no information" from
    "information that is exactly neutral".

  * Standard errors for any downstream regression should be clustered on
    `ticker` (and optionally on `date` for two-way clustering). The panel
    structure here does not impose a clustering choice; it merely exposes
    the keys needed for one.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

from .config import SentimentLabel, TICKERS, settings


# --------------------------------------------------------------------------- #
# Reshape: wide ticker flags → long (message × ticker)                        #
# --------------------------------------------------------------------------- #

def explode_to_long(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert the wide message frame to long form, one row per message-ticker
    pair. A message that mentions two tickers contributes two rows; this is
    the standard treatment in the financial-text literature and is required
    for ticker-level aggregation.
    """
    ticker_cols = [f"has_{t}" for t in TICKERS]
    missing = [c for c in ticker_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Ticker flags missing from input: {missing}")

    long = (
        df.unpivot(
            on=ticker_cols,
            index=[c for c in df.columns if c not in ticker_cols],
            variable_name="ticker_flag",
            value_name="mentioned",
        )
        .filter(pl.col("mentioned"))
        .with_columns(
            pl.col("ticker_flag").str.strip_prefix("has_").alias("ticker")
        )
        .drop("ticker_flag", "mentioned")
    )
    logger.info("Reshaped {} wide rows → {} long (message × ticker) rows.",
                df.height, long.height)
    return long


# --------------------------------------------------------------------------- #
# Date key                                                                    #
# --------------------------------------------------------------------------- #

def add_date_key(df: pl.DataFrame,
                 timestamp_col: str | None = None) -> pl.DataFrame:
    """
    Add a `date` column (calendar date, no time-of-day). Trading-day
    alignment is intentionally not enforced here: that is a downstream
    decision (e.g., shift weekend messages onto the next session).
    """
    ts = timestamp_col or settings.timestamp_column
    return df.with_columns(pl.col(ts).dt.date().alias("date"))


# --------------------------------------------------------------------------- #
# Daily aggregation                                                           #
# --------------------------------------------------------------------------- #

def daily_panel(long: pl.DataFrame) -> pl.DataFrame:
    """
    Collapse the long message frame to a (ticker, date) panel.

    `sent_mean_wls` is the mean of `sent_score` weighted by classifier
    confidence `sent_conf`. This down-weights low-confidence predictions
    without discarding them, which is preferable to a hard threshold when
    the daily message count is small.
    """
    required = {"ticker", "date", "sent_score", "sent_label",
                "sent_conf", settings.author_column}
    missing = required - set(long.columns)
    if missing:
        raise KeyError(f"Missing columns for aggregation: {missing}")

    grouped = (
        long.group_by(["ticker", "date"])
        .agg(
            pl.col("sent_score").mean().alias("sent_mean"),
            pl.col("sent_score").median().alias("sent_median"),
            pl.col("sent_score").std(ddof=1).alias("sent_std"),
            ((pl.col("sent_score") * pl.col("sent_conf")).sum()
             / pl.col("sent_conf").sum().clip(lower_bound=1e-9))
                .alias("sent_mean_wls"),
            (pl.col("sent_label") == SentimentLabel.POSITIVE.value)
                .mean().alias("share_pos"),
            (pl.col("sent_label") == SentimentLabel.NEGATIVE.value)
                .mean().alias("share_neg"),
            (pl.col("sent_label") == SentimentLabel.NEUTRAL.value)
                .mean().alias("share_neu"),
            pl.len().alias("n_messages"),
            pl.col(settings.author_column).n_unique().alias("n_authors"),
        )
        .with_columns(
            (pl.col("share_pos") - pl.col("share_neg")).alias("polarity_index")
        )
    )
    return grouped


# --------------------------------------------------------------------------- #
# Panel balancing                                                             #
# --------------------------------------------------------------------------- #

def balance_panel(panel: pl.DataFrame) -> pl.DataFrame:
    """
    Cross-join the ticker set with the full calendar span of the data and
    left-join the observed aggregates. Days without messages for a ticker
    carry `n_messages = 0` and null sentiment statistics.
    """
    if panel.is_empty():
        logger.warning("Empty panel passed to balance_panel; returning as-is.")
        return panel

    d_min, d_max = panel["date"].min(), panel["date"].max()
    calendar = pl.DataFrame({
        "date": pl.date_range(d_min, d_max, interval="1d", eager=True)
    })
    skeleton = (
        pl.DataFrame({"ticker": list(TICKERS)})
        .join(calendar, how="cross")
    )

    balanced = (
        skeleton.join(panel, on=["ticker", "date"], how="left")
        .with_columns(
            pl.col("n_messages").fill_null(0).cast(pl.UInt32),
            pl.col("n_authors").fill_null(0).cast(pl.UInt32),
        )
        .sort(["ticker", "date"])
    )

    gaps = balanced.filter(pl.col("n_messages") == 0).height
    logger.info("Balanced panel: {} ticker-day cells, {} with zero messages.",
                balanced.height, gaps)
    return balanced


# --------------------------------------------------------------------------- #
# Lagged and rolling features                                                 #
# --------------------------------------------------------------------------- #

def add_dynamics(panel: pl.DataFrame,
                 lags: tuple[int, ...] = (1, 2, 3, 5),
                 rolling_windows: tuple[int, ...] = (3, 7)) -> pl.DataFrame:
    """
    Append lagged sentiment levels and rolling means within ticker. These
    are routine regressors in event studies and local-projection
    specifications; we generate them here to keep the econometric module
    downstream free of feature engineering.
    """
    panel = panel.sort(["ticker", "date"])

    lag_exprs = [
        pl.col("sent_mean").shift(k).over("ticker").alias(f"sent_mean_l{k}")
        for k in lags
    ]
    roll_exprs = [
        pl.col("sent_mean")
          .rolling_mean(window_size=w, min_periods=max(2, w // 2))
          .over("ticker")
          .alias(f"sent_mean_ma{w}")
        for w in rolling_windows
    ]
    return panel.with_columns(*lag_exprs, *roll_exprs)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

def build_panel(scored: pl.DataFrame) -> pl.DataFrame:
    """
    End-to-end aggregation: wide → long → daily collapse → balance →
    dynamics. Returns the analysis-ready panel.
    """
    long = explode_to_long(scored)
    long = add_date_key(long)
    daily = daily_panel(long)
    balanced = balance_panel(daily)
    final = add_dynamics(balanced)
    logger.info("Panel ready: {} rows × {} columns.", final.height, final.width)
    return final


def write_panel(panel: pl.DataFrame,
                name: str = "sentiment_panel.parquet") -> Path:
    out = settings.output_dir / name
    panel.write_parquet(out, compression="zstd", statistics=True)
    logger.info("Wrote sentiment panel to {}.", out)
    return out


# --------------------------------------------------------------------------- #
# Summary diagnostics                                                         #
# --------------------------------------------------------------------------- #

def summarise(panel: pl.DataFrame) -> pl.DataFrame:
    """
    Per-ticker summary statistics. Intended for inclusion in the descriptive
    table of a research paper; not for inference.
    """
    return (
        panel.group_by("ticker")
        .agg(
            pl.col("n_messages").sum().alias("messages_total"),
            pl.col("n_messages").mean().alias("messages_per_day"),
            pl.col("sent_mean").mean().alias("sent_mean_avg"),
            pl.col("sent_mean").std(ddof=1).alias("sent_mean_sd"),
            pl.col("polarity_index").mean().alias("polarity_avg"),
            (pl.col("n_messages") > 0).sum().alias("active_days"),
        )
        .sort("ticker")
    )