"""
Visualisation.

All figures are produced with Matplotlib using a Persian-capable font
(Vazirmatn). Persian strings are passed through `arabic-reshaper` and
`python-bidi` so that RTL glyphs render correctly under LTR Matplotlib
text engines. Figures are written to `settings.figure_dir` as PDF (vector,
for inclusion in LaTeX) and PNG (for quick inspection).

Plots are intentionally austere: no decorative elements, no colour scales
that imply ordinal information where there is none, and no annotations
beyond what the underlying statistics support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from arabic_reshaper import reshape
from bidi.algorithm import get_display
from loguru import logger
from matplotlib.ticker import MaxNLocator

from .config import TICKERS, settings


# --------------------------------------------------------------------------- #
# Font handling                                                               #
# --------------------------------------------------------------------------- #

_PERSIAN_FONT_CANDIDATES = (
    "Vazirmatn", "Vazir", "Sahel", "IRANSans", "B Nazanin", "Noto Sans Arabic",
)


def _resolve_persian_font() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in _PERSIAN_FONT_CANDIDATES:
        if cand in available:
            return cand
    logger.warning(
        "No Persian font found among {}. Falling back to DejaVu Sans; "
        "Persian glyphs may render incorrectly.", _PERSIAN_FONT_CANDIDATES,
    )
    return "DejaVu Sans"


def _fa(s: str) -> str:
    """Shape and bidi-reorder a Persian string for Matplotlib."""
    return get_display(reshape(s))


def _apply_style() -> None:
    font = _resolve_persian_font()
    plt.rcParams.update({
        "font.family": font,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# --------------------------------------------------------------------------- #
# Output helpers                                                              #
# --------------------------------------------------------------------------- #

def _save(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    pdf = settings.figure_dir / f"{stem}.pdf"
    png = settings.figure_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    logger.info("Saved figure: {} / {}", pdf.name, png.name)
    return pdf, png


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #

def plot_message_counts(panel: pl.DataFrame,
                        tickers: Iterable[str] = TICKERS) -> Path:
    """Bar chart: total messages per ticker."""
    _apply_style()
    agg = (panel.group_by("ticker")
                .agg(pl.col("n_messages").sum().alias("total"))
                .sort("ticker"))
    order = [t for t in tickers if t in agg["ticker"].to_list()]
    totals = [agg.filter(pl.col("ticker") == t)["total"][0] for t in order]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(range(len(order)), totals, color="#4C72B0", edgecolor="black",
           linewidth=0.4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_fa(t) for t in order])
    ax.set_ylabel(_fa("تعداد پیام‌ها"))
    ax.set_xlabel(_fa("نماد"))
    ax.set_title(_fa("حجم پیام‌ها به تفکیک نماد"))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for i, v in enumerate(totals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    pdf, _ = _save(fig, "message_counts")
    return pdf


def plot_sentiment_distribution(scored: pl.DataFrame,
                                tickers: Iterable[str] = TICKERS) -> Path:
    """Stacked bar: share of positive/neutral/negative messages per ticker."""
    _apply_style()
    long = (scored
            .unpivot(on=[f"has_{t}" for t in tickers],
                     index=["sent_label"],
                     variable_name="ticker_flag", value_name="mentioned")
            .filter(pl.col("mentioned"))
            .with_columns(pl.col("ticker_flag").str.strip_prefix("has_").alias("ticker"))
            .drop("ticker_flag", "mentioned"))

    pivoted = (long.group_by(["ticker", "sent_label"])
                   .len().rename({"len": "n"})
                   .pivot(values="n", index="ticker", on="sent_label")
                   .fill_null(0))

    for col in ("positive", "negative", "neutral"):
        if col not in pivoted.columns:
            pivoted = pivoted.with_columns(pl.lit(0).alias(col))

    pivoted = pivoted.with_columns(
        (pl.col("positive") + pl.col("negative") + pl.col("neutral"))
            .alias("total")
    )
    shares = pivoted.with_columns(
        (pl.col("positive") / pl.col("total")).alias("p_pos"),
        (pl.col("neutral")  / pl.col("total")).alias("p_neu"),
        (pl.col("negative") / pl.col("total")).alias("p_neg"),
    ).sort("ticker")

    order = shares["ticker"].to_list()
    p_pos = shares["p_pos"].to_numpy()
    p_neu = shares["p_neu"].to_numpy()
    p_neg = shares["p_neg"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(order))
    ax.bar(x, p_pos,             color="#55A868", label=_fa("مثبت"))
    ax.bar(x, p_neu, bottom=p_pos, color="#BBBBBB", label=_fa("خنثی"))
    ax.bar(x, p_neg, bottom=p_pos + p_neu, color="#C44E52", label=_fa("منفی"))
    ax.set_xticks(x)
    ax.set_xticklabels([_fa(t) for t in order])
    ax.set_ylabel(_fa("سهم"))
    ax.set_xlabel(_fa("نماد"))
    ax.set_ylim(0, 1)
    ax.set_title(_fa("توزیع برچسب احساسات به تفکیک نماد"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=False)
    pdf, _ = _save(fig, "sentiment_distribution")
    return pdf


def plot_daily_sentiment(panel: pl.DataFrame,
                         tickers: Iterable[str] = TICKERS,
                         smoothing_window: int = 7) -> Path:
    """Time series of weighted daily mean sentiment per ticker."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9.0, 4.5))

    for t in tickers:
        sub = (panel.filter(pl.col("ticker") == t)
                    .sort("date"))
        if sub.is_empty():
            continue
        x = sub["date"].to_numpy()
        y = sub["sent_mean_wls"].to_numpy().astype(float)
        # Rolling mean for readability; raw series shown faintly underneath.
        sm = pl.Series(y).rolling_mean(window_size=smoothing_window,
                                       min_periods=2).to_numpy()
        ax.plot(x, y, alpha=0.25, linewidth=0.8)
        ax.plot(x, sm, linewidth=1.6, label=_fa(t))

    ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel(_fa("تاریخ"))
    ax.set_ylabel(_fa("میانگین احساسات روزانه"))
    ax.set_title(_fa(
        f"روند احساسات روزانه (میانگین وزنی، هموارسازی {smoothing_window} روزه)"
    ))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=len(list(tickers)), frameon=False)
    fig.autofmt_xdate()
    pdf, _ = _save(fig, "daily_sentiment")
    return pdf


def plot_polarity_heatmap(panel: pl.DataFrame,
                          tickers: Iterable[str] = TICKERS) -> Path:
    """Ticker × date heatmap of the polarity index."""
    _apply_style()
    wide = (panel.select(["ticker", "date", "polarity_index"])
                 .pivot(values="polarity_index", index="ticker", on="date")
                 .sort("ticker"))
    order = [t for t in tickers if t in wide["ticker"].to_list()]
    wide = wide.filter(pl.col("ticker").is_in(order))

    data_cols = [c for c in wide.columns if c != "ticker"]
    M = wide.select(data_cols).to_numpy().astype(float)
    dates = data_cols

    fig, ax = plt.subplots(figsize=(10.0, 0.6 * len(order) + 1.4))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_fa(t) for t in order])
    step = max(1, len(dates) // 10)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels([str(dates[i]) for i in range(0, len(dates), step)],
                       rotation=45, ha="right")
    ax.set_xlabel(_fa("تاریخ"))
    ax.set_title(_fa("شاخص قطبیت احساسات به تفکیک نماد و روز"))
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(_fa("شاخص قطبیت"))
    pdf, _ = _save(fig, "polarity_heatmap")
    return pdf