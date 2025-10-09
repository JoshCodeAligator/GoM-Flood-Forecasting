"""Utility plotting helpers for the GoM Flood Forecasting pipeline.

All functions use matplotlib (no seaborn) and return the tuple (fig, ax) so
callers can further customize or test. Each function can optionally save to
`out_path` (PNG) and will create parent directories as needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from .io import REPORTS
import pandas as pd

# -----------------------------------------------------------------------------
# Internal utils
# -----------------------------------------------------------------------------


def _resolve_out_path(out_path: Optional[Union[str, Path]]) -> Optional[Path]:
    if out_path is None:
        return None
    p = Path(out_path)
    # If caller passed a relative path (e.g., "emerson_flow.png"), save under root reports/figures
    if not p.is_absolute():
        p = REPORTS / p
    return p


def _ensure_parent(out_path: Optional[Union[str, Path]]) -> None:
    p = _resolve_out_path(out_path)
    if p is None:
        return
    p.parent.mkdir(parents=True, exist_ok=True)


def _infer_unit_label(df_like) -> str:
    try:
        col = df_like["unit_SI"]
        if hasattr(col, "dropna"):
            v = col.dropna()
            if len(v):
                return str(v.iloc[0])
    except Exception:
        pass
    return ""


def _finalize(fig: plt.Figure, out_path: Optional[Union[str, Path]]) -> Tuple[plt.Figure, plt.Axes]:
    """Tight layout and optional save, returning (fig, ax)."""
    p = _resolve_out_path(out_path)
    # enforce plain (non-scientific) y-axis across all axes
    for ax in fig.axes:
        ax.set_xlabel(ax.get_xlabel() or "Date")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.ticklabel_format(style='plain', axis='y')
    fig.autofmt_xdate()
    plt.tight_layout()
    if p:
        _ensure_parent(p)
        fig.savefig(p, dpi=200)
    # return first axes for convenience
    return fig, fig.axes[0]


# -----------------------------------------------------------------------------
# Core plots used
# -----------------------------------------------------------------------------

def plot_hydrograph(
    df: pd.DataFrame,
    station: str,
    metric: str,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot daily values for a station/metric.

    Parameters
    ----------
    df : DataFrame with columns ['station','date','metric','value_SI', 'unit_SI']
    station : station name to filter
    metric : 'flow' or 'level' (or other standardized metric)
    ylabel : y-axis label; if None, will use df['unit_SI'] for that subset
    title : plot title; if None, auto-generated
    out_path : where to save PNG; if None, only returns fig/ax
    """
    sub = (
        df.query("station == @station and metric == @metric")
          .dropna(subset=["date", "value_SI"])  # safety
          .sort_values("date")
          .set_index("date")
    )
    sub.index = pd.to_datetime(sub.index)
    fig, ax = plt.subplots(figsize=(10, 4))
    if sub.empty:
        ax.set_title(title or f"{station} — {metric} (no data)")
        return _finalize(fig, out_path)

    sub["value_SI"].plot(ax=ax, label="daily")

    ax.set_ylabel(ylabel or _infer_unit_label(sub))
    ax.set_title(title or f"{station} — {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finalize(fig, out_path)


def plot_rolling(
    df: pd.DataFrame,
    station: str,
    metric: str,
    ylabel: Optional[str] = None,
    win: int = 7,
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay a rolling mean on top of the daily series (optionally from freeze-up onward)."""
    sub = (
        df.query("station == @station and metric == @metric")
          .dropna(subset=["date", "value_SI"])  # safety
          .sort_values("date")
          .set_index("date")
    )
    sub.index = pd.to_datetime(sub.index)
    fig, ax = plt.subplots(figsize=(10, 4))
    if sub.empty:
        ax.set_title(title or f"{station} — {metric} (no data)")
        return _finalize(fig, out_path)

    sub["value_SI"].plot(ax=ax, alpha=0.35, label="daily")
    sub["value_SI"].rolling(win, min_periods=1).mean().plot(ax=ax, label=f"{win}-day mean")

    ax.set_title(title or f"{station} — {metric} (rolling {win}d)")
    if ylabel is None:
        ax.set_ylabel(_infer_unit_label(sub))
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finalize(fig, out_path)


def plot_bands(
    obs_series: pd.Series,
    scenarios: Dict[str, Union[float, int, pd.Series]],
    title: str,
    out_path: Union[str, Path],
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot observed series against scenario bands (p10–p90) and p50.

    Parameters
    ----------
    obs_series : Series indexed by datetime, observed values (SI units)
    scenarios : dict with keys 'p10','p50','p90' (scalar or Series aligned to obs index)
    title : plot title
    out_path : path to save PNG
    """
    idx = obs_series.index

    def to_series(x):
        if isinstance(x, (int, float)):
            return pd.Series([x] * len(idx), index=idx)
        if isinstance(x, pd.Series):
            return x.reindex(idx)
        return pd.Series([pd.NA] * len(idx), index=idx)

    p10 = to_series(scenarios.get("p10"))
    p50 = to_series(scenarios.get("p50"))
    p90 = to_series(scenarios.get("p90"))

    fig, ax = plt.subplots(figsize=(10, 4))
    obs_series.plot(ax=ax, label="Observed")
    ax.fill_between(idx, p10, p90, alpha=0.3, label="Forecast band (p10–p90)")
    ax.plot(idx, p50, linestyle="--", label="Official p50")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finalize(fig, out_path)


def plot_map(
    gdf,  # GeoDataFrame of points
    basins=None,  # optional GeoDataFrame of polygons
    out_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
):
    """Plot station points over optional basin polygons (no CRS changes here)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if basins is not None and not basins.empty:
        basins.plot(ax=ax, facecolor="none", edgecolor="black")
    if gdf is not None and not gdf.empty:
        gdf.plot(ax=ax, markersize=20)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.2)
    return _finalize(fig, out_path)


def simple_bar(data: Union[pd.Series, pd.DataFrame], title: str, out_path: Union[str, Path]):
    """Bar plot helper for quick comparisons (e.g., soil wetness categories)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if isinstance(data, pd.Series):
        data.plot(kind="bar", ax=ax)
    else:
        data.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return _finalize(fig, out_path)


# -----------------------------------------------------------------------------
# Optional advanced plots used in Step 1 extras / later steps
# -----------------------------------------------------------------------------

def plot_climatology_band(
    df: pd.DataFrame,
    station: str,
    metric: str,
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
    pct_low: float = 10,
    pct_high: float = 90,
) -> Tuple[plt.Figure, plt.Axes]:
    """Compute and plot a day-of-year climatology band with current year overlay (optionally from freeze-up onward).

    Expects df with multiple years. Produces a band between pct_low and pct_high
    percentiles and overlays the latest year values.
    """
    sub = (
        df.query("station == @station and metric == @metric")
          .dropna(subset=["date", "value_SI"])  # safety
          .sort_values("date")
          .set_index("date")
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    if sub.empty:
        ax.set_title(title or f"{station} — {metric} (no data)")
        return _finalize(fig, out_path)

    s = sub["value_SI"].copy()
    s.index = pd.to_datetime(s.index)
    # Group by day-of-year
    doy = s.groupby(s.index.dayofyear)
    p10 = doy.quantile(pct_low / 100.0)
    p50 = doy.quantile(0.5)
    p90 = doy.quantile(pct_high / 100.0)

    # Overlay latest year
    latest_year = s.index.max().year
    cur = s[s.index.year == latest_year]
    # Align to day-of-year index for plotting band
    idx = pd.Index(range(1, 367))
    p10 = p10.reindex(idx)
    p50 = p50.reindex(idx)
    p90 = p90.reindex(idx)
    cur_doy = pd.Series(cur.values, index=cur.index.dayofyear).reindex(idx)

    ax.fill_between(idx, p10, p90, alpha=0.3, label=f"{pct_low}-{pct_high}th pct band")
    ax.plot(idx, p50, linestyle="--", label="Median")
    ax.plot(idx, cur_doy, label=f"{latest_year}")
    ax.set_title(title or f"{station} — {metric} climatology")
    ax.set_xlabel("day of year")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finalize(fig, out_path)


def plot_anomaly(
    df: pd.DataFrame,
    station: str,
    metric: str,
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot anomaly = observed − climatological median by day-of-year (optionally from freeze-up onward)."""
    sub = (
        df.query("station == @station and metric == @metric")
          .dropna(subset=["date", "value_SI"])  # safety
          .sort_values("date")
          .set_index("date")
    )
    fig, ax = plt.subplots(figsize=(10, 3.5))
    if sub.empty:
        ax.set_title(title or f"{station} — {metric} (no data)")
        return _finalize(fig, out_path)

    s = sub["value_SI"].copy()
    s.index = pd.to_datetime(s.index)
    median_by_doy = s.groupby(s.index.dayofyear).median()
    anomaly = s - median_by_doy.reindex(s.index.dayofyear).values
    anomaly.plot(ax=ax, label="anomaly")

    ax.axhline(0, linestyle="--", alpha=0.5)
    ax.set_title(title or f"{station} — {metric} anomaly vs climatology")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finalize(fig, out_path)


def plot_fdc(
    df: pd.DataFrame,
    station: str,
    metric: str,
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Flow Duration Curve (exceedance probability vs value), optionally restricted from freeze-up onward."""
    sub = (
        df.query("station == @station and metric == @metric")
          .dropna(subset=["value_SI"])  # safety
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    if sub.empty:
        ax.set_title(title or f"{station} — {metric} (no data)")
        return _finalize(fig, out_path)

    vals = sub["value_SI"].sort_values(ascending=False).reset_index(drop=True)
    n = len(vals)
    exceed_pct = (vals.rank(method="first", ascending=True) / (n + 1)) * 100.0
    ax.plot(exceed_pct, vals)

    ax.set_xlabel("exceedance probability (%)")
    ax.set_ylabel(_infer_unit_label(sub))
    ax.set_title(title or f"{station} — {metric} FDC")
    ax.set_xscale("linear")
    # y-axis often benefits from log scale for flows; caller can modify outside if desired
    ax.grid(True, which="both", alpha=0.3)
    return _finalize(fig, out_path)