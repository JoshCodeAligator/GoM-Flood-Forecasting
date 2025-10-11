from __future__ import annotations
"""Utility plotting helpers for the GoM Flood Forecasting pipeline.

All functions use matplotlib (no seaborn) and return the tuple (fig, ax) so
callers can further customize or test. Each function can optionally save to
`out_path` (PNG) and will create parent directories as needed.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from .io import REPORTS, _require_geopandas
import pandas as pd
from shapely.geometry import (
    Polygon, MultiPolygon,
    LineString, MultiLineString,
    GeometryCollection,
)

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
        # Only auto-format dates when the x-axis is actually time-based
        try:
            if isinstance(ax.xaxis.get_major_formatter(), mdates.DateFormatter) or isinstance(ax.xaxis.get_major_locator(), mdates.AutoDateLocator):
                fig.autofmt_xdate()
        except Exception:
            pass
    plt.tight_layout()
    if p:
        _ensure_parent(p)
        fig.savefig(p, dpi=200)
    # return first axes for convenience
    return fig, fig.axes[0]

def _apply_freeze_marker(ax, freeze_marker: Optional[Dict[str, Any]], unit_label: str = "") -> None:
    """
    Draw a vertical freeze-up marker line (and optional point if value provided).

    freeze_marker: dict like {"date": "YYYY-MM-DD", "value_SI": <float>|None}
    unit_label: y-axis units for legend clarity (optional).
    """
    if not freeze_marker:
        return
    try:
        dt = pd.to_datetime(freeze_marker.get("date")).normalize()
    except Exception:
        return
    label = f"Freeze-up ({dt.date()})"
    # draw vertical line with a label so it appears in the legend
    vline = ax.axvline(dt, color="red", linestyle="--", alpha=0.7, label=label, zorder=4)
    # optional dot if freeze-up value is known
    val = freeze_marker.get("value_SI", None)
    if val is not None:
        ax.scatter([dt], [val], color="red", zorder=5)

    # ensure the marker is inside view: expand x-limits if necessary
    try:
        xmin, xmax = ax.get_xlim()
        xmin_dt = pd.Timestamp(mdates.num2date(xmin))
        xmax_dt = pd.Timestamp(mdates.num2date(xmax))
        new_min = min(xmin_dt, dt)
        new_max = max(xmax_dt, dt)
        if (new_min != xmin_dt) or (new_max != xmax_dt):
            ax.set_xlim(new_min, new_max)
    except Exception:
        # if anything goes wrong, skip expanding limits
        pass

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
    freeze_marker: Optional[Dict[str, Any]] = None,
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
    freeze_marker : optional dict {"date": "YYYY-MM-DD", "value_SI": float|None}
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
        _apply_freeze_marker(ax, freeze_marker, _infer_unit_label(sub))
        return _finalize(fig, out_path)

    sub["value_SI"].plot(ax=ax, label="daily")
    ax.set_ylabel(ylabel or _infer_unit_label(sub))
    ax.set_title(title or f"{station} — {metric}")
    _apply_freeze_marker(ax, freeze_marker, _infer_unit_label(sub))
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
    freeze_marker: Optional[Dict[str, Any]] = None,
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
        _apply_freeze_marker(ax, freeze_marker, _infer_unit_label(sub))
        return _finalize(fig, out_path)

    sub["value_SI"].plot(ax=ax, alpha=0.35, label="daily")
    sub["value_SI"].rolling(win, min_periods=1).mean().plot(ax=ax, label=f"{win}-day mean")
    ax.set_title(title or f"{station} — {metric} (rolling {win}d)")
    if ylabel is None:
        ax.set_ylabel(_infer_unit_label(sub))
    _apply_freeze_marker(ax, freeze_marker, _infer_unit_label(sub))
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
    freeze_marker: Optional[Dict[str, Any]] = None,
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
        _apply_freeze_marker(ax, freeze_marker)
        return _finalize(fig, out_path)

    s = sub["value_SI"].copy()
    s.index = pd.to_datetime(s.index)
    median_by_doy = s.groupby(s.index.dayofyear).median()
    anomaly = s - median_by_doy.reindex(s.index.dayofyear).values
    anomaly.plot(ax=ax, label="anomaly")

    ax.axhline(0, linestyle="--", alpha=0.5)
    ax.set_title(title or f"{station} — {metric} anomaly vs climatology")
    _apply_freeze_marker(ax, freeze_marker)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finalize(fig, out_path)

def plot_fdc(
    df: pd.DataFrame,
    station: str,
    metric: str,
    title: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
    freeze_marker: Optional[Dict[str, Any]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Flow Duration Curve (exceedance probability vs value), optionally restricted from freeze-up onward."""
    sub = (
        df.query("station == @station and metric == @metric")
          .dropna(subset=["value_SI"])  # safety
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    if sub.empty:
        ax.set_title(title or f"{station} — {metric} (no data)")
        # For FDC, x-axis is probability, so annotate freeze-up by value (horizontal line)
        if freeze_marker and pd.notna(freeze_marker.get("value_SI", None)):
            y = float(freeze_marker["value_SI"])  # freeze-up value in SI units
            ax.axhline(y, color="red", linestyle="--", alpha=0.7,
                       label=f"Freeze-up value ({y:.2f} { _infer_unit_label(sub) })")
        return _finalize(fig, out_path)

    vals = sub["value_SI"].sort_values(ascending=False).reset_index(drop=True)
    n = len(vals)
    exceed_pct = (vals.rank(method="first", ascending=True) / (n + 1)) * 100.0
    ax.plot(exceed_pct, vals)

    ax.set_xlabel("exceedance probability (%)")
    ax.set_ylabel(_infer_unit_label(sub))
    ax.set_title(title or f"{station} — {metric} FDC")
    ax.set_xscale("linear")
    # For FDC, x-axis is probability, so annotate freeze-up by value (horizontal line)
    if freeze_marker and pd.notna(freeze_marker.get("value_SI", None)):
        y = float(freeze_marker["value_SI"])  # freeze-up value in SI units
        ax.axhline(y, color="red", linestyle="--", alpha=0.7,
                   label=f"Freeze-up value ({y:.2f} { _infer_unit_label(sub) })")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return _finalize(fig, out_path)

def plot_basins_and_stations(basin_gdf: "gpd.GeoDataFrame", stations_gdf: "gpd.GeoDataFrame", out_path: Path):
    # Draw basins without using GeoPandas .plot() (avoids aspect errors)
    _require_geopandas()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import matplotlib.pyplot as plt
    from shapely.geometry import (
        Polygon, MultiPolygon,
        LineString, MultiLineString,
        GeometryCollection,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    def _draw_geom(g):
        # Draw outline of polygons/lines using pure Matplotlib
        if g is None or g.is_empty:
            return
        if isinstance(g, (Polygon,)):
            # exterior
            x, y = g.exterior.xy
            ax.plot(x, y, linewidth=1.0, color="black")
            # holes
            for ring in g.interiors:
                xh, yh = ring.xy
                ax.plot(xh, yh, linewidth=0.8, color="black", linestyle=":")
        elif isinstance(g, (MultiPolygon,)):
            for part in g.geoms:
                _draw_geom(part)
        elif isinstance(g, (LineString,)):
            x, y = g.xy
            ax.plot(x, y, linewidth=1.0, color="black")
        elif isinstance(g, (MultiLineString,)):
            for part in g.geoms:
                _draw_geom(part)
        elif isinstance(g, GeometryCollection):
            for part in g.geoms:
                _draw_geom(part)
        else:
            # Unknown geometry type: try .boundary if available
            try:
                b = g.boundary
                if b is not None and not b.is_empty:
                    _draw_geom(b)
            except Exception:
                pass

    # --- basins ---
    if basin_gdf is not None and not basin_gdf.empty and "geometry" in basin_gdf.columns:
        for g in basin_gdf["geometry"]:
            _draw_geom(g)

    # --- stations ---
    if stations_gdf is not None and not stations_gdf.empty and "geometry" in stations_gdf.columns:
        # plot points
        xs = stations_gdf.geometry.x.values
        ys = stations_gdf.geometry.y.values
        ax.scatter(xs, ys, s=35, zorder=5)
        for _, row in stations_gdf.iterrows():
            ax.annotate(
                row["station"],
                xy=(row.geometry.x, row.geometry.y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
            )

    # --- framing ---
    # Prefer basin bounds; fall back to station bounds
    try:
        if basin_gdf is not None and not basin_gdf.empty:
            xmin, ymin, xmax, ymax = basin_gdf.total_bounds
        elif stations_gdf is not None and not stations_gdf.empty:
            xmin, ymin, xmax, ymax = stations_gdf.total_bounds
        else:
            xmin = ymin = -1; xmax = ymax = 1
        if np.isfinite([xmin, ymin, xmax, ymax]).all():
            pad_x = (xmax - xmin) * 0.05 or 0.1
            pad_y = (ymax - ymin) * 0.05 or 0.1
            ax.set_xlim(xmin - pad_x, xmax + pad_x)
            ax.set_ylim(ymin - pad_y, ymax + pad_y)
    except Exception:
        pass

    ax.set_aspect("auto")
    ax.set_title("Red River Basin & Pilot Stations", fontsize=12)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, linewidth=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)