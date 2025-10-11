from __future__ import annotations
"""
Step 5 – Flood Outlooks (Scenario Benchmarks)

What this module does
---------------------
1) Loads observed time series (historical + last-30d API extension) from Step 1.
2) Loads scenario bands from `data/processed/scenario_peaks.csv` (p10/p50/p90).
3) Produces comparison plots: observed hydrograph vs shaded forecast bands.

Run:
    python -m hydro_pipeline.outlooks --station Emerson --days 120
    python -m hydro_pipeline.outlooks --station "James Ave" --metric level
    python -m hydro_pipeline.outlooks --station all

Outputs:
    reports/figures/step5/{station}_{metric}_scenarios.png
"""
import argparse
from pathlib import Path
import pandas as pd

from .io import (
    REPORTS,
    load_historical_flows,
    extend_with_current,
    load_scenarios,
)
from .visualize import plot_bands

STEP5_DIR = REPORTS / "step5_outlooks"
STEP5_DIR.mkdir(parents=True, exist_ok=True)

FLOW_STATIONS = {
    "Emerson": {"metric": "flow"},
    "Ste. Agathe": {"metric": "flow"},
    "Selkirk": {"metric": "flow"},
}
LEVEL_STATIONS = {
    "James Ave": {"metric": "level"},
    "Lockport": {"metric": "level"},
    "Breezy Point": {"metric": "level"},
}
VALID_STATIONS = {**FLOW_STATIONS, **LEVEL_STATIONS}

def _pick_metric_for_station(station: str, metric: str | None) -> str:
    """Return the metric to use ("flow" or "level"). If user did not
    pass --metric, infer from station type.
    """
    if metric:
        return metric
    if station in FLOW_STATIONS:
        return "flow"
    if station in LEVEL_STATIONS:
        return "level"
    return "flow"

def _scenario_band_for_station(scen_df: pd.DataFrame, station: str, metric: str) -> pd.DataFrame:
    """Return a tiny DataFrame with columns [scenario, percentile, value_SI]
    filtered for a station/metric, where scenarios are
    {favourable (p10), normal (p50), unfavourable (p90)}.

    `scenario_peaks.csv` rows are expected to include:
        station, scenario (favourable|normal|unfavourable), percentile (p10|p50|p90),
        metric (peak_flow|peak_level), value_SI
    """
    # Map desired metric -> scenario file metric label
    need_metric = "peak_flow" if metric == "flow" else "peak_level"

    sub = scen_df.query("station == @station and metric == @need_metric").copy()
    # Ensure expected categories exist
    if sub.empty:
        return sub

    # Keep only the columns we need and sort by scenario for consistent plotting
    cols = ["station", "scenario", "percentile", "metric", "value_SI", "unit_SI"]
    sub = sub[[c for c in cols if c in sub.columns]].copy()

    # Sanity: make sure we have (p10, p50, p90)
    have = set(sub["percentile"].unique())
    expected = {"p10", "p50", "p90"}
    if not expected.issubset(have):
        # We can still plot whatever is present; plot_bands should handle missing gracefully
        pass

    return sub

def run(station: str, metric: str | None, days: int) -> Path:
    metric = _pick_metric_for_station(station, metric)

    # 1) Observed series (historical + API extension)
    hist_df = load_historical_flows()
    all_df, _cur = extend_with_current(hist_df)  # io.extend_with_current returns (all_df, current_df)

    # Restrict to requested station/metric and last N days
    today = pd.Timestamp.today().normalize()
    start = (today - pd.Timedelta(days=days-1)) if days and days > 0 else None
    obs_filt = (
        all_df.query("station == @station and metric == @metric")[
            ["date", "station", "metric", "value_SI", "unit_SI", "source"]
        ]
        .sort_values("date")
    )
    if start is not None and not obs_filt.empty:
        obs_filt = obs_filt[obs_filt["date"] >= start]
    obs_filt = obs_filt.reset_index(drop=True)

    if obs_filt.empty:
        raise SystemExit(f"No observed data found for station={station!r}, metric={metric!r}.")

    # 2) Scenario bands
    scen_df = load_scenarios()
    bands = _scenario_band_for_station(scen_df, station, metric)
    if bands.empty:
        raise SystemExit(
            f"No scenario entries found in scenario_peaks.csv for station={station!r}, metric={metric!r}."
        )

    # 3) Plot observed vs bands
    ylabel = "Flow (m³/s)" if metric == "flow" else "Level (m)"
    title = f"{station} — Observed vs Forecast Bands"

    station_slug = station.replace(' ', '_').lower()
    out_png = STEP5_DIR / f"{station_slug}_{metric}_scenarios.png"
    (STEP5_DIR / "data").mkdir(parents=True, exist_ok=True)
    
    # Save the exact inputs we used for traceability
    obs_csv = STEP5_DIR / "data" / f"{station_slug}_{metric}_observed.csv"
    bands_csv = STEP5_DIR / "data" / f"{station_slug}_{metric}_bands.csv"
    obs_filt.to_csv(obs_csv, index=False)
    bands.to_csv(bands_csv, index=False)

    plot_bands(
        df_obs=obs_filt,
        station=station,
        metric=metric,
        bands_df=bands,
        ylabel=ylabel,
        title=title,
        out_path=out_png,
        freeze_marker=None,
    )

    return out_png

def run_all(days: int) -> list[Path]:
    outputs: list[Path] = []
    for st, meta in VALID_STATIONS.items():
        try:
            outputs.append(run(st, meta.get("metric"), days))
        except SystemExit as e:
            print(f"[WARN] {st}: {e}")
    return outputs

def main(outdir=None):
    p = argparse.ArgumentParser(description="Step 5 – Outlook scenario comparison plots")
    p.add_argument("--station", default="Emerson", help="Station name (e.g., Emerson, Ste. Agathe, James Ave, Lockport, Selkirk, Breezy Point, or 'all')")
    p.add_argument("--metric", choices=["flow", "level"], default=None, help="Override metric (flow|level)")
    p.add_argument("--days", type=int, default=120, help="Observed window (days) to include")
    args = p.parse_args()

    if args.station.lower() == "all":
        outs = run_all(args.days)
        if outs:
            print("[OK] Wrote:")
            for pth in outs:
                print("  ", pth)
        else:
            print("[WARN] No outputs produced (no matching data/bands).")
    else:
        out = run(args.station, args.metric, args.days)
        print(f"[OK] Wrote {out}")


if __name__ == "__main__":
    main()