from .io import (
    DATA, REPORTS, reports_path,
    load_historical_flows, extend_with_current,
    load_baseline, get_freeze_up_marker
)
from .visualize import plot_hydrograph, plot_rolling, plot_anomaly, plot_fdc
import pandas as pd

def _slug(s: str) -> str:
    """Make a safe filename slug from a station name like 'James Ave'."""
    return s.lower().replace(" ", "_").replace(".", "")


def _norm(s: str) -> str:
    """Normalize station names for matching."""
    return (s or "").strip().lower()

def main(outdir=None):
    # 1) Load historical time series and extend with current API data
    hist = load_historical_flows()
    all_df, _current_df = extend_with_current(hist)

    # 2) Load freeze-up/baseline table and extract all Red River station+metric pairs
    baseline = load_baseline().copy()
   
    if "basin" in baseline.columns:
        baseline = baseline[baseline["basin"].str.strip().str.lower() == "red"]

    # Build the list of unique (station, metric) that we must attempt to plot
    required_pairs = (
        baseline[["station", "metric"]]
        .dropna()
        .drop_duplicates()
        .assign(st_norm=lambda d: d["station"].apply(_norm))
        .to_dict(orient="records")
    )

    print(required_pairs)
    # If baseline is empty or missing basin column, fall back to whatever is in the time series
    if not required_pairs:
        stations = sorted(all_df["station"].dropna().unique().tolist())
        metrics = ["flow", "level"]
        required_pairs = [{"station": s, "metric": m, "st_norm": _norm(s)} for s in stations for m in metrics]

    # 3) Prepare output dir
    (REPORTS / "step1_flows").mkdir(parents=True, exist_ok=True)

    # 4) Plot for every baseline-required (station, metric) pair
    for pair in required_pairs:
        st = pair["station"]
        metric = pair["metric"]

        # pull freeze-up marker for this pair (may return None if not present)
        # Only include freeze-up marker for flow metrics
        freeze_marker = None
        if metric == "flow":
            freeze_marker = get_freeze_up_marker(st, metric)

        # select series (match by normalized station string)
        subset = all_df[(all_df["station"].apply(_norm) == _norm(st)) & (all_df["metric"] == metric)]
        if subset.empty:
            print(f"[flows] WARNING: no timeseries for {st} / {metric}. Skipping plots but keeping baseline marker.")
            # You could optionally render a marker-only figure here; for now skip.
            continue

        ylabel = "Flow (m³/s)" if metric == "flow" else "Level (m)"
        base = f"{_slug(st)}_{metric}"

        # Hydrograph
        plot_hydrograph(
            all_df, st, metric,
            ylabel=ylabel,
            title=f"{st} — {metric.title()} (history + current API)",
            out_path=reports_path(f"step1_flows/{base}_hydro.png"),
            freeze_marker=freeze_marker
        )

        # Rolling mean (7-day)
        plot_rolling(
            all_df, st, metric, win=7,
            title=f"{st} — 7-day rolling mean ({metric})",
            out_path=reports_path(f"step1_flows/{base}_rolling.png"),
            freeze_marker=freeze_marker
        )

        # Anomaly vs DOY median
        plot_anomaly(
            all_df, st, metric,
            title=f"{st} — {metric.title()} anomaly (vs DOY median)",
            out_path=reports_path(f"step1_flows/{base}_anom.png")
        )

        # Flow Duration Curve (only if metric is flow and enough points)
        if metric == "flow" and len(subset) >= 30:
            plot_fdc(
                all_df, st, metric,
                title=f"{st} — Flow duration curve",
                out_path=reports_path(f"step1_flows/{base}_fdc.png"),
                freeze_marker=freeze_marker
            )

    # 5) Save the unified series we used for plotting (handy for later steps)
    (DATA / "processed").mkdir(parents=True, exist_ok=True)
    all_df.to_csv(DATA / "processed/red_river_timeseries_step1.csv", index=False)

if __name__ == "__main__":
    main()