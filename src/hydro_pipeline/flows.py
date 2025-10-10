from .io import DATA, reports_path, load_historical_flows, extend_with_current
from .visualize import plot_hydrograph, plot_rolling, plot_anomaly, plot_fdc

def _slug(s: str) -> str:
    """Make a safe filename slug from a station name like 'James Ave'."""
    return s.lower().replace(" ", "_").replace(".", "")

def main(outdir=None):
    # 1) Load historical (CSV) and extend with current 
    hist = load_historical_flows()
    all_df, current_df = extend_with_current(hist)

    # 2) Determine which station/metric pairs actually exist in the merged data
    if all_df.empty:
        print("[flows] No data to plot.")
        return

    stations = sorted(all_df["station"].dropna().unique().tolist())
    metrics = ["flow", "level"]  # we will check presence per station below

    # 3) Generate plots for each available (station, metric)
    for st in stations:
        for metric in metrics:
            subset = all_df[(all_df["station"] == st) & (all_df["metric"] == metric)]
            if subset.empty:
                continue  # skip pairs that don't exist for this station

            # Common labels and filenames
            ylabel = "Flow (m³/s)" if metric == "flow" else "Level (m)"
            base = f"{_slug(st)}_{metric}"

            # Hydrograph
            plot_hydrograph(
                all_df, st, metric,
                ylabel=ylabel,
                title=f"{st} — {metric.title()} (history + current API)",
                out_path=reports_path(f"{base}_hydro.png")
            )

            # Rolling mean (7-day)
            plot_rolling(
                all_df, st, metric, win=7,
                title=f"{st} — 7-day rolling mean ({metric})",
                out_path=reports_path(f"{base}_rolling.png")
            )

            # Anomaly vs DOY median
            plot_anomaly(
                all_df, st, metric,
                title=f"{st} — {metric.title()} anomaly (vs DOY median)",
                out_path=reports_path(f"{base}_anom.png")
            )

            # Flow Duration Curve (typically most meaningful for flow)
            if metric == "flow" and len(subset) >= 30:  # need enough points
                plot_fdc(
                    all_df, st, metric,
                    title=f"{st} — Flow duration curve",
                    out_path=reports_path(f"{base}_fdc.png")
                )

    # 4) Optional: write the merged time series used for plotting
    (DATA / "processed").mkdir(parents=True, exist_ok=True)
    all_df.to_csv(DATA / "processed/red_river_timeseries_step1.csv", index=False)

if __name__ == "__main__":
    main()