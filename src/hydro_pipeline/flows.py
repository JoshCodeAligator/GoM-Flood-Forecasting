from pathlib import Path
from .io import DATA, reports_path, load_historical_flows, extend_with_current_last_n_days, load_baseline_markers
from .visualize import plot_hydrograph, plot_rolling, plot_anomaly, plot_fdc

def main(outdir=None):
    hist = load_historical_flows()
    all_df, current_df = extend_with_current_last_n_days(hist, n=30)  # fetch last 30 days via API
    markers = load_baseline_markers()

    # plots
    plot_hydrograph(all_df, "Emerson", "flow",
                    ylabel="Flow (m³/s)",
                    title="Emerson — Flow since freeze-up",
                    out_path=reports_path("emerson_flow_hydro.png"))

    plot_hydrograph(all_df, "James Ave", "level",
                    ylabel="Level (m)",
                    title="James Ave — Level since freeze-up",
                    out_path=reports_path("james_level_hydro.png"))

    plot_rolling(all_df, "Emerson", "flow",
                 win=7,
                 title="Emerson — 7-day rolling mean (flow)",
                 out_path=reports_path("emerson_flow_rolling.png"))

    plot_rolling(all_df, "James Ave", "level",
                 win=7,
                 title="James Ave — 7-day rolling mean (level)",
                 out_path=reports_path("james_level_rolling.png"))

    plot_anomaly(all_df, "Emerson", "flow",
                 title="Emerson — Flow anomalies since freeze-up",
                 out_path=reports_path("emerson_flow_anom.png"))

    plot_fdc(all_df, "Emerson", "flow",
             title="Emerson — Flow duration curve",
             out_path=reports_path("emerson_flow_fdc.png"))

    # optional save
    all_df.to_csv(DATA / "processed/red_river_timeseries_step1.csv", index=False)

if __name__ == "__main__":
    main()