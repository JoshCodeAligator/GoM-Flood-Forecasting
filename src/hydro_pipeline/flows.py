from pathlib import Path
from .io import DATA, REPORTS, load_historical_flows, extend_with_current, load_baseline_markers
from .visualize import plot_hydrograph, plot_rolling, plot_anomaly, plot_fdc

def run_step1():
    hist = load_historical_flows()
    all_df, current_df = extend_with_current(hist)  # API extends from last hist date -> today
    markers = load_baseline_markers()

    # plots
    plot_hydrograph(all_df, "Emerson", "flow",
                    ylabel="Flow (m³/s)",
                    freeze_marker=markers.get("Emerson_flow"),
                    title="Emerson — Flow since freeze-up",
                    out_path=REPORTS / "emerson_flow_hydro.png")

    plot_hydrograph(all_df, "James Ave", "level",
                    ylabel="Level (m)",
                    freeze_marker=markers.get("James_level"),
                    title="James Ave — Level since freeze-up",
                    out_path=REPORTS / "james_level_hydro.png")

    plot_rolling(all_df, "Emerson", "flow",
                 win=7,
                 freeze_marker=markers.get("Emerson_flow"),
                 title="Emerson — 7-day rolling mean (flow)",
                 out_path=REPORTS / "emerson_flow_rolling.png")

    plot_rolling(all_df, "James Ave", "level",
                 win=7,
                 freeze_marker=markers.get("James_level"),
                 title="James Ave — 7-day rolling mean (level)",
                 out_path=REPORTS / "james_level_rolling.png")

    plot_anomaly(all_df, "Emerson", "flow",
                 freeze_marker=markers.get("Emerson_flow"),
                 title="Emerson — Flow anomalies since freeze-up",
                 out_path=REPORTS / "emerson_flow_anom.png")

    plot_fdc(all_df, "Emerson", "flow",
             freeze_marker=markers.get("Emerson_flow"),
             title="Emerson — Flow duration curve",
             out_path=REPORTS / "emerson_flow_fdc.png")

    # optional save
    all_df.to_csv(DATA / "processed/red_river_timeseries_step1.csv", index=False)

if __name__ == "__main__":
    run_step1()