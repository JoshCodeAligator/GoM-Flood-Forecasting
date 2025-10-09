import argparse
from pathlib import Path
from . import flows, precip, fall_report, basins, outlooks, api_model, scenarios

def main():
    p = argparse.ArgumentParser(description="GoM Flood Forecasting CLI - run different analysis tasks")
    p.add_argument("task", choices=[
        "flows", "precip", "fall", "basins", "outlooks", "api", "whatif"
    ], help="Select which task to run: "
           "'flows' = flow hydrographs, "
           "'precip' = precipitation plots, "
           "'fall' = fall condition report, "
           "'basins' = basin aggregation, "
           "'outlooks' = flood outlook summaries, "
           "'api' = API model run, "
           "'whatif' = scenario simulations")
    args = p.parse_args()

    outdir = Path("reports/figures")

    if args.task == "flows":
        flows.main(outdir)
    elif args.task == "precip":
        precip.main(outdir)
    elif args.task == "fall":
        fall_report.main(outdir)
    elif args.task == "basins":
        basins.main(outdir)
    elif args.task == "outlooks":
        outlooks.main(outdir)
    elif args.task == "api":
        api_model.main(outdir)
    elif args.task == "whatif":
        scenarios.main(outdir)

if __name__ == "__main__":
    main()