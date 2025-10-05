import argparse
from pathlib import Path
from . import flows, precip, fall_report, basins, outlooks, api_model, scenarios

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=[
        "flows", "precip", "fall", "basins", "outlooks", "api", "whatif"
    ])
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