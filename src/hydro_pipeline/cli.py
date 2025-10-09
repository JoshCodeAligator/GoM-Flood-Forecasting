import argparse
from pathlib import Path
from .io import REPORTS
import importlib
import sys

# Map task -> module path and callable name to keep imports lazy
TASKS = {
    "flows":    ("hydro_pipeline.flows", "main"),
    "precip":   ("hydro_pipeline.precip", "main"),
    "fall":     ("hydro_pipeline.fall_report", "main"),
    "basins":   ("hydro_pipeline.basins", "main"),
    "outlooks": ("hydro_pipeline.outlooks", "main"),
    "api":      ("hydro_pipeline.api_model", "main"),
    "whatif":   ("hydro_pipeline.scenarios", "main"),
}

def _run_task(task: str, outdir: Path) -> None:
    if task not in TASKS:
        raise SystemExit(f"Unknown task: {task}")
    mod_name, func_name = TASKS[task]
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        sys.stderr.write(f"[ERROR] Could not import module '{mod_name}'. Did you create the file?\n{e}\n")
        raise SystemExit(1)
    func = getattr(mod, func_name, None)
    if func is None:
        sys.stderr.write(f"[ERROR] Module '{mod_name}' has no callable '{func_name}'.\n")
        raise SystemExit(1)
    func(outdir)


def main():
    p = argparse.ArgumentParser(
        description="GoM Flood Forecasting CLI - run different analysis tasks"
    )
    p.add_argument(
        "task",
        choices=list(TASKS.keys()),
        help=(
            "Select which task to run: "
            "'flows' = flow hydrographs, "
            "'precip' = precipitation plots, "
            "'fall' = fall condition report, "
            "'basins' = basin aggregation, "
            "'outlooks' = flood outlook summaries, "
            "'api' = API model run, "
            "'whatif' = scenario simulations"
        ),
    )
    args = p.parse_args()

    outdir = REPORTS
    outdir.mkdir(parents=True, exist_ok=True)

    _run_task(args.task, outdir)


if __name__ == "__main__":
    main()