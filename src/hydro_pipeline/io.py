from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # project_root
DATA = ROOT / "data" / "processed"

def load_baseline():
    return pd.read_csv(DATA / "baseline_observed.csv", parse_dates=["asof_date"])

def load_scenarios():
    return pd.read_csv(DATA / "scenario_peaks.csv")

def load_state():
    return pd.read_csv(DATA / "state_indices.csv", parse_dates=["asof_date"])
