from pathlib import Path
import pandas as pd

# Root paths
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "figures"

# CSV Input/Output Methods
def load_csv(filename, parse_dates=None):
    return pd.read_csv(DATA / filename, parse_dates=parse_dates)

def save_csv(df, filename):
    df.to_csv(DATA / filename, index=False)

# Load Methods from CSV
def load_baseline():
    return load_csv("baseline_observed.csv", parse_dates=["asof_date"])

def load_scenarios():
    return load_csv("scenario_peaks.csv")

def load_state():
    return load_csv("state_indices.csv", parse_dates=["asof_date"])