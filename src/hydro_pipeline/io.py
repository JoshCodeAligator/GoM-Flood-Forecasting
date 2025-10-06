from pathlib import Path
import pandas as pd

# Root paths
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" 
REPORTS = ROOT / "reports" / "figures"

# CSV Input/Output Methods
def load_csv(filename, parse_dates=None):
    return pd.read_csv(DATA / filename, parse_dates=parse_dates)

def save_csv(df, filename):
    df.to_csv(DATA / filename, index=False)

# Load Methods from CSV - Processed
def load_baseline():
    return load_csv("processed/baseline_observed.csv", parse_dates=["asof_date"])

def load_scenarios():
    return load_csv("processed/scenario_peaks.csv")

def load_state():
    return load_csv("processed/state_indices.csv", parse_dates=["asof_date"])


# Load Methods from CSV - Processed
def load_historical_flows():
    #Columns: Date (CST),Parameter ,Value (m³/s)
    discharge_df = load_csv("raw/red-river-data/emerson/emerson-daily-discharge.csv", parse_dates=["Date (CST)"])
    
    #Columns: Date (CST),Parameter ,Value (m)
    water_lvl_df = load_csv("raw/red-river-data/james-ave/james-ave-water-level.csv", parse_dates=["Date (CST)"])
    
    #Renaming All Columns in DataFrames
    discharge_df = discharge_df.rename(columns={"Date (CST)": "date", "Parameter ": "metric", "Value (m³/s)": "value_SI"})
    water_lvl_df = water_lvl_df.rename(columns={"Date (CST)": "date", "Parameter ": "metric", "Value (m)": "value_SI"})

    #Assumption for basin=Red River. asof_date is the same as date. Assigning New Columns based on baseline_observed.csv file.
    #Baseline Columns: basin, station, asof_date, metric, value, unit, value_SI, unit_SI, source. We only care about value_SI and unit_SI for now. 
    discharge_df = discharge_df.assign(
        station="Emerson",
        metric="flow",
        unit_SI="m³/s",
        source="emerson-daily-discharge.csv"
    )
    
    water_lvl_df = water_lvl_df.assign(
        station="James Ave",
        metric="level",
        unit_SI="m",
        source="james-ave-water-level.csv"
    )
    hist_df = pd.concat([discharge_df, water_lvl_df], ignore_index=True)
    
    print(hist_df)
    return hist_df

## Room for API call methods and Reporting

if __name__ == "__main__":
   load_historical_flows()


   