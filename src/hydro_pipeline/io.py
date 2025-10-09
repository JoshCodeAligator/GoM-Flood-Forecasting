from pathlib import Path
import requests
import pandas as pd

# Root paths
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" 
REPORTS = ROOT / "reports" / "figures"


def reports_path(name: str) -> Path:
    """Return absolute path under project_root/reports/figures and ensure dirs exist."""
    p = REPORTS / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

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

def load_baseline_markers():
    base = load_baseline().rename(columns={"asof_date":"date"})
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    # keep only SI columns to match plotting series
    base = base[["station","date","metric","value_SI","unit_SI","source"]]
    # return dicts for convenience
    def pick(st, met):
        row = base.query("station==@st and metric==@met").iloc[0]
        return {"date": row["date"], "value_SI": row["value_SI"]}
    return {
        "Emerson_flow": pick("Emerson","flow"),
        "James_level":  pick("James Ave","level")
    }


# Fetches Historical Observations of Discharge and Level in Red River.
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

## Extends Historical Data with Current Data Measured in the last 30+ days from ECCC API
STATION_NUMBERS = {
    "Emerson":  "05OC001",  # flow
    "James Ave":"05OC010",  # level
}

def extend_with_current(historical_df, date_end=None):
    date_end = (pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
                if date_end is None else date_end)

    # last historical dates per station/metric
    last_flow  = historical_df.query("station=='Emerson' and metric=='flow'")["date"].max()
    last_level = historical_df.query("station=='James Ave' and metric=='level'")["date"].max()

    frames = []
    if pd.notna(last_flow):
        frames.append(fetch_current_df(
            STATION_NUMBERS["Emerson"], "Emerson",
            start_date=(last_flow + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            end_date=date_end
        ))
    if pd.notna(last_level):
        frames.append(fetch_current_df(
            STATION_NUMBERS["James Ave"], "James Ave",
            start_date=(last_level + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            end_date=date_end
        ))

    current_df = pd.concat(frames, ignore_index=True) if frames else historical_df.iloc[0:0]
    all_df = (pd.concat([historical_df, current_df], ignore_index=True)
                .sort_values(["station","metric","date"])
                .drop_duplicates(subset=["station","metric","date"]))
    return all_df, current_df

# Fetch a fixed last-N-days window and merge with historical
def extend_with_current_last_n_days(historical_df, n: int = 30):
    """
    Fetch a fixed last-N-days window from the ECCC API for the tracked stations
    and merge with the provided historical dataframe.
    Returns: (all_df, current_df)
    """
    # Compute date window as YYYY-MM-DD strings
    today = pd.Timestamp.today().normalize()
    start_date = (today - pd.Timedelta(days=n-1)).strftime("%Y-%m-%d")
    end_date   = today.strftime("%Y-%m-%d")

    frames = [
        fetch_current_df(STATION_NUMBERS["Emerson"], "Emerson",
                         start_date=start_date, end_date=end_date),
        fetch_current_df(STATION_NUMBERS["James Ave"], "James Ave",
                         start_date=start_date, end_date=end_date),
    ]
    # If one of the frames is empty, concat still works
    current_df = pd.concat(frames, ignore_index=True) if frames else historical_df.iloc[0:0]

    all_df = (pd.concat([historical_df, current_df], ignore_index=True)
                .sort_values(["station","metric","date"])
                .drop_duplicates(subset=["station","metric","date"]))
    return all_df, current_df


# Helper Function in extend_with_current methods to fetch data from ECCC API
def fetch_current_df(station_number: str, station_name: str, start_date=None, end_date=None):
    url = "https://api.weather.gc.ca/collections/hydrometric-realtime/items"
    params = {
        "STATION_NUMBER": station_number,             
        "f": "json",
        "limit": 10000,
        "properties": "STATION_NUMBER,DATETIME_LST,DISCHARGE,LEVEL",
    }
    if start_date and end_date:
        params["datetime"] = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"

    r = requests.get(url, params=params); r.raise_for_status()
    js = r.json()
    feats = js.get("features", [])

    # DEBUG: see what the server actually returned
    print(f"[DEBUG] {station_name} URL:", r.url)
    print(f"[DEBUG] {station_name} features:", len(feats))

    rows = []
    for f in feats:
        p = f["properties"]
        dt = pd.to_datetime(p["DATETIME_LST"]).tz_localize(None).normalize()
        if p.get("DISCHARGE") is not None:
            rows.append({"station": station_name, "date": dt, "metric": "flow",
                         "value_SI": p["DISCHARGE"], "unit_SI": "m³/s", "source": "api_realtime"})
        if p.get("LEVEL") is not None:
            rows.append({"station": station_name, "date": dt, "metric": "level",
                         "value_SI": p["LEVEL"], "unit_SI": "m", "source": "api_realtime"})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["station","date","metric","value_SI","unit_SI","source"])
    return df.drop_duplicates(subset=["station","date","metric"]).sort_values(["station","date","metric"])


### MAIN IMPLEMENTATION CODE  
if __name__ == "__main__":
    # 1) Load your historical (Sept–Oct 2025) CSVs and unify structure
    hist_df = load_historical_flows()

    # 2) Pull a fixed last-30-days window from the API and merge
    all_df, current_df = extend_with_current_last_n_days(hist_df, n=30)

    # 3) Persist a Step-1 product for plotting/QA
    save_csv(all_df, "processed/red_river_timeseries_step1.csv")

    # 4) Quick console summary
    print("Historical rows:", len(hist_df))
    print("Current rows (last 30d):", len(current_df))
    print("Merged rows:", len(all_df))
    print(all_df.groupby(["station","metric"])["date"].agg(["min","max"]))


   