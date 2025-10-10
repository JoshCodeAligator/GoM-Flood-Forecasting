from pathlib import Path
import requests
import pandas as pd
import geopandas as gpd
from typing import Optional

# ABSOLUTE ROOT PATHS DEFINED FOR CORRECT DATA STORAGE AND USE
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" 
REPORTS = ROOT / "reports" / "figures"
PROCESSED = DATA / "processed"
MAPS = DATA / "raw" / "precip-maps"
RESOURCES = ROOT / "basin-resources"

# ---- Red River stations (authoritative) ----

STATIONS_ID_TO_NAME = {
    "05OC001": "Emerson",
    "05OC012": "Ste. Agathe",
    "05OJ015": "James Ave",
    "05OJ021": "Lockport",
    "05OJ005": "Selkirk",
    "05OJ022": "Breezy Point",
}
STATIONS_NAME_TO_ID = {v: k for k, v in STATIONS_ID_TO_NAME.items()}

# --- Helper: authoritative station catalog DataFrame ---
def station_catalog() -> pd.DataFrame:
    """
    Return a small catalog dataframe mapping station_id & station name
    from the authoritative dictionaries above. Lon/lat may be appended
    later by load_stations() if available in stations.csv.
    """
    rows = [{"station_id": sid, "station": name} for sid, name in STATIONS_ID_TO_NAME.items()]
    return pd.DataFrame(rows, columns=["station_id", "station"])


def reports_path(name: str) -> Path:
    """Return absolute path under project_root/reports/figures and ensure dirs exist."""
    p = REPORTS / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

## CSV HELPER METHODS
def load_csv(filename, parse_dates=None):
    return pd.read_csv(DATA / filename, parse_dates=parse_dates)

def save_csv(df, filename):
    df.to_csv(DATA / filename, index=False)

## LOAD METHODS FOR CSV FILES FROM DATA/PROCESSED DIRECTORY
def load_baseline():
    return load_csv("processed/baseline_observed.csv", parse_dates=["asof_date"])

def load_scenarios():
    return load_csv("processed/scenario_peaks.csv")

def load_state():
    return load_csv("processed/state_indices.csv", parse_dates=["asof_date"])

def load_baseline_markers():
    """
    Return a dict like {"Emerson_flow": {"date": dt, "value_SI": val}, ...}
    for any station/metric rows present in baseline_observed.csv that match our station list.
    """
    base = load_baseline().rename(columns={"asof_date": "date"})
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    base = base[["station", "date", "metric", "value_SI", "unit_SI", "source"]]
    # Keep only stations we care about
    base = base[base["station"].isin(STATIONS_NAME_TO_ID.keys())].copy()
    out = {}
    for _, r in base.iterrows():
        key = f"{r['station']}_{r['metric']}"
        out[key] = {"date": r["date"], "value_SI": r["value_SI"]}
    return out

def get_freeze_up_marker(station, metric):
    """Return the freeze-up date/value from baseline_observed.csv if available."""
    base = load_baseline().rename(columns={"asof_date": "date"})
    match = base.query("station == @station and metric == @metric")
    if not match.empty:
        row = match.iloc[0]
        return {"date": pd.to_datetime(row["date"]), "value_SI": row["value_SI"]}
    return None

#########################################################################
### STEP 1 LOAD and FETCH METHODS FOR WATER LEVEL AND DISCHARGE ANALYSIS. USES HISTORICAL OBSERVATIONS FROM CSVs AND EXTENDS WITH CURRENT DATA FROM ECCC API.
# Fetches Historical Observations of Discharge and Level in Red River. 
def load_historical_flows():
    """
    Load and merge historical HYDAT-format CSVs for all Red River stations.
    Each file is expected to be in wide HYDAT format:
    ID, PARAM, TYPE, YEAR, DD, Jan, SYM, Feb, SYM, ..., Dec, SYM
    """

    import glob

    raw_dir = DATA / "raw" / "stations" / "red-river"
    all_frames = []

    for csv_path in sorted(glob.glob(str(raw_dir / "*.csv"))):
        # HYDAT "Historical – Daily" wide files: row 0 is a title; real headers at row 1
        df = pd.read_csv(csv_path, header=1, skipinitialspace=True)

        required = {"ID", "PARAM", "TYPE", "YEAR", "DD"}
        if not required.issubset(df.columns):
            raise ValueError(f"{csv_path} missing expected HYDAT columns {required}")

        station_id = str(df.iloc[0]["ID"]).strip()
        param = int(pd.to_numeric(df.iloc[0]["PARAM"], errors="coerce"))
        metric = "flow" if param == 1 else "level"
        unit_SI = "m³/s" if param == 1 else "m"
        station_name = STATIONS_ID_TO_NAME.get(station_id, station_id)

        month_cols = [m for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul",
                                  "Aug","Sep","Oct","Nov","Dec"] if m in df.columns]
        if not month_cols:
            continue

        melted = df.melt(
            id_vars=["YEAR","DD"],
            value_vars=month_cols,
            var_name="month",
            value_name="value_SI"
        )
        melted["month_num"] = pd.to_datetime(melted["month"], format="%b").dt.month
        melted["date"] = pd.to_datetime(
            dict(year=melted["YEAR"], month=melted["month_num"], day=melted["DD"]),
            errors="coerce"
        )
        melted["value_SI"] = pd.to_numeric(melted["value_SI"], errors="coerce")

        # Attach metadata and retain canonical columns
        melted = (
            melted.dropna(subset=["date","value_SI"])
                  .assign(
                      station=station_name,
                      metric=metric,
                      unit_SI=unit_SI,
                      source=Path(csv_path).name
                  )[["station","date","metric","value_SI","unit_SI","source"]]
        )

        # Trim per-station series to start at 2023-01-01
        melted = melted[melted["date"] >= pd.Timestamp("2023-01-01")]

        if not melted.empty:
            all_frames.append(melted)

    if not all_frames:
        raise FileNotFoundError(f"No valid HYDAT CSVs found under {raw_dir}")

    merged = pd.concat(all_frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged = (merged
              .sort_values(["station","metric","date"])
              .drop_duplicates(subset=["station","metric","date"])
              .reset_index(drop=True))
    return merged

# Extends Historical Data with Current Data Measured in the last 30+ days from ECCC API
def extend_with_current(historical_df: pd.DataFrame, date_end: Optional[str] = None):
    """
    For each station/metric in historical_df, fetch API data from the day after the last
    historical point up to date_end (default today) and merge de-duplicated.
    """
    date_end = (pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
                if date_end is None else date_end)

    frames = []
    # Find last date per (station, metric) that we already have
    last = (historical_df
            .groupby(["station", "metric"])["date"]
            .max()
            .dropna())

    for (station_name, metric), last_dt in last.items():
        station_id = STATIONS_NAME_TO_ID.get(station_name)
        if not station_id:
            continue
        # Pull both flow & level, then keep rows matching this metric
        df_cur = fetch_current_df(
            station_number=station_id,
            station_name=station_name,
            start_date=(last_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            end_date=date_end,
        )
        if not df_cur.empty:
            df_cur = df_cur.query("metric == @metric")[["station","date","metric","value_SI","unit_SI","source"]]
            frames.append(df_cur)

    current_df = pd.concat(frames, ignore_index=True) if frames else historical_df.iloc[0:0]
    cols = ["station","date","metric","value_SI","unit_SI","source"]
    all_df = (pd.concat([historical_df, current_df], ignore_index=True)
                .sort_values(["station","metric","date"])
                .drop_duplicates(subset=["station","metric","date"]))
    all_df = all_df[cols]
    current_df = current_df[cols] if not current_df.empty else current_df
    print(current_df)
    return all_df, current_df

# Fetch a fixed last-N-days window and merge with historical. More Specific Variant of extend_with_current
def extend_with_current_last_n_days(historical_df: pd.DataFrame, n: int = 30):
    """
    Fetch a fixed last-N-days window from the ECCC API for ALL Red River stations
    (flow & level where available) and merge with historical_df (de-duplicated).
    """
    today = pd.Timestamp.today().normalize()
    start_date = (today - pd.Timedelta(days=n-1)).strftime("%Y-%m-%d")
    end_date   = today.strftime("%Y-%m-%d")

    frames = []
    for station_name, station_id in STATIONS_NAME_TO_ID.items():
        df_cur = fetch_current_df(
            station_number=station_id,
            station_name=station_name,
            start_date=start_date,
            end_date=end_date,
        )
        if not df_cur.empty:
            # Only keep requested columns and order
            df_cur = df_cur[["station","date","metric","value_SI","unit_SI","source"]]
            frames.append(df_cur)

    current_df = pd.concat(frames, ignore_index=True) if frames else historical_df.iloc[0:0]
    cols = ["station","date","metric","value_SI","unit_SI","source"]
    all_df = (pd.concat([historical_df, current_df], ignore_index=True)
                .sort_values(["station","metric","date"])
                .drop_duplicates(subset=["station","metric","date"]))
    all_df = all_df[cols]
    current_df = current_df[cols] if not current_df.empty else current_df
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
        dt = pd.to_datetime(p["DATETIME_LST"], utc=True).tz_convert(None).normalize()
        if p.get("DISCHARGE") is not None:
            rows.append({"station": station_name, "date": dt, "metric": "flow",
                         "value_SI": p["DISCHARGE"], "unit_SI": "m³/s", "source": "api_realtime"})
        if p.get("LEVEL") is not None:
            rows.append({"station": station_name, "date": dt, "metric": "level",
                         "value_SI": p["LEVEL"], "unit_SI": "m", "source": "api_realtime"})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["station","date","metric","value_SI","unit_SI","source"])
    df = df.drop_duplicates(subset=["station","date","metric"]).sort_values(["station","date","metric"])
    return df[["station","date","metric","value_SI","unit_SI","source"]]

#########################################################################
### STEP 2 LOAD METHODS TO OBSERVE HOW PRECIPITATION AFFECTS BASINS. USES JPEG MAPS FOR NOW, BUT CAN BE EXTENDED TO RASTER EXTRACTION LATER.
def load_basins(path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """
    Load Manitoba basins/watersheds polygons.

    Parameters
    ----------
    path : Optional[Path]
        Path to a vector dataset (Shapefile directory, .shp, or .geojson).
        If None, tries common defaults under basin-resources/shapefiles.

    Returns
    -------
    GeoDataFrame in EPSG:4326 (lon/lat)
    """
    if path is None:
        candidates = [
            DATA / "raw" / "basin-resources" / "shapefiles" / "500k_shp" / "500k_hyd-py.shp",
            DATA / "raw" / "basin-resources" / "shapefiles" / "500k_shp" / "500k_hyd-py.geojson",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                "Could not find a basins shapefile or GeoJSON under 'data/raw/basin-resources/shapefiles/500k_shp/'.\n"
                "Expected one of: 500k_hyd-py.shp or 500k_hyd-py.geojson. Please provide the file."
            )

    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read basins file {path}: {e}")

    if gdf.empty:
        raise ValueError(f"Basins file loaded but contains no features: {path}")

    # Filter out invalid geometries (e.g., empty, None, or invalid)
    if "geometry" not in gdf.columns:
        raise ValueError(f"Basins file at {path} does not contain a 'geometry' column.")
    gdf = gdf[~gdf["geometry"].isna() & gdf["geometry"].notnull()]
    gdf = gdf[gdf.is_valid]
    if gdf.empty:
        raise ValueError(
            f"All geometries in the basins file {path} are invalid or empty. "
            "Please check the data source."
        )

    # Reproject to EPSG:4326 (WGS84 lon/lat)
    try:
        # If no CRS defined, assume NAD83 / UTM Zone 14N (common for Manitoba base maps)
        if gdf.crs is None:
            print("[INFO] CRS missing; assuming NAD83 / UTM Zone 14N (EPSG:26914)")
            gdf.set_crs(epsg=26914, inplace=True)

        # Convert everything to geographic (lon/lat)
        gdf = gdf.to_crs(epsg=4326)

    except Exception as e:
        raise RuntimeError(f"Could not reproject basins to EPSG:4326: {e}")

    # Buffer by zero to clean up geometry errors (e.g., self-intersections)
    try:
        gdf["geometry"] = gdf["geometry"].buffer(0)
    except Exception as e:
        raise RuntimeError(f"Failed to clean geometries by buffering: {e}")

    # Remove any geometries that are still invalid after buffering
    gdf = gdf[gdf.is_valid]
    if gdf.empty:
        raise ValueError(
            f"All geometries in the basins file {path} are invalid after cleaning. "
            "Please check the data source."
        )

    return gdf


def load_stations(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load a minimal stations CSV with columns:
        station, lon, lat, (optional: basin, type, station_id)

    If no CSV exists, return a DataFrame from the authoritative
    Red River list with station_id & station columns, and NaN lon/lat.
    """
    cat = station_catalog()  # station_id, station

    if path is None:
        path = RESOURCES / "stations.csv"

    if not path.exists():
        # Return catalog with empty lon/lat so plots can still run.
        out = cat.copy()
        out["lon"] = pd.NA
        out["lat"] = pd.NA
        out["basin"] = "Red"
        return out[["station","station_id","lon","lat","basin"]]

    df = pd.read_csv(path)

    # Normalize expected columns
    if "station_id" not in df.columns:
        # add IDs by mapping station names
        df["station_id"] = df["station"].map(STATIONS_NAME_TO_ID)

    # Merge to ensure we only keep stations from the authoritative list
    merged = pd.merge(cat, df, on=["station","station_id"], how="left")
    # required columns for mapping
    for col in ["lon","lat"]:
        if col not in merged.columns:
            merged[col] = pd.NA
    if "basin" not in merged.columns:
        merged["basin"] = "Red"

    return merged[["station","station_id","lon","lat","basin"]]


### MAIN IMPLEMENTATION CODE  
if __name__ == "__main__":
    # 1) Load your historical CSVs and unify structure
    hist_df = load_historical_flows()

    # 2) Pull continuing data from the API and merge
    all_df, current_df = extend_with_current(hist_df)

    # 3) Persist a Step-1 product for plotting/QA
    save_csv(all_df, "processed/red_river_timeseries_step1.csv")
    expected_cols = ["station","date","metric","value_SI","unit_SI","source"]
    assert list(all_df.columns) == expected_cols, f"Unexpected columns in merged Step-1 timeseries: {list(all_df.columns)}"

    # 4) Quick console summary
    print("Historical rows:", len(hist_df))
    print("Current rows (last 30d):", len(current_df))
    print("Merged rows:", len(all_df))
    print(all_df.groupby(["station","metric"])["date"].agg(["min","max"]))