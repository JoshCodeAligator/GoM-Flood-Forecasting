

"""
Step 4 – Basin Maps / Station Resources

This module:
  1) Loads a Manitoba basin polygon layer (shapefile/GeoJSON).
  2) Creates a stations GeoDataFrame for the Red River pilot (Emerson flow, James Ave level).
  3) Spatially assigns stations → basin via GeoPandas spatial join.
  4) Exports `data/processed/stations_with_basin.csv`.
  5) Renders a simple basin + stations map to `reports/figures/basins_stations_map.png`.

Run:
  python -m hydro_pipeline.basins
"""

from pathlib import Path
import sys
import warnings

import pandas as pd

# Try to import geopandas; give a helpful error if not installed.
try:
    import geopandas as gpd
    from shapely.geometry import Point, box
except Exception as e:
    gpd = None

# Local helpers & paths
from .io import ROOT, DATA, REPORTS

# ----------------------------
# Configuration (edit as needed)
# ----------------------------

# Default location of the basin layer (you can point this to a GeoJSON as well)
DEFAULT_BASIN_PATHS = [
    ROOT / "basin-resources" / "shapefiles" / "manitoba_basins.shp",
    ROOT / "basin-resources" / "shapefiles" / "manitoba_basins.geojson",
    ROOT / "basin-resources" / "manitoba_basins.geojson",
]

# Name column candidates in the basin layer (one of these should exist)
BASIN_NAME_FIELDS = ["BASIN_NAME", "Basin", "NAME", "name"]

# Pilot stations for Red River (provide station_number + approx coords)
# You can extend this list or load from a CSV later.
STATION_CATALOG = [
    # Red River at Emerson (flow)
    {"station": "Emerson", "station_number": "05OC001", "lon": -97.208, "lat": 49.000},
    # Red River at James Avenue in Winnipeg (level)
    {"station": "James Ave", "station_number": "05OC010", "lon": -97.139, "lat": 49.899},
]

OUTPUT_CSV = DATA / "stations_with_basin.csv"
OUTPUT_FIG = REPORTS / "basins_stations_map.png"

# ----------------------------
# Utilities
# ----------------------------

def _require_geopandas():
    if gpd is None:
        raise RuntimeError(
            "GeoPandas/Shapely are required for Step 4.\n"
            "Install with: pip install geopandas shapely pyproj fiona rtree\n"
            "On macOS, you may need: brew install proj geos gdal"
        )

def _first_existing_path(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def load_basin_layer(basin_path: Path | None = None) -> gpd.GeoDataFrame:
    """
    Load the Manitoba basins layer. Returns a GeoDataFrame in WGS84 (EPSG:4326).
    If the file isn't found, builds a minimal bounding box around pilot stations
    as a stand-in so the rest of the pipeline still runs.
    """
    _require_geopandas()

    if basin_path is None:
        basin_path = _first_existing_path(DEFAULT_BASIN_PATHS)

    if basin_path and basin_path.exists():
        gdf = gpd.read_file(basin_path)
        if gdf.crs is None:
            # Assume WGS84 if missing
            warnings.warn("Basin layer has no CRS; assuming WGS84 (EPSG:4326).")
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf = gdf.to_crs(epsg=4326)
        return gdf

    # Fallback: create a simple bounding box polygon around the pilot stations
    warnings.warn(
        "No basin layer found. Using a simple bounding box around stations as a fallback."
    )
    stations = build_stations_gdf(STATION_CATALOG)
    minx, miny, maxx, maxy = (
        stations.geometry.x.min() - 0.3,
        stations.geometry.y.min() - 0.3,
        stations.geometry.x.max() + 0.3,
        stations.geometry.y.max() + 0.3,
    )
    poly = gpd.GeoDataFrame(
        {"BASIN_NAME": ["Red River (bbox fallback)"]},
        geometry=[box(minx, miny, maxx, maxy)],
        crs="EPSG:4326",
    )
    return poly

def select_basin(gdf: "gpd.GeoDataFrame", basin_keyword: str = "Red") -> "gpd.GeoDataFrame":
    """
    Filter the basin layer to rows whose name contains the keyword (case-insensitive).
    If no name column is found, returns the full layer.
    """
    name_col = None
    for col in BASIN_NAME_FIELDS:
        if col in gdf.columns:
            name_col = col
            break
    if name_col is None:
        warnings.warn("No recognizable basin name column; returning full basin layer.")
        return gdf

    m = gdf[name_col].astype(str).str.contains(basin_keyword, case=False, na=False)
    sub = gdf.loc[m]
    if sub.empty:
        warnings.warn(f"No basin matched keyword '{basin_keyword}'; returning full basin layer.")
        return gdf
    return sub

def build_stations_gdf(catalog: list[dict]) -> "gpd.GeoDataFrame":
    """
    Build a GeoDataFrame from a station catalog containing station, lon, lat.
    """
    _require_geopandas()
    df = pd.DataFrame(catalog)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    return gdf

def stations_to_basin_join(basin_gdf: "gpd.GeoDataFrame", stations_gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    """
    Spatially join stations to basin polygons.
    Returns a plain DataFrame with station metadata and the matched basin name when available.
    """
    _require_geopandas()
    # Ensure both layers are in the same CRS
    if basin_gdf.crs is None:
        basin_gdf.set_crs(epsg=4326, inplace=True)
    else:
        basin_gdf = basin_gdf.to_crs(epsg=4326)
    if stations_gdf.crs is None:
        stations_gdf.set_crs(epsg=4326, inplace=True)
    else:
        stations_gdf = stations_gdf.to_crs(epsg=4326)

    # Identify a usable basin name column
    basin_name_col = None
    for col in BASIN_NAME_FIELDS:
        if col in basin_gdf.columns:
            basin_name_col = col
            break
    if basin_name_col is None:
        basin_name_col = "BASIN_NAME"
        basin_gdf = basin_gdf.copy()
        basin_gdf[basin_name_col] = "Unknown"

    joined = gpd.sjoin(stations_gdf, basin_gdf[[basin_name_col, "geometry"]], how="left", predicate="within")
    out = (
        pd.DataFrame(joined.drop(columns=["index_right"]))
        .rename(columns={basin_name_col: "basin"})
        .loc[:, ["station", "station_number", "lon", "lat", "basin"]]
        .sort_values(["basin", "station"])
        .reset_index(drop=True)
    )
    return out

def plot_basins_and_stations(basin_gdf: "gpd.GeoDataFrame", stations_gdf: "gpd.GeoDataFrame", out_path: Path):
    """
    Simple static map: basin polygons + station points, saved to PNG.
    Uses GeoPandas plotting to avoid visual helper coupling.
    """
    _require_geopandas()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ax = basin_gdf.plot(edgecolor="black", facecolor="none", linewidth=1.0, figsize=(8, 6))
    stations_gdf.plot(ax=ax, markersize=35)
    for _, row in stations_gdf.iterrows():
        ax.annotate(
            row["station"],
            xy=(row.geometry.x, row.geometry.y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Red River Basin & Pilot Stations", fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_axisbelow(True)
    ax.grid(True, linewidth=0.3, linestyle="--")
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=200)

# ----------------------------
# Main entrypoint (CLI)
# ----------------------------

def run():
    _require_geopandas()

    # 1) Load basin layer (and filter to Red River if possible)
    basin_layer = load_basin_layer()
    red_basin = select_basin(basin_layer, basin_keyword="Red")

    # 2) Build stations (pilot)
    stations_gdf = build_stations_gdf(STATION_CATALOG)

    # 3) Spatial join → CSV
    df_join = stations_to_basin_join(red_basin, stations_gdf)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_join.to_csv(OUTPUT_CSV, index=False)

    # 4) Map → PNG
    plot_basins_and_stations(red_basin, stations_gdf, OUTPUT_FIG)

    print(f"[OK] stations_with_basin.csv → {OUTPUT_CSV}")
    print(f"[OK] basins_stations_map.png → {OUTPUT_FIG}")
    print(df_join)

if __name__ == "__main__":
    try:
        run()
    except RuntimeError as e:
        # Provide a helpful message if GeoPandas is missing.
        sys.stderr.write(str(e) + "\n")
        sys.exit(1)