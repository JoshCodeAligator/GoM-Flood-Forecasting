from __future__ import annotations
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
from .io import load_basin_layer, ROOT, DATA, REPORTS
from .visualize import plot_basins_and_stations

try:
    import geopandas as gpd
    from shapely.geometry import Point, box
except Exception as e:
    gpd = None


# Name column candidates in the basin layer (one of these should exist)
BASIN_NAME_FIELDS = ["BASIN_NAME", "Basin", "NAME", "name"]

# Pilot stations for Red River (provide station_number + approx coords)
STATION_CATALOG = [
    {"station": "Emerson",      "station_number": "05OC001", "lon": -97.208, "lat": 49.000},
    {"station": "Ste. Agathe",  "station_number": "05OC012", "lon": -97.138, "lat": 49.581},
    {"station": "James Ave",    "station_number": "05OJ015", "lon": -97.139, "lat": 49.899},
    {"station": "Lockport",     "station_number": "05OJ021", "lon": -96.938, "lat": 50.087},
    {"station": "Selkirk",      "station_number": "05OJ005", "lon": -96.883, "lat": 50.143},
    {"station": "Breezy Point", "station_number": "05OJ022", "lon": -96.851, "lat": 50.278},
]

def _require_geopandas():
    """Ensure GeoPandas/Shapely are available for basin/station geometry ops."""
    if gpd is None:
        raise RuntimeError(
            "GeoPandas/Shapely are required for basin loading.\n"
            "Install with: pip install geopandas shapely pyproj fiona rtree"
        )
    if box is None:
        raise RuntimeError(
            "shapely is required for creating fallback geometries.\n"
            "Install with: pip install shapely"
        )

def build_stations_gdf(catalog: list[dict]) -> "gpd.GeoDataFrame":
    """Create a GeoDataFrame from a simple station catalog (lon/lat)."""
    _require_geopandas()
    df = pd.DataFrame(catalog)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    return gdf

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

# ----------------------------
# Main entrypoint (CLI)
# ----------------------------
def main(outdir=None):
    _require_geopandas()

    # Resolve output directories (CLI can override reports root via --outdir)
    reports_root = Path(outdir) if outdir else REPORTS
    csv_out = DATA / "processed" / "stations_with_basin_step4.csv"
    fig_out = reports_root / "step4_basins" / "basins_stations_map.png"
    (csv_out.parent).mkdir(parents=True, exist_ok=True)
    (fig_out.parent).mkdir(parents=True, exist_ok=True)

    # 1) Load basin layer (and filter to Red River if possible)
    basin_layer = load_basin_layer()
    red_basin = select_basin(basin_layer, basin_keyword="Red")

    # 2) Build stations (pilot)
    stations_gdf = build_stations_gdf(STATION_CATALOG)

    # 3) Spatial join → CSV
    df_join = stations_to_basin_join(red_basin, stations_gdf)
    df_join.to_csv(csv_out, index=False)

    # 4) Map → PNG
    plot_basins_and_stations(red_basin, stations_gdf, fig_out)

    print(f"[OK] stations_with_basin.csv → {csv_out}")
    print(f"[OK] basins_stations_map.png → {fig_out}")
    print(df_join)

if __name__ == "__main__":
    main()
