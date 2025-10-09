

from __future__ import annotations

"""
Step 2 – Precipitation Context helpers.

This module provides:
- load_basins(): read Manitoba basin polygons and ensure CRS=EPSG:4326
- load_stations(): read a simple stations CSV (lon/lat in WGS84) and return a pandas DataFrame
- parse_map_date_from_filename(): best-effort date parser from JPEG filenames
- overlay_precip_map(): qualitative overlay of basins + stations on top of a precipitation JPEG
- basin_average_precip_from_raster(): (stub) numeric extraction for GeoTIFF/NetCDF rasters
- append_precip_timeseries(): helper to persist basin-averaged precipitation

Outputs:
- reports/figures/precip_overlay_<filename>.png

Notes:
- JPEGs on the Manitoba site are not georeferenced. We place the image as a background
  stretched to the geographic extent of the basin polygons for qualitative context only.
- For quantitative work, use gridded rasters (GeoTIFF/NetCDF) and implement
  `basin_average_precip_from_raster()` using rasterio/xarray.
"""

from pathlib import Path
import re
from typing import Optional, Iterable

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import geopandas as gpd

# ---- Project paths (aligned with io.py conventions) ----
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MAPS = ROOT / "precip-maps"
RESOURCES = ROOT / "basin-resources"
REPORTS = ROOT / "reports" / "figures"

# ---- Public API ----


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
        # Try common names; adjust as needed for your resource filenames.
        candidates = [
            RESOURCES / "shapefiles" / "manitoba_basins.shp",
            RESOURCES / "shapefiles" / "mb_basins.shp",
            RESOURCES / "shapefiles" / "manitoba_basins.geojson",
            RESOURCES / "manitoba_basins.geojson",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                "Could not find a basins shapefile/geojson under basin-resources/. "
                "Place it under basin-resources/shapefiles/ and point load_basins() to it."
            )

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Basins file loaded but contains no features: {path}")

    # Ensure lon/lat for plotting with imshow extent
    if gdf.crs is None:
        # Assume WGS84 if missing; change if your data uses a projected CRS.
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def load_stations(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load a minimal stations CSV with columns:
        station, type, lon, lat, basin (optional)

    If no path is provided, tries basin-resources/stations.csv.

    Returns
    -------
    pandas.DataFrame
    """
    if path is None:
        path = RESOURCES / "stations.csv"
    if not path.exists():
        # Provide a tiny default with Red River test points so plotting doesn't fail.
        df = pd.DataFrame(
            [
                {"station": "Emerson", "type": "flow", "lon": -97.21, "lat": 49.00, "basin": "Red"},
                {"station": "James Ave", "type": "level", "lon": -97.14, "lat": 49.90, "basin": "Red"},
            ]
        )
        return df

    df = pd.read_csv(path)
    required = {"station", "lon", "lat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"stations.csv missing required columns: {missing}")
    return df


def parse_map_date_from_filename(p: Path) -> Optional[pd.Timestamp]:
    """
    Extract date from a precipitation JPEG filename, e.g.,
        HRDPA_7day_obs_2025-10-05.jpeg -> 2025-10-05

    Returns pandas.Timestamp or None if not found.
    """
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", p.name)
    if m:
        try:
            return pd.to_datetime(m.group(1))
        except Exception:
            return None
    return None


def overlay_precip_map(
    map_path: Path,
    basins_gdf: gpd.GeoDataFrame,
    stations_df: pd.DataFrame,
    out_path: Optional[Path] = None,
    title: Optional[str] = None,
    alpha_img: float = 0.6,
    basin_edge: str = "black",
    basin_linewidth: float = 0.6,
):
    """
    Draw basins + stations on top of a precipitation JPEG for qualitative context.

    Parameters
    ----------
    map_path : Path
        Path to the JPEG map (not georeferenced).
    basins_gdf : GeoDataFrame (EPSG:4326)
        Basin polygons.
    stations_df : DataFrame
        Columns: station, lon, lat, (optional) type/basin
    out_path : Optional[Path]
        Where to save the PNG. If None, saves to reports/figures/precip_overlay_<stem>.png
    title : Optional[str]
        Plot title. If None, auto-generates.
    alpha_img : float
        Opacity of the JPEG background.
    basin_edge : str
        Color for basin edges.
    basin_linewidth : float
        Line width for basin edges.
    """
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Determine map extent from basin bounds (qualitative alignment)
    minx, miny, maxx, maxy = basins_gdf.total_bounds

    # Load image
    img = mpimg.imread(str(map_path))

    # Prepare figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Show JPEG stretched to basin extent
    ax.imshow(img, extent=[minx, maxx, miny, maxy], origin="upper", alpha=alpha_img)

    # Plot basins
    basins_gdf.boundary.plot(ax=ax, color=basin_edge, linewidth=basin_linewidth)

    # Plot stations
    if not stations_df.empty:
        ax.scatter(
            stations_df["lon"], stations_df["lat"],
            s=25, c="tab:red", marker="o", label="Stations", zorder=3
        )
        # Text labels with slight offset
        for _, r in stations_df.iterrows():
            ax.text(r["lon"] + 0.05, r["lat"] + 0.05, r["station"], fontsize=8, color="tab:red")

    # Title & cosmetics
    map_date = parse_map_date_from_filename(map_path)
    auto_title = f"Precipitation Map Overlay — {map_path.name}"
    if map_date is not None:
        auto_title += f" (as of {map_date.date()})"
    ax.set_title(title or auto_title)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.legend(loc="lower right")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # Save
    if out_path is None:
        out_path = REPORTS / f"precip_overlay_{map_path.stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def basin_average_precip_from_raster(raster_path: Path, basins_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    (Stub) Compute basin-averaged precipitation from a gridded raster (GeoTIFF/NetCDF).

    Suggested implementation (later):
    - If GeoTIFF: use rasterio + rasterstats (zonal statistics) or rasterio.mask + numpy mean.
    - If NetCDF (HRDPA/HRDPS): use xarray to open dataset, select variable/time, clip with basins_gdf,
      and compute spatial mean for each basin.

    Returns
    -------
    DataFrame with columns: ['date', 'basin', 'precip_mm']
    """
    raise NotImplementedError(
        "Numeric extraction not yet implemented. Use JPEG overlay for qualitative context now; "
        "switch to GeoTIFF/NetCDF + rasterio/xarray for quantitative basin means."
    )


def append_precip_timeseries(dfs: Iterable[pd.DataFrame], outfile: Optional[Path] = None) -> Path:
    """
    Append one or more precip timeseries DataFrames and persist to processed CSV.

    Each input df should have columns at least: ['date','basin','precip_mm'].

    Returns
    -------
    Path to the written CSV.
    """
    out = outfile or (PROCESSED / "precip_basin_daily.csv")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = pd.concat(list(dfs), ignore_index=True)
    # Basic order & dtypes
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["basin", "date"])
    df.to_csv(out, index=False)
    return out


# ---- Convenience CLI entrypoint (optional) ----
def _discover_maps(folder: Path = MAPS) -> list[Path]:
    exts = (".jpg", ".jpeg", ".png")
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in exts])


def run_precip_overlay():
    basins = load_basins()
    stations = load_stations()
    maps = _discover_maps(MAPS)
    if not maps:
        print(f"[precip] No JPEG maps found under: {MAPS}")
        return
    for m in maps:
        out = REPORTS / f"precip_overlay_{m.stem}.png"
        print(f"[precip] Overlay → {out.name}")
        overlay_precip_map(m, basins, stations, out_path=out)


if __name__ == "__main__":
    run_precip_overlay()