from __future__ import annotations
from pathlib import Path
"""
Step 2 – Precipitation Context in relation to flood forecasting.

This module provides:
- parse_map_date_from_filename(): best-effort date parser from JPEG filenames
- overlay_precip_map(): qualitative overlay of basins + stations on top of a precipitation JPEG
- basin_average_precip_from_raster(): (stub) numeric extraction for GeoTIFF/NetCDF rasters
- append_precip_timeseries(): helper to persist basin-averaged precipitation

Inputs: 
- load_basins(): read Manitoba basin polygons and ensure CRS=EPSG:4326. Method from io.py.
- load_stations(): read a simple stations CSV (lon/lat in WGS84) and return a pandas DataFrame. Method from io.py.

Outputs:
- reports/figures/precip_overlay_<filename>.png

Notes:
- JPEGs on the Manitoba site are not georeferenced. We place the image as a background
  stretched to the geographic extent of the basin polygons for qualitative context only.
- For quantitative work, use gridded rasters (GeoTIFF/NetCDF) and implement
  `basin_average_precip_from_raster()` using rasterio/xarray.
"""
import re
from typing import Optional, Iterable

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import geopandas as gpd
from .io import DATA, MAPS, RESOURCES, REPORTS, PROCESSED, reports_path, load_basins, load_stations


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
    ax.imshow(img, extent=[minx, maxx, miny, maxy], origin="lower", alpha=alpha_img)

    # Plot basins
    basins_gdf.boundary.plot(ax=ax, color=basin_edge, linewidth=basin_linewidth, label="Basins")

    # Plot stations (drop rows without coordinates)
    if isinstance(stations_df, pd.DataFrame) and not stations_df.empty:
        s = stations_df.dropna(subset=["lon", "lat"])
        if not s.empty:
            ax.scatter(
                s["lon"], s["lat"],
                s=25, c="tab:red", marker="o", label="Stations", zorder=3
            )
            # Text labels with slight offset
            for _, r in s.iterrows():
                ax.text(float(r["lon"]) + 0.05, float(r["lat"]) + 0.05, str(r["station"]), fontsize=8, color="tab:red")

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


def _discover_maps(folder: Path = MAPS) -> list[Path]:
    exts = (".jpg", ".jpeg", ".png")
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in exts])


# Executes Precipitation Overlay for all discovered JPEGs
def main(outdir=None):
    basins = load_basins()
    stations_ret = load_stations()
    stations = stations_ret[0] if isinstance(stations_ret, tuple) else stations_ret
    
    (REPORTS / "step2_precip").mkdir(parents=True, exist_ok=True)

    maps = _discover_maps(MAPS)
    if not maps:
        print(f"[precip] No JPEG maps found under: {MAPS}")
        return
    for m in maps:
        out = REPORTS / "step2_precip" / f"precip_overlay_{m.stem}.png"
        print(f"[precip] Overlay → {out.name}")
        overlay_precip_map(m, basins, stations, out_path=out)


if __name__ == "__main__":
    main()