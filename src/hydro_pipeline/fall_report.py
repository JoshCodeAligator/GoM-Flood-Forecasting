

"""
Step 3 – Fall Conditions (Baseline Wetness & Baseflows)

This module extracts *baseline wetness* information (e.g., soil moisture classes at freeze-up)
from the Fall Conditions PDF and persists it to `data/processed/state_indices.csv`.
It is intentionally self-contained (like `precip.py` in Step 2): it handles parsing,
upserts to CSV, and produces a small QA plot.

If structured tables are not reliably parsed, it falls back to safe manual overrides
(populated with the values you've already curated for Red and Assiniboine).
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import List, Dict, Optional

import pandas as pd

# Optional dependencies (we handle gracefully if missing)
try:
    import pdfplumber  # robust for text extraction
except Exception:  # pragma: no cover
    pdfplumber = None

try:
    import tabula  # table extraction fallback (requires Java)
except Exception:  # pragma: no cover
    tabula = None

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[2]
PDFS = ROOT / "pdf-reports"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

FALL_PDF = PDFS / "2024_fall_conditions_report.pdf"

# ---------- Utility: CSV upsert ----------
def _upsert_csv(out_path: Path, df_new: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """
    Append/merge df_new into out_path on key_cols (keep last), return merged.
    Creates file if not present.
    """
    df_new = df_new.copy()
    for c in key_cols:
        if c not in df_new.columns:
            raise ValueError(f"Missing key column '{c}' for upsert into {out_path.name}")

    if out_path.exists():
        cur = pd.read_csv(out_path, parse_dates=[c for c in df_new.columns if "date" in c])
        merged = (
            pd.concat([cur, df_new], ignore_index=True)
              .sort_values(key_cols)
              .drop_duplicates(subset=key_cols, keep="last")
        )
    else:
        merged = df_new

    merged.to_csv(out_path, index=False)
    return merged

# ---------- PDF helpers ----------
def _all_page_text(pdf_path: Path) -> List[str]:
    """
    Return list of page texts from the PDF (one string per page).
    If pdfplumber is unavailable or parsing fails, return [].
    """
    pages: List[str] = []
    if pdfplumber is None:
        return pages
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pg in pdf.pages:
                txt = pg.extract_text() or ""
                pages.append(txt)
    except Exception:
        # Silent fallback; caller will move to manual overrides or tabula
        pass
    return pages

def _search_soil_moisture_blocks(pages: List[str]) -> Dict[str, str]:
    """
    Heuristic text search to find qualitative soil moisture statements by basin.
    Returns dict {basin_name_lower: class_value}, e.g. {"red": "above_normal_to_well_above_normal"}.
    """
    results: Dict[str, str] = {}
    if not pages:
        return results

    page_text = "\n".join(pages).lower()

    # Simple, resilient regex patterns
    patterns = [
        (r"red river.*?(above normal.*?well above normal|above normal|below normal|near normal)", "red"),
        (r"assiniboine.*?(near normal.*?below normal|below normal|near normal|above normal)", "assiniboine"),
    ]

    for pat, basin in patterns:
        m = re.search(pat, page_text, flags=re.DOTALL)
        if m:
            raw = m.group(1)
            # Normalize to compact class labels
            norm = raw.replace(" ", "_").replace("-", "_")
            norm = norm.replace("__", "_")
            # Collapse common phrases
            norm = norm.replace("above_normal_to_well_above_normal", "above_normal_to_well_above_normal")
            norm = norm.replace("near_normal_to_below_normal", "near_normal_to_below_normal")
            results[basin] = norm
    return results

# ---------- Core extractors ----------
def extract_soil_wetness(pdf_path: Path) -> pd.DataFrame:
    """
    Try to parse soil moisture (qualitative class) from the Fall PDF.
    If parsing is inconclusive, fall back to manual overrides that match your curated values.
    """
    rows: List[Dict[str, Optional[str]]] = []
    parsed = _search_soil_moisture_blocks(_all_page_text(pdf_path))

    # Manual overrides (authoritative values you’ve already documented)
    manual_overrides = {
        "red": "above_normal_to_well_above_normal",
        "assiniboine": "near_normal_to_below_normal",
    }

    # Merge parsed with manual (parsed wins if present)
    final_classes = {
        "red": parsed.get("red", manual_overrides["red"]),
        "assiniboine": parsed.get("assiniboine", manual_overrides["assiniboine"]),
    }

    # Build tidy rows
    rows.append({
        "basin": "Red",
        "asof_date": "2024-12-05",
        "metric": "soil_wetness_class",
        "value": final_classes["red"],
        "unit": "class",
        "value_SI": None,
        "unit_SI": None,
        "note": "Soil Moisture at Freeze-up (parsed/confirmed)",
        "source": "Fall Report | Soil Moisture at Freeze-up",
    })
    rows.append({
        "basin": "Assiniboine",
        "asof_date": "2024-12-05",
        "metric": "soil_wetness_class",
        "value": final_classes["assiniboine"],
        "unit": "class",
        "value_SI": None,
        "unit_SI": None,
        "note": "Soil Moisture at Freeze-up (parsed/confirmed)",
        "source": "Fall Report | Soil Moisture at Freeze-up",
    })

    df = pd.DataFrame(rows)
    df["asof_date"] = pd.to_datetime(df["asof_date"])
    return df

def extract_freezeup_flows(pdf_path: Path) -> pd.DataFrame:
    """
    Optional: parse freeze-up/baseflow tables if present.
    By default, we return an empty DataFrame; you can implement Tabula parsing if needed.

    Expected schema (matching baseline_observed.csv):
    ["basin","station","asof_date","metric","value","unit","value_SI","unit_SI","source"]
    """
    # If you decide to parse a table, uncomment and implement:
    # if tabula is not None:
    #     tables = tabula.read_pdf(str(pdf_path), pages="all", multiple_tables=True, lattice=True)
    #     # TODO: locate the right table and normalize columns into the expected schema
    #     ...
    cols = ["basin","station","asof_date","metric","value","unit","value_SI","unit_SI","source"]
    return pd.DataFrame(columns=cols)

# ---------- QA plot ----------
def plot_soil_wetness_bars(df: pd.DataFrame, out_path: Path = REPORTS/"soil_wetness_bars.png") -> None:
    """
    Simple bar chart for quick QA of parsed/confirmed classes.
    """
    import matplotlib.pyplot as plt

    plot_df = df.copy()
    plot_df = plot_df[["basin","value"]].set_index("basin")

    ax = plot_df["value"].astype(str).plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("Soil wetness class")
    ax.set_title("Soil Moisture at Freeze-up (Fall 2024)")

    # Annotate class on bars
    for p, txt in zip(ax.patches, plot_df["value"].tolist()):
        ax.annotate(txt, (p.get_x() + p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", fontsize=8, rotation=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ---------- Orchestration ----------
def run_fall_step():
    """
    Main entrypoint for Step 3:
    - Extract soil wetness classes (qualitative).
    - Upsert into data/processed/state_indices.csv.
    - (Optional) Extract freeze-up/baseflow table and upsert into baseline_observed.csv.
    - Emit a QA bar plot to reports/figures/.
    """
    if not FALL_PDF.exists():
        raise FileNotFoundError(f"Expected Fall PDF at {FALL_PDF}")

    # 1) Soil wetness (qualitative)
    wet = extract_soil_wetness(FALL_PDF)
    _upsert_csv(PROCESSED / "state_indices.csv", wet,
                key_cols=["basin", "asof_date", "metric"])
    plot_soil_wetness_bars(wet)

    # 2) Optional: freeze-up/baseflow numbers (skip if you already captured these)
    extra = extract_freezeup_flows(FALL_PDF)
    if not extra.empty:
        _upsert_csv(PROCESSED / "baseline_observed.csv", extra,
                    key_cols=["basin","station","asof_date","metric"])

    print("Step 3 complete:")
    print(f"  - Updated: {PROCESSED / 'state_indices.csv'}")
    if not extra.empty:
        print(f"  - Updated: {PROCESSED / 'baseline_observed.csv'}")
    print(f"  - Figure:  {REPORTS / 'soil_wetness_bars.png'}")

if __name__ == "__main__":
    run_fall_step()