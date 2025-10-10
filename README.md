# GoM-Flood-Forecasting – Step 1 & Step 2 Exploration. 

This repository contains my exploratory work toward understanding and implementing flood forecasting workflows, guided by Dr. Fisaha. The goal is to get familiar with how data flows through hydrological analysis steps — from raw river data to basin-level precipitation insights — and later prepare for integration with more advanced models (kept confidential under Dr. Fisaha’s research).

---

## Project Overview

### Step 1 – Historical & Real-Time Flow Analysis
- Loads and merges historical hydrometric data for key Red River stations.
- Extends these records with current API data from ECCC.
- Generates time series, rolling means, anomalies, and flow duration curves.
- Each plot includes a freeze-up marker (December 2024) for context.

### Step 2 – Precipitation & Basin Visualization
- Loads Manitoba basin shapefiles (e.g., `500k_hyd-py.shp`) and station metadata.
- Overlays precipitation maps (e.g., HRDPA, HRDPS, REPS) over basin polygons.
- Produces geospatial visualizations to observe how precipitation patterns align with hydrological regions.

---

## How to Run

From the project’s `src/` directory, use:

```bash
# Step 1: Flow analysis
python3 -m hydro_pipeline.cli flows

# Step 2: Precipitation & basin visualization
python3 -m hydro_pipeline.cli precip
```

All outputs (plots and processed CSVs) are saved automatically under:

```
reports/figures/
├── step1_flows/
└── step2_precip/
```

---

## Data Structure

```
data/
├── raw/
│   ├── stations/red-river/      # HYDAT-style historical CSVs
│   ├── basin-resources/shapefiles/500k_shp/
│   └── precip-maps/             # Precipitation JPEGs or overlays
├── processed/                   # Generated merged CSVs
reports/
└── figures/                     # Auto-created plots for each step
```

---

## Notes
- This project is a learning and research exercise under academic guidance.  
- Any confidential modeling or data from Dr. Fisaha’s research remain private and excluded.  
- The focus is on building understanding and visualizing foundational hydrological data workflows.

---

### Acknowledgment
Special thanks to Dr. Fisaha for encouraging this exploration into the science and practice of flood forecasting.

---

**Author:** Joshua Debele  
Software Engineering Graduate – University of Calgary (2025)