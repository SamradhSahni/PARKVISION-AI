# ParkingIntel — Complete Feature & Technical Documentation

> **AI-Driven Parking Hotspot Intelligence for Bangalore Traffic Police**  
> Hackathon Project — Theme 1: Poor Visibility on Parking-Induced Congestion  
> June 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Feature Breakdown](#feature-breakdown)
4. [Module Reference](#module-reference)
5. [Congestion Impact Score (CIS)](#congestion-impact-score)
6. [Data Pipeline](#data-pipeline)
7. [Tech Stack](#tech-stack)
8. [Setup & Run](#setup--run)
9. [Key Findings](#key-findings)
10. [File Structure](#file-structure)

---

## Overview

ParkingIntel analyzes **298,450 parking violations** across **54 police stations** in Bangalore (November 2023 – April 2024). It detects illegal parking hotspots, quantifies congestion impact, and generates deployment-ready patrol plans for enforcement officers.

**Core capability:** Given a day, time window, and available officers, the system predicts where violations will concentrate and assigns geographically-clustered patrol routes with realistic time constraints.

---

## Architecture

```
parkingintel/
├── app/
│   └── dashboard.py           # Streamlit UI (7 pages, sidebar navigation)
├── src/
│   ├── data_loader.py         # CSV → cleaned DataFrame pipeline
│   ├── analytics.py           # Station-level metrics & temporal analysis
│   ├── clustering.py          # DBSCAN hotspot detection + grid aggregation
│   ├── blindspots.py          # Gap analysis (3 detection methods)
│   ├── visualization.py       # Folium maps + Plotly charts
│   ├── planner.py             # Enforcement planner (prediction + routing)
│   ├── cis_engine.py          # Congestion Impact Score engine
│   ├── chatbot.py             # AI assistant (DeepSeek API)
│   └── zone_classifier.py     # Vehicle-based zone classification
├── data/                      # Dataset + cached parquet
├── requirements.txt
└── README.md
```

---

## Feature Breakdown

### Page 1: Enforcement Planner 🎯

**The core solution.** Generates patrol plans for officers based on predicted hotspots.

| Feature | Description |
|---|---|
| **Day & Time Selection** | Choose day of week + time window for the patrol shift |
| **Officer Count** | 1-5 officers, determines cluster count |
| **Hotspot Prediction** | Analyzes historical patterns for that exact day/time |
| **Congestion-Aware Ranking** | Zones weighted by CIS (vehicle severity × junction proximity × temporal demand) |
| **Geographic Clustering** | K-Means groups zones into per-officer clusters |
| **Time-Constrained Routing** | Greedy nearest-neighbor within each cluster |
| **Realistic Constraints** | 20 km/h travel speed, 20 min per stop, zones beyond time window excluded |
| **Per-Officer Breakdown** | Each officer sees their own stops with travel times |
| **Patrol Route Map** | Color-coded markers per officer on satellite map |
| **Recency Boost** | Exponential decay — recent violations weighted higher |
| **Consistency Scoring** | Zones appearing on 3+ dates get higher confidence |

**Input:** Saturday, 4-8 PM, 3 officers  
**Output:** 3 clusters, ~9 zones, per-officer route with ETA and expected violations

---

### Page 2: Hotspot Map 🗺️

Interactive satellite map for exploring violation density.

| Feature | Description |
|---|---|
| **Density Overlay** | HeatMap plugin showing violation concentration |
| **DBSCAN Clusters** | Red circle markers at ~55m resolution |
| **Layer Toggle** | Checkboxes for heatmap + hotspot visibility |
| **Sidebar Filters** | Filter by station, time bucket, day, vehicle type |
| **Full Map Interaction** | Zoom, pan, click markers for details |

---

### Page 3: Station Comparison 📊

Multi-dimensional station metrics for identifying where support is needed.

| Dimension | What It Measures |
|---|---|
| **Quality Score** | Ticket accuracy (approval rate, zone-adjusted) |
| **Coverage Score** | % of 24 hours with recorded activity |
| **Responsiveness** | Processing speed |
| **Balance Score** | Weekday/weekend fit |
| **Zone Complexity** | Estimated from vehicle diversity + junction count |
| **Overall Score** | Weighted composite (35/25/25/15) |

All metrics include explanatory text emphasizing they are **not officer rankings** — they identify where the *system* needs support (better devices, training, resources).

---

### Page 4: Temporal Patterns ⏰

When violations occur.

| Feature | Description |
|---|---|
| **Day × Hour Heatmap** | Plotly heatmap showing weekly rhythm |
| **Hourly Bar Chart** | Orange-highlighted 4-7 AM spike |
| **4-7 AM Analysis** | 104,685 violations 4-7 AM vs 25,336 at 5-8 PM (4×) |
| **Balanced Interpretation** | Notes both possible explanations: overnight parking behavior AND enforcement sweep patterns |

---

### Page 5: Gap Analysis 🔍

Three indirect methods to identify areas that may benefit from additional attention.

| Method | Logic |
|---|---|
| **Vehicle Diversity** | Many vehicle types + few violations = active area, low recorded enforcement |
| **Time Coverage** | Stations active for few hours = rest of day unrepresented in data |
| **Junction Density** | Many junctions + few violations = spread thin |

All methods include explicit caveats: *"flags unusual patterns for review, not conclusions."*

---

### Page 6: Vehicle Profiles 🚘

What types of vehicles are involved at each station.

| Feature | Description |
|---|---|
| **Per-Station Pie Chart** | Top 8 vehicle types |
| **Dominant Vehicle** | Auto-detected with contextual note |
| **Multi-Station Comparison** | Side-by-side vehicle % table |
| **Enforcement Notes** | Scooter-heavy → footpath focus; car-heavy → main road focus |

---

### Page 7: Weekday vs Weekend 📅

Observed weekday/weekend split per station.

| Feature | Description |
|---|---|
| **Scatter Plot** | Each dot = station, diagonal = balanced |
| **Range Metrics** | Lowest (15.7%), highest (43.3%), spread |
| **Full Station Table** | Sortable by weekend % |
| **Observational Only** | Explicitly notes: no shift schedule or land-use data available |

---

### Chatbot 🤖

AI assistant available on every page (bottom of each tab).

| Feature | Description |
|---|---|
| **LLM** | DeepSeek (deepseek-chat) |
| **Context** | Full system prompt with all 5 key findings, station stats, vehicle breakdowns |
| **Capabilities** | Answers questions about data, explains dashboard usage, provides statistics |
| **Chat History** | Persists during session, clear button |

---

## Module Reference

### `src/data_loader.py`
Loads the 104.5 MB CSV, cleans data, engineers features.

**Engineered columns:** hour, day_of_week, day_name, is_weekend, date, month, hour_bucket, processing_hours, is_approved, is_rejected, is_unprocessed, is_parking, violation_primary, violation_list, violation_count

**Caching:** Cleaned DataFrame saved as parquet for fast reload.

---

### `src/analytics.py`
Station-level metrics and temporal analysis.

| Function | Returns |
|---|---|
| `station_summary(df)` | 54-row DataFrame with 10+ metrics per station |
| `peak_hour_profile(df)` | Hourly violation counts and percentages |
| `vehicle_fingerprint(df, station)` | Dict of vehicle_type → percentage |
| `weekday_vs_weekend(df)` | Per-station weekday/weekend split |
| `enforcement_gap_analysis(df)` | Hourly DataFrame + enforcement vs congestion totals |

---

### `src/clustering.py`
Spatial hotspot detection.

| Function | Description |
|---|---|
| `get_hotspots(df)` | DBSCAN clustering at ~55m resolution, returns top hotspots with center coordinates and violation counts |
| `grid_aggregation(df)` | Grid-based alternative (500m cells) for density visualization |

---

### `src/blindspots.py`
Gap detection using indirect data signals.

| Function | Description |
|---|---|
| `combined_blind_spot_score(df)` | Weighted composite (40/35/25) of 3 methods |
| `vehicle_diversity_blindspots(df)` | High vehicle type count ÷ low violations |
| `time_coverage_blindspots(df)` | Hours with ≥5 violations ÷ 24 |
| `junction_density_blindspots(df)` | Violations per unique junction |

---

### `src/planner.py`
The core enforcement planning engine.

| Function | Description |
|---|---|
| `predict_hotspots(df, day, start_hour, end_hour, n_officers)` | Returns top zones ranked by priority score |
| `cluster_and_assign(zones_df, n_officers, time_window_hours)` | K-Means clustering + time-constrained routing, returns per-officer assignments |
| `format_plan(assignments, day_name, start, end, n_officers)` | Human-readable patrol plan markdown |

**Priority Score Formula:**
```
priority = weighted_count × cis_factor × (1 + junction_score × 0.3) × (1 + recency_factor × 0.5) × (0.5 + consistency × 0.5)
```

---

### `src/cis_engine.py`
Congestion Impact Score — quantifies traffic impact per violation.

| Function | Description |
|---|---|
| `compute_cis(df)` | Computes CIS for each row in the DataFrame |
| `compute_zone_cis(zone_df, violation_df)` | Aggregates CIS by grid zone |

---

### `src/visualization.py`
All Folium maps and Plotly charts.

| Function | Output |
|---|---|
| `create_base_map()` | Folium map centered on Bangalore |
| `add_violation_heatmap(m, df)` | HeatMap overlay |
| `add_hotspot_markers(m, hotspots)` | Circle markers for DBSCAN clusters |
| `plot_station_scorecard(summary, metric)` | Horizontal bar chart |
| `plot_hourly_heatmap(df)` | Day×Hour heatmap |
| `plot_vehicle_pie(df, station)` | Donut chart |
| `plot_weekend_scatter(weekend_df)` | Scatter plot with diagonal |
| `plot_hourly_bar(hourly_df)` | Bar chart with 4-7 AM highlight |
| `plot_blindspot_ranking(bs_df)` | Horizontal bar chart |

---

## Congestion Impact Score

Inspired by PARKVISION's PCIS (Parking Congestion Impact Score). Simplified to work without road network data.

### Components

| Component | Range | Description |
|---|---|---|
| **Vehicle Severity** | 1.0 – 8.0 | Scooter=1.0, Car=2.0, Bus=6.0, Truck=8.0 |
| **Junction Proximity** | 1.0 – 3.0 | Near junction=3.0, isolated=1.0 |
| **Temporal Demand** | 0.05 – 1.0 | 3 AM=0.05, 8 AM/6 PM rush=1.0 |

### Formula
```
CIS = VehicleSeverity × JunctionProximity × TemporalDemand
Range: 0.05 (scooter at 3 AM, no junction) to 24.0 (truck at 8 AM, at junction)
```

### Integration
- **Enforcement Planner:** Zones ranked by CIS-weighted priority score
- **Hotspot Map:** Potential future enhancement — color-code by CIS

---

## Data Pipeline

```
Raw CSV (104.5 MB, 298,450 rows)
    │
    ▼
data_loader.py
    │ Parse JSON violation_type arrays
    │ Convert timestamps → datetime
    │ Engineer 14+ temporal/spatial features
    │ Handle NULLs, drop invalid coordinates
    │
    ▼
Cleaned DataFrame (27 columns)
    │ Cached as parquet
    │
    ├──▶ analytics.py      → Station metrics, temporal profiles
    ├──▶ clustering.py    → DBSCAN hotspots, grid cells
    ├──▶ blindspots.py    → Gap detection scores
    ├──▶ cis_engine.py    → Congestion impact scores
    └──▶ planner.py       → Predicted hotspots, patrol routes
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Data Processing** | Pandas 2.x, NumPy | ETL pipeline, feature engineering |
| **Clustering** | DBSCAN (scikit-learn) | Hotspot detection (~55m resolution) |
| **ML** | K-Means (scikit-learn) | Geographic clustering for patrol zones |
| **Spatial** | H3 (via PARKVISION) / Grid | Spatial indexing |
| **Maps** | Folium + OpenStreetMap | Interactive map with HeatMap plugin |
| **Charts** | Plotly (plotly.express + graph_objects) | All data visualizations |
| **Dashboard** | Streamlit 1.58 | Full UI with sidebar navigation |
| **LLM** | DeepSeek (deepseek-chat) | AI chatbot assistant |
| **API** | OpenAI-compatible endpoint | Chatbot backend |
| **Version Control** | Git + GitHub | Source code + Git LFS for dataset |
| **Package Mgmt** | pip + uv | Dependency management |
| **Runtime** | Python 3.11 | Core language |

---

## Setup & Run

```bash
# Clone
git clone https://github.com/Anantgoel2005/parkingintel.git
cd parkingintel

# Virtual environment
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Dataset (104.5 MB, included via Git LFS)
# Already in data/raw/ after clone

# Run
streamlit run app/dashboard.py
# Opens at http://localhost:8501
```

### Chatbot Setup (Optional)
```bash
# Create src/chatbot.py from the example
copy src\chatbot_example.py src\chatbot.py
# Set your DeepSeek API key as environment variable or edit the file
```

---

## Key Findings

1. **4-7 AM Spike** — 104,685 violations recorded 4-7 AM vs 25,336 at 5-8 PM (4× difference). Consistent across all stations and vehicle types.

2. **Quality Variation** — Rejection rates range from 21.3% (Wilson Garden) to 41.7% (K.G. Halli) across stations.

3. **Processing Times** — Median processing ranges from 25 hours (Chikkajala) to 42 hours (Vijayanagara).

4. **Vehicle Mix** — City Market: 41% scooters. Malleshwaram: 47% cars. Each station has a distinct vehicle profile reflecting local land use.

5. **Weekend Split** — Weekend share ranges from 15.7% (HAL Old Airport) to 43.3% (K.G. Halli).

---

## File Structure

```
parkingintel/
├── app/
│   └── dashboard.py              # Streamlit app (7 pages, 19.4 KB)
├── src/
│   ├── data_loader.py            # Data pipeline (5.3 KB)
│   ├── analytics.py              # Station metrics (4.2 KB)
│   ├── clustering.py             # DBSCAN + grid (4.1 KB)
│   ├── blindspots.py             # Gap analysis (4.5 KB)
│   ├── visualization.py          # Maps + charts (5.6 KB)
│   ├── planner.py                # Enforcement planner (8.2 KB)
│   ├── cis_engine.py             # Congestion Impact Score (3.5 KB)
│   ├── chatbot.py                # AI assistant (7.2 KB, gitignored)
│   └── chatbot_example.py        # Chatbot template (no API key)
├── data/
│   └── raw/                      # Dataset (Git LFS)
│   └── processed/                # Cached parquet (gitignored)
├── notebooks/                    # Jupyter notebooks
├── reports/
│   └── screenshots/              # Dashboard screenshots
├── requirements.txt              # Python dependencies
├── .gitignore
├── .gitattributes                # Git LFS config
└── README.md                     # Project overview
```

---

*Document generated June 17, 2025. For integration questions, contact the project team.*
