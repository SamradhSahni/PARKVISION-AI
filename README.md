# PARKVISION AI — Parking Congestion Intelligence System

> **An AI-powered parking violation analysis and enforcement optimization platform for Bengaluru Traffic Police (BTP)**, combining spatial clustering, traffic engineering models, machine learning prediction, and LLM-driven intelligence.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Pipeline Stages](#pipeline-stages)
4. [Key Results](#key-results)
5. [Installation & Setup](#installation--setup)
6. [Usage](#usage)
7. [API Reference](#api-reference)
8. [Data Files](#data-files)
9. [Configuration](#configuration)
10. [Technologies Used](#technologies-used)

---

## System Overview

PARKVISION AI ingests **207,781 parking violation records** from Bengaluru Traffic Police and transforms them into actionable enforcement intelligence through a 10-stage analytical pipeline.

### What It Does

| Capability | Description |
|:-----------|:------------|
| **Multi-scale Hotspot Detection** | ST-DBSCAN clustering at 3 scales (50m/150m/500m) with Getis-Ord Gi* statistical validation |
| **Congestion Impact Scoring** | 5-component PCIS formula quantifying each violation's real traffic impact |
| **Shockwave Propagation** | LWR model estimating queue buildup and speed degradation ripple effects |
| **Location Memory** | Chronic "addiction zones" where violations persist despite enforcement |
| **XGBoost Prediction** | Forecasts violations per H3 hexagon per hour for the next week |
| **Enforcement ROI** | CongestionHoursRecovered (CHR) metric ranking hotspots by enforcement value |
| **Patrol Optimization** | Ant Colony Optimization (ACO) generating shift-aware patrol routes |
| **LLM Intelligence** | Gemini 2.0 Flash agent with function calling for natural language queries |
| **Interactive Dashboard** | Real-time visualization with heatmaps, charts, and AI chat |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PARKVISION AI Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Stage 1  │───▶│  Stage 2  │───▶│  Stage 3  │───▶│  Stage 4  │   │
│  │ Ingest &  │    │ Road Net  │    │ H3 Index  │    │ ST-DBSCAN │   │
│  │  Clean    │    │ Map-Match │    │  & POI    │    │ Clustering│   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│                                                        │            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Stage 8  │◀───│  Stage 7  │◀───│  Stage 6  │◀───│  Stage 5  │   │
│  │ XGBoost   │    │ Shockwave │    │  PCIS     │    │ Gi* Stats │   │
│  │ Predict   │    │ & Memory  │    │ Scoring   │    │ Temporal  │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│       │                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Stage 9  │───▶│ Stage 10  │───▶│  Gemini   │───▶│Dashboard  │   │
│  │ CHR & ROI │    │ACO Routes │    │ LLM Agent │    │ FastAPI   │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### Stage 1: Data Ingestion & Cleaning
- **Module:** `src/data_ingestion.py`
- **Input:** Raw CSV (207,781 records)
- **Process:** Parse dates/times, extract lat/lon, normalize vehicle types, severity mapping, temporal features (hour, day_of_week, is_weekend, is_peak, month)
- **Output:** `data/cleaned_violations.parquet`

### Stage 2: Road Network & Map-Matching
- **Module:** `src/road_network.py`
- **Process:** Load OSMnx Bengaluru graph (25km radius), compute edge betweenness centrality, snap violations to nearest road edges, extract highway class/width/lanes/speed
- **Output:** `data/enriched_violations.parquet`, `data/road_edges.parquet`

### Stage 3: H3 Spatial Indexing & POI Enrichment
- **Module:** `src/spatial_indexing.py`
- **Process:** Assign H3 resolution-9 hexagons (~174m), aggregate hex statistics, detect nearby POIs (metro, bus stops, hospitals, schools, markets) from OSM data
- **Output:** `data/h3_hex_stats.parquet`, `output/h3_hex_stats.geojson`

### Stage 4: ST-DBSCAN Multi-Scale Clustering
- **Module:** `src/hotspot_engine.py`
- **Process:** 3-scale spatiotemporal clustering:
  - **Micro** (50m, 2h, min_pts=5): Street-level clusters
  - **Meso** (150m, 2h, min_pts=5): Neighborhood clusters  
  - **Macro** (500m, 2h, min_pts=5): Zone-level clusters
- **Output:** `data/hotspot_clusters.parquet`, `data/cluster_profiles.parquet`

### Stage 5: Statistical Validation & Temporal Profiling
- **Module:** `src/hotspot_stats.py`
- **Process:** Getis-Ord Gi* z-scores (count + severity-weighted), FFT dominant frequency detection, hourly/daily temporal profiles per cluster
- **Output:** `data/h3_hotspot_significance.parquet`, `data/temporal_profiles.json`

### Stage 6: PCIS Scoring Engine
- **Module:** `src/pcis_engine.py`
- **Formula:** `PCIS = 0.30×CR + 0.20×PF + 0.20×TDM + 0.15×VOF + 0.15×NC`
  - CR: Capacity Reduction (vehicle footprint / road width)
  - PF: Proximity Factor (distance to junctions)
  - TDM: Temporal Demand Multiplier (peak vs off-peak)
  - VOF: Vehicle Obstruction Factor (severity weight)
  - NC: Network Criticality (betweenness centrality)
- **Output:** `data/pcis_scored_violations.parquet`, `data/h3_pcis_scores.parquet`

### Stage 7: Congestion Propagation & Location Memory
- **Module:** `src/congestion_model.py`
- **Process:** LWR shockwave model (queue length, spillback distance, speed degradation), Location Memory Score (0.6×persistence + 0.4×repeat_vehicle_fraction), cross-jurisdiction spillover analysis
- **Output:** `output/ripple_contours.geojson`, `data/location_memory.parquet`, `data/spillover_analysis.parquet`

### Stage 8: XGBoost Violation Prediction
- **Module:** `src/prediction.py`
- **Process:** 40-feature model (temporal cyclical, PCIS components, road characteristics, location memory, historical stats). Time-series 80/20 split, 500 trees
- **Performance:** Test MAE: 1.303, RMSE: 2.448, R²: 0.601
- **Output:** `models/violation_predictor.joblib`, `data/predicted_violations.parquet`

### Stage 9: Enforcement ROI & Station Profiling
- **Module:** `src/enforcement_optimizer.py`
- **Process:** Geolocate 54 police stations, compute jurisdiction profiles, calculate CongestionHoursRecovered (CHR = PCIS × frequency × duration × capacity_affected)
- **Output:** `data/enforcement_priorities.parquet`, `data/police_stations.geojson`

### Stage 10: ACO Patrol Route Optimization
- **Module:** `src/patrol_router.py`
- **Process:** Ant Colony Optimization (50 ants, 100 iterations) across 5 shifts from 8 depot stations. Generates Daily Enforcement Intelligence Brief
- **Output:** `output/patrol_routes.geojson`, `output/daily_brief.md`

---

## Key Results

### Violation Distribution
- **207,781 total violations** across 151 observation days
- **2,420 H3 hexagons** with at least one violation
- **54 police stations** covering Bengaluru

### Top 5 Hotspot Stations

| Station | Violations | Daily Rate | PCIS | Pattern |
|:--------|:-----------|:-----------|:-----|:--------|
| Upparpet | 25,588 | 169/day | 0.536 | Morning |
| Shivajinagar | 18,189 | 120/day | 0.511 | Morning |
| Malleshwaram | 16,035 | 106/day | 0.432 | Morning |
| HAL Old Airport | 13,584 | 91/day | 0.399 | Morning |
| City Market | 12,409 | 82/day | 0.482 | Morning |

### Congestion Impact
- **Total recoverable congestion:** 5.66M vehicle-hours/day
- **Top 20 hexagons** (0.8% of zones): 21.1% of total CHR
- **Top 50 hexagons** (2.1%): 32.9% of total CHR
- **#1 URGENT zone** (Upparpet): 177,322 veh-hrs/day

### Addiction Zones
- **38 hexagons** (1.6%) with Location Memory > 0.5
- These contain **80,493 violations** (38.7% of total)
- Active on >50% of all observation days

### XGBoost Model
- **Top features:** is_no_parking (0.110), road_is_main (0.096), is_wrong_parking (0.094)
- **Test R²:** 0.601 — violation type and road characteristics are stronger predictors than time

### ACO Patrol Routes
- **40 optimized routes** across 5 shifts from 8 depots
- **25M total CHR recoverable** across all routes
- Night shift yields highest CHR per route (longer shift, more stops)

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- 8GB+ RAM recommended

### Install

```bash
# Clone the repository
git clone https://github.com/SamradhSahni/PARKVISION-AI.git
cd PARKVISION-AI

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Configure API keys
# Create a .env file in the project root directory and add:
# GEMINI_API_KEY=your_gemini_api_key_here
# TOMTOM_API_KEY=your_tomtom_api_key_here (optional)
```

### Required Python Packages

```
pandas>=2.0
geopandas>=0.14
numpy>=1.24
scipy>=1.11
scikit-learn>=1.3
xgboost>=2.0
h3>=3.7
osmnx>=1.7
shapely>=2.0
pyarrow>=14.0
fastapi>=0.104
uvicorn>=0.24
google-generativeai>=0.3
joblib>=1.3
python-dotenv>=1.0
leaflet (CDN - no install needed)
chart.js (CDN - no install needed)
```

---

## Usage

### Run Full Pipeline
```bash
python -m src.run_pipeline
```

### Resume from a Stage
```bash
python -m src.run_pipeline --from 6    # Resume from PCIS scoring
```

### Run Single Stage
```bash
python -m src.run_pipeline --stage 8   # Run only prediction
```

### Force Re-run
```bash
python -m src.run_pipeline --force     # Ignore checkpoints
```

### Launch Dashboard
```bash
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

### Interactive LLM Chat
```bash
python -m src.llm_agent               # Interactive mode
python -m src.llm_agent --demo        # Run 5 demo queries
```

---

## API Reference

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/api/summary` | GET | Overall system statistics |
| `/api/hotspots?n=50&station=X&tier=HIGH` | GET | Top hotspots by CHR |
| `/api/pcis/{h3_index}` | GET | PCIS breakdown for hexagon |
| `/api/heatmap` | GET | All hexagons for heatmap |
| `/api/hex-geojson` | GET | H3 stats as GeoJSON |
| `/api/patrol-routes?shift=morning` | GET | ACO patrol routes |
| `/api/stations` | GET | Police station profiles |
| `/api/temporal/{area}` | GET | Temporal patterns |
| `/api/predict` | GET | Next-week predictions |
| `/api/ripple-contours` | GET | Congestion ripple GeoJSON |
| `/api/cluster-profiles` | GET | Cluster profiles GeoJSON |
| `/api/chat` | POST | LLM query (body: `{"query": "..."}`) |

---

## Data Files

### Intermediate Data (`data/`)

| File | Size | Description |
|:-----|:-----|:------------|
| `cleaned_violations.parquet` | 15.4 MB | Cleaned & normalized violations |
| `enriched_violations.parquet` | 18.9 MB | Road-matched with network features |
| `h3_hex_stats.parquet` | 274 KB | H3 hexagon aggregated statistics |
| `hotspot_clusters.parquet` | 19.5 MB | ST-DBSCAN cluster assignments |
| `cluster_profiles.parquet` | 76 KB | Cluster-level profiles |
| `h3_hotspot_significance.parquet` | 319 KB | Gi* z-scores per hexagon |
| `temporal_profiles.json` | 214 KB | Hourly/daily profiles per cluster |
| `pcis_scored_violations.parquet` | 22.1 MB | PCIS-scored violations |
| `h3_pcis_scores.parquet` | 188 KB | Hex-level PCIS aggregates |
| `location_memory.parquet` | 120 KB | Location memory scores |
| `spillover_analysis.parquet` | 32 KB | Cross-station spillover |
| `predicted_violations.parquet` | 2.3 MB | Next-week predictions |
| `enforcement_priorities.parquet` | 356 KB | CHR-ranked hex priorities |
| `police_stations.geojson` | 48 KB | Station profiles + locations |

### Outputs (`output/`)

| File | Description |
|:-----|:------------|
| `h3_hex_stats.geojson` | Hex heatmap for dashboard |
| `cluster_profiles.geojson` | Cluster polygons for map |
| `ripple_contours.geojson` | Congestion ripple effect polygons |
| `patrol_routes.geojson` | ACO-optimized patrol route lines |
| `daily_brief.md` | Daily Enforcement Intelligence Brief |
| `llm_demo_conversations.md` | LLM demo query results |

### Models (`models/`)

| File | Description |
|:-----|:------------|
| `violation_predictor.joblib` | Trained XGBoost model + feature list |
| `model_metrics.json` | Train/test performance metrics |
| `feature_importance.csv` | Feature importance rankings |

---

## Configuration

All parameters are centralized in `config/settings.py`:

| Category | Key Parameters |
|:---------|:---------------|
| **PCIS Weights** | CR: 0.30, PF: 0.20, TDM: 0.20, VOF: 0.15, NC: 0.15 |
| **ST-DBSCAN** | Micro: 50m/2h/5pts, Meso: 150m/2h/5pts, Macro: 500m/2h/5pts |
| **H3 Resolution** | 9 (~174m edge length) |
| **ACO** | 50 ants, 200 iterations, α=1.0, β=2.0, evaporation=0.5 |
| **Shifts** | Morning 6-10, Midday 10-14, Afternoon 14-17, Evening 17-21, Night 21-6 |
| **Road Capacity** | 1800 veh/hr/lane (HCM urban arterial) |
| **Patrol Speed** | 20 km/h average city speed |

---

## Technologies Used

| Category | Technology |
|:---------|:-----------|
| **Spatial Analysis** | H3, OSMnx, GeoPandas, Shapely |
| **Clustering** | ST-DBSCAN (custom), Getis-Ord Gi* |
| **Machine Learning** | XGBoost, scikit-learn |
| **Optimization** | Ant Colony Optimization (custom) |
| **LLM** | Google Gemini 2.0 Flash (function calling) |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | Leaflet.js, Chart.js, h3-js |
| **Data** | Pandas, PyArrow (Parquet), GeoJSON |

---

## Project Structure

```
GridHack/
├── config/
│   ├── __init__.py
│   └── settings.py              # Central configuration
├── src/
│   ├── __init__.py
│   ├── run_pipeline.py          # End-to-end pipeline runner
│   ├── data_ingestion.py        # Stage 1: Ingest & clean
│   ├── road_network.py          # Stage 2: Road matching
│   ├── spatial_indexing.py       # Stage 3: H3 & POI
│   ├── hotspot_engine.py        # Stage 4: ST-DBSCAN
│   ├── hotspot_stats.py         # Stage 5: Gi* & temporal
│   ├── pcis_engine.py           # Stage 6: PCIS scoring
│   ├── congestion_model.py      # Stage 7: Shockwave & memory
│   ├── prediction.py            # Stage 8: XGBoost
│   ├── enforcement_optimizer.py  # Stage 9: CHR & stations
│   ├── patrol_router.py         # Stage 10: ACO routes
│   ├── llm_agent.py             # Gemini LLM agent
│   └── api_server.py            # FastAPI dashboard backend
├── dashboard/
│   └── index.html               # Interactive web dashboard
├── data/                        # Processed data files
├── output/                      # GeoJSON, reports, brief
├── models/                      # Trained ML models
├── .env                         # API keys
└── README.md                    # This file
```

---

*Built for Bengaluru Traffic Police | PARKVISION AI Parking Congestion Intelligence System*
