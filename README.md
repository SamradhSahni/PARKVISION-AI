# PARKVISION AI

**AI-driven parking enforcement intelligence for Bengaluru Traffic Police**

A full-stack data science project that processes 207,781 real-world parking violation records to identify hotspots, model congestion impact, optimize patrol routes, and power a natural-language AI assistant — all served through a live web dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Architecture](#3-architecture)
4. [Quick Start — Clone and Run](#4-quick-start--clone-and-run)
5. [Prerequisites](#5-prerequisites)
6. [Installation Steps](#6-installation-steps)
7. [Environment Variables (API Keys)](#7-environment-variables-api-keys)
8. [Running the Dashboard](#8-running-the-dashboard)
9. [Running the Full Data Pipeline](#9-running-the-full-data-pipeline)
10. [Dashboard Pages](#10-dashboard-pages)
11. [API Reference](#11-api-reference)
12. [Data Science Methodology](#12-data-science-methodology)
13. [Project File Structure](#13-project-file-structure)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Project Overview

PARKVISION AI converts raw parking challan data into actionable enforcement intelligence. It:

- Scores every hexagonal zone in Bengaluru with a **PCIS (Parking Congestion Impact Score)** built from 5 road-network-aware components
- Computes **CHR (Congestion Hours Recovered)** — the ROI of enforcement in vehicle-hours/day
- Clusters violation hotspots using **ST-DBSCAN** at micro/meso/macro scales validated by **Getis-Ord Gi\***
- Generates **optimized patrol routes** for 5 daily shifts using K-Means + greedy nearest-neighbor routing
- Predicts next-week violations using **XGBoost**
- Provides an **AI chat assistant** powered by NVIDIA NIM (LLaMA-3.3-70B) with full data context
- Serves everything via a dark-mode **interactive dashboard** built on Leaflet + Chart.js

---

## 2. Features

### Dashboard Pages

| Page | Description |
|------|-------------|
| **Violation Map** | Full-screen Leaflet map with heatmap, H3 hexagon layer, patrol routes, and station markers. Shift-based route filtering (Morning / Midday / Afternoon / Evening / Night). |
| **Filter Hotspots** | Filter 200+ ranked hotspots by station, day of week, time bucket, vehicle type, and priority tier. Card grid results with PCIS / CHR / Violations / Peak Hour. |
| **Enforcement Planner** | K-Means clustering + greedy nearest-neighbor patrol route generator. Select day, shift hours, and number of officers (1–5). See per-officer stop schedules with ETAs. |
| **Station Comparison** | 6-metric scorecard for all 54 police stations: quality, coverage, responsiveness, balance, zone complexity, and overall score. Sortable table + bar chart. |
| **Temporal Patterns** | Day × Hour violation heatmap (7 × 24). Identifies 4–7 AM early-morning enforcement sweep pattern. |
| **Gap Analysis** | 3-method indirect detection of potentially under-recorded areas using vehicle diversity, time coverage gaps, and junction density ratios. |
| **Vehicle Profiles** | Per-station vehicle type breakdown (pie chart + table) with enforcement note (e.g., "scooter-heavy → check footpaths"). |
| **Weekday vs Weekend** | Scatter plot and table showing observed weekday/weekend split per station. Highlights stations with unusual weekend enforcement patterns. |
| **AI Chat** | Natural language Q&A powered by NVIDIA NIM (LLaMA-3.3-70B). Answers questions about hotspots, PCIS scores, CHR, temporal patterns, predictions, and enforcement recommendations using live data context. |

### Core Algorithms

| Component | Method |
|-----------|--------|
| Spatial clustering | ST-DBSCAN at 3 scales (50m / 150m / 500m), min 5 points |
| Statistical validation | Getis-Ord Gi\* with permutation testing |
| Hexagonal indexing | H3 resolution 9 (~174m edge length) |
| Congestion scoring | PCIS = 5-component weighted formula |
| Enforcement ROI | CHR (Congestion Hours Recovered) in vehicle-hours/day |
| Patrol routing | K-Means clustering → greedy nearest-neighbor |
| Prediction | XGBoost regressor, day × hour features |
| AI assistant | NVIDIA NIM — meta/llama-3.3-70b-instruct via OpenAI-compatible API |

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                       │
│  Raw CSV → Clean → Enrich → Cluster → Score → Route    │
│  src/data_ingestion.py                                  │
│  src/road_network.py        ← OSM road graph            │
│  src/spatial_indexing.py    ← H3 hex binning            │
│  src/hotspot_engine.py      ← ST-DBSCAN clusters        │
│  src/hotspot_stats.py       ← Getis-Ord Gi*             │
│  src/pcis_engine.py         ← PCIS + CHR scoring        │
│  src/congestion_model.py    ← Location memory + CHR     │
│  src/enforcement_optimizer.py ← K-Means + routing       │
│  src/patrol_router.py       ← Shift-based patrol routes │
│  src/prediction.py          ← XGBoost predictions       │
│  src/llm_agent.py           ← NVIDIA NIM AI agent       │
└───────────────┬─────────────────────────────────────────┘
                │  Parquet files (data/), GeoJSON (output/)
                ▼
┌─────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                       │
│  src/api_server.py    (port 8000)                       │
│  20+ REST endpoints serving live data                   │
└───────────────┬─────────────────────────────────────────┘
                │  JSON / GeoJSON
                ▼
┌─────────────────────────────────────────────────────────┐
│                  WEB DASHBOARD                          │
│  dashboard/index.html                                   │
│  Leaflet 1.9.4 + Chart.js 4.4 + H3-js 4.1             │
│  9 interactive pages, dark theme                        │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Quick Start — Clone and Run

> **Important**: All processed data files (`data/*.parquet`, `output/*.geojson`, `models/*.joblib`) and the raw violation CSV are excluded from the repository (too large for Git). You **must** complete Steps 5 and 6 before the server will work.

### What you need before starting

| Item | How to get it |
|------|--------------|
| Raw violation CSV file | Obtain from the project owner or dataset source |
| NVIDIA NIM API key | [build.nvidia.com](https://build.nvidia.com) — free account |
| Python 3.10–3.12 | [python.org/downloads](https://www.python.org/downloads/) |

---

### Step-by-Step

```bash
# ── Step 1: Clone the repository ──────────────────────────────────
git clone https://github.com/SamradhSahni/PARKVISION-AI.git
cd PARKVISION-AI

# ── Step 2: Create and activate a virtual environment ─────────────
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# ── Step 3: Upgrade pip and install all dependencies ─────────────
python -m pip install --upgrade pip
pip install -r requirements.txt

# ── Step 4: Set up environment variables ──────────────────────────
# Windows
copy .env.example .env

# macOS / Linux
# cp .env.example .env

# Now open .env in any text editor and add your API keys:
#   NVIDIA_API_KEY=nvapi-your-key-here
#   TOMTOM_API_KEY=your-tomtom-key-here   (optional)
```

```bash
# ── Step 5: Place the raw CSV file in the project root ───────────
# The file must be named exactly:
#   "jan to may police violation_anonymized791b166.csv"
# Place it here:
#   PARKVISION-AI/
#   └── jan to may police violation_anonymized791b166.csv  ← here

# ── Step 6: Run the data pipeline (ONE-TIME SETUP, ~30-90 min) ───
# This processes the raw CSV and builds all data files the server needs.
python -m src.run_pipeline

# You will see progress logs for each of the 11 stages:
# Stage 1/11: Data ingestion ... done
# Stage 2/11: Road network (OSM download ~500 MB) ... done
# Stage 3/11: Spatial indexing (H3 hex) ... done
# ...
# Pipeline complete. All data files written to data/ and output/

# ── Step 7: Start the dashboard server ────────────────────────────
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8000

# ── Step 8: Open the dashboard ────────────────────────────────────
# Visit http://localhost:8000 in your browser
```

> **The pipeline is a one-time step.** Once `data/`, `output/`, and `models/` are populated, you only need Step 7 (start the server) on every subsequent run.

---


## 5. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.10, 3.11, or 3.12 | Python 3.13 is NOT supported (h3, osmnx incompatibility) |
| **pip** | 23+ | Run `python -m pip install --upgrade pip` |
| **Git** | Any | For cloning |
| **RAM** | 8 GB minimum | 16 GB recommended for pipeline runs |
| **Disk** | ~3 GB | For data files, OSM cache, and model files |
| **OS** | Windows 10+, Ubuntu 20.04+, macOS 12+ | All supported |
| **Internet** | Required for first run | OSM road network download (~500 MB) |

### API Keys Required

| Key | Service | Where to get it | Required for |
|-----|---------|-----------------|--------------|
| `NVIDIA_API_KEY` | NVIDIA NIM | [build.nvidia.com](https://build.nvidia.com) | AI Chat (primary) |
| `TOMTOM_API_KEY` | TomTom Maps | [developer.tomtom.com](https://developer.tomtom.com) | Night map tiles |
| `GEMINI_API_KEY` | Google Gemini | [aistudio.google.com](https://aistudio.google.com) | AI Chat (fallback only) |

> Only `NVIDIA_API_KEY` is required for AI chat. TomTom is optional (fallback to CartoDB tiles if missing). Gemini is kept as fallback.

---

## 6. Installation Steps

### Step 1 — Clone the repository

```bash
git clone https://github.com/SamradhSahni/PARKVISION-AI.git
cd PARKVISION-AI
```

### Step 2 — Create a virtual environment

**Windows (Command Prompt or PowerShell)**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3 — Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4 — Install all dependencies

```bash
pip install -r requirements.txt
```

> This installs all packages with pinned versions. Total download: ~400 MB.
> If you hit errors on `geopandas` or `pyogrio` on Windows, see the [Troubleshooting](#14-troubleshooting) section.

### Step 5 — Set up environment variables

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Then open `.env` in any text editor and fill in your API keys:

```env
NVIDIA_API_KEY=nvapi-your-key-here
TOMTOM_API_KEY=your-tomtom-key-here
GEMINI_API_KEY=your-gemini-key-here   # optional fallback
```

### Step 6 — Verify the data files are present

The processed data files should exist under `data/`. Run this to check:

```bash
python -c "
import os
files = [
    'data/pcis_scored_violations.parquet',
    'data/enforcement_priorities.parquet',
    'data/predicted_violations.parquet',
    'data/police_stations.geojson',
    'output/patrol_routes.geojson',
]
for f in files:
    status = 'OK' if os.path.exists(f) else 'MISSING'
    print(f'{status}: {f}')
"
```

If any files show `MISSING`, you need to run the full data pipeline (see Section 9).

---

## 7. Environment Variables (API Keys)

Create a `.env` file in the project root (copy from `.env.example`):

```env
# NVIDIA NIM — Primary AI backend (required for AI Chat)
# Get your free key at: https://build.nvidia.com
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# TomTom — Premium dark map tiles (optional, falls back to CartoDB)
# Get your free key at: https://developer.tomtom.com
TOMTOM_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Google Gemini — Fallback AI backend (optional)
# Get your free key at: https://aistudio.google.com
GEMINI_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> **Security**: The `.env` file is listed in `.gitignore` and will never be committed to Git. Never share your API keys publicly.

---

## 8. Running the Dashboard

### Start the server

```bash
# Make sure your venv is active first
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Open the dashboard

Open your browser and visit: **http://localhost:8000**

### Navigate the dashboard

Use the sidebar on the left. Hover over it to expand and see page names:

| Icon | Page |
|------|------|
| [P] | Enforcement Planner |
| [F] | Filter Hotspots |
| [M] | Violation Map |
| [S] | Station Comparison |
| [T] | Temporal Patterns |
| [G] | Gap Analysis |
| [V] | Vehicle Profiles |
| [W] | Weekday vs Weekend |
| [A] | AI Chat |

### Stop the server

Press `Ctrl + C` in the terminal.

---

## 9. Running the Full Data Pipeline

> **Only needed if you are starting from the raw CSV file** (e.g., replacing the data with new violation records).
> If `data/` and `output/` are already populated, skip this section.

### What the pipeline does

The pipeline has 11 sequential stages:

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `data_ingestion.py` | Clean and validate raw CSV, remove outliers, standardize columns |
| 2 | `road_network.py` | Download Bengaluru road graph from OpenStreetMap (OSMnx) |
| 3 | `spatial_indexing.py` | Assign H3 hexagonal indices (resolution 9), compute hex stats |
| 4 | `hotspot_engine.py` | Run ST-DBSCAN at micro/meso/macro scales |
| 5 | `hotspot_stats.py` | Getis-Ord Gi\* statistical validation |
| 6 | `pcis_engine.py` | Compute PCIS scores with all 5 components |
| 7 | `congestion_model.py` | Compute CHR, Location Memory scores, spillover analysis |
| 8 | `enforcement_optimizer.py` | K-Means zone clustering, CHR-based priority ranking |
| 9 | `patrol_router.py` | Generate shift-based patrol routes (GeoJSON) |
| 10 | `prediction.py` | Train XGBoost model, generate next-week predictions |
| 11 | `llm_agent.py` | Build NVIDIA NIM AI agent with live data context |

### Run the complete pipeline

```bash
python -m src.run_pipeline
```

This runs all stages in order. Expect 30–90 minutes depending on your machine (OSM download is the slowest step).

### Run individual stages

```bash
# Stage 1: Data ingestion only
python -m src.data_ingestion

# Stage 2: Road network (downloads ~500 MB OSM data)
python -m src.road_network

# Stage 3-6: Can be run individually too
python -m src.spatial_indexing
python -m src.hotspot_engine
python -m src.hotspot_stats
python -m src.pcis_engine

# Stage 7-10
python -m src.congestion_model
python -m src.enforcement_optimizer
python -m src.patrol_router
python -m src.prediction

# Test the AI agent interactively
python -m src.llm_agent
python -m src.llm_agent --demo
```

### Raw data file

Place the raw violation CSV at the project root:
```
PARKVISION-AI/
└── jan to may police violation_anonymized791b166.csv   ← place here
```

The filename must match exactly as specified in `config/settings.py`.

---

## 10. Dashboard Pages

### Violation Map (`[M]`)

The main map page with full-screen Leaflet view.

**Layer Controls (top-left):**
- **Heatmap** — Intensity map of violation density. Gradient: blue (low) → cyan → amber → red/white (high)
- **Hexagons** — H3 resolution-9 hexagons colored by PCIS score
- **Patrol Routes** — ACO-optimized routes rendered as dashed polylines
- **Stations** — Green circle markers for all 54 police stations

**When Patrol Routes are enabled**, a Shift Selector panel appears:
- All Shifts / Morning (amber) / Midday (blue) / Afternoon (purple) / Evening (pink) / Night (cyan)

**Analytics Panel (bottom-right):**
- Mini tabs: Hourly bar chart / Daily bar chart / Priority tier donut

---

### Filter Hotspots (`[F]`)

**Left panel filters:**
- **Police Station** — Dropdown of all 54 stations
- **Day of Week** — Toggle pills (Mon / Tue / Wed / Thu / Fri / Sat / Sun)
- **Time of Day** — Hour bucket pills (Early 0-6h / Morning 6-10h / Midday 10-14h / Afternoon 14-17h / Evening 17-21h / Night 21-24h)
- **Vehicle Type** — Dropdown (Car / Scooter / Motor Cycle / Passenger Auto / etc.)
- **Priority Tier** — Pills (Urgent / High / Medium / Low)

**Right panel:**
- Card grid of matching hotspots (PCIS / Violations / CHR / Peak Hour per card)
- Sort by: Priority Rank / PCIS / CHR / Violations
- Active filter summary bar
- Result count badge

---

### Enforcement Planner (`[P]`)

**Configuration:**
- Day of week (pills)
- Shift start hour (slider 0–22)
- Shift end hour (slider 1–24)
- Number of officers (stepper 1–5)

**Output (after "Generate Patrol Plan"):**
- Color-coded markers on map for each officer's stops
- Dashed polylines showing route order
- Per-officer card showing: stop number / station name / ETA / expected violations

**Algorithm:**
1. Filter violations by day and hour window
2. Aggregate by H3 hex — compute priority = `violation_count × PCIS × consistency`
3. K-Means cluster into N officer zones
4. Greedy nearest-neighbor routing within each zone
5. Check feasibility (travel time + 20 min dwell per stop)

---

### AI Chat (`[A]`)

**How it works:**
- Powered by NVIDIA NIM — `meta/llama-3.3-70b-instruct`
- Live data context (top stations, hourly patterns, vehicle types, predictions) injected into every request
- Conversation history maintained across turns in the same session
- "Reset Conversation" button clears server-side history

**Quick prompt chips:**
- Top priority station?
- 4–7 AM spike explanation
- Top hotspot hexagons
- Worst vehicle type by congestion
- Unusual patterns for review
- What is PCIS?

**Example questions it can answer:**
- "Which stations have the highest CHR score?"
- "Why is there a spike in violations at 4 AM?"
- "Compare weekday vs weekend in Upparpet"
- "Predict violations for next Monday"
- "What does the PCIS proximity factor measure?"
- "Which shift should I prioritize for Koramangala?"

---

## 11. API Reference

The FastAPI backend runs on `http://localhost:8000`. All endpoints return JSON.

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves the dashboard HTML |
| GET | `/api/config` | Returns API key availability flags |
| GET | `/api/summary` | City-wide stats (total violations, CHR, tiers, avg PCIS) |
| GET | `/api/hotspots` | Top hotspots with full filter support |
| GET | `/api/heatmap` | All hexagons for heatmap rendering |
| GET | `/api/patrol-routes` | GeoJSON patrol routes, optionally filtered by `?shift=morning` |
| GET | `/api/stations` | GeoJSON of all 54 police station locations |
| GET | `/api/temporal/{area}` | Hourly/daily patterns for a station or `city` |
| GET | `/api/predict` | XGBoost predictions for next 7 days |
| GET | `/api/filter-options` | All dropdown/pill options (stations, vehicle types, days, hours) |

### Feature Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/planner` | Generate enforcement patrol plan |
| GET | `/api/station-comparison` | 6-metric scorecard for all stations |
| GET | `/api/temporal-matrix` | 7×24 day-hour violation matrix |
| GET | `/api/gap-analysis` | 3-method under-recording detection |
| GET | `/api/vehicle-profiles` | Vehicle type breakdown per station |
| GET | `/api/weekend-split` | Weekday vs weekend split per station |

### AI Chat Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send a query to the AI agent |
| POST | `/api/chat/reset` | Clear conversation history |

### Hotspot Filter Parameters

`GET /api/hotspots` supports:

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `n` | int | `?n=50` | Max results (1–500, default 50) |
| `station` | str | `?station=Koramangala` | Filter by station name (partial match) |
| `min_pcis` | float | `?min_pcis=0.6` | Minimum PCIS threshold |
| `tier` | str | `?tier=URGENT` | Filter by priority tier |
| `day` | str | `?day=Monday` | Filter by day of week |
| `hour_start` | int | `?hour_start=6` | Filter by hour range start |
| `hour_end` | int | `?hour_end=10` | Filter by hour range end |
| `vehicle_type` | str | `?vehicle_type=SCOOTER` | Filter by vehicle type |

### AI Chat Payload

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Which station needs the most enforcement support?"}'
```

### Planner Payload

```bash
curl -X POST http://localhost:8000/api/planner \
  -H "Content-Type: application/json" \
  -d '{
    "day": "Monday",
    "start_hour": 8,
    "end_hour": 12,
    "n_officers": 3
  }'
```

---

## 12. Data Science Methodology

### PCIS (Parking Congestion Impact Score)

Five-component weighted score per H3 hexagon:

| Component | Weight | Formula |
|-----------|--------|---------|
| Capacity Reduction Factor | 30% | `vehicle_footprint_area / (lane_width × lane_length)` |
| Proximity Factor | 20% | Distance-decay from nearest junction (1.0 at junction → 0.1 residential) |
| Temporal Demand Multiplier | 20% | Demand curve by hour and weekday/weekend |
| Vehicle Obstruction Factor | 15% | Movement hindrance per vehicle class |
| Network Criticality | 15% | Road betweenness centrality (OSM graph) |

**PCIS Tiers:**

| Tier | Range | Action |
|------|-------|--------|
| MONITOR | 0.0 – 0.2 | Routine monitoring |
| LOW | 0.2 – 0.4 | Scheduled enforcement |
| MEDIUM | 0.4 – 0.6 | Prioritized deployment |
| HIGH | 0.6 – 0.8 | Immediate action |
| URGENT | 0.8 – 1.0 | Emergency intervention |

### CHR (Congestion Hours Recovered)

Quantifies the congestion relief from one enforcement action:

```
CHR = flow_rate × delay_per_vehicle × vehicles_affected_per_hour
```

Units: vehicle-hours/day. Used as the primary enforcement ROI metric for ranking patrol priorities.

### ST-DBSCAN Parameters

| Scale | Spatial epsilon | Temporal epsilon | Min points |
|-------|----------------|-----------------|------------|
| Micro | 50 m | 2 hours | 5 |
| Meso | 150 m | 2 hours | 5 |
| Macro | 500 m | 2 hours | 5 |

### Prediction Model

XGBoost regressor with features:
- Day of week (one-hot)
- Hour of day
- H3 hexagon (embedding)
- Historical violation counts (lag features)
- Road type (main/residential)
- PCIS score

---

## 13. Project File Structure

```
PARKVISION-AI/
├── .env                          # API keys (never committed)
├── .env.example                  # Template for .env
├── .gitignore
├── requirements.txt              # All dependencies with pinned versions
├── README.md
├── PROJECT_FEATURES.md           # Feature specification
│
├── config/
│   └── settings.py               # All constants, paths, PCIS weights
│
├── src/
│   ├── __init__.py
│   ├── run_pipeline.py           # Master pipeline runner (runs all stages)
│   ├── data_ingestion.py         # Stage 1: CSV cleaning and validation
│   ├── road_network.py           # Stage 2: OSMnx road graph download
│   ├── spatial_indexing.py       # Stage 3: H3 hex binning, hex stats
│   ├── hotspot_engine.py         # Stage 4: ST-DBSCAN clustering
│   ├── hotspot_stats.py          # Stage 5: Getis-Ord Gi* validation
│   ├── pcis_engine.py            # Stage 6: PCIS scoring
│   ├── congestion_model.py       # Stage 7: CHR + Location Memory
│   ├── enforcement_optimizer.py  # Stage 8: K-Means + priority ranking
│   ├── patrol_router.py          # Stage 9: Shift-based patrol routes
│   ├── prediction.py             # Stage 10: XGBoost predictions
│   ├── api_server.py             # FastAPI backend (20+ endpoints)
│   └── llm_agent.py              # NVIDIA NIM AI agent
│
├── dashboard/
│   └── index.html                # Single-page dashboard (Leaflet + Chart.js)
│
├── data/                         # Processed parquet files (auto-generated)
│   ├── pcis_scored_violations.parquet
│   ├── enforcement_priorities.parquet
│   ├── predicted_violations.parquet
│   ├── police_stations.geojson
│   └── ... (12 more files)
│
├── output/                       # GeoJSON and report outputs
│   ├── patrol_routes.geojson
│   ├── pcis_hotspots.geojson
│   ├── cluster_profiles.geojson
│   └── llm_demo_conversations.md
│
├── models/                       # Trained ML models
│   └── violation_predictor.joblib
│
└── cache/                        # OSM tile cache (auto-generated)
```

---

## 14. Troubleshooting

### `geopandas` or `pyogrio` install fails on Windows

Use pre-built wheels from the Unofficial Windows Binaries:

```powershell
# Install in this order
pip install wheel
pip install pyproj
pip install Fiona
pip install geopandas
```

Or use conda:
```bash
conda install -c conda-forge geopandas pyogrio
```

### `h3` install fails

```bash
pip install h3==4.5.0 --no-build-isolation
```

### `osmnx` fails to download road network

The OSM download requires internet access. If behind a proxy:
```python
# In config/settings.py, add:
import os
os.environ["HTTP_PROXY"] = "http://your-proxy:port"
os.environ["HTTPS_PROXY"] = "http://your-proxy:port"
```

### AI Chat returns "Rate Limit Reached"

Your NVIDIA NIM free-tier quota was hit. Solutions:
1. Wait a few seconds and retry (NVIDIA resets per-minute)
2. Check your rate limits at [build.nvidia.com](https://build.nvidia.com)
3. The system automatically falls back to Gemini if `GEMINI_API_KEY` is set

### Server fails to start — `ModuleNotFoundError`

Make sure your virtual environment is activated:
```powershell
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# Then verify
python -c "import fastapi; print('OK')"
```

### Dashboard loads but map tiles don't appear

- TomTom key is missing or invalid → the map falls back to CartoDB dark tiles automatically
- Check browser console (F12) for any tile 403 errors
- Verify `TOMTOM_API_KEY` is set in `.env`

### `data/pcis_scored_violations.parquet` not found

You need to run the data pipeline first:
```bash
python -m src.run_pipeline
```

Or check that the raw CSV file is in the project root with the exact filename:
```
jan to may police violation_anonymized791b166.csv
```

### Port 8000 already in use

```powershell
# Windows — find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <pid_number> /F

# Or use a different port
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8080
# Then open http://localhost:8080
```

### Slow startup (first run)

The first request to each endpoint triggers lazy data loading from parquet files. This takes 3–10 seconds. Subsequent requests are instant. This is normal.

---

## Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

## License

MIT License — see `LICENSE` file for details.

## Acknowledgements

- **OpenStreetMap** — Road network data
- **NVIDIA NIM** — LLaMA-3.3-70B inference
- **H3 by Uber** — Hierarchical hexagonal indexing
- **Leaflet.js** — Interactive mapping
- **Chart.js** — Data visualization
- **Bengaluru Traffic Police** — Violation data source
