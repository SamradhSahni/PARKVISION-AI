"""
PARKVISION AI — Stage 12: FastAPI Dashboard Backend
=====================================================
REST API serving processed parking violation data for the interactive dashboard.

Usage:
    uvicorn src.api_server:app --reload --port 8000
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR, OUTPUT_DIR, DASHBOARD_DIR,
    PCIS_SCORED_PARQUET, CLUSTER_PROFILES_PARQUET,
    ENFORCEMENT_PRIORITIES_PARQUET, LOCATION_MEMORY_PARQUET,
    PREDICTED_VIOLATIONS_PARQUET, H3_HOTSPOT_SIG_PARQUET,
    POLICE_STATIONS_GEOJSON, PATROL_ROUTES_GEOJSON,
    GEMINI_API_KEY,
)

logging.basicConfig(level="INFO")
logger = logging.getLogger("api_server")

app = FastAPI(title="PARKVISION AI", version="1.0.0",
              description="Parking Congestion Intelligence API")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Lazy data cache ──────────────────────────────────────────────
_cache = {}

def _load(key, path, loader="parquet"):
    if key not in _cache:
        p = Path(path)
        if not p.exists():
            return None
        if loader == "parquet":
            _cache[key] = pd.read_parquet(p)
        elif loader == "geojson":
            _cache[key] = gpd.read_file(p)
        elif loader == "json":
            with open(p) as f:
                _cache[key] = json.load(f)
    return _cache.get(key)


# ── Serve dashboard static files ─────────────────────────────────
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
async def serve_dashboard():
    index = DASHBOARD_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "PARKVISION AI API is running. Dashboard not found at /dashboard/index.html"}

app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


# ── API Endpoints ────────────────────────────────────────────────

@app.get("/api/hotspots")
async def get_hotspots(
    n: int = Query(50, ge=1, le=500),
    station: Optional[str] = None,
    min_pcis: float = Query(0.0, ge=0, le=1),
    tier: Optional[str] = None,
):
    """Get top hotspots ranked by CHR enforcement priority."""
    df = _load("priorities", ENFORCEMENT_PRIORITIES_PARQUET)
    if df is None:
        raise HTTPException(404, "Data not loaded")
    data = df.copy()
    if station:
        data = data[data["police_station"].str.contains(station, case=False, na=False)]
    if min_pcis > 0:
        data = data[data["pcis_mean"] >= min_pcis]
    if tier:
        data = data[data["priority_tier"] == tier.upper()]

    top = data.head(n)
    records = top[["h3_index","priority_rank","priority_tier","pcis_mean","pcis_max",
                    "chr","chr_normalized","violation_count","daily_frequency",
                    "avg_capacity_reduction","avg_proximity","police_station",
                    "peak_hour","centroid_lat","centroid_lon","pct_main_road",
                    "pct_junction","avg_road_width"]].to_dict(orient="records")
    return {"hotspots": records, "total": len(data)}


@app.get("/api/pcis/{h3_index}")
async def get_pcis_detail(h3_index: str):
    """Get PCIS breakdown for a specific hexagon."""
    df = _load("priorities", ENFORCEMENT_PRIORITIES_PARQUET)
    if df is None:
        raise HTTPException(404, "Data not loaded")
    row = df[df["h3_index"] == h3_index]
    if len(row) == 0:
        raise HTTPException(404, f"Hexagon {h3_index} not found")
    return row.iloc[0].to_dict()


@app.get("/api/heatmap")
async def get_heatmap():
    """Get H3 hexagon heatmap data (PCIS-colored)."""
    df = _load("priorities", ENFORCEMENT_PRIORITIES_PARQUET)
    if df is None:
        raise HTTPException(404, "Data not loaded")
    cols = ["h3_index","pcis_mean","chr_normalized","violation_count",
            "priority_tier","centroid_lat","centroid_lon","police_station"]
    available = [c for c in cols if c in df.columns]
    return {"hexagons": df[available].to_dict(orient="records")}


@app.get("/api/hex-geojson")
async def get_hex_geojson():
    """Get H3 hex stats as GeoJSON for map overlay."""
    path = OUTPUT_DIR / "h3_hex_stats.geojson"
    if not path.exists():
        raise HTTPException(404, "GeoJSON not found")
    with open(path) as f:
        return json.load(f)


@app.get("/api/patrol-routes")
async def get_patrol_routes(shift: Optional[str] = None):
    """Get ACO patrol routes, optionally filtered by shift."""
    gdf = _load("routes", PATROL_ROUTES_GEOJSON, loader="geojson")
    if gdf is None:
        raise HTTPException(404, "Routes not found")
    if shift:
        gdf = gdf[gdf["shift"] == shift.lower()]
    # Convert to GeoJSON
    return json.loads(gdf.to_json())


@app.get("/api/stations")
async def get_stations():
    """Get police station profiles."""
    gdf = _load("stations", POLICE_STATIONS_GEOJSON, loader="geojson")
    if gdf is None:
        raise HTTPException(404, "Stations not found")
    return json.loads(gdf.to_json())


@app.get("/api/temporal/{area}")
async def get_temporal(area: str):
    """Get temporal patterns for a station area or 'city'."""
    df = _load("violations", PCIS_SCORED_PARQUET)
    if df is None:
        raise HTTPException(404, "Data not loaded")
    if area.lower() != "city":
        df = df[df["police_station"].str.contains(area, case=False, na=False)]
    if len(df) == 0:
        raise HTTPException(404, f"No data for {area}")

    hourly = df["hour"].value_counts().sort_index().to_dict()
    daily = df["day_of_week"].value_counts().sort_index().to_dict()
    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    monthly = df["month"].value_counts().sort_index().to_dict()

    return {
        "area": area, "total": len(df),
        "hourly": {int(k): int(v) for k, v in hourly.items()},
        "daily": {day_names[int(k)]: int(v) for k, v in daily.items()},
        "monthly": {int(k): int(v) for k, v in monthly.items()},
        "avg_pcis": round(float(df["pcis"].mean()), 3),
        "peak_hour": int(df["hour"].value_counts().index[0]),
    }


@app.get("/api/predict")
async def get_predictions():
    """Get violation predictions for next week."""
    df = _load("predictions", PREDICTED_VIOLATIONS_PARQUET)
    if df is None:
        raise HTTPException(404, "Predictions not found")
    daily = df.groupby(["pred_date","pred_dow_name"]).agg(
        total=("predicted_violations","sum"),
        max_hex=("predicted_violations","max"),
    ).reset_index()
    top_hex = df.groupby("h3_index")["predicted_violations"].sum().nlargest(10)
    return {
        "daily": daily.to_dict(orient="records"),
        "top_hexes": [{"h3":k,"predicted":round(v,0)} for k,v in top_hex.items()],
    }


@app.get("/api/summary")
async def get_summary():
    """Get overall system summary statistics."""
    pri = _load("priorities", ENFORCEMENT_PRIORITIES_PARQUET)
    if pri is None:
        raise HTTPException(404, "Data not loaded")
    tiers = pri["priority_tier"].value_counts().to_dict()
    return {
        "total_hexagons": len(pri),
        "total_violations": int(pri["violation_count"].sum()),
        "total_chr": round(float(pri["chr"].sum()), 0),
        "avg_pcis": round(float(pri["pcis_mean"].mean()), 3),
        "tiers": tiers,
        "top_station": pri.iloc[0]["police_station"],
        "top_chr": round(float(pri.iloc[0]["chr"]), 0),
    }


@app.post("/api/chat")
async def chat_endpoint(payload: dict):
    """LLM chat proxy — sends query to Gemini with function calling."""
    query = payload.get("query", "")
    if not query:
        raise HTTPException(400, "Missing 'query' field")
    try:
        from src.llm_agent import create_agent, handle_query
        model = create_agent()
        chat = model.start_chat()
        response = handle_query(model, chat, query)
        return {"response": response}
    except Exception as e:
        return {"response": f"LLM Error: {str(e)}"}


@app.get("/api/ripple-contours")
async def get_ripple_contours():
    """Get congestion ripple contours as GeoJSON."""
    path = OUTPUT_DIR / "ripple_contours.geojson"
    if not path.exists():
        raise HTTPException(404, "Ripple contours not found")
    with open(path) as f:
        return json.load(f)


@app.get("/api/cluster-profiles")
async def get_cluster_profiles():
    """Get cluster profiles as GeoJSON."""
    path = OUTPUT_DIR / "cluster_profiles.geojson"
    if not path.exists():
        raise HTTPException(404, "Cluster profiles not found")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8000, reload=True)
