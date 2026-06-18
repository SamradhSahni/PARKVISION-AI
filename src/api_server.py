"""
PARKVISION AI — FastAPI Dashboard Backend
==========================================
REST API serving all processed parking violation data.

Usage:
    uvicorn src.api_server:app --reload --port 8000
"""

import sys
import json
import math
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List

import pandas as pd
import geopandas as gpd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import os
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR, OUTPUT_DIR, DASHBOARD_DIR,
    PCIS_SCORED_PARQUET, CLUSTER_PROFILES_PARQUET,
    ENFORCEMENT_PRIORITIES_PARQUET, LOCATION_MEMORY_PARQUET,
    PREDICTED_VIOLATIONS_PARQUET, H3_HOTSPOT_SIG_PARQUET,
    POLICE_STATIONS_GEOJSON, PATROL_ROUTES_GEOJSON,
    GEMINI_API_KEY, NVIDIA_API_KEY,
)

TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY", "")

logging.basicConfig(level="INFO")
logger = logging.getLogger("api_server")

app = FastAPI(title="PARKVISION AI", version="2.0.0",
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

def _violations():
    return _load("violations", PCIS_SCORED_PARQUET)

# ── Static file serving ──────────────────────────────────────────
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
async def serve_dashboard():
    index = DASHBOARD_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Dashboard not found"}

app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


@app.get("/api/config")
async def get_config():
    """Return public config (API keys needed by frontend)."""
    return {
        "tomtom_api_key": TOMTOM_API_KEY,
        "has_tomtom": bool(TOMTOM_API_KEY),
        "has_gemini": bool(GEMINI_API_KEY),
    }


# ── Gemini agent (lazy init, cached for the process lifetime) ────
_agent_model = None
_agent_chat = None

def _get_agent():
    """Return (agent, history) — initialize once, reuse across requests."""
    global _agent_model, _agent_chat
    if _agent_model is None:
        from src.llm_agent import create_agent
        backend = "NVIDIA NIM" if NVIDIA_API_KEY else "Gemini"
        logger.info(f"Initializing {backend} agent...")
        _agent_model = create_agent()
        _agent_chat = []   # conversation history list for NVIDIA NIM
        logger.info(f"{backend} agent ready.")
    return _agent_model, _agent_chat

# ================================================================
# EXISTING ENDPOINTS
# ================================================================

@app.get("/api/hotspots")
async def get_hotspots(
    n: int = Query(50, ge=1, le=500),
    station: Optional[str] = None,
    min_pcis: float = Query(0.0, ge=0, le=1),
    tier: Optional[str] = None,
    day: Optional[str] = None,
    hour_start: Optional[int] = None,
    hour_end: Optional[int] = None,
    vehicle_type: Optional[str] = None,
):
    """Get top hotspots with optional filters for day, hour, vehicle type."""
    df = _load("priorities", ENFORCEMENT_PRIORITIES_PARQUET)
    vdf = _violations()
    if df is None:
        raise HTTPException(404, "Data not loaded")

    data = df.copy()

    if (day or hour_start is not None or vehicle_type) and vdf is not None:
        vfilt = vdf.copy()
        if day:
            vfilt = vfilt[vfilt["day_name"].str.lower() == day.lower()]
        if hour_start is not None and hour_end is not None:
            vfilt = vfilt[(vfilt["hour"] >= hour_start) & (vfilt["hour"] < hour_end)]
        elif hour_start is not None:
            vfilt = vfilt[vfilt["hour"] >= hour_start]
        if vehicle_type:
            vfilt = vfilt[vfilt["vehicle_type"].str.upper() == vehicle_type.upper()]
        active_hexes = set(vfilt["h3_index"].unique())
        data = data[data["h3_index"].isin(active_hexes)]

    if station:
        data = data[data["police_station"].str.contains(station, case=False, na=False)]
    if min_pcis > 0:
        data = data[data["pcis_mean"] >= min_pcis]
    if tier:
        data = data[data["priority_tier"] == tier.upper()]

    top = data.head(n)
    cols = ["h3_index","priority_rank","priority_tier","pcis_mean","pcis_max",
            "chr","chr_normalized","violation_count","daily_frequency",
            "avg_capacity_reduction","avg_proximity","police_station",
            "peak_hour","centroid_lat","centroid_lon","pct_main_road",
            "pct_junction","avg_road_width"]
    available = [c for c in cols if c in top.columns]
    return {"hotspots": top[available].to_dict(orient="records"), "total": len(data)}


@app.get("/api/pcis/{h3_index}")
async def get_pcis_detail(h3_index: str):
    df = _load("priorities", ENFORCEMENT_PRIORITIES_PARQUET)
    if df is None:
        raise HTTPException(404, "Data not loaded")
    row = df[df["h3_index"] == h3_index]
    if len(row) == 0:
        raise HTTPException(404, f"Hexagon {h3_index} not found")
    return row.iloc[0].to_dict()


@app.get("/api/heatmap")
async def get_heatmap():
    df = _load("priorities", ENFORCEMENT_PRIORITIES_PARQUET)
    if df is None:
        raise HTTPException(404, "Data not loaded")
    cols = ["h3_index","pcis_mean","chr_normalized","violation_count",
            "priority_tier","centroid_lat","centroid_lon","police_station"]
    available = [c for c in cols if c in df.columns]
    return {"hexagons": df[available].to_dict(orient="records")}


@app.get("/api/patrol-routes")
async def get_patrol_routes(shift: Optional[str] = None):
    gdf = _load("routes", PATROL_ROUTES_GEOJSON, loader="geojson")
    if gdf is None:
        raise HTTPException(404, "Routes not found")
    if shift:
        gdf = gdf[gdf["shift"] == shift.lower()]
    return json.loads(gdf.to_json())


@app.get("/api/stations")
async def get_stations():
    gdf = _load("stations", POLICE_STATIONS_GEOJSON, loader="geojson")
    if gdf is None:
        raise HTTPException(404, "Stations not found")
    return json.loads(gdf.to_json())


@app.get("/api/temporal/{area}")
async def get_temporal(area: str):
    df = _violations()
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


# ================================================================
# NEW ENDPOINTS — Feature Pages F2–F9
# ================================================================

@app.get("/api/filter-options")
async def get_filter_options():
    """F3: Dropdown options for map filters."""
    df = _violations()
    if df is None:
        raise HTTPException(404, "Data not loaded")
    stations = sorted(df["police_station"].dropna().unique().tolist())
    vehicle_types = sorted(df["vehicle_type"].dropna().unique().tolist())
    return {
        "stations": stations,
        "vehicle_types": vehicle_types,
        "days": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
        "hour_buckets": [
            {"label": "Early Morning (0-6)", "start": 0, "end": 6},
            {"label": "Morning (6-10)", "start": 6, "end": 10},
            {"label": "Midday (10-14)", "start": 10, "end": 14},
            {"label": "Afternoon (14-17)", "start": 14, "end": 17},
            {"label": "Evening (17-21)", "start": 17, "end": 21},
            {"label": "Night (21-24)", "start": 21, "end": 24},
        ]
    }


@app.post("/api/planner")
async def run_planner(payload: dict):
    """F2: Enforcement Planner — K-Means cluster hotspots, greedy route per officer."""
    from sklearn.cluster import KMeans

    day = payload.get("day", "Monday")
    start_h = int(payload.get("start_hour", 8))
    end_h = int(payload.get("end_hour", 12))
    n_officers = max(1, min(5, int(payload.get("n_officers", 3))))

    df = _violations()
    if df is None:
        raise HTTPException(404, "Data not loaded")

    filt = df[
        (df["day_name"].str.lower() == day.lower()) &
        (df["hour"] >= start_h) &
        (df["hour"] < end_h)
    ].copy()

    if len(filt) < 5:
        return {"error": f"Not enough data for {day} {start_h}:00-{end_h}:00", "officers": []}

    agg = filt.groupby("h3_index").agg(
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
        violation_count=("id", "count"),
        avg_pcis=("pcis", "mean"),
        unique_dates=("date", "nunique"),
        police_station=("police_station", lambda x: x.mode().iloc[0] if len(x) > 0 else ""),
    ).reset_index()

    max_count = agg["violation_count"].max()
    total_days = filt["date"].nunique()
    agg["norm_count"] = agg["violation_count"] / max_count
    agg["consistency"] = (agg["unique_dates"] / max(total_days, 1)).clip(0, 1)
    agg["priority"] = (
        agg["norm_count"] *
        (1 + agg["avg_pcis"] * 0.5) *
        (0.5 + agg["consistency"] * 0.5)
    )

    max_zones = n_officers * 8
    top_zones = agg.nlargest(max_zones, "priority").reset_index(drop=True)

    if len(top_zones) < n_officers:
        return {"error": "Too few hotspots for this time window", "officers": []}

    coords = top_zones[["lat","lon"]].values
    if len(top_zones) == n_officers:
        top_zones["cluster"] = list(range(n_officers))
    else:
        km = KMeans(n_clusters=n_officers, n_init=10, random_state=42)
        top_zones["cluster"] = km.fit_predict(coords)

    shift_hours = end_h - start_h
    colors = ["#3b82f6","#ec4899","#10b981","#f59e0b","#8b5cf6"]
    officers = []

    for officer_id in range(n_officers):
        cluster_zones = top_zones[top_zones["cluster"] == officer_id].copy().reset_index(drop=True)
        if len(cluster_zones) == 0:
            continue

        start_lat = cluster_zones["lat"].mean()
        start_lon = cluster_zones["lon"].mean()
        unvisited = list(range(len(cluster_zones)))
        route_order = []
        cur_lat, cur_lon = start_lat, start_lon
        time_used_min = 0.0

        while unvisited:
            best_i, best_dist = None, float("inf")
            for i in unvisited:
                row = cluster_zones.iloc[i]
                dist = math.sqrt((row["lat"]-cur_lat)**2 + (row["lon"]-cur_lon)**2) * 111
                if dist < best_dist:
                    best_dist, best_i = dist, i
            travel_min = (best_dist / 20.0) * 60
            if time_used_min + travel_min + 20 > shift_hours * 60:
                break
            route_order.append(best_i)
            row = cluster_zones.iloc[best_i]
            time_used_min += travel_min + 20
            cur_lat, cur_lon = row["lat"], row["lon"]
            unvisited.remove(best_i)

        stops = []
        cumulative_min = 0.0
        prev_lat, prev_lon = start_lat, start_lon
        for idx, zi in enumerate(route_order):
            row = cluster_zones.iloc[zi]
            dist_km = math.sqrt((row["lat"]-prev_lat)**2 + (row["lon"]-prev_lon)**2) * 111
            travel_min = (dist_km / 20.0) * 60
            cumulative_min += travel_min
            eta_h = start_h + (cumulative_min / 60)
            eta_str = f"{int(eta_h):02d}:{int((eta_h % 1)*60):02d}"
            stops.append({
                "stop_num": idx + 1,
                "h3_index": row["h3_index"],
                "lat": round(float(row["lat"]), 4),
                "lon": round(float(row["lon"]), 4),
                "police_station": row["police_station"],
                "expected_violations": int(row["violation_count"]),
                "pcis": round(float(row["avg_pcis"]), 3),
                "priority": round(float(row["priority"]), 3),
                "travel_km": round(dist_km, 2),
                "travel_min": round(travel_min, 1),
                "eta": eta_str,
            })
            cumulative_min += 20
            prev_lat, prev_lon = row["lat"], row["lon"]

        officers.append({
            "officer_id": officer_id + 1,
            "color": colors[officer_id % len(colors)],
            "n_stops": len(stops),
            "total_expected_violations": sum(s["expected_violations"] for s in stops),
            "shift": f"{start_h:02d}:00 - {end_h:02d}:00",
            "stops": stops,
        })

    return {
        "day": day,
        "shift": f"{start_h:02d}:00 - {end_h:02d}:00",
        "n_officers": n_officers,
        "total_zones_found": len(top_zones),
        "officers": officers,
    }


@app.get("/api/station-comparison")
async def get_station_comparison():
    """F4: 6-metric scorecard for all 54 police stations."""
    df = _violations()
    if df is None:
        raise HTTPException(404, "Data not loaded")

    results = []
    for station, grp in df.groupby("police_station"):
        total = len(grp)
        if total < 10:
            continue
        approved = (grp["validation_status"] == "approved").sum()
        quality = round(float(approved / total * 100), 1)
        active_hours = grp["hour"].nunique()
        coverage = round(float(active_hours / 24 * 100), 1)
        unique_dates = grp["date"].nunique()
        daily_rate = total / max(unique_dates, 1)
        responsiveness = round(float(min(daily_rate / 5.0, 1.0) * 100), 1)
        weekend_pct = float(grp["is_weekend"].mean())
        expected_weekend = 2/7
        balance = round(float(max(0, 1 - abs(weekend_pct - expected_weekend) / expected_weekend) * 100), 1)
        vehicle_types = grp["vehicle_type"].nunique()
        complexity_raw = vehicle_types * math.log(total + 1)
        results.append({
            "station": station,
            "total_violations": total,
            "quality_score": quality,
            "coverage_score": coverage,
            "responsiveness_score": responsiveness,
            "balance_score": balance,
            "complexity_raw": complexity_raw,
            "avg_pcis": round(float(grp["pcis"].mean()), 3),
            "peak_hour": int(grp["hour"].value_counts().index[0]),
            "weekend_pct": round(weekend_pct * 100, 1),
            "active_hours": active_hours,
            "vehicle_types": vehicle_types,
        })

    if not results:
        raise HTTPException(404, "No station data")

    result_df = pd.DataFrame(results)
    c_min = result_df["complexity_raw"].min()
    c_max = result_df["complexity_raw"].max()
    result_df["zone_complexity_score"] = ((result_df["complexity_raw"] - c_min) / max(c_max - c_min, 1) * 100).round(1)
    result_df["overall_score"] = (
        result_df["quality_score"] * 0.35 +
        result_df["coverage_score"] * 0.25 +
        result_df["responsiveness_score"] * 0.20 +
        result_df["balance_score"] * 0.15 +
        result_df["zone_complexity_score"] * 0.05
    ).round(1)
    result_df = result_df.drop(columns=["complexity_raw"])
    result_df = result_df.sort_values("overall_score", ascending=False).reset_index(drop=True)
    result_df["rank"] = result_df.index + 1
    return {"stations": result_df.to_dict(orient="records")}


@app.get("/api/temporal-matrix")
async def get_temporal_matrix():
    """F5: 7-day x 24-hour violation count matrix."""
    df = _violations()
    if df is None:
        raise HTTPException(404, "Data not loaded")

    day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    matrix = {}
    for dow in range(7):
        day_data = df[df["day_of_week"] == dow]
        hourly = day_data["hour"].value_counts().to_dict()
        matrix[day_names[dow]] = {h: int(hourly.get(h, 0)) for h in range(24)}

    hourly_total = df["hour"].value_counts().sort_index()
    early = df[(df["hour"] >= 4) & (df["hour"] < 7)]
    evening = df[(df["hour"] >= 17) & (df["hour"] < 20)]

    return {
        "matrix": matrix,
        "hourly_total": {int(k): int(v) for k, v in hourly_total.items()},
        "early_morning_count": len(early),
        "peak_evening_count": len(evening),
        "early_morning_pct": round(len(early) / len(df) * 100, 1),
        "day_names": day_names,
    }


@app.get("/api/gap-analysis")
async def get_gap_analysis():
    """F6: 3-method gap detection per station."""
    df = _violations()
    if df is None:
        raise HTTPException(404, "Data not loaded")

    results = []
    for station, grp in df.groupby("police_station"):
        total = len(grp)
        if total < 5:
            continue
        vehicle_types = grp["vehicle_type"].nunique()
        diversity_score = vehicle_types / math.log(total + 2)
        active_hours = grp["hour"].nunique()
        coverage_gap = (24 - active_hours) / 24
        junctions = grp["has_junction"].sum()
        junction_gap = float(junctions) / max(total, 1)
        results.append({
            "station": station,
            "total_violations": total,
            "vehicle_diversity_score": round(diversity_score, 3),
            "time_coverage_gap": round(coverage_gap, 3),
            "junction_density_score": round(junction_gap, 3),
            "active_hours": active_hours,
            "vehicle_types": vehicle_types,
            "junctions_count": int(junctions),
        })

    if not results:
        raise HTTPException(404, "No data")

    result_df = pd.DataFrame(results)
    for col, norm_col in [
        ("vehicle_diversity_score", "diversity_norm"),
        ("time_coverage_gap", "coverage_norm"),
        ("junction_density_score", "junction_norm"),
    ]:
        mn, mx = result_df[col].min(), result_df[col].max()
        result_df[norm_col] = ((result_df[col] - mn) / max(mx - mn, 0.001) * 100).round(1)

    result_df["combined_score"] = (
        result_df["diversity_norm"] * 0.40 +
        result_df["coverage_norm"] * 0.35 +
        result_df["junction_norm"] * 0.25
    ).round(1)
    result_df = result_df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    result_df["gap_rank"] = result_df.index + 1
    return {"stations": result_df.to_dict(orient="records")}


@app.get("/api/vehicle-profiles")
async def get_vehicle_profiles(station: Optional[str] = None):
    """F7: Vehicle type breakdown per station."""
    df = _violations()
    if df is None:
        raise HTTPException(404, "Data not loaded")

    notes = {
        "SCOOTER": "Scooter-heavy -> focus on footpaths, narrow lanes, and two-wheeler parking bays.",
        "MOTOR CYCLE": "Two-wheeler dominant -> check footpaths, shop frontages, lane encroachments.",
        "CAR": "Car-heavy -> focus on main road lanes, no-parking zones, junction clearances.",
        "PASSENGER AUTO": "Auto-rickshaw heavy -> check auto stands, bus stop encroachments.",
        "MAXI-CAB": "Maxi-cab heavy -> focus on school/office zones and loading bay violations.",
        "LGV": "Light goods vehicle heavy -> check loading zones and market area double parking.",
        "PRIVATE BUS": "Bus heavy -> focus on bus stops, school zones, road width violations.",
    }

    if station and station.lower() != "all":
        sdf = df[df["police_station"].str.contains(station, case=False, na=False)]
        if len(sdf) == 0:
            raise HTTPException(404, f"Station not found: {station}")
        vc = sdf["vehicle_type"].value_counts().head(8)
        total = len(sdf)
        dominant = vc.index[0] if len(vc) > 0 else "UNKNOWN"
        return {
            "station": station,
            "total": total,
            "distribution": [
                {"type": t, "count": int(c), "pct": round(c/total*100, 1)}
                for t, c in vc.items()
            ],
            "dominant_vehicle": dominant,
            "enforcement_note": notes.get(dominant, f"{dominant} dominant -> enforce accordingly."),
        }

    top_types = df["vehicle_type"].value_counts().head(5).index.tolist()
    summary = []
    for stn, grp in df.groupby("police_station"):
        total = len(grp)
        row = {"station": stn, "total": total}
        vc = grp["vehicle_type"].value_counts()
        for vt in top_types:
            row[vt] = round(vc.get(vt, 0) / total * 100, 1)
        dom = vc.index[0] if len(vc) > 0 else "UNKNOWN"
        row["dominant"] = dom
        row["enforcement_note"] = notes.get(dom, "")
        summary.append(row)
    summary.sort(key=lambda x: x["total"], reverse=True)
    return {"top_vehicle_types": top_types, "stations": summary}


@app.get("/api/weekend-split")
async def get_weekend_split():
    """F8: Weekday vs weekend split per station."""
    df = _violations()
    if df is None:
        raise HTTPException(404, "Data not loaded")

    results = []
    for station, grp in df.groupby("police_station"):
        total = len(grp)
        if total < 10:
            continue
        weekend = int(grp["is_weekend"].sum())
        weekday = total - weekend
        weekend_pct = round(weekend / total * 100, 1)
        results.append({
            "station": station,
            "total": total,
            "weekday_count": weekday,
            "weekend_count": weekend,
            "weekend_pct": weekend_pct,
            "weekday_pct": round(100 - weekend_pct, 1),
        })

    results.sort(key=lambda x: x["weekend_pct"], reverse=True)
    city_weekend_pct = round(df["is_weekend"].mean() * 100, 1)
    return {
        "stations": results,
        "city_avg_weekend_pct": city_weekend_pct,
        "max_weekend_pct": results[0]["weekend_pct"] if results else 0,
        "min_weekend_pct": results[-1]["weekend_pct"] if results else 0,
        "max_station": results[0]["station"] if results else "",
        "min_station": results[-1]["station"] if results else "",
    }


@app.post("/api/chat")
async def chat_endpoint(payload: dict):
    """LLM chat proxy — NVIDIA NIM primary, Gemini fallback."""
    query = payload.get("query", "")
    if not query.strip():
        raise HTTPException(400, "Missing 'query' field")
    if not NVIDIA_API_KEY and not GEMINI_API_KEY:
        return {"response": "**Configuration Error**: No AI API key set. Add NVIDIA_API_KEY to your .env file."}
    try:
        from src.llm_agent import handle_query
        agent, history = _get_agent()
        response = handle_query(agent, history, query)
        return {"response": response}
    except Exception as e:
        err = str(e)
        logger.error(f"Chat error: {err}")
        if "429" in err or "quota" in err.lower() or "rate" in err.lower():
            return {"response": "**Rate Limit Reached** — Please wait a moment and try again."}
        if "401" in err or "unauthorized" in err.lower() or "invalid" in err.lower():
            return {"response": "**Authentication Error** — Check that your NVIDIA_API_KEY in .env is correct."}
        return {"response": f"**Error**: {err[:400]}"}


@app.post("/api/chat/reset")
async def reset_chat():
    """Reset the chat history (start a fresh conversation)."""
    global _agent_chat
    _agent_chat = []   # clear NVIDIA history list
    return {"status": "Chat history cleared"}


@app.get("/api/ripple-contours")
async def get_ripple_contours():
    path = OUTPUT_DIR / "ripple_contours.geojson"
    if not path.exists():
        raise HTTPException(404, "Not found")
    with open(path) as f:
        return json.load(f)


@app.get("/api/cluster-profiles")
async def get_cluster_profiles():
    path = OUTPUT_DIR / "cluster_profiles.geojson"
    if not path.exists():
        raise HTTPException(404, "Not found")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8000, reload=True)
