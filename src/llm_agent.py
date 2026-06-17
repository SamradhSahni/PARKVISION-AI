"""
PARKVISION AI — Stage 11: Gemini LLM Agent with Function Calling
==================================================================
Conversational AI agent powered by Google Gemini that answers natural
language queries about parking violations using function tools that
query the processed data files.

Usage:
    python -m src.llm_agent                    # Interactive chat mode
    python -m src.llm_agent --demo             # Run demo queries & save
"""

import sys
import json
import logging
import warnings
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import google.generativeai as genai

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR,
    OUTPUT_DIR,
    GEMINI_API_KEY,
    PCIS_SCORED_PARQUET,
    CLUSTER_PROFILES_PARQUET,
    ENFORCEMENT_PRIORITIES_PARQUET,
    LOCATION_MEMORY_PARQUET,
    PREDICTED_VIOLATIONS_PARQUET,
    H3_HOTSPOT_SIG_PARQUET,
    POLICE_STATIONS_GEOJSON,
    PATROL_ROUTES_GEOJSON,
    LOG_FORMAT,
    LOG_LEVEL,
)

warnings.filterwarnings("ignore")
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger("llm_agent")


# ============================================
# DATA LOADERS (lazy, cached)
# ============================================
_cache = {}

def _load(key, path, loader="parquet"):
    if key not in _cache:
        p = Path(path)
        if not p.exists():
            return None
        if loader == "parquet":
            _cache[key] = pd.read_parquet(p)
        elif loader == "json":
            import json as _json
            with open(p) as f:
                _cache[key] = _json.load(f)
        elif loader == "geojson":
            import geopandas as gpd
            _cache[key] = gpd.read_file(p)
    return _cache.get(key)


# ============================================
# FUNCTION TOOLS — query processed data
# ============================================
def get_top_hotspots(n: int = 10, police_station: Optional[str] = None,
                     min_pcis: float = 0.0) -> str:
    """Get top N parking violation hotspots ranked by enforcement priority (CHR score).
    
    Args:
        n: Number of top hotspots to return (default 10, max 50)
        police_station: Filter by police station name (optional, case-insensitive partial match)
        min_pcis: Minimum PCIS score filter (0.0 to 1.0)
    
    Returns:
        JSON string with hotspot details including rank, location, CHR score, PCIS, violations, and station.
    """
    priorities = _load("priorities", ENFORCEMENT_PRIORITIES_PARQUET)
    if priorities is None:
        return json.dumps({"error": "Enforcement priorities data not found"})
    
    df = priorities.copy()
    if police_station:
        df = df[df["police_station"].str.contains(police_station, case=False, na=False)]
    if min_pcis > 0:
        df = df[df["pcis_mean"] >= min_pcis]
    
    n = min(int(n), 50)
    top = df.head(n)
    
    results = []
    for _, row in top.iterrows():
        results.append({
            "rank": int(row["priority_rank"]),
            "h3_index": row["h3_index"],
            "chr_score": round(float(row["chr"]), 0),
            "chr_normalized": round(float(row["chr_normalized"]), 1),
            "priority_tier": row["priority_tier"],
            "pcis_mean": round(float(row["pcis_mean"]), 3),
            "daily_frequency": round(float(row["daily_frequency"]), 1),
            "violation_count": int(row["violation_count"]),
            "police_station": row["police_station"],
            "peak_hour": int(row["peak_hour"]),
            "capacity_reduction": round(float(row["avg_capacity_reduction"]), 3),
            "lat": round(float(row["centroid_lat"]), 4),
            "lon": round(float(row["centroid_lon"]), 4),
        })
    
    return json.dumps({"hotspots": results, "total_matching": len(df)})


def get_pcis_breakdown(h3_index: str) -> str:
    """Get detailed PCIS score breakdown for a specific H3 hexagon zone.
    
    Args:
        h3_index: The H3 hexagonal index identifier (e.g., '8960145b553ffff')
    
    Returns:
        JSON string with all 5 PCIS components, CHR score, and zone characteristics.
    """
    priorities = _load("priorities", ENFORCEMENT_PRIORITIES_PARQUET)
    if priorities is None:
        return json.dumps({"error": "Data not found"})
    
    row = priorities[priorities["h3_index"] == h3_index]
    if len(row) == 0:
        return json.dumps({"error": f"H3 index {h3_index} not found"})
    
    row = row.iloc[0]
    
    memory = _load("memory", LOCATION_MEMORY_PARQUET)
    memory_score = 0.0
    is_addiction = False
    if memory is not None:
        mem_row = memory[memory["h3_index"] == h3_index]
        if len(mem_row) > 0:
            memory_score = float(mem_row.iloc[0].get("location_memory_score", 0))
            is_addiction = bool(mem_row.iloc[0].get("is_addiction_zone", False))
    
    result = {
        "h3_index": h3_index,
        "priority_rank": int(row["priority_rank"]),
        "priority_tier": row["priority_tier"],
        "pcis_components": {
            "capacity_reduction": round(float(row["avg_capacity_reduction"]), 3),
            "proximity_factor": round(float(row["avg_proximity"]), 3),
            "severity": round(float(row["avg_severity"]), 3),
            "road_betweenness": round(float(row["avg_betweenness"]), 4),
        },
        "pcis_mean": round(float(row["pcis_mean"]), 3),
        "pcis_max": round(float(row["pcis_max"]), 3),
        "chr_score": round(float(row["chr"]), 0),
        "violation_count": int(row["violation_count"]),
        "daily_frequency": round(float(row["daily_frequency"]), 1),
        "unique_vehicles": int(row["unique_vehicles"]),
        "police_station": row["police_station"],
        "peak_hour": int(row["peak_hour"]),
        "pct_main_road": round(float(row["pct_main_road"]), 2),
        "pct_junction": round(float(row["pct_junction"]), 2),
        "avg_road_width_m": round(float(row["avg_road_width"]), 1),
        "location_memory_score": round(memory_score, 3),
        "is_addiction_zone": is_addiction,
    }
    
    return json.dumps(result)


def get_temporal_pattern(area: str) -> str:
    """Get temporal violation patterns for a police station area or the entire city.
    
    Args:
        area: Police station name (partial match) or 'city' for citywide patterns
    
    Returns:
        JSON with hourly distribution, daily distribution, peak times, and pattern classification.
    """
    df = _load("violations", PCIS_SCORED_PARQUET)
    if df is None:
        return json.dumps({"error": "Violation data not found"})
    
    if area.lower() != "city":
        df = df[df["police_station"].str.contains(area, case=False, na=False)]
        if len(df) == 0:
            return json.dumps({"error": f"No data found for area: {area}"})
    
    hourly = df["hour"].value_counts().sort_index()
    daily = df["day_of_week"].value_counts().sort_index()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    result = {
        "area": area,
        "total_violations": len(df),
        "hourly_distribution": {int(h): int(c) for h, c in hourly.items()},
        "peak_hour": int(hourly.idxmax()),
        "peak_hour_count": int(hourly.max()),
        "daily_distribution": {day_names[int(d)]: int(c) for d, c in daily.items()},
        "peak_day": day_names[int(daily.idxmax())],
        "weekend_percentage": round(float(df["is_weekend"].mean() * 100), 1),
        "peak_hour_percentage": round(float(df["is_peak_hour"].mean() * 100), 1),
        "avg_pcis": round(float(df["pcis"].mean()), 3),
    }
    
    return json.dumps(result)


def get_enforcement_routes(shift: str = "morning") -> str:
    """Get optimized patrol routes for a specific shift.
    
    Args:
        shift: Shift name - one of 'morning', 'midday', 'afternoon', 'evening', 'night'
    
    Returns:
        JSON with patrol routes including depot station, stops, CHR recovered, and travel time.
    """
    routes = _load("routes", PATROL_ROUTES_GEOJSON, loader="geojson")
    if routes is None:
        return json.dumps({"error": "Patrol routes data not found"})
    
    shift = shift.lower().strip()
    shift_routes = routes[routes["shift"] == shift]
    
    if len(shift_routes) == 0:
        available = routes["shift"].unique().tolist()
        return json.dumps({"error": f"Shift '{shift}' not found. Available: {available}"})
    
    results = []
    for _, row in shift_routes.nlargest(8, "total_chr").iterrows():
        results.append({
            "depot_station": row["depot_station"],
            "n_stops": int(row["n_stops"]),
            "total_chr_recovered": round(float(row["total_chr"]), 0),
            "travel_time_hours": float(row["travel_time_hours"]),
            "dwell_time_hours": float(row["dwell_time_hours"]),
            "total_time_hours": float(row["total_time_hours"]),
            "hotspot_ids": row["hotspot_ids"].split("|") if row["hotspot_ids"] else [],
        })
    
    return json.dumps({
        "shift": shift,
        "shift_window": row.get("shift_start", "") + " - " + row.get("shift_end", ""),
        "routes": results,
        "total_chr_all_routes": round(float(shift_routes["total_chr"].sum()), 0),
    })


def compare_areas(area1: str, area2: str) -> str:
    """Compare parking violation statistics between two police station areas.
    
    Args:
        area1: First police station name (partial match)
        area2: Second police station name (partial match)
    
    Returns:
        JSON comparison of violation counts, PCIS scores, temporal patterns, and enforcement needs.
    """
    df = _load("violations", PCIS_SCORED_PARQUET)
    if df is None:
        return json.dumps({"error": "Data not found"})
    
    def _area_stats(name):
        area_df = df[df["police_station"].str.contains(name, case=False, na=False)]
        if len(area_df) == 0:
            return {"error": f"No data for {name}"}
        return {
            "station": area_df["police_station"].mode().iloc[0],
            "total_violations": len(area_df),
            "unique_vehicles": int(area_df["vehicle_number"].nunique()),
            "avg_pcis": round(float(area_df["pcis"].mean()), 3),
            "max_pcis": round(float(area_df["pcis"].max()), 3),
            "avg_capacity_reduction": round(float(area_df["capacity_reduction"].mean()), 3),
            "pct_at_junction": round(float(area_df["has_junction"].mean() * 100), 1),
            "pct_main_road": round(float(area_df["road_is_main"].mean() * 100), 1),
            "peak_hour": int(area_df["hour"].value_counts().index[0]),
            "weekend_pct": round(float(area_df["is_weekend"].mean() * 100), 1),
            "avg_road_width": round(float(area_df["road_width_m"].mean()), 1),
            "daily_rate": round(len(area_df) / max(area_df["date"].nunique(), 1), 1),
        }
    
    return json.dumps({
        "area1": _area_stats(area1),
        "area2": _area_stats(area2),
    })


def get_prediction_summary(day: Optional[str] = None) -> str:
    """Get violation prediction summary for the upcoming week or a specific day.
    
    Args:
        day: Day name (e.g., 'Monday', 'Tuesday') or None for full week summary
    
    Returns:
        JSON with predicted violation counts per hexagon, top predicted hotspots.
    """
    preds = _load("predictions", PREDICTED_VIOLATIONS_PARQUET)
    if preds is None:
        return json.dumps({"error": "Prediction data not found"})
    
    if day:
        preds = preds[preds["pred_dow_name"].str.contains(day, case=False, na=False)]
        if len(preds) == 0:
            return json.dumps({"error": f"No predictions for day: {day}"})
    
    daily = preds.groupby(["pred_date", "pred_dow_name"]).agg(
        total_predicted=("predicted_violations", "sum"),
        max_hex=("predicted_violations", "max"),
        active_hexes=("predicted_violations", lambda x: int((x > 0.5).sum())),
    ).reset_index()
    
    top_hexes = preds.groupby("h3_index")["predicted_violations"].sum().nlargest(10)
    
    result = {
        "daily_predictions": [
            {
                "date": str(row["pred_date"])[:10],
                "day": row["pred_dow_name"],
                "total": round(float(row["total_predicted"]), 0),
                "max_hex": round(float(row["max_hex"]), 1),
                "active_hexes": int(row["active_hexes"]),
            }
            for _, row in daily.iterrows()
        ],
        "top_predicted_hotspots": [
            {"h3_index": idx, "predicted_weekly_total": round(float(total), 0)}
            for idx, total in top_hexes.items()
        ],
    }
    
    return json.dumps(result)


def get_station_profile(station_name: str) -> str:
    """Get detailed profile for a specific police station including workload and jurisdiction.
    
    Args:
        station_name: Police station name (partial match, case-insensitive)
    
    Returns:
        JSON with station location, violation stats, PCIS metrics, temporal pattern, and jurisdiction info.
    """
    stations = _load("stations", POLICE_STATIONS_GEOJSON, loader="geojson")
    if stations is None:
        return json.dumps({"error": "Station data not found"})
    
    matched = stations[stations["police_station"].str.contains(station_name, case=False, na=False)]
    if len(matched) == 0:
        available = stations["police_station"].tolist()
        return json.dumps({"error": f"Station not found. Available: {available}"})
    
    row = matched.iloc[0]
    result = {
        "station_name": str(row["police_station"]),
        "location": {"lat": round(float(row["centroid_lat"]), 4),
                      "lon": round(float(row["centroid_lon"]), 4)},
        "total_violations": int(float(row["total_violations"])),
        "unique_vehicles": int(float(row["unique_vehicles"])),
        "daily_rate": round(float(row["daily_rate"]), 1),
        "avg_pcis": round(float(row["avg_pcis"]), 3),
        "max_pcis": round(float(row["max_pcis"]), 3),
        "avg_severity": round(float(row["avg_severity"]), 3),
        "temporal_pattern": str(row.get("temporal_pattern", "N/A")),
        "peak_hour": str(row.get("peak_hour", "N/A")),
        "pct_junction": round(float(row["pct_junction"]) * 100, 1),
        "pct_main_road": round(float(row["pct_main_road"]) * 100, 1),
        "jurisdiction_radius_km": round(float(row["jurisdiction_radius_km"]), 2),
    }
    
    return json.dumps(result)


# ============================================
# GEMINI FUNCTION DECLARATIONS
# ============================================
TOOL_FUNCTIONS = {
    "get_top_hotspots": get_top_hotspots,
    "get_pcis_breakdown": get_pcis_breakdown,
    "get_temporal_pattern": get_temporal_pattern,
    "get_enforcement_routes": get_enforcement_routes,
    "compare_areas": compare_areas,
    "get_prediction_summary": get_prediction_summary,
    "get_station_profile": get_station_profile,
}


# ============================================
# GEMINI CLIENT SETUP
# ============================================
def create_agent():
    """Initialize Gemini model with function calling tools."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set. Add it to .env file.")
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Define tools for Gemini
    tools = [
        genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name="get_top_hotspots",
                    description="Get top N parking violation hotspots ranked by enforcement priority (CHR score). Can filter by police station and minimum PCIS score.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "n": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Number of top hotspots (default 10, max 50)"),
                            "police_station": genai.protos.Schema(type=genai.protos.Type.STRING, description="Filter by police station name (partial match)"),
                            "min_pcis": genai.protos.Schema(type=genai.protos.Type.NUMBER, description="Minimum PCIS score filter (0.0-1.0)"),
                        },
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="get_pcis_breakdown",
                    description="Get detailed PCIS (Parking Congestion Impact Score) breakdown for a specific H3 hexagon zone, including all 5 components and characteristics.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "h3_index": genai.protos.Schema(type=genai.protos.Type.STRING, description="H3 hexagonal index identifier"),
                        },
                        required=["h3_index"],
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="get_temporal_pattern",
                    description="Get temporal violation patterns (hourly, daily distributions) for a police station area or citywide ('city').",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "area": genai.protos.Schema(type=genai.protos.Type.STRING, description="Police station name or 'city' for citywide"),
                        },
                        required=["area"],
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="get_enforcement_routes",
                    description="Get optimized ACO patrol routes for a specific shift (morning/midday/afternoon/evening/night).",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "shift": genai.protos.Schema(type=genai.protos.Type.STRING, description="Shift: morning, midday, afternoon, evening, or night"),
                        },
                        required=["shift"],
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="compare_areas",
                    description="Compare parking violation statistics between two police station areas.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "area1": genai.protos.Schema(type=genai.protos.Type.STRING, description="First police station name"),
                            "area2": genai.protos.Schema(type=genai.protos.Type.STRING, description="Second police station name"),
                        },
                        required=["area1", "area2"],
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="get_prediction_summary",
                    description="Get XGBoost violation predictions for upcoming week or a specific day.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "day": genai.protos.Schema(type=genai.protos.Type.STRING, description="Day name (Monday, Tuesday, etc.) or omit for full week"),
                        },
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="get_station_profile",
                    description="Get detailed profile for a police station including workload, PCIS, temporal pattern, and jurisdiction info.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "station_name": genai.protos.Schema(type=genai.protos.Type.STRING, description="Police station name (partial match)"),
                        },
                        required=["station_name"],
                    ),
                ),
            ]
        )
    ]
    
    system_instruction = """You are PARKVISION AI, an intelligent parking enforcement assistant for Bengaluru Traffic Police (BTP).

You have access to a comprehensive analysis of 207,781 parking violations across Bengaluru, including:
- ST-DBSCAN spatial clustering at 3 scales (micro/meso/macro)
- Getis-Ord Gi* statistical hotspot validation
- PCIS (Parking Congestion Impact Score) with 5 components
- CongestionHoursRecovered (CHR) enforcement priority metric
- Location Memory scores and addiction zones
- XGBoost violation predictions
- ACO-optimized patrol routes across 5 shifts
- 54 police station jurisdiction profiles

When answering questions:
1. Use the function tools to query real data — never make up statistics
2. Provide specific numbers, ranks, and locations from the data
3. Give actionable enforcement recommendations based on the analysis
4. Explain PCIS components when discussing hotspot severity
5. Reference temporal patterns for shift-specific advice
6. Mention CHR scores when discussing enforcement ROI
7. Be concise but thorough — use tables and bullet points for clarity"""

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        tools=tools,
        system_instruction=system_instruction,
    )
    
    return model


# ============================================
# CONVERSATION HANDLER
# ============================================
def handle_query(model, chat, user_query: str) -> str:
    """Send query to Gemini, handle function calls, return final response."""
    response = chat.send_message(user_query)
    
    # Handle function calls iteratively
    max_rounds = 5
    for _ in range(max_rounds):
        # Check if model wants to call a function
        part = response.candidates[0].content.parts[0]
        
        if hasattr(part, "function_call") and part.function_call.name:
            fn_call = part.function_call
            fn_name = fn_call.name
            fn_args = dict(fn_call.args) if fn_call.args else {}
            
            logger.info(f"  Function call: {fn_name}({fn_args})")
            
            # Execute the function
            if fn_name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[fn_name](**fn_args)
            else:
                result = json.dumps({"error": f"Unknown function: {fn_name}"})
            
            # Send result back to model
            response = chat.send_message(
                genai.protos.Content(
                    parts=[genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=fn_name,
                            response={"result": result},
                        )
                    )]
                )
            )
        else:
            # No more function calls — return text response
            break
    
    return response.text


# ============================================
# DEMO MODE — run sample queries
# ============================================
DEMO_QUERIES = [
    "What are the top 5 most critical parking violation hotspots in Bengaluru and why are they dangerous?",
    "Compare the parking violation situation between Upparpet and Shivajinagar police station areas.",
    "What are the optimal patrol routes for the morning shift? Which stations should deploy units?",
    "Show me the temporal violation patterns for the city. When should enforcement be strongest?",
    "What does the AI predict for parking violations next week? Which areas will be worst?",
]


def run_demo(model):
    """Run demo queries and save conversation log."""
    import time
    logger.info("Running demo queries (with 65s delay between for rate limits) ...")
    
    demo_log = "# PARKVISION AI - LLM Demo Conversations\n\n"
    demo_log += "> Powered by Google Gemini 2.0 Flash with function calling tools\n"
    demo_log += "> Querying real processed data from 207,781 parking violations\n\n---\n\n"
    
    for i, query in enumerate(DEMO_QUERIES, 1):
        logger.info(f"\n  Demo Query {i}/{len(DEMO_QUERIES)}: {query[:60]}...")
        
        # Rate limit delay (skip for first query)
        if i > 1:
            logger.info(f"  Waiting 65s for rate limit cooldown ...")
            time.sleep(65)
        
        chat = model.start_chat()
        
        try:
            response = handle_query(model, chat, query)
            
            demo_log += f"## Query {i}\n\n"
            demo_log += f"**User:** {query}\n\n"
            demo_log += f"**PARKVISION AI:**\n\n{response}\n\n---\n\n"
            
            # Print to console
            print(f"\n{'='*70}")
            print(f"  QUERY {i}: {query}")
            print(f"{'='*70}")
            print(response)
            
        except Exception as e:
            logger.error(f"  Error on query {i}: {e}")
            demo_log += f"## Query {i}\n\n"
            demo_log += f"**User:** {query}\n\n"
            demo_log += f"**Error:** {str(e)}\n\n---\n\n"
    
    # Save demo log
    demo_path = OUTPUT_DIR / "llm_demo_conversations.md"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(demo_path, "w", encoding="utf-8") as f:
        f.write(demo_log)
    
    logger.info(f"\n  Demo conversations saved to {demo_path}")
    return demo_log


# ============================================
# INTERACTIVE CHAT MODE
# ============================================
def run_interactive(model):
    """Run interactive chat session."""
    print("\n" + "=" * 70)
    print("  PARKVISION AI - Parking Intelligence Chat")
    print("  Type your question about Bengaluru parking violations.")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 70 + "\n")
    
    chat = model.start_chat()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break
        
        try:
            response = handle_query(model, chat, user_input)
            print(f"\nPARKVISION AI: {response}")
        except Exception as e:
            print(f"\nError: {e}")
            # Reset chat on error
            chat = model.start_chat()


# ============================================
# MAIN
# ============================================
def run():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PARKVISION AI Chat Agent")
    parser.add_argument("--demo", action="store_true", help="Run demo queries")
    args = parser.parse_args()
    
    logger.info("Initializing Gemini agent ...")
    model = create_agent()
    logger.info("  Gemini agent ready")
    
    if args.demo:
        run_demo(model)
    else:
        run_interactive(model)


if __name__ == "__main__":
    run()
