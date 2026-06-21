"""
PARKVISION AI — LLM Agent (NVIDIA NIM Backend)
================================================
Conversational AI using NVIDIA NIM API (OpenAI-compatible).
Uses context injection: fetches live data from processed files and
embeds it directly into the system prompt for rich, accurate answers.

Usage:
    python -m src.llm_agent                # Interactive mode
    python -m src.llm_agent --demo         # Demo queries
"""

import sys
import json
import logging
import warnings
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR,
    OUTPUT_DIR,
    NVIDIA_API_KEY,
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
# NVIDIA NIM CONFIG
# ============================================
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "meta/llama-3.3-70b-instruct"   # High quota, tool-capable

# ============================================
# DATA LOADERS (lazy, cached)
# ============================================
_cache: Dict = {}

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
    return _cache.get(key)

def _violations():
    return _load("violations", PCIS_SCORED_PARQUET)

def _priorities():
    return _load("priorities", ENFORCEMENT_PRIORITIES_PARQUET)

def _predicted():
    return _load("predicted", PREDICTED_VIOLATIONS_PARQUET)

def _hotspot_sig():
    return _load("hotspot_sig", H3_HOTSPOT_SIG_PARQUET)

# ============================================
# CONTEXT BUILDER — injects live data into prompt
# ============================================
def build_context(station: Optional[str] = None) -> str:
    """
    Build a rich data context string from processed parquet files.
    When station is set, scope all stats to that jurisdiction only.
    """
    sections = []
    scope_label = station or "CITYWIDE"

    # ── 1. Top hotspots ─────────────────────────────────────────────
    try:
        df = _violations()
        if df is not None and station:
            df = df[df["police_station"] == station]
        if df is not None and "pcis_mean" in df.columns:
            if station:
                agg = df.groupby("h3_index").agg(
                    violation_count=("pcis_mean", "count"),
                    pcis_mean=("pcis_mean", "mean"),
                ).reset_index().sort_values("pcis_mean", ascending=False).head(10)
                rows = []
                for _, r in agg.iterrows():
                    rows.append(f"  - Hex {r['h3_index']}: {int(r['violation_count'])} violations, PCIS={r['pcis_mean']:.3f}")
                sections.append(f"TOP 10 HOTSPOT HEXAGONS IN {station.upper()}:\n" + "\n".join(rows))
            else:
                agg = df.groupby("police_station").agg(
                    violation_count=("pcis_mean", "count"),
                    pcis_mean=("pcis_mean", "mean"),
                    avg_chr=("chr", "mean") if "chr" in df.columns else ("pcis_mean", "count"),
                ).reset_index().sort_values("pcis_mean", ascending=False).head(15)
                rows = []
                for _, r in agg.iterrows():
                    rows.append(f"  - {r['police_station']}: {int(r['violation_count'])} violations, PCIS={r['pcis_mean']:.3f}")
                sections.append("TOP 15 STATIONS BY PCIS SCORE:\n" + "\n".join(rows))
    except Exception as e:
        logger.debug(f"Hotspot context error: {e}")

    # ── 2. Summary stats ────────────────────────────────────────────
    try:
        df = _violations()
        if df is not None and station:
            df = df[df["police_station"] == station]
        if df is not None:
            total = len(df)
            stations = df["police_station"].nunique() if "police_station" in df.columns else "N/A"
            avg_pcis = df["pcis_mean"].mean() if "pcis_mean" in df.columns else "N/A"
            peak_hour = df["hour"].mode()[0] if "hour" in df.columns else "N/A"
            top_vehicle = df["vehicle_type"].mode()[0] if "vehicle_type" in df.columns else "N/A"
            sections.append(
                f"{scope_label} SUMMARY:\n"
                f"  - Total validated violations: {total:,}\n"
                f"  - Unique police stations: {stations}\n"
                f"  - Average PCIS score: {avg_pcis:.4f}\n"
                f"  - Most common peak hour: {peak_hour}:00\n"
                f"  - Most common vehicle type: {top_vehicle}"
            )
    except Exception as e:
        logger.debug(f"Summary context error: {e}")

    # ── 3. Priority tiers ───────────────────────────────────────────
    try:
        df = _violations()
        if df is not None and station:
            df = df[df["police_station"] == station]
        if df is not None and "priority_tier" in df.columns:
            tiers = df["priority_tier"].value_counts()
            sections.append(f"PRIORITY TIER DISTRIBUTION ({scope_label}):\n" +
                "\n".join(f"  - {t}: {c}" for t, c in tiers.items()))
    except Exception as e:
        logger.debug(f"Tier context error: {e}")

    # ── 4. Temporal pattern ─────────────────────────────────────────
    try:
        df = _violations()
        if df is not None and station:
            df = df[df["police_station"] == station]
        if df is not None and "hour" in df.columns:
            hourly = df.groupby("hour").size()
            top3 = hourly.nlargest(3)
            bottom3 = hourly.nsmallest(3)
            sections.append(
                "HOURLY PATTERNS:\n"
                f"  - Peak hours: {', '.join([f'{h}:00 ({c:,} violations)' for h, c in top3.items()])}\n"
                f"  - Quietest hours: {', '.join([f'{h}:00 ({c:,} violations)' for h, c in bottom3.items()])}\n"
                f"  - Notable: 4-7 AM shows unusually high violations (likely early enforcement sweeps)"
            )
    except Exception as e:
        logger.debug(f"Temporal context error: {e}")

    # ── 5. Vehicle types ────────────────────────────────────────────
    try:
        df = _violations()
        if df is not None and station:
            df = df[df["police_station"] == station]
        if df is not None and "vehicle_type" in df.columns:
            vc = df["vehicle_type"].value_counts().head(8)
            sections.append("VEHICLE TYPE BREAKDOWN:\n" +
                "\n".join(f"  - {v}: {c:,} ({c/len(df)*100:.1f}%)" for v, c in vc.items()))
    except Exception as e:
        logger.debug(f"Vehicle context error: {e}")

    # ── 6. Predictions ──────────────────────────────────────────────
    try:
        pred_df = _predicted()
        if pred_df is not None:
            top_days = pred_df.groupby("pred_dow_name")["total"].sum().nlargest(3) if "pred_dow_name" in pred_df.columns else None
            if top_days is not None:
                sections.append("NEXT WEEK PREDICTIONS (XGBoost):\n" +
                    "\n".join(f"  - {d}: {int(c):,} expected violations" for d, c in top_days.items()))
    except Exception as e:
        logger.debug(f"Prediction context error: {e}")

    # ── 7. Day of week breakdown ────────────────────────────────────
    try:
        df = _violations()
        if df is not None and station:
            df = df[df["police_station"] == station]
        if df is not None and "day_of_week" in df.columns:
            dow = df.groupby("day_of_week").size().sort_values(ascending=False)
            sections.append("DAY OF WEEK PATTERN:\n" +
                "\n".join(f"  - {d}: {c:,} violations" for d, c in dow.items()))
    except Exception as e:
        logger.debug(f"DOW context error: {e}")

    # ── 8. Enforcement priorities top 10 ───────────────────────────
    try:
        pf = _priorities()
        if pf is not None and station:
            pf = pf[pf["police_station"] == station]
        if pf is not None:
            sort_col = "chr" if "chr" in pf.columns else pf.columns[0]
            top10 = pf.sort_values(sort_col, ascending=False).head(10)
            rows = []
            for _, r in top10.iterrows():
                station = r.get("police_station", "?")
                score = r.get("chr", r.get("pcis_mean", "?"))
                rows.append(f"  - {station}: CHR={score:.1f}" if isinstance(score, float) else f"  - {station}")
            sections.append("TOP 10 ENFORCEMENT PRIORITIES (by CHR):\n" + "\n".join(rows))
    except Exception as e:
        logger.debug(f"Priorities context error: {e}")

    return "\n\n".join(sections) if sections else "Data context temporarily unavailable."


# ============================================
# CITY-WIDE QUERY GUARD (for station-role users)
# ============================================

# Keywords that signal the user is asking about city-wide or cross-station data
_CITY_WIDE_KEYWORDS = [
    "all station", "all stations", "every station", "each station",
    "entire city", "whole city", "city wide", "city-wide", "citywide",
    "all of bengaluru", "all bengaluru", "bangalore city", "bengaluru city",
    "across bangalore", "across bengaluru", "overall city", "city total",
    "compare station", "compare all", "top station", "top stations",
    "worst station", "best station", "all jurisdiction", "all jurisdictions",
    "stationwise", "station wise", "other station",
    "in bengaluru", "in bangalore", "in blr",
    "of bengaluru", "of bangalore",
    "bengaluru hotspot", "bangalore hotspot",
    "bengaluru violation", "bangalore violation",
    "bengaluru traffic", "bangalore traffic",
]

# City name tokens — if these appear WITHOUT the user's station name,
# treat the query as city-wide
_CITY_NAME_TOKENS = ["bengaluru", "bangalore", "blr", "namma bengaluru"]

def _is_city_wide_query(query: str, user_station: str) -> bool:
    """
    Return True if a station-role user's query is asking about
    city-wide data, other stations, or cross-station comparisons.
    """
    q = query.lower().strip()
    my_station_lower = (user_station or "").lower()

    # Check generic city-wide keyword phrases
    for kw in _CITY_WIDE_KEYWORDS:
        if kw in q:
            return True

    # Check if the query references a city name without the user's own station name.
    # e.g. "hotspots in Bengaluru"  → bengaluru present, "adugodi" not present → BLOCK
    # e.g. "hotspots in Adugodi, Bengaluru" → both present → ALLOW (station context)
    for city_token in _CITY_NAME_TOKENS:
        if city_token in q:
            if my_station_lower and my_station_lower in q:
                continue  # they mentioned their own station alongside the city — allow
            return True   # bare city reference without their station → block

    # Check if they reference "station" in a comparative/inquiry context
    # without mentioning their own station name
    if "station" in q and my_station_lower and my_station_lower not in q:
        compare_words = ["which", "what", "where", "highest", "lowest",
                         "best", "worst", "top", "most", "least", "compare",
                         "list", "show", "all", "give"]
        if any(cw in q for cw in compare_words):
            return True

    return False




# ============================================
# SYSTEM PROMPT
# ============================================
SYSTEM_PROMPT = """You are PARKVISION AI, the intelligent parking enforcement assistant for Bengaluru Traffic Police (BTP).

You have deep expertise in:
- ST-DBSCAN spatial clustering at micro (50m), meso (150m), macro (500m) scales
- Getis-Ord Gi* statistical hotspot significance testing
- PCIS (Parking Congestion Impact Score) with 5 components:
  1. Capacity Reduction Factor (30%) — based on vehicle footprint vs lane width
  2. Proximity Factor (20%) — junction distance decay
  3. Temporal Demand Multiplier (20%) — hour/day demand curves
  4. Vehicle Obstruction Factor (15%) — movement hindrance per vehicle type
  5. Network Criticality (15%) — betweenness centrality of the road
- CHR (Congestion Hours Recovered) — hours of congestion saved per day if violations are stopped (max 24 hrs)
- H3 hexagonal indexing at resolution 9 (~174m edge length)
- XGBoost violation count predictions (next 7 days)
- ACO (Ant Colony Optimization) patrol routing across 5 shifts
- 54 police station jurisdiction profiles in Bengaluru

When answering:
1. Use the live data context below — never make up statistics
2. Give specific numbers, station names, and recommendations
3. Explain PCIS components when discussing hotspot severity
4. Reference CHR when discussing enforcement ROI
5. Suggest specific patrol shifts when relevant
6. Acknowledge data limitations honestly (e.g., GPS accuracy, enforcement bias)
7. Format responses clearly with bullet points and bold for key numbers
8. Be direct and actionable — this is for active enforcement decisions

LIVE DATA FROM PARKVISION ANALYSIS:
{context}
"""


# ============================================
# NVIDIA NIM AGENT
# ============================================
def create_agent():
    """
    Initialize the NVIDIA NIM client.
    Returns an openai.OpenAI client configured for NVIDIA NIM.
    Falls back to Gemini if NVIDIA key not available.
    """
    if NVIDIA_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=NVIDIA_BASE_URL,
                api_key=NVIDIA_API_KEY,
            )
            logger.info(f"NVIDIA NIM agent initialized (model: {NVIDIA_MODEL})")
            return {"type": "nvidia", "client": client}
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            raise

    elif GEMINI_API_KEY:
        logger.warning("NVIDIA_API_KEY not set, falling back to Gemini")
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT.format(context=build_context()),
        )
        return {"type": "gemini", "model": model}

    else:
        raise ValueError("No AI API key configured. Set NVIDIA_API_KEY or GEMINI_API_KEY in .env")


def handle_query(agent: dict, chat, user_query: str, station: Optional[str] = None) -> str:
    """
    Send a user query to the configured LLM and return the response.

    Args:
        agent: dict with 'type' and client/model
        chat: chat history list (for NVIDIA) or Gemini chat session
        user_query: the user's question
        station: station name if user is a station-role officer, else None

    Returns:
        String response from the LLM.
    """
    # ── Access guard for station-role users ─────────────────────────
    if station and _is_city_wide_query(user_query, station):
        restriction_msg = (
            f"🔒 **Access Restricted** — You are logged in as **{station}** station.\n\n"
            f"Your account is authorized to view data and answer queries related to "
            f"**{station} jurisdiction only**. City-wide statistics, comparisons across "
            f"multiple stations, or data from other police stations are not accessible "
            f"from your login.\n\n"
            f"If you need city-level insights, please contact your **City Command (Admin)** supervisor."
        )
        # Still add to history so the conversation flows naturally
        if isinstance(chat, list):
            chat.append({"role": "user", "content": user_query})
            chat.append({"role": "assistant", "content": restriction_msg})
        return restriction_msg

    if agent["type"] == "nvidia":
        return _handle_nvidia(agent["client"], chat, user_query, station)
    elif agent["type"] == "gemini":
        return _handle_gemini(agent["model"], chat, user_query)
    else:
        return "Error: Unknown agent type."


def _handle_nvidia(client, history: list, user_query: str, station: Optional[str] = None) -> str:
    """NVIDIA NIM handler using OpenAI-compatible API with context injection."""
    context = build_context(station=station)
    system_msg = SYSTEM_PROMPT.format(context=context)
    if station:
        system_msg += f"\n\nIMPORTANT: You are assisting officers at **{station}** station only. Do not discuss or compare other stations unless asked about city-wide methodology."

    messages = [{"role": "system", "content": system_msg}]
    # Add conversation history (last 10 turns to stay within context window)
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages,
        temperature=0.6,
        top_p=0.95,
        max_tokens=1024,
        stream=False,
    )

    answer = response.choices[0].message.content

    # Update history in-place
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": answer})

    return answer


def _handle_gemini(model, chat, user_query: str) -> str:
    """Gemini fallback handler."""
    response = chat.send_message(user_query)
    return response.text


# ============================================
# TOOL_FUNCTIONS (kept for backward compat)
# ============================================
def get_top_hotspots(n: int = 10, police_station: Optional[str] = None, min_pcis: float = 0.0) -> str:
    """Get top N parking violation hotspots."""
    df = _violations()
    if df is None:
        return json.dumps({"error": "Data not available"})
    agg = df.groupby("police_station").agg(
        violation_count=("pcis_mean", "count"),
        pcis_mean=("pcis_mean", "mean"),
    ).reset_index()
    if police_station:
        agg = agg[agg["police_station"].str.contains(police_station, case=False, na=False)]
    agg = agg[agg["pcis_mean"] >= min_pcis]
    agg = agg.sort_values("pcis_mean", ascending=False).head(n)
    return agg.to_json(orient="records")


def get_temporal_pattern(area: str = "city") -> str:
    """Get hourly/daily violation pattern for a station or citywide."""
    df = _violations()
    if df is None:
        return json.dumps({"error": "Data not available"})
    if area.lower() != "city":
        df = df[df["police_station"].str.contains(area, case=False, na=False)]
    hourly = df.groupby("hour").size().to_dict() if "hour" in df.columns else {}
    daily = df.groupby("day_of_week").size().to_dict() if "day_of_week" in df.columns else {}
    return json.dumps({"area": area, "hourly": hourly, "daily": daily})


TOOL_FUNCTIONS = {
    "get_top_hotspots": get_top_hotspots,
    "get_temporal_pattern": get_temporal_pattern,
}


# ============================================
# INTERACTIVE / DEMO MODES
# ============================================
def run_interactive(agent):
    """Interactive terminal chat."""
    logger.info("Starting interactive chat. Type 'quit' to exit.")
    history = []
    print("\nPARKVISION AI (NVIDIA NIM) — Interactive Mode")
    print("=" * 50)
    print("Type your question or 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break
        response = handle_query(agent, history, user_input)
        print(f"\nPARKVISION AI:\n{response}\n{'─'*50}\n")


def run_demo(agent):
    """Run demo queries."""
    demo_queries = [
        "Which police station has the highest PCIS score and what does that mean?",
        "Why is there a spike in violations at 4-7 AM?",
        "What are the top 3 vehicle types by violation count?",
        "Which day of the week has the most violations?",
    ]
    history = []
    results = []
    for q in demo_queries:
        print(f"\nQ: {q}")
        response = handle_query(agent, history, q)
        print(f"A: {response[:500]}...\n")
        results.append({"query": q, "response": response})
    return results


def run():
    parser = argparse.ArgumentParser(description="PARKVISION AI LLM Agent")
    parser.add_argument("--demo", action="store_true", help="Run demo queries")
    args = parser.parse_args()

    agent = create_agent()
    history = []

    if args.demo:
        run_demo(agent)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    run()
