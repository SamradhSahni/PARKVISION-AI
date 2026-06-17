"""
PARKVISION AI — Stage 7: Congestion Propagation & Location Memory
====================================================================
Models the ripple effect of parking violations on surrounding traffic,
computes location memory scores, and detects cross-jurisdiction spillover.

Usage:
    python -m src.congestion_model
"""

import sys
import logging
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from scipy.stats import pearsonr

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR,
    OUTPUT_DIR,
    PCIS_SCORED_PARQUET,
    H3_HEX_STATS_PARQUET,
    H3_HOTSPOT_SIG_PARQUET,
    ENRICHED_PARQUET,
    RIPPLE_CONTOURS_GEOJSON,
    LOCATION_MEMORY_PARQUET,
    SPILLOVER_PARQUET,
    STANDARD_LANE_WIDTH_M,
    LANE_CAPACITY_VPH,
    LOG_FORMAT,
    LOG_LEVEL,
)

warnings.filterwarnings("ignore")
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger("congestion_model")

# Shockwave model constants
FREE_FLOW_SPEED_KPH = 40  # typical urban free-flow speed
JAM_DENSITY_VPM = 150     # vehicles per km at jam density
WAVE_SPEED_KPH = 15       # backward shockwave propagation speed
RIPPLE_RADII_M = [50, 150, 300, 500]
SPEED_DEGRADATION = {50: 0.80, 150: 0.50, 300: 0.25, 500: 0.10}  # fraction of speed lost


# ============================================
# STEP 1: Shockwave propagation model
# ============================================
def compute_shockwave_propagation(hex_pcis: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    For each high-PCIS hexagon, estimate congestion ripple effect.
    
    Uses simplified Lighthill-Whitham-Richards (LWR) shockwave model:
    - Queue length = capacity_reduction * road_length / (1 - density_ratio)
    - Spillback distance = queue_length + wave_propagation_distance
    - Speed degradation decays with distance from violation
    """
    logger.info("Computing shockwave propagation for high-PCIS hexagons ...")

    # Filter to high/critical PCIS hexagons
    high_pcis = hex_pcis[hex_pcis["pcis_mean"] >= 0.4].copy()
    logger.info(f"  {len(high_pcis):,} hexagons with PCIS >= 0.4")

    ripple_features = []

    for _, row in high_pcis.iterrows():
        h3_idx = row["h3_index"]
        pcis = row["pcis_mean"]
        pcis_max = row["pcis_max"]
        cr = row["capacity_reduction_mean"]
        count = row["violation_count"]
        centroid = Point(row["centroid_lon"], row["centroid_lat"])

        # --- LWR Shockwave Estimation ---
        # Effective capacity remaining
        capacity_remaining_ratio = max(0.05, 1.0 - cr)

        # Queue buildup rate (vehicles/hour that can't pass)
        excess_demand = LANE_CAPACITY_VPH * cr  # demand exceeding remaining capacity

        # Estimated queue length (meters) during peak hour
        # Queue = excess_demand * avg_vehicle_length / jam_density
        queue_length_m = (excess_demand / JAM_DENSITY_VPM) * 1000  # convert to meters

        # Shockwave propagation speed (km/h, backward)
        wave_speed = WAVE_SPEED_KPH * cr  # stronger blockage = faster wave

        # Spillback distance over 15-minute window
        spillback_m = min(queue_length_m + wave_speed * 1000 * 0.25, 2000)  # cap at 2km

        # Delay per vehicle (seconds)
        delay_per_vehicle_s = (cr * 60) / capacity_remaining_ratio  # simplified

        # Total delay impact (vehicle-hours per day)
        daily_violations = count / max(1, row.get("unique_days", 30))
        total_delay_veh_hours = (delay_per_vehicle_s * excess_demand * daily_violations) / 3600

        # --- Create ripple contour polygons ---
        for radius_m in RIPPLE_RADII_M:
            # Convert meters to approximate degrees
            radius_deg = radius_m / 111000

            # Create circular buffer
            buffer_poly = centroid.buffer(radius_deg)

            # Speed degradation at this ring
            speed_loss_pct = SPEED_DEGRADATION.get(radius_m, 0.05) * cr
            remaining_speed_pct = max(0.1, 1.0 - speed_loss_pct)
            estimated_speed_kph = FREE_FLOW_SPEED_KPH * remaining_speed_pct

            ripple_features.append({
                "h3_index": h3_idx,
                "pcis_mean": round(pcis, 3),
                "pcis_max": round(pcis_max, 3),
                "capacity_reduction": round(cr, 3),
                "violation_count": int(count),
                "ripple_radius_m": radius_m,
                "queue_length_m": round(queue_length_m, 1),
                "spillback_m": round(spillback_m, 1),
                "speed_degradation_pct": round(speed_loss_pct * 100, 1),
                "estimated_speed_kph": round(estimated_speed_kph, 1),
                "delay_per_vehicle_s": round(delay_per_vehicle_s, 1),
                "total_delay_veh_hours": round(total_delay_veh_hours, 1),
                "centroid_lat": row["centroid_lat"],
                "centroid_lon": row["centroid_lon"],
                "geometry": buffer_poly,
            })

    ripple_gdf = gpd.GeoDataFrame(ripple_features, crs="EPSG:4326")
    logger.info(f"  Created {len(ripple_gdf):,} ripple contour features")
    logger.info(f"  Mean queue length: {ripple_gdf.groupby('h3_index')['queue_length_m'].first().mean():.0f}m")
    logger.info(f"  Mean spillback: {ripple_gdf.groupby('h3_index')['spillback_m'].first().mean():.0f}m")

    return ripple_gdf


# ============================================
# STEP 2: Location Memory Score
# ============================================
def compute_location_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute LocationMemoryScore per H3 hexagon.
    
    LocationMemory = persistence_ratio * 0.6 + recurrence_score * 0.4
    
    - persistence_ratio = unique_active_days / total_observation_days
    - recurrence_score = repeat_vehicle_fraction (same vehicles returning)
    """
    logger.info("Computing Location Memory Scores ...")

    # Total observation period
    date_range = df["date"].nunique()
    total_days = max(date_range, 1)
    logger.info(f"  Observation period: {total_days} unique days")

    # Group by H3 hex
    hex_groups = df.groupby("h3_index")

    memory = pd.DataFrame()

    # Persistence: fraction of total days with at least one violation
    memory["active_days"] = hex_groups["date"].nunique()
    memory["persistence_ratio"] = (memory["active_days"] / total_days).clip(0, 1)

    # Recurrence: fraction of vehicles that appear more than once at this location
    def _repeat_vehicle_fraction(group):
        vehicle_counts = group["vehicle_number"].value_counts()
        if len(vehicle_counts) == 0:
            return 0.0
        repeats = (vehicle_counts > 1).sum()
        return repeats / len(vehicle_counts)

    memory["repeat_vehicle_fraction"] = hex_groups.apply(_repeat_vehicle_fraction)

    # Composite Location Memory Score
    memory["location_memory_score"] = (
        0.6 * memory["persistence_ratio"]
        + 0.4 * memory["repeat_vehicle_fraction"]
    ).clip(0, 1)

    # Flag "addiction zones"
    memory["is_addiction_zone"] = memory["location_memory_score"] > 0.5

    # Additional metrics
    memory["violation_count"] = hex_groups.size()
    memory["unique_vehicles"] = hex_groups["vehicle_number"].nunique()
    memory["avg_pcis"] = hex_groups["pcis"].mean() if "pcis" in df.columns else 0.0

    # Centroid for spatial reference
    memory["centroid_lat"] = hex_groups["latitude"].mean()
    memory["centroid_lon"] = hex_groups["longitude"].mean()
    memory["primary_police_station"] = hex_groups["police_station"].agg(
        lambda x: x.value_counts().index[0]
    )

    memory = memory.reset_index()

    n_addiction = memory["is_addiction_zone"].sum()
    addiction_violations = memory[memory["is_addiction_zone"]]["violation_count"].sum()
    logger.info(
        f"  Location memory computed for {len(memory):,} hexagons"
    )
    logger.info(
        f"  Addiction zones (memory > 0.5): {n_addiction:,} hexagons "
        f"({n_addiction/len(memory)*100:.1f}%), "
        f"{addiction_violations:,} violations"
    )
    logger.info(
        f"  Mean memory score: {memory['location_memory_score'].mean():.3f}"
    )

    return memory


# ============================================
# STEP 3: Cross-jurisdiction spillover analysis
# ============================================
def compute_spillover_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect cross-jurisdiction spillover: if enforcement in one police station
    area causes violations to shift to adjacent stations.
    
    Method: compute monthly violation trends per station and find
    correlated increases between adjacent stations.
    """
    logger.info("Computing cross-jurisdiction spillover analysis ...")

    # Monthly violation counts per police station
    monthly = df.groupby(["police_station", "month"]).size().reset_index(name="violation_count")
    pivot = monthly.pivot(index="month", columns="police_station", values="violation_count").fillna(0)

    stations = pivot.columns.tolist()
    logger.info(f"  Analyzing {len(stations)} police stations across {len(pivot)} months")

    # Compute station centroids for adjacency detection
    station_centroids = df.groupby("police_station").agg(
        centroid_lat=("latitude", "mean"),
        centroid_lon=("longitude", "mean"),
        total_violations=("latitude", "size"),
    ).reset_index()

    # Compute pairwise correlations and adjacency
    spillover_records = []
    adjacency_threshold_km = 5.0  # stations within 5km are "adjacent"

    for i, s1 in enumerate(stations):
        for j, s2 in enumerate(stations):
            if i >= j:
                continue

            # Check adjacency (distance between station centroids)
            c1 = station_centroids[station_centroids["police_station"] == s1]
            c2 = station_centroids[station_centroids["police_station"] == s2]

            if len(c1) == 0 or len(c2) == 0:
                continue

            dist_km = _haversine_km(
                c1.iloc[0]["centroid_lat"], c1.iloc[0]["centroid_lon"],
                c2.iloc[0]["centroid_lat"], c2.iloc[0]["centroid_lon"],
            )

            is_adjacent = dist_km <= adjacency_threshold_km

            # Compute correlation of monthly trends
            series1 = pivot[s1].values
            series2 = pivot[s2].values

            if len(series1) < 3:
                continue

            try:
                corr, p_val = pearsonr(series1, series2)
            except Exception:
                corr, p_val = 0.0, 1.0

            # Compute month-over-month changes
            changes1 = np.diff(series1)
            changes2 = np.diff(series2)

            # Inverse correlation suggests spillover (one goes up, other goes down)
            # or strong positive correlation suggests shared demand
            spillover_type = "none"
            if p_val < 0.1 and is_adjacent:
                if corr < -0.5:
                    spillover_type = "displacement"  # enforcement pushes violations next door
                elif corr > 0.7:
                    spillover_type = "shared_demand"  # same underlying demand
                elif corr > 0.3:
                    spillover_type = "weak_correlation"

            spillover_records.append({
                "station_1": s1,
                "station_2": s2,
                "distance_km": round(dist_km, 2),
                "is_adjacent": is_adjacent,
                "correlation": round(corr, 3),
                "p_value": round(p_val, 4),
                "spillover_type": spillover_type,
                "s1_total": int(series1.sum()),
                "s2_total": int(series2.sum()),
                "s1_trend": round(float(np.polyfit(range(len(series1)), series1, 1)[0]), 1),
                "s2_trend": round(float(np.polyfit(range(len(series2)), series2, 1)[0]), 1),
            })

    spillover_df = pd.DataFrame(spillover_records)

    # Log findings
    n_adjacent = spillover_df["is_adjacent"].sum()
    n_displacement = (spillover_df["spillover_type"] == "displacement").sum()
    n_shared = (spillover_df["spillover_type"] == "shared_demand").sum()

    logger.info(f"  Total station pairs analyzed: {len(spillover_df):,}")
    logger.info(f"  Adjacent pairs (< {adjacency_threshold_km}km): {n_adjacent:,}")
    logger.info(f"  Displacement spillover pairs: {n_displacement:,}")
    logger.info(f"  Shared demand pairs: {n_shared:,}")

    return spillover_df


def _haversine_km(lat1, lon1, lat2, lon2):
    """Compute haversine distance in kilometers."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


# ============================================
# STEP 4: Save & Summarize
# ============================================
def save_and_summarize(ripple_gdf, memory_df, spillover_df, hex_pcis):
    """Save outputs and print summary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save ripple contours
    logger.info(f"Saving ripple contours to {RIPPLE_CONTOURS_GEOJSON} ...")
    ripple_gdf.to_file(RIPPLE_CONTOURS_GEOJSON, driver="GeoJSON")

    # Save location memory
    logger.info(f"Saving location memory to {LOCATION_MEMORY_PARQUET} ...")
    memory_df.to_parquet(LOCATION_MEMORY_PARQUET, index=False, engine="pyarrow")

    # Save spillover analysis
    logger.info(f"Saving spillover analysis to {SPILLOVER_PARQUET} ...")
    spillover_df.to_parquet(SPILLOVER_PARQUET, index=False, engine="pyarrow")

    print("\n" + "=" * 70)
    print("  PARKVISION AI - Congestion Propagation & Location Memory")
    print("=" * 70)

    # Shockwave summary
    hex_ripple = ripple_gdf.groupby("h3_index").first()
    print(f"\n  --- Shockwave Propagation Model ---")
    print(f"  Hotspot hexagons modeled:   {len(hex_ripple):,}")
    print(f"  Ripple contour features:    {len(ripple_gdf):,}")
    print(f"  Avg queue length:           {hex_ripple['queue_length_m'].mean():.0f}m")
    print(f"  Max queue length:           {hex_ripple['queue_length_m'].max():.0f}m")
    print(f"  Avg spillback distance:     {hex_ripple['spillback_m'].mean():.0f}m")
    print(f"  Total delay impact:         {hex_ripple['total_delay_veh_hours'].sum():,.0f} veh-hrs/day")

    # Top 10 worst ripple zones
    print(f"\n  --- Top 10 Worst Congestion Ripple Zones ---")
    top = hex_ripple.nlargest(10, "spillback_m")
    print(f"    {'H3 Index':>17}  {'PCIS':>5}  {'CR':>5}  {'Queue':>6}  {'Spill':>6}  {'Delay':>8}")
    for idx, row in top.iterrows():
        print(f"    {idx:>17}  {row['pcis_mean']:>5.3f}  {row['capacity_reduction']:>5.3f}  "
              f"{row['queue_length_m']:>6.0f}m  {row['spillback_m']:>5.0f}m  "
              f"{row['total_delay_veh_hours']:>7.0f}vh")

    # Location Memory summary
    print(f"\n  --- Location Memory Scores ---")
    print(f"  Total hexagons scored:     {len(memory_df):,}")
    print(f"  Mean memory score:         {memory_df['location_memory_score'].mean():.3f}")
    n_addiction = memory_df["is_addiction_zone"].sum()
    addiction_v = memory_df[memory_df["is_addiction_zone"]]["violation_count"].sum()
    print(f"  Addiction zones (>0.5):    {n_addiction:,} ({n_addiction/len(memory_df)*100:.1f}%)")
    print(f"  Violations in addictions:  {addiction_v:,}")
    print(f"  Mean persistence ratio:    {memory_df['persistence_ratio'].mean():.3f}")
    print(f"  Mean repeat vehicle rate:  {memory_df['repeat_vehicle_fraction'].mean():.3f}")

    # Top addiction zones
    print(f"\n  --- Top 10 Addiction Zones (highest memory score) ---")
    top_mem = memory_df.nlargest(10, "location_memory_score")
    for _, row in top_mem.iterrows():
        print(f"    {row['h3_index']:>17}  "
              f"memory={row['location_memory_score']:.3f}  "
              f"persist={row['persistence_ratio']:.3f}  "
              f"repeat={row['repeat_vehicle_fraction']:.3f}  "
              f"count={row['violation_count']:>5}  "
              f"station={row['primary_police_station']}")

    # Spillover summary
    print(f"\n  --- Cross-Jurisdiction Spillover Analysis ---")
    print(f"  Station pairs analyzed:    {len(spillover_df):,}")
    adj = spillover_df[spillover_df["is_adjacent"]]
    print(f"  Adjacent pairs (<5km):     {len(adj):,}")
    for stype in ["displacement", "shared_demand", "weak_correlation"]:
        n = (spillover_df["spillover_type"] == stype).sum()
        if n > 0:
            print(f"  {stype:25s}: {n:>3} pairs")

    # Show displacement pairs
    disp = spillover_df[spillover_df["spillover_type"] == "displacement"]
    if len(disp) > 0:
        print(f"\n  --- Displacement Spillover Pairs (negative correlation) ---")
        for _, row in disp.head(10).iterrows():
            print(f"    {row['station_1']:>20s} <-> {row['station_2']:<20s}  "
                  f"r={row['correlation']:>6.3f}  dist={row['distance_km']:>4.1f}km")

    print(f"\n  Outputs:")
    print(f"    {RIPPLE_CONTOURS_GEOJSON}")
    print(f"    {LOCATION_MEMORY_PARQUET}")
    print(f"    {SPILLOVER_PARQUET}")
    print("=" * 70)


# ============================================
# MAIN PIPELINE
# ============================================
def run():
    """Execute congestion propagation and location memory pipeline."""
    # Load hex-level PCIS scores
    hex_pcis_path = DATA_DIR / "h3_pcis_scores.parquet"
    logger.info(f"Loading hex PCIS scores from {hex_pcis_path} ...")
    hex_pcis = pd.read_parquet(hex_pcis_path)

    # Load hex stats for centroid info
    hex_stats = pd.read_parquet(H3_HOTSPOT_SIG_PARQUET)
    hex_pcis = hex_pcis.merge(
        hex_stats[["h3_index", "centroid_lat", "centroid_lon", "unique_days"]],
        on="h3_index", how="left",
    )
    logger.info(f"  Loaded {len(hex_pcis):,} hexagons")

    # Load PCIS-scored violations
    logger.info(f"Loading PCIS-scored violations from {PCIS_SCORED_PARQUET} ...")
    df = pd.read_parquet(PCIS_SCORED_PARQUET)
    logger.info(f"  Loaded {len(df):,} violations")

    # Step 1: Shockwave propagation
    ripple_gdf = compute_shockwave_propagation(hex_pcis)

    # Step 2: Location memory scores
    memory_df = compute_location_memory(df)

    # Step 3: Spillover analysis
    spillover_df = compute_spillover_analysis(df)

    # Step 4: Save & summarize
    save_and_summarize(ripple_gdf, memory_df, spillover_df, hex_pcis)

    return ripple_gdf, memory_df, spillover_df


if __name__ == "__main__":
    run()
