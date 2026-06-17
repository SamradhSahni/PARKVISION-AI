"""
PARKVISION AI — Stage 9: Police Station Mapping & Enforcement ROI
====================================================================
Geolocates police stations from violation data, computes jurisdiction
profiles, and ranks hotspots by CongestionHoursRecovered (CHR) metric.

Usage:
    python -m src.enforcement_optimizer
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union
from scipy.spatial import ConvexHull

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR,
    OUTPUT_DIR,
    PCIS_SCORED_PARQUET,
    CLUSTER_PROFILES_PARQUET,
    LOCATION_MEMORY_PARQUET,
    POLICE_STATIONS_GEOJSON,
    ENFORCEMENT_PRIORITIES_PARQUET,
    H3_HOTSPOT_SIG_PARQUET,
    LANE_CAPACITY_VPH,
    LOG_FORMAT,
    LOG_LEVEL,
)

warnings.filterwarnings("ignore")
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger("enforcement_optimizer")


# ============================================
# STEP 1: Geolocate police stations
# ============================================
def geolocate_stations(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Extract police station locations from violation data centroids.
    More accurate than geocoding names since it represents actual patrol areas.
    """
    logger.info("Geolocating police stations from violation data ...")

    stations = df.groupby("police_station").agg(
        centroid_lat=("latitude", "mean"),
        centroid_lon=("longitude", "mean"),
        lat_std=("latitude", "std"),
        lon_std=("longitude", "std"),
        total_violations=("latitude", "size"),
        unique_vehicles=("vehicle_number", "nunique"),
        unique_days=("date", "nunique"),
        unique_devices=("device_id", "nunique"),
        avg_pcis=("pcis", "mean"),
        max_pcis=("pcis", "max"),
        avg_severity=("violation_severity_weight", "mean"),
        avg_capacity_reduction=("capacity_reduction", "mean"),
        pct_junction=("has_junction", "mean"),
        pct_main_road=("road_is_main", "mean"),
        avg_road_width=("road_width_m", "mean"),
    ).reset_index()

    # Compute jurisdiction area (approximate from point spread)
    # Use 2*std as radius approximation
    stations["jurisdiction_radius_km"] = np.sqrt(
        stations["lat_std"].fillna(0.01)**2 + stations["lon_std"].fillna(0.01)**2
    ) * 111  # degrees to km

    # Daily violation rate
    stations["daily_rate"] = stations["total_violations"] / stations["unique_days"].clip(lower=1)

    # Enforcement intensity (devices per violation)
    stations["enforcement_intensity"] = (
        stations["unique_devices"] / stations["total_violations"].clip(lower=1)
    )

    # Create geometry
    geometry = [
        Point(row["centroid_lon"], row["centroid_lat"])
        for _, row in stations.iterrows()
    ]

    stations_gdf = gpd.GeoDataFrame(stations, geometry=geometry, crs="EPSG:4326")

    logger.info(f"  Geolocated {len(stations_gdf):,} police stations")
    logger.info(f"  Total violations covered: {stations['total_violations'].sum():,}")
    logger.info(f"  Avg violations per station: {stations['total_violations'].mean():,.0f}")
    logger.info(f"  Avg PCIS per station: {stations['avg_pcis'].mean():.3f}")

    return stations_gdf


# ============================================
# STEP 2: Compute station jurisdiction profiles
# ============================================
def compute_jurisdiction_profiles(
    stations_gdf: gpd.GeoDataFrame, df: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Enrich station profiles with temporal and violation type breakdowns."""
    logger.info("Computing jurisdiction profiles ...")

    for _, station in stations_gdf.iterrows():
        station_name = station["police_station"]
        station_df = df[df["police_station"] == station_name]

        if len(station_df) == 0:
            continue

        idx = stations_gdf[stations_gdf["police_station"] == station_name].index[0]

        # Peak hour
        stations_gdf.at[idx, "peak_hour"] = int(station_df["hour"].value_counts().index[0])

        # Dominant violation type
        if "violation_types_str" in station_df.columns:
            all_types = station_df["violation_types_str"].fillna("").str.split("|").explode()
            all_types = all_types[all_types != ""]
            if len(all_types) > 0:
                stations_gdf.at[idx, "dominant_violation"] = all_types.value_counts().index[0]

        # Weekend ratio
        stations_gdf.at[idx, "pct_weekend"] = float(station_df["is_weekend"].mean())

        # Temporal pattern
        hourly = station_df["hour"].value_counts(normalize=True)
        morning = hourly.reindex(range(7, 11), fill_value=0).sum()
        evening = hourly.reindex(range(17, 21), fill_value=0).sum()
        night = hourly.reindex(range(0, 6), fill_value=0).sum()

        if morning > 0.3:
            pattern = "morning_heavy"
        elif evening > 0.3:
            pattern = "evening_heavy"
        elif night > 0.3:
            pattern = "nighttime"
        else:
            pattern = "distributed"
        stations_gdf.at[idx, "temporal_pattern"] = pattern

    logger.info(f"  Enriched {len(stations_gdf):,} station profiles")
    return stations_gdf


# ============================================
# STEP 3: CongestionHoursRecovered (CHR) metric
# ============================================
def compute_chr_metric(df: pd.DataFrame, hex_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CongestionHoursRecovered (CHR) per H3 hexagon.
    
    CHR = PCIS × violation_frequency × avg_duration × road_capacity_affected
    
    This represents the total vehicle-hours of congestion that would be
    recovered if enforcement eliminates violations at this location.
    """
    logger.info("Computing CongestionHoursRecovered (CHR) metric ...")

    # Aggregate to hex level
    hex_chr = df.groupby("h3_index").agg(
        pcis_mean=("pcis", "mean"),
        pcis_max=("pcis", "max"),
        pcis_sum=("pcis", "sum"),
        violation_count=("pcis", "size"),
        unique_days=("date", "nunique"),
        unique_vehicles=("vehicle_number", "nunique"),
        avg_capacity_reduction=("capacity_reduction", "mean"),
        avg_proximity=("proximity_factor", "mean"),
        avg_severity=("violation_severity_weight", "mean"),
        avg_road_width=("road_width_m", "mean"),
        avg_road_lanes=("road_lanes", "mean"),
        avg_betweenness=("road_betweenness_centrality", "mean"),
        pct_main_road=("road_is_main", "mean"),
        pct_junction=("has_junction", "mean"),
        centroid_lat=("latitude", "mean"),
        centroid_lon=("longitude", "mean"),
        police_station=("police_station", lambda x: x.value_counts().index[0]),
        peak_hour=("hour", lambda x: x.value_counts().index[0]),
    ).reset_index()

    # Daily violation frequency
    hex_chr["daily_frequency"] = hex_chr["violation_count"] / hex_chr["unique_days"].clip(lower=1)

    # Estimated average violation duration (hours)
    # Based on severity: higher severity = longer parked illegally
    hex_chr["avg_duration_hours"] = (0.5 + hex_chr["avg_severity"] * 2.0).clip(0.5, 4.0)

    # Road capacity affected (vehicles per hour)
    hex_chr["capacity_affected_vph"] = (
        hex_chr["avg_capacity_reduction"] * hex_chr["avg_road_lanes"].clip(lower=1) * LANE_CAPACITY_VPH
    )

    # --- CHR Calculation ---
    # CHR = PCIS × daily_frequency × avg_duration × capacity_affected
    # Units: dimensionless × violations/day × hours × veh/hr = vehicle-hours/day
    hex_chr["chr"] = (
        hex_chr["pcis_mean"]
        * hex_chr["daily_frequency"]
        * hex_chr["avg_duration_hours"]
        * hex_chr["capacity_affected_vph"]
    )

    # Normalize CHR to [0, 100] scale for interpretability
    chr_max = hex_chr["chr"].max()
    if chr_max > 0:
        hex_chr["chr_normalized"] = (hex_chr["chr"] / chr_max * 100).clip(0, 100)
    else:
        hex_chr["chr_normalized"] = 0

    # Priority tier
    hex_chr["priority_tier"] = hex_chr["chr_normalized"].apply(_classify_priority)

    # Merge location memory if available
    memory_path = DATA_DIR / "location_memory.parquet"
    if memory_path.exists():
        memory_df = pd.read_parquet(memory_path)
        hex_chr = hex_chr.merge(
            memory_df[["h3_index", "location_memory_score", "is_addiction_zone"]],
            on="h3_index", how="left",
        )
        hex_chr["location_memory_score"] = hex_chr["location_memory_score"].fillna(0)
        hex_chr["is_addiction_zone"] = hex_chr["is_addiction_zone"].fillna(False)

    # Merge Gi* significance if available
    if hex_stats is not None and "hotspot_class" in hex_stats.columns:
        hex_chr = hex_chr.merge(
            hex_stats[["h3_index", "hotspot_class", "gi_zscore"]],
            on="h3_index", how="left",
        )

    # Sort by CHR descending
    hex_chr = hex_chr.sort_values("chr", ascending=False).reset_index(drop=True)
    hex_chr["priority_rank"] = range(1, len(hex_chr) + 1)

    logger.info(f"  Computed CHR for {len(hex_chr):,} hexagons")
    logger.info(f"  Total CHR: {hex_chr['chr'].sum():,.0f} vehicle-hours/day recoverable")
    logger.info(f"  Top hex CHR: {hex_chr['chr'].iloc[0]:,.0f}")

    tier_counts = hex_chr["priority_tier"].value_counts()
    for tier, cnt in tier_counts.items():
        logger.info(f"    {tier:15s}: {cnt:>5} hexagons")

    return hex_chr


def _classify_priority(chr_score):
    """Classify CHR into enforcement priority tiers."""
    if chr_score >= 80:
        return "URGENT"
    elif chr_score >= 50:
        return "HIGH"
    elif chr_score >= 20:
        return "MEDIUM"
    elif chr_score >= 5:
        return "LOW"
    else:
        return "MONITOR"


# ============================================
# STEP 4: Save & Summarize
# ============================================
def save_and_summarize(stations_gdf, hex_chr):
    """Save outputs and print summary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save police stations GeoJSON
    logger.info(f"Saving police stations to {POLICE_STATIONS_GEOJSON} ...")
    # Drop problematic columns for GeoJSON serialization
    save_gdf = stations_gdf.copy()
    for col in save_gdf.columns:
        if save_gdf[col].dtype == object:
            save_gdf[col] = save_gdf[col].astype(str)
    save_gdf.to_file(POLICE_STATIONS_GEOJSON, driver="GeoJSON")

    # Save enforcement priorities
    logger.info(f"Saving enforcement priorities to {ENFORCEMENT_PRIORITIES_PARQUET} ...")
    hex_chr.to_parquet(ENFORCEMENT_PRIORITIES_PARQUET, index=False, engine="pyarrow")

    print("\n" + "=" * 70)
    print("  PARKVISION AI - Enforcement Optimizer Summary")
    print("=" * 70)

    # Station summary
    print(f"\n  --- Police Station Profiles ({len(stations_gdf):,} stations) ---")
    top_stations = stations_gdf.nlargest(10, "total_violations")
    print(f"    {'Station':>25s}  {'Violations':>10s}  {'PCIS':>5s}  {'Daily':>6s}  {'Pattern':>15s}")
    for _, row in top_stations.iterrows():
        print(f"    {row['police_station']:>25s}  {row['total_violations']:>10,}  "
              f"{row['avg_pcis']:>5.3f}  {row['daily_rate']:>6.0f}  "
              f"{row.get('temporal_pattern', 'N/A'):>15s}")

    # CHR summary
    print(f"\n  --- CongestionHoursRecovered (CHR) Summary ---")
    print(f"  Total recoverable:  {hex_chr['chr'].sum():>12,.0f} vehicle-hours/day")
    print(f"  Top 20 hexagons:    {hex_chr['chr'].head(20).sum():>12,.0f} vehicle-hours/day "
          f"({hex_chr['chr'].head(20).sum() / hex_chr['chr'].sum() * 100:.1f}%)")
    print(f"  Top 50 hexagons:    {hex_chr['chr'].head(50).sum():>12,.0f} vehicle-hours/day "
          f"({hex_chr['chr'].head(50).sum() / hex_chr['chr'].sum() * 100:.1f}%)")

    # Priority tiers
    print(f"\n  --- Enforcement Priority Tiers ---")
    for tier in ["URGENT", "HIGH", "MEDIUM", "LOW", "MONITOR"]:
        tier_data = hex_chr[hex_chr["priority_tier"] == tier]
        if len(tier_data) > 0:
            chr_total = tier_data["chr"].sum()
            print(f"    {tier:10s}: {len(tier_data):>5} hexagons, "
                  f"CHR={chr_total:>10,.0f} veh-hrs/day")

    # Top 20 enforcement priorities
    print(f"\n  --- Top 20 Enforcement Priorities (by CHR) ---")
    print(f"    {'Rank':>4}  {'H3 Index':>17}  {'CHR':>10}  {'Score':>5}  "
          f"{'PCIS':>5}  {'Daily':>5}  {'Station':>20}")
    for _, row in hex_chr.head(20).iterrows():
        print(f"    {row['priority_rank']:>4}  {row['h3_index']:>17}  "
              f"{row['chr']:>10,.0f}  {row['chr_normalized']:>5.1f}  "
              f"{row['pcis_mean']:>5.3f}  {row['daily_frequency']:>5.1f}  "
              f"{row['police_station']:>20}")

    print(f"\n  Outputs:")
    print(f"    {POLICE_STATIONS_GEOJSON}")
    print(f"    {ENFORCEMENT_PRIORITIES_PARQUET}")
    print("=" * 70)


# ============================================
# MAIN PIPELINE
# ============================================
def run():
    """Execute enforcement optimization pipeline."""
    # Load PCIS-scored violations
    logger.info(f"Loading PCIS-scored violations from {PCIS_SCORED_PARQUET} ...")
    df = pd.read_parquet(PCIS_SCORED_PARQUET)
    logger.info(f"  Loaded {len(df):,} records")

    # Load hex stats for Gi* significance
    hex_stats = None
    if Path(H3_HOTSPOT_SIG_PARQUET).exists():
        hex_stats = pd.read_parquet(H3_HOTSPOT_SIG_PARQUET)
        logger.info(f"  Loaded {len(hex_stats):,} hex stats")

    # Step 1: Geolocate stations
    stations_gdf = geolocate_stations(df)

    # Step 2: Jurisdiction profiles
    stations_gdf = compute_jurisdiction_profiles(stations_gdf, df)

    # Step 3: CHR metric
    hex_chr = compute_chr_metric(df, hex_stats)

    # Step 4: Save & summarize
    save_and_summarize(stations_gdf, hex_chr)

    return stations_gdf, hex_chr


if __name__ == "__main__":
    run()
