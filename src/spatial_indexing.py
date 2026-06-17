"""
PARKVISION AI — Stage 3: H3 Hexagonal Binning & POI Enrichment
=================================================================
Assigns H3 hex indices to violations, computes per-hex aggregates,
and enriches with nearby Points of Interest from OpenStreetMap.

Usage:
    python -m src.spatial_indexing
"""

import sys
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import h3
import requests
from shapely.geometry import Point, Polygon

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR,
    OUTPUT_DIR,
    ENRICHED_PARQUET,
    H3_HEX_STATS_PARQUET,
    H3_HEX_STATS_GEOJSON,
    H3_RESOLUTION,
    POI_SEARCH_RADIUS_M,
    BENGALURU_BBOX,
    LOG_FORMAT,
    LOG_LEVEL,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger("spatial_indexing")


# ============================================
# STEP 1: Assign H3 hex indices
# ============================================
def assign_h3_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Assign H3 hexagonal index to each violation at resolution 9."""
    logger.info(f"Assigning H3 indices at resolution {H3_RESOLUTION} ...")

    # h3 v4 API: latlng_to_cell
    df["h3_index"] = df.apply(
        lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], H3_RESOLUTION),
        axis=1,
    )

    n_hexes = df["h3_index"].nunique()
    logger.info(f"  Assigned {len(df):,} violations to {n_hexes:,} unique hexagons")

    return df


# ============================================
# STEP 2: Compute per-hexagon aggregates
# ============================================
def compute_hex_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate statistics per H3 hexagon."""
    logger.info("Computing per-hexagon aggregate statistics ...")

    # Group by H3 index
    grouped = df.groupby("h3_index")

    hex_stats = pd.DataFrame()
    hex_stats["violation_count"] = grouped.size()
    hex_stats["unique_days"] = grouped["date"].nunique()
    hex_stats["unique_vehicles"] = grouped["vehicle_number"].nunique()
    hex_stats["avg_severity"] = grouped["violation_severity_weight"].mean()
    hex_stats["max_severity"] = grouped["violation_severity_weight"].max()
    hex_stats["avg_temporal_multiplier"] = grouped["temporal_demand_multiplier"].mean()

    # Dominant vehicle type
    hex_stats["dominant_vehicle_type"] = grouped["vehicle_type"].agg(
        lambda x: x.value_counts().index[0]
    )

    # Peak hour (mode)
    hex_stats["peak_hour"] = grouped["hour"].agg(
        lambda x: x.value_counts().index[0]
    )

    # Violation type prevalence
    hex_stats["pct_wrong_parking"] = grouped["is_wrong_parking"].mean()
    hex_stats["pct_no_parking"] = grouped["is_no_parking"].mean()
    hex_stats["pct_main_road"] = grouped["is_main_road_parking"].mean()

    # Road characteristics (averaged across violations in hex)
    hex_stats["avg_road_width"] = grouped["road_width_m"].mean()
    hex_stats["avg_road_lanes"] = grouped["road_lanes"].mean()
    hex_stats["avg_betweenness"] = grouped["road_betweenness_centrality"].mean()
    hex_stats["pct_on_main_road"] = grouped["road_is_main"].mean()

    # Weekend vs weekday ratio
    hex_stats["pct_weekend"] = grouped["is_weekend"].mean()

    # Time range of activity
    hex_stats["first_violation"] = grouped["created_datetime_ist"].min()
    hex_stats["last_violation"] = grouped["created_datetime_ist"].max()

    # Lat/lon of hex centroid (for spatial operations)
    hex_stats["centroid_lat"] = hex_stats.index.map(
        lambda h: h3.cell_to_latlng(h)[0]
    )
    hex_stats["centroid_lon"] = hex_stats.index.map(
        lambda h: h3.cell_to_latlng(h)[1]
    )

    # Police station (mode)
    hex_stats["primary_police_station"] = grouped["police_station"].agg(
        lambda x: x.value_counts().index[0]
    )

    # Has junction
    hex_stats["pct_at_junction"] = grouped["has_junction"].mean()

    hex_stats = hex_stats.reset_index()

    logger.info(f"  Computed stats for {len(hex_stats):,} hexagons")
    logger.info(
        f"  Violation count range: {hex_stats['violation_count'].min()} - "
        f"{hex_stats['violation_count'].max()} (mean: {hex_stats['violation_count'].mean():.1f})"
    )

    return hex_stats


# ============================================
# STEP 3: Create GeoJSON with hex boundaries
# ============================================
def create_hex_geojson(hex_stats: pd.DataFrame) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame with H3 hexagon polygon geometries."""
    logger.info("Creating hexagon geometries for GeoJSON ...")

    def h3_to_polygon(h3_index):
        """Convert H3 index to Shapely Polygon."""
        boundary = h3.cell_to_boundary(h3_index)
        # h3 v4 returns list of (lat, lng) tuples — need to swap to (lng, lat) for Shapely
        coords = [(lng, lat) for lat, lng in boundary]
        coords.append(coords[0])  # close the polygon
        return Polygon(coords)

    hex_stats["geometry"] = hex_stats["h3_index"].apply(h3_to_polygon)

    # Drop datetime columns (not JSON-serializable for GeoJSON)
    cols_to_drop = [c for c in hex_stats.columns if "violation" in c and "first" in c or "last" in c]
    geojson_df = hex_stats.drop(columns=["first_violation", "last_violation"], errors="ignore")

    gdf = gpd.GeoDataFrame(geojson_df, geometry="geometry", crs="EPSG:4326")

    logger.info(f"  Created {len(gdf):,} hexagon polygons")
    return gdf


# ============================================
# STEP 4: POI enrichment via Overpass API
# ============================================
def enrich_with_pois(hex_stats: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Enrich hexagons with POI proximity data using text-based detection from location field."""
    logger.info("Enriching hexagons with POI proximity data ...")
    logger.info("  Using text-based POI detection from violation location strings ...")

    # Extract POIs from location text (reliable, no external API needed)
    pois_by_type = _extract_pois_from_text(df)

    # Compute distances from each hex centroid to nearest POI of each type
    logger.info("  Computing nearest POI distances per hexagon ...")

    centroids = np.column_stack([
        hex_stats["centroid_lat"].values,
        hex_stats["centroid_lon"].values,
    ])

    search_radius_deg = POI_SEARCH_RADIUS_M / 111000  # rough meters-to-degrees

    for poi_type in ["metro_station", "bus_stop", "hospital", "school", "market"]:
        col_name = f"near_{poi_type}"
        dist_col = f"dist_{poi_type}_m"

        poi_points = pois_by_type.get(poi_type, [])

        if len(poi_points) == 0:
            hex_stats[col_name] = False
            hex_stats[dist_col] = 99999.0
            continue

        poi_arr = np.array(poi_points)  # (N, 2) array of (lat, lon)

        min_dists = []
        for i in range(len(centroids)):
            clat, clon = centroids[i]
            # Quick bounding-box pre-filter
            mask = (
                (np.abs(poi_arr[:, 0] - clat) < search_radius_deg * 3)
                & (np.abs(poi_arr[:, 1] - clon) < search_radius_deg * 3)
            )
            nearby = poi_arr[mask]

            if len(nearby) == 0:
                min_dists.append(99999.0)
            else:
                # Haversine-approximate distance in meters
                dlat = np.radians(nearby[:, 0] - clat)
                dlon = np.radians(nearby[:, 1] - clon)
                a = (
                    np.sin(dlat / 2) ** 2
                    + np.cos(np.radians(clat))
                    * np.cos(np.radians(nearby[:, 0]))
                    * np.sin(dlon / 2) ** 2
                )
                dists_m = 6371000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                min_dists.append(float(np.min(dists_m)))

        hex_stats[dist_col] = min_dists
        hex_stats[col_name] = hex_stats[dist_col] <= POI_SEARCH_RADIUS_M

        n_near = hex_stats[col_name].sum()
        logger.info(f"    {col_name:25s}: {n_near:,} hexagons ({n_near / len(hex_stats) * 100:.1f}%)")

    return hex_stats


def _extract_pois_from_text(df: pd.DataFrame) -> dict:
    """Extract POI locations from the 'location' text field using keyword matching."""
    pois = {
        "metro_station": [],
        "bus_stop": [],
        "hospital": [],
        "school": [],
        "market": [],
    }

    if "location" not in df.columns:
        return pois

    location_lower = df["location"].fillna("").str.lower()

    # Metro / Railway station keywords
    metro_mask = location_lower.str.contains(
        "metro|railway station|namma metro|majestic|kempegowda", regex=True
    )
    for _, row in df[metro_mask].drop_duplicates(subset=["h3_index"]).iterrows():
        pois["metro_station"].append((row["latitude"], row["longitude"]))

    # Bus stop keywords
    bus_mask = location_lower.str.contains("bus stop|bus stand|bmtc|ksrtc|majestic", regex=True)
    for _, row in df[bus_mask].drop_duplicates(subset=["h3_index"]).iterrows():
        pois["bus_stop"].append((row["latitude"], row["longitude"]))

    # Hospital keywords
    hosp_mask = location_lower.str.contains(
        "hospital|medical|clinic|health centre|nursing home", regex=True
    )
    for _, row in df[hosp_mask].drop_duplicates(subset=["h3_index"]).iterrows():
        pois["hospital"].append((row["latitude"], row["longitude"]))

    # School keywords
    school_mask = location_lower.str.contains(
        "school|college|university|institute|academy|vidyalaya", regex=True
    )
    for _, row in df[school_mask].drop_duplicates(subset=["h3_index"]).iterrows():
        pois["school"].append((row["latitude"], row["longitude"]))

    # Market / Mall keywords
    market_mask = location_lower.str.contains(
        "market|mall|commercial|bazaar|complex|forum|mantri|orion", regex=True
    )
    for _, row in df[market_mask].drop_duplicates(subset=["h3_index"]).iterrows():
        pois["market"].append((row["latitude"], row["longitude"]))

    for ptype, pts in pois.items():
        logger.info(f"    {ptype:20s}: {len(pts):,} POIs extracted from text")

    return pois


# ============================================
# STEP 5: Propagate hex/POI data back to violations
# ============================================
def propagate_to_violations(df: pd.DataFrame, hex_stats: pd.DataFrame) -> pd.DataFrame:
    """Join hex-level POI flags back to individual violations."""
    logger.info("Propagating POI data back to violation records ...")

    poi_cols = [c for c in hex_stats.columns if c.startswith("near_") or c.startswith("dist_")]
    join_cols = ["h3_index"] + poi_cols

    # Drop any existing POI columns from previous runs to prevent duplicates
    existing_poi_cols = [c for c in df.columns if c.startswith("near_") or c.startswith("dist_")]
    if existing_poi_cols:
        df = df.drop(columns=existing_poi_cols)

    df = df.merge(hex_stats[join_cols], on="h3_index", how="left")

    # Fill any NaN POI flags with False
    for col in poi_cols:
        if col.startswith("near_"):
            df[col] = df[col].fillna(False)
        elif col.startswith("dist_"):
            df[col] = df[col].fillna(99999.0)

    logger.info(f"  Propagated POI data to {len(df):,} violations")
    return df


# ============================================
# STEP 6: Save & summarize
# ============================================
def save_and_summarize(df, hex_stats, hex_gdf):
    """Save outputs and print summary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save hex stats parquet
    logger.info(f"Saving hex stats to {H3_HEX_STATS_PARQUET} ...")
    save_df = hex_stats.drop(columns=["geometry"], errors="ignore")
    save_df.to_parquet(H3_HEX_STATS_PARQUET, index=False, engine="pyarrow")

    # Save hex GeoJSON
    logger.info(f"Saving hex GeoJSON to {H3_HEX_STATS_GEOJSON} ...")
    hex_gdf.to_file(H3_HEX_STATS_GEOJSON, driver="GeoJSON")

    # Save updated enriched violations
    logger.info(f"Saving enriched violations to {ENRICHED_PARQUET} ...")
    df.to_parquet(ENRICHED_PARQUET, index=False, engine="pyarrow")

    # Print summary
    print("\n" + "=" * 70)
    print("  PARKVISION AI - H3 Binning & POI Enrichment Summary")
    print("=" * 70)

    print(f"\n  --- H3 Hexagonal Binning (Resolution {H3_RESOLUTION}) ---")
    print(f"  Total hexagons:          {len(hex_stats):,}")
    print(f"  Violations per hex:      min={hex_stats['violation_count'].min()}, "
          f"max={hex_stats['violation_count'].max()}, "
          f"mean={hex_stats['violation_count'].mean():.1f}, "
          f"median={hex_stats['violation_count'].median():.0f}")
    print(f"  Active days per hex:     mean={hex_stats['unique_days'].mean():.1f}, "
          f"max={hex_stats['unique_days'].max()}")

    # Top hexagons by violation count
    print(f"\n  --- Top 10 Hexagons by Violation Count ---")
    top = hex_stats.nlargest(10, "violation_count")
    for _, row in top.iterrows():
        print(f"    {row['h3_index']}  "
              f"count={row['violation_count']:>5}  "
              f"station={row['primary_police_station']:>20s}  "
              f"peak_hr={row['peak_hour']:>2}")

    # POI proximity
    print(f"\n  --- POI Proximity (within {POI_SEARCH_RADIUS_M}m) ---")
    for poi_type in ["metro_station", "bus_stop", "hospital", "school", "market"]:
        col = f"near_{poi_type}"
        if col in hex_stats.columns:
            n = hex_stats[col].sum()
            print(f"    {poi_type:20s}: {n:>5,} hexagons ({n/len(hex_stats)*100:.1f}%)")

    # Violation-level POI stats
    print(f"\n  --- Violations Near POIs ---")
    for poi_type in ["metro_station", "bus_stop", "hospital", "school", "market"]:
        col = f"near_{poi_type}"
        if col in df.columns:
            n = df[col].sum()
            print(f"    {poi_type:20s}: {n:>7,} violations ({n/len(df)*100:.1f}%)")

    print(f"\n  Outputs:")
    print(f"    {H3_HEX_STATS_PARQUET}")
    print(f"    {H3_HEX_STATS_GEOJSON}")
    print(f"    {ENRICHED_PARQUET} (updated with h3_index + POI flags)")
    print("=" * 70)


# ============================================
# MAIN PIPELINE
# ============================================
def run():
    """Execute H3 binning and POI enrichment pipeline."""
    # Load enriched violations
    logger.info(f"Loading enriched violations from {ENRICHED_PARQUET} ...")
    df = pd.read_parquet(ENRICHED_PARQUET)
    logger.info(f"  Loaded {len(df):,} records")

    # Step 1: Assign H3 indices
    df = assign_h3_indices(df)

    # Step 2: Compute hex stats
    hex_stats = compute_hex_stats(df)

    # Step 3: Create hex GeoJSON
    hex_gdf = create_hex_geojson(hex_stats)

    # Step 4: POI enrichment
    hex_stats = enrich_with_pois(hex_stats, df)

    # Step 5: Propagate POI data to violations
    df = propagate_to_violations(df, hex_stats)

    # Step 6: Save & summarize
    save_and_summarize(df, hex_stats, hex_gdf)

    return df, hex_stats


if __name__ == "__main__":
    run()
