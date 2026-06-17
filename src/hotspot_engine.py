"""
PARKVISION AI — Stage 4: ST-DBSCAN Multi-Scale Hotspot Clustering
==================================================================
Implements Spatio-Temporal DBSCAN at three spatial scales (micro/meso/macro)
to detect parking violation hotspots, then computes rich cluster profiles.

Usage:
    python -m src.hotspot_engine
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR,
    OUTPUT_DIR,
    ENRICHED_PARQUET,
    HOTSPOT_CLUSTERS_PARQUET,
    CLUSTER_PROFILES_PARQUET,
    CLUSTER_PROFILES_GEOJSON,
    STDBSCAN_PARAMS,
    VIOLATION_TYPE_NAMES,
    LOG_FORMAT,
    LOG_LEVEL,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger("hotspot_engine")

# Earth radius in meters (for haversine conversion)
EARTH_RADIUS_M = 6_371_000


# ============================================
# STEP 1: Run DBSCAN at 3 spatial scales
# ============================================
def run_multiscale_dbscan(df: pd.DataFrame) -> pd.DataFrame:
    """Run spatial DBSCAN at micro, meso, macro scales."""
    logger.info("Running multi-scale ST-DBSCAN clustering ...")

    # Convert lat/lon to radians (required for haversine metric)
    coords_rad = np.radians(df[["latitude", "longitude"]].values)

    for scale_name, params in STDBSCAN_PARAMS.items():
        eps_m = params["eps_spatial_m"]
        min_pts = params["min_pts"]

        # Convert eps from meters to radians
        eps_rad = eps_m / EARTH_RADIUS_M

        logger.info(
            f"  [{scale_name.upper()}] eps={eps_m}m ({eps_rad:.6f} rad), min_pts={min_pts} ..."
        )

        # Run DBSCAN with haversine metric (uses BallTree — efficient for large datasets)
        db = DBSCAN(
            eps=eps_rad,
            min_samples=min_pts,
            metric="haversine",
            algorithm="ball_tree",
            n_jobs=-1,
        )
        labels = db.fit_predict(coords_rad)

        col_name = f"cluster_{scale_name}"
        df[col_name] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        n_clustered = (labels != -1).sum()

        logger.info(
            f"    Clusters: {n_clusters:,} | "
            f"Clustered: {n_clustered:,} ({n_clustered/len(df)*100:.1f}%) | "
            f"Noise: {n_noise:,} ({n_noise/len(df)*100:.1f}%)"
        )

    return df


# ============================================
# STEP 2: Temporal sub-clustering within spatial clusters
# ============================================
def temporal_subclustering(df: pd.DataFrame, scale: str = "meso") -> pd.DataFrame:
    """Within each spatial cluster, split by temporal gaps > eps_temporal."""
    eps_temporal_h = STDBSCAN_PARAMS[scale]["eps_temporal_h"]
    col_name = f"cluster_{scale}"
    st_col = f"stcluster_{scale}"

    logger.info(f"  Temporal sub-clustering within {scale} clusters (gap > {eps_temporal_h}h) ...")

    # Convert datetime to hours since epoch for numeric comparison
    if "created_datetime_ist" in df.columns:
        dt_col = "created_datetime_ist"
    else:
        dt_col = "created_datetime"

    # For each spatial cluster, check temporal continuity
    df[st_col] = -1  # default noise
    next_st_label = 0

    spatial_clusters = df[df[col_name] != -1][col_name].unique()

    for sc in spatial_clusters:
        mask = df[col_name] == sc
        subset = df.loc[mask].sort_values(dt_col)

        if len(subset) == 0:
            continue

        # Compute time gaps in hours
        times = pd.to_datetime(subset[dt_col])
        time_diffs_h = times.diff().dt.total_seconds().fillna(0) / 3600

        # Split at gaps larger than eps_temporal
        sub_label = next_st_label
        sub_labels = []
        for gap in time_diffs_h:
            if gap > eps_temporal_h * 24 * 7:  # Use weekly gap for multi-month data
                sub_label = next_st_label
                next_st_label += 1
            sub_labels.append(sub_label)

        if sub_labels:
            df.loc[subset.index, st_col] = sub_labels
            next_st_label = sub_label + 1

    n_st_clusters = df[df[st_col] != -1][st_col].nunique()
    logger.info(f"    Spatio-temporal clusters: {n_st_clusters:,}")

    return df


# ============================================
# STEP 3: Compute cluster profiles (meso scale)
# ============================================
def compute_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute detailed profile for each meso-level spatial cluster."""
    logger.info("Computing meso-level cluster profiles ...")

    scale = "meso"
    col_name = f"cluster_{scale}"

    # Filter to clustered violations only
    clustered = df[df[col_name] != -1].copy()
    logger.info(f"  {len(clustered):,} clustered violations in {clustered[col_name].nunique():,} clusters")

    grouped = clustered.groupby(col_name)

    profiles = pd.DataFrame()

    # --- Basic counts ---
    profiles["violation_count"] = grouped.size()
    profiles["unique_vehicles"] = grouped["vehicle_number"].nunique()
    profiles["unique_days"] = grouped["date"].nunique()
    profiles["unique_devices"] = grouped["device_id"].nunique()

    # --- Centroid ---
    profiles["centroid_lat"] = grouped["latitude"].mean()
    profiles["centroid_lon"] = grouped["longitude"].mean()

    # --- Spatial extent ---
    profiles["lat_spread"] = grouped["latitude"].apply(lambda x: x.max() - x.min())
    profiles["lon_spread"] = grouped["longitude"].apply(lambda x: x.max() - x.min())

    # --- Severity ---
    profiles["avg_severity"] = grouped["violation_severity_weight"].mean()
    profiles["max_severity"] = grouped["violation_severity_weight"].max()

    # --- Dominant violation type ---
    profiles["dominant_violation"] = grouped["violation_types_str"].agg(
        lambda x: _get_dominant_violation(x)
    )

    # --- Vehicle type distribution ---
    profiles["dominant_vehicle"] = grouped["vehicle_type"].agg(
        lambda x: x.value_counts().index[0]
    )
    profiles["pct_car"] = grouped["vehicle_type"].apply(lambda x: (x == "CAR").mean())
    profiles["pct_scooter"] = grouped["vehicle_type"].apply(
        lambda x: (x.isin(["SCOOTER", "MOPED"])).mean()
    )
    profiles["pct_auto"] = grouped["vehicle_type"].apply(
        lambda x: (x == "PASSENGER AUTO").mean()
    )

    # --- Temporal patterns ---
    profiles["peak_hour"] = grouped["hour"].agg(lambda x: x.value_counts().index[0])
    profiles["peak_day"] = grouped["day_name"].agg(lambda x: x.value_counts().index[0])
    profiles["pct_peak_hour"] = grouped["is_peak_hour"].mean()
    profiles["pct_weekend"] = grouped["is_weekend"].mean()

    # Hour distribution entropy (high = spread across hours, low = concentrated)
    profiles["hour_concentration"] = grouped["hour"].apply(_compute_concentration)

    # --- Road characteristics ---
    profiles["avg_road_width"] = grouped["road_width_m"].mean()
    profiles["avg_road_lanes"] = grouped["road_lanes"].mean()
    profiles["avg_betweenness"] = grouped["road_betweenness_centrality"].mean()
    profiles["pct_main_road"] = grouped["road_is_main"].mean()
    profiles["dominant_highway"] = grouped["highway_class"].agg(
        lambda x: x.value_counts().index[0]
    )

    # --- Location context ---
    profiles["primary_police_station"] = grouped["police_station"].agg(
        lambda x: x.value_counts().index[0]
    )
    profiles["pct_at_junction"] = grouped["has_junction"].mean()

    # --- POI proximity ---
    for poi in ["metro_station", "bus_stop", "hospital", "school", "market"]:
        col = f"near_{poi}"
        if col in clustered.columns:
            profiles[f"pct_near_{poi}"] = grouped[col].mean()

    # --- Date range ---
    if "created_datetime_ist" in clustered.columns:
        profiles["first_seen"] = grouped["created_datetime_ist"].min()
        profiles["last_seen"] = grouped["created_datetime_ist"].max()
        profiles["active_span_days"] = (
            profiles["last_seen"] - profiles["first_seen"]
        ).dt.days

    # --- Violation intensity (violations per day active) ---
    profiles["daily_intensity"] = profiles["violation_count"] / profiles["unique_days"].clip(lower=1)

    profiles = profiles.reset_index().rename(columns={col_name: "cluster_id"})

    logger.info(f"  Computed profiles for {len(profiles):,} clusters")

    return profiles


def _get_dominant_violation(series):
    """Get the most common violation type from pipe-separated strings."""
    all_violations = []
    for val in series:
        if isinstance(val, str) and val:
            all_violations.extend(val.split("|"))
    if not all_violations:
        return "UNKNOWN"
    counter = Counter(all_violations)
    return counter.most_common(1)[0][0]


def _compute_concentration(series):
    """Compute concentration of violations across hours (0=uniform, 1=single hour)."""
    counts = series.value_counts(normalize=True)
    if len(counts) <= 1:
        return 1.0
    # Herfindahl-Hirschman Index
    hhi = (counts ** 2).sum()
    # Normalize: 1/24 (uniform) to 1.0 (concentrated)
    return round((hhi - 1/24) / (1 - 1/24), 4)


# ============================================
# STEP 4: Create cluster GeoJSON
# ============================================
def create_cluster_geojson(profiles: pd.DataFrame) -> gpd.GeoDataFrame:
    """Create GeoDataFrame with cluster centroid points."""
    logger.info("Creating cluster GeoJSON ...")

    geometry = [
        Point(row["centroid_lon"], row["centroid_lat"])
        for _, row in profiles.iterrows()
    ]

    # Drop datetime columns for GeoJSON serialization
    geojson_df = profiles.drop(columns=["first_seen", "last_seen"], errors="ignore")

    gdf = gpd.GeoDataFrame(geojson_df, geometry=geometry, crs="EPSG:4326")
    logger.info(f"  Created {len(gdf):,} cluster point features")

    return gdf


# ============================================
# STEP 5: Classify clusters by pattern
# ============================================
def classify_cluster_patterns(profiles: pd.DataFrame) -> pd.DataFrame:
    """Classify each cluster's temporal pattern."""
    logger.info("Classifying cluster temporal patterns ...")

    patterns = []
    for _, row in profiles.iterrows():
        peak = row["peak_hour"]
        pct_weekend = row["pct_weekend"]
        concentration = row["hour_concentration"]
        intensity = row["daily_intensity"]

        if concentration > 0.3 and 7 <= peak <= 10:
            pattern = "morning_commercial"
        elif concentration > 0.3 and 17 <= peak <= 20:
            pattern = "evening_commercial"
        elif concentration > 0.3 and 10 <= peak <= 16:
            pattern = "midday_persistent"
        elif pct_weekend > 0.5:
            pattern = "weekend_spike"
        elif concentration < 0.05:
            pattern = "all_day_uniform"
        elif 0 <= peak <= 5:
            pattern = "nighttime"
        elif intensity > 20:
            pattern = "high_intensity"
        else:
            pattern = "mixed"

        patterns.append(pattern)

    profiles["temporal_pattern"] = patterns

    pattern_counts = profiles["temporal_pattern"].value_counts()
    for pattern, count in pattern_counts.items():
        logger.info(f"    {pattern:25s}: {count:>4} clusters")

    return profiles


# ============================================
# STEP 6: Save & summarize
# ============================================
def save_and_summarize(df, profiles, cluster_gdf):
    """Save outputs and print summary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save clustered violations
    logger.info(f"Saving clustered violations to {HOTSPOT_CLUSTERS_PARQUET} ...")
    df.to_parquet(HOTSPOT_CLUSTERS_PARQUET, index=False, engine="pyarrow")

    # Save cluster profiles
    logger.info(f"Saving cluster profiles to {CLUSTER_PROFILES_PARQUET} ...")
    save_profiles = profiles.drop(columns=["first_seen", "last_seen"], errors="ignore")
    save_profiles.to_parquet(CLUSTER_PROFILES_PARQUET, index=False, engine="pyarrow")

    # Save cluster GeoJSON
    logger.info(f"Saving cluster GeoJSON to {CLUSTER_PROFILES_GEOJSON} ...")
    cluster_gdf.to_file(CLUSTER_PROFILES_GEOJSON, driver="GeoJSON")

    # Print summary
    print("\n" + "=" * 70)
    print("  PARKVISION AI - ST-DBSCAN Hotspot Clustering Summary")
    print("=" * 70)

    print(f"\n  --- Multi-Scale Clustering Results ---")
    for scale in ["micro", "meso", "macro"]:
        col = f"cluster_{scale}"
        if col in df.columns:
            n_clusters = df[df[col] != -1][col].nunique()
            n_clustered = (df[col] != -1).sum()
            pct = n_clustered / len(df) * 100
            print(f"    {scale:8s}: {n_clusters:>5,} clusters, "
                  f"{n_clustered:>7,} violations ({pct:.1f}%)")

    print(f"\n  --- Meso Cluster Profile Summary ({len(profiles):,} clusters) ---")
    print(f"    Violations per cluster:  mean={profiles['violation_count'].mean():.0f}, "
          f"median={profiles['violation_count'].median():.0f}, "
          f"max={profiles['violation_count'].max():,}")
    print(f"    Avg severity:            {profiles['avg_severity'].mean():.3f}")
    print(f"    Daily intensity:         mean={profiles['daily_intensity'].mean():.1f}")
    print(f"    Avg road width:          {profiles['avg_road_width'].mean():.1f}m")
    print(f"    On main roads:           {profiles['pct_main_road'].mean()*100:.1f}%")
    print(f"    At junctions:            {profiles['pct_at_junction'].mean()*100:.1f}%")

    # Top 15 clusters
    print(f"\n  --- Top 15 Hotspot Clusters (by violation count) ---")
    top = profiles.nlargest(15, "violation_count")
    print(f"    {'ID':>5}  {'Count':>6}  {'Sev':>5}  {'Pattern':>22}  {'Station':>20}  {'Peak':>5}  {'Road':>8}")
    print(f"    {'---':>5}  {'---':>6}  {'---':>5}  {'---':>22}  {'---':>20}  {'---':>5}  {'---':>8}")
    for _, row in top.iterrows():
        print(
            f"    {row['cluster_id']:>5}  "
            f"{row['violation_count']:>6,}  "
            f"{row['avg_severity']:>5.2f}  "
            f"{row['temporal_pattern']:>22}  "
            f"{row['primary_police_station']:>20}  "
            f"{row['peak_hour']:>5}  "
            f"{row['dominant_highway']:>8}"
        )

    # Temporal pattern distribution
    print(f"\n  --- Temporal Pattern Distribution ---")
    for pattern, count in profiles["temporal_pattern"].value_counts().items():
        total_violations = profiles[profiles["temporal_pattern"] == pattern]["violation_count"].sum()
        print(f"    {pattern:25s}: {count:>4} clusters, {total_violations:>7,} violations")

    print(f"\n  Outputs:")
    print(f"    {HOTSPOT_CLUSTERS_PARQUET}")
    print(f"    {CLUSTER_PROFILES_PARQUET}")
    print(f"    {CLUSTER_PROFILES_GEOJSON}")
    print("=" * 70)


# ============================================
# MAIN PIPELINE
# ============================================
def run():
    """Execute hotspot clustering pipeline."""
    # Load enriched violations
    logger.info(f"Loading enriched violations from {ENRICHED_PARQUET} ...")
    df = pd.read_parquet(ENRICHED_PARQUET)
    logger.info(f"  Loaded {len(df):,} records")

    # Step 1: Multi-scale spatial DBSCAN
    df = run_multiscale_dbscan(df)

    # Step 2: Temporal sub-clustering (meso scale)
    df = temporal_subclustering(df, scale="meso")

    # Step 3: Compute cluster profiles
    profiles = compute_cluster_profiles(df)

    # Step 4: Classify temporal patterns
    profiles = classify_cluster_patterns(profiles)

    # Step 5: Create GeoJSON
    cluster_gdf = create_cluster_geojson(profiles)

    # Step 6: Save & summarize
    save_and_summarize(df, profiles, cluster_gdf)

    return df, profiles


if __name__ == "__main__":
    run()
