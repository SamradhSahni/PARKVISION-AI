"""
PARKVISION AI — Stage 2: Road Network Extraction & Map-Matching
=================================================================
Downloads Bengaluru road network via OSMnx, computes network metrics,
and map-matches every violation to its nearest road segment.

Usage:
    python -m src.road_network
"""

import sys
import logging
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
from shapely.geometry import Point

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR,
    CLEANED_PARQUET,
    ENRICHED_PARQUET,
    ROAD_GRAPH_FILE,
    ROAD_EDGES_PARQUET,
    BENGALURU_CENTER,
    BENGALURU_NETWORK_DIST_M,
    STANDARD_LANE_WIDTH_M,
    DEFAULT_LANES,
    DEFAULT_SPEED_LIMIT_KMPH,
    DEFAULT_ROAD_WIDTH_M,
    MAIN_ROAD_HIGHWAY_CLASSES,
    LOG_FORMAT,
    LOG_LEVEL,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger("road_network")


# ============================================
# Default road widths by highway class (meters)
# Used when OSM lacks explicit width/lanes tags
# ============================================
HIGHWAY_DEFAULT_WIDTH = {
    "motorway": 14.0,       # 4 lanes
    "motorway_link": 7.0,   # 2 lanes
    "trunk": 10.5,          # 3 lanes
    "trunk_link": 7.0,
    "primary": 10.5,        # 3 lanes
    "primary_link": 7.0,
    "secondary": 7.0,       # 2 lanes
    "secondary_link": 7.0,
    "tertiary": 7.0,        # 2 lanes
    "tertiary_link": 5.5,
    "residential": 5.5,     # ~1.5 lanes
    "living_street": 4.0,
    "service": 4.0,
    "unclassified": 5.5,
}

HIGHWAY_DEFAULT_LANES = {
    "motorway": 4, "motorway_link": 2,
    "trunk": 3, "trunk_link": 2,
    "primary": 3, "primary_link": 2,
    "secondary": 2, "secondary_link": 1,
    "tertiary": 2, "tertiary_link": 1,
    "residential": 2, "living_street": 1,
    "service": 1, "unclassified": 2,
}

HIGHWAY_DEFAULT_SPEED = {
    "motorway": 80, "motorway_link": 60,
    "trunk": 60, "trunk_link": 40,
    "primary": 50, "primary_link": 40,
    "secondary": 40, "secondary_link": 30,
    "tertiary": 40, "tertiary_link": 30,
    "residential": 30, "living_street": 20,
    "service": 20, "unclassified": 30,
}


# ============================================
# STEP 1: Download Bengaluru road network
# ============================================
def download_road_network():
    """Download the drivable road network for Bengaluru using OSMnx."""
    logger.info("Downloading Bengaluru road network from OpenStreetMap ...")
    logger.info(f"  Center: {BENGALURU_CENTER}, Radius: {BENGALURU_NETWORK_DIST_M}m")

    # Check for cached graph
    cache_path = DATA_DIR / "bengaluru_graph.pkl"
    if cache_path.exists():
        logger.info(f"  Loading cached graph from {cache_path} ...")
        with open(cache_path, "rb") as f:
            G = pickle.load(f)
        logger.info(f"  Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G

    # Download from OSM
    G = ox.graph_from_point(
        BENGALURU_CENTER,
        dist=BENGALURU_NETWORK_DIST_M,
        network_type="drive",
        simplify=True,
    )

    logger.info(f"  Downloaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Cache for reuse
    with open(cache_path, "wb") as f:
        pickle.dump(G, f)
    logger.info(f"  Cached graph to {cache_path}")

    return G


# ============================================
# STEP 2: Extract and enrich edge attributes
# ============================================
def extract_edge_attributes(G):
    """Extract edge GeoDataFrame and compute derived road attributes."""
    logger.info("Extracting edge attributes ...")

    # Convert to GeoDataFrames
    nodes_gdf, edges_gdf = ox.convert.graph_to_gdfs(G)
    logger.info(f"  Edges GeoDataFrame: {len(edges_gdf):,} road segments")

    # --- 2a: Parse highway class ---
    # OSM highway can be a list; take the first/primary value
    edges_gdf["highway_class"] = edges_gdf["highway"].apply(_get_primary_value)
    logger.info("  Parsed highway_class")

    # --- 2b: Parse and fill lanes ---
    edges_gdf["lanes_raw"] = edges_gdf.get("lanes", pd.Series(dtype=object))
    edges_gdf["lanes_parsed"] = edges_gdf["lanes_raw"].apply(_parse_numeric)
    edges_gdf["lanes_final"] = edges_gdf.apply(
        lambda r: r["lanes_parsed"]
        if pd.notna(r["lanes_parsed"])
        else HIGHWAY_DEFAULT_LANES.get(r["highway_class"], DEFAULT_LANES),
        axis=1,
    ).astype(int)
    logger.info("  Parsed lanes (with highway-class defaults)")

    # --- 2c: Parse and fill speed limit ---
    edges_gdf["maxspeed_raw"] = edges_gdf.get("maxspeed", pd.Series(dtype=object))
    edges_gdf["maxspeed_parsed"] = edges_gdf["maxspeed_raw"].apply(_parse_speed)
    edges_gdf["maxspeed_final"] = edges_gdf.apply(
        lambda r: r["maxspeed_parsed"]
        if pd.notna(r["maxspeed_parsed"])
        else HIGHWAY_DEFAULT_SPEED.get(r["highway_class"], DEFAULT_SPEED_LIMIT_KMPH),
        axis=1,
    ).astype(int)
    logger.info("  Parsed maxspeed (with highway-class defaults)")

    # --- 2d: Estimate road width ---
    edges_gdf["width_raw"] = edges_gdf.get("width", pd.Series(dtype=object))
    edges_gdf["width_parsed"] = edges_gdf["width_raw"].apply(_parse_numeric)
    edges_gdf["width_meters"] = edges_gdf.apply(
        lambda r: r["width_parsed"]
        if pd.notna(r["width_parsed"])
        else HIGHWAY_DEFAULT_WIDTH.get(
            r["highway_class"],
            r["lanes_final"] * STANDARD_LANE_WIDTH_M,
        ),
        axis=1,
    )
    logger.info("  Estimated road width (from width/lanes/highway defaults)")

    # --- 2e: Parse oneway ---
    edges_gdf["is_oneway"] = edges_gdf.get("oneway", False)
    if edges_gdf["is_oneway"].dtype == object:
        edges_gdf["is_oneway"] = edges_gdf["is_oneway"].map(
            {"True": True, "False": False, True: True, False: False}
        ).fillna(False)

    # --- 2f: Road length ---
    edges_gdf["length_m"] = edges_gdf["length"]

    # --- 2g: Is main road flag ---
    edges_gdf["is_main_road"] = edges_gdf["highway_class"].isin(MAIN_ROAD_HIGHWAY_CLASSES)

    logger.info(
        f"  Main roads: {edges_gdf['is_main_road'].sum():,} / {len(edges_gdf):,} "
        f"({edges_gdf['is_main_road'].mean()*100:.1f}%)"
    )

    return G, nodes_gdf, edges_gdf


def _get_primary_value(val):
    """Extract primary value from potentially list-type OSM tags."""
    if isinstance(val, (list, np.ndarray)):
        return str(val[0]) if len(val) > 0 else "unclassified"
    if val is None:
        return "unclassified"
    try:
        if pd.isna(val):
            return "unclassified"
    except (ValueError, TypeError):
        pass
    return str(val)


def _parse_numeric(val):
    """Parse a numeric value from OSM tag (handles '3', '3.5', ['3','2'], arrays, etc.)."""
    if val is None:
        return np.nan
    # Handle numpy arrays and lists first
    if isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return np.nan
        val = val[0]
    # Now val should be scalar
    if isinstance(val, (int, float, np.integer, np.floating)):
        v = float(val)
        return np.nan if np.isnan(v) else v
    try:
        if pd.isna(val):
            return np.nan
    except (ValueError, TypeError):
        pass
    try:
        cleaned = str(val).split()[0].rstrip("m")
        return float(cleaned)
    except (ValueError, IndexError):
        return np.nan


def _parse_speed(val):
    """Parse speed limit from OSM (handles '40', '40 mph', ['40','30'], arrays, etc.)."""
    if val is None:
        return np.nan
    # Handle numpy arrays and lists first
    if isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return np.nan
        val = val[0]
    if isinstance(val, (int, float, np.integer, np.floating)):
        v = float(val)
        return np.nan if np.isnan(v) else v
    try:
        if pd.isna(val):
            return np.nan
    except (ValueError, TypeError):
        pass
    try:
        val_str = str(val)
        cleaned = val_str.lower().replace("mph", "").replace("kmph", "").replace("km/h", "").strip()
        speed = float(cleaned)
        if "mph" in val_str.lower():
            speed *= 1.609
        return speed
    except (ValueError, IndexError):
        return np.nan


# ============================================
# STEP 3: Compute network centrality
# ============================================
def compute_centrality(G, edges_gdf, k_samples=100):
    """Compute approximate betweenness centrality for edges."""
    logger.info(f"Computing edge betweenness centrality (k={k_samples} samples) ...")

    # Check for cached centrality
    cache_path = DATA_DIR / "edge_betweenness_centrality.pkl"
    if cache_path.exists():
        logger.info(f"  Loading cached centrality from {cache_path} ...")
        with open(cache_path, "rb") as f:
            edge_bc = pickle.load(f)
        logger.info(f"  Loaded centrality for {len(edge_bc):,} edges")
    else:
        logger.info("  This may take 5-10 minutes for a city-scale network ...")

        # Use approximate betweenness centrality (much faster than exact)
        edge_bc = nx.edge_betweenness_centrality(
            G, k=k_samples, weight="length", normalized=True
        )

        # Cache for reuse
        with open(cache_path, "wb") as f:
            pickle.dump(edge_bc, f)
        logger.info(f"  Cached centrality to {cache_path}")

    logger.info(f"  Computed centrality for {len(edge_bc):,} edges")

    # Map centrality values to edges_gdf
    # For MultiDiGraph, edge_bc keys can be (u, v) or (u, v, key)
    bc_dict = {}
    for edge_key, bc_val in edge_bc.items():
        if len(edge_key) == 3:
            u, v, k = edge_key
            bc_dict[(u, v, k)] = bc_val
        elif len(edge_key) == 2:
            u, v = edge_key
            bc_dict[(u, v)] = bc_val
        else:
            continue

    # Match to edges_gdf index (which is (u, v, key))
    def _lookup_bc(idx):
        # Try exact 3-tuple match first
        val = bc_dict.get(idx, None)
        if val is not None:
            return val
        # Try 2-tuple match (u, v)
        val = bc_dict.get((idx[0], idx[1]), None)
        if val is not None:
            return val
        return 0.0

    edges_gdf["betweenness_centrality_raw"] = [
        _lookup_bc(idx) for idx in edges_gdf.index
    ]

    # Normalize to [0, 1] range
    bc_max = edges_gdf["betweenness_centrality_raw"].max()
    if bc_max > 0:
        edges_gdf["betweenness_centrality"] = (
            edges_gdf["betweenness_centrality_raw"] / bc_max
        )
    else:
        edges_gdf["betweenness_centrality"] = 0.0

    logger.info(
        f"  Centrality stats: mean={edges_gdf['betweenness_centrality'].mean():.4f}, "
        f"max={edges_gdf['betweenness_centrality'].max():.4f}, "
        f"p95={edges_gdf['betweenness_centrality'].quantile(0.95):.4f}"
    )

    return edges_gdf


# ============================================
# STEP 4: Map-match violations to road segments
# ============================================
def map_match_violations(G, edges_gdf):
    """Snap each violation GPS point to the nearest road edge."""
    logger.info("Map-matching violations to nearest road segments ...")

    # Load cleaned violations
    df = pd.read_parquet(CLEANED_PARQUET)
    logger.info(f"  Loaded {len(df):,} violations")

    # Get violation coordinates
    lats = df["latitude"].values
    lons = df["longitude"].values

    # Find nearest edges using OSMnx (uses BallTree — fast!)
    logger.info("  Finding nearest edges (BallTree spatial index) ...")
    nearest_edges = ox.distance.nearest_edges(G, lons, lats)
    # Returns array of (u, v, key) tuples
    logger.info(f"  Matched {len(nearest_edges):,} violations to road segments")

    # Extract u, v, key from results
    u_nodes = np.array([e[0] for e in nearest_edges])
    v_nodes = np.array([e[1] for e in nearest_edges])
    keys = np.array([e[2] for e in nearest_edges])

    df["matched_u"] = u_nodes
    df["matched_v"] = v_nodes
    df["matched_key"] = keys

    # --- Build lookup dict from edges_gdf for fast enrichment ---
    logger.info("  Enriching violations with road attributes ...")

    # Columns to transfer from road data
    road_cols = [
        "highway_class",
        "lanes_final",
        "maxspeed_final",
        "width_meters",
        "is_oneway",
        "is_main_road",
        "length_m",
        "betweenness_centrality",
    ]

    # Build a lookup dictionary keyed by (u, v, key)
    edge_lookup = {}
    for idx, row in edges_gdf[road_cols].iterrows():
        edge_lookup[idx] = row.to_dict()

    # Map road attributes to violations
    road_attrs = []
    for u, v, k in zip(u_nodes, v_nodes, keys):
        attrs = edge_lookup.get((u, v, k), {})
        if not attrs:
            # Try reverse direction for undirected consideration
            attrs = edge_lookup.get((v, u, k), {})
        if not attrs:
            # Use defaults
            attrs = {
                "highway_class": "unclassified",
                "lanes_final": 2,
                "maxspeed_final": 40,
                "width_meters": 7.0,
                "is_oneway": False,
                "is_main_road": False,
                "length_m": 100.0,
                "betweenness_centrality": 0.0,
            }
        road_attrs.append(attrs)

    road_df = pd.DataFrame(road_attrs)

    # Rename for clarity
    road_df = road_df.rename(columns={
        "lanes_final": "road_lanes",
        "maxspeed_final": "road_maxspeed_kmph",
        "width_meters": "road_width_m",
        "is_oneway": "road_is_oneway",
        "is_main_road": "road_is_main",
        "length_m": "road_length_m",
        "betweenness_centrality": "road_betweenness_centrality",
    })

    # Attach to violations dataframe
    for col in road_df.columns:
        df[col] = road_df[col].values

    logger.info("  Road enrichment complete!")

    return df


# ============================================
# STEP 5: Save outputs
# ============================================
def save_outputs(edges_gdf, enriched_df):
    """Save road edges and enriched violations to parquet."""
    # Save road edges (drop geometry for parquet, keep as separate geojson)
    logger.info(f"Saving road edges to {ROAD_EDGES_PARQUET} ...")
    save_cols = [
        "highway_class", "lanes_final", "maxspeed_final", "width_meters",
        "is_oneway", "is_main_road", "length_m",
        "betweenness_centrality",
    ]
    edges_save = edges_gdf[save_cols].reset_index()
    edges_save.to_parquet(ROAD_EDGES_PARQUET, index=False, engine="pyarrow")
    logger.info(f"  Saved {len(edges_save):,} road segments")

    # Save enriched violations
    logger.info(f"Saving enriched violations to {ENRICHED_PARQUET} ...")
    enriched_df.to_parquet(ENRICHED_PARQUET, index=False, engine="pyarrow")
    logger.info(f"  Saved {len(enriched_df):,} enriched violations")


# ============================================
# STEP 6: Print summary
# ============================================
def print_summary(edges_gdf, enriched_df):
    """Print road network and enrichment summary."""
    print("\n" + "=" * 70)
    print("  PARKVISION AI - Road Network & Map-Matching Summary")
    print("=" * 70)

    print(f"\n  --- Road Network ---")
    print(f"  Total road segments:     {len(edges_gdf):,}")
    print(f"  Main roads:              {edges_gdf['is_main_road'].sum():,} ({edges_gdf['is_main_road'].mean()*100:.1f}%)")
    total_km = edges_gdf["length_m"].sum() / 1000
    print(f"  Total road length:       {total_km:,.0f} km")

    print(f"\n  --- Highway Class Distribution ---")
    hc_counts = edges_gdf["highway_class"].value_counts()
    for hc, cnt in hc_counts.head(10).items():
        print(f"    {hc:25s} {cnt:>7,}")

    print(f"\n  --- Road Width Stats ---")
    print(f"    Mean:   {edges_gdf['width_meters'].mean():.1f} m")
    print(f"    Median: {edges_gdf['width_meters'].median():.1f} m")
    print(f"    Min:    {edges_gdf['width_meters'].min():.1f} m")
    print(f"    Max:    {edges_gdf['width_meters'].max():.1f} m")

    print(f"\n  --- Centrality Stats ---")
    print(f"    Mean:   {edges_gdf['betweenness_centrality'].mean():.4f}")
    print(f"    p90:    {edges_gdf['betweenness_centrality'].quantile(0.90):.4f}")
    print(f"    p99:    {edges_gdf['betweenness_centrality'].quantile(0.99):.4f}")

    print(f"\n  --- Enriched Violations ---")
    print(f"  Total records:           {len(enriched_df):,}")
    print(f"  Matched to main roads:   {enriched_df['road_is_main'].sum():,} ({enriched_df['road_is_main'].mean()*100:.1f}%)")
    print(f"  Avg road width at violation: {enriched_df['road_width_m'].mean():.1f} m")
    print(f"  Avg road lanes:          {enriched_df['road_lanes'].mean():.1f}")
    print(f"  Avg centrality at violation: {enriched_df['road_betweenness_centrality'].mean():.4f}")

    print(f"\n  --- Top Highway Classes at Violation Sites ---")
    vc = enriched_df["highway_class"].value_counts()
    for hc, cnt in vc.head(8).items():
        pct = cnt / len(enriched_df) * 100
        print(f"    {hc:25s} {cnt:>7,}  ({pct:.1f}%)")

    print(f"\n  Outputs:")
    print(f"    {ROAD_EDGES_PARQUET}")
    print(f"    {ENRICHED_PARQUET}")
    print("=" * 70)


# ============================================
# MAIN PIPELINE
# ============================================
def run():
    """Execute road network extraction and map-matching pipeline."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download road network
    G = download_road_network()

    # Step 2: Extract edge attributes
    G, nodes_gdf, edges_gdf = extract_edge_attributes(G)

    # Step 3: Compute centrality (k=100 for 673K-edge graph; k=300 would take 30+ min)
    edges_gdf = compute_centrality(G, edges_gdf, k_samples=100)

    # Step 4: Map-match violations
    enriched_df = map_match_violations(G, edges_gdf)

    # Step 5: Save outputs
    save_outputs(edges_gdf, enriched_df)

    # Step 6: Summary
    print_summary(edges_gdf, enriched_df)

    return edges_gdf, enriched_df


if __name__ == "__main__":
    run()
