"""
PARKVISION AI — Stage 6: PCIS Engine (Capacity Reduction & Proximity Factors)
================================================================================
Computes the 5 components of the Parking Congestion Impact Score (PCIS):
  1. CapacityReduction — physical lane obstruction per violation
  2. ProximityFactor — junction & road hierarchy sensitivity
  3. TemporalDemandMultiplier — time-of-day traffic demand (already computed in ingestion)
  4. VehicleObstructionFactor — vehicle type severity
  5. NetworkCriticality — betweenness centrality (already computed in road_network)

Tasks 7 covers components 1-3. Task 8 will combine all into final PCIS.

Usage:
    python -m src.pcis_engine
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR,
    ENRICHED_PARQUET,
    PCIS_SCORED_PARQUET,
    STANDARD_LANE_WIDTH_M,
    DEFAULT_ROAD_WIDTH_M,
    DEFAULT_LANES,
    LANE_CAPACITY_VPH,
    VEHICLE_FOOTPRINTS,
    DEFAULT_VEHICLE_FOOTPRINT,
    PROXIMITY_FACTORS,
    MAIN_ROAD_HIGHWAY_CLASSES,
    RESIDENTIAL_HIGHWAY_CLASSES,
    WEEKDAY_DEMAND_CURVE,
    WEEKEND_MULTIPLIER,
    VIOLATION_SEVERITY_WEIGHTS,
    DEFAULT_VIOLATION_WEIGHT,
    PCIS_WEIGHTS,
    PCIS_TIERS,
    LOG_FORMAT,
    LOG_LEVEL,
)

warnings.filterwarnings("ignore")
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger("pcis_engine")


# ============================================
# COMPONENT 1: Capacity Reduction
# ============================================
def compute_capacity_reduction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model each violation as a physical lane obstruction.
    
    CapacityReduction = 1 - (effective_width / design_width)
    where effective_width = design_width - vehicle_footprint_width
    
    For multi-violation road segments, we aggregate footprints
    within the same H3 hex and time window.
    """
    logger.info("Computing Component 1: Capacity Reduction ...")

    # Design width = road width from OSM data
    design_width = df["road_width_m"].clip(lower=3.0)  # minimum 3m road

    # Vehicle footprint width (already in df from ingestion)
    vehicle_width = df["vehicle_footprint_width"]

    # --- Single-violation capacity reduction ---
    effective_width = (design_width - vehicle_width).clip(lower=0.0)
    single_cr = 1.0 - (effective_width / design_width)
    df["capacity_reduction_single"] = single_cr.clip(0.0, 1.0)

    # --- Multi-violation aggregation ---
    # Count concurrent violations on same road segment (same H3 hex + same hour)
    # This models the "phantom lane closure" effect from the README
    concurrent = df.groupby(["h3_index", "hour"]).agg(
        concurrent_count=("vehicle_footprint_width", "size"),
        total_footprint_width=("vehicle_footprint_width", "sum"),
    ).reset_index()

    df = df.merge(concurrent, on=["h3_index", "hour"], how="left")

    # Average concurrent violations per hour at this location
    # Divide by unique_days estimate to get typical concurrency
    hex_days = df.groupby("h3_index")["date"].transform("nunique").clip(lower=1)
    df["avg_concurrent"] = (df["concurrent_count"] / hex_days).clip(lower=1.0)

    # Multi-violation effective width: design_width - (avg_concurrent × avg_vehicle_width)
    avg_vehicle_width_in_hex = df.groupby(["h3_index", "hour"])[
        "vehicle_footprint_width"
    ].transform("mean")

    multi_footprint = df["avg_concurrent"] * avg_vehicle_width_in_hex
    effective_width_multi = (design_width - multi_footprint).clip(lower=0.0)
    df["capacity_reduction"] = (1.0 - effective_width_multi / design_width).clip(0.0, 1.0)

    # --- Capacity loss in vehicles per hour ---
    road_lanes = df["road_lanes"].clip(lower=1)
    original_capacity = road_lanes * LANE_CAPACITY_VPH
    remaining_lanes = (effective_width_multi / STANDARD_LANE_WIDTH_M).clip(lower=0.0)
    remaining_capacity = remaining_lanes * LANE_CAPACITY_VPH
    df["capacity_loss_vph"] = (original_capacity - remaining_capacity).clip(lower=0.0)

    logger.info(
        f"  Mean capacity reduction: {df['capacity_reduction'].mean():.3f} "
        f"(single: {df['capacity_reduction_single'].mean():.3f})"
    )
    logger.info(
        f"  Mean capacity loss: {df['capacity_loss_vph'].mean():.0f} veh/hr"
    )
    logger.info(
        f"  Critical (CR > 0.5): {(df['capacity_reduction'] > 0.5).sum():,} "
        f"({(df['capacity_reduction'] > 0.5).mean()*100:.1f}%)"
    )

    return df


# ============================================
# COMPONENT 2: Proximity Factor
# ============================================
def compute_proximity_factor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute proximity factor based on junction presence and road hierarchy.
    
    Higher score = violation at more congestion-sensitive location:
      1.0 — at a signalized BTP junction
      0.8 — near junction (main road)
      0.6 — main road, away from junction
      0.3 — secondary/tertiary road
      0.1 — residential/service road
    """
    logger.info("Computing Component 2: Proximity Factor ...")

    def _proximity(row):
        has_junc = row["has_junction"]
        highway = row["highway_class"]
        is_main = row["road_is_main"]

        # At a BTP junction
        if has_junc:
            if is_main:
                return 1.0   # Junction on main road — maximum impact
            else:
                return 0.8   # Junction on secondary road

        # On a main road without junction
        if is_main:
            return 0.6

        # Tertiary roads
        if highway in ("tertiary", "tertiary_link"):
            return 0.4

        # Secondary without junction
        if highway in ("secondary", "secondary_link"):
            return 0.5

        # Residential / service
        if highway in RESIDENTIAL_HIGHWAY_CLASSES:
            return 0.1

        # Default (unclassified, other)
        return 0.3

    df["proximity_factor"] = df.apply(_proximity, axis=1)

    logger.info(f"  Mean proximity factor: {df['proximity_factor'].mean():.3f}")
    logger.info(f"  At junction (1.0/0.8): {(df['proximity_factor'] >= 0.8).sum():,} "
                f"({(df['proximity_factor'] >= 0.8).mean()*100:.1f}%)")
    logger.info(f"  Main road (0.6):       {(df['proximity_factor'] == 0.6).sum():,}")
    logger.info(f"  Residential (0.1):     {(df['proximity_factor'] == 0.1).sum():,}")

    return df


# ============================================
# COMPONENT 3: Temporal Demand Multiplier
# ============================================
def validate_temporal_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate/recompute the temporal demand multiplier.
    Already computed in data_ingestion but verify and enhance.
    """
    logger.info("Validating Component 3: Temporal Demand Multiplier ...")

    if "temporal_demand_multiplier" not in df.columns:
        logger.info("  Computing temporal demand multiplier (not found in data) ...")
        df["temporal_demand_multiplier"] = df.apply(
            lambda r: _get_temporal_multiplier(r["hour"], r["is_weekend"]),
            axis=1,
        )
    else:
        logger.info("  Temporal demand multiplier already present from ingestion stage")

    logger.info(f"  Mean temporal multiplier: {df['temporal_demand_multiplier'].mean():.3f}")
    logger.info(f"  Peak hour (1.0):         {(df['temporal_demand_multiplier'] >= 0.9).sum():,}")
    logger.info(f"  Off-peak (< 0.3):        {(df['temporal_demand_multiplier'] < 0.3).sum():,}")

    return df


def _get_temporal_multiplier(hour, is_weekend):
    """Compute temporal demand multiplier."""
    mult = 0.3  # default
    for (h_start, h_end), m in WEEKDAY_DEMAND_CURVE.items():
        if h_start <= hour < h_end:
            mult = m
            break
    if is_weekend:
        mult *= WEEKEND_MULTIPLIER
    return round(mult, 3)


# ============================================
# COMPONENT 4: Vehicle Obstruction Factor
# ============================================
def compute_vehicle_obstruction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute vehicle obstruction factor from violation severity.
    Uses the violation_severity_weight already computed in ingestion.
    Normalizes to [0, 1] range.
    """
    logger.info("Computing Component 4: Vehicle Obstruction Factor ...")

    # Use the max severity weight (already computed as violation_severity_weight)
    df["vehicle_obstruction_factor"] = df["violation_severity_weight"].clip(0.0, 1.0)

    logger.info(f"  Mean obstruction factor: {df['vehicle_obstruction_factor'].mean():.3f}")

    return df


# ============================================
# COMPONENT 5: Network Criticality
# ============================================
def compute_network_criticality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize betweenness centrality to [0, 1] as network criticality score.
    Already computed in road_network stage, just normalize.
    """
    logger.info("Computing Component 5: Network Criticality ...")

    bc = df["road_betweenness_centrality"].fillna(0.0)

    # Normalize to [0, 1] — already normalized in road_network but verify
    bc_max = bc.max()
    if bc_max > 0:
        df["network_criticality"] = (bc / bc_max).clip(0.0, 1.0)
    else:
        df["network_criticality"] = 0.0

    logger.info(f"  Mean network criticality: {df['network_criticality'].mean():.4f}")
    logger.info(f"  High criticality (> 0.5): {(df['network_criticality'] > 0.5).sum():,}")

    return df


# ============================================
# COMPOSITE PCIS SCORE
# ============================================
def compute_pcis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the composite Parking Congestion Impact Score.
    
    PCIS = w1×CapacityReduction + w2×ProximityFactor + w3×TemporalDemand 
         + w4×VehicleObstruction + w5×NetworkCriticality
    """
    logger.info("Computing composite PCIS score ...")

    w = PCIS_WEIGHTS
    df["pcis"] = (
        w["capacity_reduction"] * df["capacity_reduction"]
        + w["proximity_factor"] * df["proximity_factor"]
        + w["temporal_demand_multiplier"] * df["temporal_demand_multiplier"]
        + w["vehicle_obstruction_factor"] * df["vehicle_obstruction_factor"]
        + w["network_criticality"] * df["network_criticality"]
    ).clip(0.0, 1.0)

    # Classify into tiers
    df["pcis_tier"] = df["pcis"].apply(_classify_pcis_tier)

    # Log distribution
    logger.info(f"  Mean PCIS: {df['pcis'].mean():.3f}")
    logger.info(f"  Median:    {df['pcis'].median():.3f}")
    logger.info(f"  Std:       {df['pcis'].std():.3f}")

    tier_counts = df["pcis_tier"].value_counts().sort_index()
    for tier, cnt in tier_counts.items():
        pct = cnt / len(df) * 100
        logger.info(f"    {tier:15s}: {cnt:>7,} ({pct:.1f}%)")

    return df


def _classify_pcis_tier(score):
    """Classify PCIS score into a tier."""
    for tier_name, (low, high) in PCIS_TIERS.items():
        if low <= score < high:
            return tier_name
    return "catastrophic"  # score == 1.0


# ============================================
# AGGREGATE TO CLUSTER LEVEL
# ============================================
def aggregate_pcis_to_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PCIS scores to H3 hexagon and cluster level."""
    logger.info("Aggregating PCIS to hexagon level ...")

    hex_pcis = df.groupby("h3_index").agg(
        pcis_mean=("pcis", "mean"),
        pcis_max=("pcis", "max"),
        pcis_sum=("pcis", "sum"),
        pcis_p90=("pcis", lambda x: x.quantile(0.9)),
        capacity_reduction_mean=("capacity_reduction", "mean"),
        proximity_factor_mean=("proximity_factor", "mean"),
        temporal_demand_mean=("temporal_demand_multiplier", "mean"),
        obstruction_mean=("vehicle_obstruction_factor", "mean"),
        criticality_mean=("network_criticality", "mean"),
        violation_count=("pcis", "size"),
    ).reset_index()

    hex_pcis["pcis_tier"] = hex_pcis["pcis_mean"].apply(_classify_pcis_tier)

    logger.info(f"  Hexagon PCIS tiers:")
    for tier, cnt in hex_pcis["pcis_tier"].value_counts().sort_index().items():
        logger.info(f"    {tier:15s}: {cnt:>5} hexagons")

    return hex_pcis


# ============================================
# SAVE & SUMMARIZE
# ============================================
def save_and_summarize(df, hex_pcis):
    """Save outputs and print comprehensive summary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save PCIS-scored violations
    logger.info(f"Saving PCIS-scored violations to {PCIS_SCORED_PARQUET} ...")
    df.to_parquet(PCIS_SCORED_PARQUET, index=False, engine="pyarrow")

    # Save hex-level PCIS
    hex_pcis_path = DATA_DIR / "h3_pcis_scores.parquet"
    hex_pcis.to_parquet(hex_pcis_path, index=False, engine="pyarrow")

    print("\n" + "=" * 70)
    print("  PARKVISION AI - PCIS Scoring Summary")
    print("=" * 70)

    # Component breakdown
    print(f"\n  --- PCIS Component Statistics ---")
    components = [
        ("Capacity Reduction (w=0.30)", "capacity_reduction"),
        ("Proximity Factor   (w=0.20)", "proximity_factor"),
        ("Temporal Demand     (w=0.20)", "temporal_demand_multiplier"),
        ("Vehicle Obstruction (w=0.15)", "vehicle_obstruction_factor"),
        ("Network Criticality (w=0.15)", "network_criticality"),
    ]
    print(f"    {'Component':>35s}  {'Mean':>6}  {'Med':>6}  {'Std':>6}  {'Max':>6}")
    for name, col in components:
        print(f"    {name:>35s}  {df[col].mean():>6.3f}  {df[col].median():>6.3f}  "
              f"{df[col].std():>6.3f}  {df[col].max():>6.3f}")

    # PCIS distribution
    print(f"\n  --- Composite PCIS Distribution ---")
    print(f"    Mean:   {df['pcis'].mean():.3f}")
    print(f"    Median: {df['pcis'].median():.3f}")
    print(f"    p90:    {df['pcis'].quantile(0.9):.3f}")
    print(f"    p99:    {df['pcis'].quantile(0.99):.3f}")
    print(f"    Max:    {df['pcis'].max():.3f}")

    # Tier distribution
    print(f"\n  --- PCIS Tier Distribution ---")
    tier_counts = df["pcis_tier"].value_counts().sort_index()
    for tier, cnt in tier_counts.items():
        pct = cnt / len(df) * 100
        marker = {"low": "[LOW]", "moderate": "[MOD]", "high": "[HI!]", "critical": "[CRT]", "catastrophic": "[!!!]"}.get(tier, "")
        print(f"    {marker:5s} {tier:15s}: {cnt:>7,} violations ({pct:>5.1f}%)")

    # Top 20 highest-PCIS hotspot hexagons
    print(f"\n  --- Top 20 Highest-PCIS Hexagons ---")
    top = hex_pcis.nlargest(20, "pcis_mean")
    print(f"    {'H3 Index':>17}  {'PCIS':>6}  {'Max':>6}  {'Count':>6}  {'Cap.Red':>7}  {'Prox':>5}  {'Crit':>5}")
    for _, row in top.iterrows():
        print(f"    {row['h3_index']:>17}  {row['pcis_mean']:>6.3f}  {row['pcis_max']:>6.3f}  "
              f"{row['violation_count']:>6,}  {row['capacity_reduction_mean']:>7.3f}  "
              f"{row['proximity_factor_mean']:>5.2f}  {row['criticality_mean']:>5.3f}")

    # Capacity impact
    total_capacity_loss = df["capacity_loss_vph"].sum()
    print(f"\n  --- Estimated Capacity Impact ---")
    print(f"    Total capacity loss:    {total_capacity_loss:,.0f} veh-hr/hr")
    print(f"    Avg per violation:      {df['capacity_loss_vph'].mean():.0f} veh/hr")
    print(f"    Max single violation:   {df['capacity_loss_vph'].max():.0f} veh/hr")

    print(f"\n  Outputs:")
    print(f"    {PCIS_SCORED_PARQUET}")
    print(f"    {hex_pcis_path}")
    print("=" * 70)


# ============================================
# MAIN PIPELINE
# ============================================
def run():
    """Execute PCIS scoring pipeline."""
    # Load enriched violations
    logger.info(f"Loading enriched violations from {ENRICHED_PARQUET} ...")
    df = pd.read_parquet(ENRICHED_PARQUET)
    logger.info(f"  Loaded {len(df):,} records")

    # Component 1: Capacity Reduction
    df = compute_capacity_reduction(df)

    # Component 2: Proximity Factor
    df = compute_proximity_factor(df)

    # Component 3: Temporal Demand (validate existing)
    df = validate_temporal_demand(df)

    # Component 4: Vehicle Obstruction Factor
    df = compute_vehicle_obstruction(df)

    # Component 5: Network Criticality
    df = compute_network_criticality(df)

    # Composite PCIS
    df = compute_pcis(df)

    # Aggregate to hex level
    hex_pcis = aggregate_pcis_to_clusters(df)

    # Save & summarize
    save_and_summarize(df, hex_pcis)

    return df, hex_pcis


if __name__ == "__main__":
    run()
