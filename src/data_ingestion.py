"""
PARKVISION AI — Stage 1: Data Ingestion & Cleaning
====================================================
Loads raw CSV, cleans, parses, engineers features, saves to parquet.

Usage:
    python -m src.data_ingestion
"""

import sys
import json
import logging
import ast
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    RAW_CSV_PATH,
    CLEANED_PARQUET,
    DATA_DIR,
    BENGALURU_BBOX,
    VIOLATION_SEVERITY_WEIGHTS,
    DEFAULT_VIOLATION_WEIGHT,
    VEHICLE_FOOTPRINTS,
    DEFAULT_VEHICLE_FOOTPRINT,
    WEEKDAY_DEMAND_CURVE,
    WEEKEND_MULTIPLIER,
    LOG_FORMAT,
    LOG_LEVEL,
)

# Setup logging
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger("data_ingestion")


# ============================================
# STEP 1: Load raw CSV
# ============================================
def load_raw_csv(path: Path) -> pd.DataFrame:
    """Load the raw police violation CSV."""
    logger.info(f"Loading raw CSV from {path} ...")
    df = pd.read_csv(
        path,
        dtype={
            "id": str,
            "vehicle_number": str,
            "vehicle_type": str,
            "description": str,
            "violation_type": str,
            "offence_code": str,
            "device_id": str,
            "created_by_id": str,
            "police_station": str,
            "junction_name": str,
            "updated_vehicle_number": str,
            "updated_vehicle_type": str,
            "validation_status": str,
        },
    )
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


# ============================================
# STEP 2: Parse & clean
# ============================================
def parse_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Parse datetimes, filter, deduplicate, validate GPS."""
    logger.info("Parsing and cleaning ...")

    # --- 2a: Parse datetimes (UTC+5:30 = IST) ---
    logger.info("  Parsing created_datetime ...")
    df["created_datetime"] = pd.to_datetime(
        df["created_datetime"], format="mixed", utc=True
    )
    # Convert to IST for analysis
    df["created_datetime_ist"] = df["created_datetime"].dt.tz_convert("Asia/Kolkata")

    logger.info("  Parsing modified_datetime ...")
    df["modified_datetime"] = pd.to_datetime(
        df["modified_datetime"], format="mixed", utc=True, errors="coerce"
    )

    # --- 2b: Parse violation_type from JSON string arrays ---
    logger.info("  Parsing violation_type JSON arrays ...")
    df["violation_types_list"] = df["violation_type"].apply(_parse_json_list)

    # --- 2c: Parse offence_code from JSON string arrays ---
    logger.info("  Parsing offence_code arrays ...")
    df["offence_codes_list"] = df["offence_code"].apply(_parse_offence_codes)

    # Count of violations per record
    df["violation_count"] = df["violation_types_list"].apply(len)

    # --- 2d: Filter validation_status ---
    total_before = len(df)
    # Keep 'approved' and also NaN (not yet validated) — reject 'rejected' and 'duplicate'
    valid_statuses = {"approved", "created1", "processing"}
    df["is_approved"] = df["validation_status"].isin(valid_statuses) | df[
        "validation_status"
    ].isna()
    df = df[df["is_approved"]].copy()
    logger.info(
        f"  Filtered validation_status: {total_before:,} -> {len(df):,} "
        f"(removed {total_before - len(df):,} rejected/duplicate)"
    )

    # --- 2e: Remove duplicate records ---
    total_before = len(df)
    # Round lat/lon to 5 decimal places (~1m precision) for dedup
    df["lat_round"] = df["latitude"].round(5)
    df["lon_round"] = df["longitude"].round(5)
    df = df.drop_duplicates(
        subset=["device_id", "created_datetime", "lat_round", "lon_round"], keep="first"
    )
    df = df.drop(columns=["lat_round", "lon_round"])
    logger.info(
        f"  Deduplicated: {total_before:,} -> {len(df):,} "
        f"(removed {total_before - len(df):,} duplicates)"
    )

    # --- 2f: Validate GPS within Bengaluru bounding box ---
    total_before = len(df)
    bbox = BENGALURU_BBOX
    gps_valid = (
        (df["latitude"] >= bbox["lat_min"])
        & (df["latitude"] <= bbox["lat_max"])
        & (df["longitude"] >= bbox["lon_min"])
        & (df["longitude"] <= bbox["lon_max"])
    )
    df = df[gps_valid].copy()
    logger.info(
        f"  GPS validation: {total_before:,} -> {len(df):,} "
        f"(removed {total_before - len(df):,} out-of-bounds)"
    )

    return df


def _parse_json_list(val):
    """Parse a JSON array string like '["WRONG PARKING","NO PARKING"]' into a list."""
    if pd.isna(val) or val == "NULL":
        return []
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed]
        return [str(parsed).strip()]
    except (json.JSONDecodeError, TypeError):
        return [str(val).strip()]


def _parse_offence_codes(val):
    """Parse offence_code string like '[112,104]' into a list of ints."""
    if pd.isna(val) or val == "NULL":
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [int(c) for c in parsed]
        return [int(parsed)]
    except (ValueError, SyntaxError):
        return []


# ============================================
# STEP 3: Feature Engineering
# ============================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns for analysis."""
    logger.info("Engineering features ...")

    dt = df["created_datetime_ist"]

    # --- 3a: Temporal features ---
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek  # 0=Mon, 6=Sun
    df["day_name"] = dt.dt.day_name()
    df["month"] = dt.dt.month
    df["date"] = dt.dt.date
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["is_peak_hour"] = df["hour"].isin([8, 9, 17, 18, 19])

    logger.info("  Added temporal features (hour, day, month, is_weekend, is_peak)")

    # --- 3b: Violation severity weight ---
    # For multi-code violations, take the MAX severity (worst case)
    df["violation_severity_weight"] = df["offence_codes_list"].apply(
        lambda codes: max(
            (VIOLATION_SEVERITY_WEIGHTS.get(c, DEFAULT_VIOLATION_WEIGHT) for c in codes),
            default=DEFAULT_VIOLATION_WEIGHT,
        )
    )

    # Also compute the AVERAGE severity for the composite score
    df["violation_severity_avg"] = df["offence_codes_list"].apply(
        lambda codes: np.mean(
            [VIOLATION_SEVERITY_WEIGHTS.get(c, DEFAULT_VIOLATION_WEIGHT) for c in codes]
        )
        if codes
        else DEFAULT_VIOLATION_WEIGHT
    )

    # Primary offence code (first in the list)
    df["primary_offence_code"] = df["offence_codes_list"].apply(
        lambda x: x[0] if x else None
    )

    logger.info("  Added violation severity weights (max, avg, primary)")

    # --- 3c: Vehicle footprint dimensions ---
    df["vehicle_footprint_width"] = df["vehicle_type"].map(
        lambda vt: VEHICLE_FOOTPRINTS.get(vt, DEFAULT_VEHICLE_FOOTPRINT)[0]
    )
    df["vehicle_footprint_length"] = df["vehicle_type"].map(
        lambda vt: VEHICLE_FOOTPRINTS.get(vt, DEFAULT_VEHICLE_FOOTPRINT)[1]
    )

    logger.info("  Added vehicle footprint dimensions (width, length)")

    # --- 3d: Junction-related features ---
    df["has_junction"] = (
        df["junction_name"].notna()
        & (df["junction_name"] != "No Junction")
        & (df["junction_name"].str.strip() != "")
    )
    # Extract BTP junction code where available
    df["junction_code"] = df["junction_name"].apply(_extract_junction_code)

    logger.info("  Added junction features (has_junction, junction_code)")

    # --- 3e: Temporal demand multiplier ---
    df["temporal_demand_multiplier"] = df.apply(
        lambda row: _get_temporal_multiplier(row["hour"], row["is_weekend"]), axis=1
    )

    logger.info("  Added temporal demand multiplier")

    # --- 3f: Convenience boolean flags for violation types ---
    vt_series = df["violation_types_list"]
    df["is_main_road_parking"] = vt_series.apply(
        lambda x: "PARKING IN A MAIN ROAD" in x
    )
    df["is_double_parking"] = vt_series.apply(lambda x: "DOUBLE PARKING" in x)
    df["is_no_parking"] = vt_series.apply(lambda x: "NO PARKING" in x)
    df["is_wrong_parking"] = vt_series.apply(lambda x: "WRONG PARKING" in x)
    df["is_near_busstop"] = vt_series.apply(
        lambda x: any("BUSTOP" in v or "SCHOOL" in v or "HOSPITAL" in v for v in x)
    )

    logger.info("  Added violation type boolean flags")

    return df


def _extract_junction_code(val):
    """Extract BTP junction code like 'BTP044' from 'BTP044 - Sagar Theatre Junction'."""
    if pd.isna(val) or val == "No Junction":
        return None
    val = str(val).strip()
    if val.startswith("BTP"):
        parts = val.split(" - ", 1)
        return parts[0].strip()
    return None


def _get_temporal_multiplier(hour: int, is_weekend: bool) -> float:
    """Get temporal demand multiplier for a given hour and weekend flag."""
    multiplier = DEFAULT_VIOLATION_WEIGHT  # fallback
    for (h_start, h_end), mult in WEEKDAY_DEMAND_CURVE.items():
        if h_start <= hour < h_end:
            multiplier = mult
            break
    if is_weekend:
        multiplier *= WEEKEND_MULTIPLIER
    return round(multiplier, 3)


# ============================================
# STEP 4: Summary statistics
# ============================================
def print_summary(df: pd.DataFrame):
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 70)
    print("  PARKVISION AI - Data Ingestion Summary")
    print("=" * 70)

    print(f"\n  Total cleaned records:   {len(df):,}")
    print(f"  Date range:              {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique vehicles:         {df['vehicle_number'].nunique():,}")
    print(f"  Unique devices:          {df['device_id'].nunique():,}")
    print(f"  Police stations:         {df['police_station'].nunique()}")
    print(f"  Named junctions:         {df[df['has_junction']]['junction_code'].nunique()}")

    print(f"\n  --- Vehicle Type Distribution ---")
    vt_counts = df["vehicle_type"].value_counts()
    for vt, cnt in vt_counts.head(10).items():
        pct = cnt / len(df) * 100
        print(f"    {vt:25s} {cnt:>7,}  ({pct:.1f}%)")

    print(f"\n  --- Top 10 Police Stations ---")
    ps_counts = df["police_station"].value_counts()
    for ps, cnt in ps_counts.head(10).items():
        pct = cnt / len(df) * 100
        print(f"    {ps:25s} {cnt:>7,}  ({pct:.1f}%)")

    print(f"\n  --- Violation Severity Distribution ---")
    print(f"    Mean severity weight:  {df['violation_severity_weight'].mean():.3f}")
    print(f"    Median:                {df['violation_severity_weight'].median():.3f}")
    print(f"    Critical (>= 0.8):     {(df['violation_severity_weight'] >= 0.8).sum():,}")
    print(f"    High (0.6-0.8):        {((df['violation_severity_weight'] >= 0.6) & (df['violation_severity_weight'] < 0.8)).sum():,}")

    print(f"\n  --- Temporal Distribution ---")
    print(f"    Peak hour violations:  {df['is_peak_hour'].sum():,}  ({df['is_peak_hour'].mean()*100:.1f}%)")
    print(f"    Weekend violations:    {df['is_weekend'].sum():,}  ({df['is_weekend'].mean()*100:.1f}%)")
    print(f"    At junctions (BTP):    {df['has_junction'].sum():,}  ({df['has_junction'].mean()*100:.1f}%)")

    print(f"\n  --- Violation Type Flags ---")
    for flag in ["is_main_road_parking", "is_double_parking", "is_no_parking",
                 "is_wrong_parking", "is_near_busstop"]:
        cnt = df[flag].sum()
        print(f"    {flag:30s} {cnt:>7,}  ({cnt/len(df)*100:.1f}%)")

    print(f"\n  --- Vehicle Footprint Stats ---")
    print(f"    Mean width:   {df['vehicle_footprint_width'].mean():.2f}m")
    print(f"    Mean length:  {df['vehicle_footprint_length'].mean():.2f}m")

    print(f"\n  Output: {CLEANED_PARQUET}")
    print("=" * 70)


# ============================================
# MAIN PIPELINE
# ============================================
def run():
    """Execute the full data ingestion pipeline."""
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load
    df = load_raw_csv(RAW_CSV_PATH)

    # Step 2: Parse & Clean
    df = parse_and_clean(df)

    # Step 3: Feature Engineering
    df = engineer_features(df)

    # Step 4: Select final columns and save
    # Convert list columns to strings for parquet compatibility
    df["violation_types_str"] = df["violation_types_list"].apply(
        lambda x: "|".join(x) if x else ""
    )
    df["offence_codes_str"] = df["offence_codes_list"].apply(
        lambda x: "|".join(str(c) for c in x) if x else ""
    )

    # Drop intermediate columns
    cols_to_drop = [
        "is_approved",
        "description",
        "closed_datetime",
        "action_taken_timestamp",
        "data_sent_to_scita_timestamp",
        "violation_types_list",
        "offence_codes_list",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Save to parquet
    logger.info(f"Saving {len(df):,} records to {CLEANED_PARQUET} ...")
    df.to_parquet(CLEANED_PARQUET, index=False, engine="pyarrow")
    logger.info("  Saved successfully!")

    # Print summary
    print_summary(df)

    return df


if __name__ == "__main__":
    run()
