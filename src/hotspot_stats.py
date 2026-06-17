"""
PARKVISION AI — Stage 5: Statistical Hotspot Validation & Temporal Decomposition
==================================================================================
Applies Getis-Ord Gi* to validate hotspots with statistical significance,
and decomposes temporal patterns using FFT and profile analysis.

Usage:
    python -m src.hotspot_stats
"""

import sys
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import fft as scipy_fft
from scipy.stats import norm
import libpysal
from esda.getisord import G_Local

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR,
    OUTPUT_DIR,
    H3_HEX_STATS_PARQUET,
    H3_HOTSPOT_SIG_PARQUET,
    TEMPORAL_PROFILES_JSON,
    ENRICHED_PARQUET,
    CLUSTER_PROFILES_PARQUET,
    LOG_FORMAT,
    LOG_LEVEL,
)

warnings.filterwarnings("ignore")
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger("hotspot_stats")


# ============================================
# STEP 1: Build spatial weights for H3 hexagons
# ============================================
def build_spatial_weights(hex_stats: pd.DataFrame):
    """Build distance-based spatial weights matrix for H3 hex centroids."""
    logger.info("Building spatial weights matrix for H3 hexagons ...")

    # Create point array from centroids
    points = list(zip(hex_stats["centroid_lon"].values, hex_stats["centroid_lat"].values))

    # Use KNN weights (k=6 neighbors — natural for hexagonal grids)
    w = libpysal.weights.KNN.from_array(
        np.column_stack([hex_stats["centroid_lon"].values, hex_stats["centroid_lat"].values]),
        k=6,
    )
    w.transform = "R"  # Row-standardize

    logger.info(f"  Built KNN(k=6) weights for {len(hex_stats):,} hexagons")
    logger.info(f"  Mean neighbors: {w.mean_neighbors:.1f}")

    return w


# ============================================
# STEP 2: Getis-Ord Gi* significance testing
# ============================================
def compute_getis_ord(hex_stats: pd.DataFrame, w) -> pd.DataFrame:
    """Compute Getis-Ord Gi* z-scores for violation counts."""
    logger.info("Computing Getis-Ord Gi* statistics ...")

    # Attribute: violation count
    y = hex_stats["violation_count"].values.astype(float)

    # Run Local G* (star version includes the focal unit)
    g_local = G_Local(y, w, star=True, permutations=999)

    # Extract results
    hex_stats = hex_stats.copy()
    hex_stats["gi_zscore"] = g_local.Zs
    hex_stats["gi_pvalue"] = g_local.p_sim

    # Classify significance
    hex_stats["hotspot_class"] = hex_stats.apply(
        lambda r: _classify_gi(r["gi_zscore"], r["gi_pvalue"]), axis=1
    )

    # Log results
    class_counts = hex_stats["hotspot_class"].value_counts()
    for cls, cnt in class_counts.items():
        total_violations = hex_stats[hex_stats["hotspot_class"] == cls]["violation_count"].sum()
        logger.info(f"    {cls:25s}: {cnt:>5} hexagons, {total_violations:>7,} violations")

    return hex_stats


def _classify_gi(z, p):
    """Classify hexagon based on Gi* z-score and p-value."""
    if p > 0.10:
        return "not_significant"
    if z > 0:
        if p <= 0.01:
            return "hotspot_99"       # 99% confidence
        elif p <= 0.05:
            return "hotspot_95"       # 95% confidence
        elif p <= 0.10:
            return "hotspot_90"       # 90% confidence
    else:
        if p <= 0.01:
            return "coldspot_99"
        elif p <= 0.05:
            return "coldspot_95"
        elif p <= 0.10:
            return "coldspot_90"
    return "not_significant"


# ============================================
# STEP 3: Multi-attribute Gi* (severity-weighted)
# ============================================
def compute_severity_gi(hex_stats: pd.DataFrame, w) -> pd.DataFrame:
    """Compute Gi* on severity-weighted violation count."""
    logger.info("Computing severity-weighted Gi* ...")

    # Create severity-weighted count
    y_sev = (hex_stats["violation_count"] * hex_stats["avg_severity"]).values.astype(float)

    g_local_sev = G_Local(y_sev, w, star=True, permutations=999)

    hex_stats["gi_severity_zscore"] = g_local_sev.Zs
    hex_stats["gi_severity_pvalue"] = g_local_sev.p_sim

    hex_stats["severity_hotspot_class"] = hex_stats.apply(
        lambda r: _classify_gi(r["gi_severity_zscore"], r["gi_severity_pvalue"]), axis=1
    )

    sev_hot = (hex_stats["severity_hotspot_class"].str.startswith("hotspot")).sum()
    logger.info(f"    Severity hotspots: {sev_hot} hexagons")

    return hex_stats


# ============================================
# STEP 4: Temporal profile analysis
# ============================================
def compute_temporal_profiles(df: pd.DataFrame, hex_stats: pd.DataFrame) -> dict:
    """Compute temporal profiles for significant hotspot hexagons."""
    logger.info("Computing temporal profiles for hotspot hexagons ...")

    # Focus on statistically significant hotspots
    hotspot_hexes = hex_stats[
        hex_stats["hotspot_class"].str.startswith("hotspot")
    ]["h3_index"].values

    logger.info(f"  Analyzing {len(hotspot_hexes):,} significant hotspot hexagons ...")

    # Filter violations to hotspot hexagons
    hot_df = df[df["h3_index"].isin(hotspot_hexes)].copy()
    logger.info(f"  {len(hot_df):,} violations in hotspot hexagons")

    profiles = {}

    for h3_idx in hotspot_hexes:
        hex_violations = hot_df[hot_df["h3_index"] == h3_idx]
        if len(hex_violations) < 5:
            continue

        profile = {}

        # --- Hourly profile (24 bins) ---
        hourly = hex_violations["hour"].value_counts().reindex(range(24), fill_value=0)
        profile["hourly_counts"] = hourly.tolist()
        profile["peak_hour"] = int(hourly.idxmax())
        profile["hourly_concentration"] = float(_hhi(hourly.values))

        # --- Daily profile (Mon-Sun) ---
        daily = hex_violations["day_of_week"].value_counts().reindex(range(7), fill_value=0)
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        profile["daily_counts"] = daily.tolist()
        profile["peak_day"] = day_names[int(daily.idxmax())]
        profile["weekend_ratio"] = float(daily.iloc[5:7].sum() / daily.sum()) if daily.sum() > 0 else 0

        # --- Monthly trend ---
        monthly = hex_violations["month"].value_counts().sort_index()
        profile["monthly_counts"] = {int(k): int(v) for k, v in monthly.items()}

        # --- FFT on hourly time series ---
        fft_result = _compute_fft_periods(hourly.values)
        profile["dominant_periods_hours"] = fft_result["dominant_periods"]
        profile["spectral_energy"] = fft_result["spectral_energy"]

        # --- Pattern classification ---
        profile["temporal_pattern"] = _classify_temporal_pattern(
            hourly.values, daily.values, profile["hourly_concentration"]
        )

        # --- Summary stats ---
        profile["violation_count"] = int(len(hex_violations))
        profile["unique_days"] = int(hex_violations["date"].nunique())
        profile["avg_severity"] = float(hex_violations["violation_severity_weight"].mean())
        profile["police_station"] = hex_violations["police_station"].mode().iloc[0]

        profiles[h3_idx] = profile

    logger.info(f"  Computed temporal profiles for {len(profiles):,} hotspot hexagons")

    # Aggregate pattern distribution
    patterns = [p["temporal_pattern"] for p in profiles.values()]
    pattern_counts = pd.Series(patterns).value_counts()
    for pattern, cnt in pattern_counts.items():
        logger.info(f"    {pattern:25s}: {cnt:>4} hexagons")

    return profiles


def _hhi(counts):
    """Compute Herfindahl-Hirschman Index (concentration measure)."""
    if counts.sum() == 0:
        return 0.0
    shares = counts / counts.sum()
    hhi = (shares ** 2).sum()
    # Normalize: 1/n (uniform) to 1.0 (concentrated)
    n = len(counts)
    return (hhi - 1/n) / (1 - 1/n) if n > 1 else hhi


def _compute_fft_periods(hourly_counts):
    """Apply FFT to hourly violation counts to find dominant periodicities."""
    n = len(hourly_counts)
    if n < 4:
        return {"dominant_periods": [], "spectral_energy": 0.0}

    # Remove DC component (mean)
    signal = hourly_counts - np.mean(hourly_counts)

    # FFT
    fft_vals = scipy_fft.rfft(signal)
    magnitudes = np.abs(fft_vals)

    # Frequencies (in cycles per hour-bin)
    freqs = scipy_fft.rfftfreq(n, d=1.0)

    # Find dominant periods (excluding DC at freq=0)
    dominant = []
    if len(magnitudes) > 1:
        sorted_idx = np.argsort(magnitudes[1:])[::-1] + 1  # skip DC
        for idx in sorted_idx[:3]:  # top 3 frequencies
            if freqs[idx] > 0 and magnitudes[idx] > 0.1 * magnitudes[1:].max():
                period_hours = round(1.0 / freqs[idx], 1)
                dominant.append(float(period_hours))

    total_energy = float(np.sum(magnitudes[1:] ** 2))

    return {
        "dominant_periods": dominant,
        "spectral_energy": round(total_energy, 2),
    }


def _classify_temporal_pattern(hourly, daily, concentration):
    """Classify the temporal pattern of a hotspot."""
    peak_hour = np.argmax(hourly)
    hourly_norm = hourly / hourly.sum() if hourly.sum() > 0 else hourly

    # Morning peak (7-10 AM with > 25% of violations)
    morning_pct = hourly_norm[7:11].sum()
    evening_pct = hourly_norm[17:21].sum()
    night_pct = hourly_norm[0:6].sum()
    weekend_pct = daily[5:7].sum() / daily.sum() if daily.sum() > 0 else 0

    if concentration > 0.15 and morning_pct > 0.30:
        return "morning_commercial"
    elif concentration > 0.15 and evening_pct > 0.30:
        return "evening_commercial"
    elif night_pct > 0.40:
        return "nighttime_hotspot"
    elif weekend_pct > 0.45:
        return "weekend_spike"
    elif concentration < 0.03:
        return "all_day_persistent"
    elif concentration > 0.15 and 10 <= peak_hour <= 16:
        return "midday_persistent"
    else:
        return "mixed_pattern"


# ============================================
# STEP 5: Save & summarize
# ============================================
def save_and_summarize(hex_stats, profiles):
    """Save outputs and print summary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save hex hotspot significance
    logger.info(f"Saving hotspot significance to {H3_HOTSPOT_SIG_PARQUET} ...")
    hex_stats.to_parquet(H3_HOTSPOT_SIG_PARQUET, index=False, engine="pyarrow")

    # Save temporal profiles as JSON
    logger.info(f"Saving temporal profiles to {TEMPORAL_PROFILES_JSON} ...")
    with open(TEMPORAL_PROFILES_JSON, "w") as f:
        json.dump(profiles, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("  PARKVISION AI - Statistical Hotspot Validation Summary")
    print("=" * 70)

    # Gi* results
    print(f"\n  --- Getis-Ord Gi* Results (violation count) ---")
    class_counts = hex_stats["hotspot_class"].value_counts()
    for cls in ["hotspot_99", "hotspot_95", "hotspot_90", "not_significant",
                "coldspot_90", "coldspot_95", "coldspot_99"]:
        if cls in class_counts.index:
            cnt = class_counts[cls]
            viol = hex_stats[hex_stats["hotspot_class"] == cls]["violation_count"].sum()
            pct = cnt / len(hex_stats) * 100
            print(f"    {cls:25s}: {cnt:>5} hexagons ({pct:>5.1f}%), {viol:>7,} violations")

    # Severity-weighted Gi*
    print(f"\n  --- Severity-Weighted Gi* ---")
    sev_class_counts = hex_stats["severity_hotspot_class"].value_counts()
    for cls in ["hotspot_99", "hotspot_95", "hotspot_90"]:
        if cls in sev_class_counts.index:
            cnt = sev_class_counts[cls]
            print(f"    {cls:25s}: {cnt:>5} hexagons")

    # Cross-tabulation: count vs severity hotspots
    both_hot = (
        hex_stats["hotspot_class"].str.startswith("hotspot")
        & hex_stats["severity_hotspot_class"].str.startswith("hotspot")
    ).sum()
    count_only = (
        hex_stats["hotspot_class"].str.startswith("hotspot")
        & ~hex_stats["severity_hotspot_class"].str.startswith("hotspot")
    ).sum()
    sev_only = (
        ~hex_stats["hotspot_class"].str.startswith("hotspot")
        & hex_stats["severity_hotspot_class"].str.startswith("hotspot")
    ).sum()
    print(f"\n    Both count+severity hotspot: {both_hot} hexagons")
    print(f"    Count hotspot only:         {count_only} hexagons")
    print(f"    Severity hotspot only:      {sev_only} hexagons")

    # Top Gi* hotspots
    hot = hex_stats[hex_stats["hotspot_class"].str.startswith("hotspot")]
    top = hot.nlargest(10, "gi_zscore")
    print(f"\n  --- Top 10 Hexagons by Gi* Z-Score ---")
    print(f"    {'H3 Index':>17}  {'Z-Score':>8}  {'Class':>12}  {'Violations':>11}  {'Station':>20}")
    for _, row in top.iterrows():
        print(f"    {row['h3_index']:>17}  {row['gi_zscore']:>8.2f}  "
              f"{row['hotspot_class']:>12}  {row['violation_count']:>11,}  "
              f"{row['primary_police_station']:>20}")

    # Temporal profiles summary
    print(f"\n  --- Temporal Profile Summary ({len(profiles):,} hotspots) ---")
    patterns = [p["temporal_pattern"] for p in profiles.values()]
    for pattern, cnt in pd.Series(patterns).value_counts().items():
        total_v = sum(
            p["violation_count"]
            for p in profiles.values()
            if p["temporal_pattern"] == pattern
        )
        print(f"    {pattern:25s}: {cnt:>4} hexagons, {total_v:>7,} violations")

    # FFT insights
    all_periods = []
    for p in profiles.values():
        all_periods.extend(p.get("dominant_periods_hours", []))
    if all_periods:
        period_counts = pd.Series(all_periods).value_counts().head(5)
        print(f"\n  --- Most Common Periodicities (FFT) ---")
        for period, cnt in period_counts.items():
            print(f"    {period:>6.1f} hours: detected in {cnt} hotspots")

    print(f"\n  Outputs:")
    print(f"    {H3_HOTSPOT_SIG_PARQUET}")
    print(f"    {TEMPORAL_PROFILES_JSON}")
    print("=" * 70)


# ============================================
# MAIN PIPELINE
# ============================================
def run():
    """Execute statistical hotspot validation and temporal decomposition."""
    # Load data
    logger.info(f"Loading H3 hex stats from {H3_HEX_STATS_PARQUET} ...")
    hex_stats = pd.read_parquet(H3_HEX_STATS_PARQUET)
    logger.info(f"  Loaded {len(hex_stats):,} hexagons")

    logger.info(f"Loading enriched violations from {ENRICHED_PARQUET} ...")
    df = pd.read_parquet(ENRICHED_PARQUET)
    logger.info(f"  Loaded {len(df):,} violations")

    # Step 1: Build spatial weights
    w = build_spatial_weights(hex_stats)

    # Step 2: Getis-Ord Gi* on violation count
    hex_stats = compute_getis_ord(hex_stats, w)

    # Step 3: Severity-weighted Gi*
    hex_stats = compute_severity_gi(hex_stats, w)

    # Step 4: Temporal profiles for hotspot hexagons
    profiles = compute_temporal_profiles(df, hex_stats)

    # Step 5: Save & summarize
    save_and_summarize(hex_stats, profiles)

    return hex_stats, profiles


if __name__ == "__main__":
    run()
