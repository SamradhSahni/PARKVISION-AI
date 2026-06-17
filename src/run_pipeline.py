"""
PARKVISION AI — End-to-End Pipeline Runner
=============================================
Executes all analysis stages in sequence with logging,
progress tracking, and checkpoint resume capability.

Usage:
    python -m src.run_pipeline               # Run full pipeline
    python -m src.run_pipeline --from 5      # Resume from stage 5
    python -m src.run_pipeline --stage 10    # Run only stage 10
"""

import sys
import time
import json
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, OUTPUT_DIR, MODELS_DIR

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level="INFO",
)
logger = logging.getLogger("pipeline")

CHECKPOINT_FILE = DATA_DIR / ".pipeline_checkpoint.json"


# ── Stage definitions ────────────────────────────────────────────
STAGES = [
    {
        "id": 1,
        "name": "Data Ingestion & Cleaning",
        "module": "src.data_ingestion",
        "outputs": ["data/cleaned_violations.parquet"],
    },
    {
        "id": 2,
        "name": "Road Network & Map-Matching",
        "module": "src.road_network",
        "outputs": ["data/enriched_violations.parquet", "data/road_edges.parquet"],
    },
    {
        "id": 3,
        "name": "H3 Spatial Indexing & POI",
        "module": "src.spatial_indexing",
        "outputs": ["data/h3_hex_stats.parquet"],
    },
    {
        "id": 4,
        "name": "ST-DBSCAN Hotspot Clustering",
        "module": "src.hotspot_engine",
        "outputs": ["data/hotspot_clusters.parquet", "data/cluster_profiles.parquet"],
    },
    {
        "id": 5,
        "name": "Gi* Statistics & Temporal Profiling",
        "module": "src.hotspot_stats",
        "outputs": ["data/h3_hotspot_significance.parquet", "data/temporal_profiles.json"],
    },
    {
        "id": 6,
        "name": "PCIS Scoring Engine",
        "module": "src.pcis_engine",
        "outputs": ["data/pcis_scored_violations.parquet", "data/h3_pcis_scores.parquet"],
    },
    {
        "id": 7,
        "name": "Congestion Propagation & Location Memory",
        "module": "src.congestion_model",
        "outputs": ["data/location_memory.parquet", "data/spillover_analysis.parquet",
                     "output/ripple_contours.geojson"],
    },
    {
        "id": 8,
        "name": "XGBoost Violation Prediction",
        "module": "src.prediction",
        "outputs": ["data/predicted_violations.parquet", "models/violation_predictor.joblib"],
    },
    {
        "id": 9,
        "name": "Police Station Mapping & Enforcement ROI",
        "module": "src.enforcement_optimizer",
        "outputs": ["data/enforcement_priorities.parquet", "data/police_stations.geojson"],
    },
    {
        "id": 10,
        "name": "ACO Patrol Route Optimization",
        "module": "src.patrol_router",
        "outputs": ["output/patrol_routes.geojson", "output/daily_brief.md"],
    },
]


def load_checkpoint():
    """Load pipeline checkpoint."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed_stages": [], "last_run": None}


def save_checkpoint(checkpoint):
    """Save pipeline checkpoint."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)


def check_outputs(stage):
    """Check if stage outputs already exist."""
    project_root = Path(__file__).parent.parent
    return all((project_root / out).exists() for out in stage["outputs"])


def run_stage(stage):
    """Run a single pipeline stage."""
    import importlib
    module = importlib.import_module(stage["module"])
    if hasattr(module, "run"):
        module.run()
    else:
        logger.warning(f"  Stage {stage['id']} has no run() function — skipping")


def run_pipeline(from_stage=1, single_stage=None, force=False):
    """Execute the full pipeline or a subset of stages."""
    checkpoint = load_checkpoint()

    print("\n" + "=" * 70)
    print("  PARKVISION AI — Full Analysis Pipeline")
    print("=" * 70)
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Stages: {len(STAGES)} total")
    if single_stage:
        print(f"  Mode: Single stage ({single_stage})")
    else:
        print(f"  Mode: Sequential from stage {from_stage}")
    print("=" * 70 + "\n")

    stages_to_run = STAGES
    if single_stage:
        stages_to_run = [s for s in STAGES if s["id"] == single_stage]
    else:
        stages_to_run = [s for s in STAGES if s["id"] >= from_stage]

    total = len(stages_to_run)
    results = []

    for i, stage in enumerate(stages_to_run, 1):
        stage_id = stage["id"]
        stage_name = stage["name"]

        # Check if already completed (unless forced)
        if not force and stage_id in checkpoint.get("completed_stages", []):
            if check_outputs(stage):
                logger.info(f"[{i}/{total}] Stage {stage_id}: {stage_name} — SKIPPED (already completed)")
                results.append({"stage": stage_id, "name": stage_name, "status": "skipped", "time": 0})
                continue

        # Run stage
        print(f"\n{'─' * 70}")
        print(f"  [{i}/{total}] Stage {stage_id}: {stage_name}")
        print(f"{'─' * 70}")

        t0 = time.time()
        try:
            run_stage(stage)
            elapsed = time.time() - t0

            # Update checkpoint
            if stage_id not in checkpoint["completed_stages"]:
                checkpoint["completed_stages"].append(stage_id)
            checkpoint["last_run"] = datetime.now().isoformat()
            save_checkpoint(checkpoint)

            results.append({"stage": stage_id, "name": stage_name, "status": "success", "time": elapsed})
            logger.info(f"  [OK] Stage {stage_id} completed in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"  [FAIL] Stage {stage_id} failed after {elapsed:.1f}s: {e}")
            traceback.print_exc()
            results.append({"stage": stage_id, "name": stage_name, "status": "failed",
                            "time": elapsed, "error": str(e)})

            if not single_stage:
                logger.error(f"  Pipeline halted. Resume with: python -m src.run_pipeline --from {stage_id}")
                break

    # Summary
    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"\n  {'Stage':>5}  {'Name':40s}  {'Status':>8s}  {'Time':>8s}")
    print(f"  {'─'*5}  {'─'*40}  {'─'*8}  {'─'*8}")

    total_time = 0
    for r in results:
        status_str = r["status"].upper()
        time_str = f"{r['time']:.1f}s" if r["time"] > 0 else "—"
        total_time += r["time"]
        print(f"  {r['stage']:>5}  {r['name']:40s}  {status_str:>8s}  {time_str:>8s}")

    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    succeeded = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"  Succeeded: {succeeded} | Skipped: {skipped} | Failed: {failed}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PARKVISION AI Pipeline Runner")
    parser.add_argument("--from", type=int, default=1, dest="from_stage",
                        help="Resume from this stage number")
    parser.add_argument("--stage", type=int, default=None,
                        help="Run only this single stage")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if checkpoint exists")
    args = parser.parse_args()

    run_pipeline(from_stage=args.from_stage, single_stage=args.stage, force=args.force)
