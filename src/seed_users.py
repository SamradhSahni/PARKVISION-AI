"""
Seed data/users.json with admin + 54 station accounts (shared station password).

Usage:
    python -m src.seed_users
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    USERS_FILE,
    DATA_DIR,
    PCIS_SCORED_PARQUET,
    ADMIN_USERNAME,
    ADMIN_PASSWORD,
    STATION_PASSWORD,
    hash_password,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_users")


def get_station_names() -> list:
    parquet = PCIS_SCORED_PARQUET
    if parquet.exists():
        df = pd.read_parquet(parquet, columns=["police_station"])
        return sorted(df["police_station"].dropna().unique().tolist())

    geojson = DATA_DIR / "police_stations.geojson"
    if geojson.exists():
        import geopandas as gpd
        gdf = gpd.read_file(geojson)
        col = "police_station" if "police_station" in gdf.columns else gdf.columns[0]
        return sorted(gdf[col].dropna().unique().tolist())

    raise FileNotFoundError("Run the data pipeline first to populate station names.")


def seed():
    stations = get_station_names()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "admin": {
            "username": ADMIN_USERNAME,
            "password_hash": hash_password(ADMIN_PASSWORD),
        },
        "station_password_hash": hash_password(STATION_PASSWORD),
        "stations": stations,
    }

    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Wrote {USERS_FILE}")
    logger.info(f"  Admin user: {ADMIN_USERNAME}")
    logger.info(f"  Station accounts: {len(stations)} (shared password)")
    logger.info(f"  Demo credentials — admin: {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
    logger.info(f"  Demo credentials — any station / {STATION_PASSWORD}")


if __name__ == "__main__":
    seed()
