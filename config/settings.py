"""
PARKVISION AI — Central Configuration & Constants
===================================================
All constants, mappings, weights, and parameters referenced in the README.
"""

import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# ============================================
# PROJECT PATHS
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# Raw input file
RAW_CSV_PATH = PROJECT_ROOT / "jan to may police violation_anonymized791b166.csv"

# Processed data files (created by pipeline stages)
CLEANED_PARQUET = DATA_DIR / "cleaned_violations.parquet"
ENRICHED_PARQUET = DATA_DIR / "enriched_violations.parquet"
ROAD_GRAPH_FILE = DATA_DIR / "bengaluru_road_graph.graphml"
ROAD_EDGES_PARQUET = DATA_DIR / "road_edges.parquet"
H3_HEX_STATS_PARQUET = DATA_DIR / "h3_hex_stats.parquet"
H3_HEX_STATS_GEOJSON = OUTPUT_DIR / "h3_hex_stats.geojson"
HOTSPOT_CLUSTERS_PARQUET = DATA_DIR / "hotspot_clusters.parquet"
CLUSTER_PROFILES_PARQUET = DATA_DIR / "cluster_profiles.parquet"
CLUSTER_PROFILES_GEOJSON = OUTPUT_DIR / "cluster_profiles.geojson"
H3_HOTSPOT_SIG_PARQUET = DATA_DIR / "h3_hotspot_significance.parquet"
TEMPORAL_PROFILES_JSON = DATA_DIR / "temporal_profiles.json"
PCIS_SCORED_PARQUET = DATA_DIR / "pcis_scored_violations.parquet"
PCIS_HOTSPOTS_GEOJSON = OUTPUT_DIR / "pcis_hotspots.geojson"
RIPPLE_CONTOURS_GEOJSON = OUTPUT_DIR / "ripple_contours.geojson"
LOCATION_MEMORY_PARQUET = DATA_DIR / "location_memory.parquet"
SPILLOVER_PARQUET = DATA_DIR / "spillover_analysis.parquet"
PREDICTED_VIOLATIONS_PARQUET = DATA_DIR / "predicted_violations.parquet"
VIOLATION_MODEL_PATH = MODELS_DIR / "violation_predictor.joblib"
POLICE_STATIONS_GEOJSON = DATA_DIR / "police_stations.geojson"
ENFORCEMENT_PRIORITIES_PARQUET = DATA_DIR / "enforcement_priorities.parquet"
PATROL_ROUTES_GEOJSON = OUTPUT_DIR / "patrol_routes.geojson"
DAILY_BRIEF_MD = OUTPUT_DIR / "daily_brief.md"

# ============================================
# BENGALURU BOUNDING BOX
# ============================================
# Used to filter out GPS points that fall outside Bengaluru
BENGALURU_BBOX = {
    "lat_min": 12.7500,
    "lat_max": 13.2500,
    "lon_min": 77.3500,
    "lon_max": 77.8500,
}

# Center point for OSMnx queries
BENGALURU_CENTER = (12.9716, 77.5946)
BENGALURU_NETWORK_DIST_M = 25000  # 25km radius from center

# ============================================
# VIOLATION TYPE TAXONOMY & SEVERITY WEIGHTS
# ============================================
# From README Section 4.2 — congestion impact weights per offence code
VIOLATION_SEVERITY_WEIGHTS = {
    107: 1.00,   # PARKING IN A MAIN ROAD         — Critical
    108: 0.95,   # PARKING OPPOSITE ANOTHER PARKED — Critical (blocks lane)
    109: 0.90,   # DOUBLE PARKING                  — Critical (blocks entire lane)
    111: 0.80,   # PARKING NEAR BUSTOP/SCHOOL/HOSP — High (disrupts transit)
    104: 0.75,   # PARKING NEAR ROAD CROSSING      — High (intersection safety)
    112: 0.60,   # WRONG PARKING                   — Medium (context-dependent)
    113: 0.55,   # NO PARKING                      — Medium (zone violation)
    116: 0.10,   # DEFECTIVE NUMBER PLATE          — Incidental (no congestion)
}

# Default weight for any unknown offence code
DEFAULT_VIOLATION_WEIGHT = 0.30

# Human-readable names for offence codes
VIOLATION_TYPE_NAMES = {
    107: "PARKING IN A MAIN ROAD",
    108: "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE",
    109: "DOUBLE PARKING",
    111: "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC",
    104: "PARKING NEAR ROAD CROSSING",
    112: "WRONG PARKING",
    113: "NO PARKING",
    116: "DEFECTIVE NUMBER PLATE",
}

# ============================================
# VEHICLE FOOTPRINT DIMENSIONS (meters)
# ============================================
# From README Section 6.2 — Component 1
# (width, length) in meters — used for capacity reduction calculation
VEHICLE_FOOTPRINTS = {
    "CAR":              (2.0, 4.5),
    "MAXI-CAB":         (2.0, 4.5),
    "VAN":              (2.0, 4.5),
    "SCOOTER":          (0.8, 1.8),
    "MOPED":            (0.8, 1.8),
    "MOTOR CYCLE":      (0.8, 2.0),
    "PASSENGER AUTO":   (1.5, 3.0),
    "GOODS AUTO":       (1.8, 3.5),
    "LGV":              (2.2, 5.0),
    "TANKER":           (2.5, 8.0),
    "BUS":              (2.5, 12.0),
    "OTHERS":           (1.5, 3.5),
}

# Default footprint for unknown vehicle types
DEFAULT_VEHICLE_FOOTPRINT = (1.5, 3.5)

# ============================================
# PCIS (Parking Congestion Impact Score) COEFFICIENTS
# ============================================
# From README Section 6.3 — Composite PCIS formula
PCIS_WEIGHTS = {
    "capacity_reduction":        0.30,
    "proximity_factor":          0.20,
    "temporal_demand_multiplier": 0.20,
    "vehicle_obstruction_factor": 0.15,
    "network_criticality":       0.15,
}

# PCIS classification thresholds
PCIS_TIERS = {
    "low":          (0.0, 0.2),   # 🟢 Monitor only
    "moderate":     (0.2, 0.4),   # 🟡 Scheduled enforcement
    "high":         (0.4, 0.6),   # 🟠 Prioritized enforcement
    "critical":     (0.6, 0.8),   # 🔴 Immediate deployment
    "catastrophic": (0.8, 1.0),   # ⚫ Emergency intervention
}

# ============================================
# PROXIMITY FACTOR VALUES
# ============================================
# From README Section 6.2 — Component 2
PROXIMITY_FACTORS = {
    "at_junction":        1.0,   # Within 50m of signalized BTP junction
    "near_junction_100m": 0.8,   # Within 100m
    "near_junction_200m": 0.6,   # Within 200m
    "main_road":          0.3,   # On main road, away from junction
    "residential":        0.1,   # Residential side road
}

# Highway classes considered "main road" (OSM highway tag values)
MAIN_ROAD_HIGHWAY_CLASSES = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
}

RESIDENTIAL_HIGHWAY_CLASSES = {
    "residential", "living_street", "service",
}

# ============================================
# TEMPORAL DEMAND MULTIPLIER CURVE
# ============================================
# From README Section 6.2 — Component 3
# Key: (hour_start, hour_end) → multiplier for WEEKDAYS
WEEKDAY_DEMAND_CURVE = {
    (0, 6):   0.10,   # Night / early morning
    (6, 8):   0.60,   # Pre-rush
    (8, 10):  1.00,   # Morning rush (PEAK)
    (10, 12): 0.70,   # Mid-morning
    (12, 14): 0.60,   # Midday
    (14, 16): 0.50,   # Early afternoon
    (16, 17): 0.70,   # Pre-evening rush
    (17, 20): 1.00,   # Evening rush (PEAK)
    (20, 22): 0.40,   # Post-rush
    (22, 24): 0.15,   # Late night
}

# Weekend multiplier relative to weekday
WEEKEND_MULTIPLIER = 0.70

# ============================================
# ST-DBSCAN PARAMETERS
# ============================================
# From README Section 5.1 — three spatial scales
STDBSCAN_PARAMS = {
    "micro": {
        "eps_spatial_m":  50,    # ε₁ in meters
        "eps_temporal_h": 2,     # ε₂ in hours
        "min_pts":        5,
    },
    "meso": {
        "eps_spatial_m":  150,
        "eps_temporal_h": 2,
        "min_pts":        5,
    },
    "macro": {
        "eps_spatial_m":  500,
        "eps_temporal_h": 2,
        "min_pts":        5,
    },
}

# ============================================
# H3 HEXAGONAL INDEXING
# ============================================
H3_RESOLUTION = 9   # ~174m edge length — block-level analysis

# ============================================
# ROAD NETWORK DEFAULTS
# ============================================
# Standard lane width (meters) per HCM
STANDARD_LANE_WIDTH_M = 3.5

# Default assumptions when OSM data is missing
DEFAULT_LANES = 2
DEFAULT_SPEED_LIMIT_KMPH = 40
DEFAULT_ROAD_WIDTH_M = 7.0   # 2 lanes × 3.5m

# Capacity per lane per hour (HCM urban arterial)
LANE_CAPACITY_VPH = 1800  # vehicles per hour per lane

# ============================================
# POI ENRICHMENT
# ============================================
POI_SEARCH_RADIUS_M = 300  # meters from hotspot centroid

POI_CATEGORIES = {
    "metro_station": '["railway"="station"]["station"="subway"]',
    "bus_stop":      '["highway"="bus_stop"]',
    "hospital":      '["amenity"="hospital"]',
    "school":        '["amenity"="school"]',
    "market":        '["shop"="supermarket"]',
    "mall":          '["shop"="mall"]',
}

# ============================================
# ENFORCEMENT OPTIMIZER
# ============================================
# Default shift durations (hours)
SHIFT_WINDOWS = {
    "morning":  (6, 10),
    "midday":   (10, 14),
    "afternoon": (14, 17),
    "evening":  (17, 21),
    "night":    (21, 6),
}

# ACO parameters
ACO_PARAMS = {
    "n_ants":        50,
    "n_iterations":  200,
    "alpha":         1.0,   # Pheromone influence
    "beta":          2.0,   # Heuristic influence (PCIS/travel_time)
    "evaporation":   0.5,   # Pheromone evaporation rate
    "q":             100,   # Pheromone deposit factor
}

# Average patrol speed in city (km/h)
PATROL_SPEED_KMPH = 20

# ============================================
# API KEYS (loaded from .env)
# ============================================
# Create a .env file at project root with:
#   TOMTOM_API_KEY=your_key_here
#   GEMINI_API_KEY=your_key_here
#   NVIDIA_API_KEY=your_key_here

TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# ============================================
# AUTHENTICATION
# ============================================
AUTH_SECRET = os.getenv("AUTH_SECRET", "parkvision-demo-secret-change-in-production")
AUTH_COOKIE_NAME = "parkvision_token"
SESSION_EXPIRE_HOURS = int(os.getenv("SESSION_EXPIRE_HOURS", "24"))
USERS_FILE = DATA_DIR / "users.json"
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
STATION_PASSWORD = os.getenv("STATION_PASSWORD", "station123")


def hash_password(password: str) -> str:
    return hashlib.sha256(f"{AUTH_SECRET}:{password}".encode()).hexdigest()


# ============================================
# LOGGING
# ============================================
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_LEVEL = "INFO"
