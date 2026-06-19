"""
PARKVISION AI — Authentication (JWT + JSON user store)
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import jwt
from fastapi import Cookie, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    AUTH_SECRET,
    AUTH_COOKIE_NAME,
    SESSION_EXPIRE_HOURS,
    USERS_FILE,
    hash_password,
)

logger = logging.getLogger("auth")
_bearer = HTTPBearer(auto_error=False)


@dataclass
class User:
    username: str
    role: str  # "admin" | "station"
    station_name: Optional[str] = None
    display_name: Optional[str] = None


def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


def load_users() -> dict:
    if not USERS_FILE.exists():
        raise FileNotFoundError(
            f"User store not found at {USERS_FILE}. Run: python -m src.seed_users"
        )
    with open(USERS_FILE, encoding="utf-8") as f:
        return json.load(f)


def authenticate(login_type: str, username: str, password: str, station: Optional[str] = None) -> User:
    data = load_users()

    if login_type == "admin":
        admin = data.get("admin", {})
        if username != admin.get("username"):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
        if not verify_password(password, admin.get("password_hash", "")):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
        return User(
            username=admin["username"],
            role="admin",
            display_name="City Command",
        )

    if login_type == "station":
        station_name = (station or username or "").strip()
        if not station_name:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Station is required")
        stations = data.get("stations", [])
        if station_name not in stations:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown station")
        shared_hash = data.get("station_password_hash", "")
        if not verify_password(password, shared_hash):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
        return User(
            username=station_name.lower().replace(" ", "_"),
            role="station",
            station_name=station_name,
            display_name=f"{station_name} Traffic Police",
        )

    raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid login type")


def create_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.username,
        "role": user.role,
        "station_name": user.station_name,
        "display_name": user.display_name,
        "iat": now,
        "exp": now + timedelta(hours=SESSION_EXPIRE_HOURS),
    }
    return jwt.encode(payload, AUTH_SECRET, algorithm="HS256")


def decode_token(token: str) -> User:
    try:
        payload = jwt.decode(token, AUTH_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid session")

    return User(
        username=payload["sub"],
        role=payload["role"],
        station_name=payload.get("station_name"),
        display_name=payload.get("display_name"),
    )


def _token_from_request(
    auth_cookie: Optional[str],
    creds: Optional[HTTPAuthorizationCredentials],
) -> str:
    if auth_cookie:
        return auth_cookie
    if creds and creds.credentials:
        return creds.credentials
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")


def get_current_user(
    auth_cookie: Optional[str] = Cookie(None, alias=AUTH_COOKIE_NAME),
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> User:
    token = _token_from_request(auth_cookie, creds)
    return decode_token(token)


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
    return user


def station_scope(user: User) -> Optional[str]:
    """Return station name for station-role users, else None (city-wide)."""
    return user.station_name if user.role == "station" else None
