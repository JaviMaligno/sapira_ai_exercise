from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

# Attempt to load environment variables from a .env file at repo root
# This module lives in src/sapira_etl/, so repo root is two parents up
MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent.parent
if load_dotenv is not None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

# Centralized settings
MONGO_URI: str = os.getenv("MONGO_URI", "")


def require_mongo_uri() -> str:
    """Return MONGO_URI or raise a helpful error if missing."""
    if not MONGO_URI:
        raise RuntimeError(
            "MONGO_URI is not set. Create a .env file with MONGO_URI=... or export it in your environment."
        )
    return MONGO_URI
