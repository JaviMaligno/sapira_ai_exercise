from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


def stable_hash(value: Optional[object], salt: str) -> Optional[str]:
    if value is None:
        return None
    s = f"{salt}:{value}".encode()
    return hashlib.sha256(s).hexdigest()


@dataclass
class DbConfig:
    user: str
    password: str
    host: str
    port: int
    db: str
    schema: str = "public"
    sslmode: str = "disable"

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"
            f"?sslmode={self.sslmode}"
        )


def load_env(env_path: Optional[str] = None) -> None:
    if env_path and os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()


def get_db_config() -> DbConfig:
    return DbConfig(
        user=os.getenv("POSTGRES_USER", "sapira"),
        password=os.getenv("POSTGRES_PASSWORD", "sapira_password"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        db=os.getenv("POSTGRES_DB", "fraudlab"),
        schema=os.getenv("DB_SCHEMA", "public"),
        sslmode=os.getenv("DB_SSLMODE", "disable"),
    )


def get_hash_salt() -> str:
    return os.getenv("HASH_SALT", "CHANGE_ME_SALT")



