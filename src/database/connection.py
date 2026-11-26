"""
Database connection manager for ResumeMatch AI.
Provides session factories for PostgreSQL, with automatic graceful fallback to SQLite
when PostgreSQL is not running locally.
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.database.models import Base

# Determine active database URL
_engine = None
_session_factory = None
ACTIVE_DB_TYPE = "unknown"

def get_engine():
    global _engine, ACTIVE_DB_TYPE
    if _engine is not None:
        return _engine

    # 1. Try PostgreSQL if configured
    pg_url = config.POSTGRES_URL
    try:
        test_engine = create_engine(pg_url, connect_args={"connect_timeout": 3})
        # Test connection
        with test_engine.connect() as conn:
            pass
        _engine = test_engine
        ACTIVE_DB_TYPE = "postgresql"
        print(f"[DB] Connected successfully to PostgreSQL at {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}")
        return _engine
    except Exception as e:
        # Fall back to SQLite
        sqlite_path = config.SQLITE_DB_PATH
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_url = config.SQLITE_URL
        _engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
        ACTIVE_DB_TYPE = "sqlite"
        print(f"[DB] PostgreSQL unavailable ({e}). Gracefully running on SQLite: {sqlite_path}")
        return _engine

def get_session():
    """
    Returns a new thread-safe database session.
    """
    global _session_factory
    if _session_factory is None:
        eng = get_engine()
        _session_factory = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=eng))
    return _session_factory()

def init_schema():
    """
    Initializes tables in database schema.
    """
    eng = get_engine()
    Base.metadata.create_all(bind=eng)
    return True
