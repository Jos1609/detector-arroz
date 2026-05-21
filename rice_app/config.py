from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNTIME_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else PROJECT_DIR
WEB_STORAGE_DIR = RUNTIME_DIR / "web" / "storage"
APP_STORAGE_DIR = WEB_STORAGE_DIR / "rice_app"
UPLOADS_DIR = APP_STORAGE_DIR / "uploads"
RESULTS_DIR = APP_STORAGE_DIR / "detections"
DATABASE_PATH = APP_STORAGE_DIR / "rice_system.sqlite3"


class Config:
    SECRET_KEY = os.environ.get("RICE_APP_SECRET_KEY", "cambia-esta-clave-interna")
    DATABASE = DATABASE_PATH
    APP_STORAGE_DIR = APP_STORAGE_DIR
    UPLOADS_DIR = UPLOADS_DIR
    RESULTS_DIR = RESULTS_DIR
    MAX_CONTENT_LENGTH = 15 * 1024 * 1024
    DEFAULT_USERNAME = os.environ.get("RICE_APP_USERNAME", "admin")
    DEFAULT_PASSWORD = os.environ.get("RICE_APP_PASSWORD", "admin123")
