from __future__ import annotations

import sqlite3
from pathlib import Path

from flask import current_app, g
from werkzeug.security import generate_password_hash


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS producers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    document_number TEXT,
    phone TEXT,
    location TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    producer_id INTEGER NOT NULL,
    code TEXT NOT NULL UNIQUE,
    variety TEXT NOT NULL,
    total_bags REAL NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (producer_id) REFERENCES producers (id)
);

CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lot_id INTEGER NOT NULL,
    sample_number INTEGER NOT NULL,
    sample_weight_kg REAL NOT NULL,
    image_filename TEXT NOT NULL,
    result_image_filename TEXT NOT NULL,
    result_json_filename TEXT NOT NULL,
    healthy_count INTEGER NOT NULL DEFAULT 0,
    chalky_count INTEGER NOT NULL DEFAULT 0,
    broken_count INTEGER NOT NULL DEFAULT 0,
    total_detected INTEGER NOT NULL DEFAULT 0,
    healthy_pct REAL NOT NULL DEFAULT 0,
    chalky_pct REAL NOT NULL DEFAULT 0,
    broken_pct REAL NOT NULL DEFAULT 0,
    average_confidence REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (lot_id, sample_number),
    FOREIGN KEY (lot_id) REFERENCES lots (id)
);
"""


def get_db():
    if "db" not in g:
        db_path = Path(current_app.config["DATABASE"])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(db_path)
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()
    db.executescript(SCHEMA)
    migrate_db(db)
    ensure_default_user(db)
    db.commit()


def migrate_db(db):
    # Obtener columnas actuales
    lot_columns = {row["name"] for row in db.execute("PRAGMA table_info(lots)").fetchall()}
    
    # Si todavía existe la columna vieja 'total_weight_kg', migramos la tabla completa
    if "total_weight_kg" in lot_columns:
        print("Migrando tabla 'lots' para cambiar Kilos por Sacos...")
        # 1. Crear tabla temporal con el esquema nuevo
        db.execute("""
            CREATE TABLE lots_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                producer_id INTEGER NOT NULL,
                code TEXT NOT NULL UNIQUE,
                variety TEXT NOT NULL,
                total_bags REAL NOT NULL DEFAULT 0,
                notes TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (producer_id) REFERENCES producers (id)
            )
        """)
        
        # 2. Copiar datos (si existía total_bags la usamos, si no usamos los kilos como sacos temporalmente)
        if "total_bags" in lot_columns:
            db.execute("""
                INSERT INTO lots_new (id, producer_id, code, variety, total_bags, notes, created_at)
                SELECT id, producer_id, code, variety, total_bags, notes, created_at FROM lots
            """)
        else:
            db.execute("""
                INSERT INTO lots_new (id, producer_id, code, variety, total_bags, notes, created_at)
                SELECT id, producer_id, code, variety, total_weight_kg, notes, created_at FROM lots
            """)
            
        # 3. Reemplazar tablas
        db.execute("DROP TABLE lots")
        db.execute("ALTER TABLE lots_new RENAME TO lots")
    
    # Caso secundario: Si no tiene total_bags (y no tenía la columna vieja), simplemente añadirla
    elif "total_bags" not in lot_columns:
        db.execute("ALTER TABLE lots ADD COLUMN total_bags REAL NOT NULL DEFAULT 0")


def ensure_default_user(db):
    existing = db.execute("SELECT id FROM users WHERE username = ?", (current_app.config["DEFAULT_USERNAME"],)).fetchone()
    if existing:
        return
    db.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (
            current_app.config["DEFAULT_USERNAME"],
            generate_password_hash(current_app.config["DEFAULT_PASSWORD"]),
        ),
    )


def init_app(app):
    app.teardown_appcontext(close_db)

    @app.cli.command("init-db")
    def init_db_command():
        init_db()
        print("Base de datos inicializada.")
