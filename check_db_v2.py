import sqlite3
from pathlib import Path

db_path = Path("web/storage/rice_app/rice_system.sqlite3")
if not db_path.exists():
    print(f"Base de datos no encontrada en {db_path}")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(lots)")
    columns = cursor.fetchall()
    print("Columnas en la tabla 'lots':")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    conn.close()
