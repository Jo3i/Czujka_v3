from pathlib import Path
import sqlite3
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "events.db"

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)  # tworzy folder data jeśli nie istnieje
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            label TEXT,
            score REAL
        )
    """)
    conn.commit()
    conn.close()

def log_event(label: str, score: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO events (timestamp, label, score) VALUES (?, ?, ?)",
        (datetime.now().isoformat(), label, score)
    )
    conn.commit()
    conn.close()
