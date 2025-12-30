import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "events.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            label TEXT,
            score REAL,
            duratiuon REAL
        )
    """)
    conn.commit()
    conn.close()


def log_event(label: str, score: float, duration: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO events (timestamp, label, score, duratiuon) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(), label, score, duration)
    )
    conn.commit()
    conn.close()
