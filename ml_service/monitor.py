import sqlite3
from datetime import datetime


def init_db():
    conn = sqlite3.connect("events.db")
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
    conn = sqlite3.connect("events.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO events (timestamp, label, score) VALUES (?, ?, ?)",
        (datetime.now().isoformat(), label, score)
    )
    conn.commit()
    conn.close()
