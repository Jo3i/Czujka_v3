import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "events.db"


def init_db():
    """
    Inicjalizacja bazy danych zdarzeń.
    """
    print("[INFO] Inicjalizacja bazy danych...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            label TEXT NOT NULL,
            score REAL NOT NULL,
            latitude REAL,
            longitude REAL
        )
    """)

    conn.commit()
    conn.close()


def log_event(
    label: str,
    score: float,
    location: Optional[Tuple[float, float]] = None
):
    """
    Zapis zdarzenia do bazy danych.

    Args:
        label: etykieta klasyfikacji
        score: pewność klasyfikacji
        location: (latitude, longitude) lub None
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    latitude = None
    longitude = None

    if location is not None:
        latitude, longitude = location

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO events (timestamp, label, score, latitude, longitude)
        VALUES (?, ?, ?, ?, ?)
    """, (timestamp, label, score, latitude, longitude))

    conn.commit()
    conn.close()
