from fastapi import FastAPI
from api.schemas import EventOut
from api.deps import get_db
from typing import List

app = FastAPI(
    title="Animal Sound Monitoring API",
    description="API do udostępniania zdarzeń wykrytych przez czujkę terenową",
    version="1.0"
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/events", response_model=List[EventOut])
def get_events(limit: int = 100):
    db = get_db()
    cursor = db.cursor()

    cursor.execute("""
        SELECT timestamp, label, confidence, latitude, longitude
        FROM events
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    db.close()

    return [dict(row) for row in rows]


@app.get("/events/map", response_model=List[EventOut])
def get_events_for_map():
    db = get_db()
    cursor = db.cursor()

    cursor.execute("""
        SELECT timestamp, label, confidence, latitude, longitude
        FROM events
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    """)

    rows = cursor.fetchall()
    db.close()

    return [dict(row) for row in rows]
