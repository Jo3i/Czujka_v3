import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent / "events.db"

def read_events(limit=50):
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        id,
        datetime(timestamp, 'localtime') AS time,
        label,
        ROUND(score, 2) AS score
    FROM events
    ORDER BY timestamp DESC
    LIMIT ?
    """

    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()

    return df


if __name__ == "__main__":
    df = read_events(limit=20)

    print("\n📊 Ostatnie zdarzenia (od najnowszych):\n")
    print(df.to_string(index=False))
