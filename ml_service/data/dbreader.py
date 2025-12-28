from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "events.db"  # lub "data/events.db", jeśli baza w folderze data

def read_events(label_filter=None):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM events", conn)

    # konwersja timestamp na datetime
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # sortowanie po czasie malejąco (od najnowszego)
    df = df.sort_values(by='timestamp', ascending=False)

    # filtrowanie po etykiecie, jeśli podano
    if label_filter is not None:
        df = df[df['label'].isin(label_filter)]

    conn.close()
    return df

if __name__ == "__main__":
    events_df = read_events()
    print(events_df)
