from audio.recorder import AudioRecorder
from audio.vad import EnergyVAD
from classifier.classify import AudioClassifier
from monitor import init_db, log_event
from pathlib import Path
from datetime import datetime
import sqlite3
import time
import pandas as pd

# Inicjalizacja bazy
init_db()

# Tworzenie obiektów
recorder = AudioRecorder()
vad = EnergyVAD(threshold=0.0001)
classifier = AudioClassifier()

print("[INFO] Rozpoczynam nasłuchiwanie dźwięku... (Ctrl+C aby zatrzymać)")

try:
    while True:
        # Nagrywanie próbki audio 2 sekundy
        audio = recorder.record(2.0)

        if vad.is_active(audio):
            # Klasyfikacja dźwięku
            label, confidence = classifier.classify(audio)
            print(f"[EVENT] Wykryto dźwięk: {label} (pewność={confidence:.2f})")

            # Logowanie do bazy
            log_event(label=label, score=confidence)
        else:
            print("[INFO] Cisza – brak zdarzenia")

        time.sleep(1)  # opcjonalna przerwa między nagraniami

except KeyboardInterrupt:
    print("\n[INFO] Monitorowanie zatrzymane przez użytkownika.")

# Funkcja do podglądu najnowszych zdarzeń
def show_recent_events(n=10):
    DB_PATH = Path(__file__).resolve().parent / "data" / "events.db"
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM events", conn)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.sort_values(by='timestamp', ascending=False)
    print("\n--- Ostatnie zdarzenia ---")
    print(df.head(n))
    conn.close()

# Po zakończeniu monitorowania pokaz ostatnie 10 zdarzeń
show_recent_events()
