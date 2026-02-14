import firebase_admin
from firebase_admin import credentials, db
import os
import time

# Ścieżka do klucza (w głównym folderze ml_service)
KEY_PATH = "serviceAccountKey.json"

# Adres Twojej bazy Realtime Database (znajdziesz go w ustawieniach Firebase)
DATABASE_URL = "https://czujka-1ed39-default-rtdb.europe-west1.firebasedatabase.app/"

_is_initialized = False

def init_firebase():
    """Inicjalizuje połączenie z chmurą."""
    global _is_initialized
    
    if _is_initialized:
        return

    if not os.path.exists(KEY_PATH):
        print(f"[FIREBASE] Błąd: Nie znaleziono pliku {KEY_PATH}")
        return

    try:
        cred = credentials.Certificate(KEY_PATH)
        firebase_admin.initialize_app(cred, {
            'databaseURL': DATABASE_URL
        })
        _is_initialized = True
        print("[FIREBASE] Połączono z chmurą.")
    except Exception as e:
        print(f"[FIREBASE] Błąd inicjalizacji: {e}")

def send_event_to_cloud(label, confidence, location):
    """Wysyła pojedyncze zdarzenie do bazy Realtime Database."""
    if not _is_initialized:
        init_firebase()
    
    if not _is_initialized:
        print("[FIREBASE] Nie można wysłać - brak połączenia.")
        return

    try:
        # Przygotowanie danych (JSON)
        # Firebase lubi słowniki (dict)
        event_data = {
            "label": label,
            "confidence": round(float(confidence), 2),
            "timestamp": int(time.time() * 1000), # Czas w milisekundach (dla Androida)
            "date_string": time.strftime("%Y-%m-%d %H:%M:%S"),
            "location": {
                "lat": location[0] if location else 0.0,
                "lon": location[1] if location else 0.0
            }
        }

        # Zapis do węzła 'detections'
        # push() tworzy unikalne ID dla każdego wpisu
        ref = db.reference('detections')
        ref.push(event_data)
        
        print(f"[FIREBASE] Wysłano zdarzenie: {label}")

    except Exception as e:
        print(f"[FIREBASE] Błąd wysyłania: {e}")