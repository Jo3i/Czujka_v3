from audio.recorder import AudioRecorder
from audio.vad import EnergyVAD
from classifier.classify import AudioClassifier
from monitor.db import init_db, log_event
from gps.gps_reader import GPSReader
import time

# --- KONFIGURACJA ---
# Próg ciszy (im mniej, tym czulszy). 
# 0.002 to dobra wartość dla cichego mikrofonu.
VAD_THRESHOLD = 0.002 

# Ile pewności musi mieć AI, żeby zapisać zdarzenie do bazy (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.6 
# --------------------

def main():
    init_db()

    print("[INFO] Ładowanie komponentów...")
    
    # Inicjalizacja obiektów
    recorder = AudioRecorder()
    vad = EnergyVAD(threshold=VAD_THRESHOLD)
    classifier = AudioClassifier()
    gps = GPSReader()

    print(f"[INFO] Start monitorowania (Próg VAD: {VAD_THRESHOLD})")
    print("[INFO] Naciśnij Ctrl+C, aby zakończyć.")

    try:
        while True:
            # 1. Nagrywanie próbki (2 sekundy)
            audio = recorder.record(2.0)

            # 2. Sprawdzenie czy jest cisza
            if vad.is_active(audio):
                # 3. Jeśli słychać dźwięk -> Klasyfikacja
                label, confidence = classifier.classify(audio)
                
                # 4. Pobranie GPS
                location = gps.get_location()

                # Wypisanie wyniku w terminalu
                print(f"[EVENT] {label} | Pewność: {confidence:.2f} | GPS: {location}")

                # 5. Logowanie do bazy, jeśli pewność jest wystarczająca
                if confidence >= CONFIDENCE_THRESHOLD:
                    log_event(
                        label=label,
                        score=confidence,
                        location=location
                    )
                    # Opcjonalnie: krótki komunikat, że zapisano
                    # print(" -> Zapisano w bazie.")
                else:
                    print(f" -> Odrzucono (wymagane {CONFIDENCE_THRESHOLD})")

            else:
                # Jeśli cisza, tylko krótki log (lub pass, żeby nie śmiecić)
                print(".", end="", flush=True) # Kropki oznaczają nasłuchiwanie w ciszy

            # Krótka pauza dla procesora
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] Monitorowanie zatrzymane przez użytkownika")

if __name__ == "__main__":
    main()