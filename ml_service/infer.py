from pathlib import Path
import time

from audio.recorder import AudioRecorder
from audio.vad import EnergyVAD
from classifier.classify import AudioClassifier
from monitor.db import init_db, log_event


# Konfiguracja
MODEL_PATH = Path(__file__).parent / "classifier" / "model.pkl"
CONFIDENCE_THRESHOLD = 0.2
RECORD_SECONDS = 2.0


def main():
    print("[INFO] Inicjalizacja bazy danych...")
    init_db()

    print("[INFO] Ładowanie komponentów...")
    recorder = AudioRecorder()
    vad = EnergyVAD(threshold=0.0001)
    classifier = AudioClassifier(MODEL_PATH)

    print("[INFO] Start monitorowania dźwięku (Ctrl+C aby zakończyć)")

    try:
        while True:
            print(f"[INFO] Nagrywanie {RECORD_SECONDS} s...")
            audio = recorder.record(RECORD_SECONDS)

            if vad.is_active(audio):
                label, confidence = classifier.classify(audio)

                print(f"[EVENT] {label} | pewność={confidence:.2f}")

                if confidence >= CONFIDENCE_THRESHOLD:
                    log_event(label=label, score=confidence)
                else:
                    print("[INFO] Zbyt niska pewność – zdarzenie odrzucone")

            else:
                print("[INFO] Cisza – brak zdarzenia")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] Monitorowanie zatrzymane przez użytkownika")


if __name__ == "__main__":
    main()
