from audio.recorder import AudioRecorder
from audio.vad import EnergyVAD
from classifier.classify import AudioClassifier
from monitor.db import init_db, log_event
from gps.gps_reader import GPSReader
import time
from gps.gps_reader import GPSReader
import time


def main():
    init_db()

    print("[INFO] Ładowanie komponentów...")


    recorder = AudioRecorder()
    vad = EnergyVAD(threshold=0.0001)
    classifier = AudioClassifier()
    gps = GPSReader()
    classifier = AudioClassifier()
    gps = GPSReader()

    print("[INFO] Start monitorowania dźwięku (Ctrl+C aby zakończyć)")

    try:
        while True:
            audio = recorder.record(2.0)
            audio = recorder.record(2.0)

            if vad.is_active(audio):
                label, confidence = classifier.classify(audio)

                location = gps.get_location()

                print(f"[EVENT] {label} | pewność={confidence:.2f} | GPS={location}")
                location = gps.get_location()

                print(f"[EVENT] {label} | pewność={confidence:.2f} | GPS={location}")

                if confidence >= 0.7:
                    log_event(
                        label=label,
                        score=confidence,
                        location=location
                    )
                else:
                    print("[INFO] Zbyt niska pewność – zdarzenie odrzucone")

            else:
                print("[INFO] Cisza – brak zdarzenia")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] Monitorowanie zatrzymane przez użytkownika")


if __name__ == "__main__":
    main()
