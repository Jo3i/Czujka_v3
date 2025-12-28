from audio.recorder import AudioRecorder
from audio.vad import EnergyVAD
from classifier.classify import AudioClassifier
from monitor import init_db, log_event
from datetime import datetime


def main():
    init_db()

    recorder = AudioRecorder()
    vad = EnergyVAD(threshold=0.0001)
    classifier = AudioClassifier()

    print("[INFO] Nasłuchiwanie dźwięku...")

    audio = recorder.record(2.0)

    if vad.is_active(audio):
        label, confidence = classifier.classify(audio)
        print(f"[EVENT] Wykryto dźwięk: {label} (pewność={confidence:.2f})")

        log_event(
            label=label,
            score=confidence
        )
    else:
        print("[INFO] Cisza – brak zdarzenia")


if __name__ == "__main__":
    main()
