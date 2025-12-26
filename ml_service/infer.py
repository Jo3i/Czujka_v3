from audio.recorder import AudioRecorder
from audio.vad import EnergyVAD


def main():
    recorder = AudioRecorder()
    vad = EnergyVAD(threshold=0.0001)

    audio = recorder.record(2.0)
    energy = vad.energy(audio)

    if vad.is_active(audio):
        print(f"[EVENT] Wykryto zdarzenie akustyczne (energia={energy:.6f})")
    else:
        print(f"[INFO] Cisza / szum tła (energia={energy:.6f})")


if __name__ == "__main__":
    main()
