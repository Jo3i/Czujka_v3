import sounddevice as sd
import numpy as np

class AudioRecorder:
    def __init__(self, sr=16000):
        self.sr = sr

    def record(self, duration):
        print(f"[INFO] Nagrywanie {duration} s...")

        audio = sd.rec(
            int(duration * self.sr),
            samplerate=self.sr,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        # mono → 1D
        return audio.flatten()
