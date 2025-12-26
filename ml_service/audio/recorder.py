import sounddevice as sd
import numpy as np


class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels

    def record(self, duration: float) -> np.ndarray:
        """
        Nagrywa dźwięk z mikrofonu przez określony czas.
        """
        print(f"[INFO] Nagrywanie {duration} s...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32"
        )
        sd.wait()
        return audio.flatten()
