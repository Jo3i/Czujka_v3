import numpy as np


class EnergyVAD:
    def __init__(self, threshold: float = 0.01):
        """
        threshold – próg energii sygnału
        """
        self.threshold = threshold

    def is_active(self, audio: np.ndarray) -> bool:
        """
        Sprawdza, czy w sygnale występuje zdarzenie akustyczne.
        """
        energy = np.mean(audio ** 2)
        return energy > self.threshold

    def energy(self, audio: np.ndarray) -> float:
        """
        Zwraca energię sygnału (do celów diagnostycznych).
        """
        return np.mean(audio ** 2)
