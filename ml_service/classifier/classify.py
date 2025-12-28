import numpy as np
import random
from classifier.labels import CLASS_LABELS


class AudioClassifier:
    def __init__(self):
        """
        Inicjalizacja gotowego klasyfikatora dźwięków.
        W docelowej wersji w tym miejscu ładowany jest model.
        """
        print("[INFO] Klasyfikator audio zainicjalizowany")

    def classify(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Klasyfikuje fragment audio.

        Zwraca:
        - etykietę klasy
        - poziom pewności predykcji
        """
        label = random.choice(CLASS_LABELS)
        confidence = round(random.uniform(0.6, 0.95), 2)
        return label, confidence
