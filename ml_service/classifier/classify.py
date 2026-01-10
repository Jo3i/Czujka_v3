from pathlib import Path
import numpy as np
import joblib

from vggish.extractor import VGGishExtractor
from classifier.features import aggregate_embeddings


class AudioClassifier:
    def __init__(
        self,
        model_path: Path | None = None,
        confidence_threshold: float = 0.6
    ):
        """
        AudioClassifier:
        - ładuje model + encoder
        - używa VGGish
        - zwraca (label, confidence)
        """

        if model_path is None:
            model_path = Path(__file__).parent / "model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")

        print("[INFO] Ładowanie klasyfikatora...")

        bundle = joblib.load(model_path)

        self.model = bundle["model"]
        self.label_encoder = bundle["label_encoder"]
        self.threshold = confidence_threshold

        self.extractor = VGGishExtractor()

    def classify(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Klasyfikuje próbkę audio.
        Zwraca:
        - label (str)
        - confidence (float)
        """

        # 🔹 bezpieczeństwo
        if audio is None or len(audio) == 0:
            return "unknown", 0.0

        # 🔹 VGGish
        embeddings = self.extractor.extract(audio)

        # VGGish MUSI dać (T, 128) albo (128,)
        if embeddings is None:
            print("[WARN] Brak embeddingów")
            return "unknown", 0.0

        # jeśli (128,) → OK
        if embeddings.ndim == 1:
            features = embeddings

        # jeśli (T, 128) → agregujemy
        elif embeddings.ndim == 2:
            features = aggregate_embeddings(embeddings)

        else:
            print(f"[WARN] Zły kształt embeddingów: {embeddings.shape}")
            return "unknown", 0.0

        # 🔹 predykcja prawdopodobieństw
        probs = self.model.predict_proba([features])[0]

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        # 🔹 próg pewności
        if confidence < self.threshold:
            return "unknown", confidence

        label = self.label_encoder.inverse_transform([best_idx])[0]

        return label, confidence
