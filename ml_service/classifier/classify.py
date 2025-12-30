import numpy as np
from vggish.extractor import VGGishExtractor
from classifier.features import aggregate_embeddings
from classifier.predict import predict_label

class AudioClassifier:
    def __init__(self):
        self.extractor = VGGishExtractor()

    def classify(self, audio: np.ndarray):
        """
        audio: surowy sygnał audio (1D numpy array)
        """
        # 1️⃣ Ekstrakcja cech VGGish
        embeddings = self.extractor.extract(audio)
        # embeddings.shape = (N, 128)

        if embeddings is None or len(embeddings) == 0:
            return "unknown", 0.0

        # 2️⃣ AGREGACJA (KROK 3)
        features = aggregate_embeddings(embeddings)
        # features.shape = (128,)

        # 3️⃣ Predykcja klasy
        label, confidence = predict_label(features)

        return label, confidence
