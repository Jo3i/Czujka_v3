import numpy as np
import librosa

from vggish.extractor import VGGishExtractor
from classifier.features import aggregate_embeddings


class AudioClassifier:
    def __init__(self, model_path):
        import joblib

        data = joblib.load(model_path)
        self.model = data["model"]
        self.label_encoder = data["label_encoder"]
        self.extractor = VGGishExtractor()

    def classify(self, audio, sr=16000):
        """
        audio: np.ndarray (1D)
        sr: sampling rate
        """

        # Wymuszenie float32
        audio = audio.astype(np.float32)

        # Resampling do 16 kHz
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        embeddings = self.extractor.extract(audio)


        if embeddings is None or embeddings.ndim != 2 or embeddings.shape[1] != 128:
            print("[WARN] Niepoprawne embeddingi – pomijam próbkę")
            return "unknown", 0.0

        features = aggregate_embeddings(embeddings)

        probs = self.model.predict_proba([features])[0]
        idx = probs.argmax()

        label = self.label_encoder.inverse_transform([idx])[0]
        confidence = probs[idx]

        return label, float(confidence)
