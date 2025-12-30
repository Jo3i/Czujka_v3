import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model.pkl"

def predict_label(features: np.ndarray):
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    label_encoder = bundle["label_encoder"]

    probs = model.predict_proba([features])[0]
    idx = probs.argmax()

    label = label_encoder.inverse_transform([idx])[0]
    confidence = float(probs[idx])

    return label, confidence
