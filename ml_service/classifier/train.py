from pathlib import Path
import numpy as np
import librosa
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from vggish.extractor import VGGishExtractor


# =========================
# ŚCIEŻKI
# =========================
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MODEL_PATH = Path(__file__).parent / "model.pkl"


# =========================
# ŁADOWANIE DANYCH
# =========================
def load_dataset():
    extractor = VGGishExtractor()

    X = []
    y = []

    for class_dir in DATA_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        label = class_dir.name
        print(f"[INFO] Klasa: {label}")

        for wav_file in class_dir.glob("*.wav"):
            print(f"  → {wav_file.name}")

            audio, sr = librosa.load(
                wav_file,
                sr=16000,
                mono=True
            )

            if len(audio) < 16000:
                print("    [WARN] Za krótki sygnał – pomijam")
                continue

            embedding = extractor.extract(audio)

            if embedding is None or embedding.shape != (128,):
                print("    [WARN] Niepoprawny embedding – pomijam")
                continue

            X.append(embedding)
            y.append(label)

    return np.array(X), np.array(y)


# =========================
# TRENOWANIE
# =========================
def main():
    print("[INFO] Ładowanie danych...")
    X, y = load_dataset()

    if len(X) == 0:
        raise RuntimeError("❌ Brak danych treningowych – sprawdź WAV")

    print(f"[INFO] Liczba próbek: {len(X)}")

    # Kodowanie etykiet
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    print("[INFO] Klasy:")
    for idx, cls in enumerate(label_encoder.classes_):
        print(f"  {idx} → {cls}")

    # Ważenie klas
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_enc),
        y=y_enc
    )
    class_weight_dict = dict(enumerate(class_weights))

    print("[INFO] Wagi klas:")
    for idx, w in class_weight_dict.items():
        label = label_encoder.inverse_transform([idx])[0]
        print(f"  {label} → {w:.2f}")

    # Model
    model = Pipeline([
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                class_weight=class_weight_dict
            )
        )
    ])

    print("[INFO] Trenowanie modelu...")
    model.fit(X, y_enc)

    y_pred = model.predict(X)

    print("\n[REPORT]")
    print(
        classification_report(
            y_enc,
            y_pred,
            target_names=label_encoder.classes_
        )
    )

    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder
        },
        MODEL_PATH
    )

    print(f"[INFO] Model zapisany w: {MODEL_PATH}")


if __name__ == "__main__":
    main()
