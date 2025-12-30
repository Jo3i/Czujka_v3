from pathlib import Path
import numpy as np
import librosa
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from vggish.extractor import VGGishExtractor
from classifier.features import aggregate_embeddings

# Ścieżki
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MODEL_PATH = Path(__file__).parent / "model.pkl"

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

            audio, sr = librosa.load(wav_file, sr=None)

            embeddings = extractor.extract(audio)

            if embeddings is None or len(embeddings) == 0:
                print("    [WARN] Brak embeddingów – pomijam")
                continue

            features = aggregate_embeddings(embeddings)

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

def main():
    print("[INFO] Ładowanie danych...")
    X, y = load_dataset()

    print(f"[INFO] Próbek: {len(X)}")

    # Encoder etykiet
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    # Klasyfikator
    model = Pipeline([
        ("clf", LogisticRegression(max_iter=2000))
    ])

    print("[INFO] Trenowanie modelu...")
    model.fit(X, y_enc)

    # Ewaluacja (na tym samym zbiorze – OK na start)
    y_pred = model.predict(X)
    print("\n[REPORT]")
    print(classification_report(y_enc, y_pred, target_names=label_encoder.classes_))

    # Zapis modelu + encoder
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
