from pathlib import Path
import numpy as np
import librosa
import joblib

# ZMIANA: Importujemy RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
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

    # Iteracja po folderach (klasach)
    for class_dir in DATA_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        label = class_dir.name
        print(f"[INFO] Klasa: {label}")

        # Iteracja po plikach WAV
        for wav_file in class_dir.glob("*.wav"):
            print(f"  → {wav_file.name}")

            # 1. Ładowanie audio
            try:
                audio, sr = librosa.load(
                    wav_file,
                    sr=16000,
                    mono=True
                )
            except Exception as e:
                print(f"    [ERROR] Błąd ładowania pliku {wav_file.name}: {e}")
                continue

            # 2. Sprawdzenie długości (VGGish wymaga ok. 1s)
            if len(audio) < 16000:
                print("    [WARN] Za krótki sygnał – pomijam")
                continue

            # 3. Ekstrakcja cech (VGGish)
            try:
                embedding = extractor.extract(audio)
            except Exception as e:
                print(f"    [ERROR] Błąd VGGish dla {wav_file.name}: {e}")
                continue

            # 4. Walidacja embeddingu (czy nie jest pusty/NaN)
            if embedding is None or embedding.shape != (128,) or np.isnan(embedding).any():
                print(f"    [WARN] Niepoprawny embedding (za krótki plik?) – pomijam: {wav_file.name}")
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
        raise RuntimeError("❌ Brak danych treningowych – sprawdź folder data/raw")

    print(f"[INFO] Liczba próbek: {len(X)}")

    # Kodowanie etykiet (np. bird -> 0, cat -> 1)
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    print("[INFO] Klasy:")
    for idx, cls in enumerate(label_encoder.classes_):
        print(f"  {idx} → {cls}")

    # Obliczanie wag dla klas (ważne przy nierównych danych)
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

    # Definicja modelu (Pipeline)
    model = Pipeline([
        (
            "clf",
            RandomForestClassifier(
                n_estimators=200,           # Liczba drzew (więcej = stabilniej)
                max_depth=None,             # Głębokość drzew (None = do końca)
                class_weight=class_weight_dict, # Obsługa niezbalansowanych klas
                random_state=42,            # Powtarzalność wyników
                n_jobs=-1                   # Użyj wszystkich rdzeni procesora
            )
        )
    ])

    print("[INFO] Trenowanie modelu (Random Forest)...")
    model.fit(X, y_enc)

    # Ewaluacja na zbiorze treningowym
    y_pred = model.predict(X)

    print("\n[REPORT]")
    print(
        classification_report(
            y_enc,
            y_pred,
            target_names=label_encoder.classes_
        )
    )

    # Zapis modelu i encodera do pliku
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