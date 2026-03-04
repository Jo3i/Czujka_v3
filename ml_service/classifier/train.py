from pathlib import Path
import numpy as np
import librosa
import joblib

# Import XGBoost
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

import sys
import os
# Dodajemy katalog nadrzędny (ml_service) do ścieżki
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vggish.extractor import VGGishExtractor

# =========================
# ŚCIEŻKI
# =========================
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MODEL_PATH = Path(__file__).parent / "model.pkl"
FEATURES_CACHE_X = Path(__file__).parent / "features_X.npy"
FEATURES_CACHE_Y = Path(__file__).parent / "labels_y.npy"

# =========================
# ŁADOWANIE DANYCH (Z CACHEM)
# =========================
def load_dataset():
    # Jeśli mamy zapisane cechy, ładujemy je błyskawicznie w ułamek sekundy!
    if FEATURES_CACHE_X.exists() and FEATURES_CACHE_Y.exists():
        print("[INFO] Znaleziono zcache'owane cechy na dysku! Ładowanie potrwa 1 sekundę...")
        return np.load(FEATURES_CACHE_X), np.load(FEATURES_CACHE_Y)

    print("[INFO] Brak cache. Rozpoczynam ciężką ekstrakcję VGGish (to potrwa...)")
    extractor = VGGishExtractor()

    X = []
    y = []

    # Iteracja po folderach (klasach)
    for class_dir in DATA_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        label = class_dir.name
        print(f"[INFO] Przetwarzam klasę: {label}")

        for wav_file in class_dir.glob("*.wav"):
            print(f"  → {wav_file.name}")

            try:
                audio, sr = librosa.load(wav_file, sr=16000, mono=True)
            except Exception as e:
                print(f"    [ERROR] Błąd ładowania pliku {wav_file.name}: {e}")
                continue

            if len(audio) < 16000:
                print("    [WARN] Za krótki sygnał – pomijam")
                continue

            try:
                embedding = extractor.extract(audio)
            except Exception as e:
                print(f"    [ERROR] Błąd VGGish dla {wav_file.name}: {e}")
                continue

            if embedding is None or embedding.shape != (128,) or np.isnan(embedding).any():
                print(f"    [WARN] Niepoprawny embedding – pomijam: {wav_file.name}")
                continue

            X.append(embedding)
            y.append(label)

    X_arr = np.array(X)
    y_arr = np.array(y)
    
    # Zapisujemy do cache, żeby już nigdy więcej nie czekać 2 godzin!
    np.save(FEATURES_CACHE_X, X_arr)
    np.save(FEATURES_CACHE_Y, y_arr)
    print("[INFO] Zapisano cechy do cache'u pomyślnie.")

    return X_arr, y_arr

# =========================
# TRENOWANIE XGBoost
# =========================
def main():
    print("[INFO] Start procesu...")
    X, y = load_dataset()

    if len(X) == 0:
        raise RuntimeError("❌ Brak danych treningowych – sprawdź folder data/raw")

    print(f"[INFO] Liczba poprawnych próbek: {len(X)}")

    # Kodowanie etykiet (cat -> 0, crow -> 1 itd.)
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    print("[INFO] Obliczanie zbalansowanych wag próbek dla XGBoost...")
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_enc
    )

    # Definicja modelu XGBoost
    model = Pipeline([
        (
            "clf",
            XGBClassifier(
                n_estimators=200,           
                max_depth=3,                
                learning_rate=0.1,          
                random_state=42,            
                n_jobs=-1                   
            )
        )
    ])

    print("[INFO] Trenowanie modelu XGBoost...")
    model.fit(X, y_enc, clf__sample_weight=sample_weights)

    # Ocena modelu (Wygenerowanie Twojej tabeli)
    y_pred = model.predict(X)

    print("\n" + "="*50)
    print(" [REPORT - WYNIKI KLASYFIKACJI XGBOOST]")
    print("="*50)
    print(
        classification_report(
            y_enc,
            y_pred,
            target_names=label_encoder.classes_
        )
    )
    print("="*50 + "\n")

    # Zapis gotowego modelu .pkl
    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder
        },
        MODEL_PATH
    )

    print(f"[INFO] Gotowe! Nowy model XGBoost został zapisany w:\n {MODEL_PATH}")

if __name__ == "__main__":
    main()