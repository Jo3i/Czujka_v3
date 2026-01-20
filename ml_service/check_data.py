import numpy as np
import librosa
from vggish.extractor import VGGishExtractor
from pathlib import Path

# Upewnij się, że ścieżka jest OK
DATA_DIR = Path("data/raw")
extractor = VGGishExtractor()

print(f"{'PLIK':<30} | {'ŚREDNIA':<10} | {'STD DEV':<10}")
print("-" * 55)

for wav_file in DATA_DIR.rglob("*.wav"):
    audio, sr = librosa.load(wav_file, sr=16000, mono=True)
    emb = extractor.extract(audio)
    
    # Jeśli embedding to same zera lub stała wartość -> VGGish nie działa
    mean_val = np.mean(emb)
    std_val = np.std(emb)
    
    print(f"{wav_file.name:<30} | {mean_val:.4f}     | {std_val:.4f}")