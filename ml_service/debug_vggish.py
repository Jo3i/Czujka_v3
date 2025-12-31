import librosa
from vggish.extractor import VGGishExtractor

audio, sr = librosa.load(
    "data/raw/dog/dog2.wav",
    sr=16000,
    mono=True
)

print("Długość audio (s):", len(audio) / sr)

extractor = VGGishExtractor()
embeddings = extractor.extract(audio)

print("Embeddings:", type(embeddings))
if embeddings is not None:
    print("Shape:", embeddings.shape)
