import librosa
from vggish.extractor import VGGishExtractor

audio, sr = librosa.load("data/raw/dog/test.wav", sr=None)

extractor = VGGishExtractor()
embedding = extractor.extract(audio, sr)

print("Embedding shape:", embedding.shape)
print(embedding[:10])
