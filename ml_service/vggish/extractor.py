import numpy as np
import librosa
import tensorflow as tf
import os
from . import vggish_input, vggish_slim, vggish_params


class VGGishExtractor:
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.compat.v1.Session()
            vggish_slim.define_vggish_slim()
            
            # --- ZMIANA: Ładowanie wag zamiast losowej inicjalizacji ---
            
            # Pobieramy ścieżkę do folderu, w którym znajduje się ten plik (extractor.py)
            current_dir = os.path.dirname(__file__)
            # Tworzymy pełną ścieżkę do pliku modelu (musi być w tym samym folderze co skrypt)
            checkpoint_path = os.path.join(current_dir, 'vggish_model.ckpt')
            
            # Ładujemy wytrenowany model
            vggish_slim.load_vggish_slim_checkpoint(self.session, checkpoint_path)

    def extract(self, audio, sample_rate=16000):
        # Resampling, jeśli częstotliwość jest inna niż 16kHz
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # Konwersja audio na przykłady wejściowe dla VGGish
        examples = vggish_input.waveform_to_examples(audio, 16000)

        features_tensor = self.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME
        )
        embedding_tensor = self.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME
        )

        # Uruchomienie sesji TensorFlow w celu uzyskania embeddingów
        embeddings = self.session.run(
            embedding_tensor,
            feed_dict={features_tensor: examples}
        )

        # Zwracamy średnią z embeddingów (jeden wektor dla całego nagrania)
        return np.mean(embeddings, axis=0)