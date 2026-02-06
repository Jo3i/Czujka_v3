import numpy as np
import os
# ZAMIANA: Używamy lekkiego runtime zamiast pełnego tensorflow
import tflite_runtime.interpreter as tflite

# Importujemy helpery do przetwarzania audio (te pliki masz w folderze)
from . import vggish_input, vggish_params

class VGGishExtractor:
    def __init__(self):
        # 1. Ścieżka do modelu TFLite (nie CKPT!)
        current_dir = os.path.dirname(__file__)
        # Musisz pobrać plik 'vggish.tflite' i wrzucić go do folderu ml_service/vggish/
        model_filename = 'vggish.tflite'
        self.model_path = os.path.join(current_dir, model_filename)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Nie znaleziono modelu: {self.model_path}. Pobierz go!")

        # 2. Inicjalizacja interpretera TFLite
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        # 3. Pobranie informacji o wejściu/wyjściu modelu
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Indeks tensora wejściowego (tam gdzie wkładamy spektrogram)
        self.input_index = self.input_details[0]['index']
        # Indeks tensora wyjściowego (tam gdzie wychodzą embeddingi)
        self.output_index = self.output_details[0]['index']

    def extract(self, audio, sample_rate=16000):
        """
        Przetwarza surowe audio na wektor cech (embeddings).
        """
        # 1. Przetwarzanie audio na spektrogramy Mel (log-mel)
        # To korzysta z vggish_input.py (musi być czyste od tensorflow!)
        examples = vggish_input.waveform_to_examples(audio, sample_rate)
        
        # TFLite wymaga typu float32
        examples = examples.astype(np.float32)

        # Jeśli nagranie jest ciszą lub błędem i nie ma ramek
        if examples.shape[0] == 0:
            return np.zeros((128,), dtype=np.float32)

        # 2. Obsługa dynamicznego rozmiaru (Batching)
        # VGGish dzieli audio na fragmenty po 0.96s. 
        # Oryginalny tensor ma kształt [?, 96, 64]. Musimy go dostosować do liczby fragmentów.
        
        # Zmieniamy rozmiar wejścia w locie
        self.interpreter.resize_tensor_input(self.input_index, examples.shape)
        self.interpreter.allocate_tensors()

        # 3. Wrzucamy dane i uruchamiamy
        self.interpreter.set_tensor(self.input_index, examples)
        self.interpreter.invoke()

        # 4. Odbieramy wynik
        embeddings = self.interpreter.get_tensor(self.output_index)

        # 5. Uśredniamy wynik (jeden wektor dla całego pliku audio)
        # VGGish zwraca [liczba_fragmentow, 128]. Robimy średnią -> [128]
        return np.mean(embeddings, axis=0)