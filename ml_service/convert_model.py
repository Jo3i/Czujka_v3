import tensorflow.compat.v1 as tf
import os
import sys

# Ustawienie ścieżek
sys.path.append(os.getcwd())
from vggish import vggish_slim, vggish_params

tf.disable_v2_behavior()

CKPT_PATH = os.path.join('vggish', 'vggish_model.ckpt')
TFLITE_PATH = os.path.join('vggish', 'vggish.tflite')

def main():
    if not os.path.exists(CKPT_PATH):
        print(f"BŁĄD: Nie znaleziono pliku {CKPT_PATH}")
        return

    print("1. Budowanie grafu VGGish...")
    with tf.Graph().as_default() as graph:
        # Wejście
        input_tensor = tf.placeholder(
            tf.float32, 
            shape=(None, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS),
            name='vggish_input'
        )

        # Budowa modelu (nawet jeśli doda dziwny scope, naprawimy to potem)
        embedding_tensor = vggish_slim.define_vggish_slim(input_tensor, training=False)

        # --- MAGIA: RĘCZNE MAPOWANIE ZMIENNYCH ---
        # Pobieramy wszystkie zmienne, które stworzył skrypt
        all_vars = tf.global_variables()
        var_map = {}

        print("   Tworzenie mapy zmiennych...")
        for var in all_vars:
            # Nazwa zmiennej w Twoim obecnym skrypcie
            current_name = var.op.name
            
            # Nazwa, której spodziewamy się w pliku .ckpt (musi zaczynać się od vggish/)
            # Jeśli skrypt dodał 'vggish_slim/', usuwamy to.
            ckpt_name = current_name.replace("vggish_slim/", "")
            
            # Dodajemy do mapy: { "nazwa_w_pliku_ckpt" : zmienna_w_kodzie }
            var_map[ckpt_name] = var

        # Sesja
        with tf.Session() as sess:
            print(f"2. Ładowanie wag z {CKPT_PATH}...")
            sess.run(tf.global_variables_initializer())
            
            # Używamy naszej sprytnej mapy do ładowania
            saver = tf.train.Saver(var_list=var_map)
            
            try:
                saver.restore(sess, CKPT_PATH)
            except Exception as e:
                print("\n!!! BŁĄD ŁADOWANIA WAG !!!")
                print("Sprawdź, czy plik .ckpt jest poprawny.")
                print("Szczegóły błędu:", e)
                return

            print("3. Konwersja do TFLite...")
            converter = tf.lite.TFLiteConverter.from_session(
                sess, [input_tensor], [embedding_tensor]
            )
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            tflite_model = converter.convert()

            with open(TFLITE_PATH, 'wb') as f:
                f.write(tflite_model)
            
            print(f"SUKCES! Model zapisany: {TFLITE_PATH}")

if __name__ == '__main__':
    main()