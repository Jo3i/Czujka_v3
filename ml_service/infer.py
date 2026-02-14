from audio.recorder import AudioRecorder
from audio.vad import EnergyVAD
from classifier.classify import AudioClassifier
from monitor.db import init_db, log_event
from gps.gps_reader import GPSReader
import time
import numpy as np # Musi byc zaimportowane do obliczania RMS

# --- KONFIGURACJA ---
VAD_THRESHOLD = 0.002 
CONFIDENCE_THRESHOLD = 0.6 
PRE_GAIN = 10.0  # <--- TU JEST KLUCZ! Mnożymy głośność x10 PRZED analizą
# --------------------

def main():
    init_db()
    print("[INFO] Ładowanie komponentów...")
    
    # Użyjemy try/except, żeby wyłapać błąd, jeśli GPS nie jest podłączony
    try:
        gps = GPSReader(serial_port='/dev/ttyS0') # lub /dev/ttyUSB0
    except Exception as e:
        print(f"[WARN] Błąd GPS: {e}. Uruchamiam bez GPS.")
        gps = None

    recorder = AudioRecorder()
    vad = EnergyVAD(threshold=VAD_THRESHOLD)
    classifier = AudioClassifier()

    print(f"[INFO] Start monitorowania (Próg VAD: {VAD_THRESHOLD}, Gain: {PRE_GAIN}x)")

    try:
        while True:
            # 1. Nagrywanie
            audio = recorder.record(2.0)

            # 2. -- CYFROWE WZMOCNIENIE (PRE-AMP) --
            # To sprawi, że VAD usłyszy dźwięk!
            audio = audio * PRE_GAIN
            # --------------------------------------

            # Obliczamy RMS tylko dla Twojej informacji w konsoli
            rms = np.sqrt(np.mean(audio**2))

            # 3. Sprawdzenie VAD
            if vad.is_active(audio):
                print(f"[VAD] Wykryto dźwięk! (RMS: {rms:.4f}) -> Klasyfikacja...")
                
                label, confidence = classifier.classify(audio)
                
                # Pobranie GPS (bezpiecznie)
                location = gps.get_location() if gps else None

                print(f"   >>> WYNIK: {label.upper()} | Pewność: {confidence:.2f} | GPS: {location}")

                if confidence >= CONFIDENCE_THRESHOLD:
                    log_event(label, confidence, location)
                    print("   [+] Zapisano w bazie.")
            else:
                # Zamiast kropki, wypiszmy aktualną głośność, żebyś wiedział czy mikrofon działa
                # \r pozwala nadpisywać linię w terminalu
                print(f"[CISZA] Głośność: {rms:.5f} / Próg: {VAD_THRESHOLD} (x{PRE_GAIN})", end='\r')


    except KeyboardInterrupt:
        print("\n[INFO] Stop.")

if __name__ == "__main__":
    main()