from audio.recorder import AudioRecorder
from audio.vad import EnergyVAD
from classifier.classify import AudioClassifier
from monitor.db import init_db, log_event
from gps.gps_reader import GPSReader
import time
import numpy as np

# --- KONFIGURACJA ---
VAD_THRESHOLD = 0.002 
# Obniżamy próg pewności, bo mikrofon w RPi jest gorszej jakości niż pliki treningowe
CONFIDENCE_THRESHOLD = 0.35  
# Mnożnik TYLKO dla detekcji ciszy (żeby wyzwolić nagrywanie)
VAD_GAIN = 30.0  
# --------------------

def normalize_audio(audio_data):
    """Bezpieczne podgłaśnianie dla AI (Normalizacja)"""
    peak = np.max(np.abs(audio_data))
    if peak > 0:
        # Skalujemy tak, aby najgłośniejszy punkt miał wartość 0.9
        # To zachowuje kształt fali (nie zniekształca), a jest głośne
        return (audio_data / peak) * 0.9
    return audio_data

def main():
    init_db()
    print("[INFO] Ładowanie komponentów...")
    
    try:
        gps = GPSReader(serial_port='/dev/ttyS0')
    except Exception as e:
        print(f"[WARN] Błąd GPS: {e}. Bez GPS.")
        gps = None

    recorder = AudioRecorder()
    vad = EnergyVAD(threshold=VAD_THRESHOLD)
    classifier = AudioClassifier()

    print(f"[INFO] Start. VAD Gain: {VAD_GAIN}x | Próg AI: {CONFIDENCE_THRESHOLD}")

    try:
        while True:
            # 1. Nagrywamy surowy dźwięk (cichy)
            raw_audio = recorder.record(2.0)

            # 2. Przygotowujemy wersję "dla VAD" (sztucznie głośną, może być zniekształcona)
            vad_audio = raw_audio * VAD_GAIN

            # Obliczamy RMS dla podglądu
            rms = np.sqrt(np.mean(vad_audio**2))

            # 3. Sprawdzamy czy coś słychać (na wersji głośnej)
            if vad.is_active(vad_audio):
                print(f"[VAD] Wykryto aktywność! (RMS: {rms:.4f}) -> Analiza...")
                
                # 4. -- KLUCZOWA ZMIANA --
                # Do AI wysyłamy wersję ZNORMALIZOWANĄ, a nie przesterowaną
                # Dzięki temu AI widzi poprawny kształt fali
                clean_audio_for_ai = normalize_audio(raw_audio)
                
                label, confidence = classifier.classify(clean_audio_for_ai)
                
                location = gps.get_location() if gps else None

                print(f"   >>> WYNIK: {label.upper()} | Pewność: {confidence:.2f}")

                # Zapisujemy, jeśli pewność jest powyżej progu (obniżonego do 0.35)
                if confidence >= CONFIDENCE_THRESHOLD:
                    log_event(label, confidence, location)
                    print("   [+] Zapisano w bazie.")
                else:
                    print(f"   [-] Odrzucono (za niska pewność, wymagane {CONFIDENCE_THRESHOLD})")

            else:
                # Wypisujemy poziom szumu, żebyś widział czy mikrofon żyje
                print(f"[Nasłuch] Poziom: {rms:.4f}", end='\r')

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] Stop.")

if __name__ == "__main__":
    main()