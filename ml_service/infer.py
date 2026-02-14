from audio.recorder import AudioRecorder
from audio.vad import EnergyVAD
from classifier.classify import AudioClassifier
from monitor.db import init_db, log_event
from monitor.firebase_client import init_firebase, send_event_to_cloud # <--- NOWE
from gps.gps_reader import GPSReader
import time
import numpy as np

# --- KONFIGURACJA ---
VAD_THRESHOLD = 0.002 
CONFIDENCE_THRESHOLD = 0.6
VAD_GAIN = 30.0  
# --------------------

def normalize_audio(audio_data):
    peak = np.max(np.abs(audio_data))
    if peak > 0:
        return (audio_data / peak) * 0.9
    return audio_data

def main():
    init_db()
    init_firebase() # <--- Łączymy się z chmurą na starcie
    
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
            # 1. Nagrywanie
            raw_audio = recorder.record(2.0)
            
            # 2. VAD
            vad_audio = raw_audio * VAD_GAIN
            rms = np.sqrt(np.mean(vad_audio**2))

            if vad.is_active(vad_audio):
                print(f"[VAD] Wykryto aktywność! (RMS: {rms:.4f})")
                
                clean_audio_for_ai = normalize_audio(raw_audio)
                label, confidence = classifier.classify(clean_audio_for_ai)
                location = gps.get_location() if gps else None

                print(f"   >>> WYNIK: {label.upper()} | Pewność: {confidence:.2f}")

                if confidence >= CONFIDENCE_THRESHOLD:
                    # 1. Zapis lokalny (SQLite)
                    log_event(label, confidence, location)
                    
                    # 2. Zapis do chmury (Firebase)
                    send_event_to_cloud(label, confidence, location)
                    
                    print("   [+] Zapisano w bazie i wysłano do chmury.")
                else:
                    print(f"   [-] Odrzucono (za niska pewność).")

            else:
                print(f"[Nasłuch] Poziom: {rms:.4f}", end='\r')

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] Stop.")

if __name__ == "__main__":
    main()