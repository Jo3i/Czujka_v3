from typing import Optional, Tuple
import serial
import pynmea2
import time
import logging

class GPSReader:
    """
    Moduł odpowiedzialny za odczyt fizycznej lokalizacji GPS.
    Wymaga podłączonego modułu GPS przez UART lub USB.
    """

    def __init__(self, serial_port='/dev/ttyS0', baudrate=9600):
        """
        Inicjalizacja połączenia z modułem GPS.
        
        Dla Raspberry Pi Zero 2 W / 3 / 4:
        - UART na GPIO (piny 14/15): zwykle '/dev/ttyS0' lub '/dev/serial0'
        - GPS na USB: zwykle '/dev/ttyUSB0'
        """
        self._last_location: Optional[Tuple[float, float]] = None
        self._last_update_time: Optional[float] = None
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.ser = None

        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=1.0  # Timeout, żeby nie zablokować programu na zawsze
            )
            print(f"[GPS] Połączono z portem {self.serial_port}")
        except serial.SerialException as e:
            print(f"[GPS] Błąd otwarcia portu {self.serial_port}: {e}")

    def get_location(self) -> Optional[Tuple[float, float]]:
        """
        Próbuje pobrać najnowszą dostępną lokalizację z bufora GPS.
        """
        new_location = self._read_gps()

        if new_location is not None:
            self._last_location = new_location
            self._last_update_time = time.time()
            return new_location
        
        # Jeśli nie udało się teraz odczytać, zwróć ostatnią znaną (cache)
        return self._last_location

    def _read_gps(self) -> Optional[Tuple[float, float]]:
        """
        Czyta linie z portu szeregowego i szuka zdania $GPGGA lub $GNGGA.
        """
        if self.ser is None or not self.ser.is_open:
            return None

        try:
            # Czytamy wszystkie dostępne dane z bufora, żeby dostać najświeższe
            # (GPS wysyła dane co 1 sekundę, bufor może mieć stare dane)
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                
                # Interesują nas komunikaty GGA (Fix Data) lub RMC (Recommended Minimum)
                # $GPGGA - GPS, $GNGGA - GNSS (GPS+Glonass etc.)
                if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                    try:
                        msg = pynmea2.parse(line)
                        
                        # Sprawdzamy, czy mamy "fix" (jakość sygnału > 0)
                        # msg.gps_qual: 0 = brak, 1 = GPS, 2 = DGPS
                        if msg.gps_qual > 0 and msg.latitude != 0.0:
                            return (msg.latitude, msg.longitude)
                        
                    except pynmea2.ParseError:
                        continue # Błąd parsowania tej konkretnej linii, ignorujemy
                    except Exception as e:
                        print(f"[GPS] Błąd danych: {e}")
                        continue

        except serial.SerialException:
            print("[GPS] Utracono połączenie z modułem GPS")
            # Opcjonalnie: próba ponownego otwarcia portu
        
        return None

# --- Testowanie bezpośrednie (jeśli uruchomisz ten plik sam) ---
if __name__ == "__main__":
    reader = GPSReader(serial_port='/dev/ttyS0') # Zmień na swój port
    print("Oczekiwanie na FIX GPS (wyjdź z budynków!)...")
    
    try:
        while True:
            loc = reader.get_location()
            if loc:
                print(f"Lokalizacja: {loc}")
            else:
                print("Szukanie satelitów...")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Koniec.")