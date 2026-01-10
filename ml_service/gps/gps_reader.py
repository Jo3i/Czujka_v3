from typing import Optional, Tuple
import time
import random

class GPSReader:
    """
    Moduł odpowiedzialny za odczyt lokalizacji GPS czujki terenowej.

    Wersja: symulowana (bez fizycznego modułu GPS).
    Interfejs zgodny z docelową wersją sprzętową.
    """

    def __init__(self):
        self._last_location: Optional[Tuple[float, float]] = None
        self._last_update_time: Optional[float] = None

    def get_location(self) -> Optional[Tuple[float, float]]:
        """
        Zwraca aktualną lub ostatnią znaną lokalizację GPS.

        Returns:
            (latitude, longitude) lub None, jeśli brak danych
        """

        location = self._read_gps()

        if location is not None:
            self._last_location = location
            self._last_update_time = time.time()
            return location

        # Brak nowego sygnału GPS — zwracamy ostatnią znaną pozycję
        return self._last_location

    def _read_gps(self) -> Optional[Tuple[float, float]]:
        """
        Symulacja odczytu GPS.

        Docelowo:
        - UART / USB / NMEA
        - biblioteka np. gpsd

        Na potrzeby projektu:
        - losowa pozycja
        - czasami brak sygnału
        """

        # Symulacja braku sygnału GPS (np. w lesie)
        if random.random() < 0.2:
            return None

        # Przykładowa lokalizacja (las / teren wiejski)
        latitude = 52.1 + random.uniform(-0.005, 0.005)
        longitude = 21.0 + random.uniform(-0.005, 0.005)

        return latitude, longitude
