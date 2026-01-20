from gps.gps_reader import GPSReader
import time

gps = GPSReader()

for i in range(10):
    location = gps.get_location()
    print(f"Pozycja GPS: {location}")
    time.sleep(1)
