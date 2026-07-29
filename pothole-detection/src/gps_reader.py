import serial
import pynmea2
import time
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class GPSReader:
    """Read real GPS data from a serial NMEA device."""

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0):
        """
        Initialize the GPS reader.
        :param port: serial port (e.g., '/dev/ttyUSB0')
        :param baudrate: baud rate (default 9600)
        :param timeout: serial read timeout
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self._connect()

    def _connect(self) -> None:
        """Open serial connection; raise exception on failure."""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            logger.info(f"GPS connected on {self.port}")
        except Exception as e:
            logger.error(f"Failed to open GPS port {self.port}: {e}")
            raise RuntimeError(f"GPS connection error: {e}")

    def get_location(self, max_attempts: int = 20) -> Tuple[float, float, float]:
        """
        Read NMEA sentences until a valid GGA fix is obtained.
        :param max_attempts: maximum lines to read before giving up
        :return: (latitude, longitude, altitude) as floats
        :raises RuntimeError: if no valid fix after max_attempts
        """
        if self.ser is None:
            raise RuntimeError("GPS serial not open.")
        attempts = 0
        while attempts < max_attempts:
            try:
                line = self.ser.readline().decode('ascii', errors='ignore').strip()
                if not line:
                    attempts += 1
                    continue
                if line.startswith('$GPGGA'):
                    msg = pynmea2.parse(line)
                    if msg.gps_qual > 0:  # 0 = invalid, 1 = GPS fix, 2 = DGPS, etc.
                        logger.debug(f"GPS fix: lat={msg.latitude}, lon={msg.longitude}")
                        return (msg.latitude, msg.longitude, msg.altitude)
                    else:
                        logger.debug("GPS signal not fixed yet.")
            except Exception as e:
                logger.warning(f"Error parsing NMEA sentence: {e}")
            attempts += 1
        raise RuntimeError("No valid GPS fix obtained after several attempts.")

    def close(self) -> None:
        """Close the serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.info("GPS serial closed.")