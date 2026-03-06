"""
gps_navigator.py
================
Reads NMEA sentences from a serial GPS and steers toward waypoints.
Waypoints loaded from CSV: lat,lon[,name]
"""

import math
import threading
import time

from config import EARTH_RADIUS_M


class GPSNavigator:
    """
    Reads NMEA sentences from a serial GPS and steers toward waypoints.
    Waypoints loaded from CSV: lat,lon[,name]
    """

    def __init__(self, serial_device=None, baud=9600, waypoints_file=None):
        self.lat = None
        self.lon = None
        self.speed_mps = 0.0
        self.heading_gps = 0.0  # degrees from north (from GPS RMC)
        self.fix_quality = 0
        self.hdop = 99.0
        self.last_update = 0.0
        self.lock = threading.Lock()
        self._running = False
        self._thread = None

        # Waypoints
        self.waypoints = []  # list of (lat, lon, name)
        self.current_wp_idx = 0
        self.wp_reached_radius_m = 5.0

        if waypoints_file:
            self._load_waypoints(waypoints_file)

        if serial_device:
            self._start_serial(serial_device, baud)

    def _load_waypoints(self, path):
        """Load waypoints from CSV: lat,lon[,name]"""
        self.waypoints = []
        try:
            with open(path, "r") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    lat = float(parts[0])
                    lon = float(parts[1])
                    name = parts[2].strip() if len(parts) > 2 else f"WP{i}"
                    self.waypoints.append((lat, lon, name))
            print(f"[GPS] Loaded {len(self.waypoints)} waypoints from {path}")
        except Exception as e:
            print(f"[GPS] ERROR loading waypoints: {e}")

    def _start_serial(self, device, baud):
        """Start background thread reading NMEA from serial."""
        try:
            import serial as pyserial
            self._ser = pyserial.Serial(device, baud, timeout=1)
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            print(f"[GPS] Reading from {device} @ {baud} baud")
        except ImportError:
            print("[GPS] WARNING: pyserial not installed. Run: pip install pyserial")
            print("[GPS] GPS DISABLED.")
        except Exception as e:
            print(f"[GPS] ERROR opening {device}: {e}")

    def _read_loop(self):
        """Background thread: read NMEA sentences."""
        while self._running:
            try:
                line = self._ser.readline().decode(errors="ignore").strip()
                if not line.startswith("$"):
                    continue
                self._parse_nmea(line)
            except Exception as e:
                print(f"[GPS] Read error: {e}")
                time.sleep(0.1)

    def _parse_nmea(self, sentence):
        """Parse GGA and RMC sentences."""
        parts = sentence.split(",")
        with self.lock:
            if parts[0].endswith("GGA") and len(parts) >= 10:
                lat = self._nmea_to_decimal(parts[2], parts[3])
                lon = self._nmea_to_decimal(parts[4], parts[5])
                if lat is not None and lon is not None:
                    self.lat = lat
                    self.lon = lon
                    self.last_update = time.time()
                try:
                    self.fix_quality = int(parts[6])
                except (ValueError, IndexError):
                    pass
                try:
                    hdop_val = float(parts[8])
                    self.hdop = hdop_val
                    if hdop_val > 5.0:
                        print(f"[GPS] WARNING: Poor accuracy (HDOP={hdop_val:.1f} > 5.0)")
                except (ValueError, IndexError):
                    pass

            elif parts[0].endswith("RMC") and len(parts) >= 8:
                lat = self._nmea_to_decimal(parts[3], parts[4])
                lon = self._nmea_to_decimal(parts[5], parts[6])
                if lat is not None and lon is not None:
                    self.lat = lat
                    self.lon = lon
                    self.last_update = time.time()
                try:
                    self.speed_mps = float(parts[7]) * 0.514444  # knots to m/s
                except (ValueError, IndexError):
                    pass
                try:
                    self.heading_gps = float(parts[8])
                except (ValueError, IndexError):
                    pass

    @staticmethod
    def _nmea_to_decimal(raw, hemi):
        if not raw:
            return None
        try:
            raw = float(raw)
        except ValueError:
            return None
        deg = int(raw // 100)
        minutes = raw - deg * 100
        val = deg + minutes / 60.0
        if hemi in ("S", "W"):
            val = -val
        return val

    def get_position(self):
        """Returns (lat, lon) or (None, None)."""
        with self.lock:
            return self.lat, self.lon

    def get_bearing_to_waypoint(self):
        """
        Returns (bearing_deg, distance_m, wp_name) to the current waypoint.
        bearing_deg: 0=North, 90=East, etc.
        Returns (None, None, None) if no GPS fix or no waypoints.
        """
        with self.lock:
            lat, lon = self.lat, self.lon

        if lat is None or lon is None:
            return None, None, None
        if self.current_wp_idx >= len(self.waypoints):
            return None, None, "ARRIVED"

        wp_lat, wp_lon, wp_name = self.waypoints[self.current_wp_idx]
        bearing = self._bearing(lat, lon, wp_lat, wp_lon)
        dist = self._haversine(lat, lon, wp_lat, wp_lon)

        # Auto-advance to next waypoint if close enough
        if dist < self.wp_reached_radius_m:
            print(f"[GPS] Reached waypoint: {wp_name} ({dist:.1f}m)")
            self.current_wp_idx += 1
            if self.current_wp_idx < len(self.waypoints):
                next_wp = self.waypoints[self.current_wp_idx]
                print(f"[GPS] Next waypoint: {next_wp[2]}")
            else:
                print("[GPS] All waypoints reached!")

        return bearing, dist, wp_name

    def get_gps_heading_correction(self):
        """
        Returns a correction angle (degrees) to steer toward the next waypoint.
        Positive = turn right, negative = turn left.
        Returns (0.0, None, None) if no GPS data or no valid fix.
        Returns (0.0, None, "ARRIVED") if all waypoints reached.
        """
        if self.fix_quality < 1:
            return 0.0, None, None  # No valid fix, skip GPS correction

        bearing, dist, wp_name = self.get_bearing_to_waypoint()
        if bearing is None:
            # Preserve wp_name so "ARRIVED" state is communicated
            return 0.0, dist, wp_name

        with self.lock:
            current_heading = self.heading_gps

        # Difference between desired bearing and current heading
        diff = bearing - current_heading
        # Normalize to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        return diff, dist, wp_name

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        """Distance in meters between two GPS coordinates."""
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _bearing(lat1, lon1, lat2, lon2):
        """Bearing in degrees from (lat1,lon1) to (lat2,lon2). 0=N, 90=E."""
        dlon = math.radians(lon2 - lon1)
        lat1r, lat2r = math.radians(lat1), math.radians(lat2)
        x = math.sin(dlon) * math.cos(lat2r)
        y = (math.cos(lat1r) * math.sin(lat2r) -
             math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        # Close the serial port if open
        if hasattr(self, "_ser") and self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception as e:
                print(f"[GPS] Serial close error: {e}")
