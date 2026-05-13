"""
scooter_commander.py
====================
Sends steering angle (degrees) and speed (m/s) to the scooter over serial.

Protocol: each command is a line:
    CMD,<steer_deg>,<speed_mps>\n
Example:
    CMD,-12.5,1.2\n    means steer 12.5 deg left at 1.2 m/s
    CMD,0.0,0.0\n      means stop
"""


class ScooterCommander:
    """
    Sends steering angle (degrees) and speed (m/s) to the scooter
    over a serial connection.
    """

    def __init__(self, port=None, baud=115200):
        self.ser = None
        self.port = port
        if port:
            try:
                import serial as pyserial
                self.ser = pyserial.Serial(port, baud, timeout=0.1)
                print(f"[Scooter] Serial connected: {port} @ {baud}")
            except ImportError:
                print("[Scooter] WARNING: pyserial not installed.")
            except Exception as e:
                print(f"[Scooter] ERROR: {e}")

    def send_command(self, steer_deg, speed_mps):
        """Send steering + speed command. Returns the command string."""
        cmd = f"CMD,{steer_deg:.1f},{speed_mps:.2f}\n"
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(cmd.encode())
            except Exception as e:
                print(f"[Scooter] Write error: {e}")
        return cmd.strip()

    def stop(self):
        """Emergency stop."""
        self.send_command(0.0, 0.0)
        if self.ser:
            self.ser.close()
