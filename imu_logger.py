"""
imu_logger.py - record two WitMotion WT901C485 IMUs (RS485 / Modbus RTU) to CSV.

Group 2, ACIT4035 - clip-on dual-IMU breathing-rate monitor.

The RS485 sensors never send data by themselves: this script asks each sensor
for its data ("polls" it) at a fixed rate, timestamps every reply and writes
one CSV row per sensor per reading.

Works on Windows (COM5, COM6 ...) and on the Raspberry Pi (/dev/ttyUSB0 ...).

Usage (from a terminal in the folder with this file):
    python imu_logger.py --list-ports           show which serial ports exist
    python imu_logger.py                        record until you press q / Ctrl+C
    python imu_logger.py --duration 300 --name test1
    python imu_logger.py --front COM5 --back COM6 --rate 50

While recording, press a number key (1-9) to drop a time marker, e.g. when a new
phase of the test protocol starts. Press q to stop.
"""

import argparse
import csv
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime, timedelta

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial is not installed. Run:  python -m pip install pyserial")

# --------------------------------------------------------------------------
# SETTINGS - edit these to match your setup
# --------------------------------------------------------------------------

# Default ports: Windows calls them COM5, COM6 ...; the Raspberry Pi /dev/ttyUSB0, /dev/ttyUSB1.
# They are only a starting point: the logger checks every USB serial port and finds
# each sensor by its address, so swapped or renumbered ports don't matter.
if os.name == "nt":
    FRONT_PORT, BACK_PORT = "COM5", "COM6"
else:
    FRONT_PORT, BACK_PORT = "/dev/ttyUSB0", "/dev/ttyUSB1"

SENSORS = [
    # name     serial port   Modbus address
    {"name": "front", "port": FRONT_PORT, "address": 0x50},
    {"name": "back",  "port": BACK_PORT,  "address": 0x51},
]
# If both sensors are daisy-chained on ONE cable, give them the same port.

BAUD_RATE = 115200      # must match the sensors (set in the WitMotion software)
RATE_HZ = 50            # readings per second, per sensor
OUTPUT_DIR = "recordings"

# Labels for the marker keys. Press the key during a recording to log the label.
MARKER_LABELS = {
    "1": "sync lean",
    "2": "sit, normal breathing",
    "3": "breath hold",
    "4": "paced 12/min",
    "5": "paced 18/min",
    "6": "paced 24/min",
    "7": "movement / rocking",
    "8": "walking",
    "9": "other",
}

# --------------------------------------------------------------------------
# WitMotion Modbus register map (standard WitMotion protocol)
# One read of 33 registers, 0x34 ... 0x54, gives everything in one reply:
#   0x34-0x36 acceleration X,Y,Z     0x37-0x39 angular velocity X,Y,Z
#   0x3A-0x3C magnetic field X,Y,Z   0x3D-0x3F angle X (roll), Y (pitch), Z (yaw)
#   0x40      temperature            0x41-0x50 ports/pressure/GPS (unused)
#   0x51-0x54 quaternion q0..q3
# Each register is a signed 16-bit number, high byte first.
# --------------------------------------------------------------------------

FIRST_REGISTER = 0x34
REGISTER_COUNT = 0x54 - 0x34 + 1          # 33
REPLY_LENGTH = 5 + 2 * REGISTER_COUNT     # address, function, byte count, data, 2 CRC

CSV_COLUMNS = [
    "time_s", "wall_time", "sensor", "address",
    "acc_x_g", "acc_y_g", "acc_z_g",
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    "angle_x_deg", "angle_y_deg", "angle_z_deg",
    "mag_x_raw", "mag_y_raw", "mag_z_raw",
    "temp_c",
    "q0", "q1", "q2", "q3",
]


def crc16_modbus(data: bytes) -> int:
    """Standard Modbus RTU checksum."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def build_read_request(address: int, register: int, count: int) -> bytes:
    """Modbus function 0x03 'read holding registers', CRC low byte first."""
    frame = bytes([address, 0x03, register >> 8, register & 0xFF, count >> 8, count & 0xFF])
    crc = crc16_modbus(frame)
    return frame + bytes([crc & 0xFF, crc >> 8])


def parse_reply(reply: bytes, address: int):
    """Check a reply and convert it to physical units. Returns a dict or None."""
    if len(reply) != REPLY_LENGTH:
        return None
    if reply[0] != address or reply[1] != 0x03 or reply[2] != 2 * REGISTER_COUNT:
        return None
    crc = crc16_modbus(reply[:-2])
    if reply[-2] != (crc & 0xFF) or reply[-1] != (crc >> 8):
        return None

    regs = []
    for i in range(REGISTER_COUNT):
        value = (reply[3 + 2 * i] << 8) | reply[4 + 2 * i]
        if value >= 0x8000:
            value -= 0x10000          # signed 16-bit
        regs.append(value)

    def r(register):
        return regs[register - FIRST_REGISTER]

    return {
        "acc_x_g": r(0x34) / 32768 * 16,
        "acc_y_g": r(0x35) / 32768 * 16,
        "acc_z_g": r(0x36) / 32768 * 16,
        "gyro_x_dps": r(0x37) / 32768 * 2000,
        "gyro_y_dps": r(0x38) / 32768 * 2000,
        "gyro_z_dps": r(0x39) / 32768 * 2000,
        "mag_x_raw": r(0x3A),
        "mag_y_raw": r(0x3B),
        "mag_z_raw": r(0x3C),
        "angle_x_deg": r(0x3D) / 32768 * 180,
        "angle_y_deg": r(0x3E) / 32768 * 180,
        "angle_z_deg": r(0x3F) / 32768 * 180,
        "temp_c": r(0x40) / 100,
        "q0": r(0x51) / 32768,
        "q1": r(0x52) / 32768,
        "q2": r(0x53) / 32768,
        "q3": r(0x54) / 32768,
    }


class PortPoller(threading.Thread):
    """Polls all sensors on one serial port at a fixed rate.

    One thread per port, so sensors on separate cables are read in parallel.
    """

    def __init__(self, port, sensors, rate_hz, t_start, out_queue, stop_event):
        super().__init__(daemon=True)
        self.port = port
        self.sensors = sensors
        self.period = 1.0 / rate_hz
        self.t_start = t_start
        self.out = out_queue
        self.stop_event = stop_event
        self.requests = {s["name"]: build_read_request(s["address"], FIRST_REGISTER, REGISTER_COUNT)
                         for s in sensors}
        self.stats = {s["name"]: {"ok": 0, "errors": 0, "last": None} for s in sensors}
        self.missed_deadlines = 0
        # The reply takes ~7 ms at 115200 baud; allow a generous margin.
        self.ser = serial.Serial(port, BAUD_RATE, timeout=0.05)

    def poll(self, sensor):
        self.ser.reset_input_buffer()
        t_send = time.perf_counter()
        self.ser.write(self.requests[sensor["name"]])
        reply = self.ser.read(REPLY_LENGTH)
        t_recv = time.perf_counter()
        data = parse_reply(reply, sensor["address"])
        stats = self.stats[sensor["name"]]
        if data is None:
            stats["errors"] += 1
            return
        stats["ok"] += 1
        stats["last"] = data
        # The sensor's measurement happened between sending and receiving.
        data["time_s"] = (t_send + t_recv) / 2 - self.t_start
        data["sensor"] = sensor["name"]
        data["address"] = f"0x{sensor['address']:02X}"
        self.out.put(data)

    def run(self):
        next_tick = self.t_start
        wait = self.t_start - time.perf_counter()
        if wait > 0:
            time.sleep(wait)              # all ports start together at t_start
        try:
            while not self.stop_event.is_set():
                for sensor in self.sensors:
                    self.poll(sensor)
                next_tick += self.period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    self.missed_deadlines += 1
                    if delay < -self.period:      # far behind: restart the schedule
                        next_tick = time.perf_counter()
        except serial.SerialException as e:
            print(f"\nERROR on {self.port}: {e}")
            self.stop_event.set()
        finally:
            self.ser.close()


def key_pressed():
    """Return a pressed key without waiting (Windows, or Linux/Pi terminal), else None."""
    if os.name == "nt":
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getwch()
        return None
    import select
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def list_serial_ports():
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found. Is the USB cable plugged in and the CH340 driver installed?")
    for p in ports:
        print(f"  {p.device:15s} {p.description}")


def find_sensor_ports(sensors):
    """Ask every given port for every sensor address and return {name: port}.

    COM numbers belong to the USB socket, not the sensor, so they can swap when
    cables are moved. This finds each sensor wherever it is.
    """
    ports = sorted({s["port"] for s in sensors})
    # Also try every other USB serial port on the computer (p.vid is set only for USB devices)
    ports += sorted(p.device for p in list_ports.comports() if p.vid is not None and p.device not in ports)
    found = {}
    for port in ports:
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
        except serial.SerialException as e:
            print(f"  {port}: could not open ({e})")
            continue
        for s in sensors:
            if s["name"] in found:
                continue
            for _ in range(3):
                ser.reset_input_buffer()
                ser.write(build_read_request(s["address"], FIRST_REGISTER, REGISTER_COUNT))
                if parse_reply(ser.read(REPLY_LENGTH), s["address"]) is not None:
                    found[s["name"]] = port
                    break
        ser.close()
    return found


def sanity_check(name, data):
    """Sensor at rest: |acceleration| should be ~1 g and |quaternion| ~1."""
    acc = math.sqrt(data["acc_x_g"] ** 2 + data["acc_y_g"] ** 2 + data["acc_z_g"] ** 2)
    qn = math.sqrt(sum(data[k] ** 2 for k in ("q0", "q1", "q2", "q3")))
    ok = 0.9 < acc < 1.1 and 0.95 < qn < 1.05
    status = "OK" if ok else "CHECK - values look wrong, compare with the WitMotion software"
    print(f"  {name:6s} |acc| = {acc:.3f} g, |q| = {qn:.3f}, angles = "
          f"{data['angle_x_deg']:.1f} / {data['angle_y_deg']:.1f} / {data['angle_z_deg']:.1f} deg  -> {status}")


def main():
    parser = argparse.ArgumentParser(description="Record two WitMotion RS485 IMUs to CSV.")
    parser.add_argument("--list-ports", action="store_true", help="show serial ports and exit")
    parser.add_argument("--front", help="serial port of the front sensor (e.g. COM5)")
    parser.add_argument("--back", help="serial port of the back sensor (e.g. COM6)")
    parser.add_argument("--rate", type=float, default=RATE_HZ, help="readings per second per sensor")
    parser.add_argument("--duration", type=float, default=0, help="seconds to record (0 = until q)")
    parser.add_argument("--name", default="recording", help="label used in the file name")
    args = parser.parse_args()

    if args.list_ports:
        list_serial_ports()
        return

    sensors = [dict(s) for s in SENSORS]
    for s in sensors:
        if s["name"] == "front" and args.front:
            s["port"] = args.front
        if s["name"] == "back" and args.back:
            s["port"] = args.back

    # Check which port each sensor is actually on (handles swapped COM numbers).
    print("Looking for sensors...")
    found = find_sensor_ports(sensors)
    for s in sensors:
        if s["name"] not in found:
            print(f"  {s['name']} (0x{s['address']:02X}) did not answer on any USB serial port.")
            print("  Check the cable and power, that the baud rate is "
                  f"{BAUD_RATE}, and that the WitMotion software is fully closed (Task Manager).")
            return
        if found[s["name"]] != s["port"]:
            print(f"  {s['name']} (0x{s['address']:02X}) found on {found[s['name']]} instead of {s['port']} - using that.")
        s["port"] = found[s["name"]]

    # Group sensors by port: one polling thread per port.
    by_port = {}
    for s in sensors:
        by_port.setdefault(s["port"], []).append(s)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_path = os.path.join(OUTPUT_DIR, f"{stamp}_{args.name}.csv")
    marker_path = os.path.join(OUTPUT_DIR, f"{stamp}_{args.name}_markers.csv")

    out_queue = queue.Queue()
    stop_event = threading.Event()
    t_start = time.perf_counter() + 0.2       # small head start so all threads begin together
    wall_start = datetime.now() + timedelta(seconds=0.2)

    pollers = []
    try:
        for port, port_sensors in by_port.items():
            pollers.append(PortPoller(port, port_sensors, args.rate, t_start, out_queue, stop_event))
    except serial.SerialException as e:
        print(f"Could not open serial port: {e}")
        print("Available ports:")
        list_serial_ports()
        print("Close the WitMotion software if it is open - only one program can use a port.")
        return

    print(f"Recording to {data_path}")
    print(f"Sensors: " + ", ".join(f"{s['name']} 0x{s['address']:02X} on {s['port']}" for s in sensors))
    print(f"Target rate: {args.rate:g} Hz per sensor. Press 1-9 for markers, q to stop.\n")
    for key, label in MARKER_LABELS.items():
        print(f"  {key} = {label}")
    print()

    # On Linux (the Pi, also over SSH) the terminal normally waits for Enter.
    # Switch it to single-key mode while recording and restore it afterwards.
    old_terminal = None
    if os.name != "nt" and sys.stdin.isatty():
        import termios
        import tty
        old_terminal = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    for p in pollers:
        p.start()

    stats = {}
    for p in pollers:
        stats.update(p.stats)
    checked = set()
    markers = []
    rows_written = 0
    last_status = time.perf_counter()
    last_counts = {name: 0 for name in stats}

    with open(data_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        try:
            while not stop_event.is_set():
                # Write everything the pollers have read so far.
                while not out_queue.empty():
                    row = out_queue.get_nowait()
                    row["wall_time"] = (wall_start + timedelta(seconds=row["time_s"])).strftime("%H:%M:%S.%f")[:-3]
                    writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
                    rows_written += 1
                    if row["sensor"] not in checked:
                        checked.add(row["sensor"])
                        sanity_check(row["sensor"], row)
                time.sleep(0.02)

                now = time.perf_counter()
                elapsed = now - t_start

                key = key_pressed()
                if key in ("q", "Q"):
                    break
                if key in MARKER_LABELS:
                    markers.append((round(elapsed, 3), key, MARKER_LABELS[key]))
                    print(f"\n  marker {key} at {elapsed:7.1f} s: {MARKER_LABELS[key]}")

                if now - last_status >= 1.0:
                    parts = []
                    for name, st in stats.items():
                        rate = (st["ok"] - last_counts[name]) / (now - last_status)
                        last_counts[name] = st["ok"]
                        parts.append(f"{name}: {rate:5.1f} Hz, {st['errors']} err")
                    print(f"\r  t = {elapsed:6.1f} s | " + " | ".join(parts) + "   ", end="", flush=True)
                    last_status = now
                    f.flush()

                if args.duration and elapsed >= args.duration:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            if old_terminal is not None:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal)
            for p in pollers:
                p.join(timeout=1)
            while not out_queue.empty():            # write anything still waiting
                row = out_queue.get()
                row["wall_time"] = (wall_start + timedelta(seconds=row["time_s"])).strftime("%H:%M:%S.%f")[:-3]
                writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
                rows_written += 1

    with open(marker_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "key", "label"])
        w.writerows(markers)

    total = time.perf_counter() - t_start
    print(f"\n\nStopped after {total:.1f} s. {rows_written} rows written to {data_path}")
    for name, st in stats.items():
        print(f"  {name:6s}: {st['ok']} readings ({st['ok'] / total:.1f} Hz average), {st['errors']} errors")
    missed = sum(p.missed_deadlines for p in pollers)
    if missed:
        print(f"  Note: the polling loop was late {missed} times - if this is large, try a lower --rate.")
    print(f"  {len(markers)} markers written to {marker_path}")


if __name__ == "__main__":
    main()
