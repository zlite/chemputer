"""
Physical Robot Lab — Raspberry Pi Flask Server
Controls a RAMPS 1.4 board via serial for RGB liquid mixing experiments.

Axis mapping (same as original tkinter app):
  R  → X axis (Red pump)
  G  → Y axis (Green pump)
  B  → Z axis (Blue pump)
  E  → E axis (auxiliary / extrude)
  Empty → fan pulse + extrude

Run on Pi (first time):
  bash setup.sh
Then every time:
  .venv/bin/python app.py
Then open http://<pi-ip>:5000 in a browser.
"""

import glob
import json
import math
import os
import queue
import threading
import time
from collections import deque
from typing import Optional

import serial
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from learning import (
    ALGORITHMS,
    color_distance,
    calculate_score,
    normalized_rgb_to_hex,
    create_algorithm_state,
    get_algorithm_initial_guess,
    get_algorithm_next_guess,
)

# ─────────────────────────────────────────── constants ──

BAUD = 250_000
HOME_POSITION = -7    # mm — endstops stop axes here
HOME_FEED = 20        # mm/min for homing

AXIS_MAP = {"R": "X", "G": "Y", "B": "Z", "E": "E"}

# How many mm of pump travel equals one "full tube" of liquid
# Adjust this to calibrate your actual pump displacement
TOTAL_VOLUME_MM = 0.1

# Region of interest (fraction of frame) for color sampling
# [x_start, y_start, x_end, y_end] as fractions 0-1
DEFAULT_ROI = [0.35, 0.35, 0.65, 0.65]

# Camera device index (0 = default, or Pi camera via v4l2)
CAMERA_INDEX = 0

# RP2040 LED simulator
LED_BAUD = 115_200

# File where camera-calibrated color targets are persisted
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")

# ─────────────────────────────────────────── Flask app ──

app = Flask(__name__)

# ─────────────────────────────── calibration storage ──

_cal_lock = threading.Lock()

def _load_calibration() -> dict:
    try:
        with open(CALIBRATION_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_calibration(data: dict):
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)

calibration: dict = _load_calibration()

# ─────────────────────────────────────── global state ──

class RobotState:
    def __init__(self):
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        self.port_name = ""
        self._lock = threading.Lock()

        self.log_queue: deque = deque(maxlen=500)
        self._sse_queues: list[queue.Queue] = []
        self._sse_lock = threading.Lock()

        # Experiment state
        self.experiment_running = False
        self.experiment_thread: Optional[threading.Thread] = None

        # LED simulator (RP2040)
        self.led_ser: Optional[serial.Serial] = None
        self.led_connected = False
        self.led_lock = threading.Lock()

        # Camera state
        self.camera = None
        self.camera_lock = threading.Lock()
        self.roi = list(DEFAULT_ROI)

        self.step_mm = 0.5
        self.feed_mm_min = 20.0

    # ── serial helpers ──────────────────────────────────

    def connect(self) -> dict:
        with self._lock:
            try:
                if self.ser and self.ser.is_open:
                    self.connected = False
                    self.ser.close()

                port = _find_serial_port()
                if port is None:
                    raise serial.SerialException(
                        "No USB serial device found (/dev/ttyUSB* or /dev/ttyACM*)"
                    )

                self.ser = serial.Serial(port, BAUD, timeout=1)
                time.sleep(2)  # wait for board reset after DTR toggle

                self.connected = True
                self.port_name = port
                self._log(f"Connected to {port} at {BAUD} baud", "info")

                threading.Thread(target=self._reader, daemon=True).start()

                # Startup sequence
                self._send_raw("M302 S0")  # allow cold extrusion
                self._send_raw("M120")      # enable endstops
                self._send_raw("G91")       # relative positioning

                return {"ok": True, "port": port}

            except serial.SerialException as exc:
                self.connected = False
                self.port_name = ""
                self._log(f"Connection error: {exc}", "error")
                return {"ok": False, "error": str(exc)}

    def disconnect(self):
        with self._lock:
            self.connected = False
            if self.ser and self.ser.is_open:
                self.ser.close()
            self._log("Disconnected", "info")

    def send(self, cmd: str) -> bool:
        with self._lock:
            return self._send_raw(cmd)

    def _send_raw(self, cmd: str) -> bool:
        if not self.connected or not self.ser or not self.ser.is_open:
            self._log("Not connected — command dropped", "error")
            return False
        try:
            self.ser.write((cmd.strip() + "\n").encode())
            self._log(f">> {cmd}", "sent")
            return True
        except serial.SerialException as exc:
            self._log(f"Send error: {exc}", "error")
            return False

    def _reader(self):
        while self.connected and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode(errors="replace").strip()
                    if line:
                        self._log(f"<< {line}", "recv")
            except Exception:
                break
            time.sleep(0.01)

    # ── camera helpers ──────────────────────────────────

    def open_camera(self):
        try:
            from picamera2 import Picamera2
            import cv2
            with self.camera_lock:
                if self.camera is None:
                    cam = Picamera2()
                    config = cam.create_preview_configuration(
                        main={"size": (640, 480), "format": "RGB888"}
                    )
                    cam.configure(config)
                    cam.start()
                    self.camera = cam
            return True
        except Exception:
            return False

    def get_frame_jpeg(self) -> Optional[bytes]:
        try:
            import cv2
            import numpy as np
            with self.camera_lock:
                if self.camera is None:
                    return None
                frame = self.camera.capture_array()
                # picamera2 "RGB888" format returns BGR array (OpenCV convention)
                # Draw ROI rectangle
                h, w = frame.shape[:2]
                x1 = int(self.roi[0] * w)
                y1 = int(self.roi[1] * h)
                x2 = int(self.roi[2] * w)
                y2 = int(self.roi[3] * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                return buf.tobytes()
        except Exception:
            return None

    def sample_roi_color(self) -> Optional[tuple[float, float, float]]:
        """Return mean normalized RGB (0-1) from the ROI, or None."""
        try:
            import cv2
            import numpy as np
            with self.camera_lock:
                if self.camera is None:
                    return None
                frame = self.camera.capture_array()
                # picamera2 "RGB888" format returns BGR array (OpenCV convention)
                h, w = frame.shape[:2]
                x1 = int(self.roi[0] * w)
                y1 = int(self.roi[1] * h)
                x2 = int(self.roi[2] * w)
                y2 = int(self.roi[3] * h)
                roi_crop = frame[y1:y2, x1:x2]
                mean_bgr = cv2.mean(roi_crop)[:3]
                # BGR → RGB, normalize 0-1
                b, g, r = mean_bgr
                return (r / 255.0, g / 255.0, b / 255.0)
        except Exception:
            return None

    # ── LED simulator ───────────────────────────────────

    def open_led(self) -> dict:
        with self.led_lock:
            try:
                if self.led_ser and self.led_ser.is_open:
                    self.led_ser.close()
                port = _find_led_port(skip=self.port_name)
                if port is None:
                    return {"ok": False, "error": "No RP2040 found on ttyACM*"}
                self.led_ser = serial.Serial(port, LED_BAUD, timeout=1)
                time.sleep(1)  # wait for RP2040 reset after DTR
                self.led_connected = True
                self._log(f"LED simulator connected on {port}", "info")
                return {"ok": True, "port": port}
            except serial.SerialException as exc:
                self.led_connected = False
                self._log(f"LED connect error: {exc}", "error")
                return {"ok": False, "error": str(exc)}

    def close_led(self):
        with self.led_lock:
            self.led_connected = False
            if self.led_ser and self.led_ser.is_open:
                self.led_ser.close()
            self._log("LED simulator disconnected", "info")

    def set_led_rgb(self, r: int, g: int, b: int) -> bool:
        """Send R,G,B (0–25) to the RP2040 as 'R,G,B\r'."""
        with self.led_lock:
            if not self.led_ser or not self.led_ser.is_open:
                return False
            try:
                self.led_ser.write(f"{r},{g},{b}\r".encode())
                return True
            except serial.SerialException:
                return False

    # ── SSE broadcast ───────────────────────────────────

    def _log(self, msg: str, tag: str = "info"):
        entry = {"ts": time.strftime("%H:%M:%S"), "msg": msg, "tag": tag}
        self.log_queue.append(entry)
        self._broadcast("log", entry)

    def _broadcast(self, event: str, data: dict):
        with self._sse_lock:
            dead = []
            for q in self._sse_queues:
                try:
                    q.put_nowait({"event": event, "data": data})
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._sse_queues.remove(q)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._sse_lock:
            self._sse_queues.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._sse_lock:
            if q in self._sse_queues:
                self._sse_queues.remove(q)

    def broadcast_experiment(self, event: str, data: dict):
        self._broadcast(event, data)


robot = RobotState()


# ─────────────────────────────────────── helpers ──

def _find_serial_port() -> Optional[str]:
    candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    return candidates[0] if candidates else None


def _find_led_port(skip: str = "") -> Optional[str]:
    """Find RP2040 on a ttyACM port. Prefers by VID, falls back to any ACM not in use."""
    from serial.tools import list_ports
    RP2040_VID = 0x2E8A  # Raspberry Pi USB vendor ID
    acm_ports = sorted(p for p in glob.glob("/dev/ttyACM*") if p != skip)
    # Prefer a port whose USB VID matches the RP2040
    for info in list_ports.comports():
        if info.vid == RP2040_VID and info.device in acm_ports:
            return info.device
    # Fall back to first available ACM port not used by RAMPS
    return acm_ports[0] if acm_ports else None


def _pump_rgb(r_pct: float, g_pct: float, b_pct: float, feed: float = 20.0):
    """Dispense R/G/B proportions. Values are 0-100 percentages summing to 100."""
    r_mm = TOTAL_VOLUME_MM * r_pct / 100.0
    g_mm = TOTAL_VOLUME_MM * g_pct / 100.0
    b_mm = TOTAL_VOLUME_MM * b_pct / 100.0

    if r_mm > 0.001:
        robot.send(f"G1 X{r_mm:.3f} F{feed:.0f}")
        time.sleep(0.1)
    if g_mm > 0.001:
        robot.send(f"G1 Y{g_mm:.3f} F{feed:.0f}")
        time.sleep(0.1)
    if b_mm > 0.001:
        robot.send(f"G1 Z{b_mm:.3f} F{feed:.0f}")
        time.sleep(0.1)
    robot.send("M18")  # disable steppers


def _fan_pulse_and_mix():
    """Fan on for 2s (mixing) then off. Also extrudes 0.5mm to stir."""
    robot.send("M106 S255")
    robot.send("G1 E0.5 F20")
    time.sleep(2.2)
    robot.send("M106 S0")
    robot.send("M18")


def _home_all():
    robot.send(f"G1 X{HOME_POSITION} Y{HOME_POSITION} Z{HOME_POSITION} F{HOME_FEED}")
    time.sleep(1.0)
    robot.send("G92 X0 Y0 Z0 E0")
    robot.send("M18")


# ─────────────────────────────────── experiment runner ──

def _run_experiment(target_name: str, target_rgb: dict, algorithm_name: str,
                    max_trials: int, feed: float, use_camera: bool,
                    led_mode: bool = False, also_move_steppers: bool = False):
    """Background thread: runs one full color-matching experiment."""
    try:
        robot.experiment_running = True
        target = (target_rgb["r"], target_rgb["g"], target_rgb["b"])

        robot.broadcast_experiment("experiment_start", {
            "target_name": target_name,
            "target_rgb": target_rgb,
            "algorithm": algorithm_name,
            "max_trials": max_trials,
            "led_mode": led_mode,
        })

        if led_mode:
            robot.broadcast_experiment("reasoning", {"msg": "LED mode — no homing needed"})
        else:
            robot.broadcast_experiment("reasoning", {"msg": "Homing all axes..."})
            _home_all()
            time.sleep(0.5)

        history = []
        algo_state = create_algorithm_state(algorithm_name)
        guess = get_algorithm_initial_guess(algorithm_name, algo_state, target, target_name, [])

        for trial in range(1, max_trials + 1):
            if not robot.experiment_running:
                break

            if led_mode:
                # Each channel independent (0-1 → 0-25); no normalisation
                r_val = int(max(0.0, min(1.0, guess[0])) * 25)
                g_val = int(max(0.0, min(1.0, guess[1])) * 25)
                b_val = int(max(0.0, min(1.0, guess[2])) * 25)
                r_pct = round(guess[0] * 100, 1)
                g_pct = round(guess[1] * 100, 1)
                b_pct = round(guess[2] * 100, 1)

                robot.broadcast_experiment("trial_start", {
                    "trial": trial,
                    "r": r_pct, "g": g_pct, "b": b_pct,
                })
                robot.broadcast_experiment("reasoning", {
                    "msg": f"Trial {trial}: LED R={r_val} G={g_val} B={b_val}"
                })

                robot.set_led_rgb(r_val, g_val, b_val)

                if also_move_steppers and robot.connected:
                    total = guess[0] + guess[1] + guess[2]
                    if total > 0:
                        sp = [v / total * 100 for v in guess[:3]]
                    else:
                        sp = [33.3, 33.3, 33.4]
                    robot.broadcast_experiment("reasoning", {
                        "msg": f"Debug: moving steppers R={sp[0]:.1f}% G={sp[1]:.1f}% B={sp[2]:.1f}%"
                    })
                    _pump_rgb(sp[0], sp[1], sp[2], feed)

                time.sleep(0.3)  # let camera settle on new colour

                # Score — camera required in LED mode
                robot.broadcast_experiment("reasoning", {"msg": "Sampling colour from camera..."})
                sampled = robot.sample_roi_color()
                if sampled:
                    sr, sg, sb = sampled
                    dist = color_distance(sampled, target)
                    score = calculate_score(dist)
                    mix_hex = normalized_rgb_to_hex(sr, sg, sb)
                else:
                    sr, sg, sb = guess[0], guess[1], guess[2]
                    dist = color_distance((sr, sg, sb), target)
                    score = calculate_score(dist)
                    mix_hex = normalized_rgb_to_hex(sr, sg, sb)
                    robot.broadcast_experiment("reasoning", {
                        "msg": "Camera unavailable — using LED values as estimate"
                    })

                step_result = {
                    "trial": trial,
                    "r": r_pct, "g": g_pct, "b": b_pct,
                    "score": round(score, 1),
                    "mix_hex": mix_hex,
                }
                history.append({"rgb": (guess[0], guess[1], guess[2]), "score": score})

            else:
                r_pct = round(guess[0] * 100, 1)
                g_pct = round(guess[1] * 100, 1)
                b_pct = round(guess[2] * 100, 1)

                # Normalise so percents sum to 100
                total = r_pct + g_pct + b_pct
                if total > 0:
                    r_pct = r_pct / total * 100
                    g_pct = g_pct / total * 100
                    b_pct = b_pct / total * 100
                else:
                    r_pct, g_pct, b_pct = 33.3, 33.3, 33.4

                robot.broadcast_experiment("trial_start", {
                    "trial": trial,
                    "r": round(r_pct, 1), "g": round(g_pct, 1), "b": round(b_pct, 1),
                })
                robot.broadcast_experiment("reasoning", {
                    "msg": f"Trial {trial}: pumping R={r_pct:.1f}% G={g_pct:.1f}% B={b_pct:.1f}%"
                })

                _pump_rgb(r_pct, g_pct, b_pct, feed)

                robot.broadcast_experiment("reasoning", {"msg": "Mixing (fan pulse)..."})
                _fan_pulse_and_mix()
                time.sleep(0.5)

                if use_camera:
                    robot.broadcast_experiment("reasoning", {"msg": "Sampling color from camera..."})
                    sampled = robot.sample_roi_color()
                    if sampled:
                        sr, sg, sb = sampled
                        dist = color_distance(sampled, target)
                        score = calculate_score(dist)
                        mix_hex = normalized_rgb_to_hex(sr, sg, sb)
                    else:
                        sr, sg, sb = r_pct / 100.0, g_pct / 100.0, b_pct / 100.0
                        dist = color_distance((sr, sg, sb), target)
                        score = calculate_score(dist)
                        mix_hex = normalized_rgb_to_hex(sr, sg, sb)
                        robot.broadcast_experiment("reasoning", {
                            "msg": "Camera unavailable — using simulated score"
                        })
                else:
                    sr, sg, sb = r_pct / 100.0, g_pct / 100.0, b_pct / 100.0
                    dist = color_distance((sr, sg, sb), target)
                    score = calculate_score(dist)
                    mix_hex = normalized_rgb_to_hex(sr, sg, sb)

                step_result = {
                    "trial": trial,
                    "r": round(r_pct, 1), "g": round(g_pct, 1), "b": round(b_pct, 1),
                    "score": round(score, 1),
                    "mix_hex": mix_hex,
                }
                history.append({
                    "rgb": (r_pct / 100, g_pct / 100, b_pct / 100),
                    "score": score,
                })

            robot.broadcast_experiment("trial_result", step_result)
            robot.broadcast_experiment("reasoning", {
                "msg": f"Score: {score:.1f}/100 (distance {dist:.3f})"
            })

            if score >= 90:
                robot.broadcast_experiment("experiment_end", {
                    "success": True,
                    "trials": trial,
                    "score": round(score, 1),
                    "mix_hex": mix_hex,
                    "msg": f"Success in {trial} trial(s)!",
                })
                return

            if trial < max_trials:
                algo_history = [{"rgb": h["rgb"], "score": h["score"]} for h in history]
                guess = get_algorithm_next_guess(
                    algorithm_name, algo_state, algo_history, target,
                    lambda msg: robot.broadcast_experiment("reasoning", {"msg": msg})
                )

        robot.broadcast_experiment("experiment_end", {
            "success": False,
            "trials": max_trials,
            "score": round(history[-1]["score"], 1) if history else 0,
            "mix_hex": history[-1].get("mix_hex", "#888888") if history else "#888888",
            "msg": f"Reached max trials ({max_trials}) without success.",
        })

    except Exception as exc:
        robot.broadcast_experiment("experiment_end", {
            "success": False,
            "trials": 0,
            "score": 0,
            "mix_hex": "#888888",
            "msg": f"Experiment error: {exc}",
        })
    finally:
        robot.experiment_running = False


# ─────────────────────────────────────────── routes ──

@app.route("/")
def index():
    return render_template("index.html", algorithms=list(ALGORITHMS.keys()))


# ── serial ──────────────────────────────────────────────

@app.route("/api/connect", methods=["POST"])
def api_connect():
    result = robot.connect()
    return jsonify(result)


@app.route("/api/disconnect", methods=["POST"])
def api_disconnect():
    robot.disconnect()
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify({
        "connected": robot.connected,
        "port": robot.port_name,
        "experiment_running": robot.experiment_running,
        "led_connected": robot.led_connected,
    })


# ── manual controls ──────────────────────────────────────

@app.route("/api/jog", methods=["POST"])
def api_jog():
    data = request.json or {}
    axis = data.get("axis", "X")
    direction = data.get("direction", 1)
    step = float(data.get("step", robot.step_mm))
    feed = float(data.get("feed", robot.feed_mm_min))
    amount = step * direction
    robot.send("M121")  # disable endstops so retract moves aren't blocked
    robot.send(f"G1 {axis}{amount:.3f} F{feed:.0f}")
    robot.send("M18")
    robot.send("M120")  # re-enable endstops
    return jsonify({"ok": True})


@app.route("/api/home", methods=["POST"])
def api_home():
    threading.Thread(target=_home_all, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    robot.send("M18")
    return jsonify({"ok": True})


@app.route("/api/empty", methods=["POST"])
def api_empty():
    threading.Thread(target=_fan_pulse_and_mix, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/send", methods=["POST"])
def api_send():
    data = request.json or {}
    cmd = data.get("cmd", "").strip()
    if not cmd:
        return jsonify({"ok": False, "error": "No command"})
    ok = robot.send(cmd)
    return jsonify({"ok": ok})


# ── camera ───────────────────────────────────────────────

@app.route("/api/camera/open", methods=["POST"])
def api_camera_open():
    ok = robot.open_camera()
    return jsonify({"ok": ok})


@app.route("/api/camera/frame")
def api_camera_frame():
    jpeg = robot.get_frame_jpeg()
    if jpeg is None:
        return Response(status=503)
    return Response(jpeg, mimetype="image/jpeg")


@app.route("/api/camera/roi", methods=["POST"])
def api_camera_roi():
    data = request.json or {}
    roi = data.get("roi", DEFAULT_ROI)
    if len(roi) == 4 and all(0 <= v <= 1 for v in roi):
        robot.roi = roi
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Invalid ROI"})


@app.route("/api/camera/sample")
def api_camera_sample():
    color = robot.sample_roi_color()
    if color is None:
        return jsonify({"ok": False})
    r, g, b = color
    return jsonify({
        "ok": True,
        "r": round(r, 4),
        "g": round(g, 4),
        "b": round(b, 4),
        "hex": normalized_rgb_to_hex(r, g, b),
    })


# ── calibration ─────────────────────────────────────────

@app.route("/api/calibrate")
def api_calibrate_get():
    with _cal_lock:
        return jsonify(dict(calibration))


@app.route("/api/calibrate/set", methods=["POST"])
def api_calibrate_set():
    data = request.json or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "No name"})
    entry = {
        "r": round(float(data["r"]), 4),
        "g": round(float(data["g"]), 4),
        "b": round(float(data["b"]), 4),
        "hex": data.get("hex", ""),
    }
    with _cal_lock:
        calibration[name] = entry
        _save_calibration(calibration)
    return jsonify({"ok": True})


@app.route("/api/calibrate/clear", methods=["POST"])
def api_calibrate_clear():
    data = request.json or {}
    name = data.get("name", "").strip()
    with _cal_lock:
        calibration.pop(name, None)
        _save_calibration(calibration)
    return jsonify({"ok": True})


@app.route("/api/calibrate/reset", methods=["POST"])
def api_calibrate_reset():
    with _cal_lock:
        calibration.clear()
        _save_calibration(calibration)
    return jsonify({"ok": True})


# ── color NLP (offline) ───────────────────────────────────

import colorsys
import difflib
import webcolors

# Common color names not in CSS3, stored as normalised (r, g, b)
_EXTENDED_COLORS: dict[str, tuple[float, float, float]] = {
    "army green":     (0.29, 0.33, 0.13),
    "baby blue":      (0.54, 0.81, 0.94),
    "baby pink":      (0.96, 0.76, 0.76),
    "burnt orange":   (0.80, 0.33, 0.00),
    "burnt sienna":   (0.91, 0.45, 0.32),
    "burnt umber":    (0.54, 0.20, 0.14),
    "burgundy":       (0.50, 0.00, 0.13),
    "chartreuse":     (0.50, 1.00, 0.00),
    "cobalt blue":    (0.00, 0.28, 0.67),
    "coral pink":     (0.97, 0.51, 0.47),
    "cornflower":     (0.39, 0.58, 0.93),
    "cream":          (1.00, 0.99, 0.82),
    "dusty rose":     (0.72, 0.52, 0.52),
    "electric blue":  (0.49, 0.98, 1.00),
    "emerald":        (0.31, 0.78, 0.47),
    "forest green":   (0.13, 0.55, 0.13),
    "goldenrod":      (0.85, 0.65, 0.13),
    "grass green":    (0.42, 0.56, 0.14),
    "hot pink":       (1.00, 0.41, 0.71),
    "ice blue":       (0.78, 0.93, 1.00),
    "lavender":       (0.71, 0.49, 0.86),
    "lemon yellow":   (1.00, 0.97, 0.00),
    "light green":    (0.56, 0.93, 0.56),
    "lilac":          (0.78, 0.64, 0.78),
    "lime green":     (0.20, 0.80, 0.20),
    "maroon":         (0.50, 0.00, 0.00),
    "midnight blue":  (0.10, 0.10, 0.44),
    "mint":           (0.74, 0.99, 0.79),
    "mint green":     (0.60, 0.98, 0.60),
    "mustard":        (1.00, 0.86, 0.35),
    "mustard yellow": (1.00, 0.86, 0.35),
    "navy":           (0.00, 0.00, 0.50),
    "navy blue":      (0.00, 0.00, 0.50),
    "neon green":     (0.22, 1.00, 0.08),
    "neon pink":      (1.00, 0.08, 0.58),
    "ocean blue":     (0.00, 0.47, 0.75),
    "off white":      (0.96, 0.96, 0.94),
    "olive green":    (0.42, 0.56, 0.14),
    "peach":          (1.00, 0.80, 0.64),
    "periwinkle":     (0.80, 0.80, 1.00),
    "powder blue":    (0.69, 0.88, 0.90),
    "rose":           (1.00, 0.00, 0.50),
    "rose gold":      (0.72, 0.43, 0.47),
    "royal blue":     (0.25, 0.41, 0.88),
    "rust":           (0.72, 0.25, 0.05),
    "sage":           (0.56, 0.62, 0.50),
    "sage green":     (0.56, 0.62, 0.50),
    "salmon":         (0.98, 0.50, 0.45),
    "seafoam":        (0.56, 0.99, 0.80),
    "seafoam green":  (0.56, 0.99, 0.80),
    "sky blue":       (0.53, 0.81, 0.98),
    "slate blue":     (0.42, 0.35, 0.80),
    "steel blue":     (0.27, 0.51, 0.71),
    "tan":            (0.82, 0.71, 0.55),
    "teal":           (0.00, 0.50, 0.50),
    "turquoise":      (0.25, 0.88, 0.82),
    "violet":         (0.93, 0.51, 0.93),
    "warm white":     (1.00, 0.97, 0.90),
    "wine":           (0.45, 0.18, 0.22),
}

# Modifier words and their HSL deltas: (lightness, saturation, hue_degrees)
_MODIFIERS: dict[str, tuple[float, float, float]] = {
    "light":   ( 0.20, -0.10,   0),
    "lighter": ( 0.30, -0.15,   0),
    "pale":    ( 0.25, -0.30,   0),
    "pastel":  ( 0.20, -0.35,   0),
    "soft":    ( 0.10, -0.20,   0),
    "muted":   ( 0.05, -0.25,   0),
    "faded":   ( 0.15, -0.25,   0),
    "dark":    (-0.20,  0.10,   0),
    "darker":  (-0.30,  0.15,   0),
    "deep":    (-0.15,  0.20,   0),
    "rich":    (-0.05,  0.25,   0),
    "bright":  ( 0.00,  0.30,   0),
    "vivid":   ( 0.00,  0.40,   0),
    "neon":    ( 0.10,  0.50,   0),
    "hot":     (-0.05,  0.30,   0),
    "warm":    ( 0.00,  0.05, -15),
    "cool":    ( 0.00,  0.05,  15),
    "electric": (0.05,  0.45,   0),
}

def _css3_to_rgb(name: str) -> Optional[tuple[float, float, float]]:
    """Try CSS3 name lookup; return normalised RGB or None."""
    for candidate in (name, name.replace(" ", ""), name.replace(" ", "-")):
        try:
            h = webcolors.name_to_hex(candidate)
            c = webcolors.hex_to_rgb(h)
            return (c.red / 255, c.green / 255, c.blue / 255)
        except (ValueError, AttributeError):
            pass
    return None

def _apply_modifiers(
    rgb: tuple[float, float, float],
    mods: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    r, g, b = rgb
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    for (ld, sd, hd) in mods:
        l = max(0.0, min(1.0, l + ld))
        s = max(0.0, min(1.0, s + sd))
        h = (h + hd / 360.0) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)

def resolve_color_offline(name: str) -> Optional[tuple[float, float, float]]:
    """Return normalised (r, g, b) for a plain-English color name, or None."""
    q = name.lower().strip()

    # 1. Exact match in extended dict
    if q in _EXTENDED_COLORS:
        return _EXTENDED_COLORS[q]

    # 2. Exact match in CSS3
    result = _css3_to_rgb(q)
    if result:
        return result

    # 3. Split into modifier words + base words; look up base then adjust
    words = q.split()
    mod_deltas = []
    base_words = []
    for w in words:
        if w in _MODIFIERS:
            mod_deltas.append(_MODIFIERS[w])
        else:
            base_words.append(w)

    if base_words:
        base_q = " ".join(base_words)
        base_rgb = (
            _EXTENDED_COLORS.get(base_q)
            or _EXTENDED_COLORS.get("".join(base_words))
            or _css3_to_rgb(base_q)
        )
        if base_rgb and mod_deltas:
            return _apply_modifiers(base_rgb, mod_deltas)
        if base_rgb:
            return base_rgb

    # 4. Fuzzy match (no-space version) against combined name list
    all_names = list(_EXTENDED_COLORS.keys()) + list(webcolors.CSS3_NAMES_TO_HEX.keys())
    compact = q.replace(" ", "")
    candidates = difflib.get_close_matches(compact, [n.replace(" ", "") for n in all_names],
                                           n=1, cutoff=0.72)
    if candidates:
        # Map back to original name
        match_compact = candidates[0]
        for n in all_names:
            if n.replace(" ", "") == match_compact:
                return _EXTENDED_COLORS.get(n) or _css3_to_rgb(n)

    return None


@app.route("/api/color/resolve", methods=["POST"])
def api_color_resolve():
    data = request.json or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "No name provided"})
    rgb = resolve_color_offline(name)
    if rgb is None:
        return jsonify({"ok": False, "error": f'Could not recognise "{name}"'})
    r, g, b = rgb
    return jsonify({
        "ok": True,
        "r": round(r, 3),
        "g": round(g, 3),
        "b": round(b, 3),
        "hex": normalized_rgb_to_hex(r, g, b),
    })


# ── pump ─────────────────────────────────────────────────

@app.route("/api/pump", methods=["POST"])
def api_pump():
    data = request.json or {}
    r = float(data.get("r", 0))
    g = float(data.get("g", 0))
    b = float(data.get("b", 0))
    feed = float(data.get("feed", robot.feed_mm_min))
    threading.Thread(target=_pump_rgb, args=(r, g, b, feed), daemon=True).start()
    return jsonify({"ok": True})


# ── LED simulator ────────────────────────────────────────

@app.route("/api/led/connect", methods=["POST"])
def api_led_connect():
    return jsonify(robot.open_led())


@app.route("/api/led/disconnect", methods=["POST"])
def api_led_disconnect():
    robot.close_led()
    return jsonify({"ok": True})


@app.route("/api/led/set", methods=["POST"])
def api_led_set():
    data = request.json or {}
    r = max(0, min(25, int(data.get("r", 0))))
    g = max(0, min(25, int(data.get("g", 0))))
    b = max(0, min(25, int(data.get("b", 0))))
    ok = robot.set_led_rgb(r, g, b)
    return jsonify({"ok": ok})


@app.route("/api/led/send", methods=["POST"])
def api_led_send():
    data = request.json or {}
    cmd = data.get("cmd", "").strip()
    if not cmd:
        return jsonify({"ok": False, "error": "No command"})
    with robot.led_lock:
        if not robot.led_ser or not robot.led_ser.is_open:
            return jsonify({"ok": False, "error": "LED not connected"})
        try:
            robot.led_ser.write((cmd + "\r").encode())
            return jsonify({"ok": True})
        except serial.SerialException as exc:
            return jsonify({"ok": False, "error": str(exc)})


# ── experiment ───────────────────────────────────────────

@app.route("/api/experiment/start", methods=["POST"])
def api_experiment_start():
    if robot.experiment_running:
        return jsonify({"ok": False, "error": "Experiment already running"})

    data = request.json or {}
    target_name = data.get("target_name", "Custom")
    target_rgb = data.get("target_rgb", {"r": 0.5, "g": 0.0, "b": 0.5})
    algorithm = data.get("algorithm", "Gradient Descent")
    max_trials = int(data.get("max_trials", 10))
    feed = float(data.get("feed", robot.feed_mm_min))
    use_camera = bool(data.get("use_camera", True))
    led_mode = bool(data.get("led_mode", False))
    also_move_steppers = bool(data.get("also_move_steppers", False))

    if led_mode and not robot.led_connected:
        return jsonify({"ok": False, "error": "LED simulator not connected"})

    robot.experiment_thread = threading.Thread(
        target=_run_experiment,
        args=(target_name, target_rgb, algorithm, max_trials, feed, use_camera, led_mode, also_move_steppers),
        daemon=True,
    )
    robot.experiment_thread.start()
    return jsonify({"ok": True})


@app.route("/api/experiment/stop", methods=["POST"])
def api_experiment_stop():
    robot.experiment_running = False
    robot.send("M18")
    return jsonify({"ok": True})


# ── logs ─────────────────────────────────────────────────

@app.route("/api/logs")
def api_logs():
    return jsonify(list(robot.log_queue))


# ── SSE stream ───────────────────────────────────────────

@app.route("/api/stream")
def api_stream():
    q = robot.subscribe()

    @stream_with_context
    def generate():
        yield f"data: {json.dumps({'event': 'connected'})}\n\n"
        try:
            while True:
                try:
                    item = q.get(timeout=15)
                    yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            robot.unsubscribe(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─────────────────────────────────────────── main ──

if __name__ == "__main__":
    print("Physical Robot Lab server starting on http://0.0.0.0:5000")
    print("Open this URL in a browser on the same network as this Pi.")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
