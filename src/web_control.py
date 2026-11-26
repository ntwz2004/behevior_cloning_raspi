"""
Raspberry Pi Car - Web Control Center (No VNC / No DISPLAY needed)

รันด้วย:
    ./venv/bin/python src/web_control.py

แล้วเปิดใน Browser บน Windows หรือมือถือ:
    http://<IP-ของ-Pi>:5000   (หรือ http://localhost:5000 ถ้า forward port ผ่าน SSH)

รองรับ 5 โหมดในโปรเซสเดียว (เปิดกล้องและ GPIO ครั้งเดียว):
  1. Manual Drive        - ขับด้วยคีย์บอร์ด (ลูกศร/WASD) หรือปุ่มบนจอมือถือ (ไม่เซฟภาพ)
  2. PPO Auto Drive      - ขับด้วยโมเดล PPO (.zip) รองรับทั้ง Discrete(3) และ Continuous
  3. DAgger / BC Drive   - ขับด้วยโมเดล PyTorch NvidiaCNN (.pth) เปิด/ปิดเซฟ DAgger + Excel ได้
  4. Collect Dataset     - ขับเก็บภาพสำหรับเทรน BC ลง datasets/dataset_bc/<scenario>
  5. Motor Calibration   - ทดสอบวิ่งตรง/เลี้ยวซ้าย/เลี้ยวขวาตามเวลาที่กำหนด
"""

from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import os
import signal
import socket
import sys
import threading
import time

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import car_config as car
import dagger_log

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
OBS_WIDTH = 120
OBS_HEIGHT = 100
POLL_PERIOD = 0.02            # 50 Hz สำหรับ Manual / Collect
DAGGER_SAVE_INTERVAL = 0.1    # วินาทีขั้นต่ำระหว่างการเซฟภาพ DAgger
INPUT_TIMEOUT = 0.35          # Deadman switch: ถ้าไม่มี heartbeat จากเบราว์เซอร์เกินนี้ -> ปล่อยปุ่มอัตโนมัติ
PREVIEW_INTERVAL = 0.10       # ~10 FPS สำหรับ MJPEG stream เพื่อไม่ให้แย่งแบนด์วิดท์ปุ่มกด

DISCRETE_ACTIONS = {0: "forward", 1: "left", 2: "right"}
DEFAULT_SCENARIOS = [
    "forward_normal",
    "forward_obstacle",
    "reverse_normal",
    "reverse_obstacle",
]

DAGGER_DIR = PROJECT_ROOT / "datasets" / "dagger_dataset"
DAGGER_LOG_DIR = PROJECT_ROOT / "logs" / "dagger"
BC_DATASET_ROOT = PROJECT_ROOT / "datasets" / "dataset_bc"
STATIC_DIR = SRC_DIR / "web_static"
STATIC_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
class FrameStack:
    """เรียงเฟรมเก่าสุด -> ใหม่สุด ตามแกน channel เหมือน VecFrameStack ของ SB3"""

    def __init__(self, n_stack):
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)

    def clear(self):
        self.frames.clear()

    def push(self, observation):
        if not self.frames:
            zero = np.zeros_like(observation)
            for _ in range(self.n_stack - 1):
                self.frames.append(zero)
        self.frames.append(observation)

    def get(self):
        return np.concatenate(list(self.frames), axis=-1)


def scan_assets():
    ppo_dir = PROJECT_ROOT / "models" / "ppo"
    ppo_models = sorted(ppo_dir.glob("*.zip")) if ppo_dir.is_dir() else []

    pth_models = []
    for sub in ("dagger", "bc"):
        d = PROJECT_ROOT / "models" / sub
        if d.is_dir():
            pth_models.extend(sorted(d.glob("*.pth")))

    if BC_DATASET_ROOT.is_dir():
        scenarios = sorted(d.name for d in BC_DATASET_ROOT.iterdir() if d.is_dir())
    else:
        scenarios = []
    if not scenarios:
        scenarios = list(DEFAULT_SCENARIOS)

    return ppo_models, pth_models, scenarios


def short_model_name(path):
    try:
        return str(path.relative_to(PROJECT_ROOT / "models"))
    except Exception:
        return path.name


def get_local_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


# ----------------------------------------------------------------------
# Model Cache
# ----------------------------------------------------------------------
class ModelCache:
    def __init__(self):
        self._ppo_cache = {}
        self._pth_cache = {}

    def load_ppo(self, model_path):
        key = str(model_path)
        if key in self._ppo_cache:
            return self._ppo_cache[key]

        from gymnasium import spaces
        from stable_baselines3 import PPO

        try:
            model = PPO.load(key, device="cpu")
        except Exception:
            model = PPO.load(
                key,
                device="cpu",
                custom_objects={
                    "learning_rate": 0.0,
                    "lr_schedule": lambda _: 0.0,
                    "clip_range": lambda _: 0.0,
                    "_last_obs": None,
                    "_last_episode_starts": None,
                },
            )

        expected = tuple(model.observation_space.shape)
        if len(expected) != 3:
            raise RuntimeError(f"Unsupported observation shape: {expected}")
        if expected[0] == OBS_HEIGHT and expected[1] == OBS_WIDTH:
            channels = expected[2]
        elif expected[1] == OBS_HEIGHT and expected[2] == OBS_WIDTH:
            channels = expected[0]
        else:
            raise RuntimeError(
                f"Model expects {expected}, incompatible with {OBS_WIDTH}x{OBS_HEIGHT}"
            )
        if channels % 3 != 0:
            raise RuntimeError(f"Unexpected channel count: {channels}")
        n_stack = channels // 3

        is_discrete = isinstance(model.action_space, spaces.Discrete)
        if is_discrete and model.action_space.n != len(DISCRETE_ACTIONS):
            raise RuntimeError(f"Unsupported discrete action space: {model.action_space}")

        entry = (model, n_stack, is_discrete)
        self._ppo_cache[key] = entry
        return entry

    def load_pth(self, model_path):
        key = str(model_path)
        if key in self._pth_cache:
            return self._pth_cache[key]

        import torch
        import torch.nn as nn
        from model import NvidiaCNN

        device = torch.device("cpu")
        model = NvidiaCNN().to(device)
        model.load_state_dict(torch.load(key, map_location=device))
        model.eval()
        softmax = nn.Softmax(dim=1)

        entry = (model, softmax, device, torch)
        self._pth_cache[key] = entry
        return entry


# ----------------------------------------------------------------------
# Car Controller (owns GPIO & Picamera2 in a single dedicated thread)
# ----------------------------------------------------------------------
class CarController:
    def __init__(self):
        self.lock = threading.Lock()
        self.motor_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.frame_cond = threading.Condition()
        self._latest_obs_bgr = None
        self.latest_jpeg = None
        self.frame_seq = 0
        self.stream_clients = 0

        self.default_settings = {
            "base_speed": float(car.BASE_SPEED),
            "max_diff": float(car.MAX_DIFF),
            "left_trim": float(car.LEFT_TRIM),
            "right_trim": float(car.RIGHT_TRIM),
        }

        self.preview_enabled = True
        self.cache = ModelCache()
        self.picam2 = None

        self._shutdown = threading.Event()
        self._stop_mode = threading.Event()
        self._pending_job = None
        self._job_event = threading.Event()
        self._input_event = threading.Event()

        # Live input state from web UI (protected by self.lock)
        self._held_input = None      # "forward" | "left" | "right" | "backward" | "brake" | None
        self._input_ts = 0.0
        self._last_input_seq = -1
        self._latched_input = None   # สำหรับแตะสั้นๆ ในโหมด DAgger

        self._state = {
            "mode": "idle",
            "banner": "idle",
            "title": "IDLE (STOPPED)",
            "detail": "Ready. Choose a mode to start.",
            "action": "STOPPED",
            "left_pwm": 0.0,
            "right_pwm": 0.0,
            "infer_ms": 0.0,
            "overruns": 0,
            "extra": {},
            "message": "Ready.",
            "error": False,
        }

        self._cam_worker = threading.Thread(target=self._camera_loop, daemon=True)
        self._worker = threading.Thread(target=self._run_loop, daemon=True)

    def start(self):
        car.setup_motors()
        self.picam2 = car.make_camera()
        self._cam_worker.start()
        self._worker.start()

    def shutdown(self):
        self._shutdown.set()
        self._stop_mode.set()
        self._input_event.set()
        self._job_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=3.0)
        if self._cam_worker.is_alive():
            self._cam_worker.join(timeout=2.0)
        with self.motor_lock:
            try:
                car.stop_car()
            except Exception:
                pass
            try:
                car.cleanup_motors()
            except Exception:
                pass
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
        print("[WebControl] Closed all resources safely.")

    def _motor_stop(self):
        with self.motor_lock:
            car.stop_car()

    def _motor_steer(self, steering):
        with self.motor_lock:
            return car.drive_from_steering(steering)

    def _motor_reverse(self):
        with self.motor_lock:
            return car.drive_reverse()

    # ---- Thread-safe API called by HTTP handlers ----
    def get_assets_payload(self):
        ppo_models, pth_models, scenarios = scan_assets()
        return {
            "ppo_models": [short_model_name(p) for p in ppo_models],
            "pth_models": [short_model_name(p) for p in pth_models],
            "scenarios": scenarios,
            "settings": self.get_settings(),
            "preview_enabled": self.preview_enabled,
        }

    def get_settings(self):
        return {
            "base_speed": float(car.BASE_SPEED),
            "max_diff": float(car.MAX_DIFF),
            "left_trim": float(car.LEFT_TRIM),
            "right_trim": float(car.RIGHT_TRIM),
        }

    def update_settings(self, data):
        with self.lock:
            if data.get("reset"):
                car.BASE_SPEED = self.default_settings["base_speed"]
                car.MAX_DIFF = self.default_settings["max_diff"]
                car.LEFT_TRIM = self.default_settings["left_trim"]
                car.RIGHT_TRIM = self.default_settings["right_trim"]
            else:
                if "base_speed" in data:
                    car.BASE_SPEED = float(np.clip(float(data["base_speed"]), 10.0, 100.0))
                if "max_diff" in data:
                    car.MAX_DIFF = float(np.clip(float(data["max_diff"]), 0.0, 100.0))
                if "left_trim" in data:
                    car.LEFT_TRIM = float(np.clip(float(data["left_trim"]), 0.5, 1.5))
                if "right_trim" in data:
                    car.RIGHT_TRIM = float(np.clip(float(data["right_trim"]), 0.5, 1.5))
            if "preview_enabled" in data:
                self.preview_enabled = bool(data["preview_enabled"])
            return {
                "settings": self.get_settings(),
                "preview_enabled": self.preview_enabled,
            }

    def set_input(self, key, seq=None):
        if key not in (None, "forward", "left", "right", "backward", "brake"):
            return
        now = time.monotonic()
        immediate_brake = False
        with self.lock:
            if seq is not None:
                try:
                    seq_int = int(seq)
                except (TypeError, ValueError):
                    seq_int = None
                if seq_int is not None:
                    # อนุญาตให้รีเซ็ตเลข seq ได้ถ้าหน้าเว็บรีเฟรช (seq น้อยกว่ามาก)
                    if seq_int < self._last_input_seq and (self._last_input_seq - seq_int) < 10000:
                        return
                    self._last_input_seq = seq_int
            self._held_input = key
            self._input_ts = now
            if key in ("forward", "left", "right"):
                self._latched_input = key
            if self._state["mode"] in ("ppo", "dagger") and key in ("brake", "backward"):
                immediate_brake = True
        if immediate_brake:
            self._motor_stop()
        self._input_event.set()

    def _read_input(self, consume_latch=False):
        now = time.monotonic()
        with self.lock:
            if self._held_input is not None and (now - self._input_ts) > INPUT_TIMEOUT:
                self._held_input = None
            held = self._held_input
            latched = self._latched_input
            if consume_latch:
                self._latched_input = None
            return held, latched

    def start_mode(self, job):
        with self.lock:
            self._held_input = None
            self._latched_input = None
            self._pending_job = job
        self._stop_mode.set()
        self._input_event.set()
        self._job_event.set()

    def stop_mode(self, message="Stopped by user."):
        with self.lock:
            self._held_input = None
            self._latched_input = None
            self._pending_job = None
            self._state["message"] = message
            self._state["error"] = False
        self._stop_mode.set()
        self._input_event.set()
        self._job_event.set()
        self._motor_stop()

    def get_status(self):
        with self.lock:
            out = dict(self._state)
            out["extra"] = dict(self._state["extra"])
            out["settings"] = self.get_settings()
            out["preview_enabled"] = self.preview_enabled
            return out

    def _set_state(self, **kwargs):
        with self.lock:
            self._state.update(kwargs)

    def _publish_frame(self, bgr_img):
        if not self.preview_enabled:
            return
        if self.stream_clients <= 0 and self.latest_jpeg is not None:
            return
        ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if not ok:
            return
        with self.frame_cond:
            self.latest_jpeg = buf.tobytes()
            self.frame_seq += 1
            self.frame_cond.notify_all()

    def _camera_loop(self):
        """เธรดจับภาพและเข้ารหัส MJPEG แยกต่างหาก เพื่อไม่ให้บล็อกการตอบสนองปุ่มกด"""
        last_jpg = 0.0
        while not self._shutdown.is_set():
            try:
                raw = self.picam2.capture_array()
                obs_bgr = car.prepare_observation(raw, OBS_WIDTH, OBS_HEIGHT, rgb=False)
                with self.frame_lock:
                    self._latest_obs_bgr = obs_bgr
                now = time.monotonic()
                if self.preview_enabled and (now - last_jpg >= PREVIEW_INTERVAL):
                    self._publish_frame(obs_bgr)
                    last_jpg = now
            except Exception:
                time.sleep(0.05)
                continue
            time.sleep(0.02)

    def _capture_obs(self, rgb=False):
        with self.frame_lock:
            obs_bgr = None if self._latest_obs_bgr is None else self._latest_obs_bgr.copy()
        if obs_bgr is None:
            raw = self.picam2.capture_array()
            obs_bgr = car.prepare_observation(raw, OBS_WIDTH, OBS_HEIGHT, rgb=False)
        if rgb:
            return cv2.cvtColor(obs_bgr, cv2.COLOR_BGR2RGB)
        return obs_bgr

    # ---- Background worker loop ----
    def _run_loop(self):
        while not self._shutdown.is_set():
            with self.lock:
                job = self._pending_job
                self._pending_job = None

            if job is None:
                self._motor_stop()
                self._set_state(
                    mode="idle",
                    banner="idle",
                    title="IDLE (STOPPED)",
                    detail="Select a mode and click START.",
                    action="STOPPED",
                    left_pwm=0.0,
                    right_pwm=0.0,
                    infer_ms=0.0,
                )
                self._job_event.wait(timeout=0.1)
                self._job_event.clear()
                continue

            self._stop_mode.clear()
            self._input_event.clear()
            mode = job.get("mode")
            try:
                if mode == "manual":
                    self._run_manual()
                elif mode == "ppo":
                    self._run_ppo(job)
                elif mode == "dagger":
                    self._run_dagger(job)
                elif mode == "collect":
                    self._run_collect(job)
                elif mode == "calibrate":
                    self._run_calibrate(job)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            except Exception as exc:
                self._set_state(message=f"Error: {exc}", error=True)
                print(f"[WebControl] Mode '{mode}' error: {exc}")
            finally:
                self._motor_stop()
                with self.lock:
                    self._held_input = None
                    self._latched_input = None

    # ---- Mode 1: Manual Drive ----
    def _run_manual(self):
        current_label = None
        left_pwm, right_pwm = 0.0, 0.0

        self._set_state(
            mode="manual",
            banner="idle",
            title="MANUAL DRIVE (NO SAVING)",
            detail="Hold Arrow / WASD keys or touch pad to drive.",
            action="STOPPED",
            left_pwm=0.0,
            right_pwm=0.0,
            infer_ms=0.0,
            overruns=0,
            extra={},
            message="Manual Drive active.",
            error=False,
        )

        while not self._shutdown.is_set() and not self._stop_mode.is_set():
            self._input_event.clear()
            held, _ = self._read_input()
            if held != current_label:
                if held is None or held == "brake":
                    self._motor_stop()
                    left_pwm, right_pwm = 0.0, 0.0
                elif held == "backward":
                    left_pwm, right_pwm = self._motor_reverse()
                else:
                    left_pwm, right_pwm = self._motor_steer(car.DISCRETE_STEERING[held])
                current_label = held

            if held == "brake":
                banner, action = "brake", "BRAKE"
            elif held == "backward":
                banner, action = "countdown", "REVERSE"
            elif held is not None:
                banner, action = "drive", held.upper()
            else:
                banner, action = "idle", "STOPPED"

            self._set_state(
                banner=banner,
                action=action,
                left_pwm=left_pwm,
                right_pwm=right_pwm,
            )
            # ตื่นทันทีที่มีการกด/ปล่อยปุ่ม (ไม่รอ sleep ครบ) และตรวจ deadman timeout ทุก 40ms
            self._input_event.wait(timeout=0.04)

        self._motor_stop()
        self._set_state(message="Exited Manual Drive.", error=False)

    # ---- Mode 2: PPO Auto Drive ----
    def _run_ppo(self, job):
        rel = job.get("model")
        if not rel:
            raise FileNotFoundError("No PPO model selected.")
        model_path = (PROJECT_ROOT / "models" / rel).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {rel}")

        self._set_state(
            mode="ppo",
            banner="countdown",
            title="LOADING PPO MODEL...",
            detail=rel,
            action="LOADING",
            left_pwm=0.0,
            right_pwm=0.0,
            message=f"Loading {rel}...",
            error=False,
        )
        model, n_stack, is_discrete = self.cache.load_ppo(model_path)
        frame_stack = FrameStack(n_stack)
        timer = car.LoopTimer(car.CONTROL_PERIOD)
        space_label = "Discrete(3)" if is_discrete else "Continuous Box"

        self._set_state(
            title="PPO AUTO DRIVE",
            detail=f"{rel} ({space_label}, stack={n_stack})",
            message=f"Running PPO: {rel}",
        )

        while not self._shutdown.is_set() and not self._stop_mode.is_set():
            held, _ = self._read_input()
            if held in ("brake", "backward"):
                self._motor_stop()
                frame_stack.clear()
                self._set_state(
                    banner="brake",
                    action="BRAKE",
                    left_pwm=0.0,
                    right_pwm=0.0,
                )
                self._input_event.clear()
                self._input_event.wait(timeout=0.05)
                timer.reset()
                continue

            obs_rgb = self._capture_obs(rgb=True)
            frame_stack.push(obs_rgb)
            stacked_obs = frame_stack.get()

            t0 = time.monotonic()
            action, _ = model.predict(stacked_obs, deterministic=True)
            infer_ms = (time.monotonic() - t0) * 1000.0

            if is_discrete:
                idx = int(np.asarray(action).item())
                label = DISCRETE_ACTIONS[idx]
                steering = car.DISCRETE_STEERING[label]
                action_text = label.upper()
            else:
                steering = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
                action_text = f"STEER {steering:+.2f}"

            left_pwm, right_pwm = self._motor_steer(steering)
            self._set_state(
                banner="drive",
                action=action_text,
                left_pwm=left_pwm,
                right_pwm=right_pwm,
                infer_ms=round(infer_ms, 1),
                overruns=timer.overruns,
                extra={"steering": round(steering, 2), "stack": n_stack, "space": space_label},
            )
            timer.wait()

        self._motor_stop()
        self._set_state(message=f"Stopped PPO ({model_path.name}) | overruns={timer.overruns}", error=False)

    # ---- Mode 3: DAgger / BC Drive ----
    def _run_dagger(self, job):
        rel = job.get("model")
        if not rel:
            raise FileNotFoundError("No PyTorch (.pth) model selected.")
        model_path = (PROJECT_ROOT / "models" / rel).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {rel}")

        save_enabled = bool(job.get("save_dagger", True))
        session_name = str(job.get("session_name", "")).strip()

        self._set_state(
            mode="dagger",
            banner="countdown",
            title="LOADING PYTORCH MODEL...",
            detail=rel,
            action="LOADING",
            left_pwm=0.0,
            right_pwm=0.0,
            message=f"Loading {rel}...",
            error=False,
        )
        model, softmax, device, torch = self.cache.load_pth(model_path)

        if save_enabled:
            os.makedirs(DAGGER_DIR, exist_ok=True)
        dataset_at_start = (
            sum(dagger_log.count_dataset_images(str(DAGGER_DIR)).values())
            if DAGGER_DIR.is_dir()
            else 0
        )

        timer = car.LoopTimer(car.CONTROL_PERIOD)
        tracker = dagger_log.OverrideTracker()
        saved_count = 0
        agree_count = 0
        last_saved_label = None
        last_save_time = 0.0
        session_start_dt = datetime.now()
        session_start = time.monotonic()

        self._set_state(
            title="DAGGER DRIVE" if save_enabled else "BC AUTO DRIVE (NO SAVING)",
            detail=f"{rel} | Save: {'ON' if save_enabled else 'OFF'}",
            message=f"Running {'DAgger' if save_enabled else 'BC'}: {rel}",
        )

        with torch.no_grad():
            while not self._shutdown.is_set() and not self._stop_mode.is_set():
                held, latched = self._read_input(consume_latch=True)
                if held in ("brake", "backward"):
                    self._motor_stop()
                    tracker.update(None)
                    last_saved_label = None
                    self._set_state(
                        banner="brake",
                        action="BRAKE",
                        left_pwm=0.0,
                        right_pwm=0.0,
                    )
                    self._input_event.clear()
                    self._input_event.wait(timeout=0.05)
                    timer.reset()
                    continue

                obs_bgr = self._capture_obs(rgb=False)
                input_tensor = (
                    torch.from_numpy(obs_bgr.transpose((2, 0, 1)).copy())
                    .float()
                    .div(255.0)
                    .unsqueeze(0)
                    .to(device)
                )

                t0 = time.monotonic()
                probs = softmax(model(input_tensor))[0]
                ai_choice = torch.argmax(probs).item()
                confidence = probs[ai_choice].item()
                ai_label = DISCRETE_ACTIONS[ai_choice]
                infer_ms = (time.monotonic() - t0) * 1000.0

                human_label = held if held in ("forward", "left", "right") else latched
                now = time.monotonic()
                tracker.update(human_label, now)

                if human_label is not None:
                    left_pwm, right_pwm = self._motor_steer(
                        car.DISCRETE_STEERING[human_label]
                    )
                    banner = "save" if save_enabled else "drive"
                    action_text = f"OVR: {human_label.upper()}"

                    if save_enabled:
                        should_save = (
                            human_label != last_saved_label
                            or now - last_save_time >= DAGGER_SAVE_INTERVAL
                        )
                        if should_save:
                            ts = int(time.time() * 1000)
                            cv2.imwrite(str(DAGGER_DIR / f"{human_label}_{ts}.jpg"), obs_bgr)
                            agreed = human_label == ai_label
                            saved_count += 1
                            if agreed:
                                agree_count += 1
                            tracker.add_image(agreed)
                            last_saved_label = human_label
                            last_save_time = now
                else:
                    last_saved_label = None
                    left_pwm, right_pwm = self._motor_steer(
                        car.DISCRETE_STEERING[ai_label]
                    )
                    banner = "drive"
                    action_text = f"AI: {ai_label.upper()}"

                self._set_state(
                    banner=banner,
                    action=action_text,
                    left_pwm=left_pwm,
                    right_pwm=right_pwm,
                    infer_ms=round(infer_ms, 1),
                    overruns=timer.overruns,
                    extra={
                        "ai_pred": f"{ai_label} ({confidence * 100:.1f}%)",
                        "save_enabled": save_enabled,
                        "saved_session": saved_count,
                        "dataset_total": dataset_at_start + saved_count,
                        "override_s": round(tracker.total_seconds(now), 1),
                    },
                )
                timer.wait()

        self._motor_stop()
        tracker.close()
        total_seconds = time.monotonic() - session_start

        if save_enabled:
            try:
                log_path = dagger_log.save_excel(
                    str(DAGGER_LOG_DIR),
                    session_name,
                    str(DAGGER_DIR),
                    session_start_dt,
                    datetime.now(),
                    total_seconds,
                    tracker,
                    saved_count,
                    agree_count,
                    timer.overruns,
                    {
                        "model": rel,
                        "base_speed": car.BASE_SPEED,
                        "max_diff": car.MAX_DIFF,
                        "control_period": car.CONTROL_PERIOD,
                        "save_min_interval": DAGGER_SAVE_INTERVAL,
                    },
                    dataset_at_start,
                )
                note = f" | Excel: {Path(log_path).name}" if log_path else ""
            except Exception as exc:
                note = f" | Excel error: {exc}"
            self._set_state(
                message=f"DAgger done: saved {saved_count} imgs ({total_seconds:.0f}s){note}",
                error=False,
            )
        else:
            self._set_state(
                message=f"Stopped BC Drive ({model_path.name}) after {total_seconds:.0f}s.",
                error=False,
            )

    # ---- Mode 4: Collect Dataset (BC) ----
    def _run_collect(self, job):
        scenario = str(job.get("scenario", DEFAULT_SCENARIOS[0])).strip()
        if not scenario or "/" in scenario or ".." in scenario:
            raise ValueError(f"Invalid scenario name: {scenario}")

        dataset_dir = BC_DATASET_ROOT / scenario
        os.makedirs(dataset_dir, exist_ok=True)
        existing_total = sum(dagger_log.count_dataset_images(str(dataset_dir)).values())
        saved_count = {"forward": 0, "left": 0, "right": 0}

        timer = car.LoopTimer(POLL_PERIOD)
        current_label = None
        last_saved_label = None
        last_save_time = 0.0
        last_cam = 0.0
        left_pwm, right_pwm = 0.0, 0.0

        self._set_state(
            mode="collect",
            banner="idle",
            title="COLLECT DATASET (BC)",
            detail=f"datasets/dataset_bc/{scenario}",
            action="READY",
            left_pwm=0.0,
            right_pwm=0.0,
            infer_ms=0.0,
            overruns=0,
            extra={"forward": 0, "left": 0, "right": 0, "total": 0, "folder_total": existing_total},
            message=f"Collecting into dataset_bc/{scenario}",
            error=False,
        )

        while not self._shutdown.is_set() and not self._stop_mode.is_set():
            self._input_event.clear()
            held, _ = self._read_input()
            label = held if held in ("forward", "left", "right") else None

            if label != current_label:
                if label is None:
                    self._motor_stop()
                    left_pwm, right_pwm = 0.0, 0.0
                else:
                    left_pwm, right_pwm = self._motor_steer(car.DISCRETE_STEERING[label])
                current_label = label

            now = time.monotonic()
            saved_now = False
            if label is not None:
                if label != last_saved_label or now - last_save_time >= car.CONTROL_PERIOD:
                    obs_bgr = self._capture_obs(rgb=False)
                    ts = int(time.time() * 1000)
                    cv2.imwrite(str(dataset_dir / f"{label}_{ts}.jpg"), obs_bgr)
                    saved_count[label] += 1
                    last_saved_label = label
                    last_save_time = time.monotonic()
                    saved_now = True
            else:
                last_saved_label = None

            if saved_now:
                banner, action_text = "save", f"SAVE: {label.upper()}"
            elif label is not None:
                banner, action_text = "drive", label.upper()
            else:
                banner, action_text = "idle", "READY"

            session_total = sum(saved_count.values())
            self._set_state(
                banner=banner,
                action=action_text,
                left_pwm=left_pwm,
                right_pwm=right_pwm,
                extra={
                    "forward": saved_count["forward"],
                    "left": saved_count["left"],
                    "right": saved_count["right"],
                    "total": session_total,
                    "folder_total": existing_total + session_total,
                },
            )
            wait_s = 0.04
            if label is not None:
                until_save = max(0.005, car.CONTROL_PERIOD - (time.monotonic() - last_save_time))
                wait_s = min(wait_s, until_save)
            self._input_event.wait(timeout=wait_s)

        self._motor_stop()
        self._set_state(
            message=(
                f"Collected {sum(saved_count.values())} imgs into dataset_bc/{scenario} "
                f"(F:{saved_count['forward']} L:{saved_count['left']} R:{saved_count['right']})"
            ),
            error=False,
        )

    # ---- Mode 5: Motor Calibration ----
    def _run_calibrate(self, job):
        cal_mode = job.get("cal_mode", "right")
        duration = float(np.clip(float(job.get("duration", 3.0)), 0.5, 10.0))
        modes = {"straight": 0.0, "left": -1.0, "right": 1.0}
        if cal_mode not in modes:
            raise ValueError(f"Invalid calibration mode: {cal_mode}")
        steering = modes[cal_mode]

        countdown = 3.0
        phase = "countdown"
        phase_start = time.monotonic()
        left_pwm, right_pwm = 0.0, 0.0
        last_cam = 0.0

        self._set_state(
            mode="calibrate",
            banner="countdown",
            title="MOTOR CALIBRATION",
            detail=f"{cal_mode.upper()} for {duration:.1f}s",
            action=f"START IN {countdown:.1f}s",
            left_pwm=0.0,
            right_pwm=0.0,
            infer_ms=0.0,
            overruns=0,
            extra={},
            message=f"Calibration {cal_mode} ({duration:.1f}s) starting...",
            error=False,
        )

        while not self._shutdown.is_set() and not self._stop_mode.is_set():
            now = time.monotonic()
            elapsed = now - phase_start

            if phase == "countdown":
                rem = countdown - elapsed
                if rem <= 0:
                    phase = "driving"
                    phase_start = now
                    left_pwm, right_pwm = self._motor_steer(steering)
                    rem = duration
                else:
                    self._set_state(
                        banner="countdown",
                        action=f"START IN {rem:.1f}s",
                        left_pwm=0.0,
                        right_pwm=0.0,
                        extra={"phase": "Place car at start mark", "remaining_s": round(rem, 1)},
                    )
            else:
                rem = duration - elapsed
                if rem <= 0:
                    break
                self._set_state(
                    banner="drive",
                    action=f"{cal_mode.upper()} {rem:.1f}s",
                    left_pwm=left_pwm,
                    right_pwm=right_pwm,
                    extra={"phase": "Driving", "remaining_s": round(rem, 1)},
                )

            self._stop_mode.wait(timeout=0.03)

        self._motor_stop()
        self._set_state(
            message=f"Calibration '{cal_mode}' ({duration:.1f}s) finished | L={left_pwm:.1f}% R={right_pwm:.1f}%",
            error=False,
        )


# ----------------------------------------------------------------------
# HTTP Handler
# ----------------------------------------------------------------------
def make_handler(controller):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            try:
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass

        def log_message(self, format, *args):
            # ไม่สแปม log ทุก request ใน terminal
            pass

        def _send_json(self, obj, status=200):
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self):
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            body = self.rfile.read(length)
            return json.loads(body.decode("utf-8"))

        def _send_static(self, filename):
            static_root = STATIC_DIR.resolve()
            file_path = (static_root / filename).resolve()
            if file_path.parent != static_root or not file_path.is_file():
                self.send_error(404)
                return
            mime = STATIC_MIME.get(file_path.suffix.lower(), "application/octet-stream")
            raw = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send_static("index.html")
                return

            if path.startswith("/static/"):
                self._send_static(path[len("/static/"):])
                return

            if path == "/api/assets":
                self._send_json(controller.get_assets_payload())
                return

            if path == "/api/status":
                self._send_json(controller.get_status())
                return

            if path == "/stream.mjpg":
                self.close_connection = True
                self.send_response(200)
                self.send_header("Connection", "close")
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                self.end_headers()

                controller.stream_clients += 1
                last_seq = -1
                try:
                    while not controller._shutdown.is_set():
                        with controller.frame_cond:
                            if controller.frame_seq == last_seq:
                                controller.frame_cond.wait(timeout=0.5)
                            jpg = controller.latest_jpeg
                            seq = controller.frame_seq
                        if jpg is None or seq == last_seq:
                            continue
                        last_seq = seq
                        part = (
                            b"--FRAME\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                            + jpg
                            + b"\r\n"
                        )
                        self.wfile.write(part)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    controller.stream_clients = max(0, controller.stream_clients - 1)
                return

            self.send_error(404)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            try:
                data = self._read_json()
            except Exception:
                self._send_json({"ok": False, "error": "Invalid JSON"}, status=400)
                return

            if path == "/api/start":
                controller.start_mode(data)
                self._send_json({"ok": True})
                return

            if path == "/api/stop":
                controller.stop_mode()
                self._send_json({"ok": True})
                return

            if path == "/api/input":
                controller.set_input(data.get("key"), data.get("seq"))
                self._send_json({"ok": True})
                return

            if path == "/api/settings":
                res = controller.update_settings(data)
                self._send_json({"ok": True, **res})
                return

            self.send_error(404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Car Web Control Center")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="HTTP port (default: 5000)")
    args = parser.parse_args()

    controller = CarController()
    controller.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(controller))
    server.daemon_threads = True

    def _request_stop(signum=None, frame=None):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    ip = get_local_ip()
    print("\n============================================================")
    print("  Raspberry Pi Car - Web Control Center")
    print(f"  Windows / Phone (same Wi-Fi): http://{ip}:{args.port}")
    print(f"  Local / SSH Port Forward    : http://localhost:{args.port}")
    print("  Press Ctrl+C to stop server safely.")
    print("============================================================\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        controller.shutdown()


if __name__ == "__main__":
    main()
