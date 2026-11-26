import signal
import time
from collections import deque
from pathlib import Path

import os
import cv2
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

import car_config as car

MODEL_PATH = "../models/ppo/ppo_1stackframe.zip"

# output ของโมเดลแบบ Discrete(3): 0 = forward, 1 = left, 2 = right
DISCRETE_ACTIONS = {0: "forward", 1: "left", 2: "right"}

# ----------------------------------------------------------------------
# ต้องตรงกับตอนเทรน (unity_env.py / train.py / CarAgent.cs)
# ----------------------------------------------------------------------
OBS_WIDTH = 120                 # = OBS_WIDTH ใน unity_env.py
OBS_HEIGHT = 100                # = OBS_HEIGHT ใน unity_env.py
N_STACK = 1                     # = VecFrameStack(n_stack=4) ใน train.py

# 1 step ตอนเทรน = fixedStepsPerAction x Time.fixedDeltaTime ใน Unity
FIXED_STEPS_PER_ACTION = 5      # = fixedStepsPerAction ใน CarAgent
FIXED_DELTA_TIME = 0.02         # = Project Settings > Time > Fixed Timestep
SIM_STEP_PERIOD = FIXED_STEPS_PER_ACTION * FIXED_DELTA_TIME   # 0.1 วินาที

SAVE_DEBUG_IMAGE = True         # เซฟภาพที่ส่งเข้าโมเดล 1 ครั้ง เพื่อเทียบกับ ai_vision.jpg

running = True


class FrameStack:
    """
    ทำงานเหมือน VecFrameStack ของ SB3: เรียงเก่าสุด -> ใหม่สุด ตามแกน channel
    และตอนเริ่ม episode เฟรมที่ยังไม่มีจะเป็นศูนย์ (ภาพดำ) ไม่ใช่สำเนาเฟรมแรก
    """

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


def request_stop(signum=None, frame=None):
    global running
    print("\n[Signal] Stop requested...")
    running = False


def load_model():
    try:
        return PPO.load(MODEL_PATH, device="cpu")
    except Exception as exc:
        # มักเกิดเมื่อเวอร์ชัน Python บน Pi ต่างจากเครื่องที่เทรน
        print(f"[Model] Normal load failed: {exc}")
        print("[Model] Retrying with custom_objects...")
        return PPO.load(
            MODEL_PATH,
            device="cpu",
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
                "_last_obs": None,
                "_last_episode_starts": None
            },
        )


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Not found: {MODEL_PATH}")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if abs(SIM_STEP_PERIOD - car.CONTROL_PERIOD) > 1e-9:
        print(
            f"[Warning] car.CONTROL_PERIOD={car.CONTROL_PERIOD}s but the policy was "
            f"trained with {SIM_STEP_PERIOD}s per step"
        )

    print(f"[Model] Loading: {MODEL_PATH}")
    model = load_model()

    # ตรวจว่าโมเดลคาดหวังภาพขนาด/จำนวน channel ตรงกับที่จะส่งเข้าไป
    expected = tuple(model.observation_space.shape)
    channels = N_STACK * 3
    valid_shapes = {
        (OBS_HEIGHT, OBS_WIDTH, channels),   # channel-last
        (channels, OBS_HEIGHT, OBS_WIDTH),   # channel-first (หลัง VecTransposeImage)
    }
    if expected not in valid_shapes:
        raise RuntimeError(
            f"Model expects observation shape {expected}, "
            f"but this script produces {(OBS_HEIGHT, OBS_WIDTH, channels)}"
        )

    is_discrete = isinstance(model.action_space, spaces.Discrete)
    if is_discrete and model.action_space.n != len(DISCRETE_ACTIONS):
        raise RuntimeError(
            f"Unsupported discrete action space: {model.action_space} "
            f"(expected Discrete({len(DISCRETE_ACTIONS)}))"
        )

    print(f"[Model] Loaded (observation shape {expected}, action space {model.action_space})")

    car.setup_motors()
    picam2 = car.make_camera()

    print(f"[Ready] Control rate {1.0 / car.CONTROL_PERIOD:.1f} Hz | Press Ctrl+C in terminal to STOP")

    frame_stack = FrameStack(N_STACK)
    timer = car.LoopTimer(car.CONTROL_PERIOD)
    debug_image_saved = not SAVE_DEBUG_IMAGE
    last_log_time = 0.0

    try:
        while running:
            raw_frame = picam2.capture_array()
            observation = car.prepare_observation(raw_frame, OBS_WIDTH, OBS_HEIGHT, rgb=True)
            frame_stack.push(observation)
            stacked_obs = frame_stack.get()

            if not debug_image_saved:
                cv2.imwrite(
                    "pi_vision.jpg",
                    cv2.cvtColor(observation, cv2.COLOR_RGB2BGR),
                )
                debug_image_saved = True
                print("[Debug] Saved model input: pi_vision.jpg")

            t0 = time.monotonic()
            action, _ = model.predict(stacked_obs, deterministic=True)
            infer_ms = (time.monotonic() - t0) * 1000.0

            if is_discrete:
                action_idx = int(np.asarray(action).item())
                action_label = DISCRETE_ACTIONS[action_idx]
                steering = car.DISCRETE_STEERING[action_label]
            else:
                steering = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
                action_label = None

            left_pwm, right_pwm = car.drive_from_steering(steering)

            now = time.monotonic()
            if now - last_log_time >= 0.2:
                action_text = (
                    f"action={action_label:<7} (steering={steering:+.1f})"
                    if is_discrete
                    else f"steering={steering:+.2f}"
                )
                print(
                    f"\r[AI] {action_text} | "
                    f"L={left_pwm:4.1f}% | "
                    f"R={right_pwm:4.1f}% | "
                    f"infer={infer_ms:5.1f}ms | "
                    f"overruns={timer.overruns}    ",
                    end="",
                    flush=True,
                )
                last_log_time = now

            timer.wait()

    except KeyboardInterrupt:
        print("\n[Stop] Stopped by user (Ctrl+C)")
    finally:
        print("\n[Safety] Stopping motors...")
        try:
            car.stop_car()
        except Exception:
            pass

        car.cleanup_motors()

        try:
            picam2.stop()
            picam2.close()
        except Exception:
            pass

        print("[Safety] Done")


if __name__ == "__main__":
    main()