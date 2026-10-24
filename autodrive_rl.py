import signal
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pygame
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from stable_baselines3 import PPO

MODEL_PATH = Path(__file__).resolve().parent / "ppo1.zip"

# ----------------------------------------------------------------------
# ต้องตรงกับตอนเทรน (unity_env.py / train.py / CarAgent.cs)
# ----------------------------------------------------------------------
OBS_WIDTH = 120                 # = OBS_WIDTH ใน unity_env.py
OBS_HEIGHT = 100                # = OBS_HEIGHT ใน unity_env.py
N_STACK = 4                     # = VecFrameStack(n_stack=4) ใน train.py

# 1 step ตอนเทรน = fixedStepsPerAction x Time.fixedDeltaTime ใน Unity
FIXED_STEPS_PER_ACTION = 5      # = fixedStepsPerAction ใน CarAgent
FIXED_DELTA_TIME = 0.02         # = Edit > Project Settings > Time > Fixed Timestep
CONTROL_PERIOD = FIXED_STEPS_PER_ACTION * FIXED_DELTA_TIME   # 0.1 วินาที = 10 Hz

# ----------------------------------------------------------------------
# กล้อง: จับภาพใหญ่ที่สัดส่วน 4:3 แล้วครอปกลางให้ได้สัดส่วน 120:100
# จากนั้นย่อด้วย INTER_AREA (ไม่ขอขนาดเล็กจากเซ็นเซอร์ตรงๆ เพื่อคุม FOV ได้)
# ----------------------------------------------------------------------
CAPTURE_SIZE = (640, 480)

# ภาพของ Picamera2 รูปแบบ "RGB888" จริงๆ เรียงเป็น BGR
# True = แปลง BGR เป็น RGB ก่อนส่งให้โมเดล
CAMERA_RETURNS_BGR = True

# ----------------------------------------------------------------------
# การขับมอเตอร์ (ต้อง calibrate กับรถจริง)
# ----------------------------------------------------------------------
BASE_SPEED = 30.0               # PWM % ที่ใช้เป็นความเร็วเฉลี่ย (ตรงกับ throttle คงที่ 0.4 ใน sim)

# ผลต่าง PWM ระหว่างล้อตอน |steering| = 1
# left = BASE + MAX_DIFF*s, right = BASE - MAX_DIFF*s  (s>0 = เลี้ยวขวา)
# ความเร็วเฉลี่ยคงที่เหมือน sim และอัตราการเลี้ยวเป็นเส้นตรงกับ action
MAX_DIFF = 25.0

# 1.0 = ใช้ action ของโมเดลตรงๆ ไม่บิดค่า ถ้ารถเลี้ยวไม่พอให้เพิ่ม MAX_DIFF แทน
STEERING_GAIN = 1.0

MIN_WHEEL_SPEED = 7.5           # PWM ต่ำสุดที่ล้อยังหมุนได้
MAX_WHEEL_SPEED = 100.0

IN1, IN2, IN3, IN4 = 24, 23, 27, 22
ENA, ENB = 12, 13
PWM_FREQUENCY = 100

SAVE_DEBUG_IMAGE = True         # เซฟภาพที่ส่งเข้าโมเดล 1 ครั้ง เพื่อเทียบกับ ai_vision.jpg

running = True

pwm_a = None
pwm_b = None
picam2 = None


# ----------------------------------------------------------------------
# Motor helpers
# ----------------------------------------------------------------------
def clamp_duty_cycle(value):
    return float(np.clip(value, 0.0, 100.0))


def set_speed(left_speed, right_speed):
    """ตั้ง PWM ของล้อซ้ายและขวาเป็นเปอร์เซ็นต์ 0-100"""
    pwm_a.ChangeDutyCycle(clamp_duty_cycle(left_speed))
    pwm_b.ChangeDutyCycle(clamp_duty_cycle(right_speed))


def set_forward_direction():
    """ตั้งให้มอเตอร์ทั้งสองหมุนไปข้างหน้า"""
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)

    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)


def stop_car():
    """หยุดรถทันที"""
    if pwm_a is not None:
        pwm_a.ChangeDutyCycle(0)

    if pwm_b is not None:
        pwm_b.ChangeDutyCycle(0)

    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)


def drive_from_steering(raw_steering):
    """
    -1.0 = เลี้ยวซ้ายสุด, 0.0 = ตรง, +1.0 = เลี้ยวขวาสุด (เหมือนใน Unity)
    ความเร็วเฉลี่ยของสองล้อคงที่ที่ BASE_SPEED
    """
    applied_steering = float(np.clip(raw_steering * STEERING_GAIN, -1.0, 1.0))

    left_speed = BASE_SPEED + MAX_DIFF * applied_steering
    right_speed = BASE_SPEED - MAX_DIFF * applied_steering

    left_speed = float(np.clip(left_speed, MIN_WHEEL_SPEED, MAX_WHEEL_SPEED))
    right_speed = float(np.clip(right_speed, MIN_WHEEL_SPEED, MAX_WHEEL_SPEED))

    set_forward_direction()
    set_speed(left_speed, right_speed)

    return applied_steering, left_speed, right_speed


# ----------------------------------------------------------------------
# Observation helpers
# ----------------------------------------------------------------------
def prepare_observation(frame):
    """แปลงภาพ Pi เป็น RGB uint8 ขนาด (OBS_HEIGHT, OBS_WIDTH, 3) ให้ตรงกับตอนเทรน"""

    if frame is None:
        raise RuntimeError("Picamera2 returned no frame")

    # บาง Pi camera configuration อาจส่ง 4 channels
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"Unexpected camera frame shape: {frame.shape}")

    # ครอปกลางภาพให้ได้สัดส่วนเท่ากับภาพตอนเทรน (คงมุมมองแนวตั้งไว้)
    height, width = frame.shape[:2]
    target_aspect = OBS_WIDTH / OBS_HEIGHT

    if width / height > target_aspect:
        new_width = int(round(height * target_aspect))
        x0 = (width - new_width) // 2
        frame = frame[:, x0:x0 + new_width]
    else:
        new_height = int(round(width / target_aspect))
        y0 = (height - new_height) // 2
        frame = frame[y0:y0 + new_height, :]

    frame = np.ascontiguousarray(frame)

    if CAMERA_RETURNS_BGR:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return cv2.resize(
        frame,
        (OBS_WIDTH, OBS_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


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
    running = False


def load_model():
    try:
        return PPO.load(str(MODEL_PATH), device="cpu")
    except Exception as exc:
        # มักเกิดเมื่อเวอร์ชัน Python บน Pi ต่างจากเครื่องที่เทรน
        print(f"[Model] Normal load failed: {exc}")
        print("[Model] Retrying with custom_objects...")
        return PPO.load(
            str(MODEL_PATH),
            device="cpu",
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            },
        )


def main():
    global pwm_a
    global pwm_b
    global picam2
    global running

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Not found: {MODEL_PATH}")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(f"[Model] Loading: {MODEL_PATH}")
    model = load_model()

    # ตรวจว่าโมเดลคาดหวังภาพขนาด/จำนวน channel เท่ากับที่เราจะส่งเข้าไป
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
    print(f"[Model] Loaded (observation shape {expected})")

    if BASE_SPEED - MAX_DIFF < MIN_WHEEL_SPEED or BASE_SPEED + MAX_DIFF > MAX_WHEEL_SPEED:
        print("[Warning] MAX_DIFF is clipped by wheel limits: average speed will drop in sharp turns")

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

    pwm_a = GPIO.PWM(ENA, PWM_FREQUENCY)
    pwm_b = GPIO.PWM(ENB, PWM_FREQUENCY)
    pwm_a.start(0)
    pwm_b.start(0)
    stop_car()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={
            "size": CAPTURE_SIZE,
            "format": "RGB888",
        }
    )

    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    pygame.init()
    screen = pygame.display.set_mode((320, 120))
    pygame.display.set_caption("RL Auto Drive - Q / ESC to stop")

    print(f"[Ready] Control rate {1.0 / CONTROL_PERIOD:.1f} Hz | Q / ESC = stop | Down Arrow = brake")

    frame_stack = FrameStack(N_STACK)
    debug_image_saved = not SAVE_DEBUG_IMAGE

    last_log_time = 0.0
    overruns = 0
    infer_ms = 0.0
    next_tick = time.monotonic()

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    request_stop()

            keys = pygame.key.get_pressed()

            if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
                request_stop()
                break

            # กดลูกศรลงค้างเพื่อเบรก (ล้าง stack เพื่อให้เริ่มใหม่แบบตอนต้น episode)
            if keys[pygame.K_DOWN]:
                stop_car()
                frame_stack.clear()
                screen.fill((180, 40, 40))
                pygame.display.flip()
                time.sleep(0.05)
                next_tick = time.monotonic()
                continue

            raw_frame = picam2.capture_array()
            observation = prepare_observation(raw_frame)
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

            raw_steering = float(np.clip(action[0], -1.0, 1.0))
            applied_steering, left_speed, right_speed = drive_from_steering(raw_steering)

            screen.fill((35, 100, 45))
            pygame.display.flip()

            now = time.monotonic()
            if now - last_log_time >= 0.2:
                print(
                    f"\r[AI] raw={raw_steering:+.2f} | "
                    f"applied={applied_steering:+.2f} | "
                    f"L={left_speed:4.1f}% | "
                    f"R={right_speed:4.1f}% | "
                    f"infer={infer_ms:5.1f}ms | "
                    f"overruns={overruns}    ",
                    end="",
                    flush=True,
                )
                last_log_time = now

            # รักษาจังหวะ 1 step = CONTROL_PERIOD เหมือนตอนเทรน
            next_tick += CONTROL_PERIOD
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                overruns += 1
                next_tick = time.monotonic()

    finally:
        print("\n[Safety] Stopping motors...")
        try:
            stop_car()
        except Exception:
            pass

        for pwm in (pwm_a, pwm_b):
            try:
                if pwm is not None:
                    pwm.stop()
            except Exception:
                pass

        try:
            if picam2 is not None:
                picam2.stop()
                picam2.close()
        except Exception:
            pass

        try:
            pygame.quit()
        except Exception:
            pass

        GPIO.cleanup()
        print("[Safety] Done")


if __name__ == "__main__":
    main()