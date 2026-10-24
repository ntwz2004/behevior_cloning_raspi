"""
Shared motor control, camera pre-processing and loop timing for BC / DAgger / RL.

ทุกวิธี (BC, DAgger, RL) ต้อง import ไฟล์นี้ เพื่อให้ได้
  - โปรไฟล์ความเร็ว/การเลี้ยวชุดเดียวกัน
  - มุมมองกล้อง (FOV/ครอป) เดียวกัน
  - จังหวะการควบคุม (control rate) เดียวกัน

ทดสอบความเร็วจริงด้วย:
    python car_config.py straight 3     # วิ่งตรง 3 วินาที
    python car_config.py left 3         # หักซ้ายสุด 3 วินาที
    python car_config.py right 3
"""

import sys
import time

import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# ----------------------------------------------------------------------
# Speed profile: แก้ที่นี่ที่เดียว แล้วทุกวิธีจะเปลี่ยนตามกัน
# ----------------------------------------------------------------------
BASE_SPEED = 30.0       # PWM % เฉลี่ยของสองล้อ (คงที่ทั้งตอนตรงและตอนเลี้ยว) # 30,25
MAX_DIFF = 25         # ผลต่างต่อข้างตอน |steering| = 1
                        # 37.5 = BASE_SPEED - MIN_WHEEL_SPEED: ล้อในเกือบหยุด ล้อนอก 82.5% (เลี้ยวแคบสุดโดยไม่ให้ล้อถอย)
                        # left = BASE + MAX_DIFF*s, right = BASE - MAX_DIFF*s (s>0 = ขวา)

# ชดเชยมอเตอร์สองข้างไม่เท่ากัน (ปรับจนสั่ง steering=0 แล้ววิ่งตรง)
LEFT_TRIM = 1.0
RIGHT_TRIM = 1.0

MIN_WHEEL_SPEED = 10.0   # PWM ต่ำสุดที่ล้อยังหมุนได้
MAX_WHEEL_SPEED = 100.0

# จังหวะการตัดสินใจ: 0.1 วินาที = 10 Hz (= fixedStepsPerAction x fixedDeltaTime ตอนเทรน RL)
CONTROL_PERIOD = 0.1

# action แบบ discrete ของ BC/DAgger -> steering ต่อเนื่อง
# ทำให้ "left/right" เท่ากับ steering = -1/+1 ของ RL พอดี
DISCRETE_STEERING = {
    "forward": 0.0,
    "left": -1.0,
    "right": 1.0,
}

# ----------------------------------------------------------------------
# Camera: จับภาพ 4:3 ขนาดเดียวกัน แล้วครอปกลาง + ย่อ (FOV เหมือนกันทุกวิธี)
# ----------------------------------------------------------------------
CAPTURE_SIZE = (640, 480)
CAMERA_FORMAT = "RGB888"    # Picamera2 เรียง byte เป็น B,G,R จริงๆ

# ----------------------------------------------------------------------
# GPIO
# ----------------------------------------------------------------------
IN1, IN2, IN3, IN4 = 24, 23, 27, 22
ENA, ENB = 12, 13
PWM_FREQUENCY = 100

_pwm_a = None   # ล้อซ้าย
_pwm_b = None   # ล้อขวา


def setup_motors():
    global _pwm_a, _pwm_b

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

    _pwm_a = GPIO.PWM(ENA, PWM_FREQUENCY)
    _pwm_b = GPIO.PWM(ENB, PWM_FREQUENCY)
    _pwm_a.start(0)
    _pwm_b.start(0)
    stop_car()

    print(
        f"[Car] BASE_SPEED={BASE_SPEED}% MAX_DIFF={MAX_DIFF}% "
        f"trim L={LEFT_TRIM} R={RIGHT_TRIM} period={CONTROL_PERIOD}s"
    )

    if BASE_SPEED - MAX_DIFF < MIN_WHEEL_SPEED or BASE_SPEED + MAX_DIFF > MAX_WHEEL_SPEED:
        print(
            "[Warning] MAX_DIFF is clipped by wheel limits: "
            "average speed will change in sharp turns"
        )


def stop_car():
    if _pwm_a is not None:
        _pwm_a.ChangeDutyCycle(0)

    if _pwm_b is not None:
        _pwm_b.ChangeDutyCycle(0)

    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)


def cleanup_motors():
    global _pwm_a, _pwm_b

    for pwm in (_pwm_a, _pwm_b):
        try:
            if pwm is not None:
                pwm.stop()
        except Exception:
            pass

    _pwm_a = None
    _pwm_b = None
    GPIO.cleanup()


def drive_from_steering(steering):
    """
    steering: -1 = เลี้ยวซ้ายสุด, 0 = ตรง, +1 = เลี้ยวขวาสุด
    ความเร็วเฉลี่ยของสองล้อคงที่ที่ BASE_SPEED เสมอ
    คืนค่า (left_pwm, right_pwm)
    """
    s = float(np.clip(steering, -1.0, 1.0))

    left = (BASE_SPEED + MAX_DIFF * s) * LEFT_TRIM
    right = (BASE_SPEED - MAX_DIFF * s) * RIGHT_TRIM

    left = float(np.clip(left, MIN_WHEEL_SPEED, MAX_WHEEL_SPEED))
    right = float(np.clip(right, MIN_WHEEL_SPEED, MAX_WHEEL_SPEED))

    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

    _pwm_a.ChangeDutyCycle(left)
    _pwm_b.ChangeDutyCycle(right)

    return left, right


# ----------------------------------------------------------------------
# Camera helpers
# ----------------------------------------------------------------------
def make_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CAPTURE_SIZE, "format": CAMERA_FORMAT}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    return picam2


def prepare_observation(frame, width, height, rgb=True):
    """
    ครอปกลางให้ได้สัดส่วน width:height (คงมุมมองแนวตั้งไว้) แล้วย่อด้วย INTER_AREA
    rgb=True  -> คืนภาพ RGB
    rgb=False -> คืนภาพ BGR (ลำดับเดิมของ Picamera2)
    """
    if frame is None:
        raise RuntimeError("Picamera2 returned no frame")

    # บาง configuration ส่ง 4 channels
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"Unexpected camera frame shape: {frame.shape}")

    src_h, src_w = frame.shape[:2]
    target_aspect = width / height

    if src_w / src_h > target_aspect:
        new_w = int(round(src_h * target_aspect))
        x0 = (src_w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    else:
        new_h = int(round(src_w / target_aspect))
        y0 = (src_h - new_h) // 2
        frame = frame[y0:y0 + new_h, :]

    frame = np.ascontiguousarray(frame)

    if rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------
# Fixed-rate loop
# ----------------------------------------------------------------------
class LoopTimer:
    """รักษาจังหวะ 1 รอบ = period วินาที และนับจำนวนรอบที่ช้ากว่ากำหนด"""

    def __init__(self, period):
        self.period = period
        self.overruns = 0
        self.next_tick = time.monotonic()

    def reset(self):
        self.next_tick = time.monotonic()

    def wait(self):
        self.next_tick += self.period
        remaining = self.next_tick - time.monotonic()

        if remaining > 0:
            time.sleep(remaining)
        else:
            self.overruns += 1
            self.next_tick = time.monotonic()


# ----------------------------------------------------------------------
# Calibration: python car_config.py [straight|left|right] [seconds]
# ----------------------------------------------------------------------
if __name__ == "__main__":
    modes = {"straight": 0.0, "left": -1.0, "right": 1.0}
    mode = sys.argv[1] if len(sys.argv) > 1 else "straight"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0

    if mode not in modes:
        raise SystemExit("usage: python car_config.py [straight|left|right] [seconds]")

    setup_motors()
    try:
        print(f"[Calibrate] '{mode}' for {seconds}s in 3 seconds. Place the car at the start mark.")
        time.sleep(3)
        left_pwm, right_pwm = drive_from_steering(modes[mode])
        print(f"[Calibrate] Driving L={left_pwm:.1f}% R={right_pwm:.1f}%")
        time.sleep(seconds)
    finally:
        stop_car()
        cleanup_motors()
        print("[Calibrate] Stopped. Measure the distance / turning radius now.")