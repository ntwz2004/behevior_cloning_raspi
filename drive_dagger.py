import os
import time
from datetime import datetime

import cv2
import pygame
import torch
import torch.nn as nn

import car_config as car
import dagger_log
from model import NvidiaCNN

# ตั้งชื่อโฟลเดอร์สำหรับเก็บข้อมูล DAgger
DAGGER_DIR = "dagger_dataset"
MODEL_FILE = "model_bc_dagger2.pth"

# โฟลเดอร์เก็บไฟล์ Excel (วางไว้นอกโฟลเดอร์รูป เพื่อไม่ปนกับข้อมูลเทรน)
# การรันแต่ละครั้งสร้างไฟล์ใหม่แยกกัน เช่น dagger_logs/dagger_round1_20261007_093000.xlsx
LOG_DIR = "dagger_logs"

# ชื่อเรียก session นี้ (ใส่ในชื่อไฟล์) เช่น "round1", "round2" เว้นว่างได้
SESSION_NAME = "round3"

# ขนาดภาพที่ NvidiaCNN รับ (ต้องตรงกับ RL และตอนเทรน: กว้าง 120 x สูง 100)
OBS_WIDTH = 120
OBS_HEIGHT = 100

# ลำดับสีที่ส่งเข้าโมเดล ต้องตรงกับวิธีที่ใช้ตอนเทรน
#   False = BGR (เหมือนโค้ดเดิมที่ไม่แปลงสี)
#   True  = RGB
OBS_RGB = False

# ตอนผู้ใช้แทรก: เซฟภาพใหม่เมื่อ (1) เริ่มกดใหม่ หรือ (2) กดค้างและผ่านไปอย่างน้อยกี่วินาที
# 0.3 = แตะ 1 ครั้งได้ 1 ภาพ, กดค้าง 1 วินาทีได้ ~4 ภาพ
# ตั้งเป็น 0 ถ้าต้องการเซฟทุกรอบ (10 ภาพ/วินาที)
SAVE_MIN_INTERVAL = 0.1

# output ของโมเดล: 0 = forward, 1 = left, 2 = right
AI_ACTIONS = {0: "forward", 1: "left", 2: "right"}

KEY_LABELS = {
    pygame.K_UP: "forward",
    pygame.K_LEFT: "left",
    pygame.K_RIGHT: "right",
}

os.makedirs(DAGGER_DIR, exist_ok=True)


def draw_status(screen, font, mode, session_saved, dataset_total, override_seconds):
    lines = [
        mode,
        f"Saved (session): {session_saved}",
        f"Dataset total: {dataset_total}",
        f"Override time: {override_seconds:6.1f} s",
    ]
    y = 12
    for line in lines:
        screen.blit(font.render(line, True, (255, 255, 255)), (12, y))
        y += 34


def main():
    device = torch.device("cpu")
    model = NvidiaCNN().to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    softmax = nn.Softmax(dim=1)

    car.setup_motors()
    picam2 = car.make_camera()

    pygame.init()
    screen = pygame.display.set_mode((300, 200))
    pygame.display.set_caption("DAgger Control Center")
    font = pygame.font.Font(None, 28)

    timer = car.LoopTimer(car.CONTROL_PERIOD)
    tracker = dagger_log.OverrideTracker()
    dataset_at_start = sum(dagger_log.count_dataset_images(DAGGER_DIR).values())

    running = True
    saved_count = 0
    agree_count = 0
    last_saved_label = None
    last_save_time = 0.0

    session_start_dt = datetime.now()
    session_start = time.monotonic()

    try:
        with torch.no_grad():
            while running:
                # แตะสั้นกว่า 1 รอบ (0.1 วินาที) ก็ไม่หลุด เพราะจับจาก KEYDOWN
                latched_label = None

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key in KEY_LABELS:
                        latched_label = KEY_LABELS[event.key]

                keys = pygame.key.get_pressed()

                if keys[pygame.K_q]:
                    print("\nExiting program via Q key...")
                    break

                # เบรก
                if keys[pygame.K_DOWN]:
                    car.stop_car()
                    tracker.update(None)
                    print("\r[DAgger Override] Stopped                            ", end="")
                    screen.fill((200, 50, 50))
                    draw_status(
                        screen, font, "BRAKE", saved_count,
                        dataset_at_start + saved_count, tracker.total_seconds(),
                    )
                    pygame.display.flip()
                    time.sleep(0.05)
                    last_saved_label = None
                    timer.reset()
                    continue

                # จับภาพ (ครอป/ย่อด้วยฟังก์ชันเดียวกับ RL)
                frame = picam2.capture_array()
                observation = car.prepare_observation(
                    frame, OBS_WIDTH, OBS_HEIGHT, rgb=OBS_RGB
                )

                input_tensor = (
                    torch.from_numpy(observation.transpose((2, 0, 1)).copy())
                    .float()
                    .div(255.0)
                    .unsqueeze(0)
                    .to(device)
                )

                # AI prediction (ทำทุกรอบ แม้ผู้ใช้แทรก เพื่อให้จังหวะเวลาเท่ากัน)
                probabilities = softmax(model(input_tensor))[0]
                ai_choice = torch.argmax(probabilities).item()
                confidence = probabilities[ai_choice].item()
                ai_label = AI_ACTIONS[ai_choice]

                # DAgger override: ซ้าย/ขวามาก่อน forward (เหมือน collect_data.py)
                human_label = None
                if keys[pygame.K_LEFT]:
                    human_label = "left"
                elif keys[pygame.K_RIGHT]:
                    human_label = "right"
                elif keys[pygame.K_UP]:
                    human_label = "forward"
                elif latched_label is not None:
                    human_label = latched_label

                now = time.monotonic()
                tracker.update(human_label, now)

                if human_label is not None:
                    left_pwm, right_pwm = car.drive_from_steering(
                        car.DISCRETE_STEERING[human_label]
                    )

                    should_save = (
                        human_label != last_saved_label
                        or now - last_save_time >= SAVE_MIN_INTERVAL
                    )

                    if should_save:
                        # เซฟภาพ (cv2.imwrite ต้องการ BGR)
                        save_image = (
                            cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                            if OBS_RGB
                            else observation
                        )
                        timestamp = int(time.time() * 1000)
                        cv2.imwrite(f"{DAGGER_DIR}/{human_label}_{timestamp}.jpg", save_image)

                        agreed = human_label == ai_label
                        saved_count += 1
                        if agreed:
                            agree_count += 1
                        tracker.add_image(agreed)

                        last_saved_label = human_label
                        last_save_time = now
                        screen.fill((0, 200, 100))
                    else:
                        screen.fill((0, 120, 70))

                    mode = "OVERRIDE"
                    print(
                        f"\r[DAgger Override] {human_label:<7} "
                        f"L={left_pwm:4.1f}% R={right_pwm:4.1f}% | AI said {ai_label:<7}"
                        f" | saved={saved_count} | override={tracker.total_seconds(now):5.1f}s"
                        f" | overruns={timer.overruns}   ",
                        end="",
                    )
                else:
                    # ปล่อยปุ่มแล้ว: การกดครั้งถัดไปนับเป็นการกดใหม่
                    last_saved_label = None

                    left_pwm, right_pwm = car.drive_from_steering(
                        car.DISCRETE_STEERING[ai_label]
                    )
                    screen.fill((50, 50, 50))
                    mode = "AI DRIVING"
                    print(
                        f"\r[AI Mode] {ai_label:<7} (Conf: {confidence * 100:5.1f}%) "
                        f"L={left_pwm:4.1f}% R={right_pwm:4.1f}% | overruns={timer.overruns}   ",
                        end="",
                    )

                draw_status(
                    screen, font, mode, saved_count,
                    dataset_at_start + saved_count, tracker.total_seconds(now),
                )
                pygame.display.flip()
                timer.wait()

    except KeyboardInterrupt:
        print("\nStopping the car manually...")

    finally:
        car.stop_car()
        car.cleanup_motors()
        try:
            picam2.stop()
            picam2.close()
        except Exception:
            pass
        pygame.quit()

        session_end_dt = datetime.now()
        total_seconds = time.monotonic() - session_start
        tracker.close()
        override_seconds = sum(segment["duration"] for segment in tracker.segments)
        override_pct = override_seconds / total_seconds * 100.0 if total_seconds > 0 else 0.0
        dataset_total = sum(dagger_log.count_dataset_images(DAGGER_DIR).values())

        print(
            f"\n[DAgger] Session {total_seconds:.0f}s | "
            f"Override {override_seconds:.1f}s ({override_pct:.1f}%) in "
            f"{len(tracker.segments)} presses | "
            f"Saved {saved_count} | Dataset total {dataset_total}"
        )
        if saved_count > 0:
            print(f"[DAgger] AI agreed with human: {agree_count / saved_count * 100:.1f}%")

        try:
            path = dagger_log.save_excel(
                LOG_DIR,
                SESSION_NAME,
                DAGGER_DIR,
                session_start_dt,
                session_end_dt,
                total_seconds,
                tracker,
                saved_count,
                agree_count,
                timer.overruns,
                {
                    "model": MODEL_FILE,
                    "base_speed": car.BASE_SPEED,
                    "max_diff": car.MAX_DIFF,
                    "control_period": car.CONTROL_PERIOD,
                    "save_min_interval": SAVE_MIN_INTERVAL,
                },
                dataset_at_start,
            )
            if path:
                print(f"[Excel] Saved: {os.path.abspath(path)}")
        except Exception as exc:
            print(f"[Excel] Could not save log: {exc}")

        print("Closed all resources successfully.")


if __name__ == "__main__":
    main()