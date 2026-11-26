import os
import time

import cv2
import pygame

import sys
from pathlib import Path

import source.car_config as car

BASE_DIR = "datasets/bc"
SCENARIO = "reverse_obstacle"
DATASET_DIR = f"{BASE_DIR}/{SCENARIO}"

# ขนาดภาพ (กว้าง 120 x สูง 100) ต้องตรงกับ drive_dagger.py และ RL
OBS_WIDTH = 120
OBS_HEIGHT = 100

# ลำดับสีที่เก็บ/ส่งเข้าโมเดล
OBS_RGB = False

# ตรวจปุ่มและสั่งมอเตอร์ถี่ๆ เพื่อให้รถตอบสนองไว (50 Hz)
POLL_PERIOD = 0.02
SAVE_PERIOD = car.CONTROL_PERIOD

os.makedirs(DATASET_DIR, exist_ok=True)


def main():
    car.setup_motors()
    picam2 = car.make_camera()

    pygame.init()
    # ขยายหน้าจอให้กว้างขึ้นเล็กน้อยเพื่อแสดงข้อความได้ชัดเจน
    screen = pygame.display.set_mode((350, 220))
    pygame.display.set_caption("Dataset Collector")
    
    # โหลดฟอนต์สำหรับแสดงผล
    font = pygame.font.SysFont(None, 32)
    font_large = pygame.font.SysFont(None, 40)

    poll_timer = car.LoopTimer(POLL_PERIOD)
    saved_count = {"forward": 0, "left": 0, "right": 0}
    running = True

    current_label = None
    last_saved_label = None
    last_save_time = 0.0

    try:
        while running:
            # 1. ล้าง Event Queue เพื่อป้องกันการหน่วงจากปุ่มค้าง
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()

            if keys[pygame.K_q]:
                break

            # 2. เช็กสถานะปุ่ม
            label = None
            if keys[pygame.K_LEFT]:
                label = "left"
            elif keys[pygame.K_RIGHT]:
                label = "right"
            elif keys[pygame.K_UP]:
                label = "forward"

            # 3. สั่งมอเตอร์ทันทีที่สถานะปุ่มเปลี่ยน (ลดอาการดีเลย์รถค้าง)
            if label != current_label:
                if label is None:
                    car.stop_car()
                    print("\rStopped                                   ", end="")
                else:
                    left_pwm, right_pwm = car.drive_from_steering(
                        car.DISCRETE_STEERING[label]
                    )
                    print(
                        f"\r{label:<8} L={left_pwm:4.1f}% R={right_pwm:4.1f}%   "
                        f"saved={saved_count}   ",
                        end="",
                    )
                current_label = label

            now = time.monotonic()
            is_saving_this_frame = False

            # 4. ประมวลผลและเซฟภาพ
            if label is not None:
                if label != last_saved_label or now - last_save_time >= SAVE_PERIOD:
                    frame = picam2.capture_array()
                    observation = car.prepare_observation(
                        frame, OBS_WIDTH, OBS_HEIGHT, rgb=OBS_RGB
                    )
                    image = (
                        cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
                        if OBS_RGB
                        else observation
                    )
                    timestamp = int(time.time() * 1000)
                    cv2.imwrite(f"{DATASET_DIR}/{label}_{timestamp}.jpg", image)
                    saved_count[label] += 1

                    last_saved_label = label
                    last_save_time = time.monotonic() # ควรอัปเดตเวลาหลังเซฟเสร็จ
                    is_saving_this_frame = True
                    print(f"\r{label:<8} saved={saved_count}   ", end="")
                    poll_timer.reset()
            else:
                last_saved_label = None

            # 5. วาดสีพื้นหลังหน้าจอ
            if is_saving_this_frame:
                screen.fill((0, 150, 80))    # สีเขียวสว่างเมื่อกำลังเซฟภาพ
            elif label is not None:
                screen.fill((0, 80, 40))     # สีเขียวเข้มเมื่อกดค้างรอรอบเซฟ
            else:
                screen.fill((40, 40, 40))    # สีเทาเมื่อรถหยุดนิ่ง

            # 6. วาดข้อความจำนวนภาพ
            text_fwd = font.render(f"UP (Forward): {saved_count['forward']}", True, (255, 255, 255))
            text_left = font.render(f"LEFT (Left): {saved_count['left']}", True, (255, 255, 255))
            text_right = font.render(f"RIGHT (Right): {saved_count['right']}", True, (255, 255, 255))
            
            total_images = sum(saved_count.values())
            text_total = font_large.render(f"TOTAL: {total_images}", True, (255, 215, 0)) # สีทอง

            screen.blit(text_fwd, (20, 20))
            screen.blit(text_left, (20, 60))
            screen.blit(text_right, (20, 100))
            screen.blit(text_total, (20, 160))

            pygame.display.flip()
            poll_timer.wait()

    except KeyboardInterrupt:
        print("\nStopped manually.")

    finally:
        car.stop_car()
        car.cleanup_motors()
        try:
            picam2.stop()
            picam2.close()
        except Exception:
            pass
        pygame.quit()
        print(f"\nSaved images: {saved_count}")
        print("Closed resources successfully.")

if __name__ == "__main__":
    main()