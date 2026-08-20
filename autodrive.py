import os
import sys
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from model import NvidiaCNN
from picamera2 import Picamera2
import pygame 
import torch
import torch.nn as nn

# ตั้งชื่อโฟลเดอร์สำหรับเก็บข้อมูล DAgger
DAGGER_DIR = "dagger_dataset_obstecle_5"
if not os.path.exists(DAGGER_DIR):
    os.makedirs(DAGGER_DIR)

# ตั้งค่าระดับความเร็ว
MAX_SPEED = 45    
TURN_SPEED = 85

# ตั้งค่าพิน GPIO สำหรับขับมอเตอร์ L298N
IN1, IN2, IN3, IN4 = 24, 23, 27, 22
ENA, ENB = 12, 13 

GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

# เปิดสัญญาณ PWM ความถี่ 100Hz ที่พินคุมความเร็ว
pwm_a = GPIO.PWM(ENA, 100)
pwm_b = GPIO.PWM(ENB, 100)
pwm_a.start(0)
pwm_b.start(0)

def set_speed(speed_left, speed_right):
    """ฟังก์ชันคุมความเร็วล้อซ้ายและขวาอิสระ"""
    pwm_a.ChangeDutyCycle(speed_left)
    pwm_b.ChangeDutyCycle(speed_right)

def stop_car():
    set_speed(0, 0)
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)

def move_forward(speed):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    set_speed(speed, speed)

def turn_left(speed):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW) 
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW) 
    set_speed(int(speed * 0.14), speed) 

def turn_right(speed):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW) 
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW) 
    set_speed(speed, int(speed * 0.14))

# Pygame initialization for capturing keyboard input
pygame.init()
screen = pygame.display.set_mode((300, 200))
pygame.display.set_caption("DAgger Control Center")

# Load the trained model
device = torch.device('cpu')
model = NvidiaCNN().to(device)
model.load_state_dict(torch.load("nvidia_model_5_colab.pth", map_location=device))
model.eval()

softmax = nn.Softmax(dim=1)

# ตั้งค่าระบบกล้อง Picamera2 (ใช้อัตราส่วน 240x200 ตรงกับ NvidiaCNN)
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (240, 200), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)
stop_car()

running = True

try:
    with torch.no_grad():
        while running:
            # จัดการ Event คิวของ Pygame ไม่ให้หน้าต่างค้าง
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            
            # จับภาพจากกล้อง
            frame = picam2.capture_array()
            img_resized = cv2.resize(frame, (240, 200))
            img_tensor = img_resized.transpose((2, 0, 1)) / 255.0
            input_tensor = torch.tensor(img_tensor, dtype=torch.float32).unsqueeze(0).to(device)

            # AI Prediction
            outputs = model(input_tensor)
            probabilities = softmax(outputs)[0]
            ai_choice = torch.argmax(probabilities).item()
            confidence = probabilities[ai_choice].item()

            # DAgger Override Check
            is_override = False
            label_name = ""

            if keys[pygame.K_UP]:
                move_forward(MAX_SPEED)
                label_name = "forward"
                is_override = True
                print(f"\r[DAgger Override] Forward (Speed: {MAX_SPEED}%)       ", end="")
                
            elif keys[pygame.K_LEFT]:
                turn_left(TURN_SPEED)
                label_name = "left"
                is_override = True
                print(f"\r[DAgger Override] Left (Speed: {TURN_SPEED}%)      ", end="")
                
            elif keys[pygame.K_RIGHT]:
                turn_right(TURN_SPEED)
                label_name = "right"
                is_override = True
                print(f"\r[DAgger Override] Right (Speed: {TURN_SPEED}%)     ", end="")
                
            elif keys[pygame.K_DOWN]:
                stop_car()
                print("\r[DAgger Override] Stopped                            ", end="")
                screen.fill((200, 50, 50))
                pygame.display.flip()
                time.sleep(0.05)
                continue
            
            if keys[pygame.K_q]:
                print("\nExiting program via Q key...")
                running = False
                break

            # DAgger Data Collection
            if is_override:
                timestamp = int(time.time() * 1000)
                filename = f"{DAGGER_DIR}/{label_name}_{timestamp}.jpg"
                cv2.imwrite(filename, img_resized) 
                screen.fill((0, 200, 100)) 
            else:
                # ถ้าไม่ได้กดแทรก ให้ AI ขับอัตโนมัติ
                if ai_choice == 0:
                    move_forward(MAX_SPEED)
                    print(f"\r[AI Mode] Forward (Conf: {confidence*100:.1f}%)        ", end="")
                elif ai_choice == 1:
                    turn_left(TURN_SPEED)
                    print(f"\r[AI Mode] Left (Conf: {confidence*100:.1f}%)       ", end="")
                elif ai_choice == 2:
                    turn_right(TURN_SPEED)
                    print(f"\r[AI Mode] Right (Conf: {confidence*100:.1f}%)      ", end="")
                
                screen.fill((50, 50, 50)) 

            pygame.display.flip()
            time.sleep(0.005)

except KeyboardInterrupt:
    print("\nStopping the car manually...")

finally:
    stop_car()
    try:
        pwm_a.stop()
        pwm_b.stop()
    except:
        pass
    picam2.stop()
    picam2.close()
    GPIO.cleanup()
    pygame.quit() 
    print("Closed all resources successfully.")