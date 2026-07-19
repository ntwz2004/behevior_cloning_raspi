import os
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import torch
import torch.nn as nn
import pygame 

# ตั้งชื่อโฟลเดอร์สำหรับเก็บข้อมูล DAgger
DAGGER_DIR = "dagger_dataset"
if not os.path.exists(DAGGER_DIR):
    os.makedirs(DAGGER_DIR)

# นิยามโครงสร้างโมเดล NVIDIA CNN (ภาพ 160x120)
class NvidiaCNN(nn.Module):
    def __init__(self):
        super(NvidiaCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU()
        )
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 13, 100), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 3) 
        )
        
    def forward(self, x):
        return self.linear_layers(self.conv_layers(x))

# ตั้งค่าระดับความเร็ว
MAX_SPEED = 40    
TURN_SPEED = 70   

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
    set_speed(int(speed * 0.15), speed) 

def turn_right(speed):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW) 
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW) 
    set_speed(speed, int(speed * 0.15))

# Pygame initialization for capturing keyboard input
pygame.init()
screen = pygame.display.set_mode((300, 200))
pygame.display.set_caption("Car Control")

# Load the trained model
device = torch.device('cpu')
model = NvidiaCNN()
model.load_state_dict(torch.load("nvidia_model.pth", map_location=device))
model.eval()

softmax = nn.Softmax(dim=1)

# ตั้งค่าระบบกล้อง Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

print("DAgger Interactive System with Pygame Ready!")
print("Make sure the Pygame window is active/focused, then use Arrow Keys.")
time.sleep(1)
stop_car()

try:
    with torch.no_grad():
        while True:
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            
            frame = picam2.capture_array()
            
            # ครอปภาพและเตรียมเทนเซอร์สำหรับส่งให้ AI
            roi = frame[100:240, 0:320]
            img = cv2.resize(roi, (160, 120))
            img_tensor = img.transpose((2, 0, 1))
            img_tensor = img_tensor / 255.0
            
            input_tensor = torch.tensor(img_tensor, dtype=torch.float32).unsqueeze(0).to(device)

            # AI Prediction
            outputs = model(input_tensor)
            probabilities = softmax(outputs)[0]
            ai_choice = torch.argmax(probabilities).item()
            confidence = probabilities[ai_choice].item()

            # DAgger Override
            is_override = False
            user_choice = -1
            label_name = ""

            if keys[pygame.K_UP]:
                move_forward(MAX_SPEED)
                user_choice = 0
                label_name = "forward"
                is_override = True
                print(f"\r[DAgger Override] Forward (Speed: {MAX_SPEED}%)", end="")
                
            elif keys[pygame.K_LEFT]:
                turn_left(TURN_SPEED)
                user_choice = 1
                label_name = "left"
                is_override = True
                print(f"\r[DAgger Override] <<< Left (Speed: {TURN_SPEED}%)", end="")
                
            elif keys[pygame.K_RIGHT]:
                turn_right(TURN_SPEED)
                user_choice = 2
                label_name = "right"
                is_override = True
                print(f"\r[DAgger Override] >>> Right (Speed: {TURN_SPEED}%)", end="")
                
            elif keys[pygame.K_DOWN]:
                stop_car()
                print("\r[DAgger Override] Stopped", end="")
                time.sleep(0.1)
                continue
            
            if keys[pygame.K_q]:
                print("\nExiting program via Q key...")
                break

            if is_override:
                timestamp = int(time.time() * 1000)
                filename = f"{DAGGER_DIR}/{label_name}_{timestamp}.jpg"
                cv2.imwrite(filename, frame) 
                
            else:
                if ai_choice == 0:
                    move_forward(MAX_SPEED)
                    print(f"\r AI Mode: Forward (Conf: {confidence*100:.1f}%)", end="")
                elif ai_choice == 1:
                    turn_left(TURN_SPEED)
                    print(f"\r AI Mode: Left (Conf: {confidence*100:.1f}%)", end="")
                elif ai_choice == 2:
                    turn_right(TURN_SPEED)
                    print(f"\r AI Mode: Right (Conf: {confidence*100:.1f}%)", end="")
                
            time.sleep(0.005)

except KeyboardInterrupt:
    print("\nStopping the car...")
    
finally:
    stop_car()
    pwm_a.stop()
    pwm_b.stop()
    picam2.stop()
    picam2.close()
    GPIO.cleanup()
    pygame.quit() 
    print("Closed all resources successfully.")