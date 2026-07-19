import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import torch
import torch.nn as nn

# 🧠 1. นิยามโครงสร้างโมเดล NVIDIA CNN (ภาพ 160x120)
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

# ⚙️ 2. ตั้งค่าระดับความเร็ว (ปรับแต่งตัวเลขเปอร์เซ็นต์ 0-100 ตรงนี้ได้เลยครับน้อง)
MAX_SPEED = 40    # 🚀 ความเร็วตอนวิ่งทางตรง (เช่น 60%)
TURN_SPEED = 70   # 📉 ความเร็วตอนเข้าโค้งซ้าย/ขวา (ชะลอลงเหลือ 40% กันหลุดโค้ง)

# 🛠️ 3. ตั้งค่าพิน GPIO สำหรับขับมอเตอร์ L298N
IN1, IN2, IN3, IN4 = 24, 23, 27, 22
ENA, ENB = 12, 13  # พินสำหรับส่งสัญญาณ PWM คุมความเร็ว

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
    set_speed(speed, speed)  # ส่งความเร็วเท่ากันสองล้อ

def turn_left(speed):
    # ⬅️ เลี้ยวซ้าย: ล้อซ้ายเดินหน้าช้าๆ (ชะลอ) | ล้อขวาเดินหน้าเต็มความเร็วโค้ง
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW) # ล้อซ้ายเดินหน้า
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW) # ล้อขวาเดินหน้า
    
    # ทริค: ให้ล้อซ้ายหมุนแค่ 30% ของความเร็วโค้ง (ปรับตัวเลขคูณลด/เพิ่มได้ตามใจชอบ)
    set_speed(int(speed * 0.15), speed) 

def turn_right(speed):
    # ➡️ เลี้ยวขวา: ล้อซ้ายเดินหน้าเต็มความเร็วโค้ง | ล้อขวาเดินหน้าช้าๆ (ชะลอ)
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW) # ล้อซ้ายเดินหน้า
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW) # ล้อขวาเดินหน้า
    
    # ทริค: ให้ล้อขวาหมุนแค่ 30% ของความเร็วโค้ง
    set_speed(speed, int(speed * 0.15))

# 💾 4. โหลดสมองกลโมเดล PyTorch (.pth) รันบน CPU
device = torch.device('cpu')
model = NvidiaCNN()
model.load_state_dict(torch.load("nvidia_model.pth", map_location=device))
model.eval()

softmax = nn.Softmax(dim=1)

# 📹 5. ตั้งค่าระบบกล้อง Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "BGR888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

print("Car AI PyTorch System with Speed Control is ready!")
time.sleep(1)
stop_car()

try:
    with torch.no_grad():
        while True:
            frame = picam2.capture_array()
            
            # ครอปภาพเอาเฉพาะพื้นที่ถนนครึ่งล่าง
            roi = frame[100:240, 0:320]
            img = cv2.resize(roi, (160, 120))
            img = img.transpose((2, 0, 1))
            img = img / 255.0
            
            input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

            # ให้ AI วิเคราะห์ทิศทาง
            outputs = model(input_tensor)
            probabilities = softmax(outputs)[0]
            
            ai_choice = torch.argmax(probabilities).item()
            confidence = probabilities[ai_choice].item()

            # 🕹️ สั่งการล้อรถยนต์พร้อมกำหนดความเร็วตามคำสั่ง AI
            if ai_choice == 0:
                move_forward(MAX_SPEED)
                print(f"\r Forward (Speed: {MAX_SPEED}% | Conf: {confidence*100:.1f}%)", end="")
            elif ai_choice == 1:
                turn_left(TURN_SPEED)
                print(f"\r<<< Left  (Speed: {TURN_SPEED}% | Conf: {confidence*100:.1f}%)", end="")
            elif ai_choice == 2:
                turn_right(TURN_SPEED)
                print(f"\r>>> Right (Speed: {TURN_SPEED}% | Conf: {confidence*100:.1f}%)", end="")
                
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
    print("Closed all resources successfully.")