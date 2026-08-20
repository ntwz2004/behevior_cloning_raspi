import os
import sys
import time
import cv2
import pygame
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# ตั้งค่าพิน GPIO สำหรับ L298N
IN1, IN2, IN3, IN4 = 24, 23, 27, 22
ENA, ENB = 12, 13  

GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

pwm_a = GPIO.PWM(ENA, 100)
pwm_b = GPIO.PWM(ENB, 100)
pwm_a.start(0)
pwm_b.start(0)

DRIVE_SPEED = 45  
TURN_SPEED = 75   

def set_speed(speed_left, speed_right):
    """ฟังก์ชันสั่งระดับความเร็วแยกฝั่งล้อ"""
    pwm_a.ChangeDutyCycle(speed_left)
    pwm_b.ChangeDutyCycle(speed_right)

def stop_car():
    set_speed(0, 0)
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)

def move_forward(speed):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    set_speed(speed, speed)

def move_backward(speed):
    GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
    set_speed(speed, speed)

def turn_left(speed):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    set_speed(int(speed * 0.2), speed) 

def turn_right(speed):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    set_speed(speed, int(speed * 0.2)) 

# ตั้งค่ากล้อง Picamera2 (กำหนดขนาด 240x200 ตรงตาม NvidiaCNN)
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (240, 200), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2) 

pygame.init()
screen = pygame.display.set_mode((300, 200))
pygame.display.set_caption("Dataset Collector")

DATASET_DIR = "dataset_obstacle_5"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

stop_car()
running = True
print("=======================================")
print(" Ready to collect dataset_4!")
print(" Press Arrow keys to drive | 'Q' to quit")
print("=======================================")

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        move_dir = None  
        steer_dir = None 

        if keys[pygame.K_UP]:
            move_dir = "forward"
        elif keys[pygame.K_DOWN]:
            move_dir = "backward"

        if keys[pygame.K_LEFT]:
            steer_dir = "left"
        elif keys[pygame.K_RIGHT]:
            steer_dir = "right"

        if keys[pygame.K_q]:
            running = False

        current_label = None

        if move_dir == "forward":
            if steer_dir == "left":
                current_label = "left"
                print(f"\rForward-Left (Speed: {TURN_SPEED}%)   ", end="")
                turn_left(TURN_SPEED)
            elif steer_dir == "right":
                current_label = "right"
                print(f"\rForward-Right (Speed: {TURN_SPEED}%)  ", end="")
                turn_right(TURN_SPEED)
            else:
                current_label = "forward"
                print(f"\rForward (Speed: {DRIVE_SPEED}%)        ", end="")
                move_forward(DRIVE_SPEED)

        elif move_dir == "backward":
            current_label = "backward"
            print(f"\rBackward (Speed: {DRIVE_SPEED}%)       ", end="")
            move_backward(DRIVE_SPEED)

        elif steer_dir == "left":
            current_label = "left"
            print(f"\rSpin Left (Speed: {TURN_SPEED}%)      ", end="")
            turn_left(TURN_SPEED)

        elif steer_dir == "right":
            current_label = "right"
            print(f"\rSpin Right (Speed: {TURN_SPEED}%)     ", end="")
            turn_right(TURN_SPEED)

        else:
            stop_car()
            print("\rStopped                            ", end="")

        if current_label:
            frame = picam2.capture_array()
            frame_resized = cv2.resize(frame, (240, 200))
            
            timestamp = int(time.time() * 1000)
            filename = f"{DATASET_DIR}/{current_label}_{timestamp}.jpg"
            cv2.imwrite(filename, frame_resized)
            
            screen.fill((0, 200, 100))
        else:
            screen.fill((50, 50, 50))  

        pygame.display.flip()
        time.sleep(0.005)

finally:
    stop_car()
    try:
        pwm_a.stop()
        pwm_b.stop()
    except:
        pass
    picam2.stop()
    picam2.close()
    pygame.quit()
    GPIO.cleanup()
    print("\nClosed resources successfully.")