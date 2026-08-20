import os
import sys
import time
import cv2
import pygame
import RPi.GPIO as GPIO
from picamera2 import Picamera2

IN1, IN2, IN3, IN4 = 24, 23, 27, 22
ENA, ENB = 12, 13  

GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

pwm_a = GPIO.PWM(ENA, 100)
pwm_b = GPIO.PWM(ENB, 100)
pwm_a.start(0)
pwm_b.start(0)

DRIVE_SPEED = 50   
TURN_SPEED = 70    

def set_speed(speed_left, speed_right):
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

def turn_forward_left(speed):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    set_speed(int(speed * 0.2), speed)

def turn_forward_right(speed):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    set_speed(speed, int(speed * 0.2))

def spin_left(speed):
    GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    set_speed(speed, speed)

def spin_right(speed):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
    set_speed(speed, speed)

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "BGR888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2) 

pygame.init()
screen = pygame.display.set_mode((300, 200))
pygame.display.set_caption("4WD AI Car Control")

DATASET_DIR = "dataset"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

stop_car()
running = True
print("=======================================")
print(" 4WD Ready! Press Arrow keys to drive")
print(" Press 'Q' to quit")
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
                turn_forward_left(TURN_SPEED)
            elif steer_dir == "right":
                current_label = "right"
                print(f"\rForward-Right (Speed: {TURN_SPEED}%)  ", end="")
                turn_forward_right(TURN_SPEED)
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
            spin_left(TURN_SPEED)

        elif steer_dir == "right":
            current_label = "right"
            print(f"\rSpin Right (Speed: {TURN_SPEED}%)     ", end="")
            spin_right(TURN_SPEED)

        else:
            stop_car()
            print("\rStopped", end="")

        if current_label:
            frame = picam2.capture_array()
            timestamp = int(time.time() * 1000)
            filename = f"{DATASET_DIR}/{current_label}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
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
    sys.stderr = open(os.devnull, 'w')