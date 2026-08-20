# -*- coding: utf-8 -*-
import time
import RPi.GPIO as GPIO
import pygame

# --- 1. GPIO Setup for L298N ---
IN1, IN2 = 24, 23  # Left Motor
IN3, IN4 = 27, 22  # Right Motor
ENA, ENB = 12, 13  # PWM Pins

TEST_SPEED = 100    # Speed (0 - 100%)

GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

pwm_a = GPIO.PWM(ENA, 100)
pwm_b = GPIO.PWM(ENB, 100)
pwm_a.start(0)
pwm_b.start(0)

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
    GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.HIGH)
    set_speed(speed, speed)

def turn_left(speed):
    GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    set_speed(speed, speed)

def turn_right(speed):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.HIGH)
    set_speed(speed, speed)

# --- 2. Pygame Init ---
pygame.init()
screen = pygame.display.set_mode((200, 200))
pygame.display.set_caption("Motor Test Window")

print("=======================================")
print("     Motor Testing System Ready!       ")
print("=======================================")
print("Controls:")
print("  - Arrow Up    : Move Forward")
print("  - Arrow Down  : Move Backward")
print("  - Arrow Left  : Turn Left")
print("  - Arrow Right : Turn Right")
print("  - Q Key       : Quit Test")
print("---------------------------------------")
print("Click on the Pygame window to control.")

try:
    running = True
    while running:
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_q]:
            print("\nExiting Motor Test...")
            break

        if keys[pygame.K_UP]:
            move_forward(TEST_SPEED)
            print(f"\r Status: FORWARD  (Speed: {TEST_SPEED}%)", end="")
            screen.fill((0, 200, 0))
        elif keys[pygame.K_DOWN]:
            move_backward(TEST_SPEED)
            print(f"\r Status: BACKWARD (Speed: {TEST_SPEED}%)", end="")
            screen.fill((200, 200, 0))
        elif keys[pygame.K_LEFT]:
            turn_left(TEST_SPEED)
            print(f"\r Status: <<< LEFT  (Speed: {TEST_SPEED}%)", end="")
            screen.fill((0, 100, 200))
        elif keys[pygame.K_RIGHT]:
            turn_right(TEST_SPEED)
            print(f"\r Status: >>> RIGHT (Speed: {TEST_SPEED}%)", end="")
            screen.fill((200, 0, 100))
        else:
            stop_car()
            print("\r Status: STOPPED                   ", end="")
            screen.fill((50, 50, 50))

        pygame.display.flip()
        time.sleep(0.01)

finally:
    stop_car()
    pwm_a.stop()
    pwm_b.stop()
    GPIO.cleanup()
    pygame.quit()
    print("\nCleaned up GPIO resources successfully.")