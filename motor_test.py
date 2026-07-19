import sys
import tty
import termios
import RPi.GPIO as GPIO

# ตั้งค่าพิน GPIO
IN1, IN2, IN3, IN4 = 24, 23, 27, 22
GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, IN3, IN4], GPIO.OUT)

def stop_car():
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)

def getch():
    """ฟังก์ชันดักจับปุ่มกดอักขระดิบบน Linux Terminal"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# สั่งหยุดรถไว้ก่อนเริ่ม
stop_car()

print("ระบบคีย์บอร์ดสไตล์ Terminal พร้อมรัน!")
print("กด: W (เดินหน้า) | S (ถอยหลัง) | A (เลี้ยวซ้าย) | D (เลี้ยวขวา) | Spacebar (หยุด) | Q (ออก)")

try:
    while True:
        char = getch().lower()
        
        if char == 'w':
            print("\rเดินหน้า (Forward) ", end="")
            GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
            GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
        elif char == 's':
            print("\rถอยหลัง (Backward)", end="")
            GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
            GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
        elif char == 'a':
            print("\rเลี้ยวซ้าย (Left)  ", end="")
            GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
            GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
        elif char == 'd':
            print("\rเลี้ยวขวา (Right) ", end="")
            GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
            GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
        elif char == ' ':
            print("\rหยุดรถ (Stop)     ", end="")
            stop_car()
        elif char == 'q':
            print("\rปิดระบบควบคุม      ")
            break
finally:
    stop_car()
    GPIO.cleanup()