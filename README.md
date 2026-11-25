# Senior Project: Raspberry Pi Car

โครงงานควบคุมรถด้วยภาพจากกล้องและโมเดล AI บน Raspberry Pi มีโหมดขับด้วย PPO, โมเดลพฤติกรรม (BC) และ DAgger สำหรับเก็บข้อมูลแก้ไขการตัดสินใจของโมเดล

## โครงสร้างหลัก

```text
senior_project/
├── datasets/
│   ├── dataset_bc/          # ภาพสำหรับ Behavior Cloning
│   └── dagger_dataset/      # ภาพและ label ที่เก็บจาก DAgger
├── logs/dagger/             # Excel log จากแต่ละรอบ DAgger
├── models/
│   ├── bc/model_bc.pth
│   ├── dagger/model_dagger_3.pth
│   ├── dagger/model_unity_da_3.pth
│   └── ppo/1framestacking.zip
├── source/                  # โค้ด Python ทั้งหมดอยู่ในโฟลเดอร์นี้
└── venv/                    # Python virtual environment (ไม่ควรย้าย/commit)
```

## ติดตั้ง

ใช้ Raspberry Pi OS ที่เปิดใช้งานกล้องและ GPIO แล้ว จากโฟลเดอร์โปรเจกต์:

```bash
cd ~/senior_project
sudo apt update
sudo apt install -y python3-venv python3-picamera2 python3-rpi.gpio python3-opencv
python3 -m venv --system-site-packages venv
./venv/bin/pip install -r requirements.txt
```

แพ็กเกจกล้อง, GPIO และ OpenCV ของ Raspberry Pi OS มาจาก `apt`; `--system-site-packages` ช่วยให้ virtual environment เห็นแพ็กเกจเหล่านี้ได้ การติดตั้ง PyTorch/Stable-Baselines3 อาจต่างกันตามรุ่น Python และสถาปัตยกรรม Pi; หากติดตั้ง `torch` จาก pip ไม่ได้ ให้ติดตั้ง wheel ที่รองรับอุปกรณ์ก่อน แล้วค่อยติดตั้งรายการที่เหลือจาก requirements

## วิธีรัน

รันจาก project root (`~/senior_project`) เพื่อให้ path แบบ `models/...` และ `datasets/...` อ้างอิงได้ถูกต้อง ตัวอย่าง:

```bash
cd ~/senior_project
source venv/bin/activate
python3 source/drive_rl.py
```

สคริปต์หลัก:

- `source/collect_data.py` — เก็บภาพ/label สำหรับฝึก BC
- `source/autodrive.py` — ขับรถด้วยโมเดล BC
- `source/drive_dagger.py` — ขับด้วย AI พร้อมให้ผู้ใช้แทรกการควบคุมและเก็บข้อมูล DAgger
- `source/drive_rl.py` — ขับด้วยโมเดล PPO

เปิดหน้าต่าง terminal ที่ project root และ activate `venv` ก่อนรันแต่ละครั้ง กด `Ctrl+C` เพื่อหยุดโปรแกรมอย่างปลอดภัยเมื่อสคริปต์รองรับ

## เลือกโมเดล

แต่ละสคริปต์ใช้โมเดลคนละชนิด ให้แก้ค่าคงที่ path ในไฟล์ที่จะรัน โดยใช้ path แบบ relative จาก project root:

- PPO: `MODEL_PATH = "models/ppo/1framestacking.zip"` ใน `source/drive_rl.py`
- BC: MODEL_FILE = "models/bc/model_bc.pth" in `source/autodrive.py`
- DAgger: MODEL_FILE = "models/dagger/model_dagger_3.pth" in `source/drive_dagger.py`

เลือกไฟล์ที่มีอยู่จริงใน `models/` และตรงกับสถาปัตยกรรม/ขนาดภาพที่โค้ดคาดหวัง โมเดล PPO ใช้นามสกุล `.zip`; โมเดล PyTorch BC/DAgger ใช้นามสกุล `.pth` หากเปลี่ยนโมเดล DAgger ที่ฝึกจาก Unity ให้เลือกไฟล์ `model_unity_da_3.pth` และตรวจ preprocessing กับจำนวนคลาสให้ตรงกันด้วย

## หมายเหตุ

- อย่ารันหลายโปรแกรมที่ควบคุมมอเตอร์พร้อมกัน
- ตรวจพอร์ตมอเตอร์, กล้อง, ขนาดภาพ และค่าควบคุมใน `source/car_config.py` ก่อนใช้งาน
- โฟลเดอร์ `source/testcode/` เป็นโค้ดทดสอบแยกจากสคริปต์หลัก
