from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)

picam2.configure(config)
picam2.start()

time.sleep(2)

while True:
    frame = picam2.capture_array()

    cv2.imshow("Pi Camera", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()