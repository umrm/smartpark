# scripts/capture.py
from picamera2 import Picamera2
import numpy as np

class Camera:
    def __init__(self, resolution=(1296, 972)):
        self.picam2 = Picamera2()
        self.resolution = resolution
        config = self.picam2.create_still_configuration(main={"size": self.resolution})
        self.picam2.configure(config)
        self.picam2.start()
        print(f"Camera initialized at resolution {self.resolution}")

    def capture_frame(self):
        frame = self.picam2.capture_array()
        return frame

    def stop(self):
        self.picam2.stop()
        print("Camera stopped.")


# if __name__ == "__main__":
#     cam = Camera()
#     frame = cam.capture_frame()
#     print(f"Captured frame shape: {frame.shape}")
#     cam.stop()
