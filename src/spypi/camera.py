import cv2
import time

try:
    from picamera2 import Picamera2
    _use_picamera2 = True
except ImportError:
    _use_picamera2 = False
    print("Picamera2 not found; falling back to OpenCV VideoCapture.")

class Camera:
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.use_picamera2 = _use_picamera2
        if self.use_picamera2:
            self.cam = Picamera2()
            config = self.cam.create_video_configuration(main={"size": (self.width, self.height)})
            self.cam.configure(config)
            self.cam.start()
            time.sleep(2)  # Allow camera to warm up
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def capture_frame(self):
        if self.use_picamera2:
            return self.cam.capture_array()
        else:
            ret, frame = self.cap.read()
            if ret:
                return frame
            return None
