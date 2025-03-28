import time
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from threading import Thread

from .camera import Camera
from .sound import play_greeting
from .utils import visualize
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os
import urllib.request

# Download the ML model if not present.
ML_MODEL = 'efficientdet.tflite'
if not os.path.exists(ML_MODEL):
    url = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite'
    urllib.request.urlretrieve(url, ML_MODEL)

class DetectionProcessor:
    def __init__(self, camera: Camera,
                 max_results: int = 2,
                 score_threshold: float = 0.25,
                 detection_interval: int = 5,
                 video_sleep: float = 0.05,
                 greeting_duration: int = 3):
        self.camera = camera
        self.max_results = max_results
        self.score_threshold = score_threshold
        self.detection_interval = detection_interval
        self.video_sleep = video_sleep
        self.greeting_duration = greeting_duration

        # FPS calculation.
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

        # Initialize Mediapipe detector.
        base_options = python.BaseOptions(model_asset_path=ML_MODEL)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            max_results=self.max_results,
            score_threshold=self.score_threshold
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

        self.cached_detection = None
        self.last_person_detected = False
        self.greeting_end_time = 0

    def process(self):
        frame = self.camera.capture_frame()
        if frame is None:
            print("No frame captured!")
            return None

        # If using Picamera2, the frame is in RGB.
        if self.camera.use_picamera2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame.copy()

        # Flip frame horizontally.
        frame_bgr = cv2.flip(frame_bgr, 1)

        # Update FPS every detection_interval frames.
        self.frame_count += 1
        if self.frame_count % self.detection_interval == 0:
            current_time = time.time()
            elapsed = current_time - self.start_time
            self.fps = self.detection_interval / elapsed if elapsed > 0 else 0
            self.start_time = current_time

            # Run detection.
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            try:
                self.cached_detection = self.detector.detect(mp_image)
            except Exception as e:
                print("Detection error:", e)

            # Check for person detection.
            person_detected = any(
                detection.categories[0].category_name.lower() == "person"
                for detection in (self.cached_detection.detections if self.cached_detection else [])
            )
            if person_detected and not self.last_person_detected:
                self.greeting_end_time = time.time() + self.greeting_duration
                play_greeting()
            self.last_person_detected = person_detected

        # Render detection bounding boxes.
        processed = visualize(frame_bgr, self.cached_detection)

        # Overlay current time.
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(processed, now, (10, self.camera.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        # Overlay FPS.
        cv2.putText(processed, f"FPS: {self.fps:.1f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        return processed

    def run(self, output_callback):
        """
        Continuously process frames and call output_callback(jpeg_bytes)
        to update the streaming output.
        """
        while True:
            frame = self.process()
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    output_callback(jpeg.tobytes())
            time.sleep(self.video_sleep)

    def start(self, output_callback):
        # Launch the detection loop in a separate thread.
        t = Thread(target=self.run, args=(output_callback,))
        t.daemon = True
        t.start()
