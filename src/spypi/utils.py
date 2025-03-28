import cv2
import numpy as np
import pygame
import os

HERE = os.path.dirname(__file__)

# Initialize pygame mixer.
pygame.mixer.init()

# Load the greeting sound.
_greeting_sound = pygame.mixer.Sound(f"{HERE}/sfx/hello_youre_on_camera.mp3")

def play_greeting():
    """Play the greeting sound using pygame."""
    try:
        _greeting_sound.play()
    except Exception as e:
        print("Error playing greeting:", e)



# Visualization parameters.
MARGIN = 10      # pixels
ROW_SIZE = 30    # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 0)  # black

def visualize(image: np.ndarray, detection_result) -> np.ndarray:
    """
    Draw bounding boxes on the image for detections labeled as 'person'.
    """
    if detection_result is None:
        return image

    for detection in detection_result.detections:
        category = detection.categories[0]
        category_name = category.category_name.lower()
        if category_name != "person":
            continue

        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(image, start_point, end_point, (0, 165, 255), 3)

        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return image
