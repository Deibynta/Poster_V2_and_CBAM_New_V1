from deepface import DeepFace
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time


def face_detection(img_path):
    currtime = time.strftime("%H:%M:%S")
    face_objs = DeepFace.extract_faces(np.array(img_path), detector_backend="mtcnn")

    coordinates = face_objs[0]["facial_area"]
    image = img_path
    cropped_image = image.crop(
        (
            coordinates["x"],
            coordinates["y"],
            coordinates["x"] + coordinates["w"],
            coordinates["y"] + coordinates["h"],
        )
    )
    cropped_image.save(f"Images/test_{currtime}.jpg")
    return cropped_image
