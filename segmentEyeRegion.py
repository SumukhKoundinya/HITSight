import sys
from types import ModuleType

# 1. Create fully populated mock modules for MediaPipe Audio to bypass Python 3.13 import issues
fake_audio_mod = ModuleType('mediapipe.tasks.python.audio')
fake_classifier_mod = ModuleType('mediapipe.tasks.python.audio.audio_classifier')

class FakeAudioClassifier:
    pass

class FakeAudioClassifierOptions:
    pass

class FakeAudioClassifierResult:
    pass

fake_classifier_mod.AudioClassifier = FakeAudioClassifier
fake_classifier_mod.AudioClassifierOptions = FakeAudioClassifierOptions
fake_classifier_mod.AudioClassifierResult = FakeAudioClassifierResult

fake_audio_mod.audio_classifier = fake_classifier_mod

sys.modules['mediapipe.tasks.python.audio'] = fake_audio_mod
sys.modules['mediapipe.tasks.python.audio.audio_classifier'] = fake_classifier_mod

# 2. Safely import dependencies
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import tkinter as tk
from tkinter import filedialog

# 3. Download Model
model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
model_path = "face_landmarker.task"

if not os.path.exists(model_path):
    print("Downloading face_landmarker model...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(model_url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully!")

# 4. Handle Target Image locally
target_file = 'images (1).png'

if not os.path.exists(target_file):
    print("Target image not found in script directory. Opening file picker...")
    root = tk.Tk()
    root.withdraw()  # Hide main window
    target_file = filedialog.askopenfilename(
        title="Select Face Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp")]
    )
    if not target_file:
        raise FileNotFoundError("No image file selected.")

print(f"Processing '{target_file}'...")

# 5. Load MediaPipe Vision Tasks
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

# 6. Run Segmentation, Crop High-Res Grayscale Eyes
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    mp_image = mp.Image.create_from_file(target_file)
    detection_result = landmarker.detect(mp_image)
    image = cv2.imread(target_file)
    h, w, _ = image.shape

    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]

        LEFT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        RIGHT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

        def get_contour_pts(indices):
            pts = []
            for idx in indices:
                lm = face_landmarks[idx]
                pts.append((int(lm.x * w), int(lm.y * h)))
            return np.array(pts, dtype=np.int32)

        def extract_tight_eye_crop(contour_pts, target_size=(512, 512), padding_ratio=0.3):
            x, y, box_w, box_h = cv2.boundingRect(contour_pts)

            pad_x = int(box_w * padding_ratio)
            pad_y = int(box_h * padding_ratio)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + box_w + pad_x)
            y2 = min(h, y + box_h + pad_y)

            cropped_bgr = image[y1:y2, x1:x2]
            cropped_gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)

            resized = cv2.resize(cropped_gray, target_size, interpolation=cv2.INTER_CUBIC)

            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            clean_output = clahe.apply(resized)

            return clean_output

        left_eye_pts = get_contour_pts(LEFT_EYE_CONTOUR)
        right_eye_pts = get_contour_pts(RIGHT_EYE_CONTOUR)

        left_eye_gray = extract_tight_eye_crop(left_eye_pts, target_size=(512, 512), padding_ratio=0.35)
        right_eye_gray = extract_tight_eye_crop(right_eye_pts, target_size=(512, 512), padding_ratio=0.35)

        cv2.imwrite('left_eye_gray.png', left_eye_gray)
        cv2.imwrite('right_eye_gray.png', right_eye_gray)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(left_eye_gray, cmap='gray')
        axes[0].set_title(f"Left Eye Input ({left_eye_gray.shape[1]}x{left_eye_gray.shape[0]})")
        axes[0].axis('off')

        axes[1].imshow(right_eye_gray, cmap='gray')
        axes[1].set_title(f"Right Eye Input ({right_eye_gray.shape[1]}x{right_eye_gray.shape[0]})")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()
        print("\n🎉 Extracted tight left and right eye grayscale crops successfully!")
    else:
        print("\n❌ MediaPipe could not map a face structure in that image.")