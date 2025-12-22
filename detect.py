"""
yolo11_webcam.py

Usage:
    python yolo11_webcam.py

Description:
    - Loads your best trained YOLOv11 model (best.pt).
    - Captures real-time video from the webcam.
    - Displays detections in a window.
    - Press 'q' to exit.

Requirements:
    pip install ultralytics opencv-python torch
    (and have CUDA installed if you want to use GPU)

Adjust the BEST_MODEL_PATH to point to your best.pt.
"""

import os
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# ==========================
# 1) Basic Configuration
# ==========================

# Path to the best trained model.
# If you copied best.pt to "models/best_yolo11.pt" with your training script,
# this path should work directly.
BEST_MODEL_PATH = os.getenv("YOLO_BEST_MODEL", "models/best_yolo11.pt")

# Camera index:
#   0 → built-in webcam
#   1, 2, ... → other connected cameras
CAM_INDEX = int(os.getenv("YOLO_CAM_INDEX", "0"))

# Image size that the model will use (optional but helps performance)
IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "640"))

# Confidence threshold to display detections
CONF_THRES = float(os.getenv("YOLO_CONF", "0.25"))


# ==========================
# 2) Model Verification
# ==========================

best_model_path = Path(BEST_MODEL_PATH)

if not best_model_path.exists():
    raise FileNotFoundError(
        f"Model not found at '{best_model_path.resolve()}'.\n"
        "Make sure BEST_MODEL_PATH points to your best.pt,"
        " for example: 'models/best_yolo11.pt' or 'runs-yolo11/exp/weights/best.pt'."
    )

print(f"✅ Loading model from: {best_model_path.resolve()}")


# ==========================
# 3) Device Selection
# ==========================

device = 0 if torch.cuda.is_available() else "cpu"
print(f"🖥️  PyTorch CUDA available: {torch.cuda.is_available()} | device used: {device}")


# ==========================
# 4) Load YOLO Model
# ==========================

model = YOLO(str(best_model_path))


# ==========================
# 5) Initialize Camera
# ==========================

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        f"❌ Could not open camera with index {CAM_INDEX}. "
        f"Try changing CAM_INDEX or check camera permissions."
    )

print("🎥 Camera opened successfully. Press 'q' to exit.")


# Optional: configure camera resolution (uncomment if you want to force a specific size).
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ==========================
# 6) Main Video Loop
# ==========================

try:
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("⚠ Could not read a frame from camera. Exiting...")
            break

        start_time = time.time()

        # Run prediction with YOLOv11
        # - source: the frame (numpy array)
        # - imgsz: input size for the model
        # - conf: confidence threshold to filter detections
        # - device: GPU (0) or CPU ("cpu")
        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            device=device,
            verbose=False  # to avoid spamming the console
        )

        # results is a list; we take the first prediction.
        annotated_frame = results[0].plot()  # Draws boxes, labels, etc.

        # Calculate approximate FPS
        end_time = time.time()
        fps = 1.0 / (end_time - start_time + 1e-8)

        # Draw FPS in the top left corner
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Display the annotated frame
        cv2.imshow("YOLOv11 - Webcam", annotated_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 'q' pressed. Exiting...")
            break

finally:
    # Release the camera and close windows even if there's an exception
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera released and windows closed.")


# End of script
