"""
Complete script to train YOLOv11 with a Roboflow dataset.

Flow:
1. Download the dataset from Roboflow in "yolov11" format.
2. Locate the data.yaml file and correct the paths to be relative:
   - train/images
   - valid/images
   - test/images
3. Adjust 'nc' according to the 'names' list in the YAML.
4. Train YOLOv11 with ultralytics.
5. Copy the best model (best.pt) to a final, more convenient path.
6. Execute:
   - Model validation.
   - Sample predictions on some validation images.
   - Predictions on ALL validation and test images.
"""

import os
from pathlib import Path
import yaml
import traceback
import shutil

from roboflow import Roboflow
from ultralytics import YOLO
import torch


# ==========================
# 1) Dataset Download
# ==========================

# You can also define these variables as environment variables if you prefer.
API_KEY  = os.getenv("ROBOFLOW_API_KEY", "xRR2YKoSgar7X4ACVIpO")
WORKSPACE = "yolo-3b0a6"
PROJECT_NAME = "my-first-project-4mpbz"
VERSION = 2
EXPORT_FORMAT = "yolov11"  # Format compatible with ultralytics YOLOv11

# Connection with Roboflow and dataset download.
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT_NAME)
version = project.version(VERSION)
dataset = version.download(EXPORT_FORMAT)  # Usually brings .location with the base path


# =======================================
# 2) Locate the Dataset Root Folder
# =======================================

candidates = []

# 2.1) If the dataset object has "location" attribute, use it as a candidate.
if hasattr(dataset, "location"):
    candidates.append(Path(dataset.location))

# 2.2) Common naming pattern: "<project_name>-<version>"
candidates.append(Path(f"{project.name}-{version.version}"))

# 2.3) Scan the current directory for a folder containing data.yaml
for p in Path(".").iterdir():
    if p.is_dir() and (p / "data.yaml").exists():
        candidates.append(p)

# Select the first candidate that exists.
root = None
for c in candidates:
    if c.exists():
        root = c
        break

assert root and root.exists(), f"Could not locate the dataset folder. Tried: {candidates}"
print(f"📂 Dataset root: {root.resolve()}")

# Path to the main data.yaml file.
yaml_path = root / "data.yaml"
if not yaml_path.exists():
    # Deep search in case data.yaml is not in the root.
    found = list(root.rglob("data.yaml"))
    assert found, f"data.yaml not found anywhere within {root}"
    yaml_path = found[0]

print(f"📝 data.yaml located at: {yaml_path.resolve()}")


# ============================================
# 3) Adjust data.yaml to Use Relative Paths
# ============================================

with yaml_path.open("r") as f:
    data = yaml.safe_load(f)

# Mapping of expected keys by ultralytics and typical Roboflow folder paths.
# - The key in the YAML must be 'val' (not 'valid') for ultralytics.
# - The folder is usually called 'valid/images'.
split_map = {
    "train": "train/images",
    "val":   "valid/images",
    "test":  "test/images",
}

# Force correct relative paths if folders exist.
for key, rel_path in split_map.items():
    p = yaml_path.parent / rel_path
    if p.exists():
        # If the folder exists, update the YAML entry with that relative path.
        data[key] = rel_path
        print(f"✔ {key} → {rel_path} (folder found)")
    else:
        # If the folder doesn't exist, show warning and don't touch that key.
        if key in data:
            print(f"⚠ The folder for '{key}' was not found at {rel_path}; YAML entry left as is.")
        else:
            print(f"ℹ The '{key}' split is not defined in the YAML.")

# Ensure 'nc' matches the number of classes in 'names', if it exists.
if "names" in data and isinstance(data["names"], list):
    data["nc"] = len(data["names"])

# Save the updated YAML.
with yaml_path.open("w") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print("\n✅ YAML patched correctly. Content:")
print(data)

# Quick summary of final paths for train/val/test.
for k in ("train", "val", "test"):
    if k in data:
        dp = (yaml_path.parent / data[k]).resolve()
        print(f"{k}: {data[k]} → {dp} (exists: {dp.exists()})")


# =============================
# 4) Configure and Train YOLO
# =============================

# Main training parameters (can be adjusted via environment variables).
MODEL   = os.getenv("YOLO_MODEL", "yolo11n.pt")   # e.g. yolo11s.pt, yolo11m.pt, etc.
EPOCHS  = int(os.getenv("YOLO_EPOCHS", "100"))
IMGSZ   = int(os.getenv("YOLO_IMGSZ",  "640"))
BATCH   = int(os.getenv("YOLO_BATCH",  "16"))
WORKERS = int(os.getenv("YOLO_WORKERS", "4"))
RUNS_PROJECT = os.getenv("YOLO_PROJECT", "runs-yolo11")
RUN_NAME     = os.getenv("YOLO_NAME",    "exp")

# Path where a "final" copy of the best trained model will be saved.
FINAL_WEIGHTS = os.getenv("YOLO_FINAL_WEIGHTS", "models/best_yolo11.pt")

# Automatic device selection: GPU (0) if CUDA is available, otherwise CPU.
device = 0 if torch.cuda.is_available() else "cpu"
print(f"\n🖥️  PyTorch CUDA available: {torch.cuda.is_available()} | device used by YOLO: {device}")

# Load base (pretrained) YOLOv11 model.
model = YOLO(MODEL)

# Main training.
results = model.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    workers=WORKERS,
    device=device,
    project=RUNS_PROJECT,
    name=RUN_NAME,
    exist_ok=True,  # Allows reusing the folder if it already exists.
)

print("\n🚀 Training started. Check the results folder at:")
print(Path(RUNS_PROJECT) / RUN_NAME)


# ====================================================
# 5) Validation, best.pt Copy and Prediction Generation
# ====================================================

try:
    # Path where ultralytics saves the best model from this experiment.
    best = Path(RUNS_PROJECT) / RUN_NAME / "weights" / "best.pt"

    if best.exists():
        print(f"\n✅ best.pt found at: {best.resolve()}")

        # 5.1) Copy the best model to a simpler location (for example, /models).
        final_path = Path(FINAL_WEIGHTS)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, final_path)
        print(f"💾 Best model copy saved at: {final_path.resolve()}")

        # 5.2) Validate the best model on the data defined in data.yaml.
        print("\n🔎 Running validation on best.pt ...")
        model = YOLO(str(best))
        model.val(data=str(yaml_path), imgsz=IMGSZ, device=device)

        # 5.3) Sample predictions (a few images from 'valid' split, if it exists).
        valid_img_dir = yaml_path.parent / "valid" / "images"
        if valid_img_dir.exists():
            # Take up to 6 images for quick visualization.
            sample_imgs = [str(p) for p in list(valid_img_dir.glob("*.*"))[:6]]
            if sample_imgs:
                print("\n🖼️ Generating sample predictions on some validation images...")
                model.predict(
                    source=sample_imgs,
                    imgsz=IMGSZ,
                    save=True,
                    project=RUNS_PROJECT,
                    name=f"{RUN_NAME}_pred_samples",
                    device=device,
                    conf=0.25,
                )
                print("📦 Sample predictions saved at:")
                print(Path(RUNS_PROJECT) / f"{RUN_NAME}_pred_samples")

        # 5.4) Predictions on ALL images in validation and test (if they exist).
        for split_name, split_rel in [("valid", "valid/images"),
                                      ("test",  "test/images")]:
            split_dir = yaml_path.parent / split_rel
            if split_dir.exists():
                out_name = f"{RUN_NAME}_{split_name}_pred_all"
                print(f"\n📂 Running predictions on ALL images at: {split_dir} ...")

                # Passing the directory as 'source' makes YOLO process all images in the folder.
                model.predict(
                    source=str(split_dir),
                    imgsz=IMGSZ,
                    save=True,
                    project=RUNS_PROJECT,
                    name=out_name,
                    device=device,
                    conf=0.25,
                )
                print(f"✅ Complete predictions for '{split_name}' saved at:")
                print(Path(RUNS_PROJECT) / out_name)
            else:
                print(f"ℹ Folder for '{split_name}' split not found at {split_dir}. Skipping this split.")

    else:
        # If best.pt doesn't exist, something probably failed during training.
        print("ℹ best.pt not found; validation and predictions are skipped.")

except Exception:
    # Any error in the optional validation/prediction section is caught so the script doesn't crash.
    print("⚠ An error occurred in the optional validation/prediction stage:")
    traceback.print_exc()
