[![ArXiv](https://img.shields.io/badge/ArXiv-2512.16841-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.16841)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devmuniz-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devmuniz)
[![GitHub Profile](https://img.shields.io/badge/GitHub-devMuniz02-181717?logo=github&logoColor=white)](https://github.com/devMuniz02)
[![Portfolio](https://img.shields.io/badge/Portfolio-devmuniz02.github.io-0F172A?logo=googlechrome&logoColor=white)](https://devmuniz02.github.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-manu02-FFD21E?logoColor=black)](https://huggingface.co/manu02)

# Tomato Detect YOLO

A complete object detection system for identifying tomatoes using YOLOv11, trained on a Roboflow dataset.

This project provides an end-to-end solution for tomato detection using state-of-the-art YOLO technology. It includes scripts for training a custom YOLOv11 model and deploying real-time detection on webcam feeds.

## Overview

An end-to-end computer vision pipeline designed to detect, count, and track tomatoes in real-time. This project uses the YOLO (You Only Look Once) architecture to achieve high-precision detection in complex agricultural environments.

## Repository Structure

| Path | Description |
| --- | --- |
| `assets/` | Images, figures, or other supporting media used by the project. |
| `data.yaml` | Top-level file included in the repository. |
| `detect.py` | Top-level file included in the repository. |
| `README.md` | Primary project documentation. |
| `train.py` | Top-level file included in the repository. |

## Getting Started

1. Clone the repository.

   ```bash
   git clone https://github.com/devMuniz02/Tomato-Detect-YOLO.git
   cd Tomato-Detect-YOLO
   ```

2. Prepare the local environment.

Review the repository files below to identify the appropriate local setup steps for this project.

3. Run or inspect the project entry point.

Use the project-specific scripts or notebooks in the repository root to run the workflow.

## Features

- **Dataset Management**: Automated download and preprocessing of Roboflow datasets
- **Model Training**: Full YOLOv11 training pipeline with configurable parameters
- **Real-time Detection**: Live webcam detection with FPS counter
- **Automatic Validation**: Built-in validation and sample predictions
- **Batch Predictions**: Process entire validation and test sets
- **GPU Support**: Automatic CUDA detection for accelerated training and inference

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional but recommended)
- pip package manager

### Dependencies

Install required packages:

```bash
pip install ultralytics opencv-python torch roboflow pyyaml
```

For GPU support with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Training a Model

Train a new YOLOv11 model on the tomato detection dataset:

```bash
python train.py
```

**What it does:**
1. Downloads the dataset from Roboflow
2. Validates and corrects the `data.yaml` file
3. Trains YOLOv11 model for 100 epochs (default)
4. Validates the best model
5. Generates sample predictions
6. Saves the best model to `models/best_yolo11.pt`

**Custom configuration:**

```bash
YOLO_MODEL=yolo11m.pt YOLO_EPOCHS=150 YOLO_BATCH=32 python train.py
```

### 2. Real-time Detection

Run real-time tomato detection on your webcam:

```bash
python detect.py
```

**Controls:**
- Press `q` to exit the application
- The FPS counter is displayed in the top-left corner
- Detection boxes and confidence scores are shown for each detection

**Custom camera:**

```bash
YOLO_CAM_INDEX=1 python detect.py
```

**Adjust confidence threshold:**

```bash
YOLO_CONF=0.5 python detect.py
```

## Advanced Usage

### Custom Dataset

To use your own Roboflow dataset:

1. Create a project on [Roboflow Universe](https://universe.roboflow.com/)
2. Export in YOLOv11 format
3. Update `data.yaml` with your project credentials
4. Run `train.py`

### Transfer Learning

Start from a different pretrained model:

```bash
YOLO_MODEL=yolo11l.pt python train.py
```

### Continued Training

Resume training from a checkpoint:

```python
# In train.py, modify the model loading:
model = YOLO('runs-yolo11/exp/weights/last.pt')
```

## Troubleshooting

### "Model not found" error in detect.py

**Solution**: Make sure you've run `train.py` first or place an existing model at the path specified by `YOLO_BEST_MODEL`.

### Camera not opening

**Solution**: 
- Check camera permissions
- Try different `YOLO_CAM_INDEX` values (0, 1, 2, etc.)
- Verify the camera isn't already in use by another application

### Out of Memory (OOM) error during training

**Solution**:
- Reduce batch size: `YOLO_BATCH=8`
- Use a smaller model: `YOLO_MODEL=yolo11n.pt`
- Reduce image size: `YOLO_IMGSZ=416`

### CUDA not detected

**Solution**:
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA-compatible PyTorch:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

## License

This project uses the Roboflow dataset under the CC BY 4.0 license.

## Overview

This project provides an end-to-end solution for tomato detection using state-of-the-art YOLO technology. It includes scripts for training a custom YOLOv11 model and deploying real-time detection on webcam feeds.

## Project Structure

```
Tomato-Detect-YOLO/
├── README.md           # This file
├── data.yaml           # Dataset configuration (Roboflow format)
├── train.py            # Training script
├── detect.py           # Real-time detection script
└── models/             # Models directory (created during training)
    └── best_yolo11.pt  # Best trained model
```
