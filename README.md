# Tomato Detect YOLO

A complete object detection system for identifying tomatoes using YOLOv11, trained on a Roboflow dataset.

## Overview

This project provides an end-to-end solution for tomato detection using state-of-the-art YOLO technology. It includes scripts for training a custom YOLOv11 model and deploying real-time detection on webcam feeds.

## Features

- **Dataset Management**: Automated download and preprocessing of Roboflow datasets
- **Model Training**: Full YOLOv11 training pipeline with configurable parameters
- **Real-time Detection**: Live webcam detection with FPS counter
- **Automatic Validation**: Built-in validation and sample predictions
- **Batch Predictions**: Process entire validation and test sets
- **GPU Support**: Automatic CUDA detection for accelerated training and inference

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

## Configuration

### Dataset Setup

The `data.yaml` file contains the dataset configuration:

```yaml
train: train/images
val: valid/images
test: test/images
nc: 1
names:
- tomato
roboflow:
  workspace: yolo-3b0a6
  project: my-first-project-4mpbz
  version: 2
```

Modify the Roboflow credentials to use your own dataset:
- `workspace`: Your Roboflow workspace name
- `project`: Your Roboflow project name
- `version`: Dataset version number

### Environment Variables

Configure behavior through environment variables:

#### Training (`train.py`)

```bash
# Roboflow API
ROBOFLOW_API_KEY=your_api_key

# Model configuration
YOLO_MODEL=yolo11n.pt           # Model size (n, s, m, l, x)
YOLO_EPOCHS=100                 # Number of training epochs
YOLO_IMGSZ=640                  # Input image size
YOLO_BATCH=16                   # Batch size
YOLO_WORKERS=4                  # Data loading workers

# Output paths
YOLO_PROJECT=runs-yolo11        # Results directory
YOLO_NAME=exp                   # Experiment name
YOLO_FINAL_WEIGHTS=models/best_yolo11.pt  # Final model path
```

#### Detection (`detect.py`)

```bash
# Model path
YOLO_BEST_MODEL=models/best_yolo11.pt

# Camera and inference
YOLO_CAM_INDEX=0                # Camera index (0 for default)
YOLO_IMG_SIZE=640               # Model input size
YOLO_CONF=0.25                  # Confidence threshold
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

## Output Structure

After training, the project generates the following structure:

```
runs-yolo11/
└── exp/
    ├── weights/
    │   ├── best.pt              # Best model
    │   └── last.pt              # Last checkpoint
    ├── results.csv              # Training metrics
    ├── confusion_matrix.png     # Confusion matrix
    └── ...                      # Other validation plots

models/
└── best_yolo11.pt              # Copy of best model for inference
```

## Model Sizes

Choose the appropriate model size for your use case:

| Model | Parameters | Speed (GPU) | Speed (CPU) | Use Case |
|-------|-----------|-----------|-----------|----------|
| nano (n) | 2.6M | 0.99ms | 40ms | Fast inference, limited accuracy |
| small (s) | 11.2M | 1.6ms | 98ms | Balanced |
| medium (m) | 25.9M | 2.8ms | 268ms | Better accuracy |
| large (l) | 43.7M | 4.3ms | 600ms | High accuracy |
| xlarge (x) | 68.2M | 5.5ms | 1200ms | Maximum accuracy |

## Tips for Better Performance

1. **Data Quality**: Ensure high-quality, well-labeled training images
2. **Model Size**: Start with nano/small for quick iterations, use larger models for production
3. **Batch Size**: Increase batch size (8, 16, 32, 64) if GPU memory allows for better convergence
4. **Augmentation**: YOLOv11 applies automatic augmentation; consider custom augmentation for edge cases
5. **Confidence Threshold**: Adjust `CONF_THRES` in detect.py based on precision/recall tradeoff
6. **GPU Acceleration**: Use CUDA-enabled GPU for 5-10x faster training

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

## API Reference

### train.py

Main training script that orchestrates the entire training pipeline.

**Key Functions:**
- Dataset download and validation
- YAML configuration correction
- Model training with YOLOv11
- Automatic validation and prediction generation

**Configuration:**
Environment variables control all parameters (see Environment Variables section)

### detect.py

Real-time detection script for webcam processing.

**Key Functions:**
- Model loading and device selection
- Camera initialization
- Real-time frame processing
- FPS calculation and display
- Annotation with bounding boxes

**Keyboard Controls:**
- `q`: Exit application

## Performance Metrics

After training, the model generates detailed metrics:
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive instances
- **mAP50**: Mean Average Precision at 0.5 IoU
- **mAP50-95**: Mean Average Precision across IoU thresholds
- **Confusion Matrix**: Detailed prediction breakdown

Check the `results.csv` file in the results directory for complete metrics.

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

## References

- [YOLOv11 Documentation](https://docs.ultralytics.com/models/yolov11/)
- [Roboflow Documentation](https://docs.roboflow.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)

## License

This project uses the Roboflow dataset under the CC BY 4.0 license.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Author

Tomato Detect YOLO - A YOLOv11-based tomato detection system

## Disclaimer

This project is for educational and research purposes. Ensure you have the proper permissions and follow all legal requirements when deploying this system in production environments.

---

**Last Updated**: December 2025

For questions or support, refer to the official documentation or create an issue in the repository.
