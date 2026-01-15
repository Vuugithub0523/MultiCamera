# Multi-Camera People Tracking System

A real-time multi-camera people tracking system that can identify and track individuals across multiple camera feeds simultaneously using deep learning-based object detection and feature extraction.

## 🎯 Features

- **Multi-camera support**: Track people across multiple camera feeds
- **Cross-camera re-identification**: Automatically identify the same person across different cameras
- **Real-time processing**: CUDA-accelerated inference for real-time performance
- **Persistent tracking**: Maintains person IDs even when they move between cameras
- **Visual output**: Display all camera feeds with bounding boxes and person IDs
- **Video recording**: Save tracked video output

## 📋 Prerequisites

- Ubuntu 20.04/22.04/24.04
- Python 3.10
- NVIDIA GPU with CUDA support
- CUDA 12.5 installed

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Vuugithub0523/MultiCamera.git
cd MultiCamera
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure CUDA Libraries
```bash
# Add CUDA libraries to system path
echo "/usr/local/cuda-12.5/targets/x86_64-linux/lib" | sudo tee /etc/ld.so.conf.d/cuda-12-5.conf
sudo ldconfig

# Verify CUDA is available
nvidia-smi
```

### 4. Prepare Your Data

#### Download Pre-trained Models

Place your ONNX models in the `models/` directory:
- Object detection model (e.g., YOLOv8, Faster R-CNN)
- Feature extraction model (e.g., ResNet, OSNet for person re-identification)
```bash
mkdir -p models
# Download or place your .onnx models here
```

#### Prepare Video Files

Place your camera video files in a directory (e.g., `videos/`):
```bash
mkdir -p videos
# Copy your video files (mp4, avi, etc.) into this directory
# Example: videos/camera1.mp4, videos/camera2.mp4, ...
```

### 5. Configure the System

Edit `config.yaml` to match your setup:
```yaml
# Video input
video_path: "./videos"  # Directory containing camera videos

# Model paths
object_detection_model_path: "./models/yolov8n.onnx"
feature_extraction_model_path: "./models/osnet_x1_0.onnx"
object_detection_classes_path: "./models/coco.names"

# Device configuration
inference_model_device: "cuda"  # Use "cpu" if no GPU available

# Detection settings
object_detection_threshold: 0.5
feature_extraction_threshold: 0.6
max_gallery_set_each_person: 10

# Display settings
size_each_camera_image: [640, 480]  # Width, Height
resize_all_camera_image: 0.8
display_video_camera_tracking: true

# Output settings
save_video_camera_tracking: true
output_path_name_save_video_camera_tracking: "./output"
output_name_save_video_camera_tracking: "tracked_output"
fps_save_video_camera_tracking: 30
```

### 6. Run the Tracking System
```bash
python main.py -s config.yaml
```

**Controls:**
- Press `q` to stop the program and save the output

## 📁 Project Structure
```
MultiCamera/
├── main.py                    # Main application entry point
├── object_detection.py        # Object detection module (YOLO/etc.)
├── feature_extraction.py      # Person re-identification feature extraction
├── helpers.py                 # Utility functions
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── models/                   # Pre-trained ONNX models
│   ├── yolov8n.onnx
│   ├── osnet_x1_0.onnx
│   └── coco.names
├── videos/                   # Input video files
│   ├── camera1.mp4
│   ├── camera2.mp4
│   └── ...
└── output/                   # Output tracked videos
```

## ⚙️ Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `video_path` | Directory containing input videos | `"./videos"` |
| `object_detection_threshold` | Detection confidence threshold (0-1) | `0.5` |
| `feature_extraction_threshold` | Person matching threshold (0-1) | `0.6` |
| `max_gallery_set_each_person` | Max feature embeddings per person | `10` |
| `inference_model_device` | Device for inference: "cuda" or "cpu" | `"cuda"` |
| `size_each_camera_image` | Display size [width, height] | `[640, 480]` |
| `display_video_camera_tracking` | Show live tracking window | `true` |
| `save_video_camera_tracking` | Save output video | `true` |

## 🔧 Troubleshooting

### CUDA Not Available

If you see CPU fallback warnings:
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Reinstall CUDA libraries if needed
sudo apt install cuda-toolkit-12-5
```

### Low FPS / Slow Performance

1. Reduce camera resolution in `config.yaml`:
```yaml
   size_each_camera_image: [320, 240]  # Lower resolution
```

2. Increase detection threshold to detect fewer objects:
```yaml
   object_detection_threshold: 0.7
```

3. Ensure you're using CUDA:
```yaml
   inference_model_device: "cuda"
```

### Out of Memory Error

Reduce the number of cameras or lower the resolution:
```yaml
size_each_camera_image: [320, 240]
max_gallery_set_each_person: 5
```

## 📊 Expected Performance

With CUDA acceleration on an NVIDIA RTX GPU:
- 2 cameras: 25-30 FPS
- 4 cameras: 15-20 FPS
- 6 cameras: 10-15 FPS

Performance varies based on:
- GPU model
- Video resolution
- Number of people in frame
- Model complexity

## 🛠️ Advanced Usage

### Using Custom Models

Replace the ONNX models in `models/` directory with your own trained models. Ensure they have compatible input/output shapes.

### Adding More Cameras

Simply add more video files to the `video_path` directory. The system automatically detects all videos.

### Adjusting Tracking Sensitivity

Lower `feature_extraction_threshold` for more aggressive matching (may cause false positives):
```yaml
feature_extraction_threshold: 0.4
```

Higher threshold for stricter matching (may miss some matches):
```yaml
feature_extraction_threshold: 0.8
```

## 📝 Dependencies

Key dependencies:
- `onnxruntime-gpu==1.23.2` - GPU-accelerated inference
- `opencv-python==4.5.5.64` - Video processing
- `torch==1.11.0` - Deep learning framework
- `numpy==1.23.5` - Numerical operations
- `scipy==1.9.3` - Distance calculations

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- YOLO for object detection
- OSNet for person re-identification
- ONNX Runtime for optimized inference

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: Make sure you have the appropriate licenses for any pre-trained models you use.
