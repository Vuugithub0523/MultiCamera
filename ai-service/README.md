# AI Service

AI service cho hệ thống Multi-Camera Person Tracking.

## Cấu trúc

```
ai-service/
├── core/                    # Core AI modules
│   ├── object_detection.py  # YOLO-based person detection
│   ├── feature_extraction.py # OSNet feature extraction
│   └── person_lifecycle_manager.py # Lifecycle management
├── utils/                   # Utility modules
│   ├── rtsp_loader.py      # RTSP stream loader
│   └── helpers.py          # Helper functions
└── requirements.txt         # Python dependencies
```

## Dependencies

```bash
pip install -r requirements.txt
```

## Modules

### Object Detection
- YOLO v4-tiny ONNX model
- Person class detection
- Bounding box prediction

### Feature Extraction
- OSNet ONNX model
- Re-identification features
- Cosine similarity matching

### Person Lifecycle Manager
- State management (DETECTED, TRACKING, LOST, CONFIRMED_LOST, ARCHIVED)
- Multi-camera tracking
- Feature gallery management

## Usage

```python
from ai_service.core import ObjectDetection, FeatureExtraction

# Initialize
detector = ObjectDetection(
    onnx_path="./models/yolov4-tiny.onnx",
    device="cuda"
)

feature_extractor = FeatureExtraction(
    onnx_path="./models/osnet_ain_x1_0_M.onnx",
    device="cuda"
)

# Detect persons
detections = detector.predict_img(frame)

# Extract features
features = feature_extractor.predict_img(cropped_person)
```

## Model Requirements

Download và đặt models vào thư mục `../models/pretrained_models/`:
- `yolov4-tiny.onnx`
- `osnet_ain_x1_0_M.onnx`
- `coco.names`
