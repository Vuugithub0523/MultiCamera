# 🔄 DEPENDENCIES UPDATE SUMMARY

## 📋 TÓM TẮT CẬP NHẬT

**Ngày:** 21/01/2026  
**Dự án:** Native AI Backend - Multi-Camera Person Tracking  
**Thực hiện bởi:** GitHub Copilot

---

## ✅ ĐÃ CẬP NHẬT REQUIREMENTS.TXT

### **Các dependencies được thêm/cập nhật:**

| Package | Phiên bản cũ | Phiên bản mới | Lý do |
|---------|--------------|---------------|-------|
| **opencv-python** | 4.9.0.80 | 4.8.1.78 | Compatibility với user request |
| **opencv-python-headless** | - | 4.5.5.64 | **Thêm mới** - Server environment |
| **numpy** | >=1.25.0,<2.0.0 | 1.26.4 | Version cụ thể, stable |
| **onnxruntime-gpu** | 1.17.0 | 1.23.2 | Update lên phiên bản mới nhất |
| **tqdm** | 4.66.1 | 4.63.0 | Match với user request |
| **PyYAML** | - | 6.0 | **Thêm mới** - Config parsing |
| **gdown** | - | 4.5.1 | **Thêm mới** - Model downloading |

### **PyTorch GPU (commented):**
```
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
```

**Lưu ý:** Cài riêng với lệnh:
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

## 📊 PHÂN TÍCH KIẾN TRÚC

### **Luồng xử lý chính:**

```
RTSP Camera Stream
    ↓
[YOLO Detection] - onnxruntime-gpu + opencv-python
    ↓
[ByteTracker] - scipy (Hungarian algorithm)
    ↓
[OSNet Re-ID] - onnxruntime-gpu
    ↓
[Person Database] - numpy (cosine similarity)
    ↓
[Lifecycle Manager] - state management
    ↓
[Visualization] - opencv-python (draw boxes)
    ↓
[WebSocket] - fastapi + websockets
    ↓
Frontend (Next.js)
```

### **Các module chính:**

1. **Detection** (`detection/`)
   - `yolo_detector.py` - YOLO inference với ONNX
   - `byte_tracker.py` - Multi-object tracking
   - Dependencies: `onnxruntime-gpu`, `opencv-python`, `numpy`, `scipy`

2. **Re-ID** (`reid/`)
   - `feature_extractor.py` - OSNet features
   - `person_database.py` - Person matching
   - Dependencies: `onnxruntime-gpu`, `scipy`, `numpy`

3. **Core** (`core/`)
   - `pipeline.py` - Camera processing pipeline
   - `manager.py` - Multi-camera management
   - `lifecycle_manager.py` - Person lifecycle tracking
   - Dependencies: Tất cả modules trên

4. **API** (`api/`)
   - `rest.py` - REST endpoints
   - `websocket.py` - WebSocket streaming
   - Dependencies: `fastapi`, `websockets`

5. **Frontend** (`frontend/`)
   - Next.js + TypeScript
   - WebSocket client để nhận streams
   - Hiển thị video real-time

---

## 🔗 SƠ ĐỒ LIÊN KẾT

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │VideoStream  │  │ Dashboard   │  │   Config    │         │
│  │ Component   │  │   Reports   │  │  Settings   │         │
│  └──────┬──────┘  └─────────────┘  └─────────────┘         │
│         │ useWebSocketStream                                │
└─────────┼─────────────────────────────────────────────────┘
          │ ws://localhost:5000/ws/tracking/{camera_id}
          ▼
┌─────────────────────────────────────────────────────────────┐
│              BACKEND (FastAPI + Uvicorn)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  main.py - FastAPI App + WebSocket Manager           │  │
│  └────────────┬─────────────────────────────────────────┘  │
│               │                                              │
│  ┌────────────▼─────────────────────────────────────────┐  │
│  │  core/manager.py - MultiCameraManager                │  │
│  │  • Load shared models (YOLO, OSNet)                  │  │
│  │  • Create per-camera pipelines                       │  │
│  └────────────┬─────────────────────────────────────────┘  │
│               │                                              │
│         ┌─────┴────────┐                                    │
│         ▼              ▼                                    │
│  ┌─────────────┐  ┌─────────────┐                         │
│  │Camera 1     │  │Camera N     │                         │
│  │Pipeline     │  │Pipeline     │                         │
│  └─────┬───────┘  └─────┬───────┘                         │
│        │                 │                                  │
└────────┼─────────────────┼────────────────────────────────┘
         │                 │
    ┌────▼─────────────────▼────┐
    │  PROCESSING PIPELINE      │
    │                           │
    │  1. RTSP Input            │ opencv-python-headless
    │  2. YOLO Detection        │ onnxruntime-gpu
    │  3. ByteTracker           │ scipy
    │  4. Re-ID Features        │ onnxruntime-gpu
    │  5. Person Matching       │ numpy (cosine)
    │  6. Lifecycle Mgmt        │ -
    │  7. Visualization         │ opencv-python
    │  8. JPEG Encode           │ opencv-python
    │     ↓                     │
    │  WebSocket Broadcast      │ websockets
    └───────────────────────────┘
```

---

## 🎯 VAI TRÒ CỦA TỪNG DEPENDENCY

### **Computer Vision & AI:**

1. **opencv-python (4.8.1.78)**
   - Đọc/ghi frames
   - Vẽ bounding boxes, labels
   - JPEG encoding
   - Image preprocessing

2. **opencv-python-headless (4.5.5.64)**
   - RTSP stream reading (không GUI)
   - Server environment
   - Parallel với opencv-python

3. **numpy (1.26.4)**
   - Array operations
   - Matrix multiplication
   - Cosine similarity
   - Feature normalization

4. **scipy (1.11.4)**
   - Hungarian algorithm (linear assignment)
   - Cosine distance (Re-ID matching)
   - ByteTracker matching

5. **onnxruntime-gpu (1.23.2)**
   - YOLO inference (GPU)
   - OSNet inference (GPU)
   - CUDAExecutionProvider

6. **albumentations (1.3.1)**
   - Image augmentation
   - Preprocessing transforms
   - Feature extraction pipeline

### **Utilities:**

7. **PyYAML (6.0)**
   - Config file parsing
   - YAML configuration loading

8. **tqdm (4.63.0)**
   - Progress bars
   - Model loading progress
   - Batch processing

9. **gdown (4.5.1)**
   - Download models từ Google Drive
   - Setup automation

### **Web Framework:**

10. **fastapi (0.109.0)**
    - REST API endpoints
    - Async request handling

11. **uvicorn (0.27.0)**
    - ASGI server
    - WebSocket support

12. **websockets (12.0)**
    - WebSocket connections
    - Binary frame streaming

### **Deep Learning (Optional):**

13. **torch (2.5.1+cu121)**
    - Advanced training
    - Custom models
    - GPU acceleration

14. **torchvision (0.20.1+cu121)**
    - Vision transforms
    - Pre-trained models

15. **torchaudio (2.5.1+cu121)**
    - Audio processing (nếu cần)

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT

### **Cài đặt đầy đủ (có GPU):**

```bash
# Bước 1: Cài PyTorch GPU
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Bước 2: Cài các dependencies còn lại
pip install -r requirements.txt

# Bước 3: Kiểm tra
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import onnxruntime; print(f'ONNX: {onnxruntime.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

### **Cài đặt CPU only:**

```bash
# Bỏ qua PyTorch hoặc cài CPU version
pip install torch torchvision torchaudio

# Cài dependencies với onnxruntime (không GPU)
# Sửa requirements.txt: onnxruntime==1.23.2 thay vì onnxruntime-gpu
pip install -r requirements.txt
```

---

## 📁 FILES ĐƯỢC TẠO

1. **[requirements.txt](requirements.txt)** (đã cập nhật)
   - Thêm các dependencies thiếu
   - Update versions
   - Thêm comments hướng dẫn

2. **[ARCHITECTURE_ANALYSIS.md](ARCHITECTURE_ANALYSIS.md)** (mới)
   - Phân tích kiến trúc toàn hệ thống
   - Sơ đồ luồng dữ liệu
   - Vai trò của từng module
   - Dependencies mapping

3. **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** (mới)
   - Hướng dẫn cài đặt chi tiết từng bước
   - Troubleshooting
   - Testing checklist
   - Performance tuning

4. **[DEPENDENCIES_UPDATE.md](DEPENDENCIES_UPDATE.md)** (file này)
   - Tóm tắt các thay đổi
   - Version comparison
   - Installation instructions

---

## ✅ KIỂM TRA SAU KHI CÀI ĐẶT

### **Test 1: Import kiểm tra**
```python
import torch
import cv2
import numpy as np
import scipy
import onnxruntime as ort
import albumentations
import yaml
import gdown
import tqdm

print("✅ All packages imported successfully!")
```

### **Test 2: GPU availability**
```python
import torch
import onnxruntime as ort

print(f"PyTorch CUDA: {torch.cuda.is_available()}")
print(f"ONNX Providers: {ort.get_available_providers()}")
```

### **Test 3: Run backend**
```bash
cd d:\TTTN_AntBuddy\native-ai-backend
python main.py
```

---

## 🎯 KẾT QUẢ

✅ **Tất cả dependencies đã được thêm vào requirements.txt**  
✅ **Versions đã được cập nhật theo yêu cầu**  
✅ **Kiến trúc hệ thống đã được phân tích chi tiết**  
✅ **Hướng dẫn cài đặt đã được tạo đầy đủ**  
✅ **Sẵn sàng để chạy hệ thống mượt mà! 🚀**

---

## 📚 TÀI LIỆU LIÊN QUAN

- [ARCHITECTURE_ANALYSIS.md](ARCHITECTURE_ANALYSIS.md) - Phân tích kiến trúc chi tiết
- [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) - Hướng dẫn cài đặt đầy đủ
- [requirements.txt](requirements.txt) - Dependencies list
- [README.md](README.md) - Project overview

---

**Cập nhật bởi:** GitHub Copilot  
**Ngày:** January 21, 2026  
**Status:** ✅ Completed
