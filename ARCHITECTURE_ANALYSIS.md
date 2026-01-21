# 📊 PHÂN TÍCH KIẾN TRÚC HỆ THỐNG - NATIVE AI BACKEND

## 🏗️ TỔNG QUAN KIẾN TRÚC

Hệ thống Multi-Camera Person Tracking là một ứng dụng AI real-time theo dõi người qua nhiều camera với khả năng Re-Identification (Re-ID) và quản lý vòng đời (lifecycle).

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Next.js)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Video Stream │  │ Reports      │  │ Config       │          │
│  │ Component    │  │ Dashboard    │  │ Settings     │          │
│  └──────┬───────┘  └──────────────┘  └──────────────┘          │
│         │ WebSocket Connection (ws://backend:5000/ws/tracking/) │
└─────────┼─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BACKEND (FastAPI + Python)                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              main.py - FastAPI Application                 │ │
│  │  • REST API Endpoints                                      │ │
│  │  • WebSocket Manager (broadcast frames to clients)        │ │
│  │  • Lifecycle context manager (startup/shutdown)           │ │
│  └────────────────┬───────────────────────────────────────────┘ │
│                   │                                               │
│  ┌────────────────▼───────────────────────────────────────────┐ │
│  │         core/manager.py - MultiCameraManager              │ │
│  │  • Quản lý nhiều camera pipelines                         │ │
│  │  • Load và chia sẻ models (YOLO, OSNet)                   │ │
│  │  • Khởi tạo RTSP loaders, trackers cho mỗi camera         │ │
│  │  • Thread pool cho xử lý song song                        │ │
│  └────────────────┬───────────────────────────────────────────┘ │
│                   │                                               │
│         ┌─────────┴─────────┐                                    │
│         │                   │                                    │
│  ┌──────▼─────┐      ┌─────▼──────┐                            │
│  │  Camera 1  │      │  Camera N  │  (Mỗi camera = 1 pipeline) │
│  │  Pipeline  │ ...  │  Pipeline  │                            │
│  └──────┬─────┘      └─────┬──────┘                            │
│         │                   │                                    │
└─────────┼───────────────────┼──────────────────────────────────┘
          │                   │
          ▼                   ▼
    ┌─────────────────────────────────┐
    │  PROCESSING PIPELINE (mỗi cam)  │
    │                                 │
    │  1. RTSP Stream Input          │
    │  2. YOLO Detection             │
    │  3. ByteTracker Tracking       │
    │  4. Re-ID Feature Extraction   │
    │  5. Person Database Matching   │
    │  6. Lifecycle Management       │
    │  7. Visualization & Output     │
    └─────────────────────────────────┘
```

---

## 📦 CÁC THÀNH PHẦN CHÍNH

### 1️⃣ **BACKEND - FASTAPI SERVER** (`main.py`)

**Chức năng:**
- Entry point của ứng dụng backend
- Quản lý REST API và WebSocket connections
- Khởi động và dừng MultiCameraManager

**Liên kết:**
- `FastAPI` → WebSocket Manager → Frontend (streams JPEG frames)
- `main.py` → `core/manager.py` → Camera Pipelines
- REST API endpoints từ `api/rest.py`

**Flow:**
```python
1. Startup:
   - Khởi tạo MultiCameraManager (load models, cameras)
   - Tạo WebSocketManager
   - Start processing task (asyncio)
   
2. Runtime:
   - WebSocket endpoint nhận connections từ frontend
   - Callback function broadcast frames đến clients
   - REST endpoints trả về thống kê, cấu hình
   
3. Shutdown:
   - Stop all cameras
   - Cancel processing tasks
```

---

### 2️⃣ **DETECTION MODULE** (`detection/`)

#### A. **YOLO Detector** (`yolo_detector.py`)

**Mục đích:** Object detection - phát hiện người trong frame

**Model:** YOLOv4-Tiny (ONNX format)
- Input: 416x416 RGB image
- Output: Bounding boxes [x1, y1, x2, y2, confidence]
- Target class: "person" only

**Dependencies:**
- `onnxruntime-gpu==1.23.2` - Inference engine
- `opencv-python==4.8.1.78` - Image preprocessing
- `numpy==1.26.4` - Array operations

**Processing:**
```python
1. Preprocess: Resize → Normalize → Letterbox padding
2. Inference: ONNX Runtime với CUDA provider
3. Postprocess: NMS (Non-Maximum Suppression) → Filter by confidence
```

#### B. **ByteTracker** (`byte_tracker.py`)

**Mục đích:** Multi-object tracking - theo dõi người qua các frames

**Algorithm:** BYTE (Combining high and low confidence detections)
- Kalman Filter cho motion prediction
- IoU matching + appearance matching
- Track lifecycle: Tracked → Lost → Removed

**Dependencies:**
- `scipy==1.11.4` - Linear assignment (Hungarian algorithm)
- `numpy==1.26.4` - Matrix operations

**Liên kết với Detection:**
```python
detections = detector.detect(frame)  # YOLO output
tracks = tracker.update(detections)   # ByteTracker tracking
```

---

### 3️⃣ **RE-IDENTIFICATION (Re-ID)** (`reid/`)

#### A. **Feature Extractor** (`feature_extractor.py`)

**Mục đích:** Trích xuất đặc trưng người để nhận diện xuyên camera

**Model:** OSNet (Omni-Scale Network) - ONNX format
- Input: 256x128 RGB crop của người
- Output: 512-dim feature vector (L2 normalized)

**Dependencies:**
- `onnxruntime-gpu==1.23.2` - Model inference
- `scipy==1.11.4` - Cosine distance calculation
- `opencv-python==4.8.1.78` - Image preprocessing

**Processing:**
```python
1. Crop person bbox từ frame
2. Resize to 256x128
3. Normalize (ImageNet mean/std)
4. Inference → 512-dim vector
5. L2 normalization
```

#### B. **Person Database** (`person_database.py`)

**Mục đích:** Lưu trữ và so sánh features để match người

**Structure:**
```python
{
  "person_1": {
    "features": [512-dim vectors...],  # Gallery of features
    "metadata": {...}
  }
}
```

**Matching:**
- Cosine distance < threshold (0.42) → Same person
- Time window constraint (3 seconds)

---

### 4️⃣ **CORE PIPELINE** (`core/`)

#### A. **Camera Pipeline** (`pipeline.py`)

**Trái tim của hệ thống** - Xử lý mỗi frame qua luồng AI

**Flow xử lý:**
```
Frame Input (RTSP)
    ↓
[Skip frames check]
    ↓
YOLO Detection → Bounding boxes
    ↓
ByteTracker → Track IDs
    ↓
Feature Extraction → 512-dim vectors
    ↓
Person Database → Match to global person IDs
    ↓
Lifecycle Manager → Update person state
    ↓
Visualization → Draw boxes, labels
    ↓
JPEG Encoding → WebSocket broadcast
```

**Dependencies tích hợp:**
- Detection module (YOLO + ByteTracker)
- Re-ID module (FeatureExtractor + PersonDB)
- Lifecycle Manager

#### B. **Multi-Camera Manager** (`manager.py`)

**Chức năng:**
- Quản lý N camera pipelines song song
- Load shared models (YOLO, OSNet) - dùng chung cho tất cả cameras
- Per-camera instances: Tracker, RTSP Loader
- Thread pool cho CPU-bound operations

**Optimization:**
- Shared GPU models → Tiết kiệm VRAM
- Async processing → Non-blocking
- Buffer size = 1 → Minimize latency

#### C. **Lifecycle Manager** (`lifecycle_manager.py`)

**Mục đích:** Quản lý vòng đời người (ACTIVE → LOST → ARCHIVED)

**States:**
- `ACTIVE`: Đang được track
- `LOST`: Mất track nhưng chưa lâu
- `CONFIRMED_LOST`: Mất lâu (>90 frames)
- `ARCHIVED`: Lưu vào logs

**Export:**
- `tracking_logs/tracking_summary.csv` - Lịch sử theo dõi

---

### 5️⃣ **FRONTEND - NEXT.JS** (`frontend/`)

#### A. **WebSocket Hook** (`hooks/useWebSocketStream.ts`)

**Chức năng:**
- Kết nối WebSocket đến backend
- Nhận binary JPEG frames
- Convert to Blob URL để hiển thị
- Tính FPS của stream

**Connection:**
```typescript
ws://localhost:5000/ws/tracking/{cameraId}
```

#### B. **Video Stream Component** (`components/VideoStream.tsx`)

**Chức năng:**
- Hiển thị video stream real-time
- Status indicator (FPS, connection)
- Error handling

**Liên kết với Backend:**
```
Frontend Component
    ↓
useWebSocketStream hook
    ↓
WebSocket connection
    ↓
Backend WebSocket Manager
    ↓
Camera Pipeline JPEG output
```

---

## 🔗 LUỒNG DỮ LIỆU TỔNG THỂ

### **Real-time Tracking Flow:**

```
1. RTSP Camera (IP Camera)
   ├─ Stream video → RTSP Loader
   └─ Buffer size=1 (low latency)

2. Frame Processing (Camera Pipeline)
   ├─ Every N frames → YOLO Detection
   │  └─ GPU inference (CUDA/onnxruntime-gpu)
   ├─ Every frame → ByteTracker
   │  └─ Kalman filter prediction + matching
   ├─ New track → Feature Extraction
   │  └─ OSNet ONNX inference (GPU)
   ├─ Feature → Person Database
   │  └─ Cosine similarity matching
   └─ Person state → Lifecycle Manager
      └─ Update ACTIVE/LOST/ARCHIVED

3. Output
   ├─ Draw visualizations (OpenCV)
   ├─ JPEG encoding (quality=80)
   └─ WebSocket broadcast → Frontend

4. Frontend Display
   ├─ Receive JPEG via WebSocket
   ├─ Convert to Blob URL
   └─ Update <img> src
```

---

## 📦 DEPENDENCIES ANALYSIS

### **Core AI/CV Libraries:**

| Package | Version | Mục đích | Sử dụng trong |
|---------|---------|----------|--------------|
| **opencv-python** | 4.8.1.78 | Xử lý ảnh, vẽ visualizations | Detector, Pipeline |
| **opencv-python-headless** | 4.5.5.64 | OpenCV không GUI (server) | RTSP Loader |
| **numpy** | 1.26.4 | Array operations, matrix | Tất cả modules |
| **scipy** | 1.11.4 | Linear assignment, distance | ByteTracker, Re-ID |
| **onnxruntime-gpu** | 1.23.2 | GPU inference (YOLO, OSNet) | Detector, Re-ID |
| **albumentations** | 1.3.1 | Image augmentation | Feature extraction preprocessing |
| **PyYAML** | 6.0 | Config file parsing | Config management |
| **tqdm** | 4.63.0 | Progress bars | Model loading, batch processing |
| **gdown** | 4.5.1 | Download models from Google Drive | Setup scripts |

### **PyTorch (Optional):**
| Package | Version | Mục đích |
|---------|---------|----------|
| **torch** | 2.5.1+cu121 | Deep learning framework |
| **torchvision** | 0.20.1+cu121 | Vision utilities |
| **torchaudio** | 2.5.1+cu121 | Audio processing |

**Lưu ý:** PyTorch không bắt buộc cho hệ thống hiện tại (dùng ONNX), nhưng có thể dùng cho:
- Training models mới
- Advanced augmentation
- Custom loss functions

---

## ⚙️ OPTIMIZATION & PERFORMANCE

### **GPU Acceleration:**
1. **ONNX Runtime GPU**
   - CUDAExecutionProvider cho NVIDIA GPU
   - DirectML cho Windows GPU (fallback)
   - Inference tốc độ ~30 FPS/camera

2. **Shared Models**
   - YOLO model load 1 lần → dùng cho tất cả cameras
   - OSNet model load 1 lần → shared feature extractor
   - Tiết kiệm VRAM, tăng throughput

### **Latency Reduction:**
1. **RTSP Buffer = 1**
   - Minimize frame delay
   - Trade-off: có thể drop frames nếu network lag

2. **Detection Skip Frames = 2**
   - Detect every 2 frames → tăng FPS
   - Tracking vẫn chạy mỗi frame

3. **Output FPS = 15**
   - Broadcast 15 FPS đến frontend
   - Giảm bandwidth, vẫn smooth

### **Async Processing:**
- FastAPI async endpoints
- Asyncio for WebSocket broadcast
- Thread pool cho CPU-bound tasks (JPEG encoding)

---

## 🔧 CẤU HÌNH QUAN TRỌNG (`config.py`)

```python
# Detection
DETECTION_CONFIDENCE = 0.5        # Confidence threshold
DETECTION_SKIP_FRAMES = 2         # Detect every N frames

# Tracking (ByteTracker)
TRACK_THRESH = 0.5                # High confidence track
MATCH_THRESH = 0.8                # IoU matching threshold
TRACK_BUFFER = 30                 # Keep lost tracks

# Re-ID
REID_THRESHOLD = 0.42             # Cosine distance
TIME_WINDOW_SECONDS = 3.0         # Match time window

# Lifecycle
MAX_LOST_FRAMES = 30              # 30 frames → LOST
MAX_CONFIRM_LOST_FRAMES = 90      # 90 frames → CONFIRMED_LOST
ARCHIVE_AFTER_SECONDS = 300       # 5 minutes → ARCHIVE

# Output
OUTPUT_FPS = 15                   # WebSocket broadcast FPS
JPEG_QUALITY = 80                 # JPEG compression
```

---

## 📝 KIỂM TRA DEPENDENCIES

### **Đã có trong requirements.txt:**
✅ albumentations==1.3.1
✅ gdown==4.5.1
✅ numpy==1.26.4
✅ onnxruntime-gpu==1.23.2 (updated từ 1.17.0)
✅ opencv-python==4.8.1.78 (updated từ 4.9.0.80)
✅ opencv-python-headless==4.5.5.64 (added)
✅ PyYAML==6.0 (added)
✅ scipy==1.11.4
✅ tqdm==4.63.0 (updated từ 4.66.1)

### **PyTorch (commented, cài riêng):**
✅ torch==2.5.1+cu121
✅ torchvision==0.20.1+cu121
✅ torchaudio==2.5.1+cu121

---

## 🚀 HƯỚNG DẪN SETUP

### **Bước 1: Cài PyTorch GPU (nếu cần)**
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### **Bước 2: Cài dependencies**
```bash
pip install -r requirements.txt
```

### **Bước 3: Download models**
```bash
python download_model.py  # (nếu có script)
```

### **Bước 4: Chạy backend**
```bash
python main.py
# hoặc
uvicorn main:app --host 0.0.0.0 --port 5000
```

### **Bước 5: Chạy frontend**
```bash
cd frontend
npm install
npm run dev
```

---

## 🎯 KẾT LUẬN

Hệ thống **Native AI Backend** là một kiến trúc hoàn chỉnh cho bài toán **Multi-Camera Person Tracking** với:

1. ✅ **Detection** - YOLO phát hiện người
2. ✅ **Tracking** - ByteTracker theo dõi qua frames
3. ✅ **Re-ID** - OSNet matching người xuyên camera
4. ✅ **Lifecycle** - Quản lý vòng đời người
5. ✅ **Real-time** - WebSocket streaming đến frontend

**Tất cả dependencies đã được cập nhật** trong `requirements.txt` để hệ thống chạy mượt mà!

---

**Tác giả:** AntBuddy AI Team  
**Cập nhật:** January 21, 2026  
**Version:** 1.0.0
