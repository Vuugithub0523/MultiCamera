# Multi-Camera Person Tracking System

Hệ thống theo dõi người qua nhiều camera sử dụng AI, được tổ chức thành 3 phần riêng biệt:

## 📁 Cấu trúc Project

```
MultiCamera/
├── ai-service/              # AI Service - Object Detection & Feature Extraction
│   ├── core/
│   │   ├── object_detection.py
│   │   ├── feature_extraction.py
│   │   └── person_lifecycle_manager.py
│   ├── utils/
│   │   ├── rtsp_loader.py
│   │   └── helpers.py
│   └── requirements.txt
│
├── backend-server/          # Backend Server - FastAPI WebSocket Server
│   ├── api/
│   │   └── stream_manager.py
│   ├── server.py
│   └── requirements.txt
│
├── frontend/                # Frontend - Next.js Dashboard
│   ├── app/
│   ├── components/
│   └── package.json
│
├── models/                  # AI Models
│   └── pretrained_models/
│
├── storage/                 # Data Storage
├── tracking_logs/           # Tracking Logs
└── config.yaml              # Configuration File
```

## 🚀 Quick Start

### 1. AI Service

Cài đặt dependencies:
```bash
cd ai-service
pip install -r requirements.txt
```

### 2. Backend Server

Cài đặt dependencies:
```bash
cd backend-server
pip install -r requirements.txt
```

Chạy server:
```bash
python server.py
```

Server sẽ chạy tại: `http://localhost:8080`

### 3. Frontend

Cài đặt dependencies:
```bash
cd frontend
npm install
```

Chạy development server:
```bash
npm run dev
```

Frontend sẽ chạy tại: `http://localhost:3000`

## 📋 API Endpoints

### Backend Server

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /api/cameras` - Danh sách cameras
- `GET /api/tracking/stats` - Thống kê tracking
- `WS /ws/{stream_type}/{camera_id}` - WebSocket stream
  - `stream_type`: `raw` hoặc `tracking`
  - `camera_id`: `cam01`, `cam02`, `cam03`, ...

## ⚙️ Configuration

Chỉnh sửa file `config.yaml` ở thư mục gốc:

```yaml
# RTSP URLs
rtsp_urls:
  - "rtsp://..."
  - "rtsp://..."

# AI Models
object_detection_model_path: "./models/pretrained_models/yolov4-tiny.onnx"
feature_extraction_model_path: "./models/pretrained_models/osnet_ain_x1_0_M.onnx"

# Tracking Settings
tracking_stream_enabled: true
tracking_stream_interval: 1
object_detection_threshold: 0.7
feature_extraction_threshold: 0.42
inference_model_device: "cuda"  # or "cpu"

# Lifecycle Management
max_lost_frames: 30
max_confirm_lost_frames: 90
archive_after_seconds: 3000
time_window_seconds: 3.0
```

## 🏗️ Architecture

### AI Service
- **Object Detection**: YOLO-based person detection
- **Feature Extraction**: OSNet feature extraction for Re-ID
- **Lifecycle Management**: Person tracking state management
- **RTSP Loader**: Multi-camera stream handling

### Backend Server
- **FastAPI**: High-performance async web framework
- **WebSocket**: Real-time video streaming
- **Stream Manager**: Orchestrates AI pipeline
- **API**: RESTful endpoints for frontend

### Frontend
- **Next.js**: React framework with SSR
- **WebSocket Client**: Real-time video display
- **Dashboard**: Multi-camera view and statistics

## 🔧 Development

### Chạy từng service riêng:

**Backend Server:**
```bash
cd backend-server
python server.py
```

**Frontend:**
```bash
cd frontend
npm run dev
```

### Testing

**Test backend:**
```bash
curl http://localhost:8080/health
```

**Test WebSocket:**
```bash
# Sử dụng browser hoặc WebSocket client
ws://localhost:8080/ws/tracking/cam01
```

## 📝 Notes

- Đảm bảo CUDA drivers được cài đặt nếu sử dụng GPU
- RTSP URLs phải valid và accessible
- Models phải được download vào thư mục `models/pretrained_models/`
- Frontend connect tới backend qua WebSocket

## 🐛 Troubleshooting

**Backend không start:**
- Kiểm tra RTSP URLs trong config.yaml
- Kiểm tra model paths

**Frontend không kết nối:**
- Đảm bảo backend đang chạy
- Kiểm tra WebSocket URL trong frontend code

**GPU không hoạt động:**
```bash
pip install onnxruntime-gpu
# Hoặc nếu dùng CPU:
pip install onnxruntime
```

## 📄 License

[Your License Here]
