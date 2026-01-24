# 🎥 Multi-Camera Person Tracking System

Hệ thống theo dõi và nhận diện người qua nhiều camera sử dụng AI với kiến trúc 3 tầng.

## 🚀 Quick Start

### Chạy Backend
```bash
cd backend-server
python server.py
```
Server: `http://localhost:8080`

### Chạy Frontend  
```bash
cd Front-end
npm run dev
```
Dashboard: `http://localhost:3000`

## 📁 Cấu trúc Project

```
MultiCamera/
├── ai-service/              # AI - Detection & Re-ID
│   ├── core/
│   │   ├── object_detection.py         # YOLO v4-tiny
│   │   ├── feature_extraction.py       # OSNet Re-ID
│   │   └── person_lifecycle_manager.py # Tracking lifecycle
│   ├── utils/
│   │   ├── rtsp_loader.py             # RTSP streams
│   │   └── helpers.py
│   └── requirements.txt
│
├── backend-server/          # FastAPI Server
│   ├── api/
│   │   └── stream_manager.py          # Stream & pipeline
│   ├── server.py                      # FastAPI app
│   └── requirements.txt
│
├── Front-end/               # Next.js Dashboard
│   ├── app/
│   │   ├── page.tsx                   # Dashboard (real-time events)
│   │   ├── report/page.tsx            # Reports & analytics
│   │   ├── video-events/page.tsx      # Event history
│   │   └── configuration/page.tsx     # Settings
│   ├── components/
│   │   ├── VideoStream.tsx            # WebSocket video
│   │   └── dashboard-layout.tsx
│   └── package.json
│
├── models/                  # AI Models
│   └── pretrained_models/
│       ├── yolov4-tiny.onnx
│       └── osnet_ain_x1_0_M.onnx
│
├── storage/                 # Data storage
├── tracking_logs/           # Tracking logs
└── config.yaml              # Main configuration
```

## 🔌 API Endpoints

### REST APIs
```
GET  /                              # Root
GET  /health                        # Health check
GET  /api/cameras                   # List cameras
GET  /api/tracking/stats            # Tracking statistics
GET  /api/tracking/events?limit=50  # Recent events
GET  /api/report/stats              # Report data (hourly, flow)
```

### WebSocket APIs
```
WS /ws/tracking/{camera_id}         # Video stream với bounding boxes
WS /ws/raw/{camera_id}              # Raw video stream
WS /ws/events                       # Real-time tracking events
```

**Camera IDs**: `cam01`, `cam02`, `cam03`

## ⚙️ Configuration (`config.yaml`)

```yaml
# RTSP URLs
rtsp_urls:
  - "rtsp://camera1"
  - "rtsp://camera2"
  - "rtsp://camera3"

# AI Models
object_detection_model_path: "./models/pretrained_models/yolov4-tiny.onnx"
feature_extraction_model_path: "./models/pretrained_models/osnet_ain_x1_0_M.onnx"
classes_path: "./models/pretrained_models/coco.names"

# Tracking
tracking_stream_enabled: true
tracking_stream_interval: 1
object_detection_threshold: 0.7
feature_extraction_threshold: 0.42
inference_model_device: "cuda"  # or "cpu"

# Lifecycle
max_lost_frames: 30
max_confirm_lost_frames: 90
archive_after_seconds: 3000
time_window_seconds: 3.0
max_gallery_set_each_person: 10

# Camera Topology
camera_topology:
  0: [1, 2]
  1: [0, 2]
  2: [0, 1]

camera_transition_max_time:
  "0_1": 5.0
  "1_2": 5.0
  "0_2": 10.0
```

## 🎯 Features

### Dashboard (Real-time)
- ✅ Live video streams từ tất cả cameras
- ✅ Real-time event notifications (appear, move, reappear)
- ✅ Active person count
- ✅ Click event để jump tới camera
- ✅ Dark/Light theme

### Report & Analytics
- ✅ Total unique visitors
- ✅ Average dwell time
- ✅ Peak hour traffic
- ✅ Hourly traffic chart (live data)
- ✅ Movement flow between cameras
- ✅ Auto-refresh every 5 seconds

### Video Events
- ✅ Event timeline
- ✅ Filter by camera/person
- ✅ Search functionality

### Configuration
- ✅ Camera settings
- ✅ Detection thresholds
- ✅ Tracking parameters

## 🔧 Installation

### Backend
```bash
cd backend-server
pip install -r requirements.txt

# Nếu dùng GPU (CUDA)
pip install onnxruntime-gpu

# Nếu dùng CPU
pip install onnxruntime
```

### Frontend
```bash
cd Front-end
npm install
```

## 📊 Event Types

- **appear**: Person xuất hiện lần đầu
- **move**: Person di chuyển sang camera khác
- **reappear**: Person xuất hiện lại sau khi lost
- **alert**: Cảnh báo (custom)

## 🐛 Troubleshooting

**Backend lỗi khi start:**
```bash
# Kiểm tra RTSP URLs
# Kiểm tra model files trong models/pretrained_models/
# Kiểm tra CUDA nếu dùng GPU
```

**Frontend không connect WebSocket:**
```bash
# Kiểm tra backend đang chạy: http://localhost:8080/health
# Kiểm tra CORS settings trong server.py
```

**Không detect được người:**
```bash
# Giảm object_detection_threshold trong config.yaml
# Kiểm tra camera stream quality
```

## 💡 Tips

- Dùng GPU để tăng FPS (set `inference_model_device: "cuda"`)
- Adjust `feature_extraction_threshold` để cân bằng precision/recall
- `time_window_seconds` càng lớn = re-ID càng dễ nhưng latency cao
- Kiểm tra tracking logs trong `tracking_logs/` để debug

## 📝 Tech Stack

- **Backend**: Python, FastAPI, OpenCV, ONNX Runtime
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **AI**: YOLO v4-tiny, OSNet, Re-ID
- **Communication**: WebSocket, REST API

---

**Version**: 1.0.0  
**Last Updated**: January 2026

[Your License Here]
