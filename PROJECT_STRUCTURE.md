# 🏗️ Cấu trúc Project Mới

## 📂 Tổng quan

```
MultiCamera/
│
├── 🤖 ai-service/              # AI SERVICE - Detection & Re-ID
│   ├── core/                   # Core AI modules
│   │   ├── __init__.py
│   │   ├── object_detection.py         # YOLO v4-tiny person detection
│   │   ├── feature_extraction.py       # OSNet feature extraction
│   │   └── person_lifecycle_manager.py # Person tracking lifecycle
│   │
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── rtsp_loader.py     # RTSP stream loader
│   │   └── helpers.py         # Image processing helpers
│   │
│   ├── __init__.py
│   ├── requirements.txt       # AI dependencies
│   └── README.md              # AI service docs
│
├── 🖥️ backend-server/          # BACKEND SERVER - FastAPI
│   ├── api/                    # API modules
│   │   ├── __init__.py
│   │   └── stream_manager.py  # Stream & tracking pipeline
│   │
│   ├── server.py              # FastAPI application
│   ├── requirements.txt       # Backend dependencies
│   └── README.md              # Backend docs
│
├── 🎨 frontend/                # FRONTEND - Next.js (đổi tên từ Front-end)
│   ├── app/                    # Next.js app directory
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── configuration/
│   │   ├── report/
│   │   └── video-events/
│   │
│   ├── components/            # React components
│   │   ├── dashboard-layout.tsx
│   │   ├── VideoStream.tsx
│   │   └── ui/
│   │
│   ├── hooks/                 # Custom hooks
│   │   └── useWebSocketStream.ts
│   │
│   ├── lib/                   # Utilities
│   ├── types/                 # TypeScript types
│   ├── package.json
│   └── README.md
│
├── 📦 models/                  # AI Models (shared)
│   └── pretrained_models/
│       ├── yolov4-tiny.onnx
│       ├── osnet_ain_x1_0_M.onnx
│       └── coco.names
│
├── 💾 storage/                 # Data storage (shared)
│   └── persons.json
│
├── 📊 tracking_logs/           # Tracking logs (shared)
│
├── ⚙️ config.yaml              # Configuration (shared)
│
├── 📖 README.md                # Main documentation
├── 📖 MIGRATION_GUIDE.md       # Migration guide
│
├── 🚀 start-backend.bat        # Windows: Start backend
├── 🚀 start-frontend.bat       # Windows: Start frontend
├── 🚀 start-backend.sh         # Linux/Mac: Start backend
└── 🚀 start-frontend.sh        # Linux/Mac: Start frontend
```

## 🔄 Luồng dữ liệu

```
RTSP Cameras
     ↓
[AI Service]
  - Object Detection (YOLO)
  - Feature Extraction (OSNet)
  - Person Lifecycle Management
     ↓
[Backend Server]
  - Stream Manager
  - WebSocket Server
  - REST API
     ↓
[Frontend]
  - WebSocket Client
  - Video Display
  - Dashboard UI
```

## 🎯 Phân chia trách nhiệm

### AI Service (ai-service/)
✅ Object detection (YOLO)
✅ Feature extraction (OSNet)
✅ Person tracking logic
✅ RTSP stream handling
✅ Image processing utilities

**Không có:** HTTP/WebSocket server

### Backend Server (backend-server/)
✅ FastAPI web server
✅ WebSocket streaming
✅ REST API endpoints
✅ Orchestration of AI pipeline
✅ Client request handling

**Không có:** AI model inference

### Frontend (frontend/)
✅ Next.js web application
✅ Real-time video display
✅ Dashboard UI
✅ User interactions
✅ Data visualization

**Không có:** AI logic, backend logic

## 📥 Dependencies

### AI Service
- OpenCV (computer vision)
- ONNX Runtime (model inference)
- NumPy, SciPy (computation)
- Albumentations (augmentation)

### Backend Server
- FastAPI (web framework)
- Uvicorn (ASGI server)
- + All AI service dependencies

### Frontend
- Next.js (React framework)
- TypeScript
- Tailwind CSS
- WebSocket client

## 🚀 Khởi động

### Full Stack (All services)

**Terminal 1 - Backend:**
```bash
cd backend-server
pip install -r requirements.txt
python server.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### Hoặc dùng scripts

**Windows:**
```cmd
start-backend.bat
start-frontend.bat
```

**Linux/Mac:**
```bash
./start-backend.sh
./start-frontend.sh
```

## 🔗 Communication

```
Frontend ←→ Backend ←→ AI Service
  3000         8080      (embedded)

Frontend → Backend: WebSocket (video stream)
Frontend → Backend: REST API (data/stats)
Backend → AI: Direct Python imports
```

## 📝 Configuration

Tất cả 3 services đều đọc từ `config.yaml` ở root:

```yaml
# RTSP streams
rtsp_urls: [...]

# AI models paths
object_detection_model_path: "./models/..."
feature_extraction_model_path: "./models/..."

# Tracking parameters
tracking_stream_enabled: true
object_detection_threshold: 0.7
feature_extraction_threshold: 0.42
time_window_seconds: 3.0

# Device
inference_model_device: "cuda"  # or "cpu"
```

## ✅ Advantages của cấu trúc mới

1. **Separation of Concerns**: Mỗi service có trách nhiệm rõ ràng
2. **Independent Scaling**: Có thể scale từng service riêng
3. **Better Testing**: Test từng service độc lập
4. **Easier Maintenance**: Dễ tìm và sửa bugs
5. **Modular**: Thay thế/upgrade từng phần dễ dàng
6. **Clear Dependencies**: Requirements riêng cho từng service

## 🔄 So sánh với cấu trúc cũ

| Aspect | Old | New |
|--------|-----|-----|
| **Organization** | Flat, mixed | Hierarchical, separated |
| **Dependencies** | One requirements.txt | Service-specific |
| **Imports** | Relative, messy | Organized, clean |
| **Documentation** | Single README | Per-service docs |
| **Testing** | Hard | Easier |
| **Scalability** | Monolithic | Microservices-ready |

## 📚 Documentation

- [Main README](README.md) - Overview
- [AI Service](ai-service/README.md) - AI modules
- [Backend Server](backend-server/README.md) - API server
- [Migration Guide](MIGRATION_GUIDE.md) - Hướng dẫn chuyển đổi

## 🐛 Troubleshooting

Xem [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) phần Troubleshooting.

## 📞 Support

Nếu gặp vấn đề:
1. Check logs của từng service
2. Verify config.yaml
3. Check model paths
4. Review import paths
5. Check dependencies installed

---

**Lưu ý:** Thư mục `Front-end/` cần đổi tên thành `frontend/` khi có thể (hiện đang bị lock bởi process khác).
