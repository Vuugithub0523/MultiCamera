# Native AI Backend - Multi-Camera Person Tracking

**Realtime person detection, tracking & re-identification system for multiple IP cameras**

🎯 **Optimized for Edge AI** | RTX 3050 8GB | Native Python (No Docker)

---

## ⚡ Latest Optimizations (v2.0)

### 🚀 Performance Improvements
- **Resolution:** 640x360 → **1280x720 HD** (4x increase)
- **FPS:** Increased 10-20% by eliminating frame copy/draw operations
- **RAM:** Reduced 30-40% (no frame duplication)
- **CPU:** Reduced 15-25% (backend doesn't draw annotations)

### 🎨 Architecture Changes
- **Backend:** Streams raw frames + metadata (tracking info in binary format)
- **Frontend:** Draws bounding boxes on canvas overlay using metadata
- **Benefits:** Lower latency, better performance, flexible visualization

### 📊 Binary Protocol
```
[4 bytes: metadata_length][metadata_json][frame_jpeg]
```

Metadata includes: `track_id`, `person_id`, `bbox`, `confidence`, `state`

---

## 🎯 Tracking Visualization

**Video streams with real-time tracking annotations:**
- ✅ **Bounding boxes** with unique colors per person
- ✅ **Person IDs** (ID:1, ID:2, ...) - Global across cameras
- ✅ **Track states** ([DET], [TRK], [LST], [CLT])
- ✅ **Confidence scores** (0.85)
- ✅ **Camera info & statistics** (FPS, tracks, persons)

**🧪 Test:** Open [test_optimized_stream.html](test_optimized_stream.html) in browser!

---

## 🚀 Features

- ✅ **RTSP Direct Input** - Read directly from IP cameras (minimal latency)
- ✅ **YOLO Detection** - Fast person detection with ONNX Runtime
- ✅ **BYTETracker** - Multi-object tracking with Kalman filter
- ✅ **Person Re-ID** - OSNet feature extraction for cross-camera tracking
- ✅ **WebSocket Streaming** - Real-time frame + metadata output
- ✅ **Camera Topology** - Smart tracking based on physical camera layout (NEW!)
- ✅ **Lifecycle Management** - Track person states (DETECTED, TRACKING, LOST, ARCHIVED)
- ✅ **REST API** - Camera management & statistics
- ✅ **Low RAM Usage** - ~300MB for 3 cameras (shared models)
- ✅ **Low Latency** - <80ms end-to-end processing
- ✅ **HD Quality** - 1280x720 resolution

---

## 📋 Prerequisites

### Hardware
- NVIDIA GPU (RTX 3050 or better)
- 8GB+ GPU VRAM
- 16GB+ System RAM recommended

### Software
- **Windows 10/11** or **Ubuntu 20.04+**
- **Python 3.9-3.11**
- **CUDA 11.8+** (for GPU acceleration)
- **NVIDIA Driver 520+**

### IP Cameras
- 3 IP cameras with RTSP support
- Network access to cameras
- Camera credentials

---

## 🔧 Installation

### 1. Clone or Navigate to Project

```bash
cd native-ai-backend
```

### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you don't have GPU, edit `requirements.txt` and replace `onnxruntime-gpu` with `onnxruntime`.

### 4. Verify Models

Models should already be in `models/` directory. Verify:

```bash
ls models/
# Should show:
# - yolov4-tiny.onnx
# - osnet_ain_x1_0_M.onnx
# - coco.names
```

If missing, copy from MultiCamera:
```bash
# Windows
Copy-Item -Recurse ..\MultiCamera\models\pretrained_models\* .\models\

# Linux
cp -r ../MultiCamera/models/pretrained_models/* ./models/
```

---

## ⚙️ Configuration

Edit `config.py` to configure your cameras:

```python
CAMERAS: List[Dict[str, str]] = [
    {
        "id": "cam01",
        "name": "Camera 1",
        "rtsp_url": "rtsp://admin:password@192.168.1.101:554/stream",
        "enabled": True,
    },
    {
        "id": "cam02",
        "name": "Camera 2",
        "rtsp_url": "rtsp://admin:password@192.168.1.102:554/stream",
        "enabled": True,
    },
    # Add more cameras...
]
```

### Key Settings:

- **DEVICE**: `"cuda"` or `"cpu"`
- **DETECTION_SKIP_FRAMES**: `2` = detect every 2 frames (increase for lower GPU usage)
- **OUTPUT_FPS**: `15` = send 15 FPS to frontend (lower = less bandwidth)
- **REID_THRESHOLD**: `0.42` = cosine distance threshold for same person

### 🆕 Camera Topology (NEW!)

Configure physical camera layout for intelligent tracking:

```python
# Define which cameras connect to which
CAMERA_TOPOLOGY = {
    "cam01": ["cam02"],           # Entrance -> Lobby
    "cam02": ["cam01", "cam03"],  # Lobby <-> Entrance/Warehouse
    "cam03": ["cam02"],           # Warehouse -> Lobby
}

# Maximum transition time between connected cameras
CAMERA_TRANSITION_MAX_TIME = {
    "cam01->cam02": 5.0,  # Max 5 seconds from cam01 to cam02
    "cam02->cam03": 6.0,  # Max 6 seconds from cam02 to cam03
}
```

**Benefits:**
- Prevents impossible transitions (e.g., cam01->cam03 direct)
- Rejects matches if transition time exceeds physical possibility
- Maintains same ID when person moves between connected cameras

**📖 Read more:** [CAMERA_TOPOLOGY_INTEGRATION.md](CAMERA_TOPOLOGY_INTEGRATION.md)

---

## 🏃 Running the Server

### Start Server

```bash
python main.py
```

You should see:
```
============================================================
NATIVE AI BACKEND - Multi-Camera Tracking System
============================================================

[YOLODetector] Loaded model: models/yolov4-tiny.onnx
[FeatureExtractor] Loaded model: models/osnet_ain_x1_0_M.onnx
[PersonDB] Loaded 0 persons from storage/persons.json
[Manager] Initialized camera: cam01
[Manager] Initialized camera: cam02
[Manager] Initialized camera: cam03
[Manager] Started 3 cameras

[Startup] Server ready on http://0.0.0.0:5000
============================================================
```

### Test Server

Open browser: http://localhost:5000

You should see API documentation.

---

## 📡 API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/cameras` | List all cameras |
| GET | `/api/cameras/{id}` | Get camera info |
| GET | `/api/cameras/{id}/stats` | Get camera statistics |
| GET | `/api/persons` | List all identified persons |
| GET | `/api/persons/{id}` | Get person info |
| GET | `/api/stats` | System-wide statistics |
| POST | `/api/database/save` | Save person DB to disk |

### WebSocket

**Endpoint:** `ws://localhost:5000/ws/tracking/{camera_id}`

**Usage:**
```javascript
const ws = new WebSocket('ws://localhost:5000/ws/tracking/cam01');
ws.binaryType = 'arraybuffer';

ws.onmessage = (event) => {
  // Receive JPEG frame
  const blob = new Blob([event.data], { type: 'image/jpeg' });
  const url = URL.createObjectURL(blob);
  imageElement.src = url;
};
```

---

## 🎨 Frontend Integration

### Update Frontend Config

Edit `Front-end/.env.local`:

```bash
NEXT_PUBLIC_BACKEND_URL=http://localhost:5000
```

### Update WebSocket Hook (if needed)

File: `Front-end/hooks/useWebSocketStream.ts`

Change endpoint from `/ws/stream/` to `/ws/tracking/`:

```typescript
const wsUrl = `${baseUrl}/ws/tracking/${cameraId}`;
```

### Run Frontend

```bash
cd ../ab-camera-loyalty/Front-end
npm install
npm run dev
```

Open: http://localhost:3000

---

## 📊 Performance Tuning

### For RTX 3050 8GB:

**Current settings (good balance):**
- Detection skip: 2 frames
- Input resolution: 1280x720
- Output FPS: 15
- JPEG quality: 80

**If lag/low FPS:**
- Increase `DETECTION_SKIP_FRAMES` to 3-4
- Decrease `INPUT_WIDTH` to 960
- Decrease `OUTPUT_FPS` to 10

**If want better quality:**
- Decrease `DETECTION_SKIP_FRAMES` to 1
- Increase `JPEG_QUALITY` to 90
- Increase `OUTPUT_FPS` to 20

### Memory Usage:

| Component | RAM | VRAM |
|-----------|-----|------|
| YOLO model | 200MB | 400MB |
| OSNet model | 100MB | 200MB |
| 3x frame buffers | 50MB | - |
| Tracking state | 50MB | - |
| **Total** | **~400MB** | **~600MB** |

---

## 🐛 Troubleshooting

### CUDA not available

**Error:** `CUDA not available, using CPU`

**Fix:**
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall CUDA 11.8+
3. Reinstall onnxruntime-gpu: `pip install --force-reinstall onnxruntime-gpu`

### Camera connection failed

**Error:** `Failed to open stream`

**Fix:**
1. Verify RTSP URL with VLC: `vlc rtsp://...`
2. Check camera credentials
3. Check network connectivity
4. Try reducing INPUT_WIDTH/HEIGHT

### Low FPS / High latency

**Fix:**
1. Increase `DETECTION_SKIP_FRAMES` (2 → 3)
2. Lower input resolution
3. Close other GPU applications
4. Check CPU usage (should be <50%)

### Out of memory

**Fix:**
1. Decrease `INPUT_WIDTH` and `INPUT_HEIGHT`
2. Increase `DETECTION_SKIP_FRAMES`
3. Reduce number of cameras
4. Check other GPU processes: `nvidia-smi`

---

## 📁 Project Structure

```
native-ai-backend/
├── main.py                 # FastAPI server entry point
├── config.py               # Configuration
├── requirements.txt        # Python dependencies
│
├── api/                    # FastAPI routes & WebSocket
│   ├── models.py           # Pydantic schemas
│   ├── rest.py             # REST API routes
│   └── websocket.py        # WebSocket manager
│
├── core/                   # Core processing logic
│   ├── camera_reader.py    # RTSP reader
│   ├── pipeline.py         # Per-camera pipeline
│   └── manager.py          # Multi-camera manager
│
├── detection/              # YOLO detection & tracking
│   ├── yolo_detector.py    # YOLO ONNX inference
│   ├── byte_tracker.py     # BYTETracker
│   ├── kalman_filter.py    # Kalman filter
│   ├── matching.py         # IoU matching
│   └── basetrack.py        # Base track class
│
├── reid/                   # Person re-identification
│   ├── feature_extractor.py  # OSNet features
│   └── person_database.py    # Person DB
│
├── utils/                  # Utilities
│   ├── logger.py
│   └── fps_limiter.py
│
├── models/                 # AI models (ONNX)
│   ├── yolov4-tiny.onnx
│   ├── osnet_ain_x1_0_M.onnx
│   └── coco.names
│
└── storage/                # Runtime data
    └── persons.json        # Person database
```

---

## 🔄 Architecture

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Camera 1 │  │ Camera 2 │  │ Camera 3 │  (RTSP)
└─────┬────┘  └─────┬────┘  └─────┬────┘
      │             │             │
      ▼             ▼             ▼
┌─────────────────────────────────────────┐
│         RTSPReader (per camera)         │
│  - Thread-based frame capture           │
│  - Minimal buffer (2 frames)            │
└─────────────┬───────────────────────────┘
              │ numpy array (BGR)
              ▼
┌─────────────────────────────────────────┐
│      CameraPipeline (per camera)        │
│  1. YOLO Detection (skip frames)        │
│  2. BYTETracker (every frame)           │
│  3. Feature Extraction (new tracks)     │
│  4. Person Re-ID (match/new)            │
│  5. Annotate & Encode JPEG              │
└─────────────┬───────────────────────────┘
              │ JPEG + metadata
              ▼
┌─────────────────────────────────────────┐
│      FastAPI + WebSocket Server         │
│  - WebSocket broadcast (per camera)     │
│  - REST API (stats, persons)            │
└─────────────┬───────────────────────────┘
              │ WebSocket (JPEG stream)
              ▼
┌─────────────────────────────────────────┐
│         Frontend (Next.js)              │
│  - Display 3 camera grid                │
│  - Show track IDs & bounding boxes      │
└─────────────────────────────────────────┘
```

**Key Design:**
- ✅ RTSP input (not WebSocket) → minimal latency
- ✅ Shared YOLO & OSNet models → low RAM
- ✅ Per-camera tracker state → independent tracking
- ✅ Skip-frame detection → low GPU usage
- ✅ WebSocket output only → frontend display

---

## 📝 Development

### Testing with Video Files

Instead of RTSP cameras, use video files:

```bash
# Set environment variable
export USE_VIDEO_FILES=1
export VIDEO_DIR=./videos

# Create videos directory
mkdir videos
# Copy test videos: video1.mp4, video2.mp4, video3.mp4

# Run server
python main.py
```

### Debug Mode

```bash
export ENV=development
python main.py
```

This enables:
- Debug logging
- Slower detection (easier to follow)

---

## 🚀 Production Deployment

### Using PM2 (recommended)

```bash
# Install PM2
npm install -g pm2

# Start server
pm2 start main.py --name native-ai-backend --interpreter python

# View logs
pm2 logs native-ai-backend

# Restart
pm2 restart native-ai-backend

# Stop
pm2 stop native-ai-backend
```

### Using systemd (Linux)

Create `/etc/systemd/system/native-ai-backend.service`:

```ini
[Unit]
Description=Native AI Backend
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/native-ai-backend
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable native-ai-backend
sudo systemctl start native-ai-backend
```

---

## 📄 License

MIT License - see LICENSE file

---

## 🙏 Acknowledgments

- **YOLOv4** - Alexey Bochkovskiy
- **BYTETrack** - ByteDance
- **OSNet** - Kaiyang Zhou
- **FastAPI** - Sebastián Ramírez

---

## 📧 Support

For issues or questions, check:
- GitHub Issues
- Project documentation
- Configuration comments in `config.py`

---

**Built for realtime Edge AI tracking on NVIDIA RTX 3050 8GB** 🚀
#   F E - L o c a l 
 
 