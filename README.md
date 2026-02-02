# Multi-Camera RTSP + AI Pipeline

Hệ thống giám sát multi-camera với AI detection pipeline, streaming qua LiveKit WebRTC.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────┐     ┌──────────────┐
│  RTSP Cam   │────▶│   AI Service     │────▶│   MediaMTX    │────▶│   LiveKit    │
│  (Source)   │     │  (Python/ONNX)   │     │ (RTSP Server) │     │   Server     │
└─────────────┘     └──────────────────┘     └───────────────┘     └──────────────┘
                           │                                              │
                           │ SSE Events                                   │ WebRTC
                           ▼                                              ▼
                    ┌──────────────────┐                         ┌──────────────┐
                    │   Event Feed     │◀────────────────────────│   Frontend   │
                    │   (Sidebar)      │                         │   (React)    │
                    └──────────────────┘                         └──────────────┘
```

## Components

1. **AI Service** (Python 3.10 + ONNXRuntime-GPU)
   - RTSP ingest with low-latency frame dropping
   - YOLOv4-tiny detection (person class)
   - Frame annotation (bbox, labels, FPS)
   - Re-stream to MediaMTX RTSP

2. **Publisher** (Node.js/TypeScript)
   - Pull annotated RTSP streams
   - Publish to LiveKit room

3. **Frontend** (Vite + React)
   - LiveKit video player
   - Event feed sidebar

4. **Infrastructure**
   - MediaMTX (local RTSP server)
   - LiveKit (WebRTC server)

## Quick Start

### Prerequisites
- Python 3.10.x
- Node.js 18+
- Docker & Docker Compose
- FFmpeg

### 1. Start Infrastructure

```powershell
.\scripts\start_rtsp.ps1      # Start MediaMTX
.\scripts\start_livekit.ps1   # Start LiveKit
```

### 2. Start Services

```powershell
.\scripts\start_ai.ps1        # Start AI service
.\scripts\start_publisher.ps1 # Start LiveKit publisher
.\scripts\start_frontend.ps1  # Start frontend
```

### Or Start All

```powershell
.\scripts\start_all.ps1
```

### 3. Open Browser

Navigate to `http://localhost:5173`

## Configuration

Edit `config.yaml` to configure:
- Camera RTSP URLs
- AI detection settings
- Annotation options
- Streaming parameters
- LiveKit connection

## Project Structure

```
MultiCamera/
├── config.yaml              # Root configuration
├── README.md
├── models/
│   └── pretrained_models/
│       ├── yolov4-tiny.onnx
│       ├── coco.names
│       └── osnet_ain_x1_0_M.onnx
├── ai-service/              # Python AI pipeline
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── ingest/
│   │   ├── inference/
│   │   ├── annotate/
│   │   ├── restream/
│   │   ├── events/
│   │   └── utils/
│   ├── requirements.txt
│   └── .env.example
├── publisher/               # Node.js LiveKit publisher
│   ├── src/
│   │   ├── index.ts
│   │   ├── config.ts
│   │   ├── livekit.ts
│   │   └── ffmpeg.ts
│   ├── package.json
│   └── tsconfig.json
├── frontend/                # Vite React app
├── infra/
│   ├── livekit/
│   │   ├── docker-compose.yml
│   │   └── livekit.yaml
│   └── rtsp/
│       ├── docker-compose.yml
│       └── mediamtx.yml
└── scripts/
    ├── start_rtsp.ps1
    ├── start_livekit.ps1
    ├── start_ai.ps1
    ├── start_publisher.ps1
    ├── start_frontend.ps1
    └── start_all.ps1
```

## API Endpoints

AI Service provides:
- `GET /health` - Health check
- `GET /api/cameras` - List cameras
- `GET /api/stats` - Pipeline statistics
- `GET /api/events/stream` - SSE event stream

## License

MIT
