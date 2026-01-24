# Backend Server

Backend server cho hệ thống Multi-Camera Person Tracking sử dụng FastAPI.

## Cấu trúc

```
backend-server/
├── api/
│   └── stream_manager.py    # Stream management and tracking pipeline
├── server.py                # FastAPI application
└── requirements.txt         # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python server.py
```

Server sẽ chạy tại: `http://localhost:8080`

## API Endpoints

### REST API

- `GET /` - Root endpoint, service info
- `GET /health` - Health check
- `GET /api/cameras` - Lấy danh sách cameras
- `GET /api/tracking/stats` - Lấy thống kê tracking

### WebSocket

- `WS /ws/{stream_type}/{camera_id}` - Stream video real-time
  - **stream_type**: `raw` hoặc `tracking`
  - **camera_id**: ID của camera (e.g., `cam01`, `cam02`)

## Usage Examples

### Health Check
```bash
curl http://localhost:8080/health
```

### Get Cameras
```bash
curl http://localhost:8080/api/cameras
```

### Get Tracking Stats
```bash
curl http://localhost:8080/api/tracking/stats
```

### WebSocket Stream (JavaScript)
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/tracking/cam01');

ws.onmessage = (event) => {
  const blob = event.data;
  const url = URL.createObjectURL(blob);
  imageElement.src = url;
};
```

## Configuration

Server đọc cấu hình từ `../config.yaml` ở thư mục gốc project.

## Architecture

### Stream Manager
- Quản lý RTSP streams từ cameras
- Chạy AI pipeline (detection + feature extraction)
- Cung cấp frames qua WebSocket

### Tracking Pipeline
- Object detection (YOLO)
- Feature extraction (OSNet)
- Person lifecycle management
- Multi-camera re-identification

## Development

### Run with auto-reload
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8080
```

### API Documentation

Khi server đang chạy, truy cập:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## Notes

- Server cần AI service để hoạt động
- Đảm bảo models được download
- RTSP URLs phải valid trong config.yaml
