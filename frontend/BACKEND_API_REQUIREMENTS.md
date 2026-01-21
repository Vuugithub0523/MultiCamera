# Backend API Requirements Documentation

Tài liệu này mô tả chi tiết tất cả các API endpoints và data requirements mà Frontend cần từ Backend.

## 📋 Mục lục

1. [Dashboard APIs](#1-dashboard-apis)
2. [Video Events APIs](#2-video-events-apis)
3. [Report & Analytics APIs](#3-report--analytics-apis)
4. [Configuration APIs](#4-configuration-apis)
5. [WebSocket Real-time Updates](#5-websocket-real-time-updates)
6. [Data Models](#6-data-models)

---

## 1. Dashboard APIs

### 1.1. Get All Cameras
**Endpoint:** `GET /api/cameras`

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "code": "CCTV 01",
      "name": "Meeting Room (Akasia Room)",
      "location": "Floor 2",
      "status": "online",
      "rtspUrl": "rtsp://192.168.1.101:554/stream1",
      "streamUrl": "http://localhost:8000/stream/1"
    },
    {
      "id": 2,
      "code": "CCTV 02",
      "name": "Open Workspace",
      "location": "Floor 1",
      "status": "online",
      "rtspUrl": "rtsp://192.168.1.102:554/stream1",
      "streamUrl": "http://localhost:8000/stream/2"
    }
  ]
}
```

**Description:**
- Danh sách tất cả cameras trong hệ thống
- `status`: "online" | "offline"
- `streamUrl`: URL để lấy video stream đã có bounding boxes

---

### 1.2. Get Active Tracked Persons
**Endpoint:** `GET /api/tracking/active`

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": 5,
      "personId": 5,
      "confidence": 98,
      "cameraId": 1,
      "firstSeen": "10:00:05",
      "lastSeen": "10:05:30",
      "name": "Nguyễn Văn A",
      "isKnown": true,
      "thumbnail": "http://localhost:8000/thumbnails/person_5.jpg"
    },
    {
      "id": 12,
      "personId": 12,
      "confidence": 95,
      "cameraId": 1,
      "firstSeen": "10:02:15",
      "lastSeen": "10:05:32",
      "name": "Unknown",
      "isKnown": false,
      "thumbnail": "http://localhost:8000/thumbnails/person_12.jpg"
    }
  ],
  "total": 4
}
```

**Description:**
- Danh sách người đang được tracking trong thời gian thực
- `confidence`: độ tin cậy nhận dạng (0-100)
- `isKnown`: người đã được đăng ký hay chưa

---

### 1.3. Get Recent Events
**Endpoint:** `GET /api/events/recent`

**Query Parameters:**
- `limit`: số lượng events (default: 20)
- `cameraId`: lọc theo camera (optional)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "time": "10:05:32",
      "timestamp": "2026-01-16T10:05:32Z",
      "personId": 5,
      "personName": "Nguyễn Văn A",
      "type": "appear",
      "camera": 1,
      "cameraName": "CCTV 01",
      "thumbnail": "http://localhost:8000/thumbnails/event_1.jpg",
      "message": null
    },
    {
      "id": 2,
      "time": "10:05:45",
      "timestamp": "2026-01-16T10:05:45Z",
      "personId": 5,
      "personName": "Nguyễn Văn A",
      "type": "move",
      "camera": 2,
      "cameraName": "CCTV 02",
      "thumbnail": "http://localhost:8000/thumbnails/event_2.jpg",
      "message": null
    },
    {
      "id": 4,
      "time": "10:06:30",
      "timestamp": "2026-01-16T10:06:30Z",
      "personId": 8,
      "personName": "Unknown",
      "type": "alert",
      "camera": 2,
      "cameraName": "CCTV 02",
      "thumbnail": "http://localhost:8000/thumbnails/event_4.jpg",
      "message": "Phát hiện người lạ tại khu vực kho"
    }
  ]
}
```

**Event Types:**
- `"appear"`: Người xuất hiện lần đầu
- `"move"`: Người di chuyển sang camera khác
- `"alert"`: Cảnh báo (người lạ, hành vi bất thường)

---

### 1.4. Get Camera Stream
**Endpoint:** `GET /api/stream/{camera_id}`

**Response:** Video stream (MJPEG/HLS)

**Description:**
- Stream video đã được xử lý với bounding boxes
- Bounding boxes đã vẽ sẵn trên video
- Format: MJPEG hoặc HLS

**Alternative WebSocket approach:**
```
ws://localhost:8000/ws/stream/{camera_id}
```

---

## 2. Video Events APIs

### 2.1. Get Video Events List
**Endpoint:** `GET /api/video-events`

**Query Parameters:**
- `date_range`: "today" | "yesterday" | "week" | "month"
- `camera_id`: filter by camera (optional)
- `event_type`: "person_appear" | "new_person" | "person_return" | "abnormal" | "all"
- `search`: tìm kiếm theo tên hoặc mô tả (optional)
- `page`: số trang (default: 1)
- `limit`: số lượng mỗi trang (default: 20)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "cameraId": 1,
      "cameraName": "Camera 1 - Entrance",
      "timestamp": "2026-01-16T10:05:32Z",
      "duration": 45,
      "personId": 5,
      "personName": "Nguyễn Văn A",
      "eventType": "person_appear",
      "thumbnailUrl": "http://localhost:8000/video-events/1/thumbnail.jpg",
      "videoUrl": "http://localhost:8000/video-events/1/video.mp4",
      "size": "12.5 MB",
      "sizeBytes": 13107200,
      "description": "Người xuất hiện tại lối vào chính",
      "isAlert": false
    },
    {
      "id": 2,
      "cameraId": 2,
      "cameraName": "Camera 2 - Lobby",
      "timestamp": "2026-01-16T10:06:15Z",
      "duration": 120,
      "personId": 8,
      "personName": "Unknown Person",
      "eventType": "new_person",
      "thumbnailUrl": "http://localhost:8000/video-events/2/thumbnail.jpg",
      "videoUrl": "http://localhost:8000/video-events/2/video.mp4",
      "size": "32.1 MB",
      "sizeBytes": 33672192,
      "description": "Phát hiện người mới chưa được đăng ký",
      "isAlert": false
    },
    {
      "id": 3,
      "cameraId": 2,
      "cameraName": "Camera 2 - Lobby",
      "timestamp": "2026-01-16T10:06:30Z",
      "duration": 90,
      "personId": 8,
      "personName": "Unknown Person",
      "eventType": "abnormal",
      "thumbnailUrl": "http://localhost:8000/video-events/3/thumbnail.jpg",
      "videoUrl": "http://localhost:8000/video-events/3/video.mp4",
      "size": "24.8 MB",
      "sizeBytes": 26009395,
      "description": "Phát hiện người lạ tại khu vực kho",
      "isAlert": true
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 156,
    "totalPages": 8
  }
}
```

**Event Types:**
- `"person_appear"`: Người xuất hiện
- `"new_person"`: Phát hiện người mới
- `"person_return"`: Người quen quay lại
- `"abnormal"`: Sự kiện bất thường

---

### 2.2. Get Video Events Statistics
**Endpoint:** `GET /api/video-events/statistics`

**Query Parameters:**
- `date_range`: "today" | "yesterday" | "week" | "month"

**Response:**
```json
{
  "success": true,
  "data": {
    "totalEvents": 156,
    "newPersons": 12,
    "alerts": 8,
    "storageUsed": "2.3 GB",
    "storageUsedBytes": 2469606195
  }
}
```

---

### 2.3. Download Video Event
**Endpoint:** `GET /api/video-events/{event_id}/download`

**Response:** File download (video/mp4)

**Description:**
- Download video file của event
- Response headers: `Content-Disposition: attachment; filename="event_1.mp4"`

---

### 2.4. Delete Video Event
**Endpoint:** `DELETE /api/video-events/{event_id}`

**Response:**
```json
{
  "success": true,
  "message": "Video event deleted successfully"
}
```

---

## 3. Report & Analytics APIs

### 3.1. Get Hourly Traffic Data
**Endpoint:** `GET /api/analytics/hourly-traffic`

**Query Parameters:**
- `date`: date string (format: "YYYY-MM-DD", default: today)
- `camera_id`: filter by camera (optional)

**Response:**
```json
{
  "success": true,
  "data": [
    { "hour": "06:00", "count": 12 },
    { "hour": "07:00", "count": 28 },
    { "hour": "08:00", "count": 45 },
    { "hour": "09:00", "count": 62 },
    { "hour": "10:00", "count": 78 },
    { "hour": "11:00", "count": 85 },
    { "hour": "12:00", "count": 92 }
  ]
}
```

---

### 3.2. Get KPI Statistics
**Endpoint:** `GET /api/analytics/kpi`

**Query Parameters:**
- `date_range`: "today" | "yesterday" | "week" | "month"

**Response:**
```json
{
  "success": true,
  "data": {
    "totalUniqueVisitors": 247,
    "change": "+12%",
    "avgDwellTime": "4m 32s",
    "avgDwellTimeSeconds": 272,
    "dwellTimeChange": "+8%",
    "peakHour": "17:00",
    "peakHourCount": 95,
    "activeZones": 3,
    "totalZones": 3,
    "activeZonesPercentage": "100%"
  }
}
```

---

### 3.3. Get Heatmap Data
**Endpoint:** `GET /api/analytics/heatmap`

**Query Parameters:**
- `camera_id`: required
- `date_range`: "today" | "yesterday" | "week" | "month"

**Response:**
```json
{
  "success": true,
  "data": {
    "cameraId": 1,
    "zones": [
      {
        "id": 1,
        "x": 10,
        "y": 20,
        "w": 25,
        "h": 30,
        "intensity": 0.9,
        "label": "Entrance",
        "count": 145,
        "avgDwellTime": 35
      },
      {
        "id": 2,
        "x": 45,
        "y": 15,
        "w": 20,
        "h": 25,
        "intensity": 0.7,
        "label": "Counter",
        "count": 98,
        "avgDwellTime": 120
      }
    ]
  }
}
```

**Description:**
- `x, y, w, h`: vị trí và kích thước zone (percentage)
- `intensity`: mức độ hoạt động (0-1)
- `count`: số lượt người qua zone
- `avgDwellTime`: thời gian lưu trú trung bình (giây)

---

### 3.4. Get Movement Flow Data
**Endpoint:** `GET /api/analytics/flow`

**Query Parameters:**
- `date_range`: "today" | "yesterday" | "week" | "month"

**Response:**
```json
{
  "success": true,
  "data": {
    "nodes": [
      { "id": 1, "name": "Camera 1\n(Entrance)", "count": 85 },
      { "id": 2, "name": "Camera 2\n(Lobby)", "count": 72 },
      { "id": 3, "name": "Camera 3\n(Warehouse)", "count": 45 }
    ],
    "links": [
      { "source": 1, "target": 2, "value": 45, "percentage": 53 },
      { "source": 1, "target": 3, "value": 25, "percentage": 29 },
      { "source": 2, "target": 3, "value": 35, "percentage": 49 },
      { "source": 2, "target": 1, "value": 15, "percentage": 21 },
      { "source": 3, "target": 2, "value": 20, "percentage": 44 },
      { "source": 3, "target": 1, "value": 30, "percentage": 67 }
    ]
  }
}
```

**Description:**
- `nodes`: danh sách cameras với số người qua
- `links`: luồng di chuyển giữa các cameras
- `value`: số người di chuyển
- `percentage`: phần trăm của tổng số người từ camera nguồn

---

## 4. Configuration APIs

### 4.1. Get All Cameras (Configuration)
**Endpoint:** `GET /api/config/cameras`

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "Camera 1 - Entrance",
      "url": "rtsp://192.168.1.101:554/stream1",
      "status": "online",
      "location": "Floor 1",
      "code": "CCTV 01"
    }
  ]
}
```

---

### 4.2. Create Camera
**Endpoint:** `POST /api/config/cameras`

**Request Body:**
```json
{
  "name": "Camera 4 - Parking",
  "url": "rtsp://192.168.1.104:554/stream1",
  "location": "Ground Floor",
  "code": "CCTV 04"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Camera created successfully",
  "data": {
    "id": 4,
    "name": "Camera 4 - Parking",
    "url": "rtsp://192.168.1.104:554/stream1",
    "status": "offline",
    "location": "Ground Floor",
    "code": "CCTV 04"
  }
}
```

---

### 4.3. Update Camera
**Endpoint:** `PUT /api/config/cameras/{camera_id}`

**Request Body:**
```json
{
  "name": "Camera 1 - Main Entrance",
  "url": "rtsp://192.168.1.101:554/stream1",
  "location": "Floor 1"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Camera updated successfully",
  "data": {
    "id": 1,
    "name": "Camera 1 - Main Entrance",
    "url": "rtsp://192.168.1.101:554/stream1",
    "status": "online",
    "location": "Floor 1"
  }
}
```

---

### 4.4. Delete Camera
**Endpoint:** `DELETE /api/config/cameras/{camera_id}`

**Response:**
```json
{
  "success": true,
  "message": "Camera deleted successfully"
}
```

---

### 4.5. Get Tracking Parameters
**Endpoint:** `GET /api/config/tracking-parameters`

**Response:**
```json
{
  "success": true,
  "data": {
    "confidenceThreshold": 0.6,
    "reIdThreshold": 0.7,
    "maxTrackAge": 30,
    "minTrackHits": 3,
    "iouThreshold": 0.3
  }
}
```

---

### 4.6. Update Tracking Parameters
**Endpoint:** `PUT /api/config/tracking-parameters`

**Request Body:**
```json
{
  "confidenceThreshold": 0.65,
  "reIdThreshold": 0.75,
  "maxTrackAge": 35,
  "minTrackHits": 4,
  "iouThreshold": 0.35
}
```

**Response:**
```json
{
  "success": true,
  "message": "Tracking parameters updated successfully",
  "data": {
    "confidenceThreshold": 0.65,
    "reIdThreshold": 0.75,
    "maxTrackAge": 35,
    "minTrackHits": 4,
    "iouThreshold": 0.35
  }
}
```

---

## 5. WebSocket Real-time Updates

### 5.1. Real-time Events Stream
**WebSocket URL:** `ws://localhost:8000/ws/events`

**Message Format (from server):**
```json
{
  "type": "new_event",
  "data": {
    "id": 7,
    "time": "10:08:15",
    "timestamp": "2026-01-16T10:08:15Z",
    "personId": 9,
    "personName": "Unknown",
    "type": "appear",
    "camera": 1,
    "cameraName": "CCTV 01",
    "thumbnail": "http://localhost:8000/thumbnails/event_7.jpg"
  }
}
```

---

### 5.2. Real-time Tracking Updates
**WebSocket URL:** `ws://localhost:8000/ws/tracking`

**Message Format (from server):**
```json
{
  "type": "tracking_update",
  "data": {
    "activeTracks": [
      {
        "id": 5,
        "personId": 5,
        "confidence": 98,
        "cameraId": 1,
        "position": { "x": 100, "y": 200, "w": 50, "h": 120 },
        "name": "Nguyễn Văn A"
      }
    ],
    "totalActive": 5
  }
}
```

---

### 5.3. Camera Status Updates
**WebSocket URL:** `ws://localhost:8000/ws/cameras`

**Message Format (from server):**
```json
{
  "type": "camera_status",
  "data": {
    "cameraId": 1,
    "status": "online",
    "timestamp": "2026-01-16T10:08:15Z"
  }
}
```

---

## 6. Data Models

### 6.1. Camera Model
```typescript
interface Camera {
  id: number;
  code: string;            // e.g., "CCTV 01"
  name: string;            // e.g., "Meeting Room"
  location: string;        // e.g., "Floor 2"
  status: "online" | "offline";
  rtspUrl: string;         // RTSP stream URL
  streamUrl: string;       // HTTP stream URL with bounding boxes
}
```

### 6.2. TrackedPerson Model
```typescript
interface TrackedPerson {
  id: number;
  personId: number;
  confidence: number;      // 0-100
  cameraId: number;
  firstSeen: string;       // HH:mm:ss
  lastSeen: string;        // HH:mm:ss
  name: string;
  isKnown: boolean;
  thumbnail: string;       // URL
}
```

### 6.3. Event Model
```typescript
interface Event {
  id: number;
  time: string;            // HH:mm:ss
  timestamp: string;       // ISO 8601
  personId: number;
  personName: string;
  type: "appear" | "move" | "alert";
  camera: number;
  cameraName: string;
  thumbnail: string;       // URL
  message?: string;        // For alerts
}
```

### 6.4. VideoEvent Model
```typescript
interface VideoEvent {
  id: number;
  cameraId: number;
  cameraName: string;
  timestamp: string;       // ISO 8601
  duration: number;        // seconds
  personId?: number;
  personName?: string;
  eventType: "person_appear" | "new_person" | "person_return" | "abnormal";
  thumbnailUrl: string;
  videoUrl: string;
  size: string;            // e.g., "12.5 MB"
  sizeBytes: number;
  description: string;
  isAlert: boolean;
}
```

---

## 7. Implementation Notes

### 7.1. Video Streaming
Backend nên cung cấp video stream theo một trong các cách:

**Option 1: MJPEG Stream**
```python
# FastAPI example
@app.get("/api/stream/{camera_id}")
async def stream_camera(camera_id: int):
    def generate():
        while True:
            frame = get_processed_frame(camera_id)  # Frame with bounding boxes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
```

**Option 2: HLS Stream**
- Sử dụng ffmpeg để convert RTSP sang HLS
- Serve .m3u8 và .ts files
- Frontend dùng video.js hoặc hls.js

**Option 3: WebRTC**
- Low latency
- Phức tạp hơn về implementation

### 7.2. Video Storage Structure
```
/storage/
  /video-events/
    /2026/
      /01/
        /16/
          /event_1.mp4
          /event_1_thumbnail.jpg
          /event_2.mp4
          /event_2_thumbnail.jpg
```

### 7.3. CORS Configuration
```python
# FastAPI CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 7.4. Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "CAMERA_NOT_FOUND",
    "message": "Camera with ID 99 not found",
    "details": {}
  }
}
```

### 7.5. Authentication (Future)
Nếu cần authentication:
```typescript
// Headers for authenticated requests
{
  "Authorization": "Bearer <token>",
  "Content-Type": "application/json"
}
```

---

## 8. Priority Implementation Order

### Phase 1 - Core Dashboard (Cao nhất)
1. ✅ `GET /api/cameras` - Danh sách cameras
2. ✅ `GET /api/stream/{camera_id}` - Video streaming
3. ✅ `GET /api/tracking/active` - Active tracked persons
4. ✅ `GET /api/events/recent` - Recent events
5. ✅ `ws://localhost:8000/ws/events` - Real-time events

### Phase 2 - Video Events Storage
6. ✅ `GET /api/video-events` - Video events list
7. ✅ `GET /api/video-events/statistics` - Statistics
8. ✅ `GET /api/video-events/{id}/download` - Download video

### Phase 3 - Configuration
9. ✅ `GET /api/config/cameras` - Camera management
10. ✅ `POST /api/config/cameras` - Add camera
11. ✅ `PUT /api/config/cameras/{id}` - Update camera
12. ✅ `DELETE /api/config/cameras/{id}` - Delete camera
13. ✅ `GET /api/config/tracking-parameters` - Get parameters
14. ✅ `PUT /api/config/tracking-parameters` - Update parameters

### Phase 4 - Analytics & Reports
15. ✅ `GET /api/analytics/hourly-traffic` - Traffic data
16. ✅ `GET /api/analytics/kpi` - KPI statistics
17. ✅ `GET /api/analytics/heatmap` - Heatmap data
18. ✅ `GET /api/analytics/flow` - Movement flow

---

## 9. Testing Endpoints

Có thể dùng curl hoặc Postman để test:

```bash
# Get cameras list
curl http://localhost:8000/api/cameras

# Get active tracking
curl http://localhost:8000/api/tracking/active

# Get recent events
curl http://localhost:8000/api/events/recent?limit=10

# Get video events with filters
curl "http://localhost:8000/api/video-events?date_range=today&camera_id=1&event_type=new_person"

# Update tracking parameters
curl -X PUT http://localhost:8000/api/config/tracking-parameters \
  -H "Content-Type: application/json" \
  -d '{"confidenceThreshold": 0.7, "reIdThreshold": 0.8}'
```

---

## 10. Contact & Support

Nếu cần thêm thông tin hoặc có thay đổi về API requirements, vui lòng cập nhật document này.

**Document Version:** 1.0  
**Last Updated:** January 16, 2026  
**Maintained by:** Frontend Team
