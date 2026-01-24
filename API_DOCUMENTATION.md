# API Documentation

## Backend Server APIs

Base URL: `http://localhost:8080`

### REST Endpoints

#### 1. Get Tracking Statistics
```
GET /api/tracking/stats
```
Returns overall tracking statistics including active persons, session info, and state distribution.

**Response:**
```json
{
  "session_id": "20260124_143022",
  "session_duration": 3600.5,
  "total_persons": 247,
  "active_persons": 12,
  "lost_persons": 3,
  "archived_persons": 232,
  "state_distribution": {
    "tracking": 12,
    "lost": 3,
    "archived": 232
  },
  "time_window_rejections": 45,
  "topology_rejections": 12,
  "same_camera_matches": 156,
  "topology_transitions": 89
}
```

#### 2. Get Recent Tracking Events
```
GET /api/tracking/events?limit=50
```
Returns recent tracking events (person appear, move, reappear).

**Query Parameters:**
- `limit` (optional): Number of events to return (default: 50)

**Response:**
```json
{
  "events": [
    {
      "id": 1,
      "timestamp": "2026-01-24T14:30:45.123456",
      "time": "14:30:45",
      "type": "appear",
      "person_id": 5,
      "camera_id": "cam01",
      "confidence": 95.5
    },
    {
      "id": 2,
      "timestamp": "2026-01-24T14:31:12.654321",
      "time": "14:31:12",
      "type": "move",
      "person_id": 5,
      "camera_id": "cam02",
      "confidence": 92.3,
      "from_camera": "cam01"
    }
  ]
}
```

**Event Types:**
- `appear`: Person first detected
- `move`: Person moved to different camera
- `reappear`: Person re-detected after being lost
- `alert`: Alert/warning event (custom)

#### 3. Get Report Statistics
```
GET /api/report/stats
```
Returns comprehensive statistics for report page including hourly traffic and camera flow.

**Response:**
```json
{
  "total_unique_visitors": 247,
  "avg_dwell_time_seconds": 273.5,
  "peak_hour": "17:00",
  "peak_hour_count": 95,
  "active_cameras": 3,
  "total_cameras": 3,
  "hourly_traffic": [
    {
      "hour": "06:00",
      "count": 12
    },
    {
      "hour": "07:00",
      "count": 28
    }
  ],
  "camera_flow": {
    "totals": {
      "cam01": 85,
      "cam02": 72,
      "cam03": 45
    },
    "transitions": {
      "cam01->cam02": 45,
      "cam02->cam03": 35,
      "cam01->cam03": 25
    }
  },
  "session_duration": 14523.7
}
```

#### 4. Get Available Cameras
```
GET /api/cameras
```
Returns list of available camera IDs.

**Response:**
```json
{
  "cameras": ["cam01", "cam02", "cam03"]
}
```

### WebSocket Endpoints

#### 1. Camera Video Stream
```
WS /ws/{stream_type}/{camera_id}
```
Streams camera video frames as JPEG bytes.

**Parameters:**
- `stream_type`: `raw` or `tracking` (with bounding boxes)
- `camera_id`: Camera identifier (e.g., `cam01`, `cam02`, `cam03`)

**Usage (Frontend):**
```typescript
const ws = new WebSocket(`ws://localhost:8080/ws/tracking/cam01`)
ws.onmessage = (event) => {
  const blob = new Blob([event.data], { type: 'image/jpeg' })
  const url = URL.createObjectURL(blob)
  // Display image
}
```

#### 2. Real-time Events Stream
```
WS /ws/events
```
Streams tracking events in real-time as JSON messages.

**Message Format:**
```json
{
  "id": 123,
  "timestamp": "2026-01-24T14:30:45.123456",
  "time": "14:30:45",
  "type": "appear",
  "person_id": 5,
  "camera_id": "cam01",
  "confidence": 95.5
}
```

**Usage (Frontend):**
```typescript
const ws = new WebSocket(`ws://localhost:8080/ws/events`)
ws.onmessage = (event) => {
  const trackingEvent = JSON.parse(event.data)
  console.log('New event:', trackingEvent)
}
```

## CORS Configuration

The backend enables CORS for the following origins:
- `http://localhost:3000`
- `http://127.0.0.1:3000`

## Error Responses

When tracking is not enabled:
```json
{
  "error": "Tracking not enabled"
}
```

When camera is not found:
```json
{
  "error": "Camera {camera_id} not found"
}
```

## Notes

- WebSocket connections automatically reconnect on disconnect
- Events are limited to last 100 in memory
- Video frames are encoded with 80% JPEG quality
- Statistics update every processing cycle
- All timestamps are in ISO 8601 format
