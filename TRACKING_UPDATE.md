# Native AI Backend - Cập Nhật Tracking với RTSP

## Những Gì Đã Thay Đổi

### 1. **RTSP Stream Loader Mới** (`core/rtsp_loader.py`)
- Thay thế `RTSPReader` cũ bằng `RTSPStreamLoader` từ project MultiCamera
- Zero-latency buffering với threading
- Auto-reconnect khi mất kết nối
- Support cả RTSP và video files

### 2. **Person Lifecycle Manager** (`core/lifecycle_manager.py`)
- Quản lý vòng đời của mỗi người được phát hiện
- Các trạng thái: DETECTED → TRACKING → LOST → CONFIRMED_LOST → ARCHIVED
- Time window matching (3 giây mặc định)
- Tự động archive người đã rời khỏi khung hình
- Export tracking logs sang JSON và CSV

### 3. **Enhanced Pipeline** (`core/pipeline.py`)
- Tích hợp lifecycle management
- Vẽ tracking với màu sắc nhất quán cho mỗi người
- Hiển thị trạng thái lifecycle trên video
- Thêm thống kê real-time trên frame

### 4. **Cấu Hình Mới** (`config.py`)
```python
# Lifecycle parameters
MAX_LOST_FRAMES = 30           # 30 frames trước khi đánh dấu LOST
MAX_CONFIRM_LOST_FRAMES = 90   # 90 frames trước khi CONFIRMED_LOST
ARCHIVE_AFTER_SECONDS = 300    # 5 phút trước khi archive
TIME_WINDOW_SECONDS = 3.0      # Time window cho matching
```

## Cách Sử Dụng

### 1. Cấu Hình RTSP URLs
Mở [config.py](config.py) và cập nhật RTSP URLs của cameras:

```python
CAMERAS: List[Dict[str, str]] = [
    {
        "id": "cam01",
        "name": "Camera 01 - Entrance",
        "rtsp_url": "rtsp://admin:password@192.168.1.204:554/cam/realmonitor?channel=1&subtype=0",
        "enabled": True,
    },
    # ... thêm cameras khác
]
```

### 2. Chạy Backend
```bash
cd native-ai-backend
python main.py
```

### 3. Xem Stream với Tracking
Frontend sẽ nhận WebSocket stream với tracking đã được vẽ trực tiếp:

```
ws://localhost:5000/ws/tracking/cam01
```

### 4. Xem Tracking Logs
Logs được lưu tự động trong `storage/tracking_logs/`:
- `person_{id}_{timestamp}.json` - Chi tiết từng người
- `tracking_summary.csv` - Tổng hợp tất cả

## Thông Tin Hiển Thị Trên Stream

### Trên Mỗi Bounding Box:
- **ID:{số}** - Global person ID (giữ nguyên khi di chuyển giữa cameras)
- **[TRK/LST/CLT]** - Trạng thái: Tracking/Lost/Confirmed Lost
- **Confidence score**

### Trên Frame:
- Camera ID
- FPS hiện tại
- Số tracks đang active
- Số persons đã identify
- Thống kê lifecycle (active/archived)

### Màu Sắc:
- **Màu cố định cho mỗi người** - Để dễ theo dõi
- **Màu mờ đi** - Khi người ở trạng thái LOST
- **Màu xám** - Khi CONFIRMED_LOST
- **Vàng** - Tracks chưa identify

## API Endpoints

### WebSocket Stream
```
GET ws://localhost:5000/ws/tracking/{camera_id}
```
Nhận stream video đã có tracking annotations

### REST API
```
GET /api/cameras
GET /api/persons
GET /api/stats         # Bao gồm lifecycle stats
GET /api/persons/{id}
```

## Ưu Điểm So Với Trước

1. **Stream Có Tracking Trực Tiếp**
   - Frontend không cần xử lý gì thêm
   - Chỉ cần hiển thị JPEG frames từ WebSocket
   - Tracking annotations đã được vẽ sẵn

2. **RTSP Native Support**
   - Không cần qua file reader
   - Latency thấp với buffer=1
   - Auto-reconnect khi mất kết nối

3. **Lifecycle Management**
   - Track được vòng đời của mỗi người
   - Không tạo duplicate IDs
   - Time window matching thông minh
   - Auto cleanup old tracks

4. **Visualization Tốt Hơn**
   - Màu sắc nhất quán
   - Hiển thị states rõ ràng
   - Thống kê real-time
   - Easy to debug

## Troubleshooting

### 1. RTSP không connect được
- Kiểm tra RTSP URL đúng format
- Test bằng VLC: `vlc rtsp://...`
- Kiểm tra firewall/network
- Xem logs: `[RTSPStreamLoader:{cam_id}]`

### 2. FPS thấp
- Giảm `INPUT_WIDTH/HEIGHT` trong config
- Tăng `DETECTION_SKIP_FRAMES` (detect ít hơn)
- Giảm `OUTPUT_FPS` cho WebSocket
- Disable một số cameras

### 3. Person IDs bị duplicate
- Tăng `TIME_WINDOW_SECONDS` (matching window)
- Giảm `REID_THRESHOLD` (strict hơn)
- Kiểm tra lighting/angle của cameras

### 4. Memory leak
- Giảm `TRACK_BUFFER` và `MAX_GALLERY_SIZE`
- Kiểm tra `ARCHIVE_AFTER_SECONDS` không quá lớn
- Monitor lifecycle stats: `/api/stats`

## So Sánh với MultiCamera Project

| Feature | MultiCamera | Native-AI-Backend |
|---------|-------------|-------------------|
| RTSP Loader | ✅ RTSPStreamLoader | ✅ Integrated |
| Lifecycle Manager | ✅ Full featured | ✅ Adapted |
| Time Window | ✅ 3s default | ✅ Configurable |
| Camera Topology | ✅ Yes | ⚠️ Can add |
| WebSocket Stream | ❌ No | ✅ Yes |
| REST API | ❌ No | ✅ Yes |
| Person Database | ❌ Dict only | ✅ Persistent |
| Frontend | ✅ Next.js | ✅ Next.js |

## Next Steps (Tương Lai)

1. **Camera Topology** - Thêm logic để biết cameras nào kết nối với nhau
2. **Person Events** - Trigger events khi person enter/exit zones
3. **Recording** - Lưu clips khi có events quan trọng
4. **Analytics Dashboard** - Thống kê người qua lại theo thời gian
5. **Face Recognition** - Thêm face matching để identify cụ thể

## Kiểm Tra Hoạt Động

```bash
# Terminal 1: Start backend
cd native-ai-backend
python main.py

# Terminal 2: Test WebSocket
wscat -c ws://localhost:5000/ws/tracking/cam01

# Terminal 3: Check stats
curl http://localhost:5000/api/stats | python -m json.tool
```

---

**Lưu Ý:** Frontend hiện tại đã sẵn sàng nhận và hiển thị stream. Chỉ cần chạy backend với RTSP URLs đúng.
