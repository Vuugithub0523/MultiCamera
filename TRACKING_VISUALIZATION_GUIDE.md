# 🎯 TRACKING VISUALIZATION - HƯỚNG DẪN CHI TIẾT

## ✅ XÁC NHẬN: VIDEO ĐÃ CÓ TRACKING VISUALIZATION

**Hệ thống của bạn ĐÃ được thiết kế để hiển thị tracking đầy đủ!** 

Backend đã tích hợp sẵn visualization trong pipeline xử lý frame. Mỗi frame được gửi qua WebSocket đã bao gồm:

✅ **Bounding boxes** màu sắc cho mỗi người  
✅ **Person IDs** (ID:1, ID:2, ...)  
✅ **Track states** ([DET], [TRK], [LST])  
✅ **Confidence scores** (0.85)  
✅ **Camera info & statistics**

---

## 📊 CÁC THÔNG TIN TRACKING HIỂN THỊ

### **1. Bounding Boxes với Màu Sắc**

```python
# Code: core/pipeline.py - Line 255-265
for track in tracks:
    x, y, w, h = track.bbox
    
    # Mỗi person được assign màu riêng (persistent)
    if track.person_id in self.person_colors:
        color = self.person_colors[track.person_id]
    else:
        color = (0, 255, 0)  # Default green
    
    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
```

**Hiển thị:**
- Mỗi person có màu riêng (random RGB)
- Màu được giữ nguyên xuyên suốt tracking session
- Thickness = 3 cho person đã identify, 2 cho unknown

**Màu theo state:**
- `ACTIVE/TRACKING`: Màu gốc (bright)
- `LOST`: Màu tối hơn (color // 2)
- `CONFIRMED_LOST`: Màu xám (128, 128, 128)
- `Unknown track`: Màu vàng (0, 255, 255)

---

### **2. Person ID Labels**

```python
# Code: core/pipeline.py - Line 275-285
if track.person_id:
    label = f"ID:{track.person_id}"
    if track.state:
        state_short = {
            'detected': 'DET',
            'tracking': 'TRK',
            'lost': 'LST',
            'confirmed_lost': 'CLT'
        }.get(track.state, track.state[:3].upper())
        label += f" [{state_short}]"
else:
    label = f"T{track.track_id}"  # Track ID nếu chưa có person ID
```

**Format hiển thị:**
```
ID:1 [TRK] 0.87
│    │     └─ Confidence score
│    └─────── State abbreviation
└────────────── Person global ID
```

**States:**
- `[DET]` - **Detected**: Mới phát hiện
- `[TRK]` - **Tracking**: Đang track ổn định
- `[LST]` - **Lost**: Mất track tạm thời (<30 frames)
- `[CLT]` - **Confirmed Lost**: Mất lâu (>90 frames)

---

### **3. Label với Background**

```python
# Code: core/pipeline.py - Line 295-310
# Draw label background (colored box)
cv2.rectangle(
    annotated,
    (label_x, label_y - label_h - baseline),
    (label_x + label_w, label_y + baseline),
    color,  # Same color as bbox
    -1      # Fill
)

# Draw label text (black for visibility)
cv2.putText(
    annotated,
    label,
    (label_x, label_y - baseline),
    font,
    font_scale,
    (0, 0, 0),  # Black text
    font_thickness
)
```

**Hiển thị:**
- Label nằm phía trên bounding box
- Background cùng màu với bbox
- Text màu đen để dễ đọc

---

### **4. Camera Info & Stats (Top-Left)**

```python
# Code: core/pipeline.py - Line 315-330
info_lines = [
    f"Camera: {self.camera_id}",        # Camera ID
    f"FPS: {self.stats['fps']:.1f}",    # Processing FPS
    f"Tracks: {len(tracks)}",            # Active tracks
    f"Persons: {self.stats['persons_identified']}"  # Total persons
]

y_offset = 25
for line in info_lines:
    cv2.putText(
        annotated,
        line,
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),  # Green text
        2
    )
    y_offset += 25
```

**Hiển thị (ví dụ):**
```
Camera: cam01
FPS: 28.5
Tracks: 3
Persons: 15
```

---

### **5. Lifecycle Stats (Bottom)**

```python
# Code: core/pipeline.py - Line 335-345
lifecycle_stats = self.lifecycle_manager.get_stats()
lifecycle_text = f"Active: {lifecycle_stats['total_active']} | Archived: {lifecycle_stats['total_archived']}"
cv2.putText(
    annotated,
    lifecycle_text,
    (10, frame.shape[0] - 15),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 255, 0),  # Yellow text
    1
)
```

**Hiển thị:**
```
Active: 5 | Archived: 23
```

---

## 🔄 LUỒNG PROCESSING VÀ VISUALIZATION

```
1. RTSP Frame Input
   ↓
2. YOLO Detection → Bounding boxes
   ↓
3. ByteTracker → Track IDs
   ↓
4. Feature Extraction → 512-dim vectors
   ↓
5. Person Database → Global Person IDs
   ↓
6. Lifecycle Manager → Person States
   ↓
7. ⭐ VISUALIZATION (pipeline._draw_annotations)
   │
   ├─ Draw bounding boxes (colored)
   ├─ Draw labels (ID + State + Confidence)
   ├─ Draw camera info (top-left)
   └─ Draw lifecycle stats (bottom)
   ↓
8. JPEG Encoding
   ↓
9. WebSocket Broadcast → Frontend
   ↓
10. Frontend Display (img tag)
```

**Điểm quan trọng:**
- Visualization xảy ra **TRƯỚC** khi encode JPEG
- Frame được gửi qua WebSocket **ĐÃ BAO GỒM** tracking info
- Frontend chỉ cần hiển thị JPEG, không cần vẽ thêm gì

---

## 🧪 CÁCH KIỂM TRA TRACKING VISUALIZATION

### **Option 1: Test HTML File (Khuyến nghị)**

1. **Chạy backend:**
```bash
cd d:\TTTN_AntBuddy\native-ai-backend
python main.py
```

2. **Mở test file trong browser:**
```bash
# Windows
start test_tracking_visualization.html

# Hoặc double-click file
```

3. **Click "Kết nối"** - Video stream sẽ hiển thị với tracking visualization

4. **Kiểm tra:**
   - ✅ Bounding boxes có màu sắc
   - ✅ Person IDs hiển thị (ID:1, ID:2, ...)
   - ✅ States hiển thị ([TRK], [LST], ...)
   - ✅ Confidence scores
   - ✅ Camera info (top-left)
   - ✅ Lifecycle stats (bottom)

### **Option 2: Frontend (Next.js)**

```bash
cd d:\TTTN_AntBuddy\native-ai-backend\frontend
npm run dev
```

Truy cập: `http://localhost:3000`

### **Option 3: Direct WebSocket Test**

```javascript
const ws = new WebSocket('ws://localhost:5000/ws/tracking/cam01');
ws.binaryType = 'arraybuffer';

ws.onmessage = (event) => {
    const blob = new Blob([event.data], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    document.getElementById('video').src = url;
};
```

---

## 🎨 VÍ DỤ VISUALIZATION

### **Frame với 3 người được track:**

```
┌──────────────────────────────────────────────────────┐
│ Camera: cam01                                        │
│ FPS: 28.5                                           │
│ Tracks: 3                                           │
│ Persons: 15                                         │
│                                                      │
│   ┌─────────────────┐                              │
│   │ ID:1 [TRK] 0.92 │  <- Green box                │
│   └─────────────────┘                              │
│   ┌─────────────────────┐                          │
│   │                     │                          │
│   │    Person 1        │                          │
│   │                     │                          │
│   └─────────────────────┘                          │
│                                                      │
│                  ┌─────────────────┐               │
│                  │ ID:5 [TRK] 0.87 │  <- Blue box  │
│                  └─────────────────┘               │
│                  ┌─────────────────────┐           │
│                  │                     │           │
│                  │    Person 5        │           │
│                  │                     │           │
│                  └─────────────────────┘           │
│                                                      │
│                          ┌─────────────────┐       │
│                          │ ID:3 [LST] 0.65 │ <- Dim│
│                          └─────────────────┘       │
│                          ┌─────────────────────┐   │
│                          │                     │   │
│                          │    Person 3 (Lost) │   │
│                          │                     │   │
│                          └─────────────────────┘   │
│                                                      │
│ Active: 3 | Archived: 12                           │
└──────────────────────────────────────────────────────┘
```

---

## 🐛 NẾU KHÔNG THẤY TRACKING

### **Triệu chứng:** Video stream chỉ có frame trống, không có boxes

**Nguyên nhân có thể:**

1. **Backend chưa chạy hoặc crashed**
   ```bash
   # Kiểm tra
   curl http://localhost:5000/api/status
   
   # Restart
   python main.py
   ```

2. **Camera không có người (YOLO không detect)**
   - Di chuyển trước camera
   - Kiểm tra RTSP stream có hoạt động không
   - Giảm `DETECTION_CONFIDENCE` trong config.py

3. **Detection confidence quá cao**
   ```python
   # config.py
   DETECTION_CONFIDENCE = 0.3  # Giảm từ 0.5 xuống 0.3
   ```

4. **Models chưa load đúng**
   ```bash
   # Kiểm tra models folder
   ls -lh models/
   
   # Phải có:
   # yolov4-tiny.onnx
   # osnet_ain_x1_0_M.onnx
   # coco.names
   ```

5. **GPU out of memory**
   ```python
   # config.py
   DEVICE = "cpu"  # Chuyển sang CPU để test
   ```

---

## 🔍 DEBUG TRACKING VISUALIZATION

### **1. Kiểm tra Pipeline Processing**

```python
# Thêm debug logs vào core/pipeline.py

async def process_frame(self, frame: np.ndarray):
    # ... existing code ...
    
    # After tracking
    print(f"[{self.camera_id}] Tracks: {len(tracks)}, Track IDs: {[t.track_id for t in tracks]}")
    
    # After person matching
    print(f"[{self.camera_id}] Persons: {[t.person_id for t in track_infos if t.person_id]}")
    
    # After visualization
    print(f"[{self.camera_id}] Annotated frame shape: {annotated.shape}")
```

### **2. Save Frame to File (Debug)**

```python
# Thêm vào core/pipeline.py - sau visualization

# Save annotated frame for debugging
if self.frame_count % 30 == 0:  # Every 30 frames
    cv2.imwrite(f"debug_{self.camera_id}_{self.frame_count}.jpg", annotated)
    print(f"[DEBUG] Saved frame: debug_{self.camera_id}_{self.frame_count}.jpg")
```

### **3. Kiểm tra WebSocket Data**

```javascript
// Browser console
ws.onmessage = (event) => {
    console.log('Frame size:', event.data.byteLength);
    
    // Check if it's a valid JPEG
    const arr = new Uint8Array(event.data);
    console.log('JPEG header:', arr[0], arr[1]); // Should be 255, 216
};
```

---

## 📸 SCREENSHOT VÍ DỤ

**Expected output:**

![Expected Tracking Visualization](https://via.placeholder.com/800x600/000000/00ff00?text=Camera+Feed+with+Tracking)

**Bao gồm:**
- ✅ Colored bounding boxes
- ✅ Person IDs above boxes
- ✅ State indicators [TRK], [LST]
- ✅ Confidence scores
- ✅ Camera info (green, top-left)
- ✅ Lifecycle stats (yellow, bottom)

---

## 🚀 CÁCH TỐI ƯU VISUALIZATION

### **1. Tăng FPS**
```python
# config.py
OUTPUT_FPS = 20  # Tăng từ 15 lên 20
```

### **2. Giảm Latency**
```python
# config.py
RTSP_BUFFER_SIZE = 1  # Minimum buffer
DETECTION_SKIP_FRAMES = 1  # Detect mọi frame (nếu GPU mạnh)
```

### **3. Tùy chỉnh Visualization**

```python
# core/pipeline.py - _draw_annotations()

# Thay đổi font size
font_scale = 0.8  # Tăng từ 0.6

# Thay đổi thickness
thickness = 4  # Tăng từ 3

# Thay đổi màu info text
cv2.putText(..., (255, 255, 255), ...)  # White thay vì green
```

---

## ✅ CHECKLIST TRACKING VISUALIZATION

- [ ] Backend đang chạy (`python main.py`)
- [ ] Models đã download (YOLO + OSNet)
- [ ] Camera RTSP connected
- [ ] YOLO đang detect người (`Tracks > 0`)
- [ ] Person IDs được assign
- [ ] Bounding boxes hiển thị với màu
- [ ] Labels hiển thị (ID + State + Conf)
- [ ] Camera info hiển thị (top-left)
- [ ] Lifecycle stats hiển thị (bottom)
- [ ] WebSocket stream mượt (FPS > 15)

---

## 🎯 TÓM TẮT

**Hệ thống của bạn ĐÃ SẴN SÀNG hiển thị tracking visualization!**

✅ Code đã implement đầy đủ  
✅ Visualization được vẽ trước khi encode JPEG  
✅ WebSocket gửi frame đã có tracking info  
✅ Frontend chỉ cần hiển thị JPEG

**Để test ngay:**
1. Chạy backend: `python main.py`
2. Mở `test_tracking_visualization.html` trong browser
3. Click "Kết nối"
4. Xem video với tracking boxes, IDs, và stats! 🎉

---

**Tác giả:** AntBuddy AI Team  
**Cập nhật:** January 21, 2026  
**File test:** [test_tracking_visualization.html](test_tracking_visualization.html)
