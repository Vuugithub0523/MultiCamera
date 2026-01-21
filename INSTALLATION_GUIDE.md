# 📦 HƯỚNG DẪN CÀI ĐẶT CHI TIẾT - NATIVE AI BACKEND

## 📋 YÊU CẦU HỆ THỐNG

### **Phần cứng:**
- ✅ **GPU:** NVIDIA GPU với CUDA 12.1+ (khuyến nghị: GTX 1660 trở lên)
- ✅ **RAM:** 8GB minimum, 16GB khuyến nghị
- ✅ **Storage:** 10GB free space
- ✅ **Network:** Kết nối đến IP cameras qua RTSP

### **Phần mềm:**
- ✅ **OS:** Windows 10/11 hoặc Linux
- ✅ **Python:** 3.9, 3.10, hoặc 3.11 (khuyến nghị 3.10)
- ✅ **CUDA Toolkit:** 12.1+ (nếu dùng GPU)
- ✅ **Node.js:** 18+ (cho frontend)

---

## 🔧 BƯỚC 1: KIỂM TRA MÔI TRƯỜNG

### **Kiểm tra Python:**
```bash
python --version
# Output: Python 3.10.x
```

### **Kiểm tra CUDA (nếu có GPU):**
```bash
nvidia-smi
# Kiểm tra CUDA Version >= 12.1
```

### **Kiểm tra pip:**
```bash
pip --version
python -m pip install --upgrade pip
```

---

## 📥 BƯỚC 2: CÀI ĐẶT DEPENDENCIES

### **⚠️ QUAN TRỌNG: Thứ tự cài đặt**

PyTorch GPU sử dụng index riêng, **KHÔNG THỂ** cài bằng lệnh `pip install -r requirements.txt` thông thường.

### **Bước 2.1: Cài PyTorch GPU (nếu có GPU)**

```bash
# Cài PyTorch với CUDA 12.1
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**Lưu ý:**
- Nếu có CUDA 11.8, thay `cu121` bằng `cu118`
- Nếu chỉ dùng CPU, dùng: `pip install torch torchvision torchaudio`

### **Bước 2.2: Cài các dependencies còn lại**

```bash
# Vào thư mục backend
cd d:\TTTN_AntBuddy\native-ai-backend

# Cài tất cả dependencies
pip install -r requirements.txt
```

### **Kiểm tra cài đặt:**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}'); print(f'Providers: {onnxruntime.get_available_providers()}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

**Output mong đợi:**
```
PyTorch: 2.5.1+cu121
CUDA available: True
ONNX Runtime: 1.23.2
Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
OpenCV: 4.8.1
```

---

## 📁 BƯỚC 3: DOWNLOAD MODELS

### **Models cần thiết:**

1. **YOLOv4-Tiny** (`yolov4-tiny.onnx`) - ~25MB
   - Object detection model
   - Phát hiện người trong frame

2. **OSNet** (`osnet_ain_x1_0_M.onnx`) - ~10MB
   - Re-ID feature extraction model
   - Nhận diện người xuyên camera

3. **COCO Names** (`coco.names`)
   - Class names file
   - 80 object classes

### **Cách download:**

#### **Option 1: Tự động (khuyến nghị)**
```bash
# Nếu có script download
python download_model.py
```

#### **Option 2: Thủ công**

**YOLOv4-Tiny:**
```bash
# Download từ GitHub releases hoặc Google Drive
# Đặt vào: native-ai-backend/models/yolov4-tiny.onnx
```

**OSNet:**
```bash
# Download từ model zoo
# Đặt vào: native-ai-backend/models/osnet_ain_x1_0_M.onnx
```

**COCO Names:**
```bash
# Download từ repository
# https://github.com/pjreddie/darknet/blob/master/data/coco.names
# Đặt vào: native-ai-backend/models/coco.names
```

#### **Option 3: Dùng gdown (từ Google Drive)**
```bash
pip install gdown

# YOLOv4-Tiny
gdown <google_drive_id> -O models/yolov4-tiny.onnx

# OSNet
gdown <google_drive_id> -O models/osnet_ain_x1_0_M.onnx
```

### **Kiểm tra models:**
```bash
ls -lh models/
# Output:
# yolov4-tiny.onnx         (~25MB)
# osnet_ain_x1_0_M.onnx    (~10MB)
# coco.names               (~1KB)
```

---

## ⚙️ BƯỚC 4: CẤU HÌNH HỆ THỐNG

### **Chỉnh sửa `config.py`:**

```python
# 1. Cấu hình cameras
CAMERAS = [
    {
        "id": "cam01",
        "name": "Camera 01 - Entrance",
        "rtsp_url": "rtsp://username:password@192.168.1.100:554/stream",
        "enabled": True,
    },
    # Thêm cameras khác...
]

# 2. Cấu hình device
DEVICE = "cuda"  # Đổi thành "cpu" nếu không có GPU

# 3. Cấu hình server
HOST = "0.0.0.0"  # Listen trên tất cả interfaces
PORT = 5000       # Backend port

# 4. Cấu hình detection
DETECTION_CONFIDENCE = 0.5  # Confidence threshold
DETECTION_SKIP_FRAMES = 2   # Detect every 2 frames

# 5. Cấu hình output
OUTPUT_FPS = 15      # WebSocket stream FPS
JPEG_QUALITY = 80    # JPEG compression quality
```

### **Kiểm tra RTSP URLs:**

```bash
# Test RTSP stream với ffmpeg
ffmpeg -i "rtsp://username:password@192.168.1.100:554/stream" -frames:v 1 test.jpg
```

---

## 🚀 BƯỚC 5: CHẠY BACKEND

### **Cách 1: Python trực tiếp**
```bash
cd d:\TTTN_AntBuddy\native-ai-backend
python main.py
```

### **Cách 2: Uvicorn**
```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

### **Cách 3: PowerShell script (Windows)**
```powershell
.\run.ps1
```

### **Output mong đợi:**
```
============================================================
NATIVE AI BACKEND - Multi-Camera Tracking System
============================================================

[Manager] Loading shared models...
[YOLODetector] Loaded model: models/yolov4-tiny.onnx
[YOLODetector] Input size: 416x416
[YOLODetector] Device: cuda
[YOLODetector] Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
[FeatureExtractor] Loaded model: models/osnet_ain_x1_0_M.onnx
[FeatureExtractor] Input size: 256x128
[FeatureExtractor] Device: cuda

[Startup] Server ready on http://0.0.0.0:5000
[Startup] Cameras: cam01, cam02, cam03
[Startup] WebSocket: ws://0.0.0.0:5000/ws/tracking/{camera_id}
============================================================
```

### **Kiểm tra backend:**

```bash
# REST API
curl http://localhost:5000/api/status

# WebSocket (dùng browser)
# ws://localhost:5000/ws/tracking/cam01
```

---

## 🎨 BƯỚC 6: CHẠY FRONTEND

### **Cài đặt dependencies:**
```bash
cd d:\TTTN_AntBuddy\native-ai-backend\frontend
npm install
```

### **Cấu hình backend URL:**

Tạo file `.env.local`:
```bash
NEXT_PUBLIC_BACKEND_URL=http://localhost:5000
```

### **Chạy development server:**
```bash
npm run dev
```

### **Output:**
```
▲ Next.js 14.x
- Local:        http://localhost:3000
- Network:      http://192.168.1.x:3000
```

### **Truy cập frontend:**
```
http://localhost:3000
```

---

## 🧪 BƯỚC 7: KIỂM TRA HỆ THỐNG

### **Test 1: Backend Health Check**
```bash
curl http://localhost:5000/api/status
```

**Output:**
```json
{
  "status": "running",
  "cameras": ["cam01", "cam02"],
  "uptime": 123.45
}
```

### **Test 2: WebSocket Stream**

Mở file `test_websocket.html` trong browser:
```bash
# Windows
start test_websocket.html

# Linux
xdg-open test_websocket.html
```

### **Test 3: Frontend Connection**

1. Mở `http://localhost:3000`
2. Chọn camera từ dropdown
3. Kiểm tra video stream hiển thị
4. Kiểm tra FPS counter

### **Test 4: Performance Check**

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor CPU/Memory
htop  # Linux
# hoặc Task Manager (Windows)
```

---

## 🐛 XỬ LÝ LỖI THƯỜNG GẶP

### **Lỗi 1: CUDA not available**

**Triệu chứng:**
```
CUDA available: False
WARNING: No GPU provider available, using CPU
```

**Giải pháp:**
```bash
# Kiểm tra CUDA installation
nvidia-smi

# Cài lại PyTorch với CUDA
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### **Lỗi 2: RTSP connection timeout**

**Triệu chứng:**
```
[RTSP] Failed to connect to camera
```

**Giải pháp:**
```python
# config.py
RTSP_TIMEOUT = 30  # Tăng timeout
RTSP_BUFFER_SIZE = 2  # Tăng buffer

# Test RTSP URL
ffmpeg -i "rtsp://..." -frames:v 1 test.jpg
```

### **Lỗi 3: Model not found**

**Triệu chứng:**
```
FileNotFoundError: models/yolov4-tiny.onnx
```

**Giải pháp:**
```bash
# Kiểm tra models folder
ls -lh models/

# Download models nếu thiếu
python download_model.py
```

### **Lỗi 4: WebSocket connection failed**

**Triệu chứng:**
```
[WebSocket] Error: Connection refused
```

**Giải pháp:**
```bash
# Kiểm tra backend đang chạy
curl http://localhost:5000/api/status

# Kiểm tra port không bị block
netstat -an | grep 5000

# Kiểm tra CORS settings trong main.py
```

### **Lỗi 5: Out of memory (GPU)**

**Triệu chứng:**
```
CUDA out of memory
```

**Giải pháp:**
```python
# config.py
DETECTION_SKIP_FRAMES = 3  # Skip nhiều frames hơn
OUTPUT_FPS = 10            # Giảm output FPS

# Giảm số cameras enabled
CAMERAS = [
    {..., "enabled": True},   # Chỉ enable 1-2 cameras
    {..., "enabled": False},
]
```

### **Lỗi 6: Low FPS**

**Triệu chứng:**
```
FPS < 10
```

**Giải pháp:**
```python
# config.py
DETECTION_SKIP_FRAMES = 3    # Skip nhiều frames
INPUT_WIDTH = 640            # Giảm resolution
INPUT_HEIGHT = 360
JPEG_QUALITY = 70            # Giảm quality

# Dùng YOLO tiny thay vì full
# Disable re-ID nếu không cần
```

---

## 📊 MONITORING & LOGS

### **Log locations:**
```
native-ai-backend/
  └─ storage/
      ├─ persons.json           # Person database
      └─ tracking_logs/
          └─ tracking_summary.csv  # Tracking history
```

### **View logs:**
```bash
# Backend logs (terminal output)
python main.py | tee backend.log

# Tracking logs
cat storage/tracking_logs/tracking_summary.csv
```

### **Monitoring dashboard:**
```
http://localhost:3000/report
```

---

## 🎯 TỔNG KẾT CHECKLIST

### **Pre-installation:**
- [ ] Python 3.9-3.11 installed
- [ ] CUDA 12.1+ installed (nếu dùng GPU)
- [ ] pip updated
- [ ] Git installed

### **Installation:**
- [ ] PyTorch GPU installed (bước riêng)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Models downloaded (YOLO, OSNet, COCO names)
- [ ] Frontend dependencies installed (`npm install`)

### **Configuration:**
- [ ] RTSP URLs configured in `config.py`
- [ ] Device set to "cuda" or "cpu"
- [ ] Backend URL set in frontend `.env.local`

### **Testing:**
- [ ] Backend starts without errors
- [ ] GPU/CUDA available (nếu dùng GPU)
- [ ] RTSP streams connected
- [ ] WebSocket connection works
- [ ] Frontend displays video streams
- [ ] FPS > 15

### **Optimization:**
- [ ] GPU usage monitored
- [ ] Detection skip frames tuned
- [ ] Output FPS optimized
- [ ] JPEG quality adjusted

---

## 📚 TÀI LIỆU THAM KHẢO

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Next.js Documentation](https://nextjs.org/docs)

---

## 💡 TIPS & TRICKS

### **Performance:**
1. Dùng GPU nếu có (tăng tốc 5-10x)
2. Skip frames detection (giảm load)
3. Giảm resolution input nếu cần
4. Dùng JPEG compression hợp lý

### **Stability:**
1. Set RTSP timeout hợp lý
2. Handle connection errors gracefully
3. Monitor GPU memory usage
4. Use async processing

### **Development:**
1. Use `--reload` với uvicorn
2. Enable debug logs
3. Test with video files trước khi dùng RTSP
4. Monitor logs real-time

---

**Chúc bạn cài đặt thành công! 🚀**

**Support:** AntBuddy Team  
**Updated:** January 21, 2026
