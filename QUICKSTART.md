# 🚀 QUICKSTART GUIDE

Get up and running in 5 minutes!

---

## 📦 Step 1: Setup Environment

**Windows:**
```powershell
cd native-ai-backend
.\setup.ps1
```

**Linux:**
```bash
cd native-ai-backend
bash setup.sh
```

This will:
- Create virtual environment
- Install dependencies
- Copy AI models
- Create storage directory

---

## ⚙️ Step 2: Configure Cameras

Edit `config.py`:

```python
CAMERAS = [
    {
        "id": "cam01",
        "name": "Camera 1",
        "rtsp_url": "rtsp://admin:password@192.168.1.101:554/stream",
        "enabled": True,
    },
    # Add your cameras here
]
```

**Don't have cameras?** Use test videos:

```powershell
# Windows
$env:USE_VIDEO_FILES="1"
$env:VIDEO_DIR="./videos"

# Linux  
export USE_VIDEO_FILES=1
export VIDEO_DIR=./videos
```

Then put video files in `videos/` folder named: `video1.mp4`, `video2.mp4`, etc.

---

## 🏃 Step 3: Run Server

**Windows:**
```powershell
.\run.ps1
```

**Linux:**
```bash
bash run.sh
```

**Or manually:**
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
[Manager] Started 3 cameras

Server ready on http://0.0.0.0:5000
============================================================
```

---

## 🎨 Step 4: Connect Frontend

### Option A: Use Existing Frontend

```bash
cd ../ab-camera-loyalty/Front-end

# Edit .env.local
echo "NEXT_PUBLIC_BACKEND_URL=http://localhost:5000" > .env.local

# Run
npm install
npm run dev
```

Open: http://localhost:3000

### Option B: Test with Browser

Open: http://localhost:5000

You'll see API documentation.

### Option C: Test with WebSocket

Create `test.html`:

```html
<!DOCTYPE html>
<html>
<head><title>Test Camera Stream</title></head>
<body>
    <h1>Camera Stream</h1>
    <img id="camera" style="width:100%; max-width:1280px;">
    
    <script>
        const ws = new WebSocket('ws://localhost:5000/ws/tracking/cam01');
        ws.binaryType = 'arraybuffer';
        
        ws.onmessage = (event) => {
            const blob = new Blob([event.data], { type: 'image/jpeg' });
            const url = URL.createObjectURL(blob);
            document.getElementById('camera').src = url;
        };
        
        ws.onopen = () => console.log('Connected!');
        ws.onerror = (e) => console.error('Error:', e);
    </script>
</body>
</html>
```

Open in browser and you'll see live stream!

---

## 📊 Step 5: Check Status

### REST API

```bash
# Health check
curl http://localhost:5000/health

# List cameras
curl http://localhost:5000/api/cameras

# Get statistics
curl http://localhost:5000/api/stats

# List identified persons
curl http://localhost:5000/api/persons
```

### WebSocket Endpoints

- `ws://localhost:5000/ws/tracking/cam01` - Camera 1 stream
- `ws://localhost:5000/ws/tracking/cam02` - Camera 2 stream
- `ws://localhost:5000/ws/tracking/cam03` - Camera 3 stream

---

## 🐛 Troubleshooting

### 1. "CUDA not available"

**Solution:**
- Check: `nvidia-smi`
- Install CUDA 11.8+
- Reinstall: `pip install --force-reinstall onnxruntime-gpu`

### 2. "Failed to open stream"

**Solution:**
- Test RTSP URL with VLC: `vlc rtsp://...`
- Check camera credentials
- Check network connectivity

### 3. Low FPS / Lag

**Solution:**
Edit `config.py`:
```python
DETECTION_SKIP_FRAMES = 3  # Increase (was 2)
INPUT_WIDTH = 960          # Decrease (was 1280)
OUTPUT_FPS = 10            # Decrease (was 15)
```

### 4. Out of memory

**Solution:**
- Close other GPU applications
- Reduce number of cameras
- Lower resolution in config.py

---

## 🎯 Performance Tips

### For RTX 3050 8GB (recommended settings):

```python
# config.py
DETECTION_SKIP_FRAMES = 2   # Detect every 2 frames
INPUT_WIDTH = 1280          # 720p input
INPUT_HEIGHT = 720
OUTPUT_FPS = 15             # 15 FPS to frontend
JPEG_QUALITY = 80           # Good quality
```

**Expected Performance:**
- Latency: 55-95ms
- RAM Usage: ~400MB
- VRAM Usage: ~600MB
- CPU Usage: ~30%
- FPS: 28-30 per camera

### For Lower-end GPU:

```python
DETECTION_SKIP_FRAMES = 4   # Detect every 4 frames
INPUT_WIDTH = 960           # Lower resolution
INPUT_HEIGHT = 540
OUTPUT_FPS = 10             # Lower output FPS
```

---

## 📁 Quick Reference

### Project Structure

```
native-ai-backend/
├── main.py              # Run this!
├── config.py            # Edit cameras here
├── requirements.txt     
│
├── models/              # AI models (ONNX)
├── storage/             # Person database
│
├── core/                # Processing logic
├── detection/           # YOLO + tracking
├── reid/                # Person re-ID
└── api/                 # FastAPI + WebSocket
```

### Key Files

- **config.py** - Configure cameras, models, settings
- **main.py** - FastAPI server (run this)
- **storage/persons.json** - Person database
- **README.md** - Full documentation

### Useful Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Linux
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Run
python main.py

# With video files
export USE_VIDEO_FILES=1  # Linux
$env:USE_VIDEO_FILES=1    # Windows
python main.py

# Debug mode
export ENV=development    # Linux
$env:ENV="development"    # Windows
python main.py
```

---

## ✅ Checklist

Before running, make sure:

- [ ] Python 3.9-3.11 installed
- [ ] NVIDIA driver installed (if using GPU)
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Models in `models/` folder
- [ ] `config.py` edited with camera URLs
- [ ] Cameras accessible (or video files prepared)

---

## 🎉 You're Ready!

```bash
python main.py
```

Then open:
- Backend API: http://localhost:5000
- Frontend: http://localhost:3000 (if running)
- WebSocket: ws://localhost:5000/ws/tracking/cam01

**Happy tracking!** 🚀

---

## 📚 Next Steps

1. **Read README.md** for full documentation
2. **Tune config.py** for your hardware
3. **Connect frontend** for visualization
4. **Check API docs** at http://localhost:5000/docs

For detailed info, see **README.md**.
