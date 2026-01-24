# 📋 Quick Reference

## 🚀 Start Commands

### Backend Server
```bash
cd backend-server
python server.py
# or: start-backend.bat (Windows) / ./start-backend.sh (Linux/Mac)
```

### Frontend
```bash
cd frontend
npm run dev
# or: start-frontend.bat (Windows) / ./start-frontend.sh (Linux/Mac)
```

## 🔗 URLs

- **Backend API**: http://localhost:8080
- **Frontend**: http://localhost:3000
- **Backend Health**: http://localhost:8080/health
- **API Docs**: http://localhost:8080/docs

## 📁 Important Files

| File | Purpose |
|------|---------|
| `config.yaml` | Main configuration |
| `backend-server/server.py` | FastAPI application |
| `ai-service/core/object_detection.py` | YOLO detector |
| `ai-service/core/feature_extraction.py` | OSNet Re-ID |
| `ai-service/core/person_lifecycle_manager.py` | Tracking logic |

## 🔧 Installation

### AI Service
```bash
cd ai-service
pip install -r requirements.txt
```

### Backend Server
```bash
cd backend-server
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/api/cameras` | List cameras |
| GET | `/api/tracking/stats` | Tracking statistics |
| WS | `/ws/{type}/{cam}` | Video stream |

## 🎥 WebSocket Streams

```
ws://localhost:8080/ws/raw/cam01      # Raw camera feed
ws://localhost:8080/ws/tracking/cam01  # With tracking overlay
```

## ⚙️ Config.yaml Keys

```yaml
# Cameras
rtsp_urls: [...]

# Models
object_detection_model_path: "..."
feature_extraction_model_path: "..."

# Detection
object_detection_threshold: 0.7
inference_model_device: "cuda"

# Tracking
feature_extraction_threshold: 0.42
time_window_seconds: 3.0
max_lost_frames: 30
```

## 🐛 Common Issues

### "Module not found"
```bash
# Make sure you're in the right directory
cd backend-server
pip install -r requirements.txt
```

### "Port already in use"
```bash
# Change port in server.py or kill existing process
# Windows: netstat -ano | findstr :8080
# Linux: lsof -i :8080
```

### "CUDA not available"
```yaml
# In config.yaml, change:
inference_model_device: "cpu"
```

## 📚 Documentation

- [README.md](README.md) - Main docs
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Architecture
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Migration help
- [ai-service/README.md](ai-service/README.md) - AI docs
- [backend-server/README.md](backend-server/README.md) - Backend docs

## 🔍 Debugging

### Check Backend Status
```bash
curl http://localhost:8080/health
```

### Check Cameras
```bash
curl http://localhost:8080/api/cameras
```

### View Logs
- Backend: Terminal output
- Frontend: Browser console
- Tracking: `tracking_logs/` directory

## 🎯 Next Steps

1. ✅ Install dependencies for all services
2. ✅ Configure RTSP URLs in `config.yaml`
3. ✅ Download AI models to `models/pretrained_models/`
4. ✅ Start backend server
5. ✅ Start frontend
6. ✅ Open browser to http://localhost:3000

## 💡 Tips

- Use **GPU** for better performance: `inference_model_device: "cuda"`
- Adjust **thresholds** in config.yaml for accuracy
- Check **tracking_logs/** for detailed person tracking data
- Use **API docs** at http://localhost:8080/docs for testing

## 🔗 Quick Links

- Backend: http://localhost:8080
- Frontend: http://localhost:3000
- API Docs: http://localhost:8080/docs
- WebSocket: ws://localhost:8080/ws/tracking/cam01

---

**Pro tip:** Bookmark this file for quick reference! 🔖
