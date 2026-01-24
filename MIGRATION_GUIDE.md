# Hướng dẫn Migration từ cấu trúc cũ sang mới

## Tổng quan

Project đã được tổ chức lại thành 3 phần rõ ràng:

```
OLD STRUCTURE → NEW STRUCTURE

Root files (AI)          → ai-service/
local_server.py          → backend-server/
Front-end/               → frontend/ (đổi tên)
```

## Chi tiết thay đổi

### 1. AI Service (ai-service/)

**Files đã di chuyển:**
- `object_detection.py` → `ai-service/core/object_detection.py`
- `feature_extraction.py` → `ai-service/core/feature_extraction.py`
- `person_lifecycle_manager.py` → `ai-service/core/person_lifecycle_manager.py`
- `rtsp_multicam_loader.py` → `ai-service/utils/rtsp_loader.py`
- `helpers.py` → `ai-service/utils/helpers.py`

**Files mới:**
- `ai-service/__init__.py`
- `ai-service/core/__init__.py`
- `ai-service/utils/__init__.py`
- `ai-service/requirements.txt`
- `ai-service/README.md`

### 2. Backend Server (backend-server/)

**Files đã di chuyển:**
- `local_server.py` → Logic được tách thành:
  - `backend-server/server.py` (FastAPI app)
  - `backend-server/api/stream_manager.py` (Business logic)

**Files mới:**
- `backend-server/api/__init__.py`
- `backend-server/requirements.txt`
- `backend-server/README.md`

### 3. Frontend (frontend/)

**Đổi tên:**
- `Front-end/` → `frontend/` (chữ thường, chuẩn convention)

**Không thay đổi:**
- Tất cả files trong frontend giữ nguyên

## Import Path Changes

### Old imports (không còn hoạt động):
```python
from object_detection import ObjectDetection
from feature_extraction import FeatureExtraction
from person_lifecycle_manager import PersonLifecycleManager
```

### New imports (trong backend-server):
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "ai-service"))

from core.object_detection import ObjectDetection
from core.feature_extraction import FeatureExtraction
from core.person_lifecycle_manager import PersonLifecycleManager
```

## Files không thay đổi

Các files/folders sau vẫn ở vị trí cũ:
- `config.yaml` (root)
- `models/` (root)
- `storage/` (root)
- `tracking_logs/` (root)
- `requirements.txt` (root - legacy, nên dùng service-specific)

## Files cũ có thể xóa

Sau khi verify hệ thống hoạt động tốt, bạn có thể xóa:
- `object_detection.py`
- `feature_extraction.py`
- `person_lifecycle_manager.py`
- `rtsp_multicam_loader.py`
- `helpers.py`
- `local_server.py`
- `main_ID_cycle.py` (nếu không dùng)

## Cách chạy hệ thống mới

### 1. Setup Backend
```bash
cd backend-server
pip install -r requirements.txt
python server.py
```

### 2. Setup Frontend
```bash
cd frontend
npm install
npm run dev
```

### 3. Hoặc dùng scripts
**Windows:**
```bash
start-backend.bat
start-frontend.bat
```

**Linux/Mac:**
```bash
./start-backend.sh
./start-frontend.sh
```

## Troubleshooting

### Backend không start
**Lỗi:** `ModuleNotFoundError: No module named 'core'`

**Giải pháp:** Đảm bảo `ai-service/` ở đúng vị trí và có `__init__.py`

### Frontend không connect
**Lỗi:** WebSocket connection failed

**Giải pháp:** 
1. Check backend đang chạy tại port 8080
2. Check WebSocket URL trong frontend code

### Import errors
**Lỗi:** `ImportError: cannot import name 'ObjectDetection'`

**Giải pháp:** Check sys.path setup trong `backend-server/api/stream_manager.py`

## Testing Migration

### 1. Test Backend API
```bash
curl http://localhost:8080/health
# Expected: {"status": "ok"}

curl http://localhost:8080/api/cameras
# Expected: {"cameras": ["cam01", "cam02", "cam03"]}
```

### 2. Test WebSocket (Browser Console)
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/tracking/cam01');
ws.onmessage = (e) => console.log('Frame received', e.data.size);
```

### 3. Test Frontend
1. Open http://localhost:3000
2. Check video streams hiển thị
3. Check tracking overlay hoạt động

## Rollback Plan

Nếu cần rollback về cấu trúc cũ:
1. Giữ nguyên files cũ (chưa xóa)
2. Đổi tên `frontend/` về `Front-end/`
3. Sử dụng `local_server.py` và `main_ID_cycle.py` như trước

## Next Steps

Sau khi migration thành công:
1. ✅ Verify tất cả chức năng hoạt động
2. ✅ Test với real RTSP streams
3. ✅ Test tracking accuracy
4. ⬜ Commit changes to Git
5. ⬜ Update documentation
6. ⬜ Xóa files cũ không dùng

## Support

Nếu gặp vấn đề, check:
1. README.md trong mỗi service
2. Import paths trong code
3. config.yaml paths (models, logs, etc.)
