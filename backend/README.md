# MultiCamera Backend

Run the backend API to serve RTSP camera streams to the Next.js dashboard in `Front-end`.

## Requirements

- Python 3.10+
- RTSP camera URLs configured in `config.yaml`

## Start the server

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install fastapi uvicorn
uvicorn backend.app:app --host 0.0.0.0 --port 8080 --reload
```

## Endpoints

- `GET /cameras`: camera metadata list.
- `GET /api/stream/{camera_id}`: MJPEG stream.
- `WS /ws/stream/{camera_id}`: binary JPEG frames for the dashboard.

Set `NEXT_PUBLIC_BACKEND_URL` in the Front-end to point to this server.
