import { useState, useEffect, useRef } from 'react';

interface UseWebSocketStreamOptions {
  cameraId: string;
  type?: 'stream' | 'tracking' | 'motion';
  backendUrl?: string;
}

interface StreamStatus {
  connected: boolean;
  fps: number;
  lastUpdate: number;
}

export function useWebSocketStream({ 
  cameraId, 
  type = 'stream',
  backendUrl 
}: UseWebSocketStreamOptions) {
  const [imageUrl, setImageUrl] = useState<string>('');
  const [status, setStatus] = useState<StreamStatus>({
    connected: false,
    fps: 0,
    lastUpdate: 0,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const fpsCounterRef = useRef({ count: 0, lastTime: Date.now() });

  useEffect(() => {
    // Determine WebSocket URL - connect directly to backend port 8080
    let baseUrl = backendUrl;
    
    if (!baseUrl) {
      // Use environment variable or default to localhost:8080
      const backendHttpUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8080';
      baseUrl = backendHttpUrl.replace('http://', 'ws://').replace('https://', 'wss://');
    } else {
      // Convert http to ws
      baseUrl = baseUrl.replace('http://', 'ws://').replace('https://', 'wss://');
    }
    
    // Connect to backend WebSocket
    const wsUrl = `${baseUrl}/ws/${type}/${cameraId}`;
    
    console.log(`[WebSocket] Connecting to ${wsUrl}`);
    
    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      console.log(`[WebSocket] Connected to camera ${cameraId}`);
      setStatus(prev => ({ ...prev, connected: true }));
    };

    ws.onmessage = (event) => {
      try {
        if (event.data instanceof ArrayBuffer) {
          // Binary JPEG data
          const blob = new Blob([event.data], { type: 'image/jpeg' });
          const url = URL.createObjectURL(blob);
          
          setImageUrl(prevUrl => {
            if (prevUrl) URL.revokeObjectURL(prevUrl);
            return url;
          });
        } else {
          // Text data (might be JSON)
          const data = JSON.parse(event.data);
          if (data.frame) {
            // Base64 encoded image
            setImageUrl(`data:image/jpeg;base64,${data.frame}`);
          }
        }

        // Update FPS counter
        const now = Date.now();
        fpsCounterRef.current.count++;
        if (now - fpsCounterRef.current.lastTime >= 1000) {
          const fps = fpsCounterRef.current.count;
          setStatus(prev => ({ ...prev, fps, lastUpdate: now }));
          fpsCounterRef.current = { count: 0, lastTime: now };
        }
      } catch (error) {
        console.error('[WebSocket] Parse error:', error);
      }
    };

    ws.onerror = (error) => {
      try {
        console.error(`[WebSocket] Error for camera ${cameraId}:`, error, 'readyState:', ws.readyState);
      } catch (e) {
        console.error(`[WebSocket] Error for camera ${cameraId}: (logging failed)`, e);
      }
      setStatus(prev => ({ ...prev, connected: false }));
    };

    ws.onclose = (event) => {
      try {
        console.log(`[WebSocket] Disconnected from camera ${cameraId}`, 'code:', (event && (event as CloseEvent).code), 'reason:', (event && (event as CloseEvent).reason));
      } catch (e) {
        console.log(`[WebSocket] Disconnected from camera ${cameraId}`);
      }
      setStatus(prev => ({ ...prev, connected: false, fps: 0 }));
    };

    wsRef.current = ws;

    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [cameraId, type, backendUrl]);

  return { imageUrl, status };
}
