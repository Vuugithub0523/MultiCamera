import { useState, useEffect, useRef } from 'react';

interface UseWebSocketStreamOptions {
  cameraId: string;
  type?: 'stream' | 'tracking' | 'motion';
  backendUrl?: string;
  autoReconnect?: boolean;
}

interface StreamStatus {
  connected: boolean;
  fps: number;
  lastUpdate: number;
  error?: string;
}

export interface TrackingData {
  track_id: number;
  person_id: number | null;
  bbox: [number, number, number, number]; // [x, y, w, h]
  confidence: number;
  is_new: boolean;
  state: string | null;
}

export function useWebSocketStream({ 
  cameraId, 
  type = 'stream',
  backendUrl,
  autoReconnect = true
}: UseWebSocketStreamOptions) {
  const [imageUrl, setImageUrl] = useState<string>('');
  const [trackingData, setTrackingData] = useState<TrackingData[]>([]);
  const [status, setStatus] = useState<StreamStatus>({
    connected: false,
    fps: 0,
    lastUpdate: 0,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const fpsCounterRef = useRef({ count: 0, lastTime: Date.now() });
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000;

  useEffect(() => {
    let isMounted = true;

    const connectWebSocket = () => {
      let baseUrl = backendUrl;
      
      if (!baseUrl) {
        const backendHttpUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3000';
        baseUrl = backendHttpUrl.replace('http://', 'ws://').replace('https://', 'wss://');
      } else {
        baseUrl = baseUrl.replace('http://', 'ws://').replace('https://', 'wss://');
      }
      
      const endpoint = type === 'stream' ? 'tracking' : type;
      const wsUrl = `${baseUrl}/ws/${endpoint}/${cameraId}`;
      
      console.log(`[WebSocket] Connecting to ${wsUrl}...`);
      
      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        console.log(`[WebSocket] Connected to camera ${cameraId}`);
        reconnectAttemptsRef.current = 0;
        if (isMounted) {
          setStatus(prev => ({ ...prev, connected: true, error: undefined }));
        }
      };

      ws.onmessage = (event) => {
        if (!isMounted) return;
        
        try {
          if (event.data instanceof ArrayBuffer) {
            // Parse binary format: [4 bytes: metadata_length][metadata_json][frame_jpeg]
            const buffer = event.data;
            const dataView = new DataView(buffer);
            
            // Read metadata length (first 4 bytes)
            const metadataLength = dataView.getUint32(0, false); // big-endian
            
            // Extract metadata JSON
            const metadataBytes = new Uint8Array(buffer, 4, metadataLength);
            const metadataText = new TextDecoder().decode(metadataBytes);
            const metadata = JSON.parse(metadataText) as TrackingData[];
            
            // Extract frame JPEG
            const frameBytes = new Uint8Array(buffer, 4 + metadataLength);
            const blob = new Blob([frameBytes], { type: 'image/jpeg' });
            const url = URL.createObjectURL(blob);
            
            setImageUrl(prevUrl => {
              if (prevUrl) URL.revokeObjectURL(prevUrl);
              return url;
            });
            
            setTrackingData(metadata);
          } else {
            // Fallback for JSON format (legacy)
            const data = JSON.parse(event.data);
            if (data.frame) {
              setImageUrl(`data:image/jpeg;base64,${data.frame}`);
            }
          }

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
        if (!isMounted) return; // Ignore errors after unmount
        
        const readyState = ws.readyState;
        const stateText = ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'][readyState];
        
        console.error(`[WebSocket] Error for camera ${cameraId} (state: ${stateText}/${readyState})`);
        
        let errorMsg = 'Connection error';
        if (readyState === 3) {
          errorMsg = 'Backend khong kha dung. Kiem tra: python main.py dang chay?';
        }
        
        setStatus(prev => ({ ...prev, connected: false, error: errorMsg }));
      };

      ws.onclose = (event) => {
        if (!isMounted) return; // Ignore close event after unmount
        
        const code = event.code;
        const reason = event.reason || 'Unknown';
        
        console.log(`[WebSocket] Disconnected from camera ${cameraId} (code: ${code}, reason: ${reason})`);
        
        setStatus(prev => ({ ...prev, connected: false, fps: 0 }));
        
        if (autoReconnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++;
          console.log(`[WebSocket] Reconnecting in ${reconnectDelay / 1000}s... (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            if (isMounted) {
              connectWebSocket();
            }
          }, reconnectDelay);
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          console.log(`[WebSocket] Max reconnect attempts reached for camera ${cameraId}`);
          setStatus(prev => ({ ...prev, error: 'Khong the ket noi sau nhieu lan thu' }));
        }
      };

      wsRef.current = ws;
    };

    connectWebSocket();

    return () => {
      isMounted = false;
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [cameraId, type, backendUrl, autoReconnect]);

  return { imageUrl, trackingData, status };
}
