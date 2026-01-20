// components/VideoStream.tsx
/**
 * Video stream component that renders WebSocket video feed.
 */
"use client"

import { useRef, useState } from 'react';
import { useWebSocketStream } from '@/hooks/useWebSocketStream';

interface VideoStreamProps {
    cameraId: string;
    type?: 'stream' | 'tracking' | 'motion';
    className?: string;
    showStatus?: boolean;
    backendUrl?: string;
}

export function VideoStream({
    cameraId,
    type = 'stream',
    className = '',
    showStatus = false,
    backendUrl
}: VideoStreamProps) {
    const { imageUrl, status } = useWebSocketStream({ cameraId, type, backendUrl });
    const imgRef = useRef<HTMLImageElement>(null);
    const [imgError, setImgError] = useState(false);

    return (
        <div className={`relative w-full h-full bg-black ${className}`}>
            {imageUrl && !imgError ? (
                <img
                    ref={imgRef}
                    src={imageUrl}
                    alt={`Camera ${cameraId}`}
                    className="w-full h-full object-cover"
                    onError={() => setImgError(true)}
                    onLoad={() => setImgError(false)}
                />
            ) : (
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center text-white/60">
                        <div className="animate-pulse mb-2">
                            <svg className="w-8 h-8 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                        </div>
                        <p className="text-xs">{status.connected ? 'Loading...' : 'Connecting...'}</p>
                    </div>
                </div>
            )}

            {showStatus && (
                <div className="absolute bottom-2 left-2 flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${status.connected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                    <span className="text-white text-xs bg-black/50 px-1.5 py-0.5 rounded">
                        {status.connected ? `${status.fps} FPS` : 'Offline'}
                    </span>
                </div>
            )}
        </div>
    );
}

export default VideoStream;
