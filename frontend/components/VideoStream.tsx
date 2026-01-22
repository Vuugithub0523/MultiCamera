// components/VideoStream.tsx
/**
 * Video stream component with canvas overlay for tracking annotations
 */
"use client"

import { useRef, useState, useEffect } from 'react';
import { useWebSocketStream } from '@/hooks/useWebSocketStream';
import { drawTrackingAnnotations, drawCameraInfo, clearCanvas } from '@/lib/canvas-utils';

interface VideoStreamProps {
    cameraId: string;
    type?: 'stream' | 'tracking' | 'motion';
    className?: string;
    showStatus?: boolean;
    showAnnotations?: boolean;
    backendUrl?: string;
}

export function VideoStream({
    cameraId,
    type = 'stream',
    className = '',
    showStatus = false,
    showAnnotations = true,
    backendUrl
}: VideoStreamProps) {
    const { imageUrl, trackingData, status } = useWebSocketStream({ cameraId, type, backendUrl });
    const imgRef = useRef<HTMLImageElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [imgError, setImgError] = useState(false);
    const [imageDimensions, setImageDimensions] = useState({ width: 1920, height: 1080 });

    // Draw annotations on canvas when tracking data or image changes
    useEffect(() => {
        if (!showAnnotations || !canvasRef.current || !trackingData.length) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clear previous drawings
        clearCanvas(ctx, canvas.width, canvas.height);

        // Draw tracking annotations
        drawTrackingAnnotations(
            ctx,
            trackingData,
            canvas.width,
            canvas.height,
            imageDimensions.width,
            imageDimensions.height
        );

        // Draw camera info
        drawCameraInfo(ctx, cameraId, status.fps, trackingData.length);
    }, [trackingData, showAnnotations, imageDimensions, cameraId, status.fps]);

    // Update canvas size when container resizes
    useEffect(() => {
        if (!containerRef.current || !canvasRef.current) return;

        const resizeObserver = new ResizeObserver(() => {
            if (containerRef.current && canvasRef.current) {
                const { width, height } = containerRef.current.getBoundingClientRect();
                canvasRef.current.width = width;
                canvasRef.current.height = height;
            }
        });

        resizeObserver.observe(containerRef.current);

        return () => resizeObserver.disconnect();
    }, []);

    // Handle image load to get actual dimensions
    const handleImageLoad = () => {
        if (imgRef.current) {
            setImageDimensions({
                width: imgRef.current.naturalWidth,
                height: imgRef.current.naturalHeight,
            });
        }
        setImgError(false);
    };

    return (
        <div ref={containerRef} className={`relative w-full h-full bg-black ${className}`}>
            {imageUrl && !imgError ? (
                <>
                    <img
                        ref={imgRef}
                        src={imageUrl}
                        alt={`Camera ${cameraId}`}
                        className="w-full h-full object-cover"
                        onError={() => setImgError(true)}
                        onLoad={handleImageLoad}
                    />
                    {showAnnotations && (
                        <canvas
                            ref={canvasRef}
                            className="absolute inset-0 w-full h-full pointer-events-none"
                            style={{ zIndex: 10 }}
                        />
                    )}
                </>
            ) : (
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center text-white/60">
                        <div className="animate-pulse mb-2">
                            <svg className="w-8 h-8 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                        </div>
                        {status.error ? (
                            <>
                                <p className="text-xs text-red-400 mb-1">⚠️ Lỗi kết nối</p>
                                <p className="text-[10px] text-white/40 max-w-[200px] mx-auto">{status.error}</p>
                            </>
                        ) : (
                            <p className="text-xs">{status.connected ? 'Loading...' : 'Connecting...'}</p>
                        )}
                    </div>
                </div>
            )}

            {showStatus && !showAnnotations && (
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
