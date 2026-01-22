/**
 * Canvas utility functions for drawing tracking annotations
 */

import { TrackingData } from '@/hooks/useWebSocketStream';

// Softer color palette for person IDs (pastel colors)
const COLOR_PALETTE = [
  [100, 200, 100],   // Soft green
  [200, 100, 100],   // Soft red
  [100, 150, 220],   // Soft blue
  [220, 200, 100],   // Soft yellow
  [200, 120, 200],   // Soft magenta
  [120, 200, 200],   // Soft cyan
  [220, 160, 100],   // Soft orange
  [160, 120, 220],   // Soft purple
  [120, 200, 160],   // Soft mint
  [220, 140, 180],   // Soft pink
];

/**
 * Get consistent color for a person ID
 */
function getPersonColor(personId: number | null, state: string | null): string {
  if (!personId) {
    return 'rgb(255, 255, 0)'; // Yellow for unidentified tracks
  }

  const colorIndex = personId % COLOR_PALETTE.length;
  let [r, g, b] = COLOR_PALETTE[colorIndex];

  // Modify color based on state
  if (state === 'lost') {
    r = Math.floor(r / 2);
    g = Math.floor(g / 2);
    b = Math.floor(b / 2);
  } else if (state === 'confirmed_lost') {
    r = g = b = 128; // Gray
  }

  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Get state abbreviation
 */
function getStateShort(state: string | null): string {
  const stateMap: Record<string, string> = {
    'detected': 'DET',
    'tracking': 'TRK',
    'lost': 'LST',
    'confirmed_lost': 'CLT',
  };
  return state ? (stateMap[state] || state.slice(0, 3).toUpperCase()) : '';
}

/**
 * Draw tracking annotations on canvas
 */
export function drawTrackingAnnotations(
  ctx: CanvasRenderingContext2D,
  tracks: TrackingData[],
  canvasWidth: number,
  canvasHeight: number,
  imageWidth: number,
  imageHeight: number
) {
  // Calculate scaling factors
  const scaleX = canvasWidth / imageWidth;
  const scaleY = canvasHeight / imageHeight;

  tracks.forEach((track) => {
    const [x, y, w, h] = track.bbox;
    
    // Scale coordinates
    const scaledX = x * scaleX;
    const scaledY = y * scaleY;
    const scaledW = w * scaleX;
    const scaledH = h * scaleY;

    // Get color
    const color = getPersonColor(track.person_id, track.state);
    const thickness = track.person_id ? 2 : 1;  // Reduced thickness

    // Draw bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = thickness;
    ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);

    // Simplified label - only ID and confidence percentage
    let label = track.person_id ? `ID:${track.person_id}` : `T${track.track_id}`;
    label += ` ${(track.confidence * 100).toFixed(0)}%`;

    // Draw label with smaller font
    ctx.font = 'bold 12px sans-serif';  // Smaller font
    const metrics = ctx.measureText(label);
    const textWidth = metrics.width;
    const textHeight = 16;  // Smaller height
    const padding = 3;  // Smaller padding

    const labelX = scaledX;
    const labelY = Math.max(scaledY - textHeight - padding, textHeight);

    // Semi-transparent background
    ctx.fillStyle = color.replace('rgb', 'rgba').replace(')', ', 0.85)');
    ctx.fillRect(labelX, labelY - textHeight, textWidth + padding * 2, textHeight + padding);

    // Draw label text in white for better contrast
    ctx.fillStyle = 'white';
    ctx.textBaseline = 'top';
    ctx.fillText(label, labelX + padding, labelY - textHeight + padding / 2);
  });
}

/**
 * Draw camera info overlay
 */
export function drawCameraInfo(
  ctx: CanvasRenderingContext2D,
  cameraId: string,
  fps: number,
  trackCount: number
) {
  // Compact info display
  const info = `${cameraId.toUpperCase()} | ${fps.toFixed(1)} FPS | ${trackCount} tracks`;

  ctx.font = 'bold 13px sans-serif';
  ctx.textBaseline = 'top';

  const metrics = ctx.measureText(info);
  const textWidth = metrics.width;
  const padding = 6;

  // Semi-transparent dark background
  ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
  ctx.fillRect(8, 8, textWidth + padding * 2, 22);

  // White text
  ctx.fillStyle = 'rgb(255, 255, 255)';
  ctx.fillText(info, 8 + padding, 8 + padding);
}

/**
 * Clear canvas
 */
export function clearCanvas(ctx: CanvasRenderingContext2D, width: number, height: number) {
  ctx.clearRect(0, 0, width, height);
}
