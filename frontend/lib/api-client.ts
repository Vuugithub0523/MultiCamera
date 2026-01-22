// my-app/lib/api-client.ts
/**
 * API Client for Backend Integration
 * Provides type-safe API calls to backend services
 */

const BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8080";

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination?: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

// ============================================
// VIDEO EVENTS APIs
// ============================================

export interface VideoEvent {
  id: string;
  cameraId: string;
  cameraName: string;
  timestamp: string;
  duration: number;
  personId?: number | null;
  personName?: string;
  eventType: "person_appear" | "new_person" | "person_return" | "abnormal";
  thumbnailUrl: string;
  videoUrl: string;
  fileName: string;
  size: string;
  sizeBytes: number;
  description: string;
  isAlert: boolean;
}

export interface VideoEventsStats {
  totalEvents: number;
  newPersons: number;
  alerts: number;
  storageUsed: string;
  storageUsedBytes: number;
}

export async function getVideoEvents(params?: {
  camera_id?: string;
  event_type?: string;
  date_range?: string;
  search?: string;
  page?: number;
  limit?: number;
}): Promise<PaginatedResponse<VideoEvent>> {
  const query = new URLSearchParams(
    Object.entries(params || {})
      .filter(([, v]) => v !== undefined)
      .map(([k, v]) => [k, String(v)])
  );

  const response = await fetch(`${BASE_URL}/api/video-events?${query}`);
  return response.json();
}

export async function getVideoEventsStatistics(
  dateRange?: string
): Promise<ApiResponse<VideoEventsStats>> {
  const query = dateRange ? `?date_range=${dateRange}` : "";
  const response = await fetch(`${BASE_URL}/api/video-events/statistics${query}`);
  return response.json();
}

export async function downloadVideoEvent(eventId: string): Promise<void> {
  window.open(`${BASE_URL}/api/video-events/${eventId}/download`, "_blank");
}

export async function deleteVideoEvent(eventId: string): Promise<ApiResponse<void>> {
  const response = await fetch(`${BASE_URL}/api/video-events/${eventId}`, {
    method: "DELETE",
  });
  return response.json();
}

// ============================================
// CONFIGURATION APIs
// ============================================

export interface TrackingParameters {
  confidenceThreshold: number;
  reIdThreshold: number;
  maxTrackAge: number;
  minTrackHits: number;
  iouThreshold: number;
}

export async function getTrackingParameters(): Promise<ApiResponse<TrackingParameters>> {
  const response = await fetch(`${BASE_URL}/api/config/tracking-parameters`);
  return response.json();
}

export async function updateTrackingParameters(
  params: Partial<TrackingParameters>
): Promise<ApiResponse<TrackingParameters>> {
  const response = await fetch(`${BASE_URL}/api/config/tracking-parameters`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return response.json();
}

// ============================================
// CAMERA APIs (existing)
// ============================================

export interface Camera {
  id: string;
  code: string;
  name: string;
  location: string;
  status: "online" | "offline";
  rtspUrl?: string;
  streamUrl?: string;
}

export async function getCameras(): Promise<ApiResponse<Camera[]>> {
  const response = await fetch(`${BASE_URL}/cameras`);
  const result = await response.json();
  
  // Transform backend format to frontend format
  if (result.ok && result.data) {
    return {
      success: true,
      data: result.data.map((cam: any) => ({
        id: cam.id,
        code: cam.code || cam.id,
        name: cam.name,
        location: cam.location || "",
        status: "online", // Default to online
        rtspUrl: cam.rtspUrl,
        streamUrl: cam.streamUrl,
      })),
    };
  }
  
  return { success: false, error: "Failed to fetch cameras" };
}

export async function createCamera(camera: Omit<Camera, "id" | "status">): Promise<ApiResponse<Camera>> {
  const response = await fetch(`${BASE_URL}/cameras`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(camera),
  });
  return response.json();
}

export async function updateCamera(
  id: string,
  camera: Partial<Camera>
): Promise<ApiResponse<Camera>> {
  const response = await fetch(`${BASE_URL}/cameras/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(camera),
  });
  return response.json();
}

export async function deleteCamera(id: string): Promise<ApiResponse<void>> {
  const response = await fetch(`${BASE_URL}/cameras/${id}`, {
    method: "DELETE",
  });
  return response.json();
}

// ============================================
// DETECTION APIs (existing)
// ============================================

export interface CustomerDetection {
  id: number;
  customerId: number;
  name: string;
  cameraId: string;
  cameraName: string;
  imageUrl: string;
  timestamp: string;
  confidence: number;
  location?: string;
  visitCount?: number;
  phone?: string;
  dateOfBirth?: string;
  email?: string;
  address?: string;
  isNewCustomer: boolean;
}

export async function getCustomerDetections(params?: {
  camera_id?: string;
  limit?: number;
}): Promise<ApiResponse<CustomerDetection[]>> {
  const query = new URLSearchParams(
    Object.entries(params || {})
      .filter(([, v]) => v !== undefined)
      .map(([k, v]) => [k, String(v)])
  );

  const response = await fetch(`${BASE_URL}/api/detections/customers?${query}`);
  return response.json();
}

export async function getLatestDetections(): Promise<ApiResponse<CustomerDetection[]>> {
  const response = await fetch(`${BASE_URL}/api/detections/latest`);
  return response.json();
}

// ============================================
// WEBSOCKET EVENTS
// ============================================

export interface RealtimeEvent {
  id: number;
  time: string;
  timestamp: string;
  personId: number | null;
  personName: string;
  type: "appear" | "disappear" | "move" | "alert";
  camera: string;
  cameraName: string;
  thumbnail: string | null;
  message?: string;
}

export function connectToEventsStream(
  onEvent: (event: RealtimeEvent) => void,
  onError?: (error: Error) => void
): WebSocket {
  const ws = new WebSocket("ws://localhost:5000/ws/events");

  ws.onopen = () => {
    console.log("[Events WebSocket] Connected");
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "new_event" && data.data) {
        onEvent(data.data);
      }
    } catch (e) {
      console.error("[Events WebSocket] Parse error:", e);
    }
  };

  ws.onerror = (error) => {
    console.error("[Events WebSocket] Error:", error);
    onError?.(new Error("WebSocket error"));
  };

  ws.onclose = () => {
    console.log("[Events WebSocket] Disconnected");
  };

  return ws;
}
