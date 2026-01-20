// Video event types for event-based storage

export interface VideoEvent {
  id: string;
  cameraId: string; // Using string to match backend camera IDs (cam01, cam02, cam03)
  cameraName: string;
  timestamp: Date;
  duration: number; // seconds
  personId?: number;
  personName?: string;
  eventType: 'person_appear' | 'new_person' | 'person_return' | 'abnormal';
  thumbnailUrl: string;
  videoUrl: string;
  size: string; // e.g., "12.5 MB"
  description: string;
  isAlert?: boolean;
}

export interface VideoEventFilter {
  cameraId?: string | 'all';
  eventType?: VideoEvent['eventType'] | 'all';
  dateRange?: 'today' | 'yesterday' | 'week' | 'month';
  searchQuery?: string;
}

export interface VideoEventStatistics {
  totalEvents: number;
  newPersons: number;
  alerts: number;
  storageUsed: string;
}
