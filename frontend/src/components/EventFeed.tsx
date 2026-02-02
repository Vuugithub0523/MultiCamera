import { useState, useEffect } from 'react'

interface DetectionEvent {
  timestamp: string
  camera_id: string
  count: number
  detections: Array<{
    x1: number
    y1: number
    x2: number
    y2: number
    confidence: number
    label: string
  }>
}

export function EventFeed() {
  const [events, setEvents] = useState<DetectionEvent[]>([])
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    const eventSource = new EventSource('/api/events/stream')

    eventSource.onopen = () => {
      setConnected(true)
    }

    eventSource.addEventListener('detection', (e) => {
      try {
        const event = JSON.parse(e.data) as DetectionEvent
        setEvents((prev) => [event, ...prev].slice(0, 50))
      } catch (err) {
        console.error('Failed to parse event:', err)
      }
    })

    eventSource.onerror = () => {
      setConnected(false)
    }

    return () => {
      eventSource.close()
    }
  }, [])

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
          Detection Events
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {events.length === 0 ? (
          <div className="text-gray-500 text-center py-8">
            No events yet...
          </div>
        ) : (
          <div className="space-y-2">
            {events.map((event, index) => (
              <EventCard key={`${event.timestamp}-${index}`} event={event} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function EventCard({ event }: { event: DetectionEvent }) {
  const time = new Date(event.timestamp).toLocaleTimeString()

  return (
    <div className="bg-gray-700 rounded-lg p-3 text-sm">
      <div className="flex justify-between items-center mb-1">
        <span className="font-medium text-blue-400">{event.camera_id}</span>
        <span className="text-gray-400 text-xs">{time}</span>
      </div>
      <div className="text-white">
        {event.count} person{event.count !== 1 ? 's' : ''} detected
      </div>
      {event.detections.length > 0 && (
        <div className="mt-1 text-xs text-gray-400">
          Confidence: {event.detections.map(d => `${(d.confidence * 100).toFixed(0)}%`).join(', ')}
        </div>
      )}
    </div>
  )
}
