import { useState } from 'react'
import {
  LiveKitRoom,
  GridLayout,
  ParticipantTile,
  useTracks,
  RoomAudioRenderer,
} from '@livekit/components-react'
import { Track } from 'livekit-client'
import { EventFeed } from './components/EventFeed'
import { Header } from './components/Header'
import { useToken } from './hooks/useToken'

// Configuration - matches config.yaml
const LIVEKIT_URL = 'ws://192.168.1.247:7880'
const ROOM_NAME = 'multicam'

function App() {
  const [connected, setConnected] = useState(false)
  const token = useToken(ROOM_NAME, 'viewer')

  if (!token) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Connecting to LiveKit...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      <Header connected={connected} />
      
      <div className="flex-1 flex">
        {/* Main video grid */}
        <div className="flex-1 p-4">
          <LiveKitRoom
            serverUrl={LIVEKIT_URL}
            token={token}
            connect={true}
            onConnected={() => setConnected(true)}
            onDisconnected={() => setConnected(false)}
            className="h-full"
          >
            <VideoGrid />
            <RoomAudioRenderer />
          </LiveKitRoom>
        </div>

        {/* Event sidebar */}
        <div className="w-80 bg-gray-800 border-l border-gray-700">
          <EventFeed />
        </div>
      </div>
    </div>
  )
}

function VideoGrid() {
  const tracks = useTracks([Track.Source.Camera])

  if (tracks.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-gray-400">
        <div className="text-center">
          <div className="text-2xl mb-2">📹</div>
          <div>Waiting for video streams...</div>
          <div className="text-sm mt-2">Make sure AI service and Publisher are running</div>
        </div>
      </div>
    )
  }

  return (
    <GridLayout
      tracks={tracks}
      className="h-full"
    >
      <ParticipantTile />
    </GridLayout>
  )
}

export default App
