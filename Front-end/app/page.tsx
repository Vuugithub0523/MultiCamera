"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { useTheme } from "next-themes"
import {
  Maximize2,
  Settings,
  Eye,
  Users,
  Clock,
  AlertTriangle,
  ChevronDown,
  MapPin,
  Video,
  X,
  Moon,
  Sun,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"
import { DashboardLayout } from "@/components/dashboard-layout"
import { VideoStream } from "@/components/VideoStream"

// Camera configuration matching backend camera IDs
const cameras = [
  { id: "cam01", code: "CCTV 01", name: "Camera 1 - Entrance", location: "Main Area", status: "online" },
  { id: "cam02", code: "CCTV 02", name: "Camera 2 - Lobby", location: "Secondary Area", status: "online" },
  { id: "cam03", code: "CCTV 03", name: "Camera 3 - Warehouse", location: "Storage", status: "online" },
]

// Tracked persons data - số người đang tracking (từ server)
const trackedPersons = [
  { id: 5, confidence: 98, cameraId: "cam01", firstSeen: "10:00:05" },
  { id: 12, confidence: 95, cameraId: "cam01", firstSeen: "10:02:15" },
  { id: 8, confidence: 92, cameraId: "cam02", firstSeen: "10:01:30" },
  { id: 3, confidence: 89, cameraId: "cam03", firstSeen: "09:58:45" },
]

const events = [
  { id: 1, time: "10:05:32", personId: 5, type: "appear", camera: "cam01", thumbnail: "/person-face-portrait.png" },
  { id: 2, time: "10:05:45", personId: 5, type: "move", camera: "cam02", thumbnail: "/person-face-portrait-man.jpg" },
  { id: 3, time: "10:06:12", personId: 12, type: "appear", camera: "cam01", thumbnail: "/person-face-portrait-woman.jpg" },
  {
    id: 4,
    time: "10:06:30",
    personId: 8,
    type: "alert",
    camera: "cam02",
    thumbnail: "/person-face-portrait-stranger.jpg",
    message: "Phát hiện người lạ tại khu vực kho",
  },
  { id: 5, time: "10:07:00", personId: 3, type: "appear", camera: "cam03", thumbnail: "/person-face-portrait-employee.jpg" },
  { id: 6, time: "10:07:15", personId: 5, type: "move", camera: "cam03", thumbnail: "/person-face-portrait-male.jpg" },
]

export default function Dashboard() {
  const router = useRouter()
  const { theme, setTheme } = useTheme()
  const [mainCamera, setMainCamera] = useState("cam01")
  const [currentTime, setCurrentTime] = useState(new Date())
  const [lightboxCamera, setLightboxCamera] = useState<string | null>(null)
  const [mounted, setMounted] = useState(false)
  const [isTransitioning, setIsTransitioning] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  const sidebarCameras = cameras.filter((c) => c.id !== mainCamera)
  const mainCameraData = cameras.find((c) => c.id === mainCamera)

  const handleCameraSwap = (cameraId: string) => {
    setIsTransitioning(true)
    // Small delay to show transition effect
    setTimeout(() => {
      setMainCamera(cameraId)
      setIsTransitioning(false)
    }, 150)
  }

  const CameraHeader = ({ camera, isMain = false }: { camera: (typeof cameras)[0]; isMain?: boolean }) => (
    <div className="bg-secondary/50 border border-border rounded-t-xl px-2.5 py-1.5 flex items-center justify-between">
      <div className="flex items-center gap-1.5">
        <Video className="w-3.5 h-3.5 text-primary" />
        <span className="text-foreground text-xs font-medium">
          {camera.code} / {camera.name}
        </span>
      </div>
      <div className="flex items-center gap-0.5">
        <Button
          size="icon"
          variant="ghost"
          className="h-5 w-5 text-muted-foreground hover:text-foreground hover:bg-secondary"
          onClick={() => setLightboxCamera(camera.id)}
        >
          <Maximize2 className="w-3 h-3" />
        </Button>
        <Button
          size="icon"
          variant="ghost"
          className="h-5 w-5 text-muted-foreground hover:text-foreground hover:bg-secondary"
          onClick={() => router.push(`/configuration?camera=${camera.id}`)}
        >
          <Settings className="w-3 h-3" />
        </Button>
      </div>
    </div>
  )

  return (
    <DashboardLayout>
      <div className="flex flex-col h-full">
        <div className="px-4 py-2 flex items-center justify-between border-b border-border">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-card rounded-full border border-border">
              <Users className="w-3.5 h-3.5 text-primary" />
              <span className="text-xs font-medium text-foreground">
                Active: <span className="text-primary font-semibold">{trackedPersons.length}</span>
              </span>
            </div>
            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-card rounded-full border border-border">
              <Eye className="w-3.5 h-3.5 text-emerald-500" />
              <span className="text-xs font-medium text-foreground">
                Cameras: <span className="text-emerald-500 font-semibold">3/3</span>
              </span>
            </div>
            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-card rounded-full border border-border">
              <Clock className="w-3.5 h-3.5 text-muted-foreground" />
              <span className="text-xs font-mono text-foreground">{currentTime.toLocaleTimeString("vi-VN")}</span>
            </div>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1.5 bg-card rounded-full border border-border">
            {mounted && theme === "dark" ? (
              <Moon className="w-3.5 h-3.5 text-primary" />
            ) : (
              <Sun className="w-3.5 h-3.5 text-primary" />
            )}
            <span className="text-xs text-foreground">Dark Mode</span>
            <Switch
              checked={mounted && theme === "dark"}
              onCheckedChange={(checked) => setTheme(checked ? "dark" : "light")}
              className="data-[state=checked]:bg-primary scale-75"
            />
          </div>
        </div>

        <div className="flex-1 p-4 min-h-0 overflow-hidden">
          <div className="flex gap-3 h-full">
            {/* Main Camera - 65% */}
            <div className="flex-[2] min-w-0 flex flex-col max-h-full">
              <CameraHeader camera={mainCameraData!} isMain />
              <Card className="flex-1 bg-black border-0 overflow-hidden relative rounded-t-none rounded-b-xl min-h-0">
                {/* Video stream từ backend với WebSocket */}
                <div className={`absolute inset-0 transition-opacity duration-200 ${isTransitioning ? "opacity-0" : "opacity-100"}`}>
                  <VideoStream
                    cameraId={mainCamera}
                    type="stream"
                    showStatus={true}
                    className="w-full h-full"
                  />
                </div>
                {/* Loading overlay during transition */}
                {isTransitioning && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                    <div className="flex flex-col items-center gap-2">
                      <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                      <span className="text-white text-sm">Switching camera...</span>
                    </div>
                  </div>
                )}
              </Card>
            </div>

            {/* Sidebar Cameras - 35% stacked vertically */}
            <div className="flex-1 flex flex-col gap-3 min-w-0 max-h-full">
              {sidebarCameras.map((camera) => (
                <div
                  key={camera.id}
                  className="flex-1 flex flex-col cursor-pointer group min-h-0"
                  onClick={() => handleCameraSwap(camera.id)}
                >
                  <CameraHeader camera={camera} />
                  <div className="flex-1 bg-black border-0 overflow-hidden relative rounded-t-none rounded-b-xl min-h-0">
                    {/* Video stream từ tracking service với bounding boxes */}
                    <VideoStream
                      cameraId={camera.id}
                      type="stream"
                      className="absolute inset-0 w-full h-full opacity-95 group-hover:opacity-100 transition-opacity"
                    />

                    {/* Click to Swap Overlay */}
                    <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/30 z-10">
                      <span className="text-white text-xs font-medium bg-black/60 px-2 py-1 rounded">
                        Click to Swap
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="border-t border-border bg-card flex-shrink-0">
          <div className="px-4 py-3">
            <div className="flex gap-3 overflow-x-auto pb-1">
              {events.map((event) => (
                <Card
                  key={event.id}
                  className={`p-2.5 cursor-pointer transition-all hover:shadow-md min-w-[180px] flex-shrink-0 ${event.type === "alert"
                    ? "border-red-500 dark:border-red-500 bg-red-50 dark:bg-red-950/30"
                    : "border-border bg-card hover:bg-secondary/30"
                    }`}
                  onClick={() => {
                    if (event.camera !== mainCamera) {
                      setMainCamera(event.camera)
                    }
                  }}
                >
                  <div className="flex items-start gap-2">
                    <img
                      src={event.thumbnail || "/placeholder.svg"}
                      alt={`Person ${event.personId}`}
                      className="w-9 h-9 rounded-full border border-border object-cover flex-shrink-0"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 mb-0.5">
                        <span className="text-[10px] font-mono text-muted-foreground">[{event.time}]</span>
                        {event.type === "alert" && <AlertTriangle className="w-3 h-3 text-red-500" />}
                      </div>
                      <p className="text-xs text-foreground leading-tight">
                        {event.type === "appear" && (
                          <>
                            ID{" "}
                            <span className="text-primary font-mono font-semibold">
                              #{event.personId.toString().padStart(2, "0")}
                            </span>{" "}
                            xuất hiện
                          </>
                        )}
                        {event.type === "move" && (
                          <>
                            ID{" "}
                            <span className="text-primary font-mono font-semibold">
                              #{event.personId.toString().padStart(2, "0")}
                            </span>{" "}
                            di chuyển
                          </>
                        )}
                        {event.type === "alert" && (
                          <span className="text-red-600 dark:text-red-400 text-[10px]">{event.message}</span>
                        )}
                      </p>
                      <div className="flex items-center gap-1 mt-0.5">
                        <MapPin className="w-2.5 h-2.5 text-muted-foreground" />
                        <span className="text-[10px] text-muted-foreground">
                          {cameras.find((c) => c.id === event.camera)?.code}
                        </span>
                      </div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Lightbox Dialog for Fullscreen Camera View */}
      <Dialog open={lightboxCamera !== null} onOpenChange={(open) => !open && setLightboxCamera(null)}>
        <DialogContent className="max-w-none w-auto h-auto p-0 bg-black border-0 flex items-center justify-center">
          <DialogTitle className="sr-only">
            {lightboxCamera && cameras.find((c) => c.id === lightboxCamera)?.name}
          </DialogTitle>
          {lightboxCamera && (
            <div className="relative">
              {/* Camera Header */}
              <div className="absolute top-0 left-0 right-0 z-10 bg-gradient-to-b from-black/80 to-transparent p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Video className="w-5 h-5 text-white" />
                    <div className="text-white">
                      <div className="font-semibold">
                        {cameras.find((c) => c.id === lightboxCamera)?.name}
                      </div>
                      <div className="text-xs text-white/70">
                        {cameras.find((c) => c.id === lightboxCamera)?.location}
                      </div>
                    </div>
                    <Badge className="bg-green-600 text-white">
                      <Eye className="w-3 h-3 mr-1" /> Online
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      variant="secondary"
                      className="gap-2"
                    >
                      <Video className="w-4 h-4" />
                      Disconnect
                    </Button>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="text-white hover:bg-white/20"
                      onClick={() => setLightboxCamera(null)}
                    >
                      <X className="w-5 h-5" />
                    </Button>
                  </div>
                </div>
              </div>

              {/* Camera Video Stream - Large with 16:9 ratio */}
              <div className="relative bg-black" style={{ width: '90vw', height: 'calc(90vw * 9 / 16)', maxHeight: '85vh' }}>
                <VideoStream
                  cameraId={lightboxCamera}
                  type="stream"
                  showStatus={true}
                  className="absolute inset-0 w-full h-full"
                />
              </div>

              {/* Camera Info Footer */}
              <div className="absolute bottom-0 left-0 right-0 z-10 bg-gradient-to-t from-black/80 to-transparent p-4">
                <div className="flex items-center justify-between text-white text-sm">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <Users className="w-4 h-4" />
                      <span>
                        {trackedPersons.filter((p) => p.cameraId === lightboxCamera).length} người đang tracking
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Clock className="w-4 h-4" />
                      <span>{currentTime.toLocaleTimeString("vi-VN")}</span>
                    </div>
                  </div>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => {
                      setLightboxCamera(null)
                      router.push(`/configuration?camera=${lightboxCamera}`)
                    }}
                  >
                    <Settings className="w-4 h-4 mr-2" />
                    Camera Settings
                  </Button>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </DashboardLayout>
  )
}
