"use client"

import { useState, useEffect } from "react"
import { Video, Users, AlertTriangle, Calendar, Download, Play, Search, Filter, Eye, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"
import { DashboardLayout } from "@/components/dashboard-layout"
import { 
  getVideoEvents, 
  getVideoEventsStatistics, 
  downloadVideoEvent, 
  deleteVideoEvent,
  type VideoEvent,
  type VideoEventsStats
} from "@/lib/api-client"

// Mock video events data - FALLBACK if API fails
const videoEvents = [
  {
    id: 1,
    cameraId: "cam01",
    cameraName: "CCTV 01 - Meeting Room",
    timestamp: new Date("2026-01-16T10:05:32"),
    duration: 45,
    personId: 5,
    personName: "Nguyễn Văn A",
    eventType: "person_appear",
    thumbnailUrl: "/person-face-portrait.png",
    videoUrl: "/videos/event-1.mp4",
    size: "12.5 MB",
    description: "Người xuất hiện tại phòng họp",
  },
  {
    id: 2,
    cameraId: "cam02",
    cameraName: "CCTV 02 - Open Workspace",
    timestamp: new Date("2026-01-16T10:06:15"),
    duration: 120,
    personId: 8,
    personName: "Unknown Person",
    eventType: "new_person",
    thumbnailUrl: "/person-face-portrait-stranger.jpg",
    videoUrl: "/videos/event-2.mp4",
    size: "32.1 MB",
    description: "Phát hiện người mới chưa được đăng ký",
  },
  {
    id: 3,
    cameraId: "cam02",
    cameraName: "CCTV 02 - Open Workspace",
    timestamp: new Date("2026-01-16T10:06:30"),
    duration: 90,
    personId: 8,
    personName: "Unknown Person",
    eventType: "abnormal",
    thumbnailUrl: "/person-face-portrait-stranger.jpg",
    videoUrl: "/videos/event-3.mp4",
    size: "24.8 MB",
    description: "Phát hiện người lạ tại khu vực kho",
    isAlert: true,
  },
  {
    id: 4,
    cameraId: "cam01",
    cameraName: "CCTV 01 - Meeting Room",
    timestamp: new Date("2026-01-16T09:58:45"),
    duration: 60,
    personId: 3,
    personName: "Trần Thị B",
    eventType: "person_return",
    thumbnailUrl: "/person-face-portrait-employee.jpg",
    videoUrl: "/videos/event-4.mp4",
    size: "16.2 MB",
    description: "Người quen quay lại",
  },
  {
    id: 5,
    cameraId: "cam03",
    cameraName: "CCTV 03 - Vendor Room",
    timestamp: new Date("2026-01-16T09:45:20"),
    duration: 180,
    personId: 12,
    personName: "Lê Văn C",
    eventType: "person_appear",
    thumbnailUrl: "/person-face-portrait-woman.jpg",
    videoUrl: "/videos/event-5.mp4",
    size: "48.5 MB",
    description: "Người xuất hiện tại Vendor Room",
  },
  {
    id: 6,
    cameraId: "cam01",
    cameraName: "CCTV 01 - Meeting Room",
    timestamp: new Date("2026-01-16T08:30:15"),
    duration: 75,
    personId: 15,
    personName: "Phạm Thị D",
    eventType: "person_appear",
    thumbnailUrl: "/person-face-portrait-man.jpg",
    videoUrl: "/videos/event-6.mp4",
    size: "20.3 MB",
    description: "Người xuất hiện tại phòng họp",
  },
]

// Statistics data
const statistics = [
  { title: "Total Events", value: "156", icon: Video, color: "text-blue-500" },
  { title: "New Persons", value: "12", icon: Users, color: "text-green-500" },
  { title: "Alerts", value: "8", icon: AlertTriangle, color: "text-red-500" },
  { title: "Storage Used", value: "2.3 GB", icon: Video, color: "text-purple-500" },
]

const eventTypeLabels = {
  person_appear: { label: "Người xuất hiện", color: "bg-blue-500/10 text-blue-500 border-blue-500/20" },
  new_person: { label: "Người mới", color: "bg-green-500/10 text-green-500 border-green-500/20" },
  person_return: { label: "Người quay lại", color: "bg-purple-500/10 text-purple-500 border-purple-500/20" },
  abnormal: { label: "Bất thường", color: "bg-red-500/10 text-red-500 border-red-500/20" },
}

export default function VideoEventsPage() {
  const [selectedDate, setSelectedDate] = useState("today")
  const [selectedCamera, setSelectedCamera] = useState("all")
  const [selectedEventType, setSelectedEventType] = useState("all")
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedVideo, setSelectedVideo] = useState<(typeof videoEvents)[0] | null>(null)

  const filteredEvents = videoEvents.filter((event) => {
    const matchCamera = selectedCamera === "all" || event.cameraId === selectedCamera
    const matchEventType = selectedEventType === "all" || event.eventType === selectedEventType
    const matchSearch =
      searchQuery === "" ||
      event.personName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      event.description.toLowerCase().includes(searchQuery.toLowerCase())
    return matchCamera && matchEventType && matchSearch
  })

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  const formatTimestamp = (date: Date) => {
    return date.toLocaleString("vi-VN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })
  }

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Video Events</h1>
            <p className="text-muted-foreground mt-1">Lưu trữ và xem lại các sự kiện quan trọng</p>
          </div>
          <div className="flex items-center gap-3">
            <Select value={selectedDate} onValueChange={setSelectedDate}>
              <SelectTrigger className="w-40">
                <Calendar className="w-4 h-4 mr-2" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="today">Hôm nay</SelectItem>
                <SelectItem value="yesterday">Hôm qua</SelectItem>
                <SelectItem value="week">7 ngày qua</SelectItem>
                <SelectItem value="month">30 ngày qua</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="outline" className="gap-2 bg-transparent">
              <Download className="w-4 h-4" />
              Export
            </Button>
          </div>
        </div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {statistics.map((stat) => (
            <Card key={stat.title} className="bg-card border-border">
              <CardContent className="p-4 flex items-center gap-3">
                <div className={`p-2.5 rounded-lg bg-secondary ${stat.color}`}>
                  <stat.icon className="w-5 h-5" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">{stat.title}</p>
                  <p className="text-xl font-bold">{stat.value}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Filters */}
        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Tìm kiếm theo tên người hoặc mô tả..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 bg-secondary border-border"
                />
              </div>
              <Select value={selectedCamera} onValueChange={setSelectedCamera}>
                <SelectTrigger className="w-48 bg-secondary">
                  <Filter className="w-4 h-4 mr-2" />
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Tất cả camera</SelectItem>
                  <SelectItem value="cam01">CCTV 01 - Meeting Room</SelectItem>
                  <SelectItem value="cam02">CCTV 02 - Open Workspace</SelectItem>
                  <SelectItem value="cam03">CCTV 03 - Vendor Room</SelectItem>
                </SelectContent>
              </Select>
              <Select value={selectedEventType} onValueChange={setSelectedEventType}>
                <SelectTrigger className="w-48 bg-secondary">
                  <Filter className="w-4 h-4 mr-2" />
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Tất cả loại sự kiện</SelectItem>
                  <SelectItem value="person_appear">Người xuất hiện</SelectItem>
                  <SelectItem value="new_person">Người mới</SelectItem>
                  <SelectItem value="person_return">Người quay lại</SelectItem>
                  <SelectItem value="abnormal">Bất thường</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        {/* Tabs for different views */}
        <Tabs defaultValue="grid" className="space-y-4">
          <TabsList className="bg-secondary">
            <TabsTrigger value="grid">Grid View</TabsTrigger>
            <TabsTrigger value="list">List View</TabsTrigger>
          </TabsList>

          {/* Grid View */}
          <TabsContent value="grid" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredEvents.map((event) => (
                <Card
                  key={event.id}
                  className={`bg-card border-border hover:border-primary/50 transition-all cursor-pointer ${
                    event.isAlert ? "border-red-500/50" : ""
                  }`}
                  onClick={() => setSelectedVideo(event)}
                >
                  <CardContent className="p-0">
                    <div className="relative aspect-video bg-secondary rounded-t-lg overflow-hidden">
                      <img
                        src={event.thumbnailUrl || "/placeholder.svg"}
                        alt={`Event ${event.id}`}
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                        <div className="bg-white/20 backdrop-blur-sm rounded-full p-3">
                          <Play className="w-6 h-6 text-white" />
                        </div>
                      </div>
                      <div className="absolute bottom-2 right-2 bg-black/70 px-2 py-1 rounded text-xs text-white font-mono">
                        {formatDuration(event.duration)}
                      </div>
                      {event.isAlert && (
                        <div className="absolute top-2 right-2 bg-red-500 px-2 py-1 rounded text-xs text-white font-medium flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3" />
                          Alert
                        </div>
                      )}
                    </div>
                    <div className="p-3 space-y-2">
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate">{event.description}</p>
                          <p className="text-xs text-muted-foreground">{event.cameraName}</p>
                        </div>
                        <Badge
                          variant="outline"
                          className={`text-[10px] shrink-0 ${
                            eventTypeLabels[event.eventType as keyof typeof eventTypeLabels].color
                          }`}
                        >
                          {eventTypeLabels[event.eventType as keyof typeof eventTypeLabels].label}
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span className="font-mono">{formatTimestamp(event.timestamp)}</span>
                        <span>{event.size}</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs">
                        <Users className="w-3 h-3 text-muted-foreground" />
                        <span className="text-muted-foreground">{event.personName}</span>
                        {event.personId && (
                          <Badge variant="secondary" className="text-[10px] ml-auto">
                            ID: {event.personId}
                          </Badge>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* List View */}
          <TabsContent value="list" className="space-y-2">
            {filteredEvents.map((event) => (
              <Card
                key={event.id}
                className={`bg-card border-border hover:border-primary/50 transition-all cursor-pointer ${
                  event.isAlert ? "border-red-500/50" : ""
                }`}
                onClick={() => setSelectedVideo(event)}
              >
                <CardContent className="p-4">
                  <div className="flex items-center gap-4">
                    <div className="relative w-32 h-20 bg-secondary rounded overflow-hidden shrink-0">
                      <img
                        src={event.thumbnailUrl || "/placeholder.svg"}
                        alt={`Event ${event.id}`}
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute inset-0 bg-black/30 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                        <Play className="w-5 h-5 text-white" />
                      </div>
                      <div className="absolute bottom-1 right-1 bg-black/70 px-1.5 py-0.5 rounded text-[10px] text-white font-mono">
                        {formatDuration(event.duration)}
                      </div>
                    </div>
                    <div className="flex-1 min-w-0 space-y-1">
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium">{event.description}</p>
                          <p className="text-xs text-muted-foreground">{event.cameraName}</p>
                        </div>
                        <Badge
                          variant="outline"
                          className={`text-[10px] shrink-0 ${
                            eventTypeLabels[event.eventType as keyof typeof eventTypeLabels].color
                          }`}
                        >
                          {eventTypeLabels[event.eventType as keyof typeof eventTypeLabels].label}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-muted-foreground">
                        <span className="font-mono">{formatTimestamp(event.timestamp)}</span>
                        <span className="flex items-center gap-1">
                          <Users className="w-3 h-3" />
                          {event.personName}
                        </span>
                        {event.personId && (
                          <Badge variant="secondary" className="text-[10px]">
                            ID: {event.personId}
                          </Badge>
                        )}
                        <span className="ml-auto">{event.size}</span>
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      className="shrink-0 gap-2"
                      onClick={(e) => {
                        e.stopPropagation()
                        setSelectedVideo(event)
                      }}
                    >
                      <Eye className="w-4 h-4" />
                      Xem
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </TabsContent>
        </Tabs>

        {/* Empty state */}
        {filteredEvents.length === 0 && (
          <Card className="bg-card border-border">
            <CardContent className="p-12 text-center">
              <Video className="w-12 h-12 text-muted-foreground mx-auto mb-3" />
              <p className="text-muted-foreground">Không tìm thấy video events nào</p>
              <p className="text-sm text-muted-foreground mt-1">Thử thay đổi bộ lọc hoặc tìm kiếm</p>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Video Player Dialog */}
      <Dialog open={selectedVideo !== null} onOpenChange={(open) => !open && setSelectedVideo(null)}>
        <DialogContent className="max-w-4xl">
          <DialogTitle className="sr-only">Video Event Player</DialogTitle>
          {selectedVideo && (
            <div className="space-y-4">
              <div className="flex items-start justify-between">
                <div className="space-y-1">
                  <h3 className="text-lg font-semibold">{selectedVideo.description}</h3>
                  <p className="text-sm text-muted-foreground">{selectedVideo.cameraName}</p>
                </div>
                <Badge
                  variant="outline"
                  className={eventTypeLabels[selectedVideo.eventType as keyof typeof eventTypeLabels].color}
                >
                  {eventTypeLabels[selectedVideo.eventType as keyof typeof eventTypeLabels].label}
                </Badge>
              </div>

              <div className="aspect-video bg-black rounded-lg overflow-hidden">
                <video controls className="w-full h-full" poster={selectedVideo.thumbnailUrl}>
                  <source src={selectedVideo.videoUrl} type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Thời gian:</span>
                    <span className="font-mono">{formatTimestamp(selectedVideo.timestamp)}</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Thời lượng:</span>
                    <span className="font-mono">{formatDuration(selectedVideo.duration)}</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Kích thước:</span>
                    <span>{selectedVideo.size}</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Camera:</span>
                    <span>{selectedVideo.cameraName}</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Người:</span>
                    <span>{selectedVideo.personName}</span>
                  </div>
                  {selectedVideo.personId && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Person ID:</span>
                      <Badge variant="secondary" className="text-xs">
                        ID: {selectedVideo.personId}
                      </Badge>
                    </div>
                  )}
                </div>
              </div>

              <div className="flex items-center gap-2 pt-2">
                <Button className="flex-1 gap-2">
                  <Download className="w-4 h-4" />
                  Tải xuống
                </Button>
                <Button variant="outline" className="flex-1">
                  Xem chi tiết
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </DashboardLayout>
  )
}
