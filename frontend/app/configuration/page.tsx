"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { useSearchParams } from "next/navigation"
import { Camera, Save, Trash2, Edit2, Plus, Check, X, Wifi, WifiOff } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DashboardLayout } from "@/components/dashboard-layout"

// Mock camera data
const initialCameras = [
  { id: 1, name: "Camera 1 - Entrance", url: "rtsp://192.168.1.101:554/stream1", status: "online" },
  { id: 2, name: "Camera 2 - Lobby", url: "rtsp://192.168.1.102:554/stream1", status: "online" },
  { id: 3, name: "Camera 3 - Warehouse", url: "rtsp://192.168.1.103:554/stream1", status: "offline" },
]

export default function ConfigurationPage() {
  const searchParams = useSearchParams()
  const [cameras, setCameras] = useState(initialCameras)
  const [editingCamera, setEditingCamera] = useState<number | null>(null)
  const [editedName, setEditedName] = useState("")
  const [editedUrl, setEditedUrl] = useState("")

  // Tracking parameters
  const [confidenceThreshold, setConfidenceThreshold] = useState([0.6])
  const [reIdThreshold, setReIdThreshold] = useState([0.7])

  // Tự động mở edit mode khi có camera ID từ URL
  useEffect(() => {
    const cameraId = searchParams.get("camera")
    if (cameraId) {
      const camera = cameras.find((c) => c.id === Number(cameraId))
      if (camera) {
        startEditing(camera)
      }
    }
  }, [searchParams])

  const startEditing = (camera: (typeof cameras)[0]) => {
    setEditingCamera(camera.id)
    setEditedName(camera.name)
    setEditedUrl(camera.url)
  }

  const saveEditing = () => {
    setCameras(cameras.map((c) => (c.id === editingCamera ? { ...c, name: editedName, url: editedUrl } : c)))
    setEditingCamera(null)
  }

  const cancelEditing = () => {
    setEditingCamera(null)
    setEditedName("")
    setEditedUrl("")
  }

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Configuration</h1>
            <p className="text-muted-foreground mt-1">Manage cameras and tracking parameters</p>
          </div>
        </div>

        <Tabs defaultValue="cameras" className="space-y-4">
          <TabsList className="bg-secondary">
            <TabsTrigger value="cameras">Camera Management</TabsTrigger>
            <TabsTrigger value="parameters">Tracking Parameters</TabsTrigger>
          </TabsList>

          {/* Camera Management Tab */}
          <TabsContent value="cameras" className="space-y-4">
            <div className="grid gap-4">
              {cameras.map((camera) => (
                <Card key={camera.id} className="bg-card border-border">
                  <CardContent className="p-4">
                    {editingCamera === camera.id ? (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label htmlFor="name">Camera Name</Label>
                            <Input
                              id="name"
                              value={editedName}
                              onChange={(e) => setEditedName(e.target.value)}
                              className="bg-secondary"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="url">RTSP URL</Label>
                            <Input
                              id="url"
                              value={editedUrl}
                              onChange={(e) => setEditedUrl(e.target.value)}
                              className="bg-secondary font-mono text-sm"
                            />
                          </div>
                        </div>
                        <div className="flex justify-end gap-2">
                          <Button variant="ghost" size="sm" onClick={cancelEditing}>
                            <X className="w-4 h-4 mr-1" />
                            Cancel
                          </Button>
                          <Button size="sm" onClick={saveEditing}>
                            <Check className="w-4 h-4 mr-1" />
                            Save
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div className="w-12 h-12 bg-secondary rounded-lg flex items-center justify-center">
                            <Camera className="w-6 h-6 text-primary" />
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="font-medium">{camera.name}</span>
                              <Badge
                                variant={camera.status === "online" ? "default" : "destructive"}
                                className={camera.status === "online" ? "bg-green-600" : ""}
                              >
                                {camera.status === "online" ? (
                                  <>
                                    <Wifi className="w-3 h-3 mr-1" /> Online
                                  </>
                                ) : (
                                  <>
                                    <WifiOff className="w-3 h-3 mr-1" /> Offline
                                  </>
                                )}
                              </Badge>
                            </div>
                            <div className="text-sm text-muted-foreground font-mono mt-1">{camera.url}</div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button variant="ghost" size="icon" onClick={() => startEditing(camera)}>
                            <Edit2 className="w-4 h-4" />
                          </Button>
                          <Button variant="ghost" size="icon" className="text-destructive hover:text-destructive">
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}

              <Button variant="outline" className="w-full border-dashed bg-transparent">
                <Plus className="w-4 h-4 mr-2" />
                Add New Camera
              </Button>
            </div>
          </TabsContent>

          {/* Tracking Parameters Tab */}
          <TabsContent value="parameters" className="space-y-4">
            <div className="grid grid-cols-2 gap-6">
              <Card className="bg-card border-border">
                <CardHeader>
                  <CardTitle>Detection Settings</CardTitle>
                  <CardDescription>Configure person detection parameters</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Label>Confidence Threshold</Label>
                      <span className="text-sm font-mono text-primary">
                        {(confidenceThreshold[0] * 100).toFixed(0)}%
                      </span>
                    </div>
                    <Slider
                      value={confidenceThreshold}
                      onValueChange={setConfidenceThreshold}
                      min={0.1}
                      max={1}
                      step={0.05}
                      className="w-full"
                    />
                    <p className="text-xs text-muted-foreground">
                      Only detect persons with confidence above this threshold
                    </p>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Label>Re-ID Threshold</Label>
                      <span className="text-sm font-mono text-primary">{(reIdThreshold[0] * 100).toFixed(0)}%</span>
                    </div>
                    <Slider
                      value={reIdThreshold}
                      onValueChange={setReIdThreshold}
                      min={0.1}
                      max={1}
                      step={0.05}
                      className="w-full"
                    />
                    <p className="text-xs text-muted-foreground">Sensitivity for matching persons across cameras</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-card border-border">
                <CardHeader>
                  <CardTitle>Advanced Options</CardTitle>
                  <CardDescription>Fine-tune tracking behavior</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-secondary rounded-lg">
                    <div>
                      <div className="font-medium text-sm">Max Track Age</div>
                      <div className="text-xs text-muted-foreground">Seconds to keep track without detection</div>
                    </div>
                    <Input type="number" defaultValue={30} className="w-20 bg-background text-right" />
                  </div>

                  <div className="flex items-center justify-between p-3 bg-secondary rounded-lg">
                    <div>
                      <div className="font-medium text-sm">Min Track Hits</div>
                      <div className="text-xs text-muted-foreground">Detections needed to confirm track</div>
                    </div>
                    <Input type="number" defaultValue={3} className="w-20 bg-background text-right" />
                  </div>

                  <div className="flex items-center justify-between p-3 bg-secondary rounded-lg">
                    <div>
                      <div className="font-medium text-sm">IOU Threshold</div>
                      <div className="text-xs text-muted-foreground">Overlap ratio for box matching</div>
                    </div>
                    <Input type="number" defaultValue={0.3} step={0.1} className="w-20 bg-background text-right" />
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="flex justify-end">
              <Button className="gap-2">
                <Save className="w-4 h-4" />
                Save Configuration
              </Button>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </DashboardLayout>
  )
}
