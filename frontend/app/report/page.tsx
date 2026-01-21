"use client"

import { useState } from "react"
import { Users, Clock, TrendingUp, Calendar, Download, Flame } from "lucide-react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { DashboardLayout } from "@/components/dashboard-layout"

// Hourly traffic data
const hourlyData = [
  { hour: "06:00", count: 12 },
  { hour: "07:00", count: 28 },
  { hour: "08:00", count: 45 },
  { hour: "09:00", count: 62 },
  { hour: "10:00", count: 78 },
  { hour: "11:00", count: 85 },
  { hour: "12:00", count: 92 },
  { hour: "13:00", count: 88 },
  { hour: "14:00", count: 75 },
  { hour: "15:00", count: 68 },
  { hour: "16:00", count: 72 },
  { hour: "17:00", count: 95 },
  { hour: "18:00", count: 82 },
  { hour: "19:00", count: 55 },
  { hour: "20:00", count: 32 },
  { hour: "21:00", count: 18 },
]

// Flow data for Sankey diagram
const flowData = {
  nodes: [{ name: "Camera 1\n(Entrance)" }, { name: "Camera 2\n(Lobby)" }, { name: "Camera 3\n(Warehouse)" }],
  links: [
    { source: 0, target: 1, value: 45 },
    { source: 0, target: 2, value: 25 },
    { source: 1, target: 2, value: 35 },
    { source: 1, target: 0, value: 15 },
    { source: 2, target: 1, value: 20 },
    { source: 2, target: 0, value: 30 },
  ],
}

// KPI cards data
const kpiData = [
  { title: "Total Unique Visitors", value: "247", change: "+12%", icon: Users },
  { title: "Avg Dwell Time", value: "4m 32s", change: "+8%", icon: Clock },
  { title: "Peak Hour", value: "17:00", change: "95 people", icon: TrendingUp },
  { title: "Active Zones", value: "3/3", change: "100%", icon: Flame },
]

// Heatmap zones
const heatmapZones = [
  { id: 1, x: 10, y: 20, w: 25, h: 30, intensity: 0.9, label: "Entrance" },
  { id: 2, x: 45, y: 15, w: 20, h: 25, intensity: 0.7, label: "Counter" },
  { id: 3, x: 70, y: 40, w: 25, h: 35, intensity: 0.5, label: "Storage" },
  { id: 4, x: 20, y: 60, w: 30, h: 25, intensity: 0.3, label: "Exit" },
]

export default function ReportPage() {
  const [selectedDate, setSelectedDate] = useState("today")
  const [selectedCamera, setSelectedCamera] = useState("1")

  const getHeatColor = (intensity: number) => {
    if (intensity > 0.8) return "rgba(239, 68, 68, 0.6)"
    if (intensity > 0.6) return "rgba(249, 115, 22, 0.5)"
    if (intensity > 0.4) return "rgba(234, 179, 8, 0.4)"
    return "rgba(34, 197, 94, 0.3)"
  }

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Report & Analytics</h1>
            <p className="text-muted-foreground mt-1">Traffic analysis and movement patterns</p>
          </div>
          <div className="flex items-center gap-3">
            <Select value={selectedDate} onValueChange={setSelectedDate}>
              <SelectTrigger className="w-40">
                <Calendar className="w-4 h-4 mr-2" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="today">Today</SelectItem>
                <SelectItem value="yesterday">Yesterday</SelectItem>
                <SelectItem value="week">Last 7 days</SelectItem>
                <SelectItem value="month">Last 30 days</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="outline" className="gap-2 bg-transparent">
              <Download className="w-4 h-4" />
              Export
            </Button>
          </div>
        </div>

        {/* KPI Cards */}
        <div className="grid grid-cols-4 gap-4">
          {kpiData.map((kpi, index) => (
            <Card key={index} className="bg-card border-border">
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center">
                    <kpi.icon className="w-5 h-5 text-primary" />
                  </div>
                  <span className="text-xs text-green-500 font-medium">{kpi.change}</span>
                </div>
                <div className="text-2xl font-bold mb-1">{kpi.value}</div>
                <div className="text-sm text-muted-foreground">{kpi.title}</div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Tabs */}
        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList className="bg-secondary">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
            <TabsTrigger value="flow">Movement Flow</TabsTrigger>
          </TabsList>

          {/* Tab 1: Overview */}
          <TabsContent value="overview" className="space-y-4">
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle>Hourly Traffic</CardTitle>
                <CardDescription>Unique visitors per hour</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={hourlyData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="hour" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} />
                      <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                          color: "hsl(var(--foreground))",
                        }}
                        formatter={(value: number) => [`${value} people`, "Visitors"]}
                      />
                      <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Tab 2: Heatmap */}
          <TabsContent value="heatmap" className="space-y-4">
            <Card className="bg-card border-border">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle>Density Heatmap</CardTitle>
                  <CardDescription>Area density based on dwell time</CardDescription>
                </div>
                <Select value={selectedCamera} onValueChange={setSelectedCamera}>
                  <SelectTrigger className="w-40">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">Camera 1 - Entrance</SelectItem>
                    <SelectItem value="2">Camera 2 - Lobby</SelectItem>
                    <SelectItem value="3">Camera 3 - Warehouse</SelectItem>
                  </SelectContent>
                </Select>
              </CardHeader>
              <CardContent>
                <div className="relative aspect-video bg-secondary rounded-lg overflow-hidden">
                  {/* Background Image */}
                  <img
                    src={`/floor-plan-camera-.jpg?height=400&width=700&query=floor plan camera ${selectedCamera} area top view`}
                    alt="Floor plan"
                    className="w-full h-full object-cover opacity-50"
                  />

                  {/* Heatmap Zones */}
                  {heatmapZones.map((zone) => (
                    <div
                      key={zone.id}
                      className="absolute rounded-lg transition-all hover:scale-105 cursor-pointer"
                      style={{
                        left: `${zone.x}%`,
                        top: `${zone.y}%`,
                        width: `${zone.w}%`,
                        height: `${zone.h}%`,
                        backgroundColor: getHeatColor(zone.intensity),
                        backdropFilter: "blur(4px)",
                      }}
                    >
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-xs font-medium text-white drop-shadow-lg">{zone.label}</span>
                      </div>
                    </div>
                  ))}

                  {/* Legend */}
                  <div className="absolute bottom-4 right-4 bg-card/90 rounded-lg p-3 border border-border">
                    <div className="text-xs font-medium mb-2">Density</div>
                    <div className="flex items-center gap-2">
                      <div className="flex">
                        <div className="w-6 h-3 rounded-l" style={{ backgroundColor: "rgba(34, 197, 94, 0.6)" }} />
                        <div className="w-6 h-3" style={{ backgroundColor: "rgba(234, 179, 8, 0.6)" }} />
                        <div className="w-6 h-3" style={{ backgroundColor: "rgba(249, 115, 22, 0.6)" }} />
                        <div className="w-6 h-3 rounded-r" style={{ backgroundColor: "rgba(239, 68, 68, 0.6)" }} />
                      </div>
                      <span className="text-xs text-muted-foreground">Low → High</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Tab 3: Movement Flow */}
          <TabsContent value="flow" className="space-y-4">
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle>Movement Flow</CardTitle>
                <CardDescription>Traffic flow between cameras</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-6 py-8">
                  {/* Camera 1 */}
                  <div className="flex flex-col items-center">
                    <div className="w-24 h-24 rounded-full bg-primary/20 border-2 border-primary flex items-center justify-center mb-3">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-primary">85</div>
                        <div className="text-xs text-muted-foreground">people</div>
                      </div>
                    </div>
                    <div className="text-sm font-medium">Camera 1</div>
                    <div className="text-xs text-muted-foreground">Entrance</div>
                  </div>

                  {/* Camera 2 */}
                  <div className="flex flex-col items-center">
                    <div className="w-24 h-24 rounded-full bg-chart-2/20 border-2 border-chart-2 flex items-center justify-center mb-3">
                      <div className="text-center">
                        <div className="text-2xl font-bold" style={{ color: "hsl(var(--chart-2))" }}>
                          72
                        </div>
                        <div className="text-xs text-muted-foreground">people</div>
                      </div>
                    </div>
                    <div className="text-sm font-medium">Camera 2</div>
                    <div className="text-xs text-muted-foreground">Lobby</div>
                  </div>

                  {/* Camera 3 */}
                  <div className="flex flex-col items-center">
                    <div className="w-24 h-24 rounded-full bg-chart-3/20 border-2 border-chart-3 flex items-center justify-center mb-3">
                      <div className="text-center">
                        <div className="text-2xl font-bold" style={{ color: "hsl(var(--chart-3))" }}>
                          45
                        </div>
                        <div className="text-xs text-muted-foreground">people</div>
                      </div>
                    </div>
                    <div className="text-sm font-medium">Camera 3</div>
                    <div className="text-xs text-muted-foreground">Warehouse</div>
                  </div>
                </div>

                {/* Flow Arrows */}
                <div className="relative h-32 mx-8">
                  {/* Cam 1 -> Cam 2 */}
                  <div className="absolute top-4 left-[15%] w-[35%] flex flex-col items-center">
                    <div className="w-full h-1 bg-gradient-to-r from-primary to-chart-2 rounded" />
                    <span className="text-xs mt-1 text-muted-foreground">45 people →</span>
                  </div>

                  {/* Cam 2 -> Cam 3 */}
                  <div className="absolute top-4 left-[50%] w-[35%] flex flex-col items-center">
                    <div className="w-full h-1 bg-gradient-to-r from-chart-2 to-chart-3 rounded" />
                    <span className="text-xs mt-1 text-muted-foreground">35 people →</span>
                  </div>

                  {/* Cam 1 -> Cam 3 (direct) */}
                  <div className="absolute top-16 left-[15%] w-[70%] flex flex-col items-center">
                    <div className="w-full h-0.5 bg-gradient-to-r from-primary to-chart-3 rounded opacity-50" />
                    <span className="text-xs mt-1 text-muted-foreground">25 people (direct)</span>
                  </div>
                </div>

                {/* Summary Stats */}
                <div className="grid grid-cols-3 gap-4 mt-8 pt-6 border-t border-border">
                  <div className="text-center">
                    <div className="text-lg font-bold text-primary">53%</div>
                    <div className="text-xs text-muted-foreground">Cam 1 → Cam 2</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold" style={{ color: "hsl(var(--chart-2))" }}>
                      49%
                    </div>
                    <div className="text-xs text-muted-foreground">Cam 2 → Cam 3</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold" style={{ color: "hsl(var(--chart-3))" }}>
                      29%
                    </div>
                    <div className="text-xs text-muted-foreground">Cam 1 → Cam 3</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </DashboardLayout>
  )
}
