"use client"

import { useState, useEffect } from "react"
import { Users, Clock, TrendingUp, Calendar, Download, Flame } from "lucide-react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { DashboardLayout } from "@/components/dashboard-layout"

const API_BASE_URL = "http://localhost:8080"

interface ReportStats {
  total_unique_visitors: number
  avg_dwell_time_seconds: number
  peak_hour: string
  peak_hour_count: number
  active_cameras: number
  total_cameras: number
  hourly_traffic: { hour: string; count: number }[]
  camera_flow: {
    totals: { [key: string]: number }
    transitions: { [key: string]: number }
  }
}

// Camera names mapping
const cameraNames: { [key: string]: string } = {
  cam01: "Camera 1 - Entrance",
  cam02: "Camera 2 - Lobby",
  cam03: "Camera 3 - Warehouse",
}

export default function ReportPage() {
  const [selectedDate, setSelectedDate] = useState("today")
  const [stats, setStats] = useState<ReportStats | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/report/stats`)
        const data = await response.json()
        if (!data.error) {
          setStats(data)
        }
      } catch (error) {
        console.error("Failed to fetch report stats:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchStats()
    const interval = setInterval(fetchStats, 5000) // Refresh every 5 seconds

    return () => clearInterval(interval)
  }, [])

  const formatDwellTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}m ${secs}s`
  }

  const kpiData = stats
    ? [
        {
          title: "Total Unique Visitors",
          value: stats.total_unique_visitors.toString(),
          change: "+12%",
          icon: Users,
        },
        {
          title: "Avg Dwell Time",
          value: formatDwellTime(stats.avg_dwell_time_seconds),
          change: "+8%",
          icon: Clock,
        },
        {
          title: "Peak Hour",
          value: stats.peak_hour,
          change: `${stats.peak_hour_count} people`,
          icon: TrendingUp,
        },
        {
          title: "Active Zones",
          value: `${stats.active_cameras}/${stats.total_cameras}`,
          change: "100%",
          icon: Flame,
        },
      ]
    : []

  // Extract camera flow data
  const getCameraFlowData = () => {
    if (!stats) return { cameras: [], transitions: [] }

    const cameras = Object.entries(stats.camera_flow.totals).map(([camId, count]) => ({
      id: camId,
      name: cameraNames[camId] || camId,
      count: count,
    }))

    const transitions = Object.entries(stats.camera_flow.transitions).map(([key, count]) => {
      const [from, to] = key.split("->")
      return { from, to, count }
    })

    return { cameras, transitions }
  }

  const { cameras, transitions } = getCameraFlowData()

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
          {loading ? (
            <>
              {[1, 2, 3, 4].map((i) => (
                <Card key={i} className="bg-card border-border">
                  <CardContent className="p-4">
                    <div className="animate-pulse">
                      <div className="h-10 w-10 bg-secondary rounded-lg mb-3" />
                      <div className="h-8 bg-secondary rounded mb-2" />
                      <div className="h-4 bg-secondary rounded" />
                    </div>
                  </CardContent>
                </Card>
              ))}
            </>
          ) : (
            kpiData.map((kpi, index) => (
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
            ))
          )}
        </div>

        {/* Tabs */}
        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList className="bg-secondary">
            <TabsTrigger value="overview">Overview</TabsTrigger>
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
                {loading ? (
                  <div className="h-80 flex items-center justify-center">
                    <div className="text-muted-foreground">Loading data...</div>
                  </div>
                ) : stats && stats.hourly_traffic.length > 0 ? (
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={stats.hourly_traffic}>
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
                ) : (
                  <div className="h-80 flex items-center justify-center">
                    <div className="text-muted-foreground">No data available yet</div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Tab 2: Movement Flow */}
          <TabsContent value="flow" className="space-y-4">
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle>Movement Flow</CardTitle>
                <CardDescription>Traffic flow between cameras</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="py-8 flex items-center justify-center">
                    <div className="text-muted-foreground">Loading flow data...</div>
                  </div>
                ) : cameras.length > 0 ? (
                  <>
                    <div className="grid grid-cols-3 gap-6 py-8">
                      {cameras.map((camera, idx) => (
                        <div key={camera.id} className="flex flex-col items-center">
                          <div
                            className={`w-24 h-24 rounded-full border-2 flex items-center justify-center mb-3 ${
                              idx === 0
                                ? "bg-primary/20 border-primary"
                                : idx === 1
                                ? "bg-chart-2/20 border-chart-2"
                                : "bg-chart-3/20 border-chart-3"
                            }`}
                          >
                            <div className="text-center">
                              <div
                                className={`text-2xl font-bold ${
                                  idx === 0 ? "text-primary" : idx === 1 ? "text-chart-2" : "text-chart-3"
                                }`}
                              >
                                {camera.count}
                              </div>
                              <div className="text-xs text-muted-foreground">people</div>
                            </div>
                          </div>
                          <div className="text-sm font-medium">{camera.name.split(" - ")[0]}</div>
                          <div className="text-xs text-muted-foreground">{camera.name.split(" - ")[1]}</div>
                        </div>
                      ))}
                    </div>

                    {/* Flow Summary */}
                    {transitions.length > 0 && (
                      <div className="grid grid-cols-3 gap-4 mt-8 pt-6 border-t border-border">
                        {transitions.slice(0, 3).map((trans, idx) => (
                          <div key={idx} className="text-center">
                            <div className="text-lg font-bold text-primary">{trans.count}</div>
                            <div className="text-xs text-muted-foreground">
                              {cameraNames[trans.from]?.split(" - ")[0] || trans.from} →{" "}
                              {cameraNames[trans.to]?.split(" - ")[0] || trans.to}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="py-8 flex items-center justify-center">
                    <div className="text-muted-foreground">No movement data available yet</div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </DashboardLayout>
  )
}
