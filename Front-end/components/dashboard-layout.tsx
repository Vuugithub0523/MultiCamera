"use client"

import type React from "react"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Bell, Camera, BarChart3, Settings, Video } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu"

const navigation = [
  { name: "Dashboard", href: "/", icon: Camera },
  { name: "Video Events", href: "/video-events", icon: Video },
  { name: "Report", href: "/report", icon: BarChart3 },
  { name: "Configuration", href: "/configuration", icon: Settings },
]

interface DashboardLayoutProps {
  children: React.ReactNode
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  const pathname = usePathname()

  return (
    <div className="min-h-screen bg-background">
      <header className="h-14 border-b border-border bg-card px-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Camera className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold text-foreground">TrackVision</span>
          </div>
          <div className="h-4 w-px bg-border" />
          <span className="text-sm text-muted-foreground">
            Dashboard / {pathname === "/" ? "Overview" : pathname.slice(1).charAt(0).toUpperCase() + pathname.slice(2)}
          </span>
        </div>

        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" className="relative h-9 w-9">
            <Bell className="w-4 h-4 text-muted-foreground" />
            <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-red-500 rounded-full"></span>
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-9 w-9">
                <Avatar className="w-8 h-8">
                  <AvatarImage src="/placeholder.svg?height=32&width=32" />
                  <AvatarFallback className="text-xs bg-gray-200 text-gray-600">AD</AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuLabel>Admin User</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem>Profile</DropdownMenuItem>
              <DropdownMenuItem>System Settings</DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>Sign out</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      <div className="flex">
        <aside className="w-56 border-r border-border bg-sidebar-background h-[calc(100vh-3.5rem)] flex flex-col py-4 px-3">
          {/* Navigation */}
          <nav className="flex flex-col gap-1">
            {navigation.map((item) => {
              const isActive = pathname === item.href
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-sidebar-accent hover:text-foreground"
                  }`}
                >
                  <item.icon className="w-5 h-5" />
                  <span>{item.name}</span>
                </Link>
              )
            })}
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 bg-background overflow-auto h-[calc(100vh-3.5rem)]">{children}</main>
      </div>
    </div>
  )
}
