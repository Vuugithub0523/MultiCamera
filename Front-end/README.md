This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

### Backend local RTSP streaming server

The frontend expects a WebSocket server at `ws://localhost:8080/ws/{type}/{cameraId}`. Run the local RTSP streaming server in the repo root to stream 3 cameras using `RTSPStreamLoader`:

```bash
python local_server.py
```

Update `config.yaml` with your three RTSP URLs. The server maps them to camera IDs `cam01`, `cam02`, and `cam03` to match the frontend camera layout.

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
 
 
 📊 PHÂN TÍCH PROJECT FRONTEND - TrackVision
🎯 Mô tả tổng quan:
Đây là hệ thống giám sát camera đa điểm với khả năng theo dõi người (Multi-Camera People Tracking) sử dụng Next.js 15 + React + TypeScript + Shadcn UI.
________________________________________
📱 CẤU TRÚC LAYOUT
DashboardLayout (Layout chính)
•	Header bar với:
o	Logo "TrackVision" + breadcrumb navigation
o	Search bar (tìm cameras, events)
o	Notification bell icon với red badge
o	User avatar dropdown menu
•	Sidebar (56px width):
o	Search box
o	3 menu items: Dashboard, Report, Configuration
o	Active state highlighting
________________________________________
🏠 1. DASHBOARD PAGE (/)
A. Header Statistics Bar:
•	🧑‍🤝‍🧑 Active: Số người đang được theo dõi (4)
•	👁️ Cameras: Trạng thái camera online (3/3)
•	🕐 Clock: Đồng hồ real-time
•	🎯 Smart Follow Mode: Switch tự động chuyển camera theo người di chuyển
B. Main Grid Layout (65%-35%):
Main Camera View (65%):
•	Hiển thị feed camera chính
•	Label ID #XX trên mỗi bounding box
•	Click để swap camera
Sidebar Cameras (35% - 2 cameras stacked):
•	2 camera nhỏ xếp dọc
•	Click để swap lên main view
•	Hover effect "Click to Swap"
C. Event Timeline (Bottom):
Thanh event cuộn ngang với các card event:
•	Appear: Người xuất hiện
•	Move: Người di chuyển
•	Alert: Cảnh báo người lạ (màu đỏ)
•	Mỗi event có:
o	Thumbnail avatar
o	Timestamp [HH:MM:SS]
o	Person ID
o	Camera location
o	Message (nếu là alert)
________________________________________
⚙️ 2. CONFIGURATION PAGE (/configuration)
Tab 1: Camera Management
•	Danh sách cameras (cards):
o	Camera name + status badge (Online/Offline)
o	RTSP URL (font mono)
o	Edit/Delete buttons
•	Chức năng:
o	✏️ Edit camera name & URL
o	🗑️ Delete camera
o	➕ Add new camera button

Tab 2: Tracking Parameters
2 cards:
Detection Settings:
•	📊 Confidence Threshold: Slider 0-100%
•	🎯 Re-ID Threshold: Slider 0-100%
Advanced Options:
•	Max Track Age (seconds)
•	Min Track Hits (number)
•	IOU Threshold (0-1)
________________________________________
📈 3. REPORT & ANALYTICS PAGE (/report)
A. Header:
•	Date selector (Today/Yesterday/Last 7/30 days)
•	📥 Export button
B. KPI Cards (4 cards ngang):
1.	👥 Total Unique Visitors: 247 (+12%)
2.	⏱️ Avg Dwell Time: 4m 32s (+8%)
3.	📈 Peak Hour: 17:00 (95 people)
4.	🔥 Active Zones: 3/3 (100%)
C. Tabs với 3 loại báo cáo:
Tab 1: Overview
•	Bar chart traffic theo giờ (6:00-21:00)
•	Trục X: giờ, Trục Y: số người
Tab 2: Heatmap
•	Camera selector dropdown
•	Heatmap overlay trên floor plan
•	Zones với màu intensity:
o	🟢 Green (thấp) → 🟡 Yellow → 🟠 Orange → 🔴 Red (cao)
•	Legend: Low → High density
•	Labels cho từng zone
Tab 3: Movement Flow
•	Sankey-style diagram với 3 cameras:
o	Camera 1 (Entrance): 85 people
o	Camera 2 (Lobby): 72 people
o	Camera 3 (Warehouse): 45 people
•	Flow arrows với số liệu:
o	Cam 1 → Cam 2: 45 people (53%)
o	Cam 2 → Cam 3: 35 people (49%)
o	Cam 1 → Cam 3: 25 people direct (29%)
________________________________________
🎨 UI COMPONENTS (Shadcn UI)
Có 13 UI components từ Shadcn:
•	Button, Card, Badge, Switch
•	Input, Label, Select, Slider
•	Avatar, Dropdown Menu, Tabs
•	Skeleton (loading states)
________________________________________
🔧 CÔNG NGHỆ SỬ DỤNG:
•	⚛️ Next.js 15 (App Router)
•	📘 TypeScript
•	🎨 Tailwind CSS
•	🧩 Radix UI (headless components)
•	📊 Recharts (biểu đồ)
•	🎭 Lucide React (icons)
•	🎨 Geist fonts (Google Fonts)
________________________________________
✨ TÍNH NĂNG NỔI BẬT:
1.	✅ Real-time tracking với bounding boxes
2.	✅ Smart Follow Mode tự động chuyển camera
3.	✅ Event timeline với thumbnails
4.	✅ Zone drawing bằng polygon
5.	✅ Heatmap analysis với intensity colors
6.	✅ Flow diagram di chuyển giữa cameras
7.	✅ Dark mode ready (theme system)
8.	✅ Loading states với Skeleton components
9.	✅ Click to swap cameras
10.	✅ Alert system cho người lạ
________________________________________
Tổng kết: Đây là một hệ thống surveillance dashboard hoàn chỉnh với khả năng quản lý camera, tracking người, phân tích báo cáo và cảnh báo thông minh! 🚀
