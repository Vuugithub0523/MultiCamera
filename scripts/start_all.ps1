# Start All Services
# Master script to start the entire Multi-Camera pipeline

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Multi-Camera RTSP + AI Pipeline Startup" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Function to start service in new terminal
function Start-ServiceInTerminal {
    param(
        [string]$Name,
        [string]$Script,
        [int]$WaitSeconds = 3
    )
    
    Write-Host "Starting $Name..." -ForegroundColor Yellow
    
    $scriptPath = Join-Path $scriptDir $Script
    Start-Process powershell -ArgumentList "-NoExit", "-File", $scriptPath -WorkingDirectory $projectRoot
    
    Write-Host "  Waiting ${WaitSeconds}s for $Name to initialize..." -ForegroundColor Gray
    Start-Sleep -Seconds $WaitSeconds
    
    Write-Host "  $Name started!" -ForegroundColor Green
}

# Pre-flight checks
Write-Host "Running pre-flight checks..." -ForegroundColor Cyan
Write-Host ""

# Check Docker
try {
    docker info | Out-Null
    Write-Host "  [OK] Docker is running" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check Python
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "3\.10") {
        Write-Host "  [OK] Python 3.10 detected: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Python version: $pythonVersion (recommended: 3.10.x)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [ERROR] Python not found in PATH" -ForegroundColor Red
    exit 1
}

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "  [OK] Node.js detected: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Node.js not found in PATH" -ForegroundColor Red
    exit 1
}

# Check FFmpeg
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "  [OK] FFmpeg detected" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] FFmpeg not found in PATH" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Starting services in sequence..." -ForegroundColor Cyan
Write-Host ""

# 1. Start Infrastructure (MediaMTX + LiveKit)
Start-ServiceInTerminal -Name "MediaMTX (RTSP Server)" -Script "start_rtsp.ps1" -WaitSeconds 5
Start-ServiceInTerminal -Name "LiveKit Server" -Script "start_livekit.ps1" -WaitSeconds 5

# 2. Start AI Service
Start-ServiceInTerminal -Name "AI Service" -Script "start_ai.ps1" -WaitSeconds 10

# 3. Start Publisher
Start-ServiceInTerminal -Name "LiveKit Publisher" -Script "start_publisher.ps1" -WaitSeconds 5

# 4. Start Frontend
Start-ServiceInTerminal -Name "Frontend" -Script "start_frontend.ps1" -WaitSeconds 3

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  All services started!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Frontend:    http://localhost:5173" -ForegroundColor White
Write-Host "  AI API:      http://localhost:8080" -ForegroundColor White
Write-Host "  API Docs:    http://localhost:8080/docs" -ForegroundColor White
Write-Host "  RTSP Base:   rtsp://127.0.0.1:8554" -ForegroundColor White
Write-Host "  LiveKit:     ws://127.0.0.1:7880" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to open frontend in browser..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Start-Process "http://localhost:5173"
