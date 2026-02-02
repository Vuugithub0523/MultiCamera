# Start MediaMTX RTSP Server
# Run this first before starting AI service

Write-Host "Starting MediaMTX RTSP Server..." -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$infraDir = Join-Path $scriptDir "..\infra\rtsp"

Push-Location $infraDir

try {
    # Check if Docker is running
    docker info | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Docker is not running. Please start Docker first." -ForegroundColor Red
        exit 1
    }

    # Stop existing container
    docker-compose down 2>$null

    # Start MediaMTX
    Write-Host "Starting MediaMTX container..." -ForegroundColor Yellow
    docker-compose up -d

    Write-Host "MediaMTX RTSP server started!" -ForegroundColor Green
    Write-Host "RTSP endpoint: rtsp://127.0.0.1:8554" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To check status: docker logs -f mediamtx" -ForegroundColor Gray
}
finally {
    Pop-Location
}
