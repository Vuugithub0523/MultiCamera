# Start LiveKit Server
# Run this after MediaMTX

Write-Host "Starting LiveKit Server..." -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$infraDir = Join-Path $scriptDir "..\infra\livekit"

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

    # Start LiveKit
    Write-Host "Starting LiveKit container..." -ForegroundColor Yellow
    docker-compose up -d

    Write-Host "LiveKit server started!" -ForegroundColor Green
    Write-Host "WebSocket endpoint: ws://127.0.0.1:7880" -ForegroundColor Cyan
    Write-Host "API Key: devkey" -ForegroundColor Cyan
    Write-Host "API Secret: devsecret" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To check status: docker logs -f livekit" -ForegroundColor Gray
}
finally {
    Pop-Location
}
