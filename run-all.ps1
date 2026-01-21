# Start Both Backend and Frontend
Write-Host "=" * 60 -ForegroundColor Green
Write-Host "Starting Native AI Backend System" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green

$scriptPath = $PSScriptRoot

# Function to start backend
$backendJob = Start-Job -ScriptBlock {
    param($path)
    Set-Location $path
    
    # Activate venv
    if (Test-Path "venv\Scripts\Activate.ps1") {
        . .\venv\Scripts\Activate.ps1
    }
    
    # Run backend
    python main.py
} -ArgumentList $scriptPath

Write-Host "`n[Backend] Starting in background..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Function to start frontend
$frontendJob = Start-Job -ScriptBlock {
    param($path)
    Set-Location "$path\frontend"
    
    # Check if node_modules exists
    if (-not (Test-Path "node_modules")) {
        npm install
    }
    
    # Run frontend
    npm run dev
} -ArgumentList $scriptPath

Write-Host "[Frontend] Starting in background..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host "`n" + ("=" * 60) -ForegroundColor Green
Write-Host "SYSTEM STARTED!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green
Write-Host "`nServices:" -ForegroundColor Cyan
Write-Host "  Backend:  http://localhost:5000" -ForegroundColor White
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "`nWebSocket: ws://localhost:5000/ws/tracking/{camera_id}" -ForegroundColor White
Write-Host "`nPress Ctrl+C to stop all services" -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Green

# Keep script running and show logs
try {
    while ($true) {
        # Check backend
        $backendState = Get-Job -Id $backendJob.Id
        if ($backendState.State -ne 'Running') {
            Write-Host "`n[Backend] Stopped unexpectedly!" -ForegroundColor Red
            break
        }
        
        # Check frontend
        $frontendState = Get-Job -Id $frontendJob.Id
        if ($frontendState.State -ne 'Running') {
            Write-Host "`n[Frontend] Stopped unexpectedly!" -ForegroundColor Red
            break
        }
        
        Start-Sleep -Seconds 2
    }
}
finally {
    Write-Host "`n`nStopping all services..." -ForegroundColor Yellow
    
    # Stop jobs
    Stop-Job -Id $backendJob.Id -ErrorAction SilentlyContinue
    Stop-Job -Id $frontendJob.Id -ErrorAction SilentlyContinue
    
    # Remove jobs
    Remove-Job -Id $backendJob.Id -ErrorAction SilentlyContinue
    Remove-Job -Id $frontendJob.Id -ErrorAction SilentlyContinue
    
    Write-Host "All services stopped." -ForegroundColor Green
}
