# Start Frontend
# Run after all other services

Write-Host "Starting Frontend..." -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$frontendDir = Join-Path $projectRoot "frontend"

Push-Location $frontendDir

try {
    # Check if node_modules exists
    if (!(Test-Path "node_modules")) {
        Write-Host "Installing npm packages..." -ForegroundColor Yellow
        npm install
    }

    # Start dev server
    Write-Host "Starting Vite dev server..." -ForegroundColor Yellow
    Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
    Write-Host ""
    
    npm run dev
}
finally {
    Pop-Location
}
