# Start Frontend for Native AI Backend
Write-Host "Starting Frontend..." -ForegroundColor Green

# Navigate to frontend directory
Set-Location -Path "$PSScriptRoot\frontend"

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
    npm install
}

# Start development server
Write-Host "`nStarting Next.js development server..." -ForegroundColor Cyan
Write-Host "Frontend will be available at: http://localhost:3000" -ForegroundColor Green
Write-Host "Backend should be running at: http://localhost:3000" -ForegroundColor Yellow
Write-Host "`nPress Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "=" * 60

npm run dev
