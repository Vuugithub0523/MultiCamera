#!/usr/bin/env pwsh
# Restart Backend Script

Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "Restarting Native AI Backend..." -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

# Stop running Python processes for backend
Write-Host "[1/2] Stopping old backend..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*python*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Start new backend
Write-Host "[2/2] Starting backend with new config..." -ForegroundColor Yellow
python main.py
