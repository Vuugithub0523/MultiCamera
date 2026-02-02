# Start LiveKit Publisher
# Run after AI Service

Write-Host "Starting LiveKit Publisher..." -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$publisherDir = Join-Path $projectRoot "publisher"

Push-Location $publisherDir

try {
    # Check if node_modules exists
    if (!(Test-Path "node_modules")) {
        Write-Host "Installing npm packages..." -ForegroundColor Yellow
        npm install
    }

    # Build TypeScript
    Write-Host "Building TypeScript..." -ForegroundColor Yellow
    npm run build

    # Ensure FFmpeg is available for the Node publisher.
    # If ffmpeg is not on PATH, set FFMPEG_PATH to the winget-installed ffmpeg.exe.
    if (-not $env:FFMPEG_PATH -or $env:FFMPEG_PATH.Trim().Length -eq 0) {
        $defaultFfmpeg = "C:\\Users\\tranm\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.0.1-full_build\\bin\\ffmpeg.exe"
        if (Test-Path $defaultFfmpeg) {
            $env:FFMPEG_PATH = $defaultFfmpeg
            Write-Host "Using FFmpeg at: $env:FFMPEG_PATH" -ForegroundColor Gray
        } else {
            Write-Host "WARN: FFMPEG_PATH not set and default ffmpeg.exe not found. Ensure ffmpeg is in PATH." -ForegroundColor Yellow
        }
    }

    # Start publisher
    Write-Host "Starting LiveKit publisher..." -ForegroundColor Yellow
    npm start
}
finally {
    Pop-Location
}
