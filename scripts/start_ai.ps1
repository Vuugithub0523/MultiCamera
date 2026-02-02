# Start AI Service
# Run after infrastructure (MediaMTX, LiveKit)

Write-Host "Starting AI Service..." -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$aiServiceDir = Join-Path $projectRoot "ai-service"

Push-Location $aiServiceDir

try {
    # Check Python version
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion" -ForegroundColor Gray

    # Check if venv exists
    $venvPath = Join-Path $aiServiceDir ".venv"
    if (!(Test-Path $venvPath)) {
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        python -m venv .venv
    }

    # Activate venv
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    . $activateScript

    # Install requirements if needed
    $installed = pip list 2>$null | Select-String "fastapi"
    if (!$installed) {
        Write-Host "Installing requirements..." -ForegroundColor Yellow
        pip install -r requirements.txt
    }

    # Ensure FFmpeg is available for AI restream.
    if (-not $env:FFMPEG_PATH -or $env:FFMPEG_PATH.Trim().Length -eq 0) {
        $defaultFfmpeg = "C:\\Users\\tranm\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.0.1-full_build\\bin\\ffmpeg.exe"
        if (Test-Path $defaultFfmpeg) {
            $env:FFMPEG_PATH = $defaultFfmpeg
            Write-Host "Using FFmpeg at: $env:FFMPEG_PATH" -ForegroundColor Gray
        } else {
            Write-Host "WARN: FFMPEG_PATH not set and default ffmpeg.exe not found. Ensure ffmpeg is in PATH." -ForegroundColor Yellow
        }
    }

    # Start AI service
    Write-Host "Starting AI service on port 8080..." -ForegroundColor Yellow
    Write-Host "API docs: http://localhost:8080/docs" -ForegroundColor Cyan
    Write-Host ""
    
    # Run with uvicorn
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
}
finally {
    Pop-Location
}
