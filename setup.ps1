# Setup Script for Native AI Backend
# Run this after cloning the project

Write-Host "==================================" -ForegroundColor Green
Write-Host "Native AI Backend - Setup Script" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green

# Check Python
Write-Host "`nChecking Python installation..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Please install Python 3.9-3.11" -ForegroundColor Red
    exit 1
}

# Check CUDA (optional)
Write-Host "`nChecking NVIDIA GPU..." -ForegroundColor Yellow
nvidia-smi
if ($LASTEXITCODE -eq 0) {
    Write-Host "GPU detected! Will use CUDA acceleration" -ForegroundColor Green
} else {
    Write-Host "WARNING: No GPU detected. Will use CPU (slower)" -ForegroundColor Yellow
    
    # Ask user if they want to continue with CPU
    $response = Read-Host "Continue with CPU? (y/n)"
    if ($response -ne 'y') {
        exit 1
    }
    
    # Modify requirements.txt to use CPU version
    Write-Host "Switching to CPU version of onnxruntime..." -ForegroundColor Yellow
    (Get-Content requirements.txt) | 
        ForEach-Object { $_ -replace 'onnxruntime-gpu', 'onnxruntime' } | 
        Set-Content requirements.txt
}

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Check if models exist
Write-Host "`nChecking AI models..." -ForegroundColor Yellow
$modelsExist = Test-Path "models/yolov4-tiny.onnx"
if (-not $modelsExist) {
    Write-Host "WARNING: Models not found in models/ directory" -ForegroundColor Yellow
    
    # Check if MultiCamera exists
    $multiCameraPath = "..\MultiCamera\models\pretrained_models"
    if (Test-Path $multiCameraPath) {
        Write-Host "Found MultiCamera models, copying..." -ForegroundColor Green
        Copy-Item -Recurse "$multiCameraPath\*" ".\models\"
        Write-Host "Models copied successfully!" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Please download models or copy from MultiCamera" -ForegroundColor Red
        Write-Host "Required files:" -ForegroundColor Red
        Write-Host "  - models/yolov4-tiny.onnx" -ForegroundColor Red
        Write-Host "  - models/osnet_ain_x1_0_M.onnx" -ForegroundColor Red
        Write-Host "  - models/coco.names" -ForegroundColor Red
    }
} else {
    Write-Host "Models found!" -ForegroundColor Green
}

# Create storage directory
Write-Host "`nCreating storage directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "storage" | Out-Null

# Show next steps
Write-Host "`n==================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Edit config.py and set your camera RTSP URLs" -ForegroundColor Cyan
Write-Host "2. Run: python main.py" -ForegroundColor Cyan
Write-Host "3. Open browser: http://localhost:3000" -ForegroundColor Cyan
Write-Host "4. WebSocket: ws://localhost:3000/ws/tracking/cam01" -ForegroundColor Cyan
Write-Host "`nFor testing without cameras:" -ForegroundColor Yellow
Write-Host "  `$env:USE_VIDEO_FILES=1" -ForegroundColor Cyan
Write-Host "  python main.py" -ForegroundColor Cyan
Write-Host "`n==================================" -ForegroundColor Green
