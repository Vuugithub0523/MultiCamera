#!/bin/bash
# Setup Script for Native AI Backend (Linux)
# Run with: bash setup.sh

echo "=================================="
echo "Native AI Backend - Setup Script"
echo "=================================="

# Check Python
echo -e "\nChecking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python not found. Please install Python 3.9-3.11"
    exit 1
fi

# Check CUDA (optional)
echo -e "\nChecking NVIDIA GPU..."
nvidia-smi > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "GPU detected! Will use CUDA acceleration"
else
    echo "WARNING: No GPU detected. Will use CPU (slower)"
    read -p "Continue with CPU? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    
    # Modify requirements.txt to use CPU version
    echo "Switching to CPU version of onnxruntime..."
    sed -i 's/onnxruntime-gpu/onnxruntime/g' requirements.txt
fi

# Create virtual environment
echo -e "\nCreating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo -e "\nUpgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo -e "\nInstalling dependencies..."
pip install -r requirements.txt

# Check if models exist
echo -e "\nChecking AI models..."
if [ ! -f "models/yolov4-tiny.onnx" ]; then
    echo "WARNING: Models not found in models/ directory"
    
    # Check if MultiCamera exists
    if [ -d "../MultiCamera/models/pretrained_models" ]; then
        echo "Found MultiCamera models, copying..."
        cp -r ../MultiCamera/models/pretrained_models/* ./models/
        echo "Models copied successfully!"
    else
        echo "ERROR: Please download models or copy from MultiCamera"
        echo "Required files:"
        echo "  - models/yolov4-tiny.onnx"
        echo "  - models/osnet_ain_x1_0_M.onnx"
        echo "  - models/coco.names"
    fi
else
    echo "Models found!"
fi

# Create storage directory
echo -e "\nCreating storage directory..."
mkdir -p storage

# Make scripts executable
chmod +x setup.sh
chmod +x run.sh 2>/dev/null || true

# Show next steps
echo -e "\n=================================="
echo "Setup Complete!"
echo "=================================="
echo -e "\nNext steps:"
echo "1. Edit config.py and set your camera RTSP URLs"
echo "2. Run: python main.py"
echo "3. Open browser: http://localhost:5000"
echo "4. WebSocket: ws://localhost:5000/ws/tracking/cam01"
echo -e "\nFor testing without cameras:"
echo "  export USE_VIDEO_FILES=1"
echo "  python main.py"
echo -e "\n=================================="
