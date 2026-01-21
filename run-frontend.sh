#!/bin/bash
# Start Frontend for Native AI Backend
echo "Starting Frontend..."

# Navigate to frontend directory
cd "$(dirname "$0")/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo ""
    echo "Installing dependencies..."
    npm install
fi

# Start development server
echo ""
echo "Starting Next.js development server..."
echo "Frontend will be available at: http://localhost:3000"
echo "Backend should be running at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop"
echo "============================================================"

npm run dev
