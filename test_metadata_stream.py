"""
Test script to verify optimized pipeline
Run this AFTER starting the backend: python main.py
"""
import sys

if "pytest" in sys.modules:
    import pytest

    pytest.skip(
        "Manual integration test; run directly after starting the backend.",
        allow_module_level=True,
    )

import asyncio
import websockets
import struct
import json
from datetime import datetime

BACKEND_URL = "ws://localhost:5000"
CAMERA_ID = "cam01"

async def test_websocket_metadata():
    """Test WebSocket connection and metadata parsing"""
    uri = f"{BACKEND_URL}/ws/tracking/{CAMERA_ID}"
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✓ Connected to {CAMERA_ID}")
            
            frame_count = 0
            total_tracks = 0
            
            print("\nReceiving frames... (Press Ctrl+C to stop)\n")
            print("-" * 80)
            
            async for message in websocket:
                frame_count += 1
                
                # Parse binary format: [4 bytes: length][metadata][frame]
                data = message
                metadata_length = struct.unpack('!I', data[:4])[0]
                
                # Extract metadata
                metadata_bytes = data[4:4+metadata_length]
                metadata_text = metadata_bytes.decode('utf-8')
                tracks = json.loads(metadata_text)
                
                # Extract frame
                frame_bytes = data[4+metadata_length:]
                frame_size_kb = len(frame_bytes) / 1024
                
                total_tracks += len(tracks)
                
                # Print summary every 10 frames
                if frame_count % 10 == 0:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] Frame {frame_count:4d} | "
                          f"Tracks: {len(tracks):2d} | "
                          f"Size: {frame_size_kb:6.1f} KB | "
                          f"Metadata: {metadata_length:4d} bytes")
                    
                    # Print track details
                    if tracks:
                        for track in tracks:
                            person_id = track.get('person_id', 'None')
                            bbox = track.get('bbox', [0,0,0,0])
                            confidence = track.get('confidence', 0)
                            state = track.get('state', 'unknown')
                            print(f"    → Person {person_id}: "
                                  f"bbox={bbox}, conf={confidence:.2f}, state={state}")
                
                # Test for 50 frames
                if frame_count >= 50:
                    break
            
            print("-" * 80)
            print(f"\n✓ Test completed successfully!")
            print(f"  Frames received: {frame_count}")
            print(f"  Total tracks: {total_tracks}")
            print(f"  Avg tracks/frame: {total_tracks/frame_count:.1f}")
            
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"✗ Connection failed: {e}")
        print(f"  Make sure backend is running: python main.py")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 80)
    print("WebSocket Metadata Test - Optimized Pipeline")
    print("=" * 80)
    print(f"\nTarget: {BACKEND_URL}/ws/tracking/{CAMERA_ID}")
    print("Expected format: [4 bytes length][metadata JSON][frame JPEG]\n")
    
    try:
        asyncio.run(test_websocket_metadata())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    
    print("\n" + "=" * 80)
