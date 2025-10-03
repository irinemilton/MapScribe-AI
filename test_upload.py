#!/usr/bin/env python3
"""
Test script to debug upload issues
"""

import requests
import os
import sys

def test_upload(server_url="http://localhost:5000"):
    """Test the upload functionality."""
    
    print(f"🧪 Testing upload to: {server_url}")
    
    # First, check if server is running
    try:
        response = requests.get(f"{server_url}/api/status", timeout=10)
        print(f"✅ Server status: {response.status_code}")
        print(f"📊 Status response: {response.json()}")
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        return
    
    # Check debug info
    try:
        response = requests.get(f"{server_url}/debug", timeout=10)
        print(f"🐛 Debug info: {response.status_code}")
        if response.status_code == 200:
            debug_info = response.json()
            print(f"📋 Dependencies: {debug_info.get('dependencies', {})}")
            print(f"📁 Directories: {debug_info.get('directories', {})}")
    except Exception as e:
        print(f"⚠️ Debug endpoint error: {e}")
    
    # Test with a simple image file
    test_image_path = "demo_historical_map.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"⚠️ Test image not found: {test_image_path}")
        print("Creating a dummy image file...")
        
        # Create a simple test image
        try:
            from PIL import Image
            import numpy as np
            
            # Create a simple test image
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save("test_image.jpg")
            test_image_path = "test_image.jpg"
            print(f"✅ Created test image: {test_image_path}")
            
        except ImportError:
            print("❌ Cannot create test image - PIL not available")
            return
    
    # Test upload
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': (test_image_path, f, 'image/jpeg')}
            data = {
                'task': 'task4',
                'confidence': '0.5',
                'language': 'auto'
            }
            
            print(f"🔄 Uploading {test_image_path}...")
            response = requests.post(f"{server_url}/upload", files=files, data=data, timeout=30)
            
            print(f"📤 Upload response: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Upload successful!")
                print(f"📊 Results: {result.get('detections_count', 0)} detections")
                print(f"⏱️ Processing time: {result.get('processing_time', 0)}s")
            else:
                print(f"❌ Upload failed: {response.status_code}")
                try:
                    error_info = response.json()
                    print(f"🔍 Error details: {error_info}")
                except:
                    print(f"🔍 Error text: {response.text}")
                    
    except Exception as e:
        print(f"❌ Upload test failed: {e}")

if __name__ == "__main__":
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    test_upload(server_url)