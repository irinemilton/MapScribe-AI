#!/usr/bin/env python3
"""
Quick test of the Flask app
"""

import requests
import time
import os

def test_app():
    # Start the app in background (you need to run this manually)
    base_url = "http://localhost:5000"
    
    print("Testing MapScribe AI...")
    
    # Test status endpoint
    try:
        response = requests.get(f"{base_url}/api/status", timeout=5)
        print(f"✅ Status: {response.status_code}")
        if response.status_code == 200:
            print(f"📊 Response: {response.json()}")
    except Exception as e:
        print(f"❌ Status test failed: {e}")
        return
    
    # Create a test image file
    test_content = b"fake image content for testing upload functionality"
    test_file = "test_image.jpg"
    
    with open(test_file, 'wb') as f:
        f.write(test_content)
    
    # Test upload
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'image/jpeg')}
            data = {
                'task': 'task4',
                'confidence': '0.5',
                'language': 'auto'
            }
            
            print("🔄 Testing upload...")
            response = requests.post(f"{base_url}/upload", files=files, data=data, timeout=10)
            
            print(f"📤 Upload response: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Upload successful!")
                print(f"📊 Detections: {result.get('results', {}).get('total_detections', 0)}")
                print(f"⏱️ Processing time: {result.get('results', {}).get('processing_time', 0)}s")
            else:
                print(f"❌ Upload failed: {response.status_code}")
                try:
                    error = response.json()
                    print(f"Error: {error}")
                except:
                    print(f"Error text: {response.text}")
                    
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)

if __name__ == "__main__":
    print("🧪 MapScribe AI Test")
    print("Make sure to run 'python app.py' in another terminal first!")
    input("Press Enter when the server is running...")
    test_app()