#!/usr/bin/env python3
"""
Test the complete smart system
"""

import requests
import json
import os

def test_api_status():
    """Test if the API is running."""
    try:
        response = requests.get('http://localhost:5000/api/status', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data.get('status', 'unknown')}")
            print(f"🔧 Processor: {data.get('name', 'MapScribe AI')}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_file_upload():
    """Test file upload with a small test image."""
    # Create a simple test image if none exists
    test_files = ['demo_historical_map.jpg', 'test_image.jpg', 'sample.png']
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"🧪 Testing upload with: {test_file}")
            
            try:
                with open(test_file, 'rb') as f:
                    files = {'file': f}
                    data = {
                        'task': 'task4',
                        'confidence': '0.3',
                        'language': 'auto'
                    }
                    
                    response = requests.post(
                        'http://localhost:5000/upload',
                        files=files,
                        data=data,
                        timeout=60
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        print(f"✅ Upload successful!")
                        print(f"📊 Total places: {result.get('recognized_count', 0)}")
                        print(f"⏱️ Processing time: {result.get('processing_time', 0):.2f}s")
                        
                        if result.get('smart_results_url'):
                            print(f"🌍 Smart results URL: http://localhost:5000{result['smart_results_url']}")
                        
                        return True
                    else:
                        print(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
                        return False
                else:
                    print(f"❌ Upload request failed: {response.status_code}")
                    print(f"Response: {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Upload test failed: {e}")
                return False
    
    print("❌ No test images found for upload test")
    return False

def main():
    """Run all tests."""
    print("🧪 Testing Smart Geographic System")
    print("=" * 40)
    
    # Test 1: API Status
    print("\n1. Testing API Status...")
    api_ok = test_api_status()
    
    if not api_ok:
        print("\n❌ API is not running. Please start the server first:")
        print("python start_web_server.py")
        return
    
    # Test 2: File Upload
    print("\n2. Testing File Upload...")
    upload_ok = test_file_upload()
    
    # Summary
    print("\n" + "=" * 40)
    if api_ok and upload_ok:
        print("🎉 All tests passed! Smart system is working correctly.")
        print("🌐 Open http://localhost:5000 to use the web interface")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()