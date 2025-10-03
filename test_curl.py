#!/usr/bin/env python3
"""
Test the upload with Python requests
"""

import requests
import os
import sys

def test_server(base_url="http://localhost:5000"):
    print(f"Testing server at: {base_url}")
    
    # Test basic endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Root endpoint: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False
    
    # Test /test endpoint
    try:
        response = requests.get(f"{base_url}/test")
        print(f"✅ Test endpoint: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Test endpoint failed: {e}")
    
    # Create a simple test file
    test_content = b"fake image content for testing"
    test_filename = "test.jpg"
    
    with open(test_filename, 'wb') as f:
        f.write(test_content)
    
    # Test upload
    try:
        with open(test_filename, 'rb') as f:
            files = {'file': (test_filename, f, 'image/jpeg')}
            
            print(f"🔄 Testing upload...")
            response = requests.post(f"{base_url}/upload", files=files)
            
            print(f"Upload response: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Upload successful: {result}")
            else:
                print(f"❌ Upload failed: {response.status_code}")
                try:
                    error = response.json()
                    print(f"Error details: {error}")
                except:
                    print(f"Error text: {response.text}")
                    
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
    
    # Clean up
    if os.path.exists(test_filename):
        os.remove(test_filename)

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    test_server(url)