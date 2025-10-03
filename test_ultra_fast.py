#!/usr/bin/env python3
"""
Test the ultra-fast processor with a sample image
"""

import os
import time
from ultra_fast_processor import UltraFastProcessor

def test_processor():
    """Test the ultra-fast processor."""
    processor = UltraFastProcessor()
    
    # Look for test images
    test_images = [
        'demo_historical_map.jpg',
        'test_image.jpg', 
        'sample.png',
        'map.jpg'
    ]
    
    for image_file in test_images:
        if os.path.exists(image_file):
            print(f"\n🧪 Testing ultra-fast processor with: {image_file}")
            
            # Get file size
            file_size = os.path.getsize(image_file) / (1024 * 1024)  # MB
            print(f"📁 File size: {file_size:.2f} MB")
            
            # Process the image
            start_time = time.time()
            result = processor.process_image(image_file)
            end_time = time.time()
            
            print(f"⏱️ Processing time: {end_time - start_time:.2f} seconds")
            print(f"✅ Success: {result.get('success', False)}")
            print(f"🔍 Detections: {len(result.get('detections', []))}")
            print(f"📊 Summary: {result.get('summary', {})}")
            
            if result.get('success'):
                print("🎉 Ultra-fast processor working correctly!")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            return result.get('success', False)
    
    print("❌ No test images found. Please add a test image file.")
    return False

if __name__ == "__main__":
    print("⚡ Ultra-Fast Processor Test")
    print("=" * 30)
    
    success = test_processor()
    
    if success:
        print("\n✅ Test passed! Ultra-fast processor is ready.")
        print("🚀 You can now use the web interface with large images.")
    else:
        print("\n❌ Test failed. Check the error messages above.")