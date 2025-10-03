#!/usr/bin/env python3
"""
Quick OCR test to process your image directly without the web interface.
This will show you the real OCR results immediately.
"""

from real_ocr_processor import RealOCRProcessor
from pathlib import Path
import sys

def test_your_image():
    """Test OCR with your uploaded image."""
    print("🗺️  QUICK OCR TEST")
    print("=" * 50)
    
    # Initialize processor
    processor = RealOCRProcessor()
    
    if not processor.ocr_available:
        print("❌ OCR not available")
        return
    
    # Look for your uploaded image
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
        image_files.extend(Path('.').glob(f'*.{ext}'))
        image_files.extend(Path('.').glob(f'*.{ext.upper()}'))
    
    # Filter to find likely user images
    user_images = [img for img in image_files if 'kiro' in img.name.lower() or 'test' in img.name.lower()]
    
    if not user_images:
        user_images = [img for img in image_files if not img.name.startswith('demo_')]
    
    if user_images:
        test_image = user_images[0]
        print(f"🔍 Testing with your image: {test_image.name}")
        
        # Process with lower confidence to catch more text
        result = processor.process_image(str(test_image), {
            'task': 'task4',
            'confidence_threshold': 0.1  # Very low threshold to catch faint text
        })
        
        if result['success']:
            texts = result['recognized_texts']
            print(f"\n🎉 SUCCESS! Found {len(texts)} text regions in your image!")
            
            if texts:
                print("\n📍 REAL TEXT FROM YOUR IMAGE:")
                for i, text in enumerate(texts, 1):
                    bbox = text['bbox']
                    print(f"   {i}. '{text['text']}' ({text['language']})")
                    print(f"      Location: ({bbox['x']}, {bbox['y']}) - Size: {bbox['width']}x{bbox['height']}")
                    print(f"      Confidence: {text['confidence']:.3f}")
                    print()
                
                print("🎯 This is REAL text recognition from your historical map!")
                print("No more fake generated names - these are actual words from your image!")
            else:
                print("No text detected. Possible reasons:")
                print("• Text might be very faint or unclear")
                print("• Image might be too complex or noisy")
                print("• Text might be in an unsupported language")
                print("• Try with a clearer image or different settings")
        else:
            print(f"❌ Processing failed: {result['error']}")
    
    else:
        print("📁 No user images found")
        print("Upload an image and run this script again!")

if __name__ == "__main__":
    test_your_image()