#!/usr/bin/env python3
"""
Test script for the real image processor.
"""

import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

try:
    print("🔍 Testing real image processor...")
    
    # Try to import
    print("1. Importing module...")
    import real_image_processor
    print("   ✅ Module imported successfully")
    
    # Check what's in the module
    print("2. Checking module contents...")
    contents = dir(real_image_processor)
    print(f"   Module contents: {contents}")
    
    # Try to import the class
    print("3. Importing MapScribeProcessor class...")
    try:
        from real_image_processor import MapScribeProcessor
        print("   ✅ Class imported successfully")
        
        # Try to create an instance
        print("4. Creating processor instance...")
        processor = MapScribeProcessor()
        print(f"   ✅ Processor created with engines: {processor.ocr_engines}")
        
        # Try to process the demo image
        print("5. Testing image processing...")
        demo_path = "demo_historical_map.jpg"
        
        from pathlib import Path
        if Path(demo_path).exists():
            print(f"   Processing {demo_path}...")
            result = processor.process_image(demo_path, {
                'task': 'task4',
                'confidence_threshold': 0.3,
                'language': 'auto'
            })
            
            if result['success']:
                texts = result['recognized_texts']
                print(f"   ✅ Processing successful! Found {len(texts)} text regions:")
                
                for i, text in enumerate(texts[:3], 1):  # Show first 3
                    bbox = text['bbox']
                    print(f"      {i}. '{text['text']}' at ({bbox['x']}, {bbox['y']}) - confidence: {text['confidence']:.2f}")
                
                if len(texts) > 3:
                    print(f"      ... and {len(texts) - 3} more")
                    
            else:
                print(f"   ❌ Processing failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"   ⚠️  Demo image not found at {demo_path}")
            print("   Creating a simple test...")
            
            # Test with a simple synthetic image
            import cv2
            import numpy as np
            
            # Create a simple test image with text-like regions
            test_image = np.ones((400, 600, 3), dtype=np.uint8) * 240
            cv2.rectangle(test_image, (50, 100), (200, 130), (0, 0, 0), -1)  # Black rectangle (text-like)
            cv2.rectangle(test_image, (250, 150), (400, 180), (0, 0, 0), -1)  # Another rectangle
            
            test_path = "test_synthetic.jpg"
            cv2.imwrite(test_path, test_image)
            
            print(f"   Created synthetic test image: {test_path}")
            result = processor.process_image(test_path, {
                'task': 'task4',
                'confidence_threshold': 0.3,
                'language': 'auto'
            })
            
            if result['success']:
                texts = result['recognized_texts']
                print(f"   ✅ Synthetic test successful! Found {len(texts)} regions")
                for i, text in enumerate(texts, 1):
                    print(f"      {i}. '{text['text']}' - confidence: {text['confidence']:.2f}")
            else:
                print(f"   ❌ Synthetic test failed: {result.get('error', 'Unknown error')}")
        
        print("\n🎉 All tests completed!")
        
    except ImportError as e:
        print(f"   ❌ Failed to import class: {e}")
        
        # Let's check if there's a syntax error by trying to exec the file
        print("6. Checking for syntax errors...")
        try:
            with open('real_image_processor.py', 'r') as f:
                code = f.read()
            
            compile(code, 'real_image_processor.py', 'exec')
            print("   ✅ No syntax errors found")
            
            # Try to execute the code directly
            print("7. Executing code directly...")
            exec(code)
            print("   ✅ Code executed successfully")
            
        except SyntaxError as se:
            print(f"   ❌ Syntax error: {se}")
        except Exception as ee:
            print(f"   ❌ Execution error: {ee}")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Test completed!")