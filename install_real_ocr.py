#!/usr/bin/env python3
"""
Installation guide and script for real OCR capabilities.
This will enable actual text recognition instead of computer vision approximation.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_easyocr():
    """Install EasyOCR for real text recognition."""
    print("📦 Installing EasyOCR...")
    print("This is the best OCR engine for historical maps!")
    print()
    
    try:
        # Try to install EasyOCR
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'easyocr', '--user'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ EasyOCR installed successfully!")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def test_easyocr():
    """Test EasyOCR installation."""
    print("\n🔍 Testing EasyOCR...")
    
    try:
        import easyocr
        print("✅ EasyOCR imported successfully!")
        
        # Initialize reader
        print("🔧 Initializing EasyOCR reader...")
        reader = easyocr.Reader(['en'], gpu=False)
        print("✅ EasyOCR reader initialized!")
        
        # Test with demo image if available
        demo_path = "demo_historical_map.jpg"
        if Path(demo_path).exists():
            print(f"\n🗺️  Testing with {demo_path}...")
            
            results = reader.readtext(demo_path)
            print(f"✅ Found {len(results)} text regions:")
            
            for i, (bbox, text, confidence) in enumerate(results, 1):
                if confidence > 0.3:  # Only show confident results
                    print(f"   {i}. '{text}' (confidence: {confidence:.2f})")
            
            if not results:
                print("   No text detected - this might be normal for the demo image")
        
        return True
        
    except ImportError:
        print("❌ EasyOCR not available")
        return False
    except Exception as e:
        print(f"❌ EasyOCR test failed: {e}")
        return False

def create_real_ocr_processor():
    """Create a processor that uses real OCR."""
    
    processor_code = '''#!/usr/bin/env python3
"""
Real OCR Processor using EasyOCR
This provides actual text recognition from images.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Any

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

class RealOCRProcessor:
    """Processor that uses real OCR for text recognition."""
    
    def __init__(self):
        """Initialize with EasyOCR."""
        self.name = "RealOCRProcessor"
        
        if EASYOCR_AVAILABLE:
            print("🔧 Initializing EasyOCR...")
            self.reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)
            print("✅ EasyOCR ready!")
        else:
            print("❌ EasyOCR not available")
            self.reader = None
    
    def process_image(self, image_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process image with real OCR."""
        start_time = time.time()
        
        try:
            if not self.reader:
                return {'success': False, 'error': 'EasyOCR not available'}
            
            # Load image info
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Could not load image'}
            
            height, width = image.shape[:2]
            print(f"📐 Image loaded: {width}x{height}")
            
            # Use EasyOCR to detect and recognize text
            print("🔍 Running OCR analysis...")
            results = self.reader.readtext(image_path)
            
            # Process results
            recognized_texts = []
            detections = []
            
            confidence_threshold = options.get('confidence_threshold', 0.3)
            
            for bbox, text, confidence in results:
                if confidence < confidence_threshold:
                    continue
                
                # Convert bbox format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x = int(min(x_coords))
                y = int(min(y_coords))
                w = int(max(x_coords) - min(x_coords))
                h = int(max(y_coords) - min(y_coords))
                
                # Create detection
                detection = {
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'confidence': float(confidence),
                    'detection_type': 'phrase' if len(text.split()) > 1 else 'word'
                }
                detections.append(detection)
                
                # Create recognized text
                recognized_text = {
                    'text': text.strip(),
                    'confidence': float(confidence),
                    'language': 'en',  # Could be improved with language detection
                    'bbox': detection['bbox'],
                    'analysis_method': 'easyocr'
                }
                recognized_texts.append(recognized_text)
            
            processing_time = time.time() - start_time
            
            # Create spatial links
            linked_texts = []
            for text_result in recognized_texts:
                bbox = text_result['bbox']
                center_x = bbox['x'] + bbox['width'] / 2
                center_y = bbox['y'] + bbox['height'] / 2
                
                linked_text = {
                    'text': text_result['text'],
                    'map_coordinates': [center_x, center_y],
                    'normalized_coordinates': [center_x / width, center_y / height],
                    'spatial_confidence': text_result['confidence'] * 0.9,
                    'geographic_type': 'place_name',
                    'bbox': bbox
                }
                linked_texts.append(linked_text)
            
            results = {
                'success': True,
                'task_type': options.get('task', 'task4'),
                'processing_time': round(processing_time, 2),
                'image_info': {'width': width, 'height': height, 'path': image_path},
                'recognized_texts': recognized_texts,
                'detections': detections,
                'linked_texts': linked_texts,
                'statistics': {
                    'total_recognized': len(recognized_texts),
                    'avg_confidence': sum(t['confidence'] for t in recognized_texts) / len(recognized_texts) if recognized_texts else 0,
                    'processing_method': 'real_ocr_easyocr'
                }
            }
            
            print(f"✅ Found {len(recognized_texts)} text regions in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    processor = RealOCRProcessor()
    
    # Test with demo image
    demo_path = "demo_historical_map.jpg"
    from pathlib import Path
    
    if Path(demo_path).exists():
        result = processor.process_image(demo_path, {'task': 'task4', 'confidence_threshold': 0.3})
        
        if result['success']:
            texts = result['recognized_texts']
            print(f"\\n📍 REAL OCR RESULTS:")
            for i, text in enumerate(texts, 1):
                bbox = text['bbox']
                print(f"   {i}. '{text['text']}' at ({bbox['x']}, {bbox['y']}) - confidence: {text['confidence']:.2f}")
        else:
            print(f"❌ Processing failed: {result['error']}")
    else:
        print("💡 Demo image not found")
'''
    
    with open('real_ocr_processor.py', 'w') as f:
        f.write(processor_code)
    
    print("✅ Created real_ocr_processor.py")

def main():
    """Main installation and setup function."""
    print("🗺️  MAPSCRIBE AI - REAL OCR INSTALLATION")
    print("=" * 60)
    print("This will install EasyOCR for REAL text recognition!")
    print()
    
    print("Current status:")
    try:
        import easyocr
        print("✅ EasyOCR is already installed!")
        
        # Test it
        if test_easyocr():
            print("\n🎉 EasyOCR is working perfectly!")
            print("Your MapScribe AI can now do REAL text recognition!")
        
    except ImportError:
        print("❌ EasyOCR not installed")
        print()
        
        response = input("Do you want to install EasyOCR now? (y/n): ").lower().strip()
        
        if response == 'y':
            print("\n🚀 Installing EasyOCR...")
            
            if install_easyocr():
                print("\n🎉 Installation successful!")
                
                if test_easyocr():
                    print("\n✅ EasyOCR is working!")
                    
                    # Create the real OCR processor
                    create_real_ocr_processor()
                    
                    print("\n🎯 NEXT STEPS:")
                    print("1. Test real OCR: python real_ocr_processor.py")
                    print("2. Use in MapScribe: Update mapscribe_app.py to use RealOCRProcessor")
                    print("3. Enjoy REAL text recognition!")
                    
                else:
                    print("⚠️  Installation succeeded but testing failed")
            else:
                print("❌ Installation failed")
                print("\n💡 Manual installation:")
                print("   pip install easyocr --user")
        else:
            print("Installation cancelled.")
    
    print("\n" + "=" * 60)
    print("🔧 ALTERNATIVE: Use Tesseract OCR")
    print("=" * 60)
    print("If EasyOCR doesn't work, you can also use Tesseract:")
    print()
    print("1. Install pytesseract: pip install pytesseract")
    print("2. Install Tesseract binary:")
    print("   • Windows: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   • macOS: brew install tesseract")
    print("   • Ubuntu: sudo apt install tesseract-ocr")
    print()
    print("Both will give you REAL text recognition instead of computer vision approximation!")

if __name__ == "__main__":
    main()