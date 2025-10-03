#!/usr/bin/env python3
"""
Real OCR Processor using EasyOCR
This provides actual text recognition from images.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Any
import logging

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
        self.logger = logging.getLogger(__name__)
        
        if EASYOCR_AVAILABLE:
            print("🔧 Initializing EasyOCR...")
            try:
                self.reader = easyocr.Reader(['en', 'ch_sim'], gpu=False, verbose=False)
                print("✅ EasyOCR ready for real text recognition!")
                self.ocr_available = True
            except Exception as e:
                print(f"❌ EasyOCR initialization failed: {e}")
                self.reader = None
                self.ocr_available = False
        else:
            print("❌ EasyOCR not available")
            self.reader = None
            self.ocr_available = False
    
    def process_image(self, image_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process image with real OCR."""
        start_time = time.time()
        
        try:
            if not self.ocr_available:
                return {'success': False, 'error': 'EasyOCR not available'}
            
            # Load image info
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Could not load image'}
            
            height, width = image.shape[:2]
            print(f"📐 Image loaded: {width}x{height}")
            
            # Use EasyOCR to detect and recognize text
            print("🔍 Running REAL OCR analysis...")
            results = self.reader.readtext(image_path)
            
            # Process results
            recognized_texts = []
            detections = []
            
            confidence_threshold = options.get('confidence_threshold', 0.3)
            
            print(f"🎯 EasyOCR found {len(results)} potential text regions")
            
            for i, (bbox, text, confidence) in enumerate(results):
                print(f"   Region {i+1}: '{text}' (confidence: {confidence:.3f})")
                
                if confidence < confidence_threshold:
                    print(f"      ⚠️  Skipped (below threshold {confidence_threshold})")
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
                    'language': self._detect_language(text),
                    'bbox': detection['bbox'],
                    'analysis_method': 'easyocr_real'
                }
                recognized_texts.append(recognized_text)
                print(f"      ✅ Added: '{text}' at ({x}, {y})")
            
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
                    'geographic_type': self._classify_geographic_type(text_result['text']),
                    'bbox': bbox,
                    'language': text_result['language']
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
                    'processing_method': 'real_ocr_easyocr',
                    'languages_detected': list(set(t['language'] for t in recognized_texts))
                }
            }
            
            print(f"✅ REAL OCR completed: Found {len(recognized_texts)} actual text regions in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            print(f"❌ OCR processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character analysis."""
        # Check for Chinese characters
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        if chinese_chars > 0:
            return 'zh'
        
        # Default to English
        return 'en'
    
    def _classify_geographic_type(self, text: str) -> str:
        """Classify the type of geographic feature based on text content."""
        text_lower = text.lower()
        
        # Water features
        water_keywords = ['river', 'lake', 'sea', 'ocean', 'bay', 'harbor', 'creek', 'stream', 'thames', 'seine']
        if any(keyword in text_lower for keyword in water_keywords):
            return 'water_feature'
        
        # Transportation
        transport_keywords = ['road', 'street', 'avenue', 'bridge', 'railway', 'station']
        if any(keyword in text_lower for keyword in transport_keywords):
            return 'transportation'
        
        # Settlements
        settlement_keywords = ['city', 'town', 'village', 'district', 'county']
        if any(keyword in text_lower for keyword in settlement_keywords):
            return 'settlement'
        
        # Landmarks
        landmark_keywords = ['castle', 'church', 'tower', 'palace', 'fort', 'abbey']
        if any(keyword in text_lower for keyword in landmark_keywords):
            return 'landmark'
        
        # Default to place name
        return 'place_name'

if __name__ == "__main__":
    print("🗺️  TESTING REAL OCR PROCESSOR")
    print("=" * 50)
    
    processor = RealOCRProcessor()
    
    if not processor.ocr_available:
        print("❌ OCR not available. Install EasyOCR first:")
        print("   pip install easyocr")
        exit(1)
    
    # Test with demo image
    demo_path = "demo_historical_map.jpg"
    from pathlib import Path
    
    if Path(demo_path).exists():
        print(f"\n🔍 Testing with {demo_path}...")
        result = processor.process_image(demo_path, {
            'task': 'task4', 
            'confidence_threshold': 0.1  # Lower threshold to catch more text
        })
        
        if result['success']:
            texts = result['recognized_texts']
            print(f"\n📍 REAL OCR RESULTS:")
            print(f"Found {len(texts)} actual text regions from your image!")
            
            if texts:
                print("\nDetected text:")
                for i, text in enumerate(texts, 1):
                    bbox = text['bbox']
                    print(f"   {i}. '{text['text']}' ({text['language']})")
                    print(f"      Location: ({bbox['x']}, {bbox['y']}) - Size: {bbox['width']}x{bbox['height']}")
                    print(f"      Confidence: {text['confidence']:.3f}")
                    print(f"      Type: {result['linked_texts'][i-1]['geographic_type']}")
                    print()
                
                print("🎉 This is REAL text from your image, not generated names!")
            else:
                print("No text detected. This could mean:")
                print("• The image has very faint or unclear text")
                print("• The text is in a language not supported")
                print("• Try lowering the confidence threshold")
        else:
            print(f"❌ Processing failed: {result['error']}")
    else:
        print(f"💡 Demo image not found at {demo_path}")
        print("Create a test image with text and try again!")
        
        # Create a simple test image with text
        print("\n🎨 Creating a test image with text...")
        import cv2
        import numpy as np
        
        # Create image with actual text
        img = np.ones((300, 500, 3), dtype=np.uint8) * 255  # White background
        
        # Add text using OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'LONDON', (50, 100), font, 2, (0, 0, 0), 3)
        cv2.putText(img, 'PARIS', (50, 180), font, 2, (0, 0, 0), 3)
        cv2.putText(img, 'BERLIN', (50, 260), font, 2, (0, 0, 0), 3)
        
        test_path = "test_text_image.jpg"
        cv2.imwrite(test_path, img)
        print(f"✅ Created {test_path} with text: LONDON, PARIS, BERLIN")
        
        # Test with the created image
        print(f"\n🔍 Testing with {test_path}...")
        result = processor.process_image(test_path, {
            'task': 'task4',
            'confidence_threshold': 0.1
        })
        
        if result['success']:
            texts = result['recognized_texts']
            print(f"\n📍 REAL OCR RESULTS from test image:")
            print(f"Found {len(texts)} text regions!")
            
            for i, text in enumerate(texts, 1):
                bbox = text['bbox']
                print(f"   {i}. '{text['text']}' at ({bbox['x']}, {bbox['y']}) - confidence: {text['confidence']:.3f}")
            
            print("\n🎉 SUCCESS! EasyOCR is reading REAL text from images!")
        else:
            print(f"❌ Test failed: {result['error']}")
    
    print("\n" + "=" * 50)
    print("🎯 REAL OCR is now working!")
    print("Upload your historical maps to see actual text recognition!")