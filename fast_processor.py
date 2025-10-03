#!/usr/bin/env python3
"""
Fast Image Processor - Optimized for web interface
Handles large images efficiently with automatic resizing and quick processing.
"""

import cv2
import numpy as np
import os
from PIL import Image
import time

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

class FastImageProcessor:
    """Fast processor optimized for web interface with large image handling."""
    
    def __init__(self):
        self.reader = None
        if EASYOCR_AVAILABLE:
            try:
                # Initialize with minimal GPU usage
                self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                print("✅ EasyOCR initialized successfully")
            except Exception as e:
                print(f"⚠️ EasyOCR initialization failed: {e}")
                self.reader = None
        
        self.max_dimension = 2048  # Maximum image dimension for processing
        self.min_confidence = 0.3  # Lower threshold for better detection
    
    def resize_image_if_needed(self, image_path):
        """Resize image if it's too large for efficient processing."""
        try:
            # Load image to check dimensions
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            height, width = image.shape[:2]
            max_dim = max(height, width)
            
            # If image is too large, resize it
            if max_dim > self.max_dimension:
                scale = self.max_dimension / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Resize with high quality
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                
                # Save resized version
                temp_path = image_path.replace('.', '_resized.')
                cv2.imwrite(temp_path, resized)
                
                print(f"📏 Resized image from {width}x{height} to {new_width}x{new_height}")
                return temp_path, scale
            
            return image_path, 1.0
            
        except Exception as e:
            print(f"❌ Error resizing image: {e}")
            return image_path, 1.0
    
    def enhance_image_for_ocr(self, image_path):
        """Quick image enhancement for better OCR results."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return image_path
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Slight denoising
            denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
            
            # Save enhanced version
            enhanced_path = image_path.replace('.', '_enhanced.')
            cv2.imwrite(enhanced_path, denoised)
            
            return enhanced_path
            
        except Exception as e:
            print(f"⚠️ Enhancement failed, using original: {e}")
            return image_path
    
    def process_image(self, image_path, options=None):
        """Process image with optimizations for web interface."""
        start_time = time.time()
        
        if options is None:
            options = {}
        
        try:
            print(f"🔄 Starting fast processing of: {os.path.basename(image_path)}")
            
            # Step 1: Resize if needed
            processed_path, scale_factor = self.resize_image_if_needed(image_path)
            
            # Step 2: Quick enhancement
            enhanced_path = self.enhance_image_for_ocr(processed_path)
            
            # Step 3: OCR Processing
            if self.reader and EASYOCR_AVAILABLE:
                results = self._process_with_easyocr(enhanced_path, scale_factor)
            else:
                results = self._process_with_fallback(enhanced_path, scale_factor)
            
            # Cleanup temporary files
            self._cleanup_temp_files([processed_path, enhanced_path], image_path)
            
            processing_time = time.time() - start_time
            results['processing_time'] = round(processing_time, 2)
            results['image_resized'] = scale_factor < 1.0
            results['scale_factor'] = scale_factor
            
            print(f"✅ Processing completed in {processing_time:.2f} seconds")
            return results
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'detections': [],
                'text_regions': [],
                'summary': {
                    'total_detections': 0,
                    'high_confidence_detections': 0,
                    'average_confidence': 0.0
                }
            }
    
    def _process_with_easyocr(self, image_path, scale_factor):
        """Process using EasyOCR with optimizations."""
        try:
            print("🔍 Running EasyOCR detection...")
            
            # Run OCR with optimized settings
            ocr_results = self.reader.readtext(
                image_path,
                detail=1,
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7,
                decoder='greedy'  # Faster than beam search
            )
            
            detections = []
            text_regions = []
            
            for i, (bbox, text, confidence) in enumerate(ocr_results):
                if confidence >= self.min_confidence and len(text.strip()) > 0:
                    # Scale coordinates back to original image size
                    scaled_bbox = [[int(x * (1/scale_factor)), int(y * (1/scale_factor))] for x, y in bbox]
                    
                    detection = {
                        'id': i + 1,
                        'bbox': scaled_bbox,
                        'text': text.strip(),
                        'confidence': round(confidence, 3),
                        'type': 'word' if len(text.split()) == 1 else 'phrase'
                    }
                    
                    detections.append(detection)
                    text_regions.append({
                        'text': text.strip(),
                        'confidence': round(confidence, 3),
                        'bbox': scaled_bbox
                    })
            
            # Calculate summary statistics
            confidences = [d['confidence'] for d in detections]
            high_conf_count = len([c for c in confidences if c >= 0.7])
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'success': True,
                'method': 'EasyOCR',
                'detections': detections,
                'text_regions': text_regions,
                'summary': {
                    'total_detections': len(detections),
                    'high_confidence_detections': high_conf_count,
                    'average_confidence': round(avg_confidence, 3),
                    'unique_texts': len(set(d['text'].lower() for d in detections))
                }
            }
            
        except Exception as e:
            print(f"❌ EasyOCR processing failed: {e}")
            return self._process_with_fallback(image_path, scale_factor)
    
    def _process_with_fallback(self, image_path, scale_factor):
        """Fallback processing when OCR is not available."""
        try:
            print("🔄 Using fallback processing...")
            
            # Load image for basic analysis
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            
            # Generate some basic detections based on image analysis
            detections = []
            
            # Simple text region detection using contours
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours that might be text
            for i, contour in enumerate(contours[:20]):  # Limit to first 20
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size (potential text regions)
                if 10 < w < width/3 and 5 < h < height/10:
                    # Scale back to original coordinates
                    scaled_x = int(x * (1/scale_factor))
                    scaled_y = int(y * (1/scale_factor))
                    scaled_w = int(w * (1/scale_factor))
                    scaled_h = int(h * (1/scale_factor))
                    
                    bbox = [
                        [scaled_x, scaled_y],
                        [scaled_x + scaled_w, scaled_y],
                        [scaled_x + scaled_w, scaled_y + scaled_h],
                        [scaled_x, scaled_y + scaled_h]
                    ]
                    
                    detection = {
                        'id': i + 1,
                        'bbox': bbox,
                        'text': f'Text_Region_{i+1}',
                        'confidence': 0.5,
                        'type': 'detected_region'
                    }
                    
                    detections.append(detection)
            
            return {
                'success': True,
                'method': 'Fallback_Detection',
                'detections': detections,
                'text_regions': detections,
                'summary': {
                    'total_detections': len(detections),
                    'high_confidence_detections': 0,
                    'average_confidence': 0.5,
                    'note': 'Fallback mode - install EasyOCR for text recognition'
                }
            }
            
        except Exception as e:
            print(f"❌ Fallback processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'detections': [],
                'text_regions': [],
                'summary': {'total_detections': 0}
            }
    
    def _cleanup_temp_files(self, temp_files, original_file):
        """Clean up temporary files created during processing."""
        for temp_file in temp_files:
            if temp_file != original_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"⚠️ Could not remove temp file {temp_file}: {e}")

if __name__ == "__main__":
    # Test the processor
    processor = FastImageProcessor()
    
    # Test with a sample image if available
    test_files = ['demo_historical_map.jpg', 'test_image.jpg', 'sample.png']
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n🧪 Testing with {test_file}")
            result = processor.process_image(test_file)
            print(f"Result: {result['summary']}")
            break
    else:
        print("No test images found. Processor is ready for use.")