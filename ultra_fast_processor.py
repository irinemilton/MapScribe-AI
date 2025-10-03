#!/usr/bin/env python3
"""
Ultra Fast Image Processor - Optimized for very large historical maps
Handles massive images with aggressive optimization and minimal processing time.
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

class UltraFastProcessor:
    """Ultra-fast processor for large historical maps with aggressive optimizations."""
    
    def __init__(self):
        self.reader = None
        if EASYOCR_AVAILABLE:
            try:
                # Initialize with minimal settings for speed
                self.reader = easyocr.Reader(['en'], gpu=False, verbose=False, 
                                           download_enabled=False)
                print("⚡ Ultra-fast EasyOCR initialized")
            except Exception as e:
                print(f"⚠️ EasyOCR failed, using fallback: {e}")
                self.reader = None
        
        # Ultra-aggressive settings for speed
        self.max_dimension = 1024  # Much smaller for speed
        self.min_confidence = 0.2  # Lower threshold
        self.max_regions = 50      # Limit processing regions
    
    def ultra_resize(self, image_path):
        """Ultra-aggressive resizing for maximum speed."""
        try:
            # Use PIL for faster loading
            with Image.open(image_path) as img:
                width, height = img.size
                max_dim = max(width, height)
                
                if max_dim > self.max_dimension:
                    # Calculate new size
                    scale = self.max_dimension / max_dim
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    
                    # Use fastest resampling
                    resized = img.resize((new_width, new_height), Image.NEAREST)
                    
                    # Save as temporary file
                    temp_path = image_path.replace('.', '_ultra_small.')
                    resized.save(temp_path, 'JPEG', quality=85, optimize=True)
                    
                    print(f"⚡ Ultra-resized: {width}x{height} → {new_width}x{new_height}")
                    return temp_path, scale
                
                return image_path, 1.0
                
        except Exception as e:
            print(f"❌ Ultra-resize failed: {e}")
            return image_path, 1.0
    
    def minimal_enhance(self, image_path):
        """Minimal enhancement for speed."""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return image_path
            
            # Just basic contrast enhancement
            enhanced = cv2.equalizeHist(image)
            
            # Save enhanced version
            enhanced_path = image_path.replace('.', '_enhanced.')
            cv2.imwrite(enhanced_path, enhanced)
            
            return enhanced_path
            
        except Exception as e:
            print(f"⚠️ Enhancement skipped: {e}")
            return image_path
    
    def process_image(self, image_path, options=None):
        """Ultra-fast processing optimized for large images."""
        start_time = time.time()
        
        try:
            print(f"⚡ Ultra-fast processing: {os.path.basename(image_path)}")
            
            # Step 1: Ultra-aggressive resize
            small_path, scale_factor = self.ultra_resize(image_path)
            
            # Step 2: Minimal enhancement
            enhanced_path = self.minimal_enhance(small_path)
            
            # Step 3: Fast OCR with limits
            if self.reader and EASYOCR_AVAILABLE:
                results = self._ultra_fast_ocr(enhanced_path, scale_factor)
            else:
                results = self._ultra_fast_fallback(enhanced_path, scale_factor)
            
            # Cleanup
            self._cleanup_temp_files([small_path, enhanced_path], image_path)
            
            processing_time = time.time() - start_time
            results['processing_time'] = round(processing_time, 2)
            results['ultra_optimized'] = True
            results['scale_factor'] = scale_factor
            
            print(f"⚡ Ultra-fast completed in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            print(f"❌ Ultra-fast processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'detections': [],
                'text_regions': [],
                'summary': {'total_detections': 0}
            }
    
    def _ultra_fast_ocr(self, image_path, scale_factor):
        """Ultra-fast OCR with aggressive limits."""
        try:
            print("⚡ Running ultra-fast OCR...")
            
            # Ultra-fast OCR settings
            ocr_results = self.reader.readtext(
                image_path,
                detail=1,
                paragraph=False,
                width_ths=0.9,    # More aggressive filtering
                height_ths=0.9,   # More aggressive filtering
                decoder='greedy', # Fastest decoder
                batch_size=1,     # Minimal batch
                workers=1         # Single worker for consistency
            )
            
            # Limit results for speed
            ocr_results = ocr_results[:self.max_regions]
            
            detections = []
            text_regions = []
            
            for i, (bbox, text, confidence) in enumerate(ocr_results):
                if confidence >= self.min_confidence and len(text.strip()) > 1:
                    # Scale coordinates back
                    scaled_bbox = [[int(x / scale_factor), int(y / scale_factor)] for x, y in bbox]
                    
                    detection = {
                        'id': i + 1,
                        'bbox': scaled_bbox,
                        'text': text.strip(),
                        'confidence': round(confidence, 3),
                        'type': 'text_region'
                    }
                    
                    detections.append(detection)
                    text_regions.append(detection)
            
            # Quick stats
            confidences = [d['confidence'] for d in detections]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'success': True,
                'method': 'Ultra_Fast_OCR',
                'detections': detections,
                'text_regions': text_regions,
                'summary': {
                    'total_detections': len(detections),
                    'average_confidence': round(avg_confidence, 3),
                    'processing_mode': 'ultra_optimized'
                }
            }
            
        except Exception as e:
            print(f"❌ Ultra-fast OCR failed: {e}")
            return self._ultra_fast_fallback(image_path, scale_factor)
    
    def _ultra_fast_fallback(self, image_path, scale_factor):
        """Ultra-fast fallback when OCR fails."""
        try:
            print("⚡ Ultra-fast fallback mode...")
            
            # Load small image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Cannot load image")
            
            height, width = image.shape
            
            # Very basic text region detection
            detections = []
            
            # Simple threshold
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours (limit to speed up)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[:20]  # Limit for speed
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                # Basic size filtering
                if 5 < w < width/4 and 3 < h < height/8:
                    # Scale back
                    scaled_x = int(x / scale_factor)
                    scaled_y = int(y / scale_factor)
                    scaled_w = int(w / scale_factor)
                    scaled_h = int(h / scale_factor)
                    
                    bbox = [
                        [scaled_x, scaled_y],
                        [scaled_x + scaled_w, scaled_y],
                        [scaled_x + scaled_w, scaled_y + scaled_h],
                        [scaled_x, scaled_y + scaled_h]
                    ]
                    
                    detection = {
                        'id': i + 1,
                        'bbox': bbox,
                        'text': f'Region_{i+1}',
                        'confidence': 0.3,
                        'type': 'detected_region'
                    }
                    
                    detections.append(detection)
            
            return {
                'success': True,
                'method': 'Ultra_Fast_Fallback',
                'detections': detections,
                'text_regions': detections,
                'summary': {
                    'total_detections': len(detections),
                    'average_confidence': 0.3,
                    'note': 'Ultra-fast mode - reduced accuracy for speed'
                }
            }
            
        except Exception as e:
            print(f"❌ Ultra-fast fallback failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'detections': [],
                'text_regions': [],
                'summary': {'total_detections': 0}
            }
    
    def _cleanup_temp_files(self, temp_files, original_file):
        """Clean up temporary files."""
        for temp_file in temp_files:
            if temp_file != original_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass  # Ignore cleanup errors for speed

if __name__ == "__main__":
    # Test the ultra-fast processor
    processor = UltraFastProcessor()
    print("⚡ Ultra-fast processor ready!")