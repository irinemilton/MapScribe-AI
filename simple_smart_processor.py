#!/usr/bin/env python3
"""
Simple Smart Processor - Compatible with Flask app
Provides basic geographic categorization with the expected interface
"""

import cv2
import numpy as np
import os
import time
import re
from typing import Dict, List

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

class SimpleSmartProcessor:
    """Simple smart processor compatible with Flask app interface."""
    
    def __init__(self):
        self.reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                print("🌍 Simple Smart Processor initialized")
            except Exception as e:
                print(f"⚠️ OCR initialization failed: {e}")
                self.reader = None
        
        # Simple geographic patterns
        self.categories = {
            'states': ['pradesh', 'bengal', 'bihar', 'gujarat', 'karnataka', 'kerala', 'maharashtra', 'punjab', 'rajasthan', 'tamil', 'uttar', 'andhra'],
            'rivers': ['river', 'ganga', 'ganges', 'yamuna', 'narmada', 'godavari', 'krishna', 'cauvery', 'brahmaputra'],
            'cities': ['delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore', 'hyderabad', 'pune', 'calcutta', 'bombay', 'madras'],
            'countries': ['india', 'pakistan', 'bangladesh', 'nepal', 'bhutan'],
            'districts': ['district', 'zilla', 'tehsil']
        }
        
        # Name corrections
        self.corrections = {
            'calcutta': 'Kolkata',
            'bombay': 'Mumbai', 
            'madras': 'Chennai',
            'bangalore': 'Bengaluru'
        }
    
    def categorize_text(self, text: str) -> Dict:
        """Categorize text into geographic types."""
        text_lower = text.lower().strip()
        
        if len(text_lower) < 2:
            return {'category': 'unknown', 'display_name': text, 'validated': False}
        
        # Check corrections first
        corrected_name = self.corrections.get(text_lower, text.title())
        
        # Find category
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return {
                        'category': category,
                        'display_name': corrected_name,
                        'validated': corrected_name != text.title()
                    }
        
        return {'category': 'unknown', 'display_name': text.title(), 'validated': False}
    
    def process_image(self, image_path, options=None):
        """Process image with simple smart categorization."""
        start_time = time.time()
        
        try:
            print(f"🌍 Processing with simple smart processor: {os.path.basename(image_path)}")
            
            # Basic OCR
            if self.reader and EASYOCR_AVAILABLE:
                ocr_results = self.reader.readtext(image_path, detail=1)
            else:
                return self._fallback_result()
            
            # Categorize results
            categorized = {
                'states': [],
                'rivers': [],
                'cities': [],
                'countries': [],
                'districts': [],
                'unknown': []
            }
            
            detections = []
            text_regions = []
            
            for i, (bbox, text, confidence) in enumerate(ocr_results):
                if confidence >= 0.3 and len(text.strip()) > 1:
                    # Categorize
                    category_info = self.categorize_text(text)
                    
                    detection = {
                        'id': i + 1,
                        'bbox': bbox,
                        'text': text.strip(),
                        'confidence': confidence,
                        'original_text': text.strip(),
                        'display_name': category_info['display_name'],
                        'category': category_info['category'],
                        'validated': category_info['validated']
                    }
                    
                    detections.append(detection)
                    text_regions.append(detection)
                    
                    # Add to category
                    category = category_info['category']
                    if category in categorized:
                        categorized[category].append(detection)
                    else:
                        categorized['unknown'].append(detection)
            
            # Calculate summary
            total_places = len(detections)
            validated_places = sum(1 for d in detections if d['validated'])
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'method': 'Simple_Smart_Processing',
                'processing_time': round(processing_time, 2),
                'total_detections': total_places,
                'detections': detections,
                'text_regions': text_regions,
                'categorized_results': categorized,
                'summary': {
                    'total_places': total_places,
                    'validated_places': validated_places,
                    'categories': {cat: {'count': len(items)} for cat, items in categorized.items()}
                }
            }
            
        except Exception as e:
            print(f"❌ Simple smart processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'detections': [],
                'text_regions': [],
                'categorized_results': {},
                'summary': {'total_places': 0, 'validated_places': 0}
            }
    
    def _fallback_result(self):
        """Fallback when OCR is not available."""
        return {
            'success': True,
            'method': 'Fallback_Mode',
            'processing_time': 0.1,
            'total_detections': 0,
            'detections': [],
            'text_regions': [],
            'categorized_results': {
                'states': [], 'rivers': [], 'cities': [], 
                'countries': [], 'districts': [], 'unknown': []
            },
            'summary': {
                'total_places': 0,
                'validated_places': 0,
                'note': 'OCR not available - install EasyOCR for text recognition'
            }
        }

if __name__ == "__main__":
    # Test the processor
    processor = SimpleSmartProcessor()
    
    # Test categorization
    test_texts = ["Andhra Pradesh", "Ganges", "Mumbai", "Calcutta"]
    for text in test_texts:
        result = processor.categorize_text(text)
        print(f"'{text}' → {result['category']} → {result['display_name']}")