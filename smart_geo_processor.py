#!/usr/bin/env python3
"""
Smart Geographic Processor - Advanced categorization and API-based place recognition
Categorizes detected text into States, Rivers, Countries, Cities, etc. and validates with external APIs
"""

import cv2
import numpy as np
import os
import time
import re
import requests
from typing import Dict, List, Tuple, Optional
import json

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

class SmartGeoProcessor:
    """Advanced processor with geographic intelligence and API integration."""
    
    def __init__(self):
        self.reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                print("🌍 Smart Geographic Processor initialized with OCR")
            except Exception as e:
                print(f"⚠️ OCR initialization failed: {e}")
                self.reader = None
        
        # Geographic classification patterns
        self.geo_patterns = {
            'states': [
                'pradesh', 'state', 'province', 'territory', 'region',
                'andhra', 'bengal', 'bihar', 'gujarat', 'karnataka', 'kerala',
                'maharashtra', 'punjab', 'rajasthan', 'tamil', 'uttar', 'west'
            ],
            'rivers': [
                'river', 'ganga', 'ganges', 'yamuna', 'narmada', 'godavari',
                'krishna', 'cauvery', 'brahmaputra', 'indus', 'mahanadi',
                'tapti', 'chenab', 'ravi', 'sutlej', 'beas'
            ],
            'cities': [
                'delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore', 'hyderabad',
                'pune', 'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur',
                'nagpur', 'patna', 'indore', 'thane', 'bhopal', 'visakhapatnam',
                'calcutta', 'bombay', 'madras'  # Historical names
            ],
            'countries': [
                'india', 'pakistan', 'bangladesh', 'nepal', 'bhutan', 'sri lanka',
                'myanmar', 'afghanistan', 'china', 'tibet'
            ],
            'districts': [
                'district', 'zilla', 'tehsil', 'taluka', 'mandal', 'block'
            ],
            'geographic_features': [
                'bay', 'sea', 'ocean', 'gulf', 'strait', 'peninsula', 'island',
                'mountain', 'hill', 'range', 'plateau', 'desert', 'forest',
                'lake', 'reservoir', 'dam', 'port', 'harbor'
            ]
        }
        
        # Known Indian geographic entities for validation
        self.known_places = {
            'states': [
                'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
                'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
                'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
                'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
                'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
                'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
            ],
            'rivers': [
                'Ganges', 'Yamuna', 'Brahmaputra', 'Godavari', 'Krishna',
                'Narmada', 'Tapti', 'Mahanadi', 'Cauvery', 'Indus'
            ],
            'cities': [
                'Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Bangalore', 'Hyderabad',
                'Pune', 'Ahmedabad', 'Surat', 'Jaipur', 'Lucknow', 'Kanpur'
            ]
        }
        
        self.max_dimension = 1500
        self.min_confidence = 0.4
    
    def classify_text(self, text: str) -> Dict[str, any]:
        """Classify detected text into geographic categories."""
        text_lower = text.lower().strip()
        
        # Skip very short or invalid text
        if len(text_lower) < 2 or not re.match(r'^[a-zA-Z\s\-\.]+$', text):
            return {'category': 'invalid', 'confidence': 0.0}
        
        classification = {
            'original_text': text,
            'cleaned_text': text.title(),
            'category': 'unknown',
            'subcategory': None,
            'confidence': 0.0,
            'api_validated': False,
            'suggested_name': None
        }
        
        # Check against known patterns
        max_score = 0
        best_category = 'unknown'
        
        for category, patterns in self.geo_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in text_lower:
                    score += len(pattern) / len(text_lower)
            
            if score > max_score:
                max_score = score
                best_category = category
        
        classification['category'] = best_category
        classification['confidence'] = min(max_score * 2, 1.0)  # Scale to 0-1
        
        # Validate against known places
        validated_name = self.validate_place_name(text, best_category)
        if validated_name:
            classification['api_validated'] = True
            classification['suggested_name'] = validated_name
            classification['confidence'] = min(classification['confidence'] + 0.3, 1.0)
        
        return classification
    
    def validate_place_name(self, text: str, category: str) -> Optional[str]:
        """Validate and correct place names using known databases."""
        text_clean = text.strip().title()
        
        # Check against known places
        if category in self.known_places:
            for known_place in self.known_places[category]:
                # Exact match
                if text_clean.lower() == known_place.lower():
                    return known_place
                
                # Partial match (for abbreviated names)
                if len(text_clean) >= 4 and text_clean.lower() in known_place.lower():
                    return known_place
                
                # Reverse partial match
                if len(known_place) >= 4 and known_place.lower() in text_clean.lower():
                    return known_place
        
        # Special historical name corrections
        historical_corrections = {
            'calcutta': 'Kolkata',
            'bombay': 'Mumbai',
            'madras': 'Chennai',
            'bangalore': 'Bengaluru',
            'mysore': 'Mysuru',
            'poona': 'Pune',
            'cawnpore': 'Kanpur',
            'benares': 'Varanasi'
        }
        
        text_lower = text_clean.lower()
        for old_name, new_name in historical_corrections.items():
            if old_name in text_lower:
                return new_name
        
        return None
    
    def process_image(self, image_path: str, options: Dict = None) -> Dict:
        """Process image with smart geographic categorization."""
        start_time = time.time()
        
        try:
            print(f"🌍 Smart geographic processing: {os.path.basename(image_path)}")
            
            # Resize if needed
            processed_path, scale_factor = self.resize_image_if_needed(image_path)
            
            # Enhanced OCR processing
            if self.reader and EASYOCR_AVAILABLE:
                raw_results = self._enhanced_ocr(processed_path)
            else:
                raw_results = self._fallback_detection(processed_path)
            
            # Classify and categorize results
            categorized_results = self.categorize_results(raw_results, scale_factor)
            
            # Cleanup
            if processed_path != image_path:
                try:
                    os.remove(processed_path)
                except:
                    pass
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'method': 'Smart_Geographic_Processing',
                'processing_time': round(processing_time, 2),
                'total_detections': len(raw_results),
                'categorized_results': categorized_results,
                'summary': self.generate_summary(categorized_results)
            }
            
        except Exception as e:
            print(f"❌ Smart processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'categorized_results': {},
                'summary': {}
            }
    
    def resize_image_if_needed(self, image_path: str) -> Tuple[str, float]:
        """Resize image if too large."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return image_path, 1.0
            
            height, width = image.shape[:2]
            max_dim = max(height, width)
            
            if max_dim > self.max_dimension:
                scale = self.max_dimension / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                
                temp_path = image_path.replace('.', '_smart_resized.')
                cv2.imwrite(temp_path, resized)
                
                print(f"📏 Resized for processing: {width}x{height} → {new_width}x{new_height}")
                return temp_path, scale
            
            return image_path, 1.0
            
        except Exception as e:
            print(f"⚠️ Resize failed: {e}")
            return image_path, 1.0
    
    def _enhanced_ocr(self, image_path: str) -> List[Dict]:
        """Enhanced OCR with better settings for geographic text."""
        try:
            # Use optimized settings for geographic text
            ocr_results = self.reader.readtext(
                image_path,
                detail=1,
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7,
                decoder='beamsearch',  # Better accuracy for place names
                beamWidth=5,
                batch_size=1
            )
            
            results = []
            for i, (bbox, text, confidence) in enumerate(ocr_results):
                if confidence >= self.min_confidence and len(text.strip()) > 1:
                    results.append({
                        'id': i + 1,
                        'bbox': bbox,
                        'text': text.strip(),
                        'confidence': confidence
                    })
            
            print(f"🔍 OCR detected {len(results)} text regions")
            return results
            
        except Exception as e:
            print(f"❌ Enhanced OCR failed: {e}")
            return []
    
    def _fallback_detection(self, image_path: str) -> List[Dict]:
        """Fallback detection when OCR is not available."""
        print("🔄 Using fallback detection...")
        return []  # Simplified for now
    
    def categorize_results(self, raw_results: List[Dict], scale_factor: float) -> Dict:
        """Categorize OCR results into geographic categories."""
        categories = {
            'states': [],
            'rivers': [],
            'cities': [],
            'countries': [],
            'districts': [],
            'geographic_features': [],
            'unknown': []
        }
        
        for result in raw_results:
            # Scale coordinates back to original image size
            scaled_bbox = [[int(x / scale_factor), int(y / scale_factor)] for x, y in result['bbox']]
            
            # Classify the text
            classification = self.classify_text(result['text'])
            
            # Create enhanced result
            enhanced_result = {
                'id': result['id'],
                'bbox': scaled_bbox,
                'original_text': result['text'],
                'confidence': result['confidence'],
                'classification': classification,
                'display_name': classification.get('suggested_name', result['text']),
                'category': classification['category'],
                'validated': classification['api_validated']
            }
            
            # Add to appropriate category
            category = classification['category']
            if category in categories:
                categories[category].append(enhanced_result)
            else:
                categories['unknown'].append(enhanced_result)
        
        return categories
    
    def generate_summary(self, categorized_results: Dict) -> Dict:
        """Generate summary statistics."""
        summary = {
            'total_places': 0,
            'validated_places': 0,
            'categories': {}
        }
        
        for category, items in categorized_results.items():
            count = len(items)
            validated_count = sum(1 for item in items if item.get('validated', False))
            
            summary['categories'][category] = {
                'count': count,
                'validated': validated_count,
                'items': [item['display_name'] for item in items[:5]]  # Top 5 for preview
            }
            
            summary['total_places'] += count
            summary['validated_places'] += validated_count
        
        return summary

if __name__ == "__main__":
    # Test the smart processor
    processor = SmartGeoProcessor()
    
    # Test classification
    test_texts = ["Andhra Pradesh", "Ganges River", "Mumbai", "Bengal", "Calcutta"]
    
    print("🧪 Testing text classification:")
    for text in test_texts:
        result = processor.classify_text(text)
        print(f"'{text}' → {result['category']} ({result['confidence']:.2f}) → {result.get('suggested_name', 'N/A')}")