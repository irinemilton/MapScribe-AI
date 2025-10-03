"""
Test utilities for mock objects, image generation, and test data creation.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from unittest.mock import Mock, MagicMock
import tempfile
from pathlib import Path
import json

from src.models.data_structures import (
    BoundingBox, Detection, RecognizedText, LinkedText, ProcessingResult,
    ProcessedImage, ModelMetadata, DetectionConfig, RecognitionConfig
)


class MockImageGenerator:
    """Utility class for generating synthetic test images and data."""
    
    @staticmethod
    def create_test_image(width: int = 640, height: int = 480, channels: int = 3) -> np.ndarray:
        """Create a synthetic test image with random noise."""
        if channels == 1:
            return np.random.randint(0, 255, (height, width), dtype=np.uint8)
        else:
            return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
    
    @staticmethod
    def create_historical_map_image(width: int = 1024, height: int = 768) -> np.ndarray:
        """Create a synthetic historical map image with text-like regions."""
        # Create base map with aged paper color
        image = np.full((height, width, 3), [240, 230, 210], dtype=np.uint8)
        
        # Add some noise for aging effect
        noise = np.random.normal(0, 10, (height, width, 3))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Add some rectangular regions that simulate text
        for _ in range(10):
            x = np.random.randint(0, width - 100)
            y = np.random.randint(0, height - 30)
            w = np.random.randint(50, 150)
            h = np.random.randint(15, 25)
            # Dark text color
            image[y:y+h, x:x+w] = [50, 40, 30]
        
        return image
    
    @staticmethod
    def create_degraded_image(base_image: np.ndarray, degradation_level: float = 0.3) -> np.ndarray:
        """Apply degradation effects to simulate historical document aging."""
        degraded = base_image.copy()
        
        # Add noise
        noise = np.random.normal(0, degradation_level * 50, base_image.shape)
        degraded = np.clip(degraded + noise, 0, 255).astype(np.uint8)
        
        # Reduce contrast
        degraded = np.clip(degraded * (1 - degradation_level * 0.5), 0, 255).astype(np.uint8)
        
        return degraded
    
    @staticmethod
    def create_text_region_image(text: str = "Sample Text", width: int = 200, height: int = 50) -> np.ndarray:
        """Create a simple image with text-like rectangular region."""
        image = np.full((height, width, 3), [255, 255, 255], dtype=np.uint8)
        
        # Create a dark rectangular region to simulate text
        text_width = min(len(text) * 8, width - 20)
        text_height = min(20, height - 10)
        x_start = (width - text_width) // 2
        y_start = (height - text_height) // 2
        
        image[y_start:y_start+text_height, x_start:x_start+text_width] = [0, 0, 0]
        
        return image


class MockModelObjects:
    """Factory for creating mock model objects and interfaces."""
    
    @staticmethod
    def create_mock_detector() -> Mock:
        """Create a mock text detector."""
        mock_detector = Mock()
        mock_detector.detect_words.return_value = [
            Detection(
                bbox=BoundingBox(x=10, y=20, width=100, height=30, confidence=0.9),
                detection_type='word',
                confidence=0.9,
                image_coordinates=(10, 20, 110, 50)
            )
        ]
        mock_detector.detect_phrases.return_value = [
            Detection(
                bbox=BoundingBox(x=10, y=20, width=200, height=30, confidence=0.85),
                detection_type='phrase',
                confidence=0.85,
                image_coordinates=(10, 20, 210, 50)
            )
        ]
        mock_detector.get_confidence_scores.return_value = [0.9, 0.85]
        return mock_detector
    
    @staticmethod
    def create_mock_recognizer() -> Mock:
        """Create a mock text recognizer."""
        mock_recognizer = Mock()
        mock_recognizer.recognize_words.return_value = [
            RecognizedText(
                detection=Detection(
                    bbox=BoundingBox(x=10, y=20, width=100, height=30, confidence=0.9),
                    detection_type='word',
                    confidence=0.9,
                    image_coordinates=(10, 20, 110, 50)
                ),
                text="Sample",
                language="en",
                recognition_confidence=0.95,
                character_confidences=[0.9, 0.95, 0.92, 0.98, 0.94, 0.96]
            )
        ]
        mock_recognizer.recognize_phrases.return_value = [
            RecognizedText(
                detection=Detection(
                    bbox=BoundingBox(x=10, y=20, width=200, height=30, confidence=0.85),
                    detection_type='phrase',
                    confidence=0.85,
                    image_coordinates=(10, 20, 210, 50)
                ),
                text="Sample Text",
                language="en",
                recognition_confidence=0.92,
                character_confidences=[0.9, 0.95, 0.92, 0.98, 0.94, 0.96, 0.88, 0.93, 0.91, 0.97, 0.89]
            )
        ]
        mock_recognizer.handle_multilingual.return_value = "Mixed Text 混合文本"
        return mock_recognizer
    
    @staticmethod
    def create_mock_linker() -> Mock:
        """Create a mock text linker."""
        mock_linker = Mock()
        mock_linker.link_to_coordinates.return_value = [
            LinkedText(
                recognized_text=RecognizedText(
                    detection=Detection(
                        bbox=BoundingBox(x=10, y=20, width=100, height=30, confidence=0.9),
                        detection_type='word',
                        confidence=0.9,
                        image_coordinates=(10, 20, 110, 50)
                    ),
                    text="Sample",
                    language="en",
                    recognition_confidence=0.95,
                    character_confidences=[0.9, 0.95, 0.92, 0.98, 0.94, 0.96]
                ),
                map_coordinates=(60.0, 35.0),
                spatial_confidence=0.88,
                related_features=["landmark", "place_name"]
            )
        ]
        mock_linker.group_related_text.return_value = [[]]
        mock_linker.calculate_spatial_confidence.return_value = 0.88
        return mock_linker
    
    @staticmethod
    def create_mock_image_processor() -> Mock:
        """Create a mock image processor."""
        mock_processor = Mock()
        mock_processor.preprocess.return_value = MockImageGenerator.create_test_image(640, 480, 3)
        mock_processor.enhance_text_visibility.return_value = MockImageGenerator.create_test_image(640, 480, 3)
        mock_processor.normalize_resolution.return_value = MockImageGenerator.create_test_image(640, 480, 3)
        mock_processor.correct_orientation.return_value = MockImageGenerator.create_test_image(640, 480, 3)
        return mock_processor


class TestDataFactory:
    """Factory for creating test data sets and fixtures."""
    
    @staticmethod
    def create_sample_detections(count: int = 5) -> List[Detection]:
        """Create a list of sample detections."""
        detections = []
        for i in range(count):
            bbox = BoundingBox(
                x=i * 50 + 10,
                y=20,
                width=80 + i * 10,
                height=25,
                confidence=0.8 + i * 0.02
            )
            detection = Detection(
                bbox=bbox,
                detection_type='word' if i % 2 == 0 else 'phrase',
                confidence=0.85 + i * 0.02,
                image_coordinates=(bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)
            )
            detections.append(detection)
        return detections
    
    @staticmethod
    def create_sample_recognized_texts(detections: List[Detection]) -> List[RecognizedText]:
        """Create recognized texts from detections."""
        sample_texts = ["Paris", "London", "Berlin", "Madrid", "Rome"]
        recognized_texts = []
        
        for i, detection in enumerate(detections):
            text = sample_texts[i % len(sample_texts)]
            recognized_text = RecognizedText(
                detection=detection,
                text=text,
                language="en",
                recognition_confidence=0.90 + i * 0.01,
                character_confidences=[0.9 + j * 0.01 for j in range(len(text))]
            )
            recognized_texts.append(recognized_text)
        
        return recognized_texts
    
    @staticmethod
    def create_sample_linked_texts(recognized_texts: List[RecognizedText]) -> List[LinkedText]:
        """Create linked texts from recognized texts."""
        linked_texts = []
        
        for i, recognized_text in enumerate(recognized_texts):
            # Generate map coordinates based on detection bbox center
            bbox = recognized_text.detection.bbox
            map_x = bbox.x + bbox.width / 2
            map_y = bbox.y + bbox.height / 2
            
            linked_text = LinkedText(
                recognized_text=recognized_text,
                map_coordinates=(map_x, map_y),
                spatial_confidence=0.85 + i * 0.02,
                related_features=[f"feature_{i}", "landmark"]
            )
            linked_texts.append(linked_text)
        
        return linked_texts
    
    @staticmethod
    def create_multilingual_test_data() -> Dict[str, List[Dict[str, Any]]]:
        """Create multilingual test data for English and Chinese."""
        return {
            'english': [
                {
                    'text': 'London',
                    'language': 'en',
                    'confidence': 0.95,
                    'character_confidences': [0.9, 0.95, 0.92, 0.98, 0.94, 0.96]
                },
                {
                    'text': 'Paris',
                    'language': 'en',
                    'confidence': 0.92,
                    'character_confidences': [0.9, 0.94, 0.91, 0.96, 0.93]
                },
                {
                    'text': 'Berlin',
                    'language': 'en',
                    'confidence': 0.88,
                    'character_confidences': [0.85, 0.90, 0.87, 0.92, 0.89, 0.91]
                }
            ],
            'chinese': [
                {
                    'text': '北京',
                    'language': 'zh',
                    'confidence': 0.90,
                    'character_confidences': [0.88, 0.92]
                },
                {
                    'text': '上海',
                    'language': 'zh',
                    'confidence': 0.87,
                    'character_confidences': [0.85, 0.89]
                },
                {
                    'text': '广州',
                    'language': 'zh',
                    'confidence': 0.85,
                    'character_confidences': [0.83, 0.87]
                }
            ]
        }


class ValidationTestUtilities:
    """Utilities for testing validation functions."""
    
    @staticmethod
    def test_confidence_validation(validation_func, valid_range: Tuple[float, float] = (0.0, 1.0)):
        """Test confidence score validation function."""
        # Test valid values
        assert validation_func(valid_range[0]) is True
        assert validation_func(valid_range[1]) is True
        assert validation_func((valid_range[0] + valid_range[1]) / 2) is True
        
        # Test invalid values
        assert validation_func(valid_range[0] - 0.1) is False
        assert validation_func(valid_range[1] + 0.1) is False


class PerformanceTestUtilities:
    """Utilities for performance testing."""
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure execution time of a function."""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time