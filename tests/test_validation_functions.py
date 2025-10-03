"""
Unit tests for validation functions and error handling.
"""

import pytest
import numpy as np
from typing import List, Tuple, Dict, Any
from unittest.mock import Mock, patch

from src.models.data_structures import (
    BoundingBox, Detection, RecognizedText, LinkedText, ProcessingResult,
    ProcessedImage, ModelMetadata, DetectionConfig, RecognitionConfig
)
from tests.test_utilities import ValidationTestUtilities, TestDataFactory


class TestBoundingBoxValidation:
    """Test validation functions for BoundingBox."""
    
    def test_coordinate_validation(self):
        """Test coordinate validation in BoundingBox."""
        # Valid coordinates
        valid_bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.8)
        assert valid_bbox.x == 10
        assert valid_bbox.y == 20
        
        # Test negative coordinates
        with pytest.raises(ValueError, match="Coordinates must be non-negative"):
            BoundingBox(x=-1, y=20, width=100, height=50, confidence=0.8)
        
        with pytest.raises(ValueError, match="Coordinates must be non-negative"):
            BoundingBox(x=10, y=-1, width=100, height=50, confidence=0.8)
    
    def test_dimension_validation(self):
        """Test dimension validation in BoundingBox."""
        # Valid dimensions
        valid_bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.8)
        assert valid_bbox.width == 100
        assert valid_bbox.height == 50
        
        # Test zero/negative dimensions
        with pytest.raises(ValueError, match="Width and height must be positive"):
            BoundingBox(x=10, y=20, width=0, height=50, confidence=0.8)
        
        with pytest.raises(ValueError, match="Width and height must be positive"):
            BoundingBox(x=10, y=20, width=100, height=-1, confidence=0.8)
    
    def test_confidence_validation(self):
        """Test confidence validation in BoundingBox."""
        # Valid confidence
        valid_bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.8)
        assert valid_bbox.confidence == 0.8
        
        # Test invalid confidence range
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            BoundingBox(x=10, y=20, width=100, height=50, confidence=-0.1)
        
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            BoundingBox(x=10, y=20, width=100, height=50, confidence=1.1)
        
        # Test boundary values
        boundary_bbox_low = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.0)
        assert boundary_bbox_low.confidence == 0.0
        
        boundary_bbox_high = BoundingBox(x=10, y=20, width=100, height=50, confidence=1.0)
        assert boundary_bbox_high.confidence == 1.0
    
    def test_area_calculation_validation(self):
        """Test area calculation and validation."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.8)
        assert bbox.area == 5000
        
        # Test with different dimensions
        small_bbox = BoundingBox(x=0, y=0, width=1, height=1, confidence=0.5)
        assert small_bbox.area == 1
        
        large_bbox = BoundingBox(x=0, y=0, width=1000, height=800, confidence=0.9)
        assert large_bbox.area == 800000
    
    def test_coordinates_property_validation(self):
        """Test coordinates property calculation."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.8)
        coords = bbox.coordinates
        assert coords == (10, 20, 110, 70)
        
        # Test edge case with zero position
        zero_bbox = BoundingBox(x=0, y=0, width=50, height=30, confidence=0.7)
        assert zero_bbox.coordinates == (0, 0, 50, 30)


class TestDetectionValidation:
    """Test validation functions for Detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.valid_bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.8)
    
    def test_detection_type_validation(self):
        """Test detection type validation."""
        # Valid detection types
        word_detection = Detection(
            bbox=self.valid_bbox,
            detection_type='word',
            confidence=0.9,
            image_coordinates=(10, 20, 110, 70)
        )
        assert word_detection.detection_type == 'word'
        
        phrase_detection = Detection(
            bbox=self.valid_bbox,
            detection_type='phrase',
            confidence=0.9,
            image_coordinates=(10, 20, 110, 70)
        )
        assert phrase_detection.detection_type == 'phrase'
        
        # Invalid detection type
        with pytest.raises(ValueError, match="Detection type must be 'word' or 'phrase'"):
            Detection(
                bbox=self.valid_bbox,
                detection_type='invalid',
                confidence=0.9,
                image_coordinates=(10, 20, 110, 70)
            )
    
    def test_image_coordinates_validation(self):
        """Test image coordinates validation."""
        # Valid coordinates
        valid_detection = Detection(
            bbox=self.valid_bbox,
            detection_type='word',
            confidence=0.9,
            image_coordinates=(10, 20, 110, 70)
        )
        assert valid_detection.image_coordinates == (10, 20, 110, 70)
        
        # Test invalid coordinates - negative values
        with pytest.raises(ValueError, match="Invalid image coordinates"):
            Detection(
                bbox=self.valid_bbox,
                detection_type='word',
                confidence=0.9,
                image_coordinates=(-1, 20, 110, 70)
            )
        
        # Test invalid coordinates - x2 <= x1
        with pytest.raises(ValueError, match="Invalid image coordinates"):
            Detection(
                bbox=self.valid_bbox,
                detection_type='word',
                confidence=0.9,
                image_coordinates=(110, 20, 110, 70)
            )
        
        # Test invalid coordinates - y2 <= y1
        with pytest.raises(ValueError, match="Invalid image coordinates"):
            Detection(
                bbox=self.valid_bbox,
                detection_type='word',
                confidence=0.9,
                image_coordinates=(10, 70, 110, 70)
            )
    
    def test_detection_confidence_validation(self):
        """Test detection confidence validation."""
        # Valid confidence
        valid_detection = Detection(
            bbox=self.valid_bbox,
            detection_type='word',
            confidence=0.85,
            image_coordinates=(10, 20, 110, 70)
        )
        assert valid_detection.confidence == 0.85
        
        # Invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Detection(
                bbox=self.valid_bbox,
                detection_type='word',
                confidence=1.5,
                image_coordinates=(10, 20, 110, 70)
            )


class TestRecognizedTextValidation:
    """Test validation functions for RecognizedText."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.valid_bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.8)
        self.valid_detection = Detection(
            bbox=self.valid_bbox,
            detection_type='word',
            confidence=0.9,
            image_coordinates=(10, 20, 110, 70)
        )
    
    def test_language_validation(self):
        """Test language code validation."""
        # Valid languages
        valid_languages = ['en', 'zh', 'fr', 'de', 'es', 'it', 'ja', 'ko', 'ar', 'ru']
        
        for lang in valid_languages:
            recognized_text = RecognizedText(
                detection=self.valid_detection,
                text="Sample",
                language=lang,
                recognition_confidence=0.95,
                character_confidences=[0.9, 0.95, 0.92, 0.98, 0.94, 0.96]
            )
            assert recognized_text.language == lang
        
        # Invalid language
        with pytest.raises(ValueError, match="Language must be one of"):
            RecognizedText(
                detection=self.valid_detection,
                text="Sample",
                language="invalid_lang",
                recognition_confidence=0.95,
                character_confidences=[0.9, 0.95, 0.92, 0.98, 0.94, 0.96]
            )
    
    def test_character_confidences_validation(self):
        """Test character confidences validation."""
        # Valid character confidences
        text = "Hello"
        valid_confidences = [0.9, 0.95, 0.92, 0.98, 0.94]
        
        recognized_text = RecognizedText(
            detection=self.valid_detection,
            text=text,
            language="en",
            recognition_confidence=0.95,
            character_confidences=valid_confidences
        )
        assert len(recognized_text.character_confidences) == len(text)
        
        # Mismatched length
        with pytest.raises(ValueError, match="Character confidences must match text length"):
            RecognizedText(
                detection=self.valid_detection,
                text="Hello",
                language="en",
                recognition_confidence=0.95,
                character_confidences=[0.9, 0.95, 0.92]  # Too few
            )
        
        # Invalid confidence values
        with pytest.raises(ValueError, match="Character confidence must be between 0.0 and 1.0"):
            RecognizedText(
                detection=self.valid_detection,
                text="Hello",
                language="en",
                recognition_confidence=0.95,
                character_confidences=[0.9, 1.5, 0.92, 0.98, 0.94]  # Invalid value
            )
    
    def test_recognition_confidence_validation(self):
        """Test recognition confidence validation."""
        # Valid confidence
        recognized_text = RecognizedText(
            detection=self.valid_detection,
            text="Hello",
            language="en",
            recognition_confidence=0.95,
            character_confidences=[0.9, 0.95, 0.92, 0.98, 0.94]
        )
        assert recognized_text.recognition_confidence == 0.95
        
        # Invalid confidence
        with pytest.raises(ValueError, match="Recognition confidence must be between 0.0 and 1.0"):
            RecognizedText(
                detection=self.valid_detection,
                text="Hello",
                language="en",
                recognition_confidence=1.5,
                character_confidences=[0.9, 0.95, 0.92, 0.98, 0.94]
            )
    
    def test_empty_text_validation(self):
        """Test validation with empty text."""
        # Empty text should be valid with empty character confidences
        recognized_text = RecognizedText(
            detection=self.valid_detection,
            text="",
            language="en",
            recognition_confidence=0.0,
            character_confidences=[]
        )
        assert recognized_text.text == ""
        assert len(recognized_text.character_confidences) == 0


class TestLinkedTextValidation:
    """Test validation functions for LinkedText."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.valid_bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.8)
        self.valid_detection = Detection(
            bbox=self.valid_bbox,
            detection_type='word',
            confidence=0.9,
            image_coordinates=(10, 20, 110, 70)
        )
        self.valid_recognized_text = RecognizedText(
            detection=self.valid_detection,
            text="Paris",
            language="en",
            recognition_confidence=0.95,
            character_confidences=[0.9, 0.95, 0.92, 0.98, 0.94]
        )
    
    def test_map_coordinates_validation(self):
        """Test map coordinates validation."""
        # Valid coordinates
        linked_text = LinkedText(
            recognized_text=self.valid_recognized_text,
            map_coordinates=(48.8566, 2.3522),
            spatial_confidence=0.85,
            related_features=["city"]
        )
        assert linked_text.map_coordinates == (48.8566, 2.3522)
        
        # Invalid coordinates - non-numeric
        with pytest.raises(ValueError, match="Map coordinates must be numeric"):
            LinkedText(
                recognized_text=self.valid_recognized_text,
                map_coordinates=("invalid", 2.3522),
                spatial_confidence=0.85,
                related_features=["city"]
            )
        
        with pytest.raises(ValueError, match="Map coordinates must be numeric"):
            LinkedText(
                recognized_text=self.valid_recognized_text,
                map_coordinates=(48.8566, "invalid"),
                spatial_confidence=0.85,
                related_features=["city"]
            )
    
    def test_spatial_confidence_validation(self):
        """Test spatial confidence validation."""
        # Valid confidence
        linked_text = LinkedText(
            recognized_text=self.valid_recognized_text,
            map_coordinates=(48.8566, 2.3522),
            spatial_confidence=0.85,
            related_features=["city"]
        )
        assert linked_text.spatial_confidence == 0.85
        
        # Invalid confidence
        with pytest.raises(ValueError, match="Spatial confidence must be between 0.0 and 1.0"):
            LinkedText(
                recognized_text=self.valid_recognized_text,
                map_coordinates=(48.8566, 2.3522),
                spatial_confidence=1.5,
                related_features=["city"]
            )
    
    def test_related_features_validation(self):
        """Test related features validation."""
        # Valid features
        linked_text = LinkedText(
            recognized_text=self.valid_recognized_text,
            map_coordinates=(48.8566, 2.3522),
            spatial_confidence=0.85,
            related_features=["city", "capital", "location"]
        )
        assert len(linked_text.related_features) == 3
        assert "city" in linked_text.related_features
        
        # Empty features should be valid
        linked_text_empty = LinkedText(
            recognized_text=self.valid_recognized_text,
            map_coordinates=(48.8566, 2.3522),
            spatial_confidence=0.85,
            related_features=[]
        )
        assert len(linked_text_empty.related_features) == 0


class TestProcessingResultValidation:
    """Test validation functions for ProcessingResult."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_detections = TestDataFactory.create_sample_detections(3)
        self.sample_recognized_texts = TestDataFactory.create_sample_recognized_texts(self.sample_detections)
        self.sample_linked_texts = TestDataFactory.create_sample_linked_texts(self.sample_recognized_texts)
    
    def test_task_type_validation(self):
        """Test task type validation."""
        valid_task_types = [
            'word_detection',
            'phrase_detection', 
            'word_detection_recognition',
            'phrase_detection_recognition'
        ]
        
        for task_type in valid_task_types:
            result = ProcessingResult(
                task_type=task_type,
                detections=self.sample_detections,
                recognized_texts=self.sample_recognized_texts,
                linked_texts=self.sample_linked_texts,
                processing_time=1.5,
                metrics={}
            )
            assert result.task_type == task_type
        
        # Invalid task type
        with pytest.raises(ValueError, match="Task type must be one of"):
            ProcessingResult(
                task_type='invalid_task',
                detections=[],
                recognized_texts=[],
                linked_texts=[],
                processing_time=1.5,
                metrics={}
            )
    
    def test_processing_time_validation(self):
        """Test processing time validation."""
        # Valid processing time
        result = ProcessingResult(
            task_type='word_detection',
            detections=self.sample_detections,
            recognized_texts=[],
            linked_texts=[],
            processing_time=2.5,
            metrics={}
        )
        assert result.processing_time == 2.5
        
        # Zero processing time should be valid
        result_zero = ProcessingResult(
            task_type='word_detection',
            detections=[],
            recognized_texts=[],
            linked_texts=[],
            processing_time=0.0,
            metrics={}
        )
        assert result_zero.processing_time == 0.0
        
        # Negative processing time should be invalid
        with pytest.raises(ValueError, match="Processing time must be non-negative"):
            ProcessingResult(
                task_type='word_detection',
                detections=[],
                recognized_texts=[],
                linked_texts=[],
                processing_time=-1.0,
                metrics={}
            )
    
    def test_result_integrity_validation(self):
        """Test result integrity validation."""
        # Valid result with proper relationships
        result = ProcessingResult(
            task_type='word_detection_recognition',
            detections=self.sample_detections,
            recognized_texts=self.sample_recognized_texts,
            linked_texts=self.sample_linked_texts,
            processing_time=1.5,
            metrics={}
        )
        assert len(result.detections) == len(result.recognized_texts)
        
        # More recognized texts than detections should fail
        with pytest.raises(ValueError, match="Cannot have more recognized texts than detections"):
            ProcessingResult(
                task_type='word_detection_recognition',
                detections=[],
                recognized_texts=self.sample_recognized_texts,
                linked_texts=[],
                processing_time=1.5,
                metrics={}
            )
        
        # More linked texts than recognized texts should fail
        with pytest.raises(ValueError, match="Cannot have more linked texts than recognized texts"):
            ProcessingResult(
                task_type='word_detection_recognition',
                detections=self.sample_detections,
                recognized_texts=[],
                linked_texts=self.sample_linked_texts,
                processing_time=1.5,
                metrics={}
            )
    
    def test_metrics_validation(self):
        """Test metrics validation."""
        # Valid metrics
        valid_metrics = {
            'precision': 0.95,
            'recall': 0.90,
            'f1_score': 0.925,
            'accuracy': 0.88
        }
        
        result = ProcessingResult(
            task_type='word_detection',
            detections=self.sample_detections,
            recognized_texts=[],
            linked_texts=[],
            processing_time=1.5,
            metrics=valid_metrics
        )
        assert result.metrics == valid_metrics
        
        # Invalid metric value - non-numeric
        with pytest.raises(ValueError, match="Metric .* must be numeric"):
            ProcessingResult(
                task_type='word_detection',
                detections=[],
                recognized_texts=[],
                linked_texts=[],
                processing_time=1.5,
                metrics={'precision': 'invalid'}
            )


class TestConfigurationValidation:
    """Test validation functions for configuration classes."""
    
    def test_detection_config_validation(self):
        """Test DetectionConfig validation."""
        # Valid configuration
        config = DetectionConfig(
            model_architecture="YOLO",
            input_size=(640, 640),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            batch_size=8
        )
        assert config.model_architecture == "YOLO"
        
        # Invalid architecture
        with pytest.raises(ValueError, match="Architecture must be one of"):
            DetectionConfig(
                model_architecture="INVALID",
                input_size=(640, 640),
                confidence_threshold=0.5,
                nms_threshold=0.4,
                batch_size=8
            )
        
        # Invalid thresholds
        with pytest.raises(ValueError, match="Confidence threshold must be between 0.0 and 1.0"):
            DetectionConfig(
                model_architecture="YOLO",
                input_size=(640, 640),
                confidence_threshold=1.5,
                nms_threshold=0.4,
                batch_size=8
            )
        
        # Invalid batch size
        with pytest.raises(ValueError, match="Batch size must be positive"):
            DetectionConfig(
                model_architecture="YOLO",
                input_size=(640, 640),
                confidence_threshold=0.5,
                nms_threshold=0.4,
                batch_size=0
            )
    
    def test_recognition_config_validation(self):
        """Test RecognitionConfig validation."""
        # Valid configuration
        config = RecognitionConfig(
            model_architecture="CRNN",
            max_sequence_length=25,
            vocabulary_size=1000,
            languages=["en", "zh"],
            beam_width=5
        )
        assert config.model_architecture == "CRNN"
        
        # Invalid architecture
        with pytest.raises(ValueError, match="Architecture must be one of"):
            RecognitionConfig(
                model_architecture="INVALID",
                max_sequence_length=25,
                vocabulary_size=1000,
                languages=["en"],
                beam_width=5
            )
        
        # Invalid sequence length
        with pytest.raises(ValueError, match="Max sequence length must be positive"):
            RecognitionConfig(
                model_architecture="CRNN",
                max_sequence_length=0,
                vocabulary_size=1000,
                languages=["en"],
                beam_width=5
            )
        
        # Invalid language
        with pytest.raises(ValueError, match="Language .* not supported"):
            RecognitionConfig(
                model_architecture="CRNN",
                max_sequence_length=25,
                vocabulary_size=1000,
                languages=["invalid_lang"],
                beam_width=5
            )


class TestValidationUtilitiesIntegration:
    """Test integration of validation utilities with actual validation functions."""
    
    def test_confidence_validation_utility_integration(self):
        """Test confidence validation utility with real validation functions."""
        def bbox_confidence_validation(confidence):
            try:
                BoundingBox(x=10, y=20, width=100, height=50, confidence=confidence)
                return True
            except ValueError:
                return False
        
        # This should pass without exceptions
        ValidationTestUtilities.test_confidence_validation(bbox_confidence_validation)
    
    def test_coordinate_validation_utility_integration(self):
        """Test coordinate validation utility with real validation functions."""
        def detection_coordinate_validation(coords):
            try:
                bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.8)
                Detection(
                    bbox=bbox,
                    detection_type='word',
                    confidence=0.9,
                    image_coordinates=coords
                )
                return True
            except ValueError:
                return False
        
        # This should pass without exceptions
        ValidationTestUtilities.test_coordinate_validation(detection_coordinate_validation)
    
    def test_size_validation_utility_integration(self):
        """Test size validation utility with real validation functions."""
        def input_size_validation(size):
            try:
                DetectionConfig(
                    model_architecture="YOLO",
                    input_size=size,
                    confidence_threshold=0.5,
                    nms_threshold=0.4,
                    batch_size=8
                )
                return True
            except ValueError:
                return False
        
        # This should pass without exceptions
        ValidationTestUtilities.test_size_validation(input_size_validation)