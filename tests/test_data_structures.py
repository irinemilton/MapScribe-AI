"""
Unit tests for core data structures and validation functions.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.models.data_structures import (
    BoundingBox, Detection, RecognizedText, LinkedText, ProcessingResult,
    ProcessedImage, ModelMetadata, DetectionConfig, RecognitionConfig
)


class TestBoundingBox:
    """Test cases for BoundingBox data structure."""
    
    def test_valid_bounding_box_creation(self):
        """Test creating a valid bounding box."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.8)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.confidence == 0.8
    
    def test_bounding_box_area_calculation(self):
        """Test area calculation."""
        bbox = BoundingBox(x=0, y=0, width=100, height=50, confidence=0.8)
        assert bbox.area == 5000
    
    def test_bounding_box_coordinates_property(self):
        """Test coordinates property."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.8)
        assert bbox.coordinates == (10, 20, 110, 70)
    
    def test_invalid_negative_coordinates(self):
        """Test validation of negative coordinates."""
        with pytest.raises(ValueError, match="Coordinates must be non-negative"):
            BoundingBox(x=-1, y=0, width=100, height=50, confidence=0.8)
        
        with pytest.raises(ValueError, match="Coordinates must be non-negative"):
            BoundingBox(x=0, y=-1, width=100, height=50, confidence=0.8)
    
    def test_invalid_zero_dimensions(self):
        """Test validation of zero or negative dimensions."""
        with pytest.raises(ValueError, match="Width and height must be positive"):
            BoundingBox(x=0, y=0, width=0, height=50, confidence=0.8)
        
        with pytest.raises(ValueError, match="Width and height must be positive"):
            BoundingBox(x=0, y=0, width=100, height=-1, confidence=0.8)
    
    def test_invalid_confidence_range(self):
        """Test validation of confidence score range."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            BoundingBox(x=0, y=0, width=100, height=50, confidence=-0.1)
        
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            BoundingBox(x=0, y=0, width=100, height=50, confidence=1.1)
    
    def test_bounding_box_serialization(self):
        """Test serialization to dictionary and JSON."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.8)
        
        # Test dictionary serialization
        bbox_dict = bbox.to_dict()
        expected_dict = {
            'x': 10, 'y': 20, 'width': 100, 'height': 50, 'confidence': 0.8
        }
        assert bbox_dict == expected_dict
        
        # Test JSON serialization
        bbox_json = bbox.to_json()
        assert isinstance(bbox_json, str)
        assert json.loads(bbox_json) == expected_dict
    
    def test_bounding_box_deserialization(self):
        """Test deserialization from dictionary and JSON."""
        original_bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.8)
        
        # Test dictionary deserialization
        bbox_dict = original_bbox.to_dict()
        restored_bbox = BoundingBox.from_dict(bbox_dict)
        assert restored_bbox == original_bbox
        
        # Test JSON deserialization
        bbox_json = original_bbox.to_json()
        restored_bbox = BoundingBox.from_json(bbox_json)
        assert restored_bbox == original_bbox


class TestDetection:
    """Test cases for Detection data structure."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.valid_bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.8)
        self.valid_image_coords = (10, 20, 110, 70)
    
    def test_valid_detection_creation(self):
        """Test creating a valid detection."""
        detection = Detection(
            bbox=self.valid_bbox,
            detection_type='word',
            confidence=0.9,
            image_coordinates=self.valid_image_coords
        )
        assert detection.bbox == self.valid_bbox
        assert detection.detection_type == 'word'
        assert detection.confidence == 0.9
        assert detection.image_coordinates == self.valid_image_coords
    
    def test_invalid_detection_type(self):
        """Test validation of detection type."""
        with pytest.raises(ValueError, match="Detection type must be 'word' or 'phrase'"):
            Detection(
                bbox=self.valid_bbox,
                detection_type='invalid',
                confidence=0.9,
                image_coordinates=self.valid_image_coords
            )
    
    def test_invalid_confidence_range(self):
        """Test validation of confidence score range."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Detection(
                bbox=self.valid_bbox,
                detection_type='word',
                confidence=1.5,
                image_coordinates=self.valid_image_coords
            )
    
    def test_invalid_image_coordinates(self):
        """Test validation of image coordinates."""
        with pytest.raises(ValueError, match="Invalid image coordinates"):
            Detection(
                bbox=self.valid_bbox,
                detection_type='word',
                confidence=0.9,
                image_coordinates=(-1, 20, 110, 70)  # negative x1
            )
        
        with pytest.raises(ValueError, match="Invalid image coordinates"):
            Detection(
                bbox=self.valid_bbox,
                detection_type='word',
                confidence=0.9,
                image_coordinates=(110, 20, 10, 70)  # x2 <= x1
            )
    
    def test_detection_serialization(self):
        """Test detection serialization and deserialization."""
        detection = Detection(
            bbox=self.valid_bbox,
            detection_type='word',
            confidence=0.9,
            image_coordinates=self.valid_image_coords
        )
        
        # Test dictionary serialization
        detection_dict = detection.to_dict()
        restored_detection = Detection.from_dict(detection_dict)
        assert restored_detection.bbox.x == detection.bbox.x
        assert restored_detection.detection_type == detection.detection_type
        
        # Test JSON serialization
        detection_json = detection.to_json()
        restored_detection = Detection.from_json(detection_json)
        assert restored_detection.confidence == detection.confidence


class TestRecognizedText:
    """Test cases for RecognizedText data structure."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.valid_bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.8)
        self.valid_detection = Detection(
            bbox=self.valid_bbox,
            detection_type='word',
            confidence=0.9,
            image_coordinates=(10, 20, 110, 70)
        )
    
    def test_valid_recognized_text_creation(self):
        """Test creating valid recognized text."""
        recognized_text = RecognizedText(
            detection=self.valid_detection,
            text="Hello",
            language="en",
            recognition_confidence=0.95,
            character_confidences=[0.9, 0.95, 0.92, 0.98, 0.94]
        )
        assert recognized_text.text == "Hello"
        assert recognized_text.language == "en"
        assert len(recognized_text.character_confidences) == 5
    
    def test_invalid_recognition_confidence(self):
        """Test validation of recognition confidence."""
        with pytest.raises(ValueError, match="Recognition confidence must be between 0.0 and 1.0"):
            RecognizedText(
                detection=self.valid_detection,
                text="Hello",
                language="en",
                recognition_confidence=1.5,
                character_confidences=[0.9, 0.95, 0.92, 0.98, 0.94]
            )
    
    def test_mismatched_character_confidences(self):
        """Test validation of character confidences length."""
        with pytest.raises(ValueError, match="Character confidences must match text length"):
            RecognizedText(
                detection=self.valid_detection,
                text="Hello",
                language="en",
                recognition_confidence=0.95,
                character_confidences=[0.9, 0.95, 0.92]  # Too few confidences
            )
    
    def test_invalid_character_confidence_range(self):
        """Test validation of character confidence ranges."""
        with pytest.raises(ValueError, match="Character confidence must be between 0.0 and 1.0"):
            RecognizedText(
                detection=self.valid_detection,
                text="Hello",
                language="en",
                recognition_confidence=0.95,
                character_confidences=[0.9, 1.5, 0.92, 0.98, 0.94]  # Invalid confidence
            )
    
    def test_invalid_language(self):
        """Test validation of language code."""
        with pytest.raises(ValueError, match="Language must be one of"):
            RecognizedText(
                detection=self.valid_detection,
                text="Hello",
                language="invalid",
                recognition_confidence=0.95,
                character_confidences=[0.9, 0.95, 0.92, 0.98, 0.94]
            )
    
    def test_recognized_text_serialization(self):
        """Test recognized text serialization."""
        recognized_text = RecognizedText(
            detection=self.valid_detection,
            text="Hello",
            language="en",
            recognition_confidence=0.95,
            character_confidences=[0.9, 0.95, 0.92, 0.98, 0.94]
        )
        
        # Test serialization and deserialization
        text_dict = recognized_text.to_dict()
        restored_text = RecognizedText.from_dict(text_dict)
        assert restored_text.text == recognized_text.text
        assert restored_text.language == recognized_text.language


class TestLinkedText:
    """Test cases for LinkedText data structure."""
    
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
    
    def test_valid_linked_text_creation(self):
        """Test creating valid linked text."""
        linked_text = LinkedText(
            recognized_text=self.valid_recognized_text,
            map_coordinates=(48.8566, 2.3522),  # Paris coordinates
            spatial_confidence=0.85,
            related_features=["city", "capital"]
        )
        assert linked_text.map_coordinates == (48.8566, 2.3522)
        assert linked_text.spatial_confidence == 0.85
        assert "city" in linked_text.related_features
    
    def test_invalid_spatial_confidence(self):
        """Test validation of spatial confidence."""
        with pytest.raises(ValueError, match="Spatial confidence must be between 0.0 and 1.0"):
            LinkedText(
                recognized_text=self.valid_recognized_text,
                map_coordinates=(48.8566, 2.3522),
                spatial_confidence=1.5,
                related_features=["city"]
            )
    
    def test_invalid_map_coordinates(self):
        """Test validation of map coordinates."""
        with pytest.raises(ValueError, match="Map coordinates must be numeric"):
            LinkedText(
                recognized_text=self.valid_recognized_text,
                map_coordinates=("invalid", 2.3522),
                spatial_confidence=0.85,
                related_features=["city"]
            )
    
    def test_linked_text_serialization(self):
        """Test linked text serialization."""
        linked_text = LinkedText(
            recognized_text=self.valid_recognized_text,
            map_coordinates=(48.8566, 2.3522),
            spatial_confidence=0.85,
            related_features=["city", "capital"]
        )
        
        # Test serialization and deserialization
        linked_dict = linked_text.to_dict()
        restored_linked = LinkedText.from_dict(linked_dict)
        assert restored_linked.map_coordinates == linked_text.map_coordinates
        assert restored_linked.spatial_confidence == linked_text.spatial_confidence


class TestProcessingResult:
    """Test cases for ProcessingResult data structure."""
    
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
        self.valid_linked_text = LinkedText(
            recognized_text=self.valid_recognized_text,
            map_coordinates=(48.8566, 2.3522),
            spatial_confidence=0.85,
            related_features=["city"]
        )
    
    def test_valid_processing_result_creation(self):
        """Test creating valid processing result."""
        result = ProcessingResult(
            task_type='word_detection_recognition',
            detections=[self.valid_detection],
            recognized_texts=[self.valid_recognized_text],
            linked_texts=[self.valid_linked_text],
            processing_time=1.5,
            metrics={'precision': 0.95, 'recall': 0.90, 'f1_score': 0.925}
        )
        assert result.task_type == 'word_detection_recognition'
        assert len(result.detections) == 1
        assert result.processing_time == 1.5
    
    def test_invalid_task_type(self):
        """Test validation of task type."""
        with pytest.raises(ValueError, match="Task type must be one of"):
            ProcessingResult(
                task_type='invalid_task',
                detections=[],
                recognized_texts=[],
                linked_texts=[],
                processing_time=1.5,
                metrics={}
            )
    
    def test_invalid_processing_time(self):
        """Test validation of processing time."""
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
        """Test validation of result integrity."""
        # Test more recognized texts than detections
        with pytest.raises(ValueError, match="Cannot have more recognized texts than detections"):
            ProcessingResult(
                task_type='word_detection_recognition',
                detections=[],
                recognized_texts=[self.valid_recognized_text],
                linked_texts=[],
                processing_time=1.5,
                metrics={}
            )
        
        # Test more linked texts than recognized texts
        with pytest.raises(ValueError, match="Cannot have more linked texts than recognized texts"):
            ProcessingResult(
                task_type='word_detection_recognition',
                detections=[self.valid_detection],
                recognized_texts=[],
                linked_texts=[self.valid_linked_text],
                processing_time=1.5,
                metrics={}
            )
    
    def test_invalid_metrics(self):
        """Test validation of metrics."""
        with pytest.raises(ValueError, match="Metric .* must be numeric"):
            ProcessingResult(
                task_type='word_detection',
                detections=[],
                recognized_texts=[],
                linked_texts=[],
                processing_time=1.5,
                metrics={'precision': 'invalid'}
            )
    
    def test_processing_result_methods(self):
        """Test processing result utility methods."""
        result = ProcessingResult(
            task_type='word_detection_recognition',
            detections=[self.valid_detection],
            recognized_texts=[self.valid_recognized_text],
            linked_texts=[self.valid_linked_text],
            processing_time=1.5,
            metrics={'precision': 0.95}
        )
        
        assert result.get_detection_count() == 1
        assert result.get_recognition_count() == 1
        assert result.get_linking_count() == 1
        assert result.get_average_confidence() == 0.9
    
    def test_processing_result_file_operations(self):
        """Test file save and load operations."""
        result = ProcessingResult(
            task_type='word_detection',
            detections=[self.valid_detection],
            recognized_texts=[],
            linked_texts=[],
            processing_time=1.5,
            metrics={'precision': 0.95}
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test JSON file operations
            json_path = Path(temp_dir) / "result.json"
            result.save_to_file(json_path)
            loaded_result = ProcessingResult.load_from_file(json_path)
            assert loaded_result.task_type == result.task_type
            
            # Test pickle file operations
            pkl_path = Path(temp_dir) / "result.pkl"
            result.save_to_file(pkl_path)
            loaded_result = ProcessingResult.load_from_file(pkl_path)
            assert loaded_result.processing_time == result.processing_time
            
            # Test unsupported format
            with pytest.raises(ValueError, match="Unsupported file format"):
                result.save_to_file(Path(temp_dir) / "result.txt")


class TestProcessedImage:
    """Test cases for ProcessedImage data structure."""
    
    def test_valid_processed_image_creation(self):
        """Test creating valid processed image."""
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        processed_image = ProcessedImage(
            image=image,
            original_size=(300, 400),
            processed_size=(100, 200),
            preprocessing_steps=["resize", "normalize"],
            enhancement_applied=True
        )
        assert processed_image.image.shape == (100, 200, 3)
        assert processed_image.original_size == (300, 400)
        assert processed_image.enhancement_applied is True
    
    def test_invalid_image_dimensions(self):
        """Test validation of image dimensions."""
        with pytest.raises(ValueError, match="Image must be 2D or 3D array"):
            ProcessedImage(
                image=np.random.randint(0, 255, (100, 200, 3, 4)),  # 4D array
                original_size=(300, 400),
                processed_size=(100, 200),
                preprocessing_steps=[],
                enhancement_applied=False
            )
    
    def test_invalid_processed_size(self):
        """Test validation of processed size."""
        image = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        with pytest.raises(ValueError, match="Processed size must be positive"):
            ProcessedImage(
                image=image,
                original_size=(300, 400),
                processed_size=(0, 200),  # Invalid size
                preprocessing_steps=[],
                enhancement_applied=False
            )


class TestModelMetadata:
    """Test cases for ModelMetadata data structure."""
    
    def test_valid_model_metadata_creation(self):
        """Test creating valid model metadata."""
        metadata = ModelMetadata(
            model_name="YOLO_v5",
            version="1.0.0",
            architecture="YOLO",
            input_size=(640, 640),
            supported_languages=["en", "zh"],
            training_dataset="ICDAR2025",
            performance_metrics={"precision": 0.95, "recall": 0.90, "f1_score": 0.925}
        )
        assert metadata.model_name == "YOLO_v5"
        assert metadata.version == "1.0.0"
        assert len(metadata.supported_languages) == 2
    
    def test_invalid_basic_info(self):
        """Test validation of basic model information."""
        with pytest.raises(ValueError, match="Model name and version are required"):
            ModelMetadata(
                model_name="",
                version="1.0.0",
                architecture="YOLO",
                input_size=(640, 640),
                supported_languages=["en"],
                training_dataset="ICDAR2025",
                performance_metrics={}
            )
    
    def test_invalid_input_size(self):
        """Test validation of input size."""
        with pytest.raises(ValueError, match="Input size must be positive"):
            ModelMetadata(
                model_name="YOLO_v5",
                version="1.0.0",
                architecture="YOLO",
                input_size=(0, 640),
                supported_languages=["en"],
                training_dataset="ICDAR2025",
                performance_metrics={}
            )
    
    def test_invalid_performance_metrics(self):
        """Test validation of performance metrics."""
        with pytest.raises(ValueError, match="Performance metric .* must be numeric"):
            ModelMetadata(
                model_name="YOLO_v5",
                version="1.0.0",
                architecture="YOLO",
                input_size=(640, 640),
                supported_languages=["en"],
                training_dataset="ICDAR2025",
                performance_metrics={"precision": "invalid"}
            )
        
        with pytest.raises(ValueError, match="Performance metric .* must be between 0 and 1"):
            ModelMetadata(
                model_name="YOLO_v5",
                version="1.0.0",
                architecture="YOLO",
                input_size=(640, 640),
                supported_languages=["en"],
                training_dataset="ICDAR2025",
                performance_metrics={"precision": 1.5}
            )


class TestDetectionConfig:
    """Test cases for DetectionConfig data structure."""
    
    def test_valid_detection_config_creation(self):
        """Test creating valid detection configuration."""
        config = DetectionConfig(
            model_architecture="YOLO",
            input_size=(640, 640),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            batch_size=8
        )
        assert config.model_architecture == "YOLO"
        assert config.confidence_threshold == 0.5
        assert config.batch_size == 8
    
    def test_invalid_architecture(self):
        """Test validation of model architecture."""
        with pytest.raises(ValueError, match="Architecture must be one of"):
            DetectionConfig(
                model_architecture="INVALID",
                input_size=(640, 640),
                confidence_threshold=0.5,
                nms_threshold=0.4,
                batch_size=8
            )
    
    def test_invalid_thresholds(self):
        """Test validation of threshold parameters."""
        with pytest.raises(ValueError, match="Confidence threshold must be between 0.0 and 1.0"):
            DetectionConfig(
                model_architecture="YOLO",
                input_size=(640, 640),
                confidence_threshold=1.5,
                nms_threshold=0.4,
                batch_size=8
            )
    
    def test_invalid_batch_size(self):
        """Test validation of batch size."""
        with pytest.raises(ValueError, match="Batch size must be positive"):
            DetectionConfig(
                model_architecture="YOLO",
                input_size=(640, 640),
                confidence_threshold=0.5,
                nms_threshold=0.4,
                batch_size=0
            )


class TestRecognitionConfig:
    """Test cases for RecognitionConfig data structure."""
    
    def test_valid_recognition_config_creation(self):
        """Test creating valid recognition configuration."""
        config = RecognitionConfig(
            model_architecture="CRNN",
            max_sequence_length=25,
            vocabulary_size=1000,
            languages=["en", "zh"],
            beam_width=5
        )
        assert config.model_architecture == "CRNN"
        assert config.max_sequence_length == 25
        assert len(config.languages) == 2
    
    def test_invalid_architecture(self):
        """Test validation of model architecture."""
        with pytest.raises(ValueError, match="Architecture must be one of"):
            RecognitionConfig(
                model_architecture="INVALID",
                max_sequence_length=25,
                vocabulary_size=1000,
                languages=["en"],
                beam_width=5
            )
    
    def test_invalid_sequence_params(self):
        """Test validation of sequence parameters."""
        with pytest.raises(ValueError, match="Max sequence length must be positive"):
            RecognitionConfig(
                model_architecture="CRNN",
                max_sequence_length=0,
                vocabulary_size=1000,
                languages=["en"],
                beam_width=5
            )
    
    def test_invalid_languages(self):
        """Test validation of language configuration."""
        with pytest.raises(ValueError, match="Language .* not supported"):
            RecognitionConfig(
                model_architecture="CRNN",
                max_sequence_length=25,
                vocabulary_size=1000,
                languages=["invalid_lang"],
                beam_width=5
            )