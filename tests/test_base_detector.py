"""
Unit tests for base text detection functionality.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch

from src.models.data_structures import (
    BoundingBox, Detection, ProcessedImage, DetectionConfig
)
from src.detection.base_detector import TextDetector, BaseDetectionModel
from src.detection.bbox_utils import BoundingBoxUtils


class MockDetectionModel(BaseDetectionModel):
    """Mock detection model for testing."""
    
    def load_model(self, model_path: str) -> None:
        """Mock model loading."""
        self.model = Mock()
    
    def predict(self, image: ProcessedImage) -> list:
        """Mock prediction returning test bounding boxes."""
        return [
            BoundingBox(x=10, y=10, width=50, height=20, confidence=0.9),
            BoundingBox(x=70, y=10, width=40, height=20, confidence=0.8),
            BoundingBox(x=10, y=40, width=60, height=25, confidence=0.7),
        ]


class TestTextDetector(unittest.TestCase):
    """Test cases for TextDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DetectionConfig(
            model_architecture='YOLO',
            input_size=(640, 640),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            batch_size=1
        )
        self.detector = TextDetector(self.config)
        
        # Create test image
        self.test_image = ProcessedImage(
            image=np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8),
            original_size=(100, 200),
            processed_size=(100, 200),
            preprocessing_steps=['resize'],
            enhancement_applied=False
        )
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.config, self.config)
        self.assertIsNone(self.detector.model)
        self.assertIsNone(self.detector.metadata)
    
    def test_load_model_success(self):
        """Test successful model loading."""
        with patch('pathlib.Path.exists', return_value=True):
            self.detector.load_model('test_model.pth', MockDetectionModel)
            
            self.assertIsNotNone(self.detector.model)
            self.assertIsNotNone(self.detector.metadata)
            self.assertEqual(self.detector.metadata.architecture, 'YOLO')
    
    def test_load_model_file_not_found(self):
        """Test model loading with missing file."""
        with patch('pathlib.Path.exists', return_value=False):
            with self.assertRaises(FileNotFoundError):
                self.detector.load_model('missing_model.pth', MockDetectionModel)
    
    def test_detect_words_without_model(self):
        """Test word detection without loaded model."""
        with self.assertRaises(RuntimeError):
            self.detector.detect_words(self.test_image)
    
    def test_detect_words_with_model(self):
        """Test word detection with loaded model."""
        with patch('pathlib.Path.exists', return_value=True):
            self.detector.load_model('test_model.pth', MockDetectionModel)
            
            detections = self.detector.detect_words(self.test_image)
            
            self.assertIsInstance(detections, list)
            self.assertTrue(len(detections) > 0)
            
            for detection in detections:
                self.assertIsInstance(detection, Detection)
                self.assertEqual(detection.detection_type, 'word')
                self.assertGreaterEqual(detection.confidence, self.config.confidence_threshold)
    
    def test_detect_phrases(self):
        """Test phrase detection."""
        with patch('pathlib.Path.exists', return_value=True):
            self.detector.load_model('test_model.pth', MockDetectionModel)
            
            phrases = self.detector.detect_phrases(self.test_image)
            
            self.assertIsInstance(phrases, list)
            for phrase in phrases:
                self.assertIsInstance(phrase, Detection)
                self.assertEqual(phrase.detection_type, 'phrase')
    
    def test_get_confidence_scores(self):
        """Test confidence score extraction."""
        detections = [
            Detection(
                bbox=BoundingBox(x=0, y=0, width=10, height=10, confidence=0.9),
                detection_type='word',
                confidence=0.9,
                image_coordinates=(0, 0, 10, 10)
            ),
            Detection(
                bbox=BoundingBox(x=20, y=0, width=10, height=10, confidence=0.7),
                detection_type='word', 
                confidence=0.7,
                image_coordinates=(20, 0, 30, 10)
            )
        ]
        
        scores = self.detector.get_confidence_scores(detections)
        self.assertEqual(scores, [0.9, 0.7])
    
    def test_filter_detections(self):
        """Test detection filtering by confidence."""
        detections = [
            Detection(
                bbox=BoundingBox(x=0, y=0, width=10, height=10, confidence=0.9),
                detection_type='word',
                confidence=0.9,
                image_coordinates=(0, 0, 10, 10)
            ),
            Detection(
                bbox=BoundingBox(x=20, y=0, width=10, height=10, confidence=0.3),
                detection_type='word',
                confidence=0.3,
                image_coordinates=(20, 0, 30, 10)
            )
        ]
        
        filtered = self.detector.filter_detections(detections, 0.5)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].confidence, 0.9)
    
    def test_filter_detections_invalid_threshold(self):
        """Test detection filtering with invalid threshold."""
        detections = []
        
        with self.assertRaises(ValueError):
            self.detector.filter_detections(detections, -0.1)
        
        with self.assertRaises(ValueError):
            self.detector.filter_detections(detections, 1.1)
    
    def test_calculate_iou(self):
        """Test IoU calculation."""
        bbox1 = BoundingBox(x=0, y=0, width=10, height=10, confidence=0.9)
        bbox2 = BoundingBox(x=5, y=5, width=10, height=10, confidence=0.8)
        
        iou = self.detector.calculate_iou(bbox1, bbox2)
        
        # Expected IoU: intersection = 5*5 = 25, union = 100 + 100 - 25 = 175
        expected_iou = 25 / 175
        self.assertAlmostEqual(iou, expected_iou, places=5)
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation with no overlap."""
        bbox1 = BoundingBox(x=0, y=0, width=10, height=10, confidence=0.9)
        bbox2 = BoundingBox(x=20, y=20, width=10, height=10, confidence=0.8)
        
        iou = self.detector.calculate_iou(bbox1, bbox2)
        self.assertEqual(iou, 0.0)
    
    def test_validate_bounding_box(self):
        """Test bounding box validation."""
        # Valid bounding box
        valid_bbox = BoundingBox(x=10, y=10, width=50, height=30, confidence=0.8)
        self.assertTrue(self.detector.validate_bounding_box(valid_bbox, (100, 200)))
        
        # Invalid bounding box (exceeds image bounds)
        invalid_bbox2 = BoundingBox(x=150, y=10, width=100, height=30, confidence=0.8)
        self.assertFalse(self.detector.validate_bounding_box(invalid_bbox2, (100, 200)))
        
        # Invalid bounding box (too small)
        invalid_bbox3 = BoundingBox(x=10, y=10, width=2, height=2, confidence=0.8)
        self.assertFalse(self.detector.validate_bounding_box(invalid_bbox3, (100, 200)))


class TestBoundingBoxUtils(unittest.TestCase):
    """Test cases for BoundingBoxUtils class."""
    
    def test_calculate_iou(self):
        """Test IoU calculation utility."""
        bbox1 = BoundingBox(x=0, y=0, width=10, height=10, confidence=0.9)
        bbox2 = BoundingBox(x=5, y=5, width=10, height=10, confidence=0.8)
        
        iou = BoundingBoxUtils.calculate_iou(bbox1, bbox2)
        expected_iou = 25 / 175  # intersection=25, union=175
        self.assertAlmostEqual(iou, expected_iou, places=5)
    
    def test_merge_bboxes(self):
        """Test bounding box merging."""
        bboxes = [
            BoundingBox(x=0, y=0, width=10, height=10, confidence=0.9),
            BoundingBox(x=5, y=5, width=10, height=10, confidence=0.8),
            BoundingBox(x=10, y=0, width=5, height=15, confidence=0.7)
        ]
        
        merged = BoundingBoxUtils.merge_bboxes(bboxes)
        
        self.assertEqual(merged.x, 0)
        self.assertEqual(merged.y, 0)
        self.assertEqual(merged.width, 15)  # 0 to 15
        self.assertEqual(merged.height, 15)  # 0 to 15
        self.assertAlmostEqual(merged.confidence, 0.8, places=1)  # average
    
    def test_merge_bboxes_empty_list(self):
        """Test merging empty list of bounding boxes."""
        with self.assertRaises(ValueError):
            BoundingBoxUtils.merge_bboxes([])
    
    def test_validate_bbox(self):
        """Test bounding box validation utility."""
        valid_bbox = BoundingBox(x=10, y=10, width=50, height=30, confidence=0.8)
        self.assertTrue(BoundingBoxUtils.validate_bbox(valid_bbox, (100, 200)))
        
        # Test with bbox that exceeds bounds
        invalid_bbox = BoundingBox(x=150, y=10, width=100, height=30, confidence=0.8)
        self.assertFalse(BoundingBoxUtils.validate_bbox(invalid_bbox, (100, 200)))


if __name__ == '__main__':
    unittest.main()