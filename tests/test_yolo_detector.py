"""
Unit tests for YOLO-based word detection.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path

from src.models.data_structures import ProcessedImage, DetectionConfig
from src.detection.yolo_detector import YOLODetectionModel, YOLOWordDetector


class TestYOLODetectionModel(unittest.TestCase):
    """Test cases for YOLODetectionModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DetectionConfig(
            model_architecture='YOLO',
            input_size=(640, 640),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            batch_size=1
        )
        self.model = YOLODetectionModel(self.config)
        
        # Create test image
        self.test_image = ProcessedImage(
            image=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
            original_size=(480, 640),
            processed_size=(640, 640),
            preprocessing_steps=['resize', 'normalize'],
            enhancement_applied=True
        )
        
        # Create temporary model file
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
        self.temp_model_file.write(b'mock model data')
        self.temp_model_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_model_file.name):
            os.unlink(self.temp_model_file.name)
    
    def test_initialization(self):
        """Test YOLO model initialization."""
        self.assertEqual(self.model.config, self.config)
        self.assertIsNone(self.model.model_weights)
        self.assertEqual(self.model.class_names, ['text'])
    
    def test_load_model_success(self):
        """Test successful model loading."""
        self.model.load_model(self.temp_model_file.name)
        
        self.assertIsNotNone(self.model.model_weights)
        self.assertTrue(self.model.model)
        self.assertEqual(self.model.model_weights['format'], 'pytorch')
    
    def test_load_model_file_not_found(self):
        """Test model loading with missing file."""
        with self.assertRaises(FileNotFoundError):
            self.model.load_model('nonexistent_model.pt')
    
    def test_predict_without_model(self):
        """Test prediction without loaded model."""
        with self.assertRaises(RuntimeError):
            self.model.predict(self.test_image)
    
    def test_predict_with_model(self):
        """Test prediction with loaded model."""
        self.model.load_model(self.temp_model_file.name)
        
        predictions = self.model.predict(self.test_image)
        
        self.assertIsInstance(predictions, list)
        # Should have some predictions from mock inference
        for bbox in predictions:
            self.assertGreaterEqual(bbox.confidence, self.config.confidence_threshold)
            self.assertGreater(bbox.width, 0)
            self.assertGreater(bbox.height, 0)
    
    def test_sigmoid_function(self):
        """Test sigmoid activation function."""
        # Test normal values
        self.assertAlmostEqual(YOLODetectionModel._sigmoid(0), 0.5, places=5)
        self.assertAlmostEqual(YOLODetectionModel._sigmoid(1), 0.7311, places=3)
        self.assertAlmostEqual(YOLODetectionModel._sigmoid(-1), 0.2689, places=3)
        
        # Test extreme values (should not cause overflow)
        self.assertAlmostEqual(YOLODetectionModel._sigmoid(1000), 1.0, places=5)
        self.assertAlmostEqual(YOLODetectionModel._sigmoid(-1000), 0.0, places=5)
    
    def test_filter_by_confidence(self):
        """Test confidence filtering."""
        from src.models.data_structures import BoundingBox
        
        # Create test bounding boxes with different confidences
        bboxes = [
            BoundingBox(x=0, y=0, width=10, height=10, confidence=0.9),
            BoundingBox(x=20, y=0, width=10, height=10, confidence=0.3),
            BoundingBox(x=40, y=0, width=10, height=10, confidence=0.7),
        ]
        
        self.model.load_model(self.temp_model_file.name)
        filtered = self.model._filter_by_confidence(bboxes)
        
        # Should keep boxes with confidence >= 0.5
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0].confidence, 0.9)
        self.assertEqual(filtered[1].confidence, 0.7)
    
    def test_get_model_info(self):
        """Test model information retrieval."""
        self.model.load_model(self.temp_model_file.name)
        
        info = self.model.get_model_info()
        
        self.assertEqual(info['model_type'], 'YOLO')
        self.assertEqual(info['architecture'], 'YOLO')
        self.assertEqual(info['num_classes'], 1)
        self.assertEqual(info['class_names'], ['text'])
        self.assertEqual(info['weights_format'], 'pytorch')


class TestYOLOWordDetector(unittest.TestCase):
    """Test cases for YOLOWordDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary model file
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
        self.temp_model_file.write(b'mock model data')
        self.temp_model_file.close()
        
        # Create test image
        self.test_image = ProcessedImage(
            image=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
            original_size=(480, 640),
            processed_size=(640, 640),
            preprocessing_steps=['resize'],
            enhancement_applied=False
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_model_file.name):
            os.unlink(self.temp_model_file.name)
    
    def test_initialization_with_default_config(self):
        """Test detector initialization with default configuration."""
        detector = YOLOWordDetector(self.temp_model_file.name)
        
        self.assertEqual(detector.config.model_architecture, 'YOLO')
        self.assertEqual(detector.config.input_size, (640, 640))
        self.assertEqual(detector.config.confidence_threshold, 0.5)
        self.assertEqual(detector.config.nms_threshold, 0.4)
    
    def test_initialization_with_custom_config(self):
        """Test detector initialization with custom configuration."""
        config = DetectionConfig(
            model_architecture='YOLO',
            input_size=(512, 512),
            confidence_threshold=0.6,
            nms_threshold=0.3,
            batch_size=2
        )
        
        detector = YOLOWordDetector(self.temp_model_file.name, config)
        
        self.assertEqual(detector.config, config)
    
    def test_detect_words(self):
        """Test word detection functionality."""
        detector = YOLOWordDetector(self.temp_model_file.name)
        
        words = detector.detect_words(self.test_image)
        
        self.assertIsInstance(words, list)
        # Should have some detections from mock model
        for bbox in words:
            self.assertGreaterEqual(bbox.confidence, detector.config.confidence_threshold)
    
    def test_detect_words_with_threshold_override(self):
        """Test word detection with threshold overrides."""
        detector = YOLOWordDetector(self.temp_model_file.name)
        
        # Test with higher confidence threshold
        words_high_conf = detector.detect_words(
            self.test_image, 
            confidence_threshold=0.8
        )
        
        # Test with lower confidence threshold
        words_low_conf = detector.detect_words(
            self.test_image,
            confidence_threshold=0.2
        )
        
        # Should have fewer detections with higher threshold
        self.assertLessEqual(len(words_high_conf), len(words_low_conf))
        
        # Original config should be restored
        self.assertEqual(detector.config.confidence_threshold, 0.5)
    
    def test_get_performance_stats(self):
        """Test performance statistics retrieval."""
        detector = YOLOWordDetector(self.temp_model_file.name)
        
        stats = detector.get_performance_stats()
        
        self.assertIn('model_info', stats)
        self.assertIn('config', stats)
        self.assertIn('thresholds', stats)
        
        self.assertEqual(stats['thresholds']['confidence'], 0.5)
        self.assertEqual(stats['thresholds']['nms'], 0.4)


class TestYOLOIntegration(unittest.TestCase):
    """Integration tests for YOLO detection pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create temporary model file
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        self.temp_model_file.write(b'mock onnx model data')
        self.temp_model_file.close()
        
        # Create realistic test image
        self.test_image = ProcessedImage(
            image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            original_size=(480, 640),
            processed_size=(640, 640),
            preprocessing_steps=['resize', 'normalize', 'enhance'],
            enhancement_applied=True
        )
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        if os.path.exists(self.temp_model_file.name):
            os.unlink(self.temp_model_file.name)
    
    def test_end_to_end_detection(self):
        """Test complete detection pipeline."""
        # Initialize detector
        config = DetectionConfig(
            model_architecture='YOLO',
            input_size=(640, 640),
            confidence_threshold=0.3,
            nms_threshold=0.5,
            batch_size=1
        )
        
        detector = YOLOWordDetector(self.temp_model_file.name, config)
        
        # Run detection
        detections = detector.detect_words(self.test_image)
        
        # Verify results
        self.assertIsInstance(detections, list)
        
        # Check that all detections meet minimum requirements
        for bbox in detections:
            self.assertGreaterEqual(bbox.confidence, config.confidence_threshold)
            self.assertGreater(bbox.width, 0)
            self.assertGreater(bbox.height, 0)
            self.assertGreaterEqual(bbox.x, 0)
            self.assertGreaterEqual(bbox.y, 0)
    
    def test_different_model_formats(self):
        """Test loading different model formats."""
        formats = ['.pt', '.onnx', '.weights']
        
        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=fmt, delete=False) as temp_file:
                temp_file.write(b'mock model data')
                temp_file.close()
                
                try:
                    model = YOLODetectionModel(self.config)
                    model.load_model(temp_file.name)
                    
                    # Verify format detection
                    expected_format = {
                        '.pt': 'pytorch',
                        '.onnx': 'onnx', 
                        '.weights': 'darknet'
                    }[fmt]
                    
                    self.assertEqual(model.model_weights['format'], expected_format)
                    
                finally:
                    os.unlink(temp_file.name)
    
    @property
    def config(self):
        """Get test configuration."""
        return DetectionConfig(
            model_architecture='YOLO',
            input_size=(640, 640),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            batch_size=1
        )


if __name__ == '__main__':
    unittest.main()