"""
Unit tests for mock objects and validation functions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple

from tests.test_utilities import (
    MockImageGenerator, MockModelObjects, TestDataFactory,
    ValidationTestUtilities, PerformanceTestUtilities
)
from src.models.data_structures import (
    BoundingBox, Detection, RecognizedText, LinkedText, ProcessingResult
)


class TestMockImageGenerator:
    """Test cases for MockImageGenerator utility class."""
    
    def test_create_test_image_default(self):
        """Test creating default test image."""
        image = MockImageGenerator.create_test_image()
        assert image.shape == (480, 640, 3)
        assert image.dtype == np.uint8
        assert np.all(image >= 0) and np.all(image <= 255)
    
    def test_create_test_image_custom_dimensions(self):
        """Test creating test image with custom dimensions."""
        image = MockImageGenerator.create_test_image(width=800, height=600, channels=1)
        assert image.shape == (600, 800)
        assert image.dtype == np.uint8
    
    def test_create_test_image_grayscale(self):
        """Test creating grayscale test image."""
        image = MockImageGenerator.create_test_image(channels=1)
        assert len(image.shape) == 2
        assert image.dtype == np.uint8
    
    def test_create_historical_map_image(self):
        """Test creating historical map image."""
        image = MockImageGenerator.create_historical_map_image()
        assert image.shape == (768, 1024, 3)
        assert image.dtype == np.uint8
        
        # Check that image has aged paper-like colors (not pure white)
        mean_color = np.mean(image, axis=(0, 1))
        assert mean_color[0] < 255  # Red channel
        assert mean_color[1] < 255  # Green channel
        assert mean_color[2] < 255  # Blue channel
    
    def test_create_degraded_image(self):
        """Test image degradation functionality."""
        base_image = MockImageGenerator.create_test_image(width=100, height=100)
        degraded_image = MockImageGenerator.create_degraded_image(base_image, degradation_level=0.5)
        
        assert degraded_image.shape == base_image.shape
        assert degraded_image.dtype == base_image.dtype
        
        # Degraded image should be different from original
        assert not np.array_equal(base_image, degraded_image)
    
    def test_create_text_region_image(self):
        """Test creating text region image."""
        image = MockImageGenerator.create_text_region_image("Test", width=200, height=50)
        assert image.shape == (50, 200, 3)
        
        # Should have some dark pixels (text region)
        dark_pixels = np.sum(image < 50)
        assert dark_pixels > 0


class TestMockModelObjects:
    """Test cases for MockModelObjects factory class."""
    
    def test_create_mock_detector(self):
        """Test creating mock detector."""
        mock_detector = MockModelObjects.create_mock_detector()
        
        # Test word detection
        word_detections = mock_detector.detect_words(Mock())
        assert len(word_detections) == 1
        assert word_detections[0].detection_type == 'word'
        assert isinstance(word_detections[0], Detection)
        
        # Test phrase detection
        phrase_detections = mock_detector.detect_phrases(Mock())
        assert len(phrase_detections) == 1
        assert phrase_detections[0].detection_type == 'phrase'
        
        # Test confidence scores
        confidence_scores = mock_detector.get_confidence_scores([])
        assert isinstance(confidence_scores, list)
        assert len(confidence_scores) == 2
    
    def test_create_mock_recognizer(self):
        """Test creating mock recognizer."""
        mock_recognizer = MockModelObjects.create_mock_recognizer()
        
        # Test word recognition
        word_results = mock_recognizer.recognize_words([])
        assert len(word_results) == 1
        assert isinstance(word_results[0], RecognizedText)
        assert word_results[0].text == "Sample"
        
        # Test phrase recognition
        phrase_results = mock_recognizer.recognize_phrases([])
        assert len(phrase_results) == 1
        assert phrase_results[0].text == "Sample Text"
        
        # Test multilingual handling
        multilingual_result = mock_recognizer.handle_multilingual(Mock())
        assert "Mixed Text" in multilingual_result
        assert "混合文本" in multilingual_result
    
    def test_create_mock_linker(self):
        """Test creating mock linker."""
        mock_linker = MockModelObjects.create_mock_linker()
        
        # Test coordinate linking
        linked_results = mock_linker.link_to_coordinates([])
        assert len(linked_results) == 1
        assert isinstance(linked_results[0], LinkedText)
        assert linked_results[0].recognized_text.text == "Paris"
        assert linked_results[0].map_coordinates == (48.8566, 2.3522)
        
        # Test text grouping
        grouped_results = mock_linker.group_related_text([])
        assert len(grouped_results) == 1
        assert isinstance(grouped_results[0], list)
        
        # Test spatial confidence calculation
        spatial_confidence = mock_linker.calculate_spatial_confidence(Mock())
        assert isinstance(spatial_confidence, float)
        assert 0.0 <= spatial_confidence <= 1.0
    
    def test_create_mock_model_manager(self):
        """Test creating mock model manager."""
        mock_manager = MockModelObjects.create_mock_model_manager()
        
        # Test model loading
        detection_model = mock_manager.load_detection_model("test_path")
        assert detection_model is not None
        
        recognition_model = mock_manager.load_recognition_model("test_path")
        assert recognition_model is not None
        
        # Test version switching
        mock_manager.switch_model_version("2.0.0")
        mock_manager.switch_model_version.assert_called_with("2.0.0")
        
        # Test metadata retrieval
        metadata = mock_manager.get_model_metadata()
        assert metadata.model_name == "Test_Model"
        assert metadata.version == "1.0.0"
    
    def test_create_mock_image_processor(self):
        """Test creating mock image processor."""
        mock_processor = MockModelObjects.create_mock_image_processor()
        
        # Test preprocessing
        processed_result = mock_processor.preprocess(Mock())
        assert processed_result.original_size == (1024, 768)
        assert processed_result.processed_size == (640, 480)
        assert processed_result.enhancement_applied is True
        
        # Test enhancement methods
        enhanced_image = mock_processor.enhance_text_visibility(Mock())
        assert isinstance(enhanced_image, np.ndarray)
        
        normalized_image = mock_processor.normalize_resolution(Mock())
        assert isinstance(normalized_image, np.ndarray)
        
        corrected_image = mock_processor.correct_orientation(Mock())
        assert isinstance(corrected_image, np.ndarray)


class TestTestDataFactory:
    """Test cases for TestDataFactory utility class."""
    
    def test_create_sample_detections(self):
        """Test creating sample detections."""
        detections = TestDataFactory.create_sample_detections(count=3)
        assert len(detections) == 3
        
        for i, detection in enumerate(detections):
            assert isinstance(detection, Detection)
            assert detection.bbox.x == i * 50 + 10
            assert detection.detection_type in ['word', 'phrase']
            assert 0.8 <= detection.confidence <= 1.0
    
    def test_create_sample_recognized_texts(self):
        """Test creating sample recognized texts."""
        detections = TestDataFactory.create_sample_detections(count=3)
        recognized_texts = TestDataFactory.create_sample_recognized_texts(detections)
        
        assert len(recognized_texts) == len(detections)
        for recognized_text in recognized_texts:
            assert isinstance(recognized_text, RecognizedText)
            assert recognized_text.language == "en"
            assert len(recognized_text.character_confidences) == len(recognized_text.text)
    
    def test_create_sample_linked_texts(self):
        """Test creating sample linked texts."""
        detections = TestDataFactory.create_sample_detections(count=2)
        recognized_texts = TestDataFactory.create_sample_recognized_texts(detections)
        linked_texts = TestDataFactory.create_sample_linked_texts(recognized_texts)
        
        assert len(linked_texts) == len(recognized_texts)
        for linked_text in linked_texts:
            assert isinstance(linked_text, LinkedText)
            assert len(linked_text.map_coordinates) == 2
            assert 0.0 <= linked_text.spatial_confidence <= 1.0
            assert "city" in linked_text.related_features
    
    def test_create_sample_processing_result(self):
        """Test creating sample processing result."""
        result = TestDataFactory.create_sample_processing_result()
        
        assert isinstance(result, ProcessingResult)
        assert result.task_type == 'word_detection_recognition'
        assert len(result.detections) == 3
        assert len(result.recognized_texts) == 3
        assert len(result.linked_texts) == 3
        assert result.processing_time == 2.5
        assert 'precision' in result.metrics
    
    def test_create_multilingual_test_data(self):
        """Test creating multilingual test data."""
        multilingual_data = TestDataFactory.create_multilingual_test_data()
        
        assert len(multilingual_data) == 4
        
        # Check language distribution
        languages = [text.language for text in multilingual_data]
        assert 'en' in languages
        assert 'zh' in languages
        
        # Check text content
        texts = [text.text for text in multilingual_data]
        assert any('Paris' in text or 'London' in text for text in texts)
        assert any('北京' in text or '上海' in text for text in texts)


class TestValidationUtilities:
    """Test cases for ValidationTestUtilities."""
    
    def test_confidence_validation_utility(self):
        """Test confidence validation utility function."""
        def sample_validation_func(value):
            return 0.0 <= value <= 1.0
        
        # This should not raise any exceptions
        ValidationTestUtilities.test_confidence_validation(sample_validation_func)
        
        # Test with custom range
        def custom_validation_func(value):
            return 0.5 <= value <= 0.9
        
        ValidationTestUtilities.test_confidence_validation(
            custom_validation_func, 
            valid_range=(0.5, 0.9)
        )
    
    def test_coordinate_validation_utility(self):
        """Test coordinate validation utility function."""
        def sample_coordinate_validation(coords):
            x1, y1, x2, y2 = coords
            return x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1
        
        # This should not raise any exceptions
        ValidationTestUtilities.test_coordinate_validation(sample_coordinate_validation)
    
    def test_size_validation_utility(self):
        """Test size validation utility function."""
        def sample_size_validation(size):
            width, height = size
            return width > 0 and height > 0
        
        # This should not raise any exceptions
        ValidationTestUtilities.test_size_validation(sample_size_validation)


class TestPerformanceUtilities:
    """Test cases for PerformanceTestUtilities."""
    
    def test_measure_execution_time(self):
        """Test execution time measurement."""
        def sample_function(n):
            return sum(range(n))
        
        result, exec_time = PerformanceTestUtilities.measure_execution_time(
            sample_function, 1000
        )
        
        assert result == sum(range(1000))
        assert exec_time >= 0
        assert isinstance(exec_time, float)
    
    def test_measure_memory_usage(self):
        """Test memory usage measurement."""
        def sample_function():
            # Create a list to use some memory
            return list(range(1000))
        
        result, memory_used = PerformanceTestUtilities.measure_memory_usage(sample_function)
        
        assert len(result) == 1000
        assert isinstance(memory_used, float)
        assert memory_used >= 0  # Memory usage should be non-negative
    
    def test_create_performance_benchmark(self):
        """Test performance benchmark creation."""
        def sample_function(n, multiplier=1):
            return sum(range(n)) * multiplier
        
        test_cases = [
            (100, 1),
            (500, 2),
            (1000, 1)
        ]
        
        benchmark_results = PerformanceTestUtilities.create_performance_benchmark(
            sample_function, test_cases, iterations=3
        )
        
        assert len(benchmark_results) == 3
        for i in range(3):
            test_case_key = f'test_case_{i}'
            assert test_case_key in benchmark_results
            assert 'avg_time' in benchmark_results[test_case_key]
            assert 'min_time' in benchmark_results[test_case_key]
            assert 'max_time' in benchmark_results[test_case_key]
            
            # Verify time relationships
            times = benchmark_results[test_case_key]
            assert times['min_time'] <= times['avg_time'] <= times['max_time']


class TestMockIntegration:
    """Integration tests for mock objects working together."""
    
    def test_complete_pipeline_with_mocks(self):
        """Test complete pipeline using mock objects."""
        # Create mock components
        mock_detector = MockModelObjects.create_mock_detector()
        mock_recognizer = MockModelObjects.create_mock_recognizer()
        mock_linker = MockModelObjects.create_mock_linker()
        mock_processor = MockModelObjects.create_mock_image_processor()
        
        # Create test image
        test_image = MockImageGenerator.create_historical_map_image()
        
        # Simulate pipeline execution
        processed_image = mock_processor.preprocess(test_image)
        detections = mock_detector.detect_words(processed_image)
        recognized_texts = mock_recognizer.recognize_words(detections)
        linked_texts = mock_linker.link_to_coordinates(recognized_texts)
        
        # Verify pipeline results
        assert processed_image is not None
        assert len(detections) > 0
        assert len(recognized_texts) > 0
        assert len(linked_texts) > 0
        
        # Verify data flow consistency
        assert isinstance(detections[0], Detection)
        assert isinstance(recognized_texts[0], RecognizedText)
        assert isinstance(linked_texts[0], LinkedText)
    
    def test_error_handling_with_mocks(self):
        """Test error handling scenarios with mock objects."""
        # Create mock that raises exceptions
        mock_detector = Mock()
        mock_detector.detect_words.side_effect = RuntimeError("Model loading failed")
        
        # Test exception handling
        with pytest.raises(RuntimeError, match="Model loading failed"):
            mock_detector.detect_words(Mock())
    
    def test_mock_configuration_flexibility(self):
        """Test that mocks can be reconfigured for different test scenarios."""
        mock_recognizer = MockModelObjects.create_mock_recognizer()
        
        # Reconfigure for different language
        mock_recognizer.recognize_words.return_value = [
            RecognizedText(
                detection=Detection(
                    bbox=BoundingBox(x=10, y=20, width=100, height=30, confidence=0.9),
                    detection_type='word',
                    confidence=0.9,
                    image_coordinates=(10, 20, 110, 50)
                ),
                text="北京",
                language="zh",
                recognition_confidence=0.92,
                character_confidences=[0.9, 0.94]
            )
        ]
        
        # Test reconfigured behavior
        results = mock_recognizer.recognize_words([])
        assert results[0].text == "北京"
        assert results[0].language == "zh"
    
    def test_mock_call_verification(self):
        """Test that mock calls can be verified for testing."""
        mock_manager = MockModelObjects.create_mock_model_manager()
        
        # Make some calls
        mock_manager.load_detection_model("path/to/model")
        mock_manager.switch_model_version("2.0.0")
        
        # Verify calls were made
        mock_manager.load_detection_model.assert_called_with("path/to/model")
        mock_manager.switch_model_version.assert_called_with("2.0.0")
        
        # Verify call counts
        assert mock_manager.load_detection_model.call_count == 1
        assert mock_manager.switch_model_version.call_count == 1


class TestDataValidationIntegration:
    """Integration tests for data validation with test utilities."""
    
    def test_generated_data_passes_validation(self):
        """Test that generated test data passes validation."""
        # Generate test data
        detections = TestDataFactory.create_sample_detections(5)
        recognized_texts = TestDataFactory.create_sample_recognized_texts(detections)
        linked_texts = TestDataFactory.create_sample_linked_texts(recognized_texts)
        
        # Validate all generated data
        for detection in detections:
            assert detection.bbox.confidence >= 0.0
            assert detection.bbox.confidence <= 1.0
            assert detection.detection_type in ['word', 'phrase']
        
        for recognized_text in recognized_texts:
            assert recognized_text.recognition_confidence >= 0.0
            assert recognized_text.recognition_confidence <= 1.0
            assert len(recognized_text.character_confidences) == len(recognized_text.text)
        
        for linked_text in linked_texts:
            assert linked_text.spatial_confidence >= 0.0
            assert linked_text.spatial_confidence <= 1.0
            assert len(linked_text.map_coordinates) == 2
    
    def test_image_generation_produces_valid_arrays(self):
        """Test that image generation produces valid numpy arrays."""
        # Test different image types
        test_cases = [
            (640, 480, 3),
            (800, 600, 1),
            (1024, 768, 3)
        ]
        
        for width, height, channels in test_cases:
            image = MockImageGenerator.create_test_image(width, height, channels)
            
            # Validate image properties
            if channels == 1:
                assert image.shape == (height, width)
            else:
                assert image.shape == (height, width, channels)
            
            assert image.dtype == np.uint8
            assert np.all(image >= 0)
            assert np.all(image <= 255)
    
    def test_performance_measurement_accuracy(self):
        """Test that performance measurements are reasonably accurate."""
        def known_duration_function():
            import time
            time.sleep(0.1)  # Sleep for 100ms
            return "completed"
        
        result, exec_time = PerformanceTestUtilities.measure_execution_time(
            known_duration_function
        )
        
        assert result == "completed"
        # Allow some tolerance for timing accuracy (90ms to 200ms)
        assert 0.09 <= exec_time <= 0.2