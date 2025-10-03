"""
Unit tests for base text recognition components.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
from PIL import Image

from src.recognition.base_recognizer import (
    BaseTextRecognizer, TextPreprocessor, CharacterConfidenceTracker
)
from src.models.data_structures import (
    Detection, BoundingBox, RecognitionConfig, RecognizedText
)


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RecognitionConfig(
            model_architecture='CRNN',
            max_sequence_length=32,
            vocabulary_size=1000,
            languages=['en'],
            beam_width=5
        )
        self.preprocessor = TextPreprocessor(self.config)
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        self.assertIsNotNone(self.preprocessor.config)
        self.assertEqual(self.preprocessor.config.model_architecture, 'CRNN')
    
    def test_initialization_with_default_config(self):
        """Test preprocessor initialization with default config."""
        preprocessor = TextPreprocessor()
        self.assertIsNotNone(preprocessor.config)
        self.assertEqual(preprocessor.config.languages, ['en'])
    
    def test_preprocess_text_region(self):
        """Test text region preprocessing."""
        # Create test image and detection
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        bbox = BoundingBox(x=10, y=10, width=50, height=20, confidence=0.9)
        detection = Detection(
            bbox=bbox,
            detection_type='word',
            confidence=0.9,
            image_coordinates=(10, 10, 60, 30)
        )
        
        # Preprocess region
        processed = self.preprocessor.preprocess_text_region(image, detection)
        
        # Check output shape and type
        self.assertEqual(len(processed.shape), 3)
        self.assertEqual(processed.shape[0], 32)  # Target height
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(np.all(processed >= 0.0))
        self.assertTrue(np.all(processed <= 1.0))
    
    def test_preprocess_empty_region(self):
        """Test preprocessing with empty region."""
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        bbox = BoundingBox(x=10, y=10, width=1, height=1, confidence=0.9)  # Minimal valid size
        detection = Detection(
            bbox=bbox,
            detection_type='word',
            confidence=0.9,
            image_coordinates=(10, 10, 11, 11)  # Minimal region
        )
        
        processed = self.preprocessor.preprocess_text_region(image, detection)
        
        # Should return processed region with target dimensions
        self.assertEqual(processed.shape[0], 32)  # Target height
        self.assertEqual(len(processed.shape), 3)  # Should have 3 channels
    
    def test_normalize_region(self):
        """Test region normalization."""
        region = np.random.randint(0, 255, (32, 64, 3), dtype=np.uint8)
        normalized = self.preprocessor._normalize_region(region)
        
        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(np.all(normalized >= 0.0))
        self.assertTrue(np.all(normalized <= 1.0))
    
    def test_enhance_contrast(self):
        """Test contrast enhancement."""
        region = np.random.rand(32, 64, 3).astype(np.float32)
        enhanced = self.preprocessor._enhance_contrast(region)
        
        self.assertEqual(enhanced.dtype, np.float32)
        self.assertEqual(enhanced.shape, region.shape)
    
    def test_resize_region(self):
        """Test region resizing."""
        region = np.random.rand(16, 32, 3).astype(np.float32)
        resized = self.preprocessor._resize_region(region, target_height=32)
        
        self.assertEqual(resized.shape[0], 32)
        self.assertEqual(len(resized.shape), 3)
    
    def test_apply_padding(self):
        """Test padding application."""
        region = np.random.rand(32, 64, 3).astype(np.float32)
        padded = self.preprocessor._apply_padding(region, target_width=128)
        
        self.assertEqual(padded.shape[1], 128)
        self.assertEqual(padded.shape[0], 32)


class TestCharacterConfidenceTracker(unittest.TestCase):
    """Test cases for CharacterConfidenceTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = CharacterConfidenceTracker()
    
    def test_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(len(self.tracker.character_scores), 0)
        self.assertEqual(len(self.tracker.sequence_scores), 0)
    
    def test_track_character_confidence(self):
        """Test tracking single character confidence."""
        self.tracker.track_character_confidence('A', 0.95)
        self.tracker.track_character_confidence('B', 0.87)
        
        self.assertEqual(len(self.tracker.character_scores), 2)
        self.assertEqual(self.tracker.character_scores[0], 0.95)
        self.assertEqual(self.tracker.character_scores[1], 0.87)
    
    def test_track_character_confidence_invalid_score(self):
        """Test tracking with invalid confidence score."""
        self.tracker.track_character_confidence('A', 1.5)  # Invalid score
        
        # Should be clamped to valid range
        self.assertEqual(self.tracker.character_scores[0], 1.0)
    
    def test_track_sequence_confidence(self):
        """Test tracking sequence confidence."""
        sequence = "HELLO"
        confidences = [0.9, 0.8, 0.7, 0.85, 0.92]
        
        self.tracker.track_sequence_confidence(sequence, confidences)
        
        self.assertEqual(len(self.tracker.sequence_scores), 1)
        self.assertEqual(len(self.tracker.character_scores), 5)
        self.assertEqual(self.tracker.character_scores, confidences)
    
    def test_track_sequence_confidence_length_mismatch(self):
        """Test tracking sequence with mismatched lengths."""
        sequence = "HELLO"
        confidences = [0.9, 0.8, 0.7]  # Too short
        
        self.tracker.track_sequence_confidence(sequence, confidences)
        
        # Should pad with default confidence
        self.assertEqual(len(self.tracker.character_scores), 5)
        self.assertEqual(self.tracker.character_scores[-2:], [0.5, 0.5])
    
    def test_get_average_confidence(self):
        """Test average confidence calculation."""
        confidences = [0.9, 0.8, 0.7, 0.85, 0.92]
        for i, conf in enumerate(confidences):
            self.tracker.track_character_confidence(str(i), conf)
        
        expected_avg = sum(confidences) / len(confidences)
        self.assertAlmostEqual(self.tracker.get_average_confidence(), expected_avg)
    
    def test_get_average_confidence_empty(self):
        """Test average confidence with no data."""
        self.assertEqual(self.tracker.get_average_confidence(), 0.0)
    
    def test_get_min_confidence(self):
        """Test minimum confidence calculation."""
        confidences = [0.9, 0.8, 0.7, 0.85, 0.92]
        for i, conf in enumerate(confidences):
            self.tracker.track_character_confidence(str(i), conf)
        
        self.assertEqual(self.tracker.get_min_confidence(), 0.7)
    
    def test_get_confidence_distribution(self):
        """Test confidence distribution statistics."""
        confidences = [0.9, 0.8, 0.7, 0.85, 0.92]
        for i, conf in enumerate(confidences):
            self.tracker.track_character_confidence(str(i), conf)
        
        distribution = self.tracker.get_confidence_distribution()
        
        self.assertIn('mean', distribution)
        self.assertIn('std', distribution)
        self.assertIn('min', distribution)
        self.assertIn('max', distribution)
        self.assertEqual(distribution['min'], 0.7)
        self.assertEqual(distribution['max'], 0.92)
    
    def test_reset(self):
        """Test tracker reset."""
        self.tracker.track_character_confidence('A', 0.9)
        self.tracker.track_sequence_confidence("BC", [0.8, 0.7])
        
        self.tracker.reset()
        
        self.assertEqual(len(self.tracker.character_scores), 0)
        self.assertEqual(len(self.tracker.sequence_scores), 0)


class MockRecognitionModel:
    """Mock recognition model for testing."""
    
    def predict(self, text_region):
        """Mock prediction method."""
        return "MOCK_TEXT", [0.9, 0.8, 0.7, 0.85, 0.92, 0.88, 0.91, 0.89, 0.93]


class ConcreteTextRecognizer(BaseTextRecognizer):
    """Concrete implementation for testing."""
    
    def _recognize_text_region(self, text_region, language):
        """Mock implementation of text recognition."""
        # Return mock text and confidences
        mock_text = "SAMPLE"
        mock_confidences = [0.9, 0.8, 0.85, 0.92, 0.88, 0.91]
        return mock_text, mock_confidences


class TestBaseTextRecognizer(unittest.TestCase):
    """Test cases for BaseTextRecognizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RecognitionConfig(
            model_architecture='CRNN',
            max_sequence_length=32,
            vocabulary_size=1000,
            languages=['en', 'zh'],
            beam_width=5
        )
        self.recognizer = ConcreteTextRecognizer(self.config)
    
    def test_initialization(self):
        """Test recognizer initialization."""
        self.assertIsNotNone(self.recognizer.config)
        self.assertIsNotNone(self.recognizer.preprocessor)
        self.assertIsNotNone(self.recognizer.confidence_tracker)
        self.assertEqual(self.recognizer.current_language, 'en')
    
    def test_initialization_with_default_config(self):
        """Test recognizer initialization with default config."""
        recognizer = ConcreteTextRecognizer()
        self.assertIsNotNone(recognizer.config)
        self.assertEqual(recognizer.config.languages, ['en'])
    
    def test_recognize_words(self):
        """Test word recognition."""
        # Create test image and detections
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        bbox = BoundingBox(x=10, y=10, width=50, height=20, confidence=0.9)
        detection = Detection(
            bbox=bbox,
            detection_type='word',
            confidence=0.9,
            image_coordinates=(10, 10, 60, 30)
        )
        
        results = self.recognizer.recognize_words(image, [detection])
        
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RecognizedText)
        self.assertEqual(results[0].text, "SAMPLE")
        self.assertEqual(len(results[0].character_confidences), 6)
    
    def test_recognize_phrases(self):
        """Test phrase recognition."""
        # Create test image and detections
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        bbox = BoundingBox(x=10, y=10, width=100, height=20, confidence=0.9)
        detection = Detection(
            bbox=bbox,
            detection_type='phrase',
            confidence=0.9,
            image_coordinates=(10, 10, 110, 30)
        )
        
        results = self.recognizer.recognize_phrases(image, [detection])
        
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RecognizedText)
        self.assertEqual(results[0].text, "SAMPLE")
    
    def test_recognize_wrong_detection_type(self):
        """Test recognition with wrong detection type."""
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        bbox = BoundingBox(x=10, y=10, width=50, height=20, confidence=0.9)
        detection = Detection(
            bbox=bbox,
            detection_type='phrase',  # Wrong type for word recognition
            confidence=0.9,
            image_coordinates=(10, 10, 60, 30)
        )
        
        results = self.recognizer.recognize_words(image, [detection])
        
        # Should still process but log warning
        self.assertEqual(len(results), 1)
    
    def test_handle_multilingual(self):
        """Test multilingual text handling."""
        text_region = np.random.rand(32, 64, 3).astype(np.float32)
        
        result = self.recognizer.handle_multilingual(text_region)
        
        self.assertEqual(result, "SAMPLE")
    
    def test_detect_language_english(self):
        """Test language detection for English-like text."""
        # Wide region (typical for Latin text)
        text_region = np.random.rand(32, 128, 3).astype(np.float32)
        
        language = self.recognizer.detect_language(text_region)
        
        self.assertEqual(language, 'en')
    
    def test_detect_language_chinese(self):
        """Test language detection for Chinese-like text."""
        # Square region (typical for Chinese characters)
        text_region = np.random.rand(32, 32, 3).astype(np.float32)
        
        language = self.recognizer.detect_language(text_region)
        
        self.assertEqual(language, 'zh')
    
    def test_set_current_language(self):
        """Test setting current language."""
        self.recognizer.set_current_language('zh')
        self.assertEqual(self.recognizer.current_language, 'zh')
    
    def test_set_invalid_language(self):
        """Test setting invalid language."""
        with self.assertRaises(ValueError):
            self.recognizer.set_current_language('fr')  # Not supported
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = self.recognizer.get_supported_languages()
        self.assertEqual(languages, ['en', 'zh'])
    
    def test_get_preprocessing_config(self):
        """Test getting preprocessing configuration."""
        config = self.recognizer.get_preprocessing_config()
        
        self.assertIn('target_height', config)
        self.assertIn('target_width', config)
        self.assertEqual(config['target_height'], 32)
        self.assertEqual(config['target_width'], 128)
    
    def test_update_preprocessing_config(self):
        """Test updating preprocessing configuration."""
        # This should not raise an error
        self.recognizer.update_preprocessing_config(target_height=64)


if __name__ == '__main__':
    unittest.main()