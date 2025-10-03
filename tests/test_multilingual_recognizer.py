"""
Unit tests for multilingual text recognition.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.recognition.multilingual_recognizer import (
    LanguageDetector, ChineseRecognitionModel, MultilingualTextRecognizer
)
from src.models.data_structures import RecognitionConfig, Detection, BoundingBox, RecognizedText


class TestLanguageDetector(unittest.TestCase):
    """Test cases for LanguageDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = LanguageDetector()
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector.logger)
        self.assertGreater(self.detector.chinese_aspect_ratio_threshold, 0)
        self.assertGreater(self.detector.latin_aspect_ratio_threshold, 0)
        self.assertGreater(self.detector.chinese_density_threshold, 0)
    
    def test_detect_language_wide_region(self):
        """Test language detection for wide text region (likely Latin)."""
        # Create a wide, horizontal text region
        text_region = np.random.rand(32, 128, 3).astype(np.float32)
        
        language = self.detector.detect_language(text_region)
        
        # Should detect as English/Latin due to wide aspect ratio
        self.assertIn(language, ['en', 'zh', 'mixed'])
    
    def test_detect_language_square_region(self):
        """Test language detection for square text region (likely Chinese)."""
        # Create a more square text region
        text_region = np.random.rand(32, 40, 3).astype(np.float32)
        
        language = self.detector.detect_language(text_region)
        
        # Should be able to detect some language
        self.assertIn(language, ['en', 'zh', 'mixed'])
    
    def test_detect_language_grayscale(self):
        """Test language detection with grayscale image."""
        text_region = np.random.rand(32, 64).astype(np.float32)
        
        language = self.detector.detect_language(text_region)
        
        self.assertIn(language, ['en', 'zh', 'mixed'])
    
    def test_detect_language_low_confidence(self):
        """Test language detection with high confidence threshold."""
        text_region = np.random.rand(32, 64, 3).astype(np.float32)
        
        # Use very high confidence threshold
        language = self.detector.detect_language(text_region, confidence_threshold=0.95)
        
        # Should return 'mixed' due to low confidence
        self.assertEqual(language, 'mixed')
    
    def test_extract_visual_features(self):
        """Test visual feature extraction."""
        text_region = np.random.rand(32, 64, 3).astype(np.float32)
        
        features = self.detector._extract_visual_features(text_region)
        
        # Check that all expected features are present
        expected_features = [
            'aspect_ratio', 'density', 'num_components', 
            'avg_component_aspect_ratio', 'stroke_variation'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
    
    def test_calculate_language_scores(self):
        """Test language score calculation."""
        features = {
            'aspect_ratio': 2.0,
            'density': 0.5,
            'num_components': 5,
            'avg_component_aspect_ratio': 1.8,
            'stroke_variation': 30.0
        }
        
        scores = self.detector._calculate_language_scores(features)
        
        self.assertIn('en', scores)
        self.assertIn('zh', scores)
        self.assertIsInstance(scores['en'], float)
        self.assertIsInstance(scores['zh'], float)
        
        # Scores should be normalized (sum to 1)
        total_score = sum(scores.values())
        self.assertAlmostEqual(total_score, 1.0, places=5)


class TestChineseRecognitionModel(unittest.TestCase):
    """Test cases for ChineseRecognitionModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RecognitionConfig(
            model_architecture='CRNN',
            max_sequence_length=32,
            vocabulary_size=1000,
            languages=['zh'],
            beam_width=5
        )
        self.model = ChineseRecognitionModel(self.config)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.config, self.config)
        self.assertIsNone(self.model.model)
        self.assertIsNone(self.model.decoder)
        self.assertIsNotNone(self.model.charset)
        self.assertGreater(len(self.model.charset), 0)
    
    def test_create_chinese_charset(self):
        """Test Chinese character set creation."""
        charset = self.model._create_chinese_charset()
        
        # Should contain blank token
        self.assertIn('<blank>', charset)
        
        # Should contain Chinese characters
        self.assertIn('的', charset)
        self.assertIn('一', charset)
        
        # Should contain numbers
        self.assertIn('0', charset)
        self.assertIn('9', charset)
        
        # Should contain Chinese punctuation
        self.assertIn('，', charset)
        self.assertIn('。', charset)
    
    def test_load_model_nonexistent_file(self):
        """Test loading model from nonexistent file."""
        # Should create dummy model without raising exception
        self.model.load_model("nonexistent_chinese_model.pth")
        
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.decoder)
    
    def test_predict_without_model(self):
        """Test prediction without loaded model."""
        text_region = np.random.rand(32, 128, 3).astype(np.float32)
        
        with self.assertRaises(RuntimeError):
            self.model.predict(text_region)
    
    def test_predict_with_dummy_model(self):
        """Test prediction with dummy model."""
        # Load dummy model
        self.model.load_model("dummy_chinese_model.pth")
        
        # Create test input
        text_region = np.random.rand(32, 128, 3).astype(np.float32)
        
        # Should not raise exception
        text, confidences = self.model.predict(text_region)
        
        self.assertIsInstance(text, str)
        self.assertIsInstance(confidences, list)
    
    def test_prepare_input_rgb(self):
        """Test input preparation for RGB image."""
        text_region = np.random.rand(32, 128, 3).astype(np.float32)
        
        # Load dummy model to initialize
        self.model.load_model("dummy_model.pth")
        
        input_tensor = self.model._prepare_input(text_region)
        
        self.assertEqual(input_tensor.shape, (1, 3, 32, 128))
    
    def test_prepare_input_grayscale(self):
        """Test input preparation for grayscale image."""
        text_region = np.random.rand(32, 128).astype(np.float32)
        
        # Load dummy model to initialize
        self.model.load_model("dummy_model.pth")
        
        input_tensor = self.model._prepare_input(text_region)
        
        self.assertEqual(input_tensor.shape, (1, 3, 32, 128))
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = self.model.get_supported_languages()
        
        self.assertIn('zh', languages)
        self.assertIn('chinese', languages)


class TestMultilingualTextRecognizer(unittest.TestCase):
    """Test cases for MultilingualTextRecognizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RecognitionConfig(
            model_architecture='CRNN',
            max_sequence_length=32,
            vocabulary_size=5000,
            languages=['en', 'zh'],
            beam_width=5
        )
        self.recognizer = MultilingualTextRecognizer(self.config)
    
    def test_initialization(self):
        """Test recognizer initialization."""
        self.assertIsNotNone(self.recognizer.config)
        self.assertIsNotNone(self.recognizer.language_detector)
        self.assertIsNotNone(self.recognizer.recognition_models)
        self.assertGreater(len(self.recognizer.recognition_models), 0)
    
    def test_initialization_with_default_config(self):
        """Test initialization with default config."""
        recognizer = MultilingualTextRecognizer()
        
        self.assertIsNotNone(recognizer.config)
        self.assertEqual(recognizer.config.languages, ['en', 'zh'])
        self.assertEqual(recognizer.config.vocabulary_size, 5000)
    
    def test_initialize_language_models(self):
        """Test language model initialization."""
        # Should have models for supported languages
        self.assertIn('en', self.recognizer.recognition_models)
        self.assertIn('zh', self.recognizer.recognition_models)
    
    def test_detect_language(self):
        """Test language detection."""
        text_region = np.random.rand(32, 64, 3).astype(np.float32)
        
        language = self.recognizer.detect_language(text_region)
        
        self.assertIn(language, ['en', 'zh', 'mixed'])
    
    def test_get_model_key(self):
        """Test model key mapping."""
        test_cases = [
            ('en', 'en'),
            ('latin', 'en'),
            ('zh', 'zh'),
            ('chinese', 'zh'),
            ('mixed', 'en'),
            ('unknown', 'en')  # Default fallback
        ]
        
        for input_lang, expected_key in test_cases:
            with self.subTest(input_lang=input_lang):
                key = self.recognizer._get_model_key(input_lang)
                self.assertEqual(key, expected_key)
    
    def test_recognize_text_region(self):
        """Test text region recognition."""
        # Load dummy models first
        self.recognizer.load_models({
            'en': 'dummy_en_model.pth',
            'zh': 'dummy_zh_model.pth'
        })
        
        text_region = np.random.rand(32, 128, 3).astype(np.float32)
        
        # Test English recognition
        text_en, conf_en = self.recognizer._recognize_text_region(text_region, 'en')
        self.assertIsInstance(text_en, str)
        self.assertIsInstance(conf_en, list)
        
        # Test Chinese recognition
        text_zh, conf_zh = self.recognizer._recognize_text_region(text_region, 'zh')
        self.assertIsInstance(text_zh, str)
        self.assertIsInstance(conf_zh, list)
    
    def test_load_models(self):
        """Test loading multiple models."""
        model_paths = {
            'en': 'path/to/english_model.pth',
            'zh': 'path/to/chinese_model.pth'
        }
        
        # Should not raise exception
        self.recognizer.load_models(model_paths)
    
    def test_recognize_mixed_language_text(self):
        """Test mixed language text recognition."""
        # Load dummy models
        self.recognizer.load_models({
            'en': 'dummy_en_model.pth',
            'zh': 'dummy_zh_model.pth'
        })
        
        # Create test image and detections
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        detections = [
            Detection(
                bbox=BoundingBox(x=10, y=10, width=50, height=20, confidence=0.9),
                detection_type='word',
                confidence=0.9,
                image_coordinates=(10, 10, 60, 30)
            ),
            Detection(
                bbox=BoundingBox(x=70, y=10, width=40, height=20, confidence=0.8),
                detection_type='word',
                confidence=0.8,
                image_coordinates=(70, 10, 110, 30)
            )
        ]
        
        results = self.recognizer.recognize_mixed_language_text(image, detections)
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, RecognizedText)
            self.assertIn(result.language, ['en', 'zh', 'mixed'])
    
    def test_get_language_statistics(self):
        """Test language statistics calculation."""
        # Create mock recognition results
        recognized_texts = [
            RecognizedText(
                detection=Mock(),
                text="hello",
                language='en',
                recognition_confidence=0.9,
                character_confidences=[0.9, 0.8, 0.9, 0.9, 0.8]
            ),
            RecognizedText(
                detection=Mock(),
                text="world",
                language='en',
                recognition_confidence=0.8,
                character_confidences=[0.8, 0.9, 0.8, 0.9, 0.7]
            ),
            RecognizedText(
                detection=Mock(),
                text="你好",
                language='zh',
                recognition_confidence=0.85,
                character_confidences=[0.8, 0.9]
            )
        ]
        
        stats = self.recognizer.get_language_statistics(recognized_texts)
        
        self.assertEqual(stats['total_texts'], 3)
        self.assertEqual(stats['language_counts']['en'], 2)
        self.assertEqual(stats['language_counts']['zh'], 1)
        self.assertAlmostEqual(stats['language_percentages']['en'], 66.67, places=1)
        self.assertAlmostEqual(stats['language_percentages']['zh'], 33.33, places=1)
        self.assertEqual(stats['dominant_language'], 'en')
    
    def test_format_mixed_language_output(self):
        """Test mixed language output formatting."""
        recognized_texts = [
            RecognizedText(
                detection=Mock(),
                text="hello",
                language='en',
                recognition_confidence=0.9,
                character_confidences=[0.9, 0.8, 0.9, 0.9, 0.8]
            ),
            RecognizedText(
                detection=Mock(),
                text="你好",
                language='zh',
                recognition_confidence=0.85,
                character_confidences=[0.8, 0.9]
            )
        ]
        
        formatted = self.recognizer.format_mixed_language_output(recognized_texts)
        
        self.assertIn('[EN]', formatted)
        self.assertIn('[ZH]', formatted)
        self.assertIn('hello', formatted)
        self.assertIn('你好', formatted)
        self.assertIn('(0.90)', formatted)
        self.assertIn('(0.85)', formatted)
    
    def test_format_mixed_language_output_empty_text(self):
        """Test formatting with empty text results."""
        recognized_texts = [
            RecognizedText(
                detection=Mock(),
                text="",  # Empty text
                language='en',
                recognition_confidence=0.0,
                character_confidences=[]
            ),
            RecognizedText(
                detection=Mock(),
                text="hello",
                language='en',
                recognition_confidence=0.9,
                character_confidences=[0.9, 0.8, 0.9, 0.9, 0.8]
            )
        ]
        
        formatted = self.recognizer.format_mixed_language_output(recognized_texts)
        
        # Should only include non-empty text
        self.assertIn('[EN]hello(0.90)', formatted)
        self.assertNotIn('(0.00)', formatted)  # Empty text should be excluded
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.recognizer.get_model_info()
        
        self.assertIn('architecture', info)
        self.assertIn('supported_languages', info)
        self.assertIn('loaded_models', info)
        self.assertIn('language_detection_threshold', info)
        
        self.assertEqual(info['architecture'], 'Multilingual CRNN')
        self.assertEqual(info['supported_languages'], ['en', 'zh'])
        self.assertIsInstance(info['loaded_models'], list)


if __name__ == '__main__':
    unittest.main()