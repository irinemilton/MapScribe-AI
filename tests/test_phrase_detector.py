"""
Unit tests for phrase-level text detection.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from src.models.data_structures import (
    ProcessedImage, BoundingBox, Detection, DetectionConfig
)
from src.detection.phrase_detector import PhraseDetector, PhraseGroupingConfig


class TestPhraseDetector(unittest.TestCase):
    """Test cases for PhraseDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DetectionConfig(
            model_architecture='YOLO',
            input_size=(640, 640),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            batch_size=1
        )
        
        self.phrase_config = PhraseGroupingConfig(
            line_overlap_threshold=0.5,
            horizontal_gap_ratio=0.8,
            min_words_per_phrase=1,
            max_words_per_phrase=5,
            confidence_weight=0.3
        )
        
        self.detector = PhraseDetector(self.config, self.phrase_config)
        
        # Create test image
        self.test_image = ProcessedImage(
            image=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
            original_size=(480, 640),
            processed_size=(640, 640),
            preprocessing_steps=['resize'],
            enhancement_applied=False
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
        """Test phrase detector initialization."""
        self.assertEqual(self.detector.config, self.config)
        self.assertEqual(self.detector.phrase_config, self.phrase_config)
        self.assertIsNone(self.detector.word_detector)
    
    def test_load_word_detector(self):
        """Test loading word detector."""
        self.detector.load_word_detector(self.temp_model_file.name)
        self.assertIsNotNone(self.detector.word_detector)
    
    def test_detect_phrases_without_word_detector(self):
        """Test phrase detection without loaded word detector."""
        with self.assertRaises(RuntimeError):
            self.detector.detect_phrases(self.test_image)
    
    def test_words_on_same_line(self):
        """Test line detection algorithm."""
        # Words on same line (significant vertical overlap)
        word1 = Detection(
            bbox=BoundingBox(x=10, y=10, width=30, height=20, confidence=0.8),
            detection_type='word',
            confidence=0.8,
            image_coordinates=(10, 10, 40, 30)
        )
        
        word2 = Detection(
            bbox=BoundingBox(x=50, y=15, width=25, height=18, confidence=0.7),
            detection_type='word',
            confidence=0.7,
            image_coordinates=(50, 15, 75, 33)
        )
        
        self.assertTrue(self.detector._words_on_same_line(word1, word2))
        
        # Words on different lines (no vertical overlap)
        word3 = Detection(
            bbox=BoundingBox(x=10, y=50, width=30, height=20, confidence=0.8),
            detection_type='word',
            confidence=0.8,
            image_coordinates=(10, 50, 40, 70)
        )
        
        self.assertFalse(self.detector._words_on_same_line(word1, word3))
    
    def test_should_group_words(self):
        """Test word grouping decision algorithm."""
        word1 = Detection(
            bbox=BoundingBox(x=10, y=10, width=30, height=20, confidence=0.8),
            detection_type='word',
            confidence=0.8,
            image_coordinates=(10, 10, 40, 30)
        )
        
        # Close word should be grouped
        word2_close = Detection(
            bbox=BoundingBox(x=45, y=10, width=25, height=20, confidence=0.7),
            detection_type='word',
            confidence=0.7,
            image_coordinates=(45, 10, 70, 30)
        )
        
        current_phrase = [word1]
        self.assertTrue(self.detector._should_group_words(word1, word2_close, current_phrase))
        
        # Distant word should not be grouped
        word2_far = Detection(
            bbox=BoundingBox(x=100, y=10, width=25, height=20, confidence=0.7),
            detection_type='word',
            confidence=0.7,
            image_coordinates=(100, 10, 125, 30)
        )
        
        self.assertFalse(self.detector._should_group_words(word1, word2_far, current_phrase))
    
    def test_group_words_by_lines(self):
        """Test grouping words into text lines."""
        # Create words on two different lines
        words = [
            # Line 1
            Detection(
                bbox=BoundingBox(x=10, y=10, width=30, height=20, confidence=0.8),
                detection_type='word', confidence=0.8, image_coordinates=(10, 10, 40, 30)
            ),
            Detection(
                bbox=BoundingBox(x=50, y=12, width=25, height=18, confidence=0.7),
                detection_type='word', confidence=0.7, image_coordinates=(50, 12, 75, 30)
            ),
            # Line 2
            Detection(
                bbox=BoundingBox(x=10, y=50, width=35, height=22, confidence=0.9),
                detection_type='word', confidence=0.9, image_coordinates=(10, 50, 45, 72)
            ),
            Detection(
                bbox=BoundingBox(x=55, y=52, width=28, height=20, confidence=0.6),
                detection_type='word', confidence=0.6, image_coordinates=(55, 52, 83, 72)
            )
        ]
        
        lines = self.detector._group_words_by_lines(words)
        
        self.assertEqual(len(lines), 2)  # Should have 2 lines
        self.assertEqual(len(lines[0]), 2)  # First line has 2 words
        self.assertEqual(len(lines[1]), 2)  # Second line has 2 words
        
        # Check that words in each line are sorted by x-coordinate
        for line in lines:
            x_coords = [word.bbox.x for word in line]
            self.assertEqual(x_coords, sorted(x_coords))
    
    def test_group_line_words_into_phrases(self):
        """Test grouping words within a line into phrases."""
        # Create words that should form 2 phrases
        line_words = [
            Detection(
                bbox=BoundingBox(x=10, y=10, width=20, height=15, confidence=0.8),
                detection_type='word', confidence=0.8, image_coordinates=(10, 10, 30, 25)
            ),
            Detection(
                bbox=BoundingBox(x=35, y=10, width=18, height=15, confidence=0.7),
                detection_type='word', confidence=0.7, image_coordinates=(35, 10, 53, 25)
            ),
            # Gap here - should start new phrase
            Detection(
                bbox=BoundingBox(x=100, y=10, width=22, height=15, confidence=0.9),
                detection_type='word', confidence=0.9, image_coordinates=(100, 10, 122, 25)
            ),
            Detection(
                bbox=BoundingBox(x=125, y=10, width=20, height=15, confidence=0.6),
                detection_type='word', confidence=0.6, image_coordinates=(125, 10, 145, 25)
            )
        ]
        
        phrases = self.detector._group_line_words_into_phrases(line_words)
        
        self.assertEqual(len(phrases), 2)  # Should have 2 phrases
        self.assertEqual(len(phrases[0]), 2)  # First phrase has 2 words
        self.assertEqual(len(phrases[1]), 2)  # Second phrase has 2 words
    
    def test_create_phrase_from_words(self):
        """Test creating phrase detection from word group."""
        words = [
            Detection(
                bbox=BoundingBox(x=10, y=10, width=20, height=15, confidence=0.8),
                detection_type='word', confidence=0.8, image_coordinates=(10, 10, 30, 25)
            ),
            Detection(
                bbox=BoundingBox(x=35, y=12, width=18, height=13, confidence=0.7),
                detection_type='word', confidence=0.7, image_coordinates=(35, 12, 53, 25)
            )
        ]
        
        phrase = self.detector._create_phrase_from_words(words)
        
        self.assertEqual(phrase.detection_type, 'phrase')
        self.assertEqual(phrase.bbox.x, 10)  # Leftmost x
        self.assertEqual(phrase.bbox.y, 10)  # Topmost y
        self.assertEqual(phrase.bbox.width, 43)  # 53 - 10
        self.assertEqual(phrase.bbox.height, 15)  # 25 - 10
        
        # Confidence should be weighted average
        expected_conf = (0.8 * 300 + 0.7 * 234) / (300 + 234)  # area-weighted
        self.assertAlmostEqual(phrase.confidence, expected_conf, places=2)
    
    def test_create_phrase_from_empty_words(self):
        """Test creating phrase from empty word list."""
        with self.assertRaises(ValueError):
            self.detector._create_phrase_from_words([])
    
    def test_get_phrase_statistics(self):
        """Test phrase statistics calculation."""
        phrases = [
            Detection(
                bbox=BoundingBox(x=10, y=10, width=50, height=20, confidence=0.8),
                detection_type='phrase', confidence=0.8, image_coordinates=(10, 10, 60, 30)
            ),
            Detection(
                bbox=BoundingBox(x=10, y=40, width=40, height=18, confidence=0.6),
                detection_type='phrase', confidence=0.6, image_coordinates=(10, 40, 50, 58)
            )
        ]
        
        stats = self.detector.get_phrase_statistics(phrases)
        
        self.assertEqual(stats['num_phrases'], 2)
        self.assertEqual(stats['avg_confidence'], 0.7)
        self.assertEqual(stats['avg_width'], 45.0)
        self.assertEqual(stats['avg_height'], 19.0)
        self.assertEqual(stats['min_confidence'], 0.6)
        self.assertEqual(stats['max_confidence'], 0.8)
    
    def test_get_phrase_statistics_empty(self):
        """Test phrase statistics with empty list."""
        stats = self.detector.get_phrase_statistics([])
        
        self.assertEqual(stats['num_phrases'], 0)
        self.assertEqual(stats['avg_confidence'], 0.0)
    
    def test_split_long_phrases(self):
        """Test splitting overly long phrases."""
        # Create a long phrase group (6 words, max is 5)
        long_group = []
        for i in range(6):
            word = Detection(
                bbox=BoundingBox(x=i*25, y=10, width=20, height=15, confidence=0.7),
                detection_type='word', confidence=0.7, 
                image_coordinates=(i*25, 10, i*25+20, 25)
            )
            long_group.append(word)
        
        chunks = self.detector._split_phrase_group(long_group)
        
        self.assertEqual(len(chunks), 2)  # Should split into 2 chunks
        self.assertEqual(len(chunks[0]), 5)  # First chunk has 5 words
        self.assertEqual(len(chunks[1]), 1)  # Second chunk has 1 word
    
    def test_merge_close_phrases(self):
        """Test merging phrases that are very close."""
        # Create two phrase groups that are very close
        phrase1 = [
            Detection(
                bbox=BoundingBox(x=10, y=10, width=20, height=15, confidence=0.8),
                detection_type='word', confidence=0.8, image_coordinates=(10, 10, 30, 25)
            )
        ]
        
        phrase2 = [
            Detection(
                bbox=BoundingBox(x=35, y=12, width=18, height=13, confidence=0.7),
                detection_type='word', confidence=0.7, image_coordinates=(35, 12, 53, 25)
            )
        ]
        
        merged = self.detector._merge_close_phrases([phrase1, phrase2])
        
        # Should merge into single phrase since they're close
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]), 2)


class TestPhraseGroupingConfig(unittest.TestCase):
    """Test cases for PhraseGroupingConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PhraseGroupingConfig()
        
        self.assertEqual(config.line_overlap_threshold, 0.5)
        self.assertEqual(config.horizontal_gap_ratio, 0.8)
        self.assertEqual(config.min_words_per_phrase, 1)
        self.assertEqual(config.max_words_per_phrase, 10)
        self.assertEqual(config.confidence_weight, 0.3)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PhraseGroupingConfig(
            line_overlap_threshold=0.6,
            horizontal_gap_ratio=1.0,
            min_words_per_phrase=2,
            max_words_per_phrase=8,
            confidence_weight=0.5
        )
        
        self.assertEqual(config.line_overlap_threshold, 0.6)
        self.assertEqual(config.horizontal_gap_ratio, 1.0)
        self.assertEqual(config.min_words_per_phrase, 2)
        self.assertEqual(config.max_words_per_phrase, 8)
        self.assertEqual(config.confidence_weight, 0.5)


class TestPhraseDetectionIntegration(unittest.TestCase):
    """Integration tests for complete phrase detection pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = DetectionConfig(
            model_architecture='YOLO',
            input_size=(640, 640),
            confidence_threshold=0.3,
            nms_threshold=0.4,
            batch_size=1
        )
        
        self.detector = PhraseDetector(self.config)
        
        # Create temporary model file
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
        self.temp_model_file.write(b'mock model data')
        self.temp_model_file.close()
        
        # Create test image
        self.test_image = ProcessedImage(
            image=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
            original_size=(480, 640),
            processed_size=(640, 640),
            preprocessing_steps=['resize', 'enhance'],
            enhancement_applied=True
        )
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        if os.path.exists(self.temp_model_file.name):
            os.unlink(self.temp_model_file.name)
    
    def test_end_to_end_phrase_detection(self):
        """Test complete phrase detection pipeline."""
        # Load word detector
        self.detector.load_word_detector(self.temp_model_file.name)
        
        # Run phrase detection
        phrases = self.detector.detect_phrases(self.test_image)
        
        # Verify results
        self.assertIsInstance(phrases, list)
        
        for phrase in phrases:
            self.assertEqual(phrase.detection_type, 'phrase')
            self.assertGreaterEqual(phrase.confidence, 0.0)
            self.assertLessEqual(phrase.confidence, 1.0)
            self.assertGreater(phrase.bbox.width, 0)
            self.assertGreater(phrase.bbox.height, 0)
    
    def test_phrase_detection_with_custom_config(self):
        """Test phrase detection with custom grouping configuration."""
        custom_config = PhraseGroupingConfig(
            line_overlap_threshold=0.6,
            horizontal_gap_ratio=0.5,  # Stricter grouping
            min_words_per_phrase=2,    # Require at least 2 words
            max_words_per_phrase=3,    # Limit phrase length
            confidence_weight=0.5
        )
        
        detector = PhraseDetector(self.config, custom_config)
        detector.load_word_detector(self.temp_model_file.name)
        
        phrases = detector.detect_phrases(self.test_image)
        
        # Verify that configuration is applied
        self.assertIsInstance(phrases, list)
        # With stricter grouping, might have different number of phrases


if __name__ == '__main__':
    unittest.main()