"""
Unit tests for recognition accuracy metrics.
"""

import unittest
from io import StringIO
import sys

from src.recognition.accuracy_metrics import RecognitionAccuracyMetrics


class TestRecognitionAccuracyMetrics(unittest.TestCase):
    """Test cases for RecognitionAccuracyMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = RecognitionAccuracyMetrics()
    
    def test_initialization(self):
        """Test metrics calculator initialization."""
        self.assertIsNotNone(self.metrics.logger)
    
    def test_calculate_character_accuracy_perfect_match(self):
        """Test character accuracy with perfect matches."""
        predicted = ["hello", "world", "test"]
        ground_truth = ["hello", "world", "test"]
        
        result = self.metrics.calculate_character_accuracy(predicted, ground_truth)
        
        self.assertEqual(result['character_accuracy'], 1.0)
        self.assertEqual(result['character_error_rate'], 0.0)
        self.assertEqual(result['correct_characters'], 14)  # 5+5+4 characters
        self.assertEqual(result['total_characters'], 14)
        self.assertEqual(result['average_edit_distance'], 0.0)
        self.assertEqual(result['normalized_edit_distance'], 0.0)
    
    def test_calculate_character_accuracy_partial_match(self):
        """Test character accuracy with partial matches."""
        predicted = ["helo", "wrld", "tst"]  # Missing some characters
        ground_truth = ["hello", "world", "test"]
        
        result = self.metrics.calculate_character_accuracy(predicted, ground_truth)
        
        # Should have some accuracy but not perfect
        self.assertLess(result['character_accuracy'], 1.0)
        self.assertGreater(result['character_accuracy'], 0.0)
        self.assertGreater(result['character_error_rate'], 0.0)
        self.assertGreater(result['average_edit_distance'], 0.0)
    
    def test_calculate_character_accuracy_no_match(self):
        """Test character accuracy with no matches."""
        predicted = ["xyz", "abc", "def"]
        ground_truth = ["hello", "world", "test"]
        
        result = self.metrics.calculate_character_accuracy(predicted, ground_truth)
        
        # There might be some accidental matches, so just check it's very low
        self.assertLess(result['character_accuracy'], 0.2)
        self.assertGreater(result['character_error_rate'], 0.8)
        self.assertLessEqual(result['correct_characters'], 1)  # At most 1 accidental match
    
    def test_calculate_character_accuracy_different_lengths(self):
        """Test character accuracy with different string lengths."""
        predicted = ["hi", "world!", "testing123"]
        ground_truth = ["hello", "world", "test"]
        
        result = self.metrics.calculate_character_accuracy(predicted, ground_truth)
        
        # Should handle different lengths gracefully
        self.assertIsInstance(result['character_accuracy'], float)
        self.assertGreaterEqual(result['character_accuracy'], 0.0)
        self.assertLessEqual(result['character_accuracy'], 1.0)
    
    def test_calculate_character_accuracy_empty_lists(self):
        """Test character accuracy with empty lists."""
        predicted = []
        ground_truth = []
        
        result = self.metrics.calculate_character_accuracy(predicted, ground_truth)
        
        self.assertEqual(result['character_accuracy'], 0.0)
        self.assertEqual(result['total_characters'], 0)
        self.assertEqual(result['correct_characters'], 0)
    
    def test_calculate_character_accuracy_mismatched_lengths(self):
        """Test character accuracy with mismatched list lengths."""
        predicted = ["hello", "world"]
        ground_truth = ["hello"]  # Different length
        
        with self.assertRaises(ValueError):
            self.metrics.calculate_character_accuracy(predicted, ground_truth)
    
    def test_calculate_word_accuracy_perfect_match(self):
        """Test word accuracy with perfect matches."""
        predicted = ["hello world", "test case", "perfect match"]
        ground_truth = ["hello world", "test case", "perfect match"]
        
        result = self.metrics.calculate_word_accuracy(predicted, ground_truth)
        
        self.assertEqual(result['word_accuracy'], 1.0)
        self.assertEqual(result['word_error_rate'], 0.0)
        self.assertEqual(result['exact_match_accuracy'], 1.0)
        self.assertEqual(result['correct_words'], 6)  # 2+2+2 words
        self.assertEqual(result['total_words'], 6)
    
    def test_calculate_word_accuracy_partial_match(self):
        """Test word accuracy with partial matches."""
        predicted = ["hello world", "test fail", "partial match"]
        ground_truth = ["hello world", "test case", "perfect match"]
        
        result = self.metrics.calculate_word_accuracy(predicted, ground_truth)
        
        # Should have some accuracy but not perfect
        self.assertLess(result['word_accuracy'], 1.0)
        self.assertGreater(result['word_accuracy'], 0.0)
        self.assertLess(result['exact_match_accuracy'], 1.0)
        self.assertEqual(result['correct_words'], 4)  # hello, world, test, match
    
    def test_calculate_word_accuracy_case_insensitive(self):
        """Test word accuracy is case insensitive."""
        predicted = ["Hello World", "TEST case"]
        ground_truth = ["hello world", "test CASE"]
        
        result = self.metrics.calculate_word_accuracy(predicted, ground_truth)
        
        self.assertEqual(result['exact_match_accuracy'], 1.0)
        self.assertEqual(result['word_accuracy'], 1.0)
    
    def test_calculate_sequence_accuracy_perfect_match(self):
        """Test sequence accuracy with perfect matches."""
        predicted = ["hello world", "test case"]
        ground_truth = ["hello world", "test case"]
        
        result = self.metrics.calculate_sequence_accuracy(predicted, ground_truth)
        
        self.assertEqual(result['sequence_similarity'], 1.0)
        self.assertEqual(result['sequence_error_rate'], 0.0)
        self.assertGreaterEqual(result['jaro_winkler_similarity'], 0.9)  # Should be high
    
    def test_calculate_sequence_accuracy_partial_match(self):
        """Test sequence accuracy with partial matches."""
        predicted = ["hello wrld", "tst case"]  # Slight differences
        ground_truth = ["hello world", "test case"]
        
        result = self.metrics.calculate_sequence_accuracy(predicted, ground_truth)
        
        self.assertLess(result['sequence_similarity'], 1.0)
        self.assertGreater(result['sequence_similarity'], 0.5)  # Should be reasonably high
        self.assertGreater(result['sequence_error_rate'], 0.0)
    
    def test_calculate_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        predicted = ["hello world", "test case", "sample text"]
        ground_truth = ["hello world", "test case", "sample text"]
        
        result = self.metrics.calculate_comprehensive_metrics(predicted, ground_truth)
        
        # Check structure
        self.assertIn('character_metrics', result)
        self.assertIn('word_metrics', result)
        self.assertIn('sequence_metrics', result)
        self.assertIn('summary', result)
        self.assertIn('sample_count', result)
        
        # Check summary
        summary = result['summary']
        self.assertIn('character_accuracy', summary)
        self.assertIn('word_accuracy', summary)
        self.assertIn('exact_match_accuracy', summary)
        self.assertIn('sequence_similarity', summary)
        
        # Perfect match should have perfect scores
        self.assertEqual(summary['character_accuracy'], 1.0)
        self.assertEqual(summary['word_accuracy'], 1.0)
        self.assertEqual(summary['exact_match_accuracy'], 1.0)
        self.assertEqual(summary['sequence_similarity'], 1.0)
        
        self.assertEqual(result['sample_count'], 3)
    
    def test_calculate_edit_distance(self):
        """Test edit distance calculation."""
        # Test cases with known edit distances
        test_cases = [
            ("hello", "hello", 0),  # Identical
            ("hello", "helo", 1),   # One deletion
            ("hello", "helloo", 1), # One insertion
            ("hello", "hallo", 1),  # One substitution
            ("", "hello", 5),       # Empty to string
            ("hello", "", 5),       # String to empty
            ("abc", "def", 3),      # Complete substitution
        ]
        
        for s1, s2, expected_distance in test_cases:
            with self.subTest(s1=s1, s2=s2):
                distance = self.metrics._calculate_edit_distance(s1, s2)
                self.assertEqual(distance, expected_distance)
    
    def test_calculate_jaro_winkler_similarity(self):
        """Test Jaro-Winkler similarity calculation."""
        # Test identical strings
        similarity = self.metrics._calculate_jaro_winkler_similarity("hello", "hello")
        self.assertEqual(similarity, 1.0)
        
        # Test completely different strings
        similarity = self.metrics._calculate_jaro_winkler_similarity("abc", "xyz")
        self.assertLess(similarity, 0.5)
        
        # Test similar strings with common prefix
        similarity = self.metrics._calculate_jaro_winkler_similarity("hello", "helo")
        self.assertGreater(similarity, 0.8)  # Should be high due to common prefix
    
    def test_char_level_accuracy_helper(self):
        """Test character-level accuracy helper method."""
        correct, total = self.metrics._calculate_char_level_accuracy("hello", "hello")
        self.assertEqual(correct, 5)
        self.assertEqual(total, 5)
        
        correct, total = self.metrics._calculate_char_level_accuracy("helo", "hello")
        self.assertEqual(correct, 3)  # h, e, l match (o is at different position)
        self.assertEqual(total, 5)    # Based on ground truth length
        
        correct, total = self.metrics._calculate_char_level_accuracy("", "hello")
        self.assertEqual(correct, 0)
        self.assertEqual(total, 5)
    
    def test_word_level_accuracy_helper(self):
        """Test word-level accuracy helper method."""
        pred_words = ["hello", "world"]
        gt_words = ["hello", "world"]
        correct, total = self.metrics._calculate_word_level_accuracy(pred_words, gt_words)
        self.assertEqual(correct, 2)
        self.assertEqual(total, 2)
        
        pred_words = ["hello", "wrld"]
        gt_words = ["hello", "world"]
        correct, total = self.metrics._calculate_word_level_accuracy(pred_words, gt_words)
        self.assertEqual(correct, 1)  # Only "hello" matches
        self.assertEqual(total, 2)
        
        pred_words = ["hello"]
        gt_words = ["hello", "world"]
        correct, total = self.metrics._calculate_word_level_accuracy(pred_words, gt_words)
        self.assertEqual(correct, 1)  # Only "hello" matches
        self.assertEqual(total, 2)    # Based on longer list
    
    def test_print_metrics_report(self):
        """Test metrics report printing."""
        predicted = ["hello world", "test case"]
        ground_truth = ["hello world", "test case"]
        
        metrics = self.metrics.calculate_comprehensive_metrics(predicted, ground_truth)
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        self.metrics.print_metrics_report(metrics)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        
        # Check that report contains expected sections
        self.assertIn("TEXT RECOGNITION ACCURACY REPORT", output)
        self.assertIn("SUMMARY:", output)
        self.assertIn("CHARACTER-LEVEL METRICS:", output)
        self.assertIn("WORD-LEVEL METRICS:", output)
        self.assertIn("SEQUENCE-LEVEL METRICS:", output)
        self.assertIn("Sample Count:", output)


if __name__ == '__main__':
    unittest.main()