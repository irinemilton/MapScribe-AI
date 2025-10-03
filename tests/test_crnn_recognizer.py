"""
Unit tests for CRNN-based text recognition.
"""

import unittest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.recognition.crnn_recognizer import (
    CRNNModel, BeamSearchDecoder, CRNNRecognitionModel, SingleLanguageCRNNRecognizer
)
from src.models.data_structures import RecognitionConfig, Detection, BoundingBox


class TestCRNNModel(unittest.TestCase):
    """Test cases for CRNNModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = CRNNModel(
            img_height=32,
            img_width=128,
            num_channels=3,
            num_classes=37,
            hidden_size=256
        )
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.img_height, 32)
        self.assertEqual(self.model.img_width, 128)
        self.assertEqual(self.model.num_channels, 3)
        self.assertEqual(self.model.num_classes, 37)
        self.assertEqual(self.model.hidden_size, 256)
    
    def test_forward_pass(self):
        """Test forward pass through model."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 32, 128)
        
        output = self.model(input_tensor)
        
        # Check output shape
        self.assertEqual(len(output.shape), 3)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[2], 37)  # num_classes
        
        # Output should be logits (can be any real number)
        self.assertTrue(torch.is_tensor(output))
    
    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        params = list(self.model.parameters())
        self.assertGreater(len(params), 0)
        
        # Check that parameters require gradients
        for param in params:
            self.assertTrue(param.requires_grad)


class TestBeamSearchDecoder(unittest.TestCase):
    """Test cases for BeamSearchDecoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.charset = ['<blank>', 'a', 'b', 'c', 'd', 'e', 'h', 'l', 'o', 'r', 'w']
        self.decoder = BeamSearchDecoder(
            charset=self.charset,
            beam_width=3,
            blank_id=0
        )
    
    def test_initialization(self):
        """Test decoder initialization."""
        self.assertEqual(self.decoder.charset, self.charset)
        self.assertEqual(self.decoder.beam_width, 3)
        self.assertEqual(self.decoder.blank_id, 0)
        self.assertEqual(len(self.decoder.char_to_id), len(self.charset))
        self.assertEqual(len(self.decoder.id_to_char), len(self.charset))
    
    def test_decode_simple_sequence(self):
        """Test decoding a simple sequence."""
        # Create mock log probabilities for "hello"
        seq_len = 10
        num_classes = len(self.charset)
        batch_size = 1
        
        # Create log probabilities that favor "hello"
        log_probs = np.full((batch_size, seq_len, num_classes), -10.0)
        
        # Set high probabilities for target characters
        # h=5, e=4, l=7, l=7, o=8
        target_sequence = [5, 4, 7, 7, 8]  # "hello"
        
        for t, char_id in enumerate(target_sequence):
            if t < seq_len:
                log_probs[0, t, char_id] = -0.1  # High probability
        
        # Set blank probabilities for other timesteps
        for t in range(len(target_sequence), seq_len):
            log_probs[0, t, 0] = -0.1  # Blank token
        
        results = self.decoder.decode(log_probs)
        
        self.assertEqual(len(results), 1)
        decoded_text, confidence, char_confidences = results[0]
        
        # Should decode to something reasonable
        self.assertIsInstance(decoded_text, str)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(char_confidences, list)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_decode_empty_sequence(self):
        """Test decoding empty sequence."""
        batch_size = 1
        seq_len = 5
        num_classes = len(self.charset)
        
        # All blank tokens
        log_probs = np.full((batch_size, seq_len, num_classes), -10.0)
        log_probs[:, :, 0] = -0.1  # High probability for blank
        
        results = self.decoder.decode(log_probs)
        
        self.assertEqual(len(results), 1)
        decoded_text, confidence, char_confidences = results[0]
        
        # Should decode to empty string or very short string
        self.assertIsInstance(decoded_text, str)
        self.assertLessEqual(len(decoded_text), 1)
    
    def test_decode_batch(self):
        """Test decoding multiple sequences in batch."""
        batch_size = 3
        seq_len = 8
        num_classes = len(self.charset)
        
        log_probs = np.random.randn(batch_size, seq_len, num_classes)
        
        results = self.decoder.decode(log_probs)
        
        self.assertEqual(len(results), batch_size)
        
        for decoded_text, confidence, char_confidences in results:
            self.assertIsInstance(decoded_text, str)
            self.assertIsInstance(confidence, float)
            self.assertIsInstance(char_confidences, list)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)


class TestCRNNRecognitionModel(unittest.TestCase):
    """Test cases for CRNNRecognitionModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RecognitionConfig(
            model_architecture='CRNN',
            max_sequence_length=32,
            vocabulary_size=100,
            languages=['en'],
            beam_width=5
        )
        self.recognition_model = CRNNRecognitionModel(self.config)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.recognition_model.config, self.config)
        self.assertIsNone(self.recognition_model.model)
        self.assertIsNone(self.recognition_model.decoder)
        self.assertIsNotNone(self.recognition_model.charset)
        self.assertGreater(len(self.recognition_model.charset), 0)
    
    def test_create_charset(self):
        """Test character set creation."""
        charset = self.recognition_model._create_charset()
        
        # Should contain blank token
        self.assertIn('<blank>', charset)
        
        # Should contain letters
        self.assertIn('a', charset)
        self.assertIn('A', charset)
        self.assertIn('z', charset)
        self.assertIn('Z', charset)
        
        # Should contain digits
        self.assertIn('0', charset)
        self.assertIn('9', charset)
        
        # Should contain common punctuation
        self.assertIn(' ', charset)
        self.assertIn('.', charset)
        self.assertIn(',', charset)
    
    def test_load_model_nonexistent_file(self):
        """Test loading model from nonexistent file."""
        # Should create dummy model without raising exception
        self.recognition_model.load_model("nonexistent_model.pth")
        
        self.assertIsNotNone(self.recognition_model.model)
        self.assertIsNotNone(self.recognition_model.decoder)
    
    def test_predict_without_model(self):
        """Test prediction without loaded model."""
        text_region = np.random.rand(32, 128, 3).astype(np.float32)
        
        with self.assertRaises(RuntimeError):
            self.recognition_model.predict(text_region)
    
    def test_predict_with_dummy_model(self):
        """Test prediction with dummy model."""
        # Load dummy model
        self.recognition_model.load_model("dummy_model.pth")
        
        # Create test input
        text_region = np.random.rand(32, 128, 3).astype(np.float32)
        
        # Should not raise exception
        text, confidences = self.recognition_model.predict(text_region)
        
        self.assertIsInstance(text, str)
        self.assertIsInstance(confidences, list)
    
    def test_prepare_input_rgb(self):
        """Test input preparation for RGB image."""
        text_region = np.random.rand(32, 128, 3).astype(np.float32)
        
        # Load dummy model to initialize
        self.recognition_model.load_model("dummy_model.pth")
        
        input_tensor = self.recognition_model._prepare_input(text_region)
        
        self.assertEqual(input_tensor.shape, (1, 3, 32, 128))
        self.assertTrue(torch.is_tensor(input_tensor))
    
    def test_prepare_input_grayscale(self):
        """Test input preparation for grayscale image."""
        text_region = np.random.rand(32, 128).astype(np.float32)
        
        # Load dummy model to initialize
        self.recognition_model.load_model("dummy_model.pth")
        
        input_tensor = self.recognition_model._prepare_input(text_region)
        
        self.assertEqual(input_tensor.shape, (1, 3, 32, 128))
        self.assertTrue(torch.is_tensor(input_tensor))
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = self.recognition_model.get_supported_languages()
        
        self.assertIn('en', languages)
        self.assertIn('latin', languages)


class TestSingleLanguageCRNNRecognizer(unittest.TestCase):
    """Test cases for SingleLanguageCRNNRecognizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RecognitionConfig(
            model_architecture='CRNN',
            max_sequence_length=32,
            vocabulary_size=100,
            languages=['en'],
            beam_width=5
        )
        self.recognizer = SingleLanguageCRNNRecognizer(self.config)
    
    def test_initialization(self):
        """Test recognizer initialization."""
        self.assertIsNotNone(self.recognizer.config)
        self.assertIsNotNone(self.recognizer.crnn_model)
        self.assertEqual(self.recognizer.config.languages, ['en'])
    
    def test_initialization_with_default_config(self):
        """Test initialization with default config."""
        recognizer = SingleLanguageCRNNRecognizer()
        
        self.assertIsNotNone(recognizer.config)
        self.assertEqual(recognizer.config.languages, ['en'])
        self.assertEqual(recognizer.config.model_architecture, 'CRNN')
    
    def test_initialization_force_single_language(self):
        """Test that multi-language config is forced to single language."""
        multi_config = RecognitionConfig(
            model_architecture='CRNN',
            max_sequence_length=32,
            vocabulary_size=100,
            languages=['en', 'zh', 'chinese'],  # Multiple valid languages
            beam_width=5
        )
        
        recognizer = SingleLanguageCRNNRecognizer(multi_config)
        
        # Should be forced to single language
        self.assertEqual(len(recognizer.config.languages), 1)
        self.assertEqual(recognizer.config.languages[0], 'en')
    
    def test_get_default_charset(self):
        """Test default charset creation."""
        charset = self.recognizer._get_default_charset()
        
        self.assertIn('<blank>', charset)
        self.assertIn('a', charset)
        self.assertIn('A', charset)
        self.assertIn('0', charset)
        self.assertIn(' ', charset)
    
    def test_recognize_text_region(self):
        """Test text region recognition."""
        # Load dummy model first
        self.recognizer.load_model("dummy_model.pth")
        
        text_region = np.random.rand(32, 128, 3).astype(np.float32)
        
        text, confidences = self.recognizer._recognize_text_region(text_region, 'en')
        
        self.assertIsInstance(text, str)
        self.assertIsInstance(confidences, list)
    
    def test_load_model(self):
        """Test model loading."""
        # Should not raise exception
        self.recognizer.load_model("dummy_model.pth")
        
        # Model should be loaded
        self.assertIsNotNone(self.recognizer.crnn_model.model)
        self.assertIsNotNone(self.recognizer.crnn_model.decoder)
    
    def test_get_model_info(self):
        """Test getting model information."""
        self.recognizer.load_model("dummy_model.pth")
        
        info = self.recognizer.get_model_info()
        
        self.assertIn('architecture', info)
        self.assertIn('languages', info)
        self.assertIn('charset_size', info)
        self.assertIn('beam_width', info)
        self.assertIn('device', info)
        
        self.assertEqual(info['architecture'], 'CRNN')
        self.assertEqual(info['languages'], ['en'])
    
    def test_recognize_words_integration(self):
        """Test word recognition integration."""
        # Load dummy model
        self.recognizer.load_model("dummy_model.pth")
        
        # Create test image and detection
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
        self.assertIsInstance(results[0].text, str)
        self.assertIsInstance(results[0].character_confidences, list)
        self.assertEqual(results[0].language, 'en')


if __name__ == '__main__':
    unittest.main()