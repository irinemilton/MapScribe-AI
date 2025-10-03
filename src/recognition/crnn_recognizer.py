"""
CRNN-based text recognition implementation for single-language (Latin) text recognition.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import json

from .base_recognizer import BaseTextRecognizer
from .interfaces import RecognitionModelInterface
from ..models.data_structures import RecognitionConfig


class CRNNModel(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) model for text recognition.
    
    Architecture:
    - CNN backbone for feature extraction
    - RNN (LSTM) for sequence modeling
    - CTC loss for sequence-to-sequence learning
    """
    
    def __init__(self, 
                 img_height: int = 32,
                 img_width: int = 128,
                 num_channels: int = 3,
                 num_classes: int = 37,  # 26 letters + 10 digits + blank
                 hidden_size: int = 256):
        """
        Initialize CRNN model.
        
        Args:
            img_height: Input image height
            img_width: Input image width
            num_channels: Number of input channels
            num_classes: Number of output classes (including CTC blank)
            hidden_size: LSTM hidden size
        """
        super(CRNNModel, self).__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # CNN backbone
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x128 -> 16x64
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x64 -> 8x32
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Fourth conv block
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 8x32 -> 4x32
            
            # Fifth conv block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Sixth conv block
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 4x32 -> 2x32
            
            # Final conv block
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),  # 2x32 -> 1x31
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # RNN layers
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Classification layer
        self.classifier = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CRNN model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, num_classes)
        """
        # CNN feature extraction
        conv_features = self.cnn(x)  # (batch_size, 512, 1, width)
        
        # Reshape for RNN: (batch_size, width, 512)
        batch_size, channels, height, width = conv_features.size()
        conv_features = conv_features.squeeze(2)  # Remove height dimension
        conv_features = conv_features.permute(0, 2, 1)  # (batch_size, width, channels)
        
        # RNN sequence modeling
        rnn_output, _ = self.rnn(conv_features)  # (batch_size, width, hidden_size*2)
        
        # Classification
        output = self.classifier(rnn_output)  # (batch_size, width, num_classes)
        
        return output


class BeamSearchDecoder:
    """
    Beam search decoder for CTC outputs.
    """
    
    def __init__(self, 
                 charset: List[str],
                 beam_width: int = 5,
                 blank_id: int = 0):
        """
        Initialize beam search decoder.
        
        Args:
            charset: List of characters in vocabulary
            beam_width: Number of beams to keep
            blank_id: ID of CTC blank token
        """
        self.charset = charset
        self.beam_width = beam_width
        self.blank_id = blank_id
        self.char_to_id = {char: i for i, char in enumerate(charset)}
        self.id_to_char = {i: char for i, char in enumerate(charset)}
    
    def decode(self, 
               log_probs: np.ndarray,
               sequence_lengths: Optional[List[int]] = None) -> List[Tuple[str, float, List[float]]]:
        """
        Decode CTC outputs using beam search.
        
        Args:
            log_probs: Log probabilities of shape (batch_size, seq_len, num_classes)
            sequence_lengths: Optional sequence lengths for each batch item
            
        Returns:
            List of (decoded_text, confidence, char_confidences) tuples
        """
        batch_size, seq_len, num_classes = log_probs.shape
        results = []
        
        for batch_idx in range(batch_size):
            batch_log_probs = log_probs[batch_idx]
            seq_length = sequence_lengths[batch_idx] if sequence_lengths else seq_len
            
            # Perform beam search for this sequence
            decoded_text, confidence, char_confidences = self._beam_search_single(
                batch_log_probs[:seq_length]
            )
            
            results.append((decoded_text, confidence, char_confidences))
        
        return results
    
    def _beam_search_single(self, log_probs: np.ndarray) -> Tuple[str, float, List[float]]:
        """
        Perform beam search for a single sequence.
        
        Args:
            log_probs: Log probabilities of shape (seq_len, num_classes)
            
        Returns:
            Tuple of (decoded_text, confidence, char_confidences)
        """
        seq_len, num_classes = log_probs.shape
        
        # Initialize beams: (prefix, log_prob, char_confidences)
        beams = [("", 0.0, [])]
        
        for t in range(seq_len):
            new_beams = []
            
            for prefix, log_prob, char_confs in beams:
                # Get top-k characters for this timestep
                top_k_indices = np.argsort(log_probs[t])[-self.beam_width:]
                
                for char_id in top_k_indices:
                    char_log_prob = log_probs[t, char_id]
                    char_prob = np.exp(char_log_prob)
                    
                    if char_id == self.blank_id:
                        # Blank token - don't extend prefix
                        new_log_prob = log_prob + char_log_prob
                        new_beams.append((prefix, new_log_prob, char_confs.copy()))
                    else:
                        # Non-blank token
                        char = self.id_to_char.get(char_id, '')
                        
                        # Avoid consecutive duplicates (CTC rule)
                        if len(prefix) > 0 and prefix[-1] == char:
                            new_log_prob = log_prob + char_log_prob
                            new_beams.append((prefix, new_log_prob, char_confs.copy()))
                        else:
                            new_prefix = prefix + char
                            new_log_prob = log_prob + char_log_prob
                            new_char_confs = char_confs.copy() + [char_prob]
                            new_beams.append((new_prefix, new_log_prob, new_char_confs))
            
            # Keep top beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:self.beam_width]
        
        # Return best beam
        if beams:
            best_prefix, best_log_prob, best_char_confs = beams[0]
            # Normalize confidence to [0, 1] range
            confidence = np.exp(best_log_prob / max(len(best_prefix), 1))
            confidence = min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
            return best_prefix, confidence, best_char_confs
        else:
            return "", 0.0, []


class CRNNRecognitionModel(RecognitionModelInterface):
    """
    CRNN-based recognition model implementation.
    """
    
    def __init__(self, config: RecognitionConfig):
        """
        Initialize CRNN recognition model.
        
        Args:
            config: Recognition configuration
        """
        self.config = config
        self.model: Optional[CRNNModel] = None
        self.decoder: Optional[BeamSearchDecoder] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.charset = self._create_charset()
        self.logger = logging.getLogger(__name__)
        
    def _create_charset(self) -> List[str]:
        """Create character set for Latin text recognition."""
        # CTC blank token (index 0)
        charset = ['<blank>']
        
        # Lowercase letters
        charset.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
        
        # Uppercase letters
        charset.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        
        # Digits
        charset.extend([str(i) for i in range(10)])
        
        # Common punctuation and symbols
        charset.extend([' ', '.', ',', ':', ';', '!', '?', '-', "'", '"', '(', ')', '[', ']'])
        
        return charset
    
    def load_model(self, model_path: str, language: Optional[str] = None) -> None:
        """
        Load CRNN model from file.
        
        Args:
            model_path: Path to model file
            language: Optional language specification (ignored for single-language model)
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                # Create a dummy model for testing/demo purposes
                self.logger.warning(f"Model file {model_path} not found. Creating dummy model.")
                self._create_dummy_model()
                return
            
            # Load model state
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model architecture
            self.model = CRNNModel(
                img_height=32,
                img_width=128,
                num_channels=3,
                num_classes=len(self.charset)
            )
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize decoder
            self.decoder = BeamSearchDecoder(
                charset=self.charset,
                beam_width=self.config.beam_width,
                blank_id=0
            )
            
            self.logger.info(f"Loaded CRNN model from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self._create_dummy_model()
    
    def _create_dummy_model(self) -> None:
        """Create a dummy model for testing purposes."""
        self.model = CRNNModel(
            img_height=32,
            img_width=128,
            num_channels=3,
            num_classes=len(self.charset)
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.decoder = BeamSearchDecoder(
            charset=self.charset,
            beam_width=self.config.beam_width,
            blank_id=0
        )
        
        self.logger.info("Created dummy CRNN model for testing")
    
    def predict(self, text_region: np.ndarray) -> Tuple[str, List[float]]:
        """
        Predict text and character confidences for a region.
        
        Args:
            text_region: Image region containing text (H, W, C) in range [0, 1]
            
        Returns:
            Tuple of (recognized_text, character_confidences)
        """
        if self.model is None or self.decoder is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Prepare input tensor
            input_tensor = self._prepare_input(text_region)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)  # (1, seq_len, num_classes)
                log_probs = F.log_softmax(output, dim=2)
            
            # Convert to numpy for decoding
            log_probs_np = log_probs.cpu().numpy()
            
            # Decode using beam search
            results = self.decoder.decode(log_probs_np)
            
            if results:
                text, confidence, char_confidences = results[0]
                return text, char_confidences
            else:
                return "", []
                
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return "", []
    
    def _prepare_input(self, text_region: np.ndarray) -> torch.Tensor:
        """
        Prepare input tensor from text region.
        
        Args:
            text_region: Image region (H, W, C) in range [0, 1]
            
        Returns:
            Input tensor ready for model
        """
        # Ensure correct shape and type
        if len(text_region.shape) == 2:
            # Grayscale to RGB
            text_region = np.stack([text_region] * 3, axis=-1)
        
        # Resize to model input size if needed
        if text_region.shape[:2] != (32, 128):
            from PIL import Image
            pil_image = Image.fromarray((text_region * 255).astype(np.uint8))
            pil_image = pil_image.resize((128, 32), Image.LANCZOS)
            text_region = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convert to tensor: (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(text_region).permute(2, 0, 1)
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        tensor = tensor.unsqueeze(0)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        return tensor
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        return ['en', 'latin']


class SingleLanguageCRNNRecognizer(BaseTextRecognizer):
    """
    Single-language CRNN-based text recognizer for Latin characters.
    """
    
    def __init__(self, config: Optional[RecognitionConfig] = None):
        """
        Initialize single-language CRNN recognizer.
        
        Args:
            config: Recognition configuration
        """
        # Ensure single language configuration
        if config is None:
            config = RecognitionConfig(
                model_architecture='CRNN',
                max_sequence_length=32,
                vocabulary_size=len(self._get_default_charset()),
                languages=['en'],
                beam_width=5
            )
        elif len(config.languages) > 1:
            # Force single language
            config.languages = [config.languages[0]]
            
        super().__init__(config)
        
        # Initialize CRNN model
        self.crnn_model = CRNNRecognitionModel(config)
        
    def _get_default_charset(self) -> List[str]:
        """Get default character set."""
        charset = ['<blank>']
        charset.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
        charset.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        charset.extend([str(i) for i in range(10)])
        charset.extend([' ', '.', ',', ':', ';', '!', '?', '-', "'", '"', '(', ')', '[', ']'])
        return charset
    
    def _recognize_text_region(self, text_region: np.ndarray, language: str) -> Tuple[str, List[float]]:
        """
        Recognize text in a preprocessed region using CRNN.
        
        Args:
            text_region: Preprocessed text region
            language: Language code (ignored for single-language model)
            
        Returns:
            Tuple of (recognized_text, character_confidences)
        """
        return self.crnn_model.predict(text_region)
    
    def load_model(self, model_path: str, language: Optional[str] = None) -> None:
        """
        Load CRNN model from file.
        
        Args:
            model_path: Path to model file
            language: Optional language specification (ignored)
        """
        self.crnn_model.load_model(model_path, language)
        self.logger.info(f"Loaded single-language CRNN model from {model_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'architecture': 'CRNN',
            'languages': self.config.languages,
            'charset_size': len(self.crnn_model.charset),
            'beam_width': self.config.beam_width,
            'device': str(self.crnn_model.device)
        }