"""
Base text recognition implementation with character-level confidence tracking
and preprocessing capabilities.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2
from PIL import Image
import logging

from .interfaces import TextRecognizerInterface, RecognitionModelInterface
from ..models.data_structures import Detection, RecognizedText, RecognitionConfig
from ..core.error_handling import (
    RecognitionError, PreprocessingError, ErrorContext, handle_errors
)
from ..core.graceful_degradation import with_graceful_degradation
from ..core.monitoring import monitor_performance


class TextPreprocessor:
    """Handles preprocessing of text regions for recognition."""
    
    def __init__(self, config: Optional[RecognitionConfig] = None):
        """
        Initialize text preprocessor.
        
        Args:
            config: Recognition configuration
        """
        self.config = config or RecognitionConfig(
            model_architecture='CRNN',
            max_sequence_length=32,
            vocabulary_size=1000,
            languages=['en'],
            beam_width=5
        )
        self.logger = logging.getLogger(__name__)
    
    def preprocess_text_region(self, image: np.ndarray, detection: Detection) -> np.ndarray:
        """
        Preprocess a text region for recognition.
        
        Args:
            image: Full input image
            detection: Detection containing bounding box
            
        Returns:
            Preprocessed text region ready for recognition
        """
        # Extract region from image
        x1, y1, x2, y2 = detection.image_coordinates
        text_region = image[y1:y2, x1:x2]
        
        if text_region.size == 0:
            self.logger.warning("Empty text region detected")
            return np.zeros((32, 128, 3), dtype=np.uint8)
        
        # Apply preprocessing steps
        processed_region = self._normalize_region(text_region)
        processed_region = self._enhance_contrast(processed_region)
        processed_region = self._resize_region(processed_region)
        processed_region = self._apply_padding(processed_region)
        
        return processed_region
    
    def _normalize_region(self, region: np.ndarray) -> np.ndarray:
        """Normalize pixel values to [0, 1] range."""
        if region.dtype != np.float32:
            region = region.astype(np.float32) / 255.0
        return region
    
    def _enhance_contrast(self, region: np.ndarray) -> np.ndarray:
        """Enhance contrast for better text visibility."""
        # Convert to uint8 for OpenCV operations
        if region.dtype == np.float32:
            region_uint8 = (region * 255).astype(np.uint8)
        else:
            region_uint8 = region
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(region_uint8.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(region_uint8, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(region_uint8)
        
        # Convert back to float32
        return enhanced.astype(np.float32) / 255.0
    
    def _resize_region(self, region: np.ndarray, target_height: int = 32) -> np.ndarray:
        """Resize region to standard height while maintaining aspect ratio."""
        h, w = region.shape[:2]
        
        if h == 0 or w == 0:
            return np.zeros((target_height, 128, 3), dtype=np.float32)
        
        # Calculate new width maintaining aspect ratio
        aspect_ratio = w / h
        new_width = int(target_height * aspect_ratio)
        
        # Ensure minimum width
        new_width = max(new_width, 16)
        
        # Resize using PIL for better quality
        if len(region.shape) == 3:
            pil_image = Image.fromarray((region * 255).astype(np.uint8))
        else:
            pil_image = Image.fromarray((region * 255).astype(np.uint8), mode='L')
        
        resized_pil = pil_image.resize((new_width, target_height), Image.LANCZOS)
        resized = np.array(resized_pil).astype(np.float32) / 255.0
        
        # Ensure 3 channels
        if len(resized.shape) == 2:
            resized = np.stack([resized] * 3, axis=-1)
        
        return resized
    
    def _apply_padding(self, region: np.ndarray, target_width: int = 128) -> np.ndarray:
        """Apply padding to reach target width."""
        h, w = region.shape[:2]
        
        if w >= target_width:
            # Crop if too wide
            return region[:, :target_width]
        
        # Pad with white background
        pad_width = target_width - w
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        
        if len(region.shape) == 3:
            padded = np.pad(region, ((0, 0), (pad_left, pad_right), (0, 0)), 
                          mode='constant', constant_values=1.0)
        else:
            padded = np.pad(region, ((0, 0), (pad_left, pad_right)), 
                          mode='constant', constant_values=1.0)
        
        return padded


class CharacterConfidenceTracker:
    """Tracks character-level confidence scores during recognition."""
    
    def __init__(self):
        """Initialize confidence tracker."""
        self.character_scores: List[float] = []
        self.sequence_scores: List[List[float]] = []
        self.logger = logging.getLogger(__name__)
    
    def track_character_confidence(self, char: str, confidence: float) -> None:
        """
        Track confidence for a single character.
        
        Args:
            char: Recognized character
            confidence: Confidence score for the character
        """
        if confidence < 0.0 or confidence > 1.0:
            self.logger.warning(f"Invalid confidence score {confidence} for character '{char}'")
            confidence = max(0.0, min(1.0, confidence))
        
        self.character_scores.append(confidence)
    
    def track_sequence_confidence(self, sequence: str, confidences: List[float]) -> None:
        """
        Track confidence for a character sequence.
        
        Args:
            sequence: Recognized text sequence
            confidences: List of confidence scores for each character
        """
        if len(sequence) != len(confidences):
            self.logger.warning(
                f"Sequence length {len(sequence)} doesn't match confidence length {len(confidences)}"
            )
            # Pad or truncate confidences to match sequence length
            if len(confidences) < len(sequence):
                confidences.extend([0.5] * (len(sequence) - len(confidences)))
            else:
                confidences = confidences[:len(sequence)]
        
        self.sequence_scores.append(confidences)
        self.character_scores.extend(confidences)
    
    def get_average_confidence(self) -> float:
        """Get average confidence across all tracked characters."""
        if not self.character_scores:
            return 0.0
        return sum(self.character_scores) / len(self.character_scores)
    
    def get_min_confidence(self) -> float:
        """Get minimum confidence score."""
        if not self.character_scores:
            return 0.0
        return min(self.character_scores)
    
    def get_confidence_distribution(self) -> Dict[str, float]:
        """Get confidence score distribution statistics."""
        if not self.character_scores:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        scores = np.array(self.character_scores)
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores))
        }
    
    def reset(self) -> None:
        """Reset all tracked confidence scores."""
        self.character_scores.clear()
        self.sequence_scores.clear()


class BaseTextRecognizer(TextRecognizerInterface):
    """
    Base implementation of text recognition with preprocessing and confidence tracking.
    """
    
    def __init__(self, config: Optional[RecognitionConfig] = None):
        """
        Initialize base text recognizer.
        
        Args:
            config: Recognition configuration
        """
        self.config = config or RecognitionConfig(
            model_architecture='CRNN',
            max_sequence_length=32,
            vocabulary_size=1000,
            languages=['en'],
            beam_width=5
        )
        
        self.preprocessor = TextPreprocessor(self.config)
        self.confidence_tracker = CharacterConfidenceTracker()
        self.models: Dict[str, RecognitionModelInterface] = {}
        self.current_language = 'en'
        self.logger = logging.getLogger(__name__)
        
        # Language detection confidence threshold
        self.language_detection_threshold = 0.7
    
    @monitor_performance("recognition")
    @with_graceful_degradation("recognition")
    def recognize_words(self, image: np.ndarray, detections: List[Detection]) -> List[RecognizedText]:
        """
        Recognize text in detected word regions.
        
        Args:
            image: Input image array
            detections: List of word detections
            
        Returns:
            List of recognized text results
        """
        context = ErrorContext(
            component="text_recognizer",
            operation="recognize_words",
            input_data={
                "image_shape": image.shape,
                "num_detections": len(detections)
            }
        )
        
        results = []
        
        for i, detection in enumerate(detections):
            if detection.detection_type != 'word':
                self.logger.warning(f"Expected word detection, got {detection.detection_type}")
                # Still process but treat as word
                detection.detection_type = 'word'
            
            try:
                # Preprocess text region
                text_region = self.preprocessor.preprocess_text_region(image, detection)
                
                # Detect language if multi-language support is enabled
                if len(self.config.languages) > 1:
                    detected_language = self.detect_language(text_region)
                else:
                    detected_language = self.config.languages[0]
                
                # Recognize text using appropriate model
                recognized_text, char_confidences = self._recognize_text_region(
                    text_region, detected_language
                )
                
                # Calculate overall recognition confidence
                recognition_confidence = (
                    sum(char_confidences) / len(char_confidences) 
                    if char_confidences else 0.0
                )
                
                # Create recognized text result
                result = RecognizedText(
                    detection=detection,
                    text=recognized_text,
                    language=detected_language,
                    recognition_confidence=recognition_confidence,
                    character_confidences=char_confidences
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error recognizing word {i}: {e}")
                
                # Create error context for this specific recognition failure
                word_context = ErrorContext(
                    component="text_recognizer",
                    operation="recognize_single_word",
                    input_data={
                        "detection_index": i,
                        "detection_bbox": detection.bbox.__dict__ if detection.bbox else None
                    }
                )
                
                # Raise recognition error but continue with other detections
                recognition_error = RecognitionError(
                    f"Word recognition failed at index {i}: {str(e)}",
                    context=word_context
                )
                
                # Create empty result for failed recognition
                result = RecognizedText(
                    detection=detection,
                    text="",
                    language=self.current_language,
                    recognition_confidence=0.0,
                    character_confidences=[]
                )
                results.append(result)
        
        return results
    
    def recognize_phrases(self, image: np.ndarray, detections: List[Detection]) -> List[RecognizedText]:
        """
        Recognize text in detected phrase regions.
        
        Args:
            image: Input image array
            detections: List of phrase detections
            
        Returns:
            List of recognized text results
        """
        results = []
        
        for detection in detections:
            if detection.detection_type != 'phrase':
                self.logger.warning(f"Expected phrase detection, got {detection.detection_type}")
                # Still process but treat as phrase
                detection.detection_type = 'phrase'
            
            try:
                # Preprocess text region
                text_region = self.preprocessor.preprocess_text_region(image, detection)
                
                # Detect language if multi-language support is enabled
                if len(self.config.languages) > 1:
                    detected_language = self.detect_language(text_region)
                else:
                    detected_language = self.config.languages[0]
                
                # Recognize text using appropriate model
                recognized_text, char_confidences = self._recognize_text_region(
                    text_region, detected_language
                )
                
                # Calculate overall recognition confidence
                recognition_confidence = (
                    sum(char_confidences) / len(char_confidences) 
                    if char_confidences else 0.0
                )
                
                # Create recognized text result
                result = RecognizedText(
                    detection=detection,
                    text=recognized_text,
                    language=detected_language,
                    recognition_confidence=recognition_confidence,
                    character_confidences=char_confidences
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error recognizing phrase: {e}")
                # Create empty result for failed recognition
                result = RecognizedText(
                    detection=detection,
                    text="",
                    language=self.current_language,
                    recognition_confidence=0.0,
                    character_confidences=[]
                )
                results.append(result)
        
        return results
    
    def handle_multilingual(self, text_region: np.ndarray) -> str:
        """
        Handle multi-language text recognition.
        
        Args:
            text_region: Image region containing text
            
        Returns:
            Recognized text string
        """
        # Detect language first
        detected_language = self.detect_language(text_region)
        
        # Use appropriate model for recognition
        recognized_text, _ = self._recognize_text_region(text_region, detected_language)
        
        return recognized_text
    
    def detect_language(self, text_region: np.ndarray) -> str:
        """
        Detect the language of text in the region.
        
        Args:
            text_region: Image region containing text
            
        Returns:
            Detected language code
        """
        # This is a placeholder implementation
        # In a real implementation, this would use a language detection model
        
        # For now, use simple heuristics based on image characteristics
        # or default to the first supported language
        
        if 'zh' in self.config.languages or 'chinese' in self.config.languages:
            # Simple heuristic: if the region is more square-like, it might be Chinese
            h, w = text_region.shape[:2]
            aspect_ratio = w / h if h > 0 else 1.0
            
            if aspect_ratio < 2.0:  # More square-like regions might be Chinese characters
                return 'zh' if 'zh' in self.config.languages else 'chinese'
        
        # Default to first language or English
        if 'en' in self.config.languages:
            return 'en'
        elif 'latin' in self.config.languages:
            return 'latin'
        else:
            return self.config.languages[0] if self.config.languages else 'en'
    
    @abstractmethod
    def _recognize_text_region(self, text_region: np.ndarray, language: str) -> Tuple[str, List[float]]:
        """
        Abstract method for recognizing text in a preprocessed region.
        
        Args:
            text_region: Preprocessed text region
            language: Detected or specified language
            
        Returns:
            Tuple of (recognized_text, character_confidences)
        """
        pass
    
    def load_model(self, model_path: str, language: Optional[str] = None) -> None:
        """
        Load recognition model for specified language.
        
        Args:
            model_path: Path to model file
            language: Language code for the model
        """
        if language is None:
            language = self.current_language
        
        # This will be implemented by concrete subclasses
        self.logger.info(f"Loading model from {model_path} for language {language}")
    
    def set_current_language(self, language: str) -> None:
        """
        Set the current language for recognition.
        
        Args:
            language: Language code to set as current
        """
        if language not in self.config.languages:
            raise ValueError(f"Language {language} not supported. Supported: {self.config.languages}")
        
        self.current_language = language
        self.logger.info(f"Current language set to {language}")
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        return self.config.languages.copy()
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """
        Get current preprocessing configuration.
        
        Returns:
            Dictionary of preprocessing parameters
        """
        return {
            'target_height': 32,
            'target_width': 128,
            'contrast_enhancement': True,
            'normalization': True,
            'padding': True
        }
    
    def update_preprocessing_config(self, **kwargs) -> None:
        """
        Update preprocessing configuration.
        
        Args:
            **kwargs: Preprocessing parameters to update
        """
        # This would update the preprocessor configuration
        # Implementation depends on specific preprocessing needs
        self.logger.info(f"Updating preprocessing config: {kwargs}")