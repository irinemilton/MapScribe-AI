"""
Abstract interfaces for text recognition components.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from ..models.data_structures import Detection, RecognizedText


class TextRecognizerInterface(ABC):
    """Abstract interface for text recognition components."""
    
    @abstractmethod
    def recognize_words(self, image: np.ndarray, detections: List[Detection]) -> List[RecognizedText]:
        """
        Recognize text in detected word regions.
        
        Args:
            image: Input image array
            detections: List of word detections
            
        Returns:
            List of recognized text results
        """
        pass
    
    @abstractmethod
    def recognize_phrases(self, image: np.ndarray, detections: List[Detection]) -> List[RecognizedText]:
        """
        Recognize text in detected phrase regions.
        
        Args:
            image: Input image array
            detections: List of phrase detections
            
        Returns:
            List of recognized text results
        """
        pass
    
    @abstractmethod
    def handle_multilingual(self, text_region: np.ndarray) -> str:
        """
        Handle multi-language text recognition.
        
        Args:
            text_region: Image region containing text
            
        Returns:
            Recognized text string
        """
        pass
    
    @abstractmethod
    def detect_language(self, text_region: np.ndarray) -> str:
        """
        Detect the language of text in the region.
        
        Args:
            text_region: Image region containing text
            
        Returns:
            Detected language code
        """
        pass


class RecognitionModelInterface(ABC):
    """Abstract interface for recognition model implementations."""
    
    @abstractmethod
    def load_model(self, model_path: str, language: Optional[str] = None) -> None:
        """
        Load recognition model from file.
        
        Args:
            model_path: Path to model file
            language: Optional language specification
        """
        pass
    
    @abstractmethod
    def predict(self, text_region: np.ndarray) -> tuple[str, List[float]]:
        """
        Predict text and character confidences for a region.
        
        Args:
            text_region: Image region containing text
            
        Returns:
            Tuple of (recognized_text, character_confidences)
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        pass