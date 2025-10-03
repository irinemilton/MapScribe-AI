"""
Abstract interfaces for image processing components.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple
from ..models.data_structures import ProcessedImage


class ImageProcessorInterface(ABC):
    """Abstract interface for image preprocessing components."""
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> ProcessedImage:
        """
        Preprocess input image for text detection and recognition.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            ProcessedImage: Preprocessed image with metadata
        """
        pass
    
    @abstractmethod
    def enhance_text_visibility(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance text visibility in historical maps.
        
        Args:
            image: Input image array
            
        Returns:
            Enhanced image array
        """
        pass
    
    @abstractmethod
    def normalize_resolution(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Normalize image resolution to target size.
        
        Args:
            image: Input image array
            target_size: Target (width, height)
            
        Returns:
            Resized image array
        """
        pass
    
    @abstractmethod
    def correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct image orientation.
        
        Args:
            image: Input image array
            
        Returns:
            Orientation-corrected image array
        """
        pass