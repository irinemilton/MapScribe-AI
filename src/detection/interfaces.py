"""
Abstract interfaces for text detection components.
"""

from abc import ABC, abstractmethod
from typing import List
from ..models.data_structures import ProcessedImage, Detection, BoundingBox


class TextDetectorInterface(ABC):
    """Abstract interface for text detection components."""
    
    @abstractmethod
    def detect_words(self, image: ProcessedImage) -> List[Detection]:
        """
        Detect individual word regions in the image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of word detections with bounding boxes
        """
        pass
    
    @abstractmethod
    def detect_phrases(self, image: ProcessedImage) -> List[Detection]:
        """
        Detect phrase regions in the image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of phrase detections with bounding boxes
        """
        pass
    
    @abstractmethod
    def get_confidence_scores(self, detections: List[Detection]) -> List[float]:
        """
        Get confidence scores for detections.
        
        Args:
            detections: List of detection results
            
        Returns:
            List of confidence scores
        """
        pass
    
    @abstractmethod
    def filter_detections(self, detections: List[Detection], threshold: float) -> List[Detection]:
        """
        Filter detections based on confidence threshold.
        
        Args:
            detections: List of detection results
            threshold: Minimum confidence threshold
            
        Returns:
            Filtered list of detections
        """
        pass


class DetectionModelInterface(ABC):
    """Abstract interface for detection model implementations."""
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """
        Load detection model from file.
        
        Args:
            model_path: Path to model file
        """
        pass
    
    @abstractmethod
    def predict(self, image: ProcessedImage) -> List[BoundingBox]:
        """
        Predict text regions in image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of bounding boxes
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary with model information
        """
        pass