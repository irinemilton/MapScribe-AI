"""
Main system interface for the historical map text processing system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

from ..models.data_structures import ProcessingResult, ProcessedImage, Detection, RecognizedText, LinkedText
from ..config.base_config import SystemConfig
from ..processing.interfaces import ImageProcessorInterface
from ..detection.interfaces import TextDetectorInterface
from ..recognition.interfaces import TextRecognizerInterface
from ..evaluation.interfaces import EvaluatorInterface


class HistoricalMapProcessorInterface(ABC):
    """Main interface for the historical map text processing system."""
    
    @abstractmethod
    def initialize(self, config: SystemConfig) -> None:
        """
        Initialize the system with configuration.
        
        Args:
            config: System configuration
        """
        pass
    
    @abstractmethod
    def process_image(self, image_path: str, task_type: str) -> ProcessingResult:
        """
        Process a single historical map image.
        
        Args:
            image_path: Path to input image
            task_type: Type of processing task
            
        Returns:
            ProcessingResult: Complete processing results
        """
        pass
    
    @abstractmethod
    def process_batch(self, image_paths: List[str], task_type: str) -> List[ProcessingResult]:
        """
        Process multiple historical map images.
        
        Args:
            image_paths: List of paths to input images
            task_type: Type of processing task
            
        Returns:
            List of processing results
        """
        pass
    
    @abstractmethod
    def evaluate_results(self, results: List[ProcessingResult], ground_truth_path: str) -> Dict[str, float]:
        """
        Evaluate processing results against ground truth.
        
        Args:
            results: List of processing results
            ground_truth_path: Path to ground truth annotations
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass


class ModelManagerInterface(ABC):
    """Interface for managing detection and recognition models."""
    
    @abstractmethod
    def load_detection_model(self, model_path: str, config: Dict[str, Any]) -> TextDetectorInterface:
        """
        Load and initialize detection model.
        
        Args:
            model_path: Path to detection model
            config: Model configuration
            
        Returns:
            Initialized text detector
        """
        pass
    
    @abstractmethod
    def load_recognition_model(self, model_path: str, config: Dict[str, Any]) -> TextRecognizerInterface:
        """
        Load and initialize recognition model.
        
        Args:
            model_path: Path to recognition model
            config: Model configuration
            
        Returns:
            Initialized text recognizer
        """
        pass
    
    @abstractmethod
    def switch_model_version(self, model_type: str, version: str) -> None:
        """
        Switch to different model version.
        
        Args:
            model_type: Type of model ('detection' or 'recognition')
            version: Model version identifier
        """
        pass
    
    @abstractmethod
    def get_model_metadata(self, model_type: str) -> Dict[str, Any]:
        """
        Get metadata for loaded model.
        
        Args:
            model_type: Type of model
            
        Returns:
            Model metadata dictionary
        """
        pass


class CompetitionFormatterInterface(ABC):
    """Interface for formatting results according to ICDAR 2025 competition requirements."""
    
    @abstractmethod
    def format_task1_output(self, detections: List[Detection]) -> str:
        """
        Format Task 1 (Word Detection) output.
        
        Args:
            detections: List of word detections
            
        Returns:
            Formatted output string
        """
        pass
    
    @abstractmethod
    def format_task2_output(self, detections: List[Detection]) -> str:
        """
        Format Task 2 (Phrase Detection) output.
        
        Args:
            detections: List of phrase detections
            
        Returns:
            Formatted output string
        """
        pass
    
    @abstractmethod
    def format_task3_output(self, results: List[RecognizedText]) -> str:
        """
        Format Task 3 (Word Detection + Recognition) output.
        
        Args:
            results: List of recognized words
            
        Returns:
            Formatted output string
        """
        pass
    
    @abstractmethod
    def format_task4_output(self, results: List[RecognizedText]) -> str:
        """
        Format Task 4 (Phrase Detection + Recognition) output.
        
        Args:
            results: List of recognized phrases
            
        Returns:
            Formatted output string
        """
        pass
    
    @abstractmethod
    def validate_output_format(self, output: str, task_type: str) -> bool:
        """
        Validate output format compliance.
        
        Args:
            output: Formatted output string
            task_type: Competition task type
            
        Returns:
            True if format is valid
        """
        pass


class TextLinkerInterface(ABC):
    """Interface for linking text to map coordinates and features."""
    
    @abstractmethod
    def link_to_coordinates(self, recognized_text: List[RecognizedText], image: np.ndarray) -> List[LinkedText]:
        """
        Link recognized text to map coordinates.
        
        Args:
            recognized_text: List of recognized text
            image: Original map image
            
        Returns:
            List of linked text with coordinates
        """
        pass
    
    @abstractmethod
    def group_related_text(self, linked_text: List[LinkedText]) -> List[List[LinkedText]]:
        """
        Group related text elements.
        
        Args:
            linked_text: List of linked text
            
        Returns:
            List of text groups
        """
        pass
    
    @abstractmethod
    def calculate_spatial_confidence(self, linkage: LinkedText) -> float:
        """
        Calculate confidence score for spatial linkage.
        
        Args:
            linkage: Linked text object
            
        Returns:
            Spatial confidence score
        """
        pass