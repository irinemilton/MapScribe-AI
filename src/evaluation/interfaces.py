"""
Abstract interfaces for evaluation and metrics components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..models.data_structures import Detection, RecognizedText, LinkedText, ProcessingResult


class EvaluatorInterface(ABC):
    """Abstract interface for evaluation components."""
    
    @abstractmethod
    def evaluate_detection(self, predictions: List[Detection], ground_truth: List[Detection]) -> Dict[str, float]:
        """
        Evaluate detection performance against ground truth.
        
        Args:
            predictions: Predicted detections
            ground_truth: Ground truth detections
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def evaluate_recognition(self, predictions: List[RecognizedText], ground_truth: List[str]) -> Dict[str, float]:
        """
        Evaluate recognition performance against ground truth.
        
        Args:
            predictions: Predicted recognized text
            ground_truth: Ground truth text strings
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def evaluate_linking(self, predictions: List[LinkedText], ground_truth: List[LinkedText]) -> Dict[str, float]:
        """
        Evaluate text linking performance against ground truth.
        
        Args:
            predictions: Predicted linked text
            ground_truth: Ground truth linked text
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def calculate_iou(self, box1: tuple, box2: tuple) -> float:
        """
        Calculate Intersection over Union for two bounding boxes.
        
        Args:
            box1: First bounding box (x1, y1, x2, y2)
            box2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU score
        """
        pass


class MetricsCalculatorInterface(ABC):
    """Abstract interface for metrics calculation."""
    
    @abstractmethod
    def calculate_precision_recall_f1(self, tp: int, fp: int, fn: int) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            tp: True positives
            fp: False positives
            fn: False negatives
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        pass
    
    @abstractmethod
    def calculate_character_accuracy(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate character-level accuracy.
        
        Args:
            predicted: Predicted text
            ground_truth: Ground truth text
            
        Returns:
            Character accuracy score
        """
        pass
    
    @abstractmethod
    def calculate_word_accuracy(self, predicted: List[str], ground_truth: List[str]) -> float:
        """
        Calculate word-level accuracy.
        
        Args:
            predicted: List of predicted words
            ground_truth: List of ground truth words
            
        Returns:
            Word accuracy score
        """
        pass