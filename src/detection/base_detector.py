"""
Base text detection implementation with model loading and bounding box utilities.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import logging
from pathlib import Path

from ..models.data_structures import (
    ProcessedImage, Detection, BoundingBox, DetectionConfig, ModelMetadata
)
from .interfaces import TextDetectorInterface, DetectionModelInterface
from ..core.error_handling import (
    DetectionError, ModelLoadingError, ErrorContext, handle_errors
)
from ..core.graceful_degradation import with_graceful_degradation
from ..core.monitoring import monitor_performance


class TextDetector(TextDetectorInterface):
    """
    Base implementation of text detector with common functionality.
    
    This class provides the foundation for text detection with model management,
    bounding box processing, and validation utilities.
    """
    
    def __init__(self, config: DetectionConfig):
        """
        Initialize text detector with configuration.
        
        Args:
            config: Detection configuration parameters
        """
        self.config = config
        self.model: Optional[DetectionModelInterface] = None
        self.metadata: Optional[ModelMetadata] = None
        self.logger = logging.getLogger(__name__)
        
    @handle_errors(error_types=[FileNotFoundError, ValueError], log_errors=True)
    def load_model(self, model_path: str, model_class: type) -> None:
        """
        Load detection model from file.
        
        Args:
            model_path: Path to model file
            model_class: Class implementing DetectionModelInterface
            
        Raises:
            ModelLoadingError: If model loading fails
        """
        context = ErrorContext(
            component="text_detector",
            operation="load_model",
            input_data={"model_path": model_path, "model_class": model_class.__name__}
        )
        
        if not Path(model_path).exists():
            raise ModelLoadingError(
                f"Model file not found: {model_path}",
                model_path,
                context=context
            )
        
        try:
            self.model = model_class(self.config)
            self.model.load_model(model_path)
            self.metadata = ModelMetadata(
                model_name=Path(model_path).stem,
                version="1.0",
                architecture=self.config.model_architecture,
                input_size=self.config.input_size,
                supported_languages=["en", "zh"],
                training_dataset="synthetic",
                performance_metrics={}
            )
            self.logger.info(f"Successfully loaded model: {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model {model_path}: {str(e)}")
            raise ModelLoadingError(
                f"Model loading failed: {str(e)}",
                model_path,
                context=context
            )
    
    @monitor_performance("detection")
    @with_graceful_degradation("detection")
    def detect_words(self, image: ProcessedImage) -> List[Detection]:
        """
        Detect individual word regions in the image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of word detections with bounding boxes
            
        Raises:
            DetectionError: If detection fails
        """
        context = ErrorContext(
            component="text_detector",
            operation="detect_words",
            input_data={"image_shape": image.image.shape if hasattr(image, 'image') else None}
        )
        
        if self.model is None:
            raise DetectionError(
                "Model not loaded. Call load_model() first.",
                context=context
            )
        
        try:
            # Get raw predictions from model
            raw_boxes = self.model.predict(image)
            
            # Convert to Detection objects
            detections = []
            for bbox in raw_boxes:
                if bbox.confidence >= self.config.confidence_threshold:
                    detection = Detection(
                        bbox=bbox,
                        detection_type='word',
                        confidence=bbox.confidence,
                        image_coordinates=bbox.coordinates
                    )
                    detections.append(detection)
            
            # Apply Non-Maximum Suppression
            filtered_detections = self.apply_nms(detections, self.config.nms_threshold)
            
            self.logger.info(f"Detected {len(filtered_detections)} words from {len(raw_boxes)} raw predictions")
            return filtered_detections
            
        except Exception as e:
            self.logger.error(f"Word detection failed: {str(e)}")
            raise DetectionError(
                f"Word detection failed: {str(e)}",
                context=context
            )
    
    @monitor_performance("detection")
    @with_graceful_degradation("detection")
    def detect_phrases(self, image: ProcessedImage) -> List[Detection]:
        """
        Detect phrase regions in the image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of phrase detections with bounding boxes
        """
        context = ErrorContext(
            component="text_detector",
            operation="detect_phrases",
            input_data={"image_shape": image.image.shape if hasattr(image, 'image') else None}
        )
        
        try:
            # First detect words
            word_detections = self.detect_words(image)
            
            # Group words into phrases based on spatial proximity
            phrase_detections = self.group_words_to_phrases(word_detections)
            
            self.logger.info(f"Grouped {len(word_detections)} words into {len(phrase_detections)} phrases")
            return phrase_detections
            
        except Exception as e:
            self.logger.error(f"Phrase detection failed: {str(e)}")
            raise DetectionError(
                f"Phrase detection failed: {str(e)}",
                context=context
            )
    
    def get_confidence_scores(self, detections: List[Detection]) -> List[float]:
        """
        Get confidence scores for detections.
        
        Args:
            detections: List of detection results
            
        Returns:
            List of confidence scores
        """
        return [detection.confidence for detection in detections]
    
    def filter_detections(self, detections: List[Detection], threshold: float) -> List[Detection]:
        """
        Filter detections based on confidence threshold.
        
        Args:
            detections: List of detection results
            threshold: Minimum confidence threshold
            
        Returns:
            Filtered list of detections
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        filtered = [d for d in detections if d.confidence >= threshold]
        self.logger.info(f"Filtered {len(detections)} detections to {len(filtered)} with threshold {threshold}")
        return filtered
    
    def apply_nms(self, detections: List[Detection], nms_threshold: float) -> List[Detection]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            detections: List of detection results
            nms_threshold: IoU threshold for NMS
            
        Returns:
            Filtered list of detections after NMS
        """
        if not detections:
            return []
        
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("NMS threshold must be between 0.0 and 1.0")
        
        # Sort detections by confidence (descending)
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while sorted_detections:
            # Keep the detection with highest confidence
            current = sorted_detections.pop(0)
            keep.append(current)
            
            # Remove detections with high IoU overlap
            remaining = []
            for detection in sorted_detections:
                iou = self.calculate_iou(current.bbox, detection.bbox)
                if iou < nms_threshold:
                    remaining.append(detection)
            
            sorted_detections = remaining
        
        return keep
    
    def calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            IoU value between 0.0 and 1.0
        """
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1.coordinates
        x1_2, y1_2, x2_2, y2_2 = bbox2.coordinates
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = bbox1.area
        area2 = bbox2.area
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def group_words_to_phrases(self, word_detections: List[Detection]) -> List[Detection]:
        """
        Group word detections into phrase detections based on spatial proximity.
        
        Args:
            word_detections: List of word detection results
            
        Returns:
            List of phrase detections
        """
        if not word_detections:
            return []
        
        # Sort words by y-coordinate (top to bottom), then x-coordinate (left to right)
        sorted_words = sorted(word_detections, key=lambda d: (d.bbox.y, d.bbox.x))
        
        phrases = []
        current_phrase_words = [sorted_words[0]]
        
        for i in range(1, len(sorted_words)):
            current_word = sorted_words[i]
            last_word = current_phrase_words[-1]
            
            # Check if words are on the same line and close horizontally
            if self._are_words_in_same_phrase(last_word, current_word):
                current_phrase_words.append(current_word)
            else:
                # Create phrase from current group
                if current_phrase_words:
                    phrase = self._create_phrase_from_words(current_phrase_words)
                    phrases.append(phrase)
                
                # Start new phrase
                current_phrase_words = [current_word]
        
        # Add the last phrase
        if current_phrase_words:
            phrase = self._create_phrase_from_words(current_phrase_words)
            phrases.append(phrase)
        
        return phrases
    
    def _are_words_in_same_phrase(self, word1: Detection, word2: Detection) -> bool:
        """
        Determine if two words belong to the same phrase based on spatial proximity.
        
        Args:
            word1: First word detection
            word2: Second word detection
            
        Returns:
            True if words should be grouped in same phrase
        """
        # Calculate vertical overlap
        y1_1, y2_1 = word1.bbox.y, word1.bbox.y + word1.bbox.height
        y1_2, y2_2 = word2.bbox.y, word2.bbox.y + word2.bbox.height
        
        vertical_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        min_height = min(word1.bbox.height, word2.bbox.height)
        
        # Words should have significant vertical overlap (same line)
        if vertical_overlap < 0.5 * min_height:
            return False
        
        # Calculate horizontal distance
        x1_right = word1.bbox.x + word1.bbox.width
        x2_left = word2.bbox.x
        horizontal_gap = x2_left - x1_right
        
        # Words should be close horizontally (within reasonable gap)
        max_gap = max(word1.bbox.width, word2.bbox.width) * 0.5
        
        return horizontal_gap <= max_gap
    
    def _create_phrase_from_words(self, words: List[Detection]) -> Detection:
        """
        Create a phrase detection from a group of word detections.
        
        Args:
            words: List of word detections to group
            
        Returns:
            Phrase detection encompassing all words
        """
        if not words:
            raise ValueError("Cannot create phrase from empty word list")
        
        # Calculate bounding box that encompasses all words
        min_x = min(word.bbox.x for word in words)
        min_y = min(word.bbox.y for word in words)
        max_x = max(word.bbox.x + word.bbox.width for word in words)
        max_y = max(word.bbox.y + word.bbox.height for word in words)
        
        # Calculate average confidence
        avg_confidence = sum(word.confidence for word in words) / len(words)
        
        # Create phrase bounding box
        phrase_bbox = BoundingBox(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            confidence=avg_confidence
        )
        
        # Create phrase detection
        phrase_detection = Detection(
            bbox=phrase_bbox,
            detection_type='phrase',
            confidence=avg_confidence,
            image_coordinates=(min_x, min_y, max_x, max_y)
        )
        
        return phrase_detection
    
    def validate_bounding_box(self, bbox: BoundingBox, image_shape: Tuple[int, int]) -> bool:
        """
        Validate bounding box coordinates against image dimensions.
        
        Args:
            bbox: Bounding box to validate
            image_shape: Image dimensions (height, width)
            
        Returns:
            True if bounding box is valid
        """
        height, width = image_shape
        
        # Check if bounding box is within image bounds
        if bbox.x < 0 or bbox.y < 0:
            return False
        
        if bbox.x + bbox.width > width or bbox.y + bbox.height > height:
            return False
        
        # Check minimum size requirements
        if bbox.width < 5 or bbox.height < 5:
            return False
        
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "no_model_loaded"}
        
        info = {
            "status": "loaded",
            "config": self.config.to_dict(),
        }
        
        if self.metadata:
            info["metadata"] = self.metadata.to_dict()
        
        if hasattr(self.model, 'get_model_info'):
            info["model_details"] = self.model.get_model_info()
        
        return info


class BaseDetectionModel(DetectionModelInterface):
    """
    Base class for detection model implementations.
    
    This provides common functionality for model loading and prediction.
    """
    
    def __init__(self, config: DetectionConfig):
        """
        Initialize detection model with configuration.
        
        Args:
            config: Detection configuration parameters
        """
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
    
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary with model information
        """
        return {
            "architecture": self.config.model_architecture,
            "input_size": self.config.input_size,
            "confidence_threshold": self.config.confidence_threshold,
            "nms_threshold": self.config.nms_threshold,
            "batch_size": self.config.batch_size,
            "model_loaded": self.model is not None
        }
    
    def preprocess_image(self, image: ProcessedImage) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Processed image data
            
        Returns:
            Preprocessed image array
        """
        # Resize to model input size
        target_height, target_width = self.config.input_size
        
        if len(image.image.shape) == 3:
            # Color image
            resized = np.resize(image.image, (target_height, target_width, image.image.shape[2]))
        else:
            # Grayscale image
            resized = np.resize(image.image, (target_height, target_width))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def postprocess_predictions(self, predictions: np.ndarray, 
                              original_size: Tuple[int, int]) -> List[BoundingBox]:
        """
        Postprocess model predictions to bounding boxes.
        
        Args:
            predictions: Raw model predictions
            original_size: Original image size for coordinate scaling
            
        Returns:
            List of bounding boxes
        """
        # This is a placeholder implementation
        # Specific models will override this method
        bboxes = []
        
        # Scale coordinates back to original image size
        scale_x = original_size[1] / self.config.input_size[1]
        scale_y = original_size[0] / self.config.input_size[0]
        
        # Extract bounding boxes from predictions
        # This would be model-specific implementation
        
        return bboxes