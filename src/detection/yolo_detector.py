"""
YOLO-based text detection implementation for word-level detection.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from pathlib import Path

from ..models.data_structures import (
    ProcessedImage, BoundingBox, DetectionConfig
)
from .base_detector import BaseDetectionModel


class YOLODetectionModel(BaseDetectionModel):
    """
    YOLO-based text detection model implementation.
    
    This class implements word-level text detection using YOLO architecture
    with confidence thresholding and Non-Maximum Suppression.
    """
    
    def __init__(self, config: DetectionConfig):
        """
        Initialize YOLO detection model.
        
        Args:
            config: Detection configuration parameters
        """
        super().__init__(config)
        self.model_weights = None
        self.class_names = ['text']  # Single class for text detection
        self.anchors = None
        
    def load_model(self, model_path: str) -> None:
        """
        Load YOLO model from file.
        
        Args:
            model_path: Path to YOLO model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model loading fails
        """
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # In a real implementation, this would load actual YOLO weights
            # For now, we'll simulate model loading
            self.model_weights = self._load_mock_weights(model_path)
            self.model = True  # Mark as loaded
            
            self.logger.info(f"Successfully loaded YOLO model from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {str(e)}")
            raise ValueError(f"YOLO model loading failed: {str(e)}")
    
    def predict(self, image: ProcessedImage) -> List[BoundingBox]:
        """
        Predict text regions using YOLO model.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of bounding boxes with confidence scores
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("YOLO model not loaded")
        
        # Preprocess image for YOLO input
        processed_input = self.preprocess_image(image)
        
        # Run YOLO inference (mock implementation)
        raw_predictions = self._run_inference(processed_input)
        
        # Post-process predictions to bounding boxes
        bboxes = self.postprocess_predictions(raw_predictions, image.processed_size)
        
        # Apply confidence filtering
        filtered_bboxes = self._filter_by_confidence(bboxes)
        
        self.logger.info(f"YOLO detected {len(filtered_bboxes)} text regions from {len(bboxes)} raw predictions")
        
        return filtered_bboxes
    
    def _load_mock_weights(self, model_path: str) -> dict:
        """
        Mock implementation of weight loading.
        
        In a real implementation, this would load actual YOLO weights
        from formats like .pt, .onnx, or .weights files.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Mock weights dictionary
        """
        # Simulate loading weights based on file extension
        file_ext = Path(model_path).suffix.lower()
        
        if file_ext in ['.pt', '.pth']:
            # PyTorch format
            weights = {'format': 'pytorch', 'layers': 85}
        elif file_ext == '.onnx':
            # ONNX format
            weights = {'format': 'onnx', 'layers': 85}
        elif file_ext == '.weights':
            # Darknet format
            weights = {'format': 'darknet', 'layers': 85}
        else:
            weights = {'format': 'unknown', 'layers': 85}
        
        # Add mock anchor boxes for different scales
        weights['anchors'] = [
            # Small objects
            [[10, 13], [16, 30], [33, 23]],
            # Medium objects  
            [[30, 61], [62, 45], [59, 119]],
            # Large objects
            [[116, 90], [156, 198], [373, 326]]
        ]
        
        return weights
    
    def _run_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run YOLO inference on input tensor.
        
        This is a mock implementation. In a real scenario, this would:
        1. Pass input through YOLO network layers
        2. Apply activation functions
        3. Return raw network outputs
        
        Args:
            input_tensor: Preprocessed input image
            
        Returns:
            Raw YOLO predictions
        """
        batch_size = 1
        height, width = self.config.input_size
        
        # Mock YOLO output format: [batch, anchors, grid_h, grid_w, (x, y, w, h, conf, classes)]
        # For simplicity, using 3 scales with 3 anchors each
        scales = [8, 16, 32]  # Downsampling factors
        predictions = []
        
        for scale in scales:
            grid_h, grid_w = height // scale, width // scale
            num_anchors = 3
            num_classes = 1  # Just 'text' class
            
            # Output shape: [batch, anchors, grid_h, grid_w, 5 + num_classes]
            output_shape = (batch_size, num_anchors, grid_h, grid_w, 5 + num_classes)
            
            # Generate mock predictions with some realistic patterns
            scale_predictions = np.random.random(output_shape).astype(np.float32)
            
            # Make some predictions more confident (simulate actual text detections)
            if scale == 16:  # Medium scale - most text detections
                # Add some high-confidence predictions
                scale_predictions[0, :, 2:5, 3:8, 4] = 0.8  # High confidence
                scale_predictions[0, :, 2:5, 3:8, 5] = 0.9  # High class probability
            
            predictions.append(scale_predictions)
        
        return predictions
    
    def postprocess_predictions(self, predictions: List[np.ndarray], 
                              original_size: Tuple[int, int]) -> List[BoundingBox]:
        """
        Convert YOLO predictions to bounding boxes.
        
        Args:
            predictions: Raw YOLO predictions for each scale
            original_size: Original image size for coordinate scaling
            
        Returns:
            List of bounding boxes
        """
        all_boxes = []
        scales = [8, 16, 32]
        
        for i, (pred, scale) in enumerate(zip(predictions, scales)):
            boxes = self._decode_predictions(pred, scale, original_size)
            all_boxes.extend(boxes)
        
        return all_boxes
    
    def _decode_predictions(self, predictions: np.ndarray, scale: int,
                          original_size: Tuple[int, int]) -> List[BoundingBox]:
        """
        Decode YOLO predictions for a single scale.
        
        Args:
            predictions: Predictions for this scale [batch, anchors, grid_h, grid_w, outputs]
            scale: Downsampling scale factor
            original_size: Original image size
            
        Returns:
            List of decoded bounding boxes
        """
        batch_size, num_anchors, grid_h, grid_w, num_outputs = predictions.shape
        boxes = []
        
        # Get anchor boxes for this scale
        anchor_idx = {8: 0, 16: 1, 32: 2}[scale]
        anchors = self.model_weights['anchors'][anchor_idx]
        
        # Scale factors for converting to original image coordinates
        scale_x = original_size[1] / self.config.input_size[1]
        scale_y = original_size[0] / self.config.input_size[0]
        
        for anchor_idx in range(num_anchors):
            for grid_y in range(grid_h):
                for grid_x in range(grid_w):
                    # Extract prediction values
                    pred_box = predictions[0, anchor_idx, grid_y, grid_x]
                    
                    # YOLO output format: [x, y, w, h, confidence, class_prob]
                    x_offset, y_offset, w_pred, h_pred = pred_box[:4]
                    confidence = pred_box[4]
                    class_prob = pred_box[5] if num_outputs > 5 else 1.0
                    
                    # Apply sigmoid to x, y offsets and confidence
                    x_offset = self._sigmoid(x_offset)
                    y_offset = self._sigmoid(y_offset)
                    confidence = self._sigmoid(confidence)
                    class_prob = self._sigmoid(class_prob)
                    
                    # Calculate final confidence
                    final_confidence = confidence * class_prob
                    
                    # Skip low-confidence predictions
                    if final_confidence < 0.1:  # Very low threshold for raw predictions
                        continue
                    
                    # Convert to absolute coordinates
                    center_x = (grid_x + x_offset) * scale
                    center_y = (grid_y + y_offset) * scale
                    
                    # Apply exponential to width/height and multiply by anchors
                    width = anchors[anchor_idx][0] * np.exp(w_pred)
                    height = anchors[anchor_idx][1] * np.exp(h_pred)
                    
                    # Convert center coordinates to top-left coordinates
                    x = int((center_x - width / 2) * scale_x)
                    y = int((center_y - height / 2) * scale_y)
                    w = int(width * scale_x)
                    h = int(height * scale_y)
                    
                    # Ensure positive dimensions
                    if w > 0 and h > 0:
                        bbox = BoundingBox(
                            x=max(0, x),
                            y=max(0, y),
                            width=w,
                            height=h,
                            confidence=float(final_confidence)
                        )
                        boxes.append(bbox)
        
        return boxes
    
    def _filter_by_confidence(self, bboxes: List[BoundingBox]) -> List[BoundingBox]:
        """
        Filter bounding boxes by confidence threshold.
        
        Args:
            bboxes: List of bounding boxes
            
        Returns:
            Filtered list of bounding boxes
        """
        return [bbox for bbox in bboxes if bbox.confidence >= self.config.confidence_threshold]
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """Apply sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def get_model_info(self) -> dict:
        """
        Get YOLO model information.
        
        Returns:
            Dictionary with model information
        """
        base_info = super().get_model_info()
        
        yolo_info = {
            'model_type': 'YOLO',
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
        }
        
        if self.model_weights:
            yolo_info.update({
                'weights_format': self.model_weights.get('format', 'unknown'),
                'num_layers': self.model_weights.get('layers', 0),
                'anchor_boxes': len(self.model_weights.get('anchors', [])),
            })
        
        base_info.update(yolo_info)
        return base_info


class YOLOWordDetector:
    """
    High-level interface for YOLO-based word detection.
    
    This class provides a simplified interface for word detection
    with built-in confidence filtering and NMS.
    """
    
    def __init__(self, model_path: str, config: Optional[DetectionConfig] = None):
        """
        Initialize YOLO word detector.
        
        Args:
            model_path: Path to YOLO model file
            config: Optional detection configuration
        """
        if config is None:
            config = DetectionConfig(
                model_architecture='YOLO',
                input_size=(640, 640),
                confidence_threshold=0.5,
                nms_threshold=0.4,
                batch_size=1
            )
        
        self.config = config
        self.model = YOLODetectionModel(config)
        self.model.load_model(model_path)
        self.logger = logging.getLogger(__name__)
    
    def detect_words(self, image: ProcessedImage, 
                    confidence_threshold: Optional[float] = None,
                    nms_threshold: Optional[float] = None) -> List[BoundingBox]:
        """
        Detect words in image with optional parameter overrides.
        
        Args:
            image: Preprocessed image
            confidence_threshold: Override confidence threshold
            nms_threshold: Override NMS threshold
            
        Returns:
            List of word bounding boxes after filtering and NMS
        """
        # Override thresholds if provided
        original_conf = self.config.confidence_threshold
        original_nms = self.config.nms_threshold
        
        if confidence_threshold is not None:
            self.config.confidence_threshold = confidence_threshold
        if nms_threshold is not None:
            self.config.nms_threshold = nms_threshold
        
        try:
            # Get raw predictions
            raw_boxes = self.model.predict(image)
            
            # Apply Non-Maximum Suppression
            from .bbox_utils import NonMaximumSuppression
            filtered_boxes = NonMaximumSuppression.apply_nms(
                [self._bbox_to_detection(bbox) for bbox in raw_boxes],
                self.config.nms_threshold
            )
            
            # Convert back to bounding boxes
            result_boxes = [det.bbox for det in filtered_boxes]
            
            self.logger.info(f"Word detection: {len(raw_boxes)} -> {len(result_boxes)} after NMS")
            return result_boxes
            
        finally:
            # Restore original thresholds
            self.config.confidence_threshold = original_conf
            self.config.nms_threshold = original_nms
    
    def _bbox_to_detection(self, bbox: BoundingBox):
        """Convert BoundingBox to Detection for NMS processing."""
        from ..models.data_structures import Detection
        return Detection(
            bbox=bbox,
            detection_type='word',
            confidence=bbox.confidence,
            image_coordinates=bbox.coordinates
        )
    
    def get_performance_stats(self) -> dict:
        """
        Get performance statistics for the detector.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'model_info': self.model.get_model_info(),
            'config': self.config.to_dict(),
            'thresholds': {
                'confidence': self.config.confidence_threshold,
                'nms': self.config.nms_threshold
            }
        }