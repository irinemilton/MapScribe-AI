"""
Text detection components for word and phrase detection.
"""

from .interfaces import TextDetectorInterface, DetectionModelInterface
from .base_detector import TextDetector, BaseDetectionModel
from .bbox_utils import BoundingBoxUtils, NonMaximumSuppression
from .model_loader import ModelLoader, ModelValidator
from .yolo_detector import YOLODetectionModel, YOLOWordDetector
from .phrase_detector import PhraseDetector, PhraseGroupingConfig

__all__ = [
    'TextDetectorInterface',
    'DetectionModelInterface', 
    'TextDetector',
    'BaseDetectionModel',
    'BoundingBoxUtils',
    'NonMaximumSuppression',
    'ModelLoader',
    'ModelValidator',
    'YOLODetectionModel',
    'YOLOWordDetector',
    'PhraseDetector',
    'PhraseGroupingConfig'
]