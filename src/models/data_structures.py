"""
Core data structures for the historical map text processing system.
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Any
from abc import ABC, abstractmethod
import numpy as np
import json
import pickle
from pathlib import Path


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates and confidence."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    
    def __post_init__(self):
        """Validate bounding box coordinates and confidence."""
        self.validate_coordinates()
        self.validate_confidence()
    
    def validate_coordinates(self):
        """Validate coordinate bounds."""
        if self.x < 0 or self.y < 0:
            raise ValueError("Coordinates must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
    
    def validate_confidence(self):
        """Validate confidence score range."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    @property
    def area(self) -> int:
        """Calculate the area of the bounding box."""
        return self.width * self.height
    
    @property
    def coordinates(self) -> Tuple[int, int, int, int]:
        """Return coordinates as (x, y, x+width, y+height)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundingBox':
        """Deserialize from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BoundingBox':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Detection:
    """Base class for text detection results."""
    bbox: BoundingBox
    detection_type: str  # 'word' or 'phrase'
    confidence: float
    image_coordinates: Tuple[int, int, int, int]
    
    def __post_init__(self):
        """Validate detection parameters."""
        self.validate_confidence()
        self.validate_detection_type()
        self.validate_image_coordinates()
    
    def validate_confidence(self):
        """Validate confidence score range."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def validate_detection_type(self):
        """Validate detection type."""
        if self.detection_type not in ['word', 'phrase']:
            raise ValueError("Detection type must be 'word' or 'phrase'")
    
    def validate_image_coordinates(self):
        """Validate image coordinate bounds."""
        x1, y1, x2, y2 = self.image_coordinates
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid image coordinates")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = asdict(self)
        data['bbox'] = self.bbox.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Detection':
        """Deserialize from dictionary."""
        bbox_data = data.pop('bbox')
        bbox = BoundingBox.from_dict(bbox_data)
        return cls(bbox=bbox, **data)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Detection':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class RecognizedText:
    """Represents recognized text with confidence scores."""
    detection: Detection
    text: str
    language: str
    recognition_confidence: float
    character_confidences: List[float]
    
    def __post_init__(self):
        """Validate recognition parameters."""
        self.validate_confidence()
        self.validate_character_confidences()
        self.validate_language()
    
    def validate_confidence(self):
        """Validate recognition confidence score range."""
        if self.recognition_confidence < 0.0 or self.recognition_confidence > 1.0:
            raise ValueError("Recognition confidence must be between 0.0 and 1.0")
    
    def validate_character_confidences(self):
        """Validate character confidence scores."""
        if len(self.character_confidences) != len(self.text):
            raise ValueError("Character confidences must match text length")
        for conf in self.character_confidences:
            if conf < 0.0 or conf > 1.0:
                raise ValueError("Character confidence must be between 0.0 and 1.0")
    
    def validate_language(self):
        """Validate language code."""
        valid_languages = ['en', 'zh', 'latin', 'chinese', 'mixed']
        if self.language not in valid_languages:
            raise ValueError(f"Language must be one of {valid_languages}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = asdict(self)
        data['detection'] = self.detection.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecognizedText':
        """Deserialize from dictionary."""
        detection_data = data.pop('detection')
        detection = Detection.from_dict(detection_data)
        return cls(detection=detection, **data)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RecognizedText':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class LinkedText:
    """Represents text linked to map coordinates."""
    recognized_text: RecognizedText
    map_coordinates: Tuple[float, float]
    spatial_confidence: float
    related_features: List[str]
    
    def __post_init__(self):
        """Validate linking parameters."""
        self.validate_spatial_confidence()
        self.validate_map_coordinates()
    
    def validate_spatial_confidence(self):
        """Validate spatial confidence score range."""
        if self.spatial_confidence < 0.0 or self.spatial_confidence > 1.0:
            raise ValueError("Spatial confidence must be between 0.0 and 1.0")
    
    def validate_map_coordinates(self):
        """Validate map coordinate bounds."""
        x, y = self.map_coordinates
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError("Map coordinates must be numeric")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = asdict(self)
        data['recognized_text'] = self.recognized_text.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LinkedText':
        """Deserialize from dictionary."""
        recognized_text_data = data.pop('recognized_text')
        recognized_text = RecognizedText.from_dict(recognized_text_data)
        return cls(recognized_text=recognized_text, **data)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'LinkedText':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ProcessingResult:
    """Container for complete processing results."""
    task_type: str
    detections: List[Detection]
    recognized_texts: List[RecognizedText]
    linked_texts: List[LinkedText]
    processing_time: float
    metrics: Dict[str, float]
    
    def __post_init__(self):
        """Validate processing result parameters."""
        self.validate_task_type()
        self.validate_processing_time()
        self.validate_result_integrity()
    
    def validate_task_type(self):
        """Validate task type."""
        valid_tasks = ['word_detection', 'phrase_detection', 
                      'word_detection_recognition', 'phrase_detection_recognition']
        if self.task_type not in valid_tasks:
            raise ValueError(f"Task type must be one of {valid_tasks}")
    
    def validate_processing_time(self):
        """Validate processing time."""
        if self.processing_time < 0:
            raise ValueError("Processing time must be non-negative")
    
    def validate_result_integrity(self):
        """Validate integrity of processing results."""
        # Check that recognition results have corresponding detections
        if self.task_type in ['word_detection_recognition', 'phrase_detection_recognition']:
            if len(self.recognized_texts) > len(self.detections):
                raise ValueError("Cannot have more recognized texts than detections")
        
        # Check that linked texts have corresponding recognized texts
        if len(self.linked_texts) > len(self.recognized_texts):
            raise ValueError("Cannot have more linked texts than recognized texts")
        
        # Validate metrics
        for metric_name, value in self.metrics.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Metric {metric_name} must be numeric")
    
    def get_detection_count(self) -> int:
        """Get total number of detections."""
        return len(self.detections)
    
    def get_recognition_count(self) -> int:
        """Get total number of recognized texts."""
        return len(self.recognized_texts)
    
    def get_linking_count(self) -> int:
        """Get total number of linked texts."""
        return len(self.linked_texts)
    
    def get_average_confidence(self) -> float:
        """Calculate average detection confidence."""
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = asdict(self)
        data['detections'] = [d.to_dict() for d in self.detections]
        data['recognized_texts'] = [rt.to_dict() for rt in self.recognized_texts]
        data['linked_texts'] = [lt.to_dict() for lt in self.linked_texts]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingResult':
        """Deserialize from dictionary."""
        detections = [Detection.from_dict(d) for d in data.pop('detections', [])]
        recognized_texts = [RecognizedText.from_dict(rt) for rt in data.pop('recognized_texts', [])]
        linked_texts = [LinkedText.from_dict(lt) for lt in data.pop('linked_texts', [])]
        
        return cls(
            detections=detections,
            recognized_texts=recognized_texts,
            linked_texts=linked_texts,
            **data
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ProcessingResult':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def save_to_file(self, filepath: Path):
        """Save processing result to file."""
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                f.write(self.to_json())
        elif filepath.suffix == '.pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError("Unsupported file format. Use .json or .pkl")
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'ProcessingResult':
        """Load processing result from file."""
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                return cls.from_json(f.read())
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("Unsupported file format. Use .json or .pkl")


@dataclass
class ProcessedImage:
    """Container for preprocessed image data."""
    image: np.ndarray
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    preprocessing_steps: List[str]
    enhancement_applied: bool
    
    def __post_init__(self):
        """Validate processed image parameters."""
        if len(self.image.shape) not in [2, 3]:
            raise ValueError("Image must be 2D or 3D array")
        if self.processed_size[0] <= 0 or self.processed_size[1] <= 0:
            raise ValueError("Processed size must be positive")


@dataclass
class ModelMetadata:
    """Metadata for model information and versioning."""
    model_name: str
    version: str
    architecture: str
    input_size: Tuple[int, int]
    supported_languages: List[str]
    training_dataset: str
    performance_metrics: Dict[str, float]
    
    def __post_init__(self):
        """Validate model metadata."""
        self.validate_basic_info()
        self.validate_input_size()
        self.validate_performance_metrics()
    
    def validate_basic_info(self):
        """Validate basic model information."""
        if not self.model_name or not self.version:
            raise ValueError("Model name and version are required")
        if not self.architecture:
            raise ValueError("Architecture is required")
    
    def validate_input_size(self):
        """Validate input size parameters."""
        if self.input_size[0] <= 0 or self.input_size[1] <= 0:
            raise ValueError("Input size must be positive")
    
    def validate_performance_metrics(self):
        """Validate performance metrics."""
        for metric_name, value in self.performance_metrics.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Performance metric {metric_name} must be numeric")
            if metric_name in ['precision', 'recall', 'f1_score', 'accuracy'] and (value < 0 or value > 1):
                raise ValueError(f"Performance metric {metric_name} must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Deserialize from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ModelMetadata':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class DetectionConfig:
    """Configuration for text detection models."""
    model_architecture: str  # 'YOLO', 'RCNN', 'CRAFT'
    input_size: Tuple[int, int]
    confidence_threshold: float
    nms_threshold: float
    batch_size: int
    
    def __post_init__(self):
        """Validate detection configuration."""
        self.validate_architecture()
        self.validate_thresholds()
        self.validate_batch_size()
    
    def validate_architecture(self):
        """Validate model architecture."""
        valid_architectures = ['YOLO', 'RCNN', 'CRAFT', 'EAST', 'DBNet']
        if self.model_architecture not in valid_architectures:
            raise ValueError(f"Architecture must be one of {valid_architectures}")
        
        if self.input_size[0] <= 0 or self.input_size[1] <= 0:
            raise ValueError("Input size must be positive")
    
    def validate_thresholds(self):
        """Validate threshold parameters."""
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        if self.nms_threshold < 0.0 or self.nms_threshold > 1.0:
            raise ValueError("NMS threshold must be between 0.0 and 1.0")
    
    def validate_batch_size(self):
        """Validate batch size."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionConfig':
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class RecognitionConfig:
    """Configuration for text recognition models."""
    model_architecture: str  # 'CRNN', 'TrOCR', 'PARSeq'
    max_sequence_length: int
    vocabulary_size: int
    languages: List[str]
    beam_width: int
    
    def __post_init__(self):
        """Validate recognition configuration."""
        self.validate_architecture()
        self.validate_sequence_params()
        self.validate_languages()
    
    def validate_architecture(self):
        """Validate model architecture."""
        valid_architectures = ['CRNN', 'TrOCR', 'PARSeq', 'SVTR', 'ABINet']
        if self.model_architecture not in valid_architectures:
            raise ValueError(f"Architecture must be one of {valid_architectures}")
    
    def validate_sequence_params(self):
        """Validate sequence parameters."""
        if self.max_sequence_length <= 0:
            raise ValueError("Max sequence length must be positive")
        if self.vocabulary_size <= 0:
            raise ValueError("Vocabulary size must be positive")
        if self.beam_width <= 0:
            raise ValueError("Beam width must be positive")
    
    def validate_languages(self):
        """Validate language configuration."""
        valid_languages = ['en', 'zh', 'latin', 'chinese', 'mixed']
        for lang in self.languages:
            if lang not in valid_languages:
                raise ValueError(f"Language {lang} not supported. Valid languages: {valid_languages}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecognitionConfig':
        """Deserialize from dictionary."""
        return cls(**data)