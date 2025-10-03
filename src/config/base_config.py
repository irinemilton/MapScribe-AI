"""
Base configuration classes for the historical map text processing system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
import json
import yaml
from pathlib import Path


@dataclass
class DetectionConfig:
    """Configuration for text detection models."""
    model_architecture: str = 'YOLO'  # 'YOLO', 'RCNN', 'CRAFT'
    model_path: str = ''
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    batch_size: int = 1
    device: str = 'cpu'  # 'cpu', 'cuda', 'mps'
    
    def __post_init__(self):
        """Validate detection configuration."""
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        if self.nms_threshold < 0.0 or self.nms_threshold > 1.0:
            raise ValueError("NMS threshold must be between 0.0 and 1.0")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")


@dataclass
class RecognitionConfig:
    """Configuration for text recognition models."""
    model_architecture: str = 'CRNN'  # 'CRNN', 'TrOCR', 'PARSeq'
    model_path: str = ''
    max_sequence_length: int = 100
    vocabulary_size: int = 10000
    languages: List[str] = field(default_factory=lambda: ['en', 'zh'])
    beam_width: int = 5
    device: str = 'cpu'
    
    def __post_init__(self):
        """Validate recognition configuration."""
        if self.max_sequence_length <= 0:
            raise ValueError("Max sequence length must be positive")
        if self.vocabulary_size <= 0:
            raise ValueError("Vocabulary size must be positive")
        if self.beam_width <= 0:
            raise ValueError("Beam width must be positive")


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""
    target_size: Tuple[int, int] = (1024, 1024)
    enhance_contrast: bool = True
    denoise: bool = True
    normalize: bool = True
    auto_rotate: bool = True
    color_space: str = 'RGB'  # 'RGB', 'GRAY', 'HSV'
    
    def __post_init__(self):
        """Validate preprocessing configuration."""
        if self.target_size[0] <= 0 or self.target_size[1] <= 0:
            raise ValueError("Target size must be positive")
        valid_color_spaces = ['RGB', 'GRAY', 'HSV', 'LAB']
        if self.color_space not in valid_color_spaces:
            raise ValueError(f"Color space must be one of {valid_color_spaces}")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    iou_threshold: float = 0.5
    confidence_threshold: float = 0.5
    calculate_character_accuracy: bool = True
    calculate_word_accuracy: bool = True
    save_detailed_results: bool = True
    output_format: str = 'json'  # 'json', 'csv', 'txt'
    
    def __post_init__(self):
        """Validate evaluation configuration."""
        if self.iou_threshold < 0.0 or self.iou_threshold > 1.0:
            raise ValueError("IoU threshold must be between 0.0 and 1.0")
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")


@dataclass
class SystemConfig:
    """Main system configuration."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # System-wide settings
    log_level: str = 'INFO'
    output_dir: str = 'outputs'
    temp_dir: str = 'temp'
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    
    def __post_init__(self):
        """Validate system configuration."""
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Log level must be one of {valid_log_levels}")
        if self.max_workers <= 0:
            raise ValueError("Max workers must be positive")
        if self.memory_limit_gb <= 0:
            raise ValueError("Memory limit must be positive")


class ConfigManagerInterface(ABC):
    """Abstract interface for configuration management."""
    
    @abstractmethod
    def load_config(self, config_path: str) -> SystemConfig:
        """Load configuration from file."""
        pass
    
    @abstractmethod
    def save_config(self, config: SystemConfig, config_path: str) -> None:
        """Save configuration to file."""
        pass
    
    @abstractmethod
    def validate_config(self, config: SystemConfig) -> bool:
        """Validate configuration parameters."""
        pass
    
    @abstractmethod
    def get_default_config(self) -> SystemConfig:
        """Get default system configuration."""
        pass