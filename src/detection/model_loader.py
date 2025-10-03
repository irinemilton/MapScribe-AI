"""
Model loading and initialization utilities for text detection.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Type
from abc import ABC, abstractmethod

from ..models.data_structures import DetectionConfig, ModelMetadata
from .interfaces import DetectionModelInterface


class ModelLoader:
    """Utility class for loading and managing detection models."""
    
    def __init__(self):
        """Initialize model loader."""
        self.logger = logging.getLogger(__name__)
        self._model_registry: Dict[str, Type[DetectionModelInterface]] = {}
    
    def register_model(self, architecture: str, model_class: Type[DetectionModelInterface]) -> None:
        """
        Register a model class for a specific architecture.
        
        Args:
            architecture: Model architecture name (e.g., 'YOLO', 'RCNN')
            model_class: Class implementing DetectionModelInterface
        """
        self._model_registry[architecture] = model_class
        self.logger.info(f"Registered model class for architecture: {architecture}")
    
    def load_model(self, model_path: str, config: DetectionConfig) -> DetectionModelInterface:
        """
        Load a detection model from file.
        
        Args:
            model_path: Path to model file
            config: Detection configuration
            
        Returns:
            Loaded model instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If architecture not supported or loading fails
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        architecture = config.model_architecture
        if architecture not in self._model_registry:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        try:
            model_class = self._model_registry[architecture]
            model = model_class(config)
            model.load_model(model_path)
            
            self.logger.info(f"Successfully loaded {architecture} model from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_path}: {str(e)}")
            raise ValueError(f"Model loading failed: {str(e)}")
    
    def get_supported_architectures(self) -> list:
        """Get list of supported model architectures."""
        return list(self._model_registry.keys())


class ModelValidator:
    """Utility class for validating model files and configurations."""
    
    @staticmethod
    def validate_model_file(model_path: str) -> bool:
        """
        Validate that model file exists and is readable.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if model file is valid
        """
        path = Path(model_path)
        
        if not path.exists():
            return False
        
        if not path.is_file():
            return False
        
        if not os.access(path, os.R_OK):
            return False
        
        return True
    
    @staticmethod
    def validate_config_compatibility(config: DetectionConfig, 
                                    model_metadata: Optional[ModelMetadata] = None) -> bool:
        """
        Validate configuration compatibility with model.
        
        Args:
            config: Detection configuration
            model_metadata: Optional model metadata for validation
            
        Returns:
            True if configuration is compatible
        """
        # Basic config validation
        if config.confidence_threshold < 0.0 or config.confidence_threshold > 1.0:
            return False
        
        if config.nms_threshold < 0.0 or config.nms_threshold > 1.0:
            return False
        
        if config.batch_size <= 0:
            return False
        
        if config.input_size[0] <= 0 or config.input_size[1] <= 0:
            return False
        
        # Model-specific validation if metadata available
        if model_metadata:
            # Check input size compatibility
            if hasattr(model_metadata, 'input_size'):
                if config.input_size != model_metadata.input_size:
                    return False
        
        return True