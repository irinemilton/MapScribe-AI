# -*- coding: utf-8 -*-
"""
Model management system for loading, versioning, and switching between models.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""
    version_id: str
    model_path: str
    metadata: Any
    created_at: datetime
    file_hash: str
    performance_metrics: Dict[str, float]
    is_active: bool = False
    
    def __post_init__(self):
        """Validate model version parameters."""
        if not self.version_id or not self.model_path:
            raise ValueError("Version ID and model path are required")
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = asdict(self)
        if hasattr(self.metadata, 'to_dict'):
            data['metadata'] = self.metadata.to_dict()
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Deserialize from dictionary."""
        from .data_structures import ModelMetadata
        
        metadata_data = data.pop('metadata')
        metadata = ModelMetadata.from_dict(metadata_data)
        created_at = datetime.fromisoformat(data.pop('created_at'))
        return cls(metadata=metadata, created_at=created_at, **data)


class ModelRegistry:
    """Registry for managing model versions and metadata."""
    
    def __init__(self, detection_models: Dict[str, List[ModelVersion]], 
                 recognition_models: Dict[str, List[ModelVersion]], 
                 registry_path: str):
        """Initialize model registry."""
        self.detection_models = detection_models
        self.recognition_models = recognition_models
        self.registry_path = registry_path
    
    def add_detection_model(self, model_name: str, version: ModelVersion):
        """Add a detection model version."""
        if model_name not in self.detection_models:
            self.detection_models[model_name] = []
        
        # If this is the first version, make it active
        if not self.detection_models[model_name]:
            version.is_active = True
        
        self.detection_models[model_name].append(version)
    
    def add_recognition_model(self, model_name: str, version: ModelVersion):
        """Add a recognition model version."""
        if model_name not in self.recognition_models:
            self.recognition_models[model_name] = []
        
        # If this is the first version, make it active
        if not self.recognition_models[model_name]:
            version.is_active = True
        
        self.recognition_models[model_name].append(version)
    
    def get_active_version(self, model_name: str, model_type: str) -> Optional[ModelVersion]:
        """Get the active version of a model."""
        models = self._get_models_dict(model_type)
        if model_name not in models:
            return None
        
        for version in models[model_name]:
            if version.is_active:
                return version
        return None
    
    def set_active_version(self, model_name: str, model_type: str, version_id: str):
        """Set the active version of a model."""
        models = self._get_models_dict(model_type)
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found")
        
        # Deactivate all versions
        for version in models[model_name]:
            version.is_active = False
        
        # Activate the specified version
        for version in models[model_name]:
            if version.version_id == version_id:
                version.is_active = True
                return
        
        raise ValueError(f"Version {version_id} not found for model {model_name}")
    
    def get_model_versions(self, model_name: str, model_type: str) -> List[ModelVersion]:
        """Get all versions of a model."""
        models = self._get_models_dict(model_type)
        return models.get(model_name, [])
    
    def _get_models_dict(self, model_type: str) -> Dict[str, List[ModelVersion]]:
        """Get the appropriate models dictionary."""
        if model_type == "detection":
            return self.detection_models
        elif model_type == "recognition":
            return self.recognition_models
        else:
            raise ValueError(f"Invalid model type: {model_type}")
    
    def save_registry(self):
        """Save registry to file."""
        registry_data = {
            "detection_models": {},
            "recognition_models": {}
        }
        
        for model_name, versions in self.detection_models.items():
            registry_data["detection_models"][model_name] = [v.to_dict() for v in versions]
        
        for model_name, versions in self.recognition_models.items():
            registry_data["recognition_models"][model_name] = [v.to_dict() for v in versions]
        
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    @classmethod
    def load_registry(cls, registry_path: str) -> 'ModelRegistry':
        """Load registry from file."""
        if not os.path.exists(registry_path):
            return cls({}, {}, registry_path)
        
        with open(registry_path, 'r') as f:
            registry_data = json.load(f)
        
        detection_models = {}
        for model_name, versions_data in registry_data.get("detection_models", {}).items():
            detection_models[model_name] = [ModelVersion.from_dict(v) for v in versions_data]
        
        recognition_models = {}
        for model_name, versions_data in registry_data.get("recognition_models", {}).items():
            recognition_models[model_name] = [ModelVersion.from_dict(v) for v in versions_data]
        
        return cls(detection_models, recognition_models, registry_path)


class ModelManager:
    """Manages model loading, versioning, and switching."""
    
    def __init__(self, models_dir: str = "models", registry_path: Optional[str] = None):
        """Initialize model manager."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        if registry_path is None:
            registry_path = str(self.models_dir / "registry.json")
        self.registry_path = registry_path
        
        # Load existing registry or create new one
        self.registry = ModelRegistry.load_registry(self.registry_path)
        
        # Cache for loaded models
        self.loaded_detection_models: Dict[str, Any] = {}
        self.loaded_recognition_models: Dict[str, Any] = {}
        
        logger.info(f"ModelManager initialized with models directory: {self.models_dir}")
    
    def register_detection_model(self, model_name: str, model_path: str, 
                               metadata: Any, config: Any,
                               version_id: Optional[str] = None) -> str:
        """Register a new detection model version."""
        if version_id is None:
            version_id = f"v{metadata.version}"
        
        # Calculate file hash for integrity checking
        file_hash = self.calculate_file_hash(model_path)
        
        # Create model version
        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            metadata=metadata,
            created_at=datetime.now(),
            file_hash=file_hash,
            performance_metrics=metadata.performance_metrics.copy()
        )
        
        # Add to registry
        self.registry.add_detection_model(model_name, version)
        
        # Save registry
        self.registry.save_registry()
        
        logger.info(f"Registered detection model {model_name} version {version_id}")
        return version_id
    
    def register_recognition_model(self, model_name: str, model_path: str,
                                 metadata: Any, config: Any,
                                 version_id: Optional[str] = None) -> str:
        """Register a new recognition model version."""
        if version_id is None:
            version_id = f"v{metadata.version}"
        
        # Calculate file hash for integrity checking
        file_hash = self.calculate_file_hash(model_path)
        
        # Create model version
        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            metadata=metadata,
            created_at=datetime.now(),
            file_hash=file_hash,
            performance_metrics=metadata.performance_metrics.copy()
        )
        
        # Add to registry
        self.registry.add_recognition_model(model_name, version)
        
        # Save registry
        self.registry.save_registry()
        
        logger.info(f"Registered recognition model {model_name} version {version_id}")
        return version_id
    
    def load_detection_model(self, model_name: str, version_id: Optional[str] = None) -> Any:
        """Load a detection model."""
        # Get the version to load
        if version_id is None:
            version = self.registry.get_active_version(model_name, "detection")
            if version is None:
                raise ValueError(f"No active version found for detection model {model_name}")
        else:
            versions = self.registry.get_model_versions(model_name, "detection")
            version = next((v for v in versions if v.version_id == version_id), None)
            if version is None:
                raise ValueError(f"Version {version_id} not found for model {model_name}")
        
        # Check if model is already loaded
        cache_key = f"{model_name}:{version.version_id}"
        if cache_key in self.loaded_detection_models:
            return self.loaded_detection_models[cache_key]
        
        # Load the model based on architecture
        model = self._load_model_by_architecture(version, "detection")
        
        # Cache the loaded model
        self.loaded_detection_models[cache_key] = model
        
        logger.info(f"Loaded detection model {model_name} version {version.version_id}")
        return model
    
    def load_recognition_model(self, model_name: str, version_id: Optional[str] = None) -> Any:
        """Load a recognition model."""
        # Get the version to load
        if version_id is None:
            version = self.registry.get_active_version(model_name, "recognition")
            if version is None:
                raise ValueError(f"No active version found for recognition model {model_name}")
        else:
            versions = self.registry.get_model_versions(model_name, "recognition")
            version = next((v for v in versions if v.version_id == version_id), None)
            if version is None:
                raise ValueError(f"Version {version_id} not found for model {model_name}")
        
        # Check if model is already loaded
        cache_key = f"{model_name}:{version.version_id}"
        if cache_key in self.loaded_recognition_models:
            return self.loaded_recognition_models[cache_key]
        
        # Load the model based on architecture
        model = self._load_model_by_architecture(version, "recognition")
        
        # Cache the loaded model
        self.loaded_recognition_models[cache_key] = model
        
        logger.info(f"Loaded recognition model {model_name} version {version.version_id}")
        return model
    
    def _load_model_by_architecture(self, version: ModelVersion, model_type: str) -> Any:
        """Load model based on its architecture."""
        architecture = version.metadata.architecture
        
        if model_type == "detection":
            if architecture == "YOLO":
                try:
                    from ..detection.yolo_detector import YOLODetector
                    from .data_structures import DetectionConfig
                    config = DetectionConfig(
                        model_architecture=architecture,
                        input_size=version.metadata.input_size,
                        confidence_threshold=0.5,
                        nms_threshold=0.4,
                        batch_size=1
                    )
                    return YOLODetector(version.model_path, config)
                except ImportError:
                    return f"MockYOLODetector({version.model_path})"
            else:
                raise ValueError(f"Unsupported detection architecture: {architecture}")
        
        elif model_type == "recognition":
            if architecture == "CRNN":
                try:
                    from ..recognition.crnn_recognizer import CRNNRecognizer
                    from .data_structures import RecognitionConfig
                    config = RecognitionConfig(
                        model_architecture=architecture,
                        max_sequence_length=25,
                        vocabulary_size=95,
                        languages=version.metadata.supported_languages,
                        beam_width=5
                    )
                    return CRNNRecognizer(version.model_path, config)
                except ImportError:
                    return f"MockCRNNRecognizer({version.model_path})"
            else:
                raise ValueError(f"Unsupported recognition architecture: {architecture}")
        
        else:
            raise ValueError(f"Invalid model type: {model_type}")
    
    def switch_model_version(self, model_name: str, model_type: str, version_id: str):
        """Switch the active version of a model."""
        self.registry.set_active_version(model_name, model_type, version_id)
        self.registry.save_registry()
        
        # Clear cached model if it exists
        if model_type == "detection":
            cache_keys_to_remove = [k for k in self.loaded_detection_models.keys() 
                                  if k.startswith(f"{model_name}:")]
            for key in cache_keys_to_remove:
                del self.loaded_detection_models[key]
        elif model_type == "recognition":
            cache_keys_to_remove = [k for k in self.loaded_recognition_models.keys() 
                                  if k.startswith(f"{model_name}:")]
            for key in cache_keys_to_remove:
                del self.loaded_recognition_models[key]
        
        logger.info(f"Switched {model_type} model {model_name} to version {version_id}")
    
    def get_model_metadata(self, model_name: str, model_type: str, 
                          version_id: Optional[str] = None) -> Any:
        """Get metadata for a specific model version."""
        if version_id is None:
            version = self.registry.get_active_version(model_name, model_type)
        else:
            versions = self.registry.get_model_versions(model_name, model_type)
            version = next((v for v in versions if v.version_id == version_id), None)
        
        if version is None:
            raise ValueError(f"Model version not found: {model_name}:{version_id}")
        
        return version.metadata
    
    def compare_model_performance(self, model_name: str, model_type: str) -> Dict[str, Dict[str, float]]:
        """Compare performance metrics across all versions of a model."""
        versions = self.registry.get_model_versions(model_name, model_type)
        
        comparison = {}
        for version in versions:
            comparison[version.version_id] = version.performance_metrics.copy()
        
        return comparison
    
    def get_best_performing_version(self, model_name: str, model_type: str, 
                                  metric: str) -> str:
        """Get the version with the best performance for a specific metric."""
        versions = self.registry.get_model_versions(model_name, model_type)
        
        if not versions:
            raise ValueError(f"No versions found for model {model_name}")
        
        best_version = None
        best_score = -1
        
        for version in versions:
            if metric in version.performance_metrics:
                score = version.performance_metrics[metric]
                if score > best_score:
                    best_score = score
                    best_version = version.version_id
        
        if best_version is None:
            raise ValueError(f"Metric {metric} not found in any version")
        
        return best_version
    
    def update_model_performance(self, model_name: str, model_type: str, 
                               version_id: str, new_metrics: Dict[str, float]):
        """Update performance metrics for a model version."""
        versions = self.registry.get_model_versions(model_name, model_type)
        version = next((v for v in versions if v.version_id == version_id), None)
        
        if version is None:
            raise ValueError(f"Version {version_id} not found for model {model_name}")
        
        # Update metrics
        version.performance_metrics.update(new_metrics)
        version.metadata.performance_metrics.update(new_metrics)
        
        # Save registry
        self.registry.save_registry()
        
        logger.info(f"Updated performance metrics for {model_name}:{version_id}")
    
    def list_available_models(self, model_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List all available models."""
        result = {}
        
        if model_type is None or model_type == "detection":
            result["detection"] = list(self.registry.detection_models.keys())
        
        if model_type is None or model_type == "recognition":
            result["recognition"] = list(self.registry.recognition_models.keys())
        
        return result
    
    def cleanup_old_versions(self, model_name: str, model_type: str, keep_count: int = 5):
        """Clean up old model versions, keeping only the most recent ones."""
        versions = self.registry.get_model_versions(model_name, model_type)
        
        if len(versions) <= keep_count:
            return  # Nothing to clean up
        
        # Sort by creation date (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)
        
        # Keep the most recent versions
        versions_to_keep = versions[:keep_count]
        versions_to_remove = versions[keep_count:]
        
        # Remove old versions
        models_dict = self.registry._get_models_dict(model_type)
        models_dict[model_name] = versions_to_keep
        
        # Delete model files for removed versions
        for version in versions_to_remove:
            try:
                if os.path.exists(version.model_path):
                    os.remove(version.model_path)
                logger.info(f"Removed old model file: {version.model_path}")
            except Exception as e:
                logger.warning(f"Failed to remove model file {version.model_path}: {e}")
        
        # Save updated registry
        self.registry.save_registry()
        
        logger.info(f"Cleaned up {len(versions_to_remove)} old versions of {model_name}")
    
    def export_model_info(self, export_path: str):
        """Export model information to a JSON file."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "detection_models": {},
            "recognition_models": {}
        }
        
        # Export detection models
        for model_name, versions in self.registry.detection_models.items():
            export_data["detection_models"][model_name] = []
            for version in versions:
                version_info = {
                    "version_id": version.version_id,
                    "metadata": version.metadata.to_dict() if hasattr(version.metadata, 'to_dict') else str(version.metadata),
                    "performance_metrics": version.performance_metrics,
                    "is_active": version.is_active,
                    "created_at": version.created_at.isoformat()
                }
                export_data["detection_models"][model_name].append(version_info)
        
        # Export recognition models
        for model_name, versions in self.registry.recognition_models.items():
            export_data["recognition_models"][model_name] = []
            for version in versions:
                version_info = {
                    "version_id": version.version_id,
                    "metadata": version.metadata.to_dict() if hasattr(version.metadata, 'to_dict') else str(version.metadata),
                    "performance_metrics": version.performance_metrics,
                    "is_active": version.is_active,
                    "created_at": version.created_at.isoformat()
                }
                export_data["recognition_models"][model_name].append(version_info)
        
        # Write to file
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported model information to {export_path}")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


print("Model manager module loaded successfully")