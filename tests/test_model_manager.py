"""
Tests for the ModelManager class and related components.
"""

import pytest
import tempfile
import shutil
import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.models.model_manager import ModelManager, ModelVersion, ModelRegistry
from src.models.data_structures import ModelMetadata, DetectionConfig, RecognitionConfig


class TestModelVersion:
    """Test ModelVersion dataclass."""
    
    def test_model_version_creation(self):
        """Test creating a ModelVersion instance."""
        # Create a temporary model file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"dummy model data")
            model_path = tmp_file.name
        
        try:
            metadata = ModelMetadata(
                model_name="test_model",
                version="1.0.0",
                architecture="YOLO",
                input_size=(640, 640),
                supported_languages=["en"],
                training_dataset="synthetic",
                performance_metrics={"f1_score": 0.85}
            )
            
            version = ModelVersion(
                version_id="v1.0.0",
                model_path=model_path,
                metadata=metadata,
                created_at=datetime.now(),
                file_hash="abc123",
                performance_metrics={"f1_score": 0.85}
            )
            
            assert version.version_id == "v1.0.0"
            assert version.model_path == model_path
            assert version.metadata.model_name == "test_model"
            assert not version.is_active
        finally:
            os.unlink(model_path)
    
    def test_model_version_validation(self):
        """Test ModelVersion validation."""
        metadata = ModelMetadata(
            model_name="test_model",
            version="1.0.0",
            architecture="YOLO",
            input_size=(640, 640),
            supported_languages=["en"],
            training_dataset="synthetic",
            performance_metrics={"f1_score": 0.85}
        )
        
        # Test missing version ID
        with pytest.raises(ValueError, match="Version ID and model path are required"):
            ModelVersion(
                version_id="",
                model_path="/nonexistent/path",
                metadata=metadata,
                created_at=datetime.now(),
                file_hash="abc123",
                performance_metrics={}
            )
        
        # Test nonexistent model file
        with pytest.raises(FileNotFoundError):
            ModelVersion(
                version_id="v1.0.0",
                model_path="/nonexistent/path",
                metadata=metadata,
                created_at=datetime.now(),
                file_hash="abc123",
                performance_metrics={}
            )
    
    def test_model_version_serialization(self):
        """Test ModelVersion serialization and deserialization."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"dummy model data")
            model_path = tmp_file.name
        
        try:
            metadata = ModelMetadata(
                model_name="test_model",
                version="1.0.0",
                architecture="YOLO",
                input_size=(640, 640),
                supported_languages=["en"],
                training_dataset="synthetic",
                performance_metrics={"f1_score": 0.85}
            )
            
            original_version = ModelVersion(
                version_id="v1.0.0",
                model_path=model_path,
                metadata=metadata,
                created_at=datetime.now(),
                file_hash="abc123",
                performance_metrics={"f1_score": 0.85}
            )
            
            # Test serialization
            version_dict = original_version.to_dict()
            assert version_dict["version_id"] == "v1.0.0"
            assert "created_at" in version_dict
            
            # Test deserialization
            restored_version = ModelVersion.from_dict(version_dict)
            assert restored_version.version_id == original_version.version_id
            assert restored_version.metadata.model_name == original_version.metadata.model_name
        finally:
            os.unlink(model_path)


class TestModelRegistry:
    """Test ModelRegistry class."""
    
    def test_registry_creation(self):
        """Test creating a ModelRegistry instance."""
        registry = ModelRegistry({}, {}, "/tmp/registry.json")
        assert registry.detection_models == {}
        assert registry.recognition_models == {}
        assert registry.registry_path == "/tmp/registry.json"
    
    def test_add_models(self):
        """Test adding models to registry."""
        registry = ModelRegistry({}, {}, "/tmp/registry.json")
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"dummy model data")
            model_path = tmp_file.name
        
        try:
            metadata = ModelMetadata(
                model_name="test_detector",
                version="1.0.0",
                architecture="YOLO",
                input_size=(640, 640),
                supported_languages=["en"],
                training_dataset="synthetic",
                performance_metrics={"f1_score": 0.85}
            )
            
            version = ModelVersion(
                version_id="v1.0.0",
                model_path=model_path,
                metadata=metadata,
                created_at=datetime.now(),
                file_hash="abc123",
                performance_metrics={"f1_score": 0.85}
            )
            
            registry.add_detection_model("test_detector", version)
            
            assert "test_detector" in registry.detection_models
            assert len(registry.detection_models["test_detector"]) == 1
            assert registry.detection_models["test_detector"][0].version_id == "v1.0.0"
        finally:
            os.unlink(model_path)
    
    def test_active_version_management(self):
        """Test active version management."""
        registry = ModelRegistry({}, {}, "/tmp/registry.json")
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"dummy model data")
            model_path = tmp_file.name
        
        try:
            metadata = ModelMetadata(
                model_name="test_detector",
                version="1.0.0",
                architecture="YOLO",
                input_size=(640, 640),
                supported_languages=["en"],
                training_dataset="synthetic",
                performance_metrics={"f1_score": 0.85}
            )
            
            version1 = ModelVersion(
                version_id="v1.0.0",
                model_path=model_path,
                metadata=metadata,
                created_at=datetime.now(),
                file_hash="abc123",
                performance_metrics={"f1_score": 0.85},
                is_active=True
            )
            
            version2 = ModelVersion(
                version_id="v2.0.0",
                model_path=model_path,
                metadata=metadata,
                created_at=datetime.now(),
                file_hash="def456",
                performance_metrics={"f1_score": 0.90}
            )
            
            registry.add_detection_model("test_detector", version1)
            registry.add_detection_model("test_detector", version2)
            
            # Test getting active version
            active = registry.get_active_version("test_detector", "detection")
            assert active.version_id == "v1.0.0"
            
            # Test switching active version
            registry.set_active_version("test_detector", "detection", "v2.0.0")
            active = registry.get_active_version("test_detector", "detection")
            assert active.version_id == "v2.0.0"
            
            # Verify old version is no longer active
            versions = registry.get_model_versions("test_detector", "detection")
            v1 = next(v for v in versions if v.version_id == "v1.0.0")
            assert not v1.is_active
        finally:
            os.unlink(model_path)
    
    def test_registry_persistence(self):
        """Test saving and loading registry."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = os.path.join(tmp_dir, "registry.json")
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(b"dummy model data")
                model_path = tmp_file.name
            
            try:
                # Create and populate registry
                registry = ModelRegistry({}, {}, registry_path)
                
                metadata = ModelMetadata(
                    model_name="test_detector",
                    version="1.0.0",
                    architecture="YOLO",
                    input_size=(640, 640),
                    supported_languages=["en"],
                    training_dataset="synthetic",
                    performance_metrics={"f1_score": 0.85}
                )
                
                version = ModelVersion(
                    version_id="v1.0.0",
                    model_path=model_path,
                    metadata=metadata,
                    created_at=datetime.now(),
                    file_hash="abc123",
                    performance_metrics={"f1_score": 0.85}
                )
                
                registry.add_detection_model("test_detector", version)
                registry.save_registry()
                
                # Load registry from file
                loaded_registry = ModelRegistry.load_registry(registry_path)
                
                assert "test_detector" in loaded_registry.detection_models
                assert len(loaded_registry.detection_models["test_detector"]) == 1
                assert loaded_registry.detection_models["test_detector"][0].version_id == "v1.0.0"
            finally:
                os.unlink(model_path)


class TestModelManager:
    """Test ModelManager class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.tmp_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.tmp_dir, "models")
        self.registry_path = os.path.join(self.models_dir, "registry.json")
        
        # Create a dummy model file
        self.model_file = tempfile.NamedTemporaryFile(delete=False)
        self.model_file.write(b"dummy model data")
        self.model_file.close()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.tmp_dir)
        os.unlink(self.model_file.name)
    
    def test_model_manager_initialization(self):
        """Test ModelManager initialization."""
        manager = ModelManager(self.models_dir, self.registry_path)
        
        assert manager.models_dir == Path(self.models_dir)
        assert manager.registry_path == self.registry_path
        assert isinstance(manager.registry, ModelRegistry)
        assert manager.loaded_detection_models == {}
        assert manager.loaded_recognition_models == {}
    
    def test_register_detection_model(self):
        """Test registering a detection model."""
        manager = ModelManager(self.models_dir, self.registry_path)
        
        metadata = ModelMetadata(
            model_name="yolo_detector",
            version="1.0.0",
            architecture="YOLO",
            input_size=(640, 640),
            supported_languages=["en"],
            training_dataset="synthetic",
            performance_metrics={"f1_score": 0.85}
        )
        
        config = DetectionConfig(
            model_architecture="YOLO",
            input_size=(640, 640),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            batch_size=1
        )
        
        version_id = manager.register_detection_model(
            "yolo_detector",
            self.model_file.name,
            metadata,
            config
        )
        
        assert version_id.startswith("v")
        assert "yolo_detector" in manager.registry.detection_models
        assert len(manager.registry.detection_models["yolo_detector"]) == 1
        
        # Verify the model is set as active (first version)
        active_version = manager.registry.get_active_version("yolo_detector", "detection")
        assert active_version is not None
        assert active_version.version_id == version_id
    
    def test_register_recognition_model(self):
        """Test registering a recognition model."""
        manager = ModelManager(self.models_dir, self.registry_path)
        
        metadata = ModelMetadata(
            model_name="crnn_recognizer",
            version="1.0.0",
            architecture="CRNN",
            input_size=(32, 128),
            supported_languages=["en"],
            training_dataset="synthetic",
            performance_metrics={"accuracy": 0.92}
        )
        
        config = RecognitionConfig(
            model_architecture="CRNN",
            max_sequence_length=25,
            vocabulary_size=95,
            languages=["en"],
            beam_width=5
        )
        
        version_id = manager.register_recognition_model(
            "crnn_recognizer",
            self.model_file.name,
            metadata,
            config
        )
        
        assert version_id.startswith("v")
        assert "crnn_recognizer" in manager.registry.recognition_models
        assert len(manager.registry.recognition_models["crnn_recognizer"]) == 1
    
    def test_model_switching(self):
        """Test switching between model versions."""
        manager = ModelManager(self.models_dir, self.registry_path)
        
        # Register two versions of the same model
        metadata1 = ModelMetadata(
            model_name="test_detector",
            version="1.0.0",
            architecture="YOLO",
            input_size=(640, 640),
            supported_languages=["en"],
            training_dataset="synthetic",
            performance_metrics={"f1_score": 0.85}
        )
        
        metadata2 = ModelMetadata(
            model_name="test_detector",
            version="2.0.0",
            architecture="YOLO",
            input_size=(640, 640),
            supported_languages=["en"],
            training_dataset="synthetic",
            performance_metrics={"f1_score": 0.90}
        )
        
        config = DetectionConfig(
            model_architecture="YOLO",
            input_size=(640, 640),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            batch_size=1
        )
        
        version1_id = manager.register_detection_model(
            "test_detector", self.model_file.name, metadata1, config, "v1.0.0"
        )
        version2_id = manager.register_detection_model(
            "test_detector", self.model_file.name, metadata2, config, "v2.0.0"
        )
        
        # First version should be active
        active = manager.registry.get_active_version("test_detector", "detection")
        assert active.version_id == version1_id
        
        # Switch to second version
        manager.switch_model_version("test_detector", "detection", version2_id)
        
        active = manager.registry.get_active_version("test_detector", "detection")
        assert active.version_id == version2_id
    
    def test_model_performance_comparison(self):
        """Test comparing model performance."""
        manager = ModelManager(self.models_dir, self.registry_path)
        
        # Register models with different performance
        metadata1 = ModelMetadata(
            model_name="test_detector",
            version="1.0.0",
            architecture="YOLO",
            input_size=(640, 640),
            supported_languages=["en"],
            training_dataset="synthetic",
            performance_metrics={"f1_score": 0.85, "precision": 0.80}
        )
        
        metadata2 = ModelMetadata(
            model_name="test_detector",
            version="2.0.0",
            architecture="YOLO",
            input_size=(640, 640),
            supported_languages=["en"],
            training_dataset="synthetic",
            performance_metrics={"f1_score": 0.90, "precision": 0.88}
        )
        
        config = DetectionConfig(
            model_architecture="YOLO",
            input_size=(640, 640),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            batch_size=1
        )
        
        manager.register_detection_model(
            "test_detector", self.model_file.name, metadata1, config, "v1.0.0"
        )
        manager.register_detection_model(
            "test_detector", self.model_file.name, metadata2, config, "v2.0.0"
        )
        
        # Compare performance
        comparison = manager.compare_model_performance("test_detector", "detection")
        
        assert "v1.0.0" in comparison
        assert "v2.0.0" in comparison
        assert comparison["v1.0.0"]["f1_score"] == 0.85
        assert comparison["v2.0.0"]["f1_score"] == 0.90
        
        # Test getting best performing version
        best_version = manager.get_best_performing_version("test_detector", "detection", "f1_score")
        assert best_version == "v2.0.0"
    
    def test_update_model_performance(self):
        """Test updating model performance metrics."""
        manager = ModelManager(self.models_dir, self.registry_path)
        
        metadata = ModelMetadata(
            model_name="test_detector",
            version="1.0.0",
            architecture="YOLO",
            input_size=(640, 640),
            supported_languages=["en"],
            training_dataset="synthetic",
            performance_metrics={"f1_score": 0.85}
        )
        
        config = DetectionConfig(
            model_architecture="YOLO",
            input_size=(640, 640),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            batch_size=1
        )
        
        version_id = manager.register_detection_model(
            "test_detector", self.model_file.name, metadata, config
        )
        
        # Update performance metrics
        new_metrics = {"f1_score": 0.88, "precision": 0.85, "recall": 0.91}
        manager.update_model_performance("test_detector", "detection", version_id, new_metrics)
        
        # Verify metrics were updated
        updated_metadata = manager.get_model_metadata("test_detector", "detection", version_id)
        assert updated_metadata.performance_metrics["f1_score"] == 0.88
        assert updated_metadata.performance_metrics["precision"] == 0.85
        assert updated_metadata.performance_metrics["recall"] == 0.91
    
    def test_list_available_models(self):
        """Test listing available models."""
        manager = ModelManager(self.models_dir, self.registry_path)
        
        # Register some models
        detection_metadata = ModelMetadata(
            model_name="yolo_detector",
            version="1.0.0",
            architecture="YOLO",
            input_size=(640, 640),
            supported_languages=["en"],
            training_dataset="synthetic",
            performance_metrics={"f1_score": 0.85}
        )
        
        recognition_metadata = ModelMetadata(
            model_name="crnn_recognizer",
            version="1.0.0",
            architecture="CRNN",
            input_size=(32, 128),
            supported_languages=["en"],
            training_dataset="synthetic",
            performance_metrics={"accuracy": 0.92}
        )
        
        detection_config = DetectionConfig(
            model_architecture="YOLO",
            input_size=(640, 640),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            batch_size=1
        )
        
        recognition_config = RecognitionConfig(
            model_architecture="CRNN",
            max_sequence_length=25,
            vocabulary_size=95,
            languages=["en"],
            beam_width=5
        )
        
        manager.register_detection_model(
            "yolo_detector", self.model_file.name, detection_metadata, detection_config
        )
        manager.register_recognition_model(
            "crnn_recognizer", self.model_file.name, recognition_metadata, recognition_config
        )
        
        # Test listing all models
        all_models = manager.list_available_models()
        assert "detection" in all_models
        assert "recognition" in all_models
        assert "yolo_detector" in all_models["detection"]
        assert "crnn_recognizer" in all_models["recognition"]
        
        # Test listing specific model type
        detection_models = manager.list_available_models("detection")
        assert "detection" in detection_models
        assert "recognition" not in detection_models
    
    def test_cleanup_old_versions(self):
        """Test cleaning up old model versions."""
        manager = ModelManager(self.models_dir, self.registry_path)
        
        config = DetectionConfig(
            model_architecture="YOLO",
            input_size=(640, 640),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            batch_size=1
        )
        
        # Register multiple versions
        for i in range(7):
            metadata = ModelMetadata(
                model_name="test_detector",
                version=f"{i+1}.0.0",
                architecture="YOLO",
                input_size=(640, 640),
                supported_languages=["en"],
                training_dataset="synthetic",
                performance_metrics={"f1_score": 0.8 + i * 0.01}
            )
            
            manager.register_detection_model(
                "test_detector", self.model_file.name, metadata, config, f"v{i+1}.0.0"
            )
        
        # Verify all versions are registered
        versions = manager.registry.get_model_versions("test_detector", "detection")
        assert len(versions) == 7
        
        # Cleanup old versions, keep only 3
        manager.cleanup_old_versions("test_detector", "detection", keep_count=3)
        
        # Verify only 3 versions remain
        versions = manager.registry.get_model_versions("test_detector", "detection")
        assert len(versions) == 3
        
        # Verify the most recent versions are kept
        version_ids = [v.version_id for v in versions]
        assert "v7.0.0" in version_ids
        assert "v6.0.0" in version_ids
        assert "v5.0.0" in version_ids
    
    def test_export_model_info(self):
        """Test exporting model information."""
        manager = ModelManager(self.models_dir, self.registry_path)
        
        # Register a model
        metadata = ModelMetadata(
            model_name="test_detector",
            version="1.0.0",
            architecture="YOLO",
            input_size=(640, 640),
            supported_languages=["en"],
            training_dataset="synthetic",
            performance_metrics={"f1_score": 0.85}
        )
        
        config = DetectionConfig(
            model_architecture="YOLO",
            input_size=(640, 640),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            batch_size=1
        )
        
        manager.register_detection_model(
            "test_detector", self.model_file.name, metadata, config
        )
        
        # Export model info
        export_path = os.path.join(self.tmp_dir, "model_info.json")
        manager.export_model_info(export_path)
        
        # Verify export file exists and contains expected data
        assert os.path.exists(export_path)
        
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        assert "export_timestamp" in exported_data
        assert "detection_models" in exported_data
        assert "recognition_models" in exported_data
        assert "test_detector" in exported_data["detection_models"]
    
    @patch('src.models.model_manager.YOLODetector')
    def test_load_detection_model_mock(self, mock_yolo):
        """Test loading detection model with mocked dependencies."""
        manager = ModelManager(self.models_dir, self.registry_path)
        
        # Setup mock
        mock_instance = Mock()
        mock_yolo.return_value = mock_instance
        
        # Register model
        metadata = ModelMetadata(
            model_name="yolo_detector",
            version="1.0.0",
            architecture="YOLO",
            input_size=(640, 640),
            supported_languages=["en"],
            training_dataset="synthetic",
            performance_metrics={"f1_score": 0.85}
        )
        
        config = DetectionConfig(
            model_architecture="YOLO",
            input_size=(640, 640),
            confidence_threshold=0.5,
            nms_threshold=0.4,
            batch_size=1
        )
        
        version_id = manager.register_detection_model(
            "yolo_detector", self.model_file.name, metadata, config
        )
        
        # Load model
        loaded_model = manager.load_detection_model("yolo_detector")
        
        # Verify mock was called correctly
        mock_yolo.assert_called_once_with(self.model_file.name, config)
        assert loaded_model == mock_instance
        
        # Verify model is cached
        assert f"yolo_detector:{version_id}" in manager.loaded_detection_models
    
    def test_file_hash_calculation(self):
        """Test file hash calculation."""
        manager = ModelManager(self.models_dir, self.registry_path)
        
        # Calculate hash of test file
        file_hash = manager.calculate_file_hash(self.model_file.name)
        
        assert isinstance(file_hash, str)
        assert len(file_hash) == 64  # SHA256 hash length
        
        # Verify hash is consistent
        file_hash2 = manager.calculate_file_hash(self.model_file.name)
        assert file_hash == file_hash2