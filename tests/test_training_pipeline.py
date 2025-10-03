"""
Tests for the training pipeline components.
"""

import pytest
import tempfile
import shutil
import os
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from torch.utils.data import Dataset

from src.models.training_pipeline import (
    TrainingConfig, TrainingMetrics, DataAugmentation, EarlyStopping,
    DetectionTrainingPipeline, RecognitionTrainingPipeline, TrainingManager
)


class MockDataset(Dataset):
    """Mock dataset for testing."""
    
    def __init__(self, size: int = 100, image_shape: tuple = (3, 224, 224), 
                 num_classes: int = 10):
        self.size = size
        self.image_shape = image_shape
        self.num_classes = num_classes
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random image
        image = torch.randn(*self.image_shape)
        
        # Generate random target (simplified)
        target = torch.randn(self.num_classes)
        
        return image, target


class MockRecognitionDataset(Dataset):
    """Mock dataset for recognition testing."""
    
    def __init__(self, size: int = 100, image_shape: tuple = (3, 32, 128), 
                 vocab_size: int = 95, max_length: int = 25):
        self.size = size
        self.image_shape = image_shape
        self.vocab_size = vocab_size
        self.max_length = max_length
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random image
        image = torch.randn(*self.image_shape)
        
        # Generate random text tokens
        length = torch.randint(5, self.max_length, (1,)).item()
        text_tokens = torch.randint(1, self.vocab_size, (self.max_length,))
        text_tokens[length:] = 0  # Padding
        
        return image, text_tokens, length


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""
    
    def test_training_config_creation(self):
        """Test creating a TrainingConfig instance."""
        config = TrainingConfig(
            model_name="test_model",
            model_type="detection",
            architecture="YOLO",
            learning_rate=0.001,
            batch_size=16,
            num_epochs=100
        )
        
        assert config.model_name == "test_model"
        assert config.model_type == "detection"
        assert config.architecture == "YOLO"
        assert config.learning_rate == 0.001
        assert config.batch_size == 16
        assert config.num_epochs == 100
    
    def test_training_config_serialization(self):
        """Test TrainingConfig serialization and deserialization."""
        config = TrainingConfig(
            model_name="test_model",
            model_type="detection",
            architecture="YOLO"
        )
        
        # Test serialization
        config_dict = config.to_dict()
        assert config_dict["model_name"] == "test_model"
        assert config_dict["model_type"] == "detection"
        
        # Test deserialization
        restored_config = TrainingConfig.from_dict(config_dict)
        assert restored_config.model_name == config.model_name
        assert restored_config.model_type == config.model_type
        assert restored_config.architecture == config.architecture


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""
    
    def test_training_metrics_creation(self):
        """Test creating a TrainingMetrics instance."""
        metrics = TrainingMetrics(
            epoch=10,
            train_loss=0.5,
            val_loss=0.6,
            train_accuracy=0.85,
            val_accuracy=0.80,
            learning_rate=0.001,
            epoch_time=120.5
        )
        
        assert metrics.epoch == 10
        assert metrics.train_loss == 0.5
        assert metrics.val_loss == 0.6
        assert metrics.train_accuracy == 0.85
        assert metrics.val_accuracy == 0.80
        assert metrics.learning_rate == 0.001
        assert metrics.epoch_time == 120.5
    
    def test_training_metrics_serialization(self):
        """Test TrainingMetrics serialization."""
        metrics = TrainingMetrics(
            epoch=5,
            train_loss=0.3,
            val_loss=0.4
        )
        
        metrics_dict = metrics.to_dict()
        assert metrics_dict["epoch"] == 5
        assert metrics_dict["train_loss"] == 0.3
        
        restored_metrics = TrainingMetrics.from_dict(metrics_dict)
        assert restored_metrics.epoch == metrics.epoch
        assert restored_metrics.train_loss == metrics.train_loss


class TestDataAugmentation:
    """Test DataAugmentation class."""
    
    def test_data_augmentation_initialization(self):
        """Test DataAugmentation initialization."""
        config = TrainingConfig(
            model_name="test",
            model_type="detection",
            architecture="YOLO",
            data_augmentation=True
        )
        
        augmentation = DataAugmentation(config)
        assert augmentation.enabled == True
        assert augmentation.config == config
    
    def test_detection_data_augmentation(self):
        """Test detection data augmentation."""
        config = TrainingConfig(
            model_name="test",
            model_type="detection",
            architecture="YOLO",
            data_augmentation=True
        )
        
        augmentation = DataAugmentation(config)
        
        # Create test image and bboxes
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        bboxes = [(10, 10, 50, 50), (100, 100, 150, 150)]
        
        # Apply augmentation
        aug_image, aug_bboxes = augmentation.augment_detection_data(image, bboxes)
        
        # Check that output has correct shape and type
        assert aug_image.shape == image.shape
        assert aug_image.dtype == np.uint8
        assert len(aug_bboxes) == len(bboxes)
        assert all(len(bbox) == 4 for bbox in aug_bboxes)
    
    def test_recognition_data_augmentation(self):
        """Test recognition data augmentation."""
        config = TrainingConfig(
            model_name="test",
            model_type="recognition",
            architecture="CRNN",
            data_augmentation=True
        )
        
        augmentation = DataAugmentation(config)
        
        # Create test image and text
        image = np.random.randint(0, 255, (32, 128, 3), dtype=np.uint8)
        text = "test_text"
        
        # Apply augmentation
        aug_image, aug_text = augmentation.augment_recognition_data(image, text)
        
        # Check that output has correct shape and type
        assert aug_image.shape == image.shape
        assert aug_image.dtype == np.uint8
        assert aug_text == text  # Text should remain unchanged
    
    def test_disabled_augmentation(self):
        """Test that augmentation is disabled when configured."""
        config = TrainingConfig(
            model_name="test",
            model_type="detection",
            architecture="YOLO",
            data_augmentation=False
        )
        
        augmentation = DataAugmentation(config)
        
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        bboxes = [(10, 10, 50, 50)]
        
        # Apply augmentation (should return unchanged)
        aug_image, aug_bboxes = augmentation.augment_detection_data(image, bboxes)
        
        np.testing.assert_array_equal(aug_image, image)
        assert aug_bboxes == bboxes


class TestEarlyStopping:
    """Test EarlyStopping class."""
    
    def test_early_stopping_initialization(self):
        """Test EarlyStopping initialization."""
        early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode='min')
        
        assert early_stopping.patience == 5
        assert early_stopping.min_delta == 0.001
        assert early_stopping.mode == 'min'
        assert early_stopping.best_score is None
        assert early_stopping.counter == 0
        assert early_stopping.early_stop == False
    
    def test_early_stopping_improvement(self):
        """Test early stopping with improving scores."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')
        
        # First score
        assert early_stopping(1.0) == False
        assert early_stopping.counter == 0
        
        # Improving score
        assert early_stopping(0.8) == False
        assert early_stopping.counter == 0
        
        # Another improving score
        assert early_stopping(0.6) == False
        assert early_stopping.counter == 0
    
    def test_early_stopping_no_improvement(self):
        """Test early stopping with no improvement."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, mode='min')
        
        # Initial score
        early_stopping(1.0)
        
        # No improvement
        assert early_stopping(1.0) == False
        assert early_stopping.counter == 1
        
        # Still no improvement
        assert early_stopping(1.0) == False
        assert early_stopping.counter == 2
        
        # Should trigger early stopping
        assert early_stopping(1.0) == True
        assert early_stopping.early_stop == True
    
    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, mode='max')
        
        # Initial score
        early_stopping(0.5)
        
        # Improving score (higher is better)
        assert early_stopping(0.7) == False
        assert early_stopping.counter == 0
        
        # No improvement
        assert early_stopping(0.7) == False
        assert early_stopping.counter == 1


class TestDetectionTrainingPipeline:
    """Test DetectionTrainingPipeline class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.tmp_dir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            model_name="test_detector",
            model_type="detection",
            architecture="YOLO",
            num_epochs=2,
            batch_size=4,
            checkpoint_interval=1
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.tmp_dir)
    
    def test_pipeline_initialization(self):
        """Test DetectionTrainingPipeline initialization."""
        pipeline = DetectionTrainingPipeline(self.config, self.tmp_dir)
        
        assert pipeline.config == self.config
        assert pipeline.output_dir == Path(self.tmp_dir)
        assert pipeline.device in [torch.device('cuda'), torch.device('cpu')]
        assert pipeline.training_history == []
        assert pipeline.best_val_loss == float('inf')
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_model_setup_yolo(self, mock_cuda):
        """Test YOLO model setup."""
        pipeline = DetectionTrainingPipeline(self.config, self.tmp_dir)
        
        model = pipeline.setup_model("YOLO", num_classes=1)
        
        assert pipeline.model is not None
        assert pipeline.optimizer is not None
        assert pipeline.scheduler is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_model_setup_unsupported(self):
        """Test unsupported model architecture."""
        pipeline = DetectionTrainingPipeline(self.config, self.tmp_dir)
        
        with pytest.raises(ValueError, match="Unsupported detection architecture"):
            pipeline.setup_model("UNSUPPORTED")
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_data_loader_setup(self, mock_cuda):
        """Test data loader setup."""
        pipeline = DetectionTrainingPipeline(self.config, self.tmp_dir)
        
        train_dataset = MockDataset(size=20)
        val_dataset = MockDataset(size=10)
        
        pipeline.setup_data_loaders(train_dataset, val_dataset)
        
        assert pipeline.train_loader is not None
        assert pipeline.val_loader is not None
        assert len(pipeline.train_loader.dataset) == 20
        assert len(pipeline.val_loader.dataset) == 10
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_checkpoint_save_load(self, mock_cuda):
        """Test checkpoint saving and loading."""
        pipeline = DetectionTrainingPipeline(self.config, self.tmp_dir)
        
        # Setup model
        pipeline.setup_model("YOLO", num_classes=1)
        
        # Add some training history
        metrics = TrainingMetrics(epoch=0, train_loss=0.5, val_loss=0.6)
        pipeline.training_history.append(metrics)
        
        # Save checkpoint
        pipeline.save_checkpoint(epoch=0, is_best=True)
        
        # Check that files were created
        checkpoint_path = pipeline.output_dir / "checkpoint_epoch_0.pth"
        best_path = pipeline.output_dir / "best_model.pth"
        
        assert checkpoint_path.exists()
        assert best_path.exists()
        
        # Test loading
        loaded_epoch = pipeline.load_checkpoint(str(checkpoint_path))
        assert loaded_epoch == 0
        assert len(pipeline.training_history) == 1
    
    @patch('torch.cuda.is_available', return_value=False)
    @patch('src.models.training_pipeline.SummaryWriter')
    def test_training_loop(self, mock_writer, mock_cuda):
        """Test basic training loop."""
        # Use very small config for fast testing
        config = TrainingConfig(
            model_name="test_detector",
            model_type="detection",
            architecture="YOLO",
            num_epochs=1,
            batch_size=2,
            checkpoint_interval=1
        )
        
        pipeline = DetectionTrainingPipeline(config, self.tmp_dir)
        
        # Setup model
        pipeline.setup_model("YOLO", num_classes=1)
        
        # Create small datasets
        train_dataset = MockDataset(size=4)
        val_dataset = MockDataset(size=2)
        
        # Run training
        best_model_path = pipeline.train(train_dataset, val_dataset)
        
        # Check that training completed
        assert best_model_path is not None
        assert len(pipeline.training_history) == 1
        assert pipeline.training_history[0].epoch == 0


class TestRecognitionTrainingPipeline:
    """Test RecognitionTrainingPipeline class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.tmp_dir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            model_name="test_recognizer",
            model_type="recognition",
            architecture="CRNN",
            num_epochs=2,
            batch_size=4,
            checkpoint_interval=1
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.tmp_dir)
    
    def test_pipeline_initialization(self):
        """Test RecognitionTrainingPipeline initialization."""
        pipeline = RecognitionTrainingPipeline(self.config, self.tmp_dir)
        
        assert pipeline.config == self.config
        assert pipeline.output_dir == Path(self.tmp_dir)
        assert pipeline.training_history == []
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_model_setup_crnn(self, mock_cuda):
        """Test CRNN model setup."""
        pipeline = RecognitionTrainingPipeline(self.config, self.tmp_dir)
        
        model = pipeline.setup_model("CRNN", vocab_size=95)
        
        assert pipeline.model is not None
        assert pipeline.optimizer is not None
        assert isinstance(model, torch.nn.Module)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_model_setup_trocr(self, mock_cuda):
        """Test TrOCR model setup."""
        config = TrainingConfig(
            model_name="test_recognizer",
            model_type="recognition",
            architecture="TrOCR"
        )
        
        pipeline = RecognitionTrainingPipeline(config, self.tmp_dir)
        
        model = pipeline.setup_model("TrOCR", vocab_size=95)
        
        assert pipeline.model is not None
        assert isinstance(model, torch.nn.Module)
    
    @patch('torch.cuda.is_available', return_value=False)
    @patch('src.models.training_pipeline.SummaryWriter')
    def test_training_loop(self, mock_writer, mock_cuda):
        """Test basic training loop."""
        # Use very small config for fast testing
        config = TrainingConfig(
            model_name="test_recognizer",
            model_type="recognition",
            architecture="CRNN",
            num_epochs=1,
            batch_size=2,
            checkpoint_interval=1
        )
        
        pipeline = RecognitionTrainingPipeline(config, self.tmp_dir)
        
        # Setup model
        pipeline.setup_model("CRNN", vocab_size=95)
        
        # Create small datasets
        train_dataset = MockRecognitionDataset(size=4)
        val_dataset = MockRecognitionDataset(size=2)
        
        # Run training
        best_model_path = pipeline.train(train_dataset, val_dataset)
        
        # Check that training completed
        assert best_model_path is not None
        assert len(pipeline.training_history) == 1


class TestTrainingManager:
    """Test TrainingManager class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.tmp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.tmp_dir)
    
    def test_training_manager_initialization(self):
        """Test TrainingManager initialization."""
        manager = TrainingManager(self.tmp_dir)
        
        assert manager.base_output_dir == Path(self.tmp_dir)
        assert manager.active_trainings == {}
    
    def test_create_detection_pipeline(self):
        """Test creating detection pipeline."""
        manager = TrainingManager(self.tmp_dir)
        
        config = TrainingConfig(
            model_name="test_detector",
            model_type="detection",
            architecture="YOLO"
        )
        
        pipeline = manager.create_detection_pipeline(config)
        
        assert isinstance(pipeline, DetectionTrainingPipeline)
        assert "detection_test_detector" in manager.active_trainings
    
    def test_create_recognition_pipeline(self):
        """Test creating recognition pipeline."""
        manager = TrainingManager(self.tmp_dir)
        
        config = TrainingConfig(
            model_name="test_recognizer",
            model_type="recognition",
            architecture="CRNN"
        )
        
        pipeline = manager.create_recognition_pipeline(config)
        
        assert isinstance(pipeline, RecognitionTrainingPipeline)
        assert "recognition_test_recognizer" in manager.active_trainings
    
    def test_get_training_status(self):
        """Test getting training status."""
        manager = TrainingManager(self.tmp_dir)
        
        # Test non-existent training
        status = manager.get_training_status("nonexistent", "detection")
        assert status["status"] == "not_found"
        
        # Create a pipeline
        config = TrainingConfig(
            model_name="test_detector",
            model_type="detection",
            architecture="YOLO"
        )
        
        pipeline = manager.create_detection_pipeline(config)
        
        # Test not started
        status = manager.get_training_status("test_detector", "detection")
        assert status["status"] == "not_started"
        
        # Add some training history
        metrics = TrainingMetrics(epoch=5, train_loss=0.5, val_loss=0.6)
        pipeline.training_history.append(metrics)
        
        # Test training status
        status = manager.get_training_status("test_detector", "detection")
        assert status["status"] == "training"
        assert status["current_epoch"] == 5
    
    def test_stop_training(self):
        """Test stopping training."""
        manager = TrainingManager(self.tmp_dir)
        
        config = TrainingConfig(
            model_name="test_detector",
            model_type="detection",
            architecture="YOLO"
        )
        
        manager.create_detection_pipeline(config)
        assert "detection_test_detector" in manager.active_trainings
        
        manager.stop_training("test_detector", "detection")
        assert "detection_test_detector" not in manager.active_trainings
    
    def test_list_active_trainings(self):
        """Test listing active trainings."""
        manager = TrainingManager(self.tmp_dir)
        
        # Initially empty
        assert manager.list_active_trainings() == []
        
        # Create some pipelines
        detection_config = TrainingConfig(
            model_name="detector",
            model_type="detection",
            architecture="YOLO"
        )
        
        recognition_config = TrainingConfig(
            model_name="recognizer",
            model_type="recognition",
            architecture="CRNN"
        )
        
        manager.create_detection_pipeline(detection_config)
        manager.create_recognition_pipeline(recognition_config)
        
        active_trainings = manager.list_active_trainings()
        assert len(active_trainings) == 2
        assert "detection_detector" in active_trainings
        assert "recognition_recognizer" in active_trainings