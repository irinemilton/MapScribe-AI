"""
Tests for the integrated Historical Map Text Processing System.
Validates complete system functionality and component integration.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.core.system_integration import HistoricalMapSystem, create_system
from src.core.system_configuration import SystemConfigurationManager, RuntimeConfiguration
from src.config.base_config import SystemConfig
from src.models.data_structures import ProcessingResult, Detection, RecognizedText, BoundingBox


class TestSystemIntegration:
    """Test suite for system integration functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_image_path = self.temp_dir / "test_image.jpg"
        self.config_path = self.temp_dir / "test_config.yaml"
        
        # Create mock test image
        self._create_mock_image()
        
        # Create test configuration
        self._create_test_config()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_image(self):
        """Create a mock image file for testing."""
        # Create a simple test image
        import cv2
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(self.test_image_path), test_image)
    
    def _create_test_config(self):
        """Create a test configuration file."""
        config_content = """
detection:
  model_architecture: "yolo"
  model_path: ""
  input_size: [640, 640]
  confidence_threshold: 0.5
  nms_threshold: 0.4
  batch_size: 1
  device: "cpu"

recognition:
  model_architecture: "crnn"
  model_path: ""
  max_sequence_length: 32
  vocabulary_size: 1000
  languages: ["en"]
  beam_width: 5
  device: "cpu"

preprocessing:
  target_size: [1024, 1024]
  enhance_contrast: true
  denoise: true
  normalize: true
  auto_rotate: false
  color_space: "RGB"

evaluation:
  iou_threshold: 0.5
  confidence_threshold: 0.5
  calculate_character_accuracy: true
  calculate_word_accuracy: true
  save_detailed_results: false
  output_format: "txt"

log_level: "INFO"
output_dir: "./results"
temp_dir: "./temp"
max_workers: 1
memory_limit_gb: 4.0
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)
    
    @patch('src.core.system_integration.BasicImageProcessor')
    @patch('src.core.system_integration.HistoricalMapProcessor')
    @patch('src.core.system_integration.YOLODetector')
    @patch('src.core.system_integration.CRNNRecognizer')
    def test_system_initialization(self, mock_crnn, mock_yolo, mock_hist_proc, mock_img_proc):
        """Test system initialization with configuration."""
        # Mock the components
        mock_img_proc.return_value = Mock()
        mock_hist_proc.return_value = Mock()
        mock_yolo.return_value = Mock()
        mock_crnn.return_value = Mock()
        
        # Initialize system
        system = HistoricalMapSystem(str(self.config_path))
        
        # Verify system is created but not initialized
        assert not system._initialized
        
        # Initialize system
        system.initialize()
        
        # Verify system is initialized
        assert system._initialized
        assert system.config is not None
        
        # Verify components are created
        assert hasattr(system, 'image_processor')
        assert hasattr(system, 'historical_processor')
        assert hasattr(system, 'word_detector')
        assert hasattr(system, 'crnn_recognizer')
    
    @patch('src.core.system_integration.BasicImageProcessor')
    @patch('src.core.system_integration.HistoricalMapProcessor')
    @patch('src.core.system_integration.YOLODetector')
    @patch('src.core.system_integration.CRNNRecognizer')
    def test_single_image_processing(self, mock_crnn, mock_yolo, mock_hist_proc, mock_img_proc):
        """Test processing a single image."""
        # Mock components
        mock_img_processor = Mock()
        mock_hist_processor = Mock()
        mock_detector = Mock()
        mock_recognizer = Mock()
        
        mock_img_proc.return_value = mock_img_processor
        mock_hist_proc.return_value = mock_hist_processor
        mock_yolo.return_value = mock_detector
        mock_crnn.return_value = mock_recognizer
        
        # Mock processing results
        from src.models.data_structures import ProcessedImage
        mock_processed_image = ProcessedImage(
            image=np.ones((480, 640, 3), dtype=np.uint8),
            metadata={'processed': True}
        )
        
        mock_detection = Detection(
            bbox=BoundingBox(x=100, y=100, width=200, height=50, confidence=0.85),
            detection_type='word',
            confidence=0.85,
            image_coordinates=(100, 100, 300, 150)
        )
        
        mock_recognized = RecognizedText(
            detection=mock_detection,
            text="Test Text",
            language="en",
            recognition_confidence=0.80,
            character_confidences=[0.8, 0.85, 0.9, 0.75]
        )
        
        # Configure mocks
        mock_img_processor.load_image.return_value = np.ones((480, 640, 3), dtype=np.uint8)
        mock_hist_processor.preprocess_image.return_value = mock_processed_image
        mock_detector.detect_words.return_value = [mock_detection]
        mock_recognizer.recognize_text.return_value = mock_recognized
        
        # Initialize and test system
        system = HistoricalMapSystem(str(self.config_path))
        system.initialize()
        
        # Process image
        result = system.process_image(str(self.test_image_path), "task3")
        
        # Verify result
        assert isinstance(result, ProcessingResult)
        assert result.task_type == "task3"
        assert len(result.detections) > 0
        assert len(result.recognized_texts) > 0
        assert result.processing_time > 0
    
    @patch('src.core.system_integration.BasicImageProcessor')
    @patch('src.core.system_integration.HistoricalMapProcessor')
    @patch('src.core.system_integration.YOLODetector')
    def test_batch_processing(self, mock_yolo, mock_hist_proc, mock_img_proc):
        """Test batch processing of multiple images."""
        # Create additional test images
        test_images = []
        for i in range(3):
            img_path = self.temp_dir / f"test_image_{i}.jpg"
            import cv2
            test_image = np.ones((480, 640, 3), dtype=np.uint8) * (100 + i * 50)
            cv2.imwrite(str(img_path), test_image)
            test_images.append(str(img_path))
        
        # Mock components
        mock_img_processor = Mock()
        mock_hist_processor = Mock()
        mock_detector = Mock()
        
        mock_img_proc.return_value = mock_img_processor
        mock_hist_proc.return_value = mock_hist_processor
        mock_yolo.return_value = mock_detector
        
        # Mock processing results
        from src.models.data_structures import ProcessedImage
        mock_processed_image = ProcessedImage(
            image=np.ones((480, 640, 3), dtype=np.uint8),
            metadata={'processed': True}
        )
        
        mock_detection = Detection(
            bbox=BoundingBox(x=100, y=100, width=200, height=50, confidence=0.85),
            detection_type='word',
            confidence=0.85,
            image_coordinates=(100, 100, 300, 150)
        )
        
        # Configure mocks
        mock_img_processor.load_image.return_value = np.ones((480, 640, 3), dtype=np.uint8)
        mock_hist_processor.preprocess_image.return_value = mock_processed_image
        mock_detector.detect_words.return_value = [mock_detection]
        
        # Initialize system
        system = HistoricalMapSystem(str(self.config_path))
        system.initialize()
        
        # Process batch
        results = system.process_batch(test_images, "task1")
        
        # Verify results
        assert len(results) == 3
        for result in results:
            assert isinstance(result, ProcessingResult)
            assert result.task_type == "task1"
    
    def test_system_configuration_manager(self):
        """Test system configuration management."""
        config_manager = SystemConfigurationManager()
        
        # Test loading system configuration
        system_config = config_manager.load_system_configuration(str(self.config_path))
        assert isinstance(system_config, SystemConfig)
        assert system_config.detection.model_architecture == "yolo"
        assert system_config.recognition.model_architecture == "crnn"
        
        # Test loading runtime configuration
        runtime_config = config_manager.load_runtime_configuration()
        assert isinstance(runtime_config, RuntimeConfiguration)
        
        # Test parameter updates
        config_manager.update_system_parameter("detection.confidence_threshold", 0.7)
        assert system_config.detection.confidence_threshold == 0.7
        
        config_manager.update_runtime_parameter("batch_size", 4)
        assert runtime_config.batch_size == 4
        
        # Test configuration validation
        assert config_manager.validate_configuration(system_config)
        assert config_manager.validate_configuration(runtime_config)
    
    def test_error_handling(self):
        """Test error handling and graceful degradation."""
        system = HistoricalMapSystem(str(self.config_path))
        
        # Test processing non-existent file
        with pytest.raises(FileNotFoundError):
            system.process_image("non_existent.jpg", "task1")
        
        # Test invalid task type
        with pytest.raises(ValueError):
            system.process_image(str(self.test_image_path), "invalid_task")
        
        # Test processing without initialization
        with pytest.raises(RuntimeError):
            system.process_image(str(self.test_image_path), "task1")
    
    @patch('src.core.system_integration.BasicImageProcessor')
    @patch('src.core.system_integration.HistoricalMapProcessor')
    @patch('src.core.system_integration.YOLODetector')
    def test_system_status_monitoring(self, mock_yolo, mock_hist_proc, mock_img_proc):
        """Test system status and monitoring functionality."""
        # Mock components
        mock_img_proc.return_value = Mock()
        mock_hist_proc.return_value = Mock()
        mock_yolo.return_value = Mock()
        
        system = HistoricalMapSystem(str(self.config_path))
        system.initialize()
        
        # Get system status
        status = system.get_system_status()
        
        # Verify status structure
        assert 'initialized' in status
        assert 'configuration' in status
        assert 'processing_stats' in status
        assert 'monitoring_data' in status
        
        assert status['initialized'] is True
        assert status['processing_stats']['total_processed'] == 0
    
    def test_factory_function(self):
        """Test the create_system factory function."""
        with patch('src.core.system_integration.HistoricalMapSystem') as mock_system_class:
            mock_system = Mock()
            mock_system_class.return_value = mock_system
            
            # Test factory function
            system = create_system(str(self.config_path))
            
            # Verify system creation and initialization
            mock_system_class.assert_called_once_with(str(self.config_path))
            mock_system.initialize.assert_called_once()
    
    @patch('src.core.system_integration.BasicImageProcessor')
    @patch('src.core.system_integration.HistoricalMapProcessor')
    @patch('src.core.system_integration.YOLODetector')
    def test_system_shutdown(self, mock_yolo, mock_hist_proc, mock_img_proc):
        """Test system shutdown functionality."""
        # Mock components
        mock_img_proc.return_value = Mock()
        mock_hist_proc.return_value = Mock()
        mock_yolo.return_value = Mock()
        
        system = HistoricalMapSystem(str(self.config_path))
        system.initialize()
        
        assert system._initialized is True
        
        # Shutdown system
        system.shutdown()
        
        # Verify shutdown
        assert system._initialized is False
    
    def test_configuration_export_import(self):
        """Test configuration export and import functionality."""
        config_manager = SystemConfigurationManager()
        
        # Load configurations
        system_config = config_manager.load_system_configuration(str(self.config_path))
        runtime_config = config_manager.load_runtime_configuration()
        
        # Export configuration
        export_path = self.temp_dir / "exported_config.json"
        config_manager.export_configuration(str(export_path), "json")
        
        assert export_path.exists()
        
        # Verify export content
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        assert 'system_config' in exported_data
        assert 'runtime_config' in exported_data
        assert 'export_timestamp' in exported_data
        
        # Test import
        config_manager.reset_to_defaults()
        config_manager.import_configuration(str(export_path))
        
        # Verify imported configuration
        imported_system_config = config_manager.get_system_configuration()
        imported_runtime_config = config_manager.get_runtime_configuration()
        
        assert imported_system_config.detection.model_architecture == "yolo"
        assert imported_runtime_config.batch_size == 1


class TestSystemConfigurationManager:
    """Test suite for system configuration management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_manager = SystemConfigurationManager()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_runtime_configuration_validation(self):
        """Test runtime configuration validation."""
        # Test valid configuration
        valid_config = RuntimeConfiguration(
            batch_size=2,
            confidence_threshold=0.7,
            output_format='json'
        )
        assert self.config_manager.validate_configuration(valid_config)
        
        # Test invalid batch size
        with pytest.raises(ValueError):
            RuntimeConfiguration(batch_size=0)
        
        # Test invalid confidence threshold
        with pytest.raises(ValueError):
            RuntimeConfiguration(confidence_threshold=1.5)
        
        # Test invalid output format
        with pytest.raises(ValueError):
            RuntimeConfiguration(output_format='invalid')
    
    def test_parameter_update_validation(self):
        """Test parameter update validation."""
        # Load default configurations
        system_config = self.config_manager.load_system_configuration()
        runtime_config = self.config_manager.load_runtime_configuration()
        
        # Test valid system parameter update
        self.config_manager.update_system_parameter("detection.confidence_threshold", 0.8)
        assert system_config.detection.confidence_threshold == 0.8
        
        # Test invalid system parameter
        with pytest.raises(ValueError):
            self.config_manager.update_system_parameter("invalid.parameter", 0.5)
        
        # Test valid runtime parameter update
        self.config_manager.update_runtime_parameter("batch_size", 8)
        assert runtime_config.batch_size == 8
        
        # Test invalid runtime parameter value
        with pytest.raises(ValueError):
            self.config_manager.update_runtime_parameter("batch_size", -1)
    
    def test_configuration_history(self):
        """Test configuration change history tracking."""
        # Load configurations
        self.config_manager.load_system_configuration()
        self.config_manager.load_runtime_configuration()
        
        # Make some changes
        self.config_manager.update_system_parameter("detection.confidence_threshold", 0.6)
        self.config_manager.update_runtime_parameter("batch_size", 3)
        
        # Check history
        history = self.config_manager.get_configuration_history()
        
        assert len(history) >= 4  # load_system, load_runtime, 2 updates
        
        # Verify history entries have required fields
        for entry in history:
            assert 'timestamp' in entry
            assert 'action' in entry
            assert 'details' in entry
    
    def test_configuration_summary(self):
        """Test configuration summary generation."""
        # Load configurations
        self.config_manager.load_system_configuration()
        self.config_manager.load_runtime_configuration()
        
        # Get summary
        summary = self.config_manager.get_configuration_summary()
        
        # Verify summary structure
        assert 'system_config_loaded' in summary
        assert 'runtime_config_loaded' in summary
        assert 'config_files' in summary
        assert 'file_exists' in summary
        
        assert summary['system_config_loaded'] is True
        assert summary['runtime_config_loaded'] is True


if __name__ == '__main__':
    pytest.main([__file__])