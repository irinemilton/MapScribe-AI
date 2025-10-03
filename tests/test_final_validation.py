"""
Tests for final validation and competition preparation functionality.
Validates the complete final validation system for task 12.2.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from final_validation import FinalValidationRunner
from src.models.data_structures import ProcessingResult, Detection, RecognizedText, BoundingBox


class TestFinalValidationRunner:
    """Test suite for final validation runner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_config.yaml"
        
        # Create test configuration
        self._create_test_config()
        
        # Initialize runner
        self.runner = FinalValidationRunner(str(self.config_path))
        self.runner.output_dir = self.temp_dir / "validation_output"
        self.runner.sample_data_dir = self.temp_dir / "sample_data"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_config(self):
        """Create a test configuration file."""
        config_content = """
detection:
  model_architecture: "yolo"
  confidence_threshold: 0.5
recognition:
  model_architecture: "crnn"
  languages: ["en"]
preprocessing:
  enhance_contrast: true
evaluation:
  iou_threshold: 0.5
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)
    
    def _create_mock_sample_data(self):
        """Create mock sample data for testing."""
        sample_data = {
            'images': {
                'test_image_1.jpg': {
                    'type': 'standard_map',
                    'dimensions': [1024, 768, 3],
                    'text_regions': 5
                },
                'test_image_2.jpg': {
                    'type': 'degraded_map',
                    'dimensions': [800, 600, 3],
                    'text_regions': 3
                }
            },
            'ground_truth': {
                'test_image_1.jpg': {
                    'detections_count': 5,
                    'texts_count': 5
                }
            }
        }
        
        # Save sample data metadata
        self.runner.sample_data_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = self.runner.sample_data_dir / "dataset_info.json"
        with open(metadata_file, 'w') as f:
            json.dump(sample_data, f)
        
        return sample_data
    
    @patch('scripts.final_validation.create_system')
    def test_system_initialization(self, mock_create_system):
        """Test system initialization."""
        mock_system = Mock()
        mock_create_system.return_value = mock_system
        
        self.runner._initialize_system()
        
        mock_create_system.assert_called_once_with(str(self.config_path))
        assert self.runner.system == mock_system
    
    def test_prepare_sample_data_existing(self):
        """Test preparing sample data when it already exists."""
        # Create mock sample data
        sample_data = self._create_mock_sample_data()
        
        # Test loading existing data
        result = self.runner._prepare_sample_data()
        
        assert result == sample_data
        assert len(result['images']) == 2
    
    @patch('scripts.final_validation.IntegrationSampleDataGenerator')
    def test_prepare_sample_data_generation(self, mock_generator_class):
        """Test sample data generation when it doesn't exist."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        sample_data = {'images': {'generated.jpg': {}}}
        mock_generator.generate_complete_test_dataset.return_value = sample_data
        
        result = self.runner._prepare_sample_data()
        
        mock_generator.generate_complete_test_dataset.assert_called_once()
        assert result == sample_data
    
    def test_mock_process_image(self):
        """Test mock image processing."""
        image_info = {
            'type': 'standard_map',
            'text_regions': 3
        }
        
        result = self.runner._mock_process_image(image_info, 'task3')
        
        assert isinstance(result, ProcessingResult)
        assert result.task_type == 'task3'
        assert len(result.detections) == 3
        assert len(result.recognized_texts) == 3
        assert result.processing_time > 0
    
    def test_format_results_for_validation(self):
        """Test formatting results for validation."""
        # Create mock processing results
        detection = Detection(
            bbox=BoundingBox(x=100, y=100, width=200, height=50, confidence=0.85),
            detection_type='word',
            confidence=0.85,
            image_coordinates=(100, 100, 300, 150)
        )
        
        recognized_text = RecognizedText(
            detection=detection,
            text="Test Text",
            language="en",
            recognition_confidence=0.80,
            character_confidences=[0.8, 0.85, 0.9, 0.75]
        )
        
        result = ProcessingResult(
            task_type='task3',
            detections=[detection],
            recognized_texts=[recognized_text],
            linked_texts=[],
            processing_time=1.5,
            metrics={}
        )
        
        # Test task1 formatting (detection only)
        formatted_task1 = self.runner._format_results_for_validation([result], 'task1')
        expected_task1 = "100,100,200,50,0.85"
        assert formatted_task1 == expected_task1
        
        # Test task3 formatting (detection + recognition)
        formatted_task3 = self.runner._format_results_for_validation([result], 'task3')
        expected_task3 = "100,100,200,50,0.85,Test Text"
        assert formatted_task3 == expected_task3
    
    @patch('scripts.final_validation.create_system')
    def test_test_system_with_sample_data(self, mock_create_system):
        """Test system testing with sample data."""
        mock_system = Mock()
        mock_create_system.return_value = mock_system
        
        # Initialize system
        self.runner._initialize_system()
        
        # Create sample data
        sample_data = self._create_mock_sample_data()
        
        # Test system
        results = self.runner._test_system_with_sample_data(sample_data)
        
        # Verify results structure
        assert 'task1_results' in results
        assert 'task2_results' in results
        assert 'task3_results' in results
        assert 'task4_results' in results
        assert 'processing_times' in results
        assert 'success_count' in results
        assert 'error_count' in results
        
        # Verify processing times structure
        for task in ['task1', 'task2', 'task3', 'task4']:
            assert task in results['processing_times']
            timing = results['processing_times'][task]
            assert 'total_time' in timing
            assert 'average_time' in timing
            assert 'min_time' in timing
            assert 'max_time' in timing
    
    def test_validate_competition_formats(self):
        """Test competition format validation."""
        # Create mock test results
        test_results = {
            'task1_results': [
                self.runner._mock_process_image({'text_regions': 2}, 'task1')
            ],
            'task3_results': [
                self.runner._mock_process_image({'text_regions': 2}, 'task3')
            ]
        }
        
        # Test format validation
        format_results = self.runner._validate_competition_formats(test_results)
        
        # Verify results structure
        assert 'task1' in format_results
        assert 'task3' in format_results
        assert 'overall_compliance' in format_results
        
        # Verify individual task results
        for task in ['task1', 'task3']:
            task_result = format_results[task]
            assert 'status' in task_result
            assert 'errors' in task_result
            assert 'warnings' in task_result
            assert 'result_count' in task_result
        
        # Verify overall compliance
        compliance = format_results['overall_compliance']
        assert 'valid_tasks' in compliance
        assert 'total_tasks' in compliance
        assert 'compliance_rate' in compliance
        assert 'is_compliant' in compliance
    
    @patch('scripts.final_validation.SubmissionPackager')
    def test_create_submission_package(self, mock_packager_class):
        """Test submission package creation."""
        mock_packager = Mock()
        mock_packager_class.return_value = mock_packager
        
        # Mock packager methods
        submission_dir = self.temp_dir / "submission"
        archive_path = self.temp_dir / "submission.zip"
        
        mock_packager.create_submission_package.return_value = submission_dir
        mock_packager.validate_submission_package.return_value = Mock(
            is_valid=True, errors=[], warnings=[]
        )
        mock_packager.generate_submission_checklist.return_value = []
        mock_packager.optimize_submission_size.return_value = {'size_reduction_percent': 10}
        mock_packager.create_submission_archive.return_value = archive_path
        
        # Create test results
        test_results = {
            'task1_results': [self.runner._mock_process_image({'text_regions': 2}, 'task1')]
        }
        
        # Test submission package creation
        submission_results = self.runner._create_submission_package(test_results)
        
        # Verify results
        assert submission_results['status'] == 'SUCCESS'
        assert 'submission_directory' in submission_results
        assert 'archive_path' in submission_results
        assert 'package_validation' in submission_results
        assert 'checklist' in submission_results
        assert 'optimization_stats' in submission_results
        
        # Verify packager was called correctly
        mock_packager.create_submission_package.assert_called_once()
        mock_packager.validate_submission_package.assert_called_once()
        mock_packager.generate_submission_checklist.assert_called_once()
        mock_packager.optimize_submission_size.assert_called_once()
        mock_packager.create_submission_archive.assert_called_once()
    
    def test_generate_performance_report(self):
        """Test performance report generation."""
        # Create test results with timing data
        test_results = {
            'success_count': 8,
            'error_count': 0,
            'processing_times': {
                'task1': {
                    'total_time': 10.0,
                    'average_time': 2.5,
                    'min_time': 2.0,
                    'max_time': 3.0
                },
                'task3': {
                    'total_time': 15.0,
                    'average_time': 3.75,
                    'min_time': 3.0,
                    'max_time': 4.5
                }
            }
        }
        
        # Generate performance report
        performance_report = self.runner._generate_performance_report(test_results)
        
        # Verify report structure
        assert 'processing_summary' in performance_report
        assert 'timing_analysis' in performance_report
        assert 'throughput_analysis' in performance_report
        assert 'optimization_recommendations' in performance_report
        
        # Verify processing summary
        summary = performance_report['processing_summary']
        assert summary['total_images_processed'] == 8
        assert summary['total_errors'] == 0
        assert summary['success_rate'] == 1.0
        
        # Verify timing analysis
        timing = performance_report['timing_analysis']
        assert 'task1' in timing
        assert 'task3' in timing
        
        for task in ['task1', 'task3']:
            task_timing = timing[task]
            assert 'average_time_seconds' in task_timing
            assert 'throughput_images_per_second' in task_timing
        
        # Verify throughput analysis
        throughput = performance_report['throughput_analysis']
        assert 'overall_average_time' in throughput
        assert 'overall_throughput' in throughput
        assert 'estimated_daily_capacity' in throughput
        
        # Verify recommendations exist
        assert isinstance(performance_report['optimization_recommendations'], list)
        
        # Verify files are created
        json_report = self.runner.output_dir / "performance_evaluation_report.json"
        txt_report = self.runner.output_dir / "performance_evaluation_report.txt"
        
        assert json_report.exists()
        assert txt_report.exists()
    
    @patch('scripts.final_validation.create_system')
    def test_run_complete_validation(self, mock_create_system):
        """Test complete validation run."""
        mock_system = Mock()
        mock_create_system.return_value = mock_system
        
        # Create sample data
        self._create_mock_sample_data()
        
        # Mock submission packager
        with patch('scripts.final_validation.SubmissionPackager') as mock_packager_class:
            mock_packager = Mock()
            mock_packager_class.return_value = mock_packager
            
            # Mock packager methods
            mock_packager.create_submission_package.return_value = self.temp_dir / "submission"
            mock_packager.validate_submission_package.return_value = Mock(
                is_valid=True, errors=[], warnings=[]
            )
            mock_packager.generate_submission_checklist.return_value = []
            mock_packager.optimize_submission_size.return_value = {'size_reduction_percent': 10}
            mock_packager.create_submission_archive.return_value = self.temp_dir / "submission.zip"
            
            # Run complete validation
            results = self.runner.run_complete_validation(create_submission=True)
            
            # Verify results structure
            assert 'validation_timestamp' in results
            assert 'system_test_results' in results
            assert 'format_validation' in results
            assert 'submission_preparation' in results
            assert 'performance_report' in results
            assert 'overall_status' in results
            
            # Verify final report is saved
            final_report = self.runner.output_dir / "final_validation_report.json"
            assert final_report.exists()
    
    def test_determine_overall_status(self):
        """Test overall status determination."""
        status = self.runner._determine_overall_status()
        
        # Verify status structure
        assert 'status' in status
        assert 'confidence' in status
        assert 'critical_issues' in status
        assert 'warnings' in status
        assert 'recommendations' in status
        
        # Verify types
        assert isinstance(status['status'], str)
        assert isinstance(status['confidence'], (int, float))
        assert isinstance(status['critical_issues'], int)
        assert isinstance(status['warnings'], int)
        assert isinstance(status['recommendations'], list)


class TestFinalValidationIntegration:
    """Integration tests for final validation system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('scripts.final_validation.create_system')
    @patch('scripts.final_validation.SubmissionPackager')
    def test_end_to_end_validation_workflow(self, mock_packager_class, mock_create_system):
        """Test complete end-to-end validation workflow."""
        # Mock system
        mock_system = Mock()
        mock_create_system.return_value = mock_system
        
        # Mock packager
        mock_packager = Mock()
        mock_packager_class.return_value = mock_packager
        mock_packager.create_submission_package.return_value = self.temp_dir / "submission"
        mock_packager.validate_submission_package.return_value = Mock(
            is_valid=True, errors=[], warnings=[]
        )
        mock_packager.generate_submission_checklist.return_value = [
            {'item': 'Test item', 'status': 'PASS', 'description': 'Test', 'details': 'OK'}
        ]
        mock_packager.optimize_submission_size.return_value = {
            'size_reduction_percent': 15,
            'original_size_bytes': 1000,
            'final_size_bytes': 850
        }
        mock_packager.create_submission_archive.return_value = self.temp_dir / "submission.zip"
        
        # Create config file
        config_path = self.temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            f.write("detection:\n  model_architecture: yolo\n")
        
        # Initialize runner
        runner = FinalValidationRunner(str(config_path))
        runner.output_dir = self.temp_dir / "output"
        runner.sample_data_dir = self.temp_dir / "sample_data"
        
        # Create sample data
        runner.sample_data_dir.mkdir(parents=True)
        sample_data = {
            'images': {
                'test1.jpg': {'type': 'standard_map', 'text_regions': 3},
                'test2.jpg': {'type': 'degraded_map', 'text_regions': 2}
            }
        }
        
        metadata_file = runner.sample_data_dir / "dataset_info.json"
        with open(metadata_file, 'w') as f:
            json.dump(sample_data, f)
        
        # Run validation
        results = runner.run_complete_validation(create_submission=True)
        
        # Verify comprehensive results
        assert results is not None
        assert 'validation_timestamp' in results
        
        # Verify system testing occurred
        system_results = results['system_test_results']
        assert system_results['success_count'] > 0
        assert 'processing_times' in system_results
        
        # Verify format validation
        format_results = results['format_validation']
        assert 'overall_compliance' in format_results
        
        # Verify submission preparation
        submission_results = results['submission_preparation']
        assert submission_results['status'] == 'SUCCESS'
        
        # Verify performance report
        performance_results = results['performance_report']
        assert 'processing_summary' in performance_results
        assert 'optimization_recommendations' in performance_results
        
        # Verify files were created
        assert (runner.output_dir / "final_validation_report.json").exists()
        assert (runner.output_dir / "performance_evaluation_report.json").exists()
        assert (runner.output_dir / "performance_evaluation_report.txt").exists()


if __name__ == '__main__':
    pytest.main([__file__])