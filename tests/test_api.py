"""
Unit tests for the REST API functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import io
from pathlib import Path
from werkzeug.datastructures import FileStorage

from src.api.main_api import HistoricalMapAPI, create_app
from src.api.request_validator import RequestValidator
from src.api.response_formatter import ResponseFormatter
from src.api.file_manager import APIFileManager
from src.api.error_handlers import APIErrorHandler


class TestHistoricalMapAPI(unittest.TestCase):
    """Test cases for the main API class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api = HistoricalMapAPI({'debug': True})
        self.client = self.api.app.test_client()
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
        self.assertIn('services', data)
    
    def test_system_info(self):
        """Test system info endpoint."""
        response = self.client.get('/info')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('system', data)
        self.assertIn('api', data)
        self.assertIn('tasks', data)
    
    def test_process_image_no_file(self):
        """Test process endpoint without file."""
        response = self.client.post('/process/task1')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_process_image_invalid_task(self):
        """Test process endpoint with invalid task."""
        # Create mock file
        data = {'file': (io.BytesIO(b'fake image data'), 'test.jpg')}
        
        response = self.client.post('/process/invalid_task', data=data)
        self.assertEqual(response.status_code, 400)
    
    @patch('src.api.main_api.HistoricalMapAPI._process_image_file')
    def test_process_image_success(self, mock_process):
        """Test successful image processing."""
        # Mock processing result
        from src.models.data_structures import ProcessingResult, Detection, BoundingBox
        
        mock_detection = Detection(
            bbox=BoundingBox(x=100, y=50, width=200, height=30, confidence=0.85),
            detection_type='word',
            confidence=0.85,
            image_coordinates=(100, 50, 300, 80)
        )
        
        mock_result = ProcessingResult(
            task_type='task1',
            detections=[mock_detection],
            recognized_texts=[],
            linked_texts=[],
            processing_time=1.5,
            metrics={'confidence_avg': 0.85}
        )
        
        mock_process.return_value = mock_result
        
        # Create test file
        data = {
            'file': (io.BytesIO(b'fake image data'), 'test.jpg'),
            'confidence': '0.7'
        }
        
        response = self.client.post('/process/task1', data=data)
        self.assertEqual(response.status_code, 200)
        
        result_data = json.loads(response.data)
        self.assertIn('result_id', result_data)
        self.assertEqual(result_data['task'], 'task1')
        self.assertIn('results', result_data)
    
    def test_batch_process_no_files(self):
        """Test batch process without files."""
        response = self.client.post('/batch/task1')
        self.assertEqual(response.status_code, 400)
    
    def test_get_nonexistent_result(self):
        """Test getting non-existent result."""
        response = self.client.get('/results/nonexistent-id')
        self.assertEqual(response.status_code, 404)
    
    def test_get_config(self):
        """Test get configuration endpoint."""
        response = self.client.get('/config')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('configuration', data)
    
    def test_update_config_invalid_json(self):
        """Test update configuration with invalid JSON."""
        response = self.client.post('/config', data='invalid json')
        self.assertEqual(response.status_code, 400)
    
    def test_update_config_success(self):
        """Test successful configuration update."""
        config_data = {
            'detection.confidence_threshold': 0.8
        }
        
        response = self.client.post('/config', 
                                  data=json.dumps(config_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('message', data)
        self.assertIn('updated_keys', data)
    
    def test_list_models(self):
        """Test list models endpoint."""
        response = self.client.get('/models')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('detection_models', data)
        self.assertIn('recognition_models', data)
    
    def test_switch_model_no_json(self):
        """Test switch model without JSON."""
        response = self.client.post('/models/detection/switch')
        self.assertEqual(response.status_code, 400)
    
    def test_switch_model_success(self):
        """Test successful model switching."""
        model_data = {'version': '2.0'}
        
        response = self.client.post('/models/detection/switch',
                                  data=json.dumps(model_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('message', data)
        self.assertEqual(data['version'], '2.0')
    
    def test_evaluate_invalid_task(self):
        """Test evaluation with invalid task."""
        eval_data = {
            'results': [],
            'ground_truth': []
        }
        
        response = self.client.post('/evaluate/invalid_task',
                                  data=json.dumps(eval_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
    
    def test_evaluate_success(self):
        """Test successful evaluation."""
        eval_data = {
            'results': [
                {'bbox': [100, 50, 300, 80], 'confidence': 0.85}
            ],
            'ground_truth': [
                {'bbox': [105, 52, 295, 78]}
            ]
        }
        
        response = self.client.post('/evaluate/task1',
                                  data=json.dumps(eval_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('metrics', data['evaluation'])


class TestRequestValidator(unittest.TestCase):
    """Test cases for request validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = RequestValidator()
    
    def test_validate_task(self):
        """Test task validation."""
        self.assertTrue(self.validator.validate_task('task1'))
        self.assertTrue(self.validator.validate_task('task4'))
        self.assertFalse(self.validator.validate_task('task5'))
        self.assertFalse(self.validator.validate_task('invalid'))
    
    def test_validate_file(self):
        """Test file validation."""
        allowed_extensions = {'jpg', 'png', 'tiff'}
        
        # Valid file
        valid_file = Mock()
        valid_file.filename = 'test.jpg'
        self.assertTrue(self.validator.validate_file(valid_file, allowed_extensions))
        
        # Invalid extension
        invalid_file = Mock()
        invalid_file.filename = 'test.txt'
        self.assertFalse(self.validator.validate_file(invalid_file, allowed_extensions))
        
        # No filename
        no_name_file = Mock()
        no_name_file.filename = None
        self.assertFalse(self.validator.validate_file(no_name_file, allowed_extensions))
        
        # No extension
        no_ext_file = Mock()
        no_ext_file.filename = 'test'
        self.assertFalse(self.validator.validate_file(no_ext_file, allowed_extensions))
    
    def test_validate_confidence_threshold(self):
        """Test confidence threshold validation."""
        self.assertTrue(self.validator.validate_confidence_threshold(0.5))
        self.assertTrue(self.validator.validate_confidence_threshold(0.0))
        self.assertTrue(self.validator.validate_confidence_threshold(1.0))
        self.assertFalse(self.validator.validate_confidence_threshold(-0.1))
        self.assertFalse(self.validator.validate_confidence_threshold(1.1))
    
    def test_validate_format(self):
        """Test format validation."""
        self.assertTrue(self.validator.validate_format('txt'))
        self.assertTrue(self.validator.validate_format('json'))
        self.assertFalse(self.validator.validate_format('xml'))
        self.assertFalse(self.validator.validate_format('invalid'))
    
    def test_validate_metrics(self):
        """Test metrics validation."""
        self.assertTrue(self.validator.validate_metrics(['precision', 'recall']))
        self.assertTrue(self.validator.validate_metrics(['f1', 'iou']))
        self.assertFalse(self.validator.validate_metrics(['invalid_metric']))
        self.assertFalse(self.validator.validate_metrics([]))
    
    def test_validate_config_updates(self):
        """Test configuration updates validation."""
        # Valid updates
        valid_config = {
            'detection.confidence_threshold': 0.8,
            'recognition.languages': ['en', 'zh']
        }
        self.assertTrue(self.validator.validate_config_updates(valid_config))
        
        # Invalid key
        invalid_config = {
            'invalid.key': 'value'
        }
        self.assertFalse(self.validator.validate_config_updates(invalid_config))
        
        # Invalid type
        invalid_type_config = {
            'detection.confidence_threshold': 'not_a_number'
        }
        self.assertFalse(self.validator.validate_config_updates(invalid_type_config))
    
    def test_validate_result_id(self):
        """Test result ID validation."""
        # Valid UUID
        valid_uuid = '550e8400-e29b-41d4-a716-446655440000'
        self.assertTrue(self.validator.validate_result_id(valid_uuid))
        
        # Valid batch ID
        valid_batch_id = '550e8400-e29b-41d4-a716-446655440000_1'
        self.assertTrue(self.validator.validate_result_id(valid_batch_id))
        
        # Invalid format
        self.assertFalse(self.validator.validate_result_id('invalid-id'))
        self.assertFalse(self.validator.validate_result_id('123'))
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Normal filename
        self.assertEqual(
            self.validator.sanitize_filename('test.jpg'),
            'test.jpg'
        )
        
        # Filename with dangerous characters
        self.assertEqual(
            self.validator.sanitize_filename('test<>file.jpg'),
            'test__file.jpg'
        )
        
        # Filename with path
        self.assertEqual(
            self.validator.sanitize_filename('/path/to/file.jpg'),
            'file.jpg'
        )


class TestResponseFormatter(unittest.TestCase):
    """Test cases for response formatting."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formatter = ResponseFormatter()
    
    def test_format_processing_response(self):
        """Test processing response formatting."""
        from src.models.data_structures import ProcessingResult, Detection, BoundingBox
        
        # Create mock processing result
        detection = Detection(
            bbox=BoundingBox(x=100, y=50, width=200, height=30, confidence=0.85),
            detection_type='word',
            confidence=0.85,
            image_coordinates=(100, 50, 300, 80)
        )
        
        result = ProcessingResult(
            task_type='task1',
            detections=[detection],
            recognized_texts=[],
            linked_texts=[],
            processing_time=1.5,
            metrics={'confidence_avg': 0.85}
        )
        
        response = self.formatter.format_processing_response(
            result_id='test-id',
            task='task1',
            filename='test.jpg',
            processing_result=result,
            result_file_path=Path('/tmp/result.txt')
        )
        
        self.assertEqual(response['result_id'], 'test-id')
        self.assertEqual(response['task'], 'task1')
        self.assertEqual(response['filename'], 'test.jpg')
        self.assertIn('results', response)
        self.assertIn('detections', response['results'])
        self.assertEqual(len(response['results']['detections']), 1)
    
    def test_format_error_response(self):
        """Test error response formatting."""
        response = self.formatter.format_error_response(
            error_message='Test error',
            error_code='TEST_ERROR',
            details={'field': 'value'}
        )
        
        self.assertEqual(response['error'], 'Test error')
        self.assertEqual(response['error_code'], 'TEST_ERROR')
        self.assertEqual(response['details'], {'field': 'value'})
        self.assertIn('timestamp', response)
    
    def test_format_batch_response(self):
        """Test batch response formatting."""
        results = [
            {
                'success': True,
                'processing_time': 1.5,
                'detection_count': 3,
                'recognition_count': 2
            },
            {
                'success': False,
                'error': 'Processing failed'
            }
        ]
        
        response = self.formatter.format_batch_response(
            batch_id='batch-123',
            task='task1',
            results=results
        )
        
        self.assertEqual(response['batch_id'], 'batch-123')
        self.assertEqual(response['task'], 'task1')
        self.assertEqual(response['summary']['total_files'], 2)
        self.assertEqual(response['summary']['successful'], 1)
        self.assertEqual(response['summary']['failed'], 1)


class TestAPIFileManager(unittest.TestCase):
    """Test cases for API file management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.file_manager = APIFileManager(
            upload_folder=f"{self.temp_dir}/uploads",
            results_folder=f"{self.temp_dir}/results"
        )
    
    def test_save_uploaded_file(self):
        """Test saving uploaded files."""
        # Create mock file
        file_content = b'fake image data'
        mock_file = Mock()
        mock_file.filename = 'test.jpg'
        mock_file.save = Mock()
        
        # Save file
        result_id = 'test-result-id'
        file_path = self.file_manager.save_uploaded_file(mock_file, result_id)
        
        # Verify
        expected_path = Path(self.temp_dir) / 'uploads' / f'{result_id}.jpg'
        self.assertEqual(file_path, expected_path)
        mock_file.save.assert_called_once()
    
    def test_save_result(self):
        """Test saving result files."""
        result_id = 'test-result-id'
        content = '100,50,300,80,0.85\n150,100,250,130,0.92'
        task = 'task1'
        
        result_path = self.file_manager.save_result(result_id, content, task)
        
        # Verify file was created
        expected_path = Path(self.temp_dir) / 'results' / f'{result_id}_{task}_results.txt'
        self.assertEqual(result_path, expected_path)
        
        # Verify metadata was saved
        metadata = self.file_manager._load_result_metadata(result_id)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata['task'], task)
    
    def test_get_storage_stats(self):
        """Test storage statistics."""
        stats = self.file_manager.get_storage_stats()
        
        self.assertIn('upload_folder', stats)
        self.assertIn('results_folder', stats)
        self.assertIn('upload_count', stats)
        self.assertIn('result_count', stats)
        self.assertIn('total_size', stats)


class TestAPIErrorHandler(unittest.TestCase):
    """Test cases for API error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_handler = APIErrorHandler()
    
    def test_handle_bad_request(self):
        """Test bad request error handling."""
        mock_error = Mock()
        mock_error.description = 'Invalid parameter'
        
        response, status_code = self.error_handler.handle_bad_request(mock_error)
        
        self.assertEqual(status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Bad Request')
        self.assertIn('timestamp', data)
    
    def test_handle_validation_error(self):
        """Test validation error handling."""
        validation_errors = {
            'confidence': 'Must be between 0 and 1',
            'format': 'Must be txt or json'
        }
        
        response, status_code = self.error_handler.handle_validation_error(validation_errors)
        
        self.assertEqual(status_code, 422)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Validation Error')
        self.assertEqual(data['validation_errors'], validation_errors)
    
    def test_handle_processing_error(self):
        """Test processing error handling."""
        error_message = 'Model inference failed'
        details = {'model': 'yolo', 'stage': 'detection'}
        
        response, status_code = self.error_handler.handle_processing_error(error_message, details)
        
        self.assertEqual(status_code, 500)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Processing Error')
        self.assertEqual(data['message'], error_message)
        self.assertEqual(data['details'], details)


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for API components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = create_app({'debug': True})
        self.client = self.app.test_client()
    
    def test_full_processing_workflow(self):
        """Test complete processing workflow."""
        # 1. Check health
        health_response = self.client.get('/health')
        self.assertEqual(health_response.status_code, 200)
        
        # 2. Get system info
        info_response = self.client.get('/info')
        self.assertEqual(info_response.status_code, 200)
        
        # 3. List models
        models_response = self.client.get('/models')
        self.assertEqual(models_response.status_code, 200)
        
        # 4. Get configuration
        config_response = self.client.get('/config')
        self.assertEqual(config_response.status_code, 200)


if __name__ == '__main__':
    unittest.main()