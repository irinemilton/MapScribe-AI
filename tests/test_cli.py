"""
Unit tests for the CLI functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path
import argparse

from src.cli.main_cli import HistoricalMapCLI
from src.cli.config_cli import ConfigCLI
from src.cli.batch_processor import BatchProcessor, BatchResult
from src.cli.output_manager import OutputManager


class TestHistoricalMapCLI(unittest.TestCase):
    """Test cases for the main CLI class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = HistoricalMapCLI()
    
    def test_create_parser(self):
        """Test argument parser creation."""
        parser = self.cli.create_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)
        
        # Test that all main commands are present
        with patch('sys.argv', ['cli.py', '--help']):
            with self.assertRaises(SystemExit):
                parser.parse_args(['--help'])
    
    def test_process_command_parsing(self):
        """Test process command argument parsing."""
        parser = self.cli.create_parser()
        
        # Valid process command
        args = parser.parse_args(['process', 'image.jpg', 'task1'])
        self.assertEqual(args.command, 'process')
        self.assertEqual(args.input, 'image.jpg')
        self.assertEqual(args.task, 'task1')
        self.assertEqual(args.confidence, 0.5)  # default
    
    def test_batch_command_parsing(self):
        """Test batch command argument parsing."""
        parser = self.cli.create_parser()
        
        # Valid batch command
        args = parser.parse_args(['batch', '/path/to/images', 'task2', '--parallel', '4'])
        self.assertEqual(args.command, 'batch')
        self.assertEqual(args.input_dir, '/path/to/images')
        self.assertEqual(args.task, 'task2')
        self.assertEqual(args.parallel, 4)
    
    def test_config_command_parsing(self):
        """Test config command argument parsing."""
        parser = self.cli.create_parser()
        
        # Config show command
        args = parser.parse_args(['config', 'show'])
        self.assertEqual(args.command, 'config')
        self.assertEqual(args.config_action, 'show')
        
        # Config set command
        args = parser.parse_args(['config', 'set', 'detection.threshold', '0.8'])
        self.assertEqual(args.config_action, 'set')
        self.assertEqual(args.key, 'detection.threshold')
        self.assertEqual(args.value, '0.8')
    
    def test_evaluate_command_parsing(self):
        """Test evaluate command argument parsing."""
        parser = self.cli.create_parser()
        
        args = parser.parse_args(['evaluate', '/results', '/ground_truth', 'task1'])
        self.assertEqual(args.command, 'evaluate')
        self.assertEqual(args.results_dir, '/results')
        self.assertEqual(args.ground_truth, '/ground_truth')
        self.assertEqual(args.task, 'task1')
    
    def test_info_command_parsing(self):
        """Test info command argument parsing."""
        parser = self.cli.create_parser()
        
        args = parser.parse_args(['info', '--models'])
        self.assertEqual(args.command, 'info')
        self.assertTrue(args.models)
    
    @patch('src.cli.main_cli.Path')
    def test_get_output_path_auto_generation(self, mock_path):
        """Test automatic output path generation."""
        mock_input_path = Mock()
        mock_input_path.stem = 'test_image'
        
        output_path = self.cli._get_output_path(None, mock_input_path, 'task1', 'txt')
        expected = Path('test_image_task1_results.txt')
        self.assertEqual(output_path, expected)
    
    def test_get_output_path_explicit(self):
        """Test explicit output path."""
        mock_input_path = Mock()
        output_path = self.cli._get_output_path('/custom/output.txt', mock_input_path, 'task1', 'txt')
        expected = Path('/custom/output.txt')
        self.assertEqual(output_path, expected)
    
    def test_convert_to_json(self):
        """Test JSON conversion of processing results."""
        from src.models.data_structures import ProcessingResult, Detection, RecognizedText, BoundingBox
        
        # Create mock result
        detection = Detection(
            bbox=BoundingBox(x=100, y=50, width=200, height=50, confidence=0.85),
            detection_type='word',
            confidence=0.85,
            image_coordinates=(100, 50, 300, 100)
        )
        
        recognized = RecognizedText(
            detection=detection,
            text="Sample Text",
            language="en",
            recognition_confidence=0.80,
            character_confidences=[0.8, 0.85, 0.9]
        )
        
        result = ProcessingResult(
            task_type='task3',
            detections=[detection],
            recognized_texts=[recognized],
            linked_texts=[],
            processing_time=1.5,
            metrics={'confidence_avg': 0.825}
        )
        
        json_output = self.cli._convert_to_json(result, 'task3')
        parsed = json.loads(json_output)
        
        self.assertEqual(parsed['task'], 'task3')
        self.assertEqual(parsed['processing_time'], 1.5)
        self.assertEqual(len(parsed['detections']), 1)
        self.assertEqual(len(parsed['recognized_texts']), 1)
        self.assertEqual(parsed['recognized_texts'][0]['text'], 'Sample Text')


class TestConfigCLI(unittest.TestCase):
    """Test cases for the configuration CLI."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_cli = ConfigCLI()
    
    def test_parse_config_value_json(self):
        """Test JSON value parsing."""
        # Test number
        self.assertEqual(self.config_cli._parse_config_value('42'), 42)
        self.assertEqual(self.config_cli._parse_config_value('3.14'), 3.14)
        
        # Test boolean
        self.assertEqual(self.config_cli._parse_config_value('true'), True)
        self.assertEqual(self.config_cli._parse_config_value('false'), False)
        
        # Test list
        self.assertEqual(self.config_cli._parse_config_value('["a", "b"]'), ["a", "b"])
        
        # Test dict
        self.assertEqual(self.config_cli._parse_config_value('{"key": "value"}'), {"key": "value"})
    
    def test_parse_config_value_boolean_strings(self):
        """Test boolean string parsing."""
        # True values
        for val in ['true', 'True', 'yes', 'Yes', 'on', 'On', '1']:
            self.assertTrue(self.config_cli._parse_config_value(val))
        
        # False values
        for val in ['false', 'False', 'no', 'No', 'off', 'Off', '0']:
            self.assertFalse(self.config_cli._parse_config_value(val))
    
    def test_parse_config_value_numbers(self):
        """Test number parsing."""
        self.assertEqual(self.config_cli._parse_config_value('123'), 123)
        self.assertEqual(self.config_cli._parse_config_value('45.67'), 45.67)
    
    def test_parse_config_value_string(self):
        """Test string value parsing."""
        self.assertEqual(self.config_cli._parse_config_value('hello world'), 'hello world')
        self.assertEqual(self.config_cli._parse_config_value('path/to/file'), 'path/to/file')
    
    def test_get_config_template(self):
        """Test configuration template generation."""
        template = self.config_cli.get_config_template()
        
        self.assertIsInstance(template, dict)
        self.assertIn('detection', template)
        self.assertIn('recognition', template)
        self.assertIn('preprocessing', template)
        self.assertIn('system', template)
        
        # Check nested structure
        self.assertIn('confidence_threshold', template['detection'])
        self.assertIn('model_type', template['recognition'])


class TestBatchProcessor(unittest.TestCase):
    """Test cases for the batch processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_processor = BatchProcessor()
    
    def test_find_images_simple_pattern(self):
        """Test image finding with simple pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / 'image1.jpg').touch()
            (temp_path / 'image2.png').touch()
            (temp_path / 'document.txt').touch()
            
            # Find JPG files
            jpg_files = self.batch_processor._find_images(temp_path, '*.jpg')
            self.assertEqual(len(jpg_files), 1)
            self.assertEqual(jpg_files[0].name, 'image1.jpg')
    
    def test_find_images_multiple_extensions(self):
        """Test image finding with multiple extensions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / 'image1.jpg').touch()
            (temp_path / 'image2.png').touch()
            (temp_path / 'image3.tiff').touch()
            (temp_path / 'document.txt').touch()
            
            # Find multiple image types
            image_files = self.batch_processor._find_images(temp_path, '*.{jpg,png,tiff}')
            self.assertEqual(len(image_files), 3)
            
            # Check sorted order
            names = [f.name for f in image_files]
            self.assertEqual(names, ['image1.jpg', 'image2.png', 'image3.tiff'])
    
    def test_get_batch_output_path(self):
        """Test batch output path generation."""
        output_dir = Path('/output')
        image_path = Path('/input/test_image.jpg')
        task = 'task1'
        
        output_path = self.batch_processor._get_batch_output_path(output_dir, image_path, task)
        expected = Path('/output/test_image_task1_results.txt')
        self.assertEqual(output_path, expected)
    
    def test_count_detections_in_file(self):
        """Test detection counting in output files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("100,50,300,100,0.85\n")
            temp_file.write("150,120,250,160,0.92\n")
            temp_file.write("\n")  # Empty line should be ignored
            temp_file.flush()
            
            count = self.batch_processor._count_detections_in_file(Path(temp_file.name))
            self.assertEqual(count, 2)
            
            Path(temp_file.name).unlink()  # Clean up
    
    def test_batch_result_creation(self):
        """Test BatchResult dataclass creation."""
        result = BatchResult(
            image_path='/path/to/image.jpg',
            task='task1',
            success=True,
            processing_time=1.5,
            output_path='/path/to/output.txt',
            detection_count=3,
            recognition_count=0
        )
        
        self.assertEqual(result.image_path, '/path/to/image.jpg')
        self.assertEqual(result.task, 'task1')
        self.assertTrue(result.success)
        self.assertEqual(result.processing_time, 1.5)
        self.assertEqual(result.detection_count, 3)
        self.assertIsNone(result.error_message)


class TestOutputManager(unittest.TestCase):
    """Test cases for the output manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.output_manager = OutputManager()
    
    def test_save_output(self):
        """Test output saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test_output.txt'
            content = "100,50,300,100,0.85\n150,120,250,160,0.92"
            
            self.output_manager.save_output(content, output_path, 'txt')
            
            # Verify file was created and content is correct
            self.assertTrue(output_path.exists())
            with open(output_path, 'r') as f:
                saved_content = f.read()
            self.assertEqual(saved_content, content)
    
    def test_format_json_output(self):
        """Test JSON formatting."""
        data = {
            'task': 'task1',
            'results': [{'x': 100, 'y': 50}],
            'count': 1
        }
        
        # Pretty formatting
        json_pretty = self.output_manager.format_json_output(data, pretty=True)
        self.assertIn('\n', json_pretty)  # Should have newlines for pretty format
        
        # Compact formatting
        json_compact = self.output_manager.format_json_output(data, pretty=False)
        self.assertNotIn('\n', json_compact)  # Should be compact
        
        # Verify content is the same
        self.assertEqual(json.loads(json_pretty), json.loads(json_compact))
    
    def test_create_output_directory(self):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            task = 'task1'
            
            output_dir = self.output_manager.create_output_directory(base_path, task)
            expected = base_path / task
            
            self.assertEqual(output_dir, expected)
            self.assertTrue(output_dir.exists())
            self.assertTrue(output_dir.is_dir())
    
    def test_get_output_filename(self):
        """Test output filename generation."""
        input_path = Path('/path/to/test_image.jpg')
        
        # Text format
        filename_txt = self.output_manager.get_output_filename(input_path, 'task1', 'txt')
        self.assertEqual(filename_txt, 'test_image_task1_results.txt')
        
        # JSON format
        filename_json = self.output_manager.get_output_filename(input_path, 'task2', 'json')
        self.assertEqual(filename_json, 'test_image_task2_results.json')


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for CLI components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = HistoricalMapCLI()
    
    @patch('src.cli.main_cli.Path')
    @patch('src.cli.main_cli.HistoricalMapCLI._process_single_image')
    @patch('src.cli.main_cli.OutputManager.save_output')
    def test_handle_process_command_integration(self, mock_save, mock_process, mock_path):
        """Test process command integration."""
        # Mock file existence
        mock_input_path = Mock()
        mock_input_path.exists.return_value = True
        mock_input_path.name = 'test.jpg'
        mock_path.return_value = mock_input_path
        
        # Mock processing result
        from src.models.data_structures import ProcessingResult
        mock_result = ProcessingResult(
            task_type='task1',
            detections=[],
            recognized_texts=[],
            linked_texts=[],
            processing_time=1.0,
            metrics={}
        )
        mock_process.return_value = mock_result
        
        # Create mock arguments
        args = Mock()
        args.input = 'test.jpg'
        args.task = 'task1'
        args.output = None
        args.config = None
        args.format = 'txt'
        args.save_intermediate = False
        args.confidence = 0.5
        
        # Test the command
        result = self.cli._handle_process_command(args)
        
        # Verify success
        self.assertEqual(result, 0)
        mock_process.assert_called_once()
        mock_save.assert_called_once()


if __name__ == '__main__':
    unittest.main()