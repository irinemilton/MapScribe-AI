"""
Tests for ICDAR 2025 competition output formatter.
"""

import pytest
import tempfile
from pathlib import Path
from src.evaluation.competition_formatter import ICDAR2025Formatter
from src.models.data_structures import (
    BoundingBox, Detection, RecognizedText, ProcessingResult
)


class TestICDAR2025Formatter:
    """Test cases for ICDAR 2025 competition formatter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ICDAR2025Formatter()
        
        # Create sample bounding boxes
        self.bbox1 = BoundingBox(x=10, y=20, width=100, height=30, confidence=0.95)
        self.bbox2 = BoundingBox(x=150, y=50, width=80, height=25, confidence=0.87)
        
        # Create sample detections
        self.word_detection1 = Detection(
            bbox=self.bbox1,
            detection_type='word',
            confidence=0.95,
            image_coordinates=(10, 20, 110, 50)
        )
        
        self.word_detection2 = Detection(
            bbox=self.bbox2,
            detection_type='word', 
            confidence=0.87,
            image_coordinates=(150, 50, 230, 75)
        )
        
        self.phrase_detection1 = Detection(
            bbox=BoundingBox(x=10, y=20, width=220, height=30, confidence=0.92),
            detection_type='phrase',
            confidence=0.92,
            image_coordinates=(10, 20, 230, 50)
        )
        
        # Create sample recognized texts
        self.recognized_word1 = RecognizedText(
            detection=self.word_detection1,
            text="London",
            language="en",
            recognition_confidence=0.93,
            character_confidences=[0.95, 0.92, 0.94, 0.91, 0.96, 0.89]
        )
        
        self.recognized_word2 = RecognizedText(
            detection=self.word_detection2,
            text="Bridge",
            language="en", 
            recognition_confidence=0.88,
            character_confidences=[0.90, 0.87, 0.89, 0.85, 0.91, 0.86]
        )
        
        self.recognized_phrase1 = RecognizedText(
            detection=self.phrase_detection1,
            text="London Bridge",
            language="en",
            recognition_confidence=0.90,
            character_confidences=[0.95, 0.92, 0.94, 0.91, 0.96, 0.89, 0.88, 0.90, 0.87, 0.89, 0.85, 0.91, 0.86]
        )
    
    def test_format_task1_output(self):
        """Test Task 1 (Word Detection) output formatting."""
        detections = [self.word_detection1, self.word_detection2]
        output = self.formatter.format_task1_output(detections)
        
        expected_lines = [
            "10,20,110,50,0.9500",
            "150,50,230,75,0.8700"
        ]
        
        assert output == "\n".join(expected_lines)
    
    def test_format_task1_output_empty(self):
        """Test Task 1 output with empty detections."""
        output = self.formatter.format_task1_output([])
        assert output == ""
    
    def test_format_task1_output_filters_phrases(self):
        """Test Task 1 output filters out phrase detections."""
        detections = [self.word_detection1, self.phrase_detection1]
        output = self.formatter.format_task1_output(detections)
        
        # Should only include word detection
        expected = "10,20,110,50,0.9500"
        assert output == expected
    
    def test_format_task2_output(self):
        """Test Task 2 (Phrase Detection) output formatting."""
        detections = [self.phrase_detection1]
        output = self.formatter.format_task2_output(detections)
        
        expected = "10,20,230,50,0.9200"
        assert output == expected
    
    def test_format_task2_output_filters_words(self):
        """Test Task 2 output filters out word detections."""
        detections = [self.word_detection1, self.phrase_detection1]
        output = self.formatter.format_task2_output(detections)
        
        # Should only include phrase detection
        expected = "10,20,230,50,0.9200"
        assert output == expected
    
    def test_format_task3_output(self):
        """Test Task 3 (Word Detection + Recognition) output formatting."""
        recognized_texts = [self.recognized_word1, self.recognized_word2]
        output = self.formatter.format_task3_output(recognized_texts)
        
        expected_lines = [
            '10,20,110,50,0.9500,"London"',
            '150,50,230,75,0.8700,"Bridge"'
        ]
        
        assert output == "\n".join(expected_lines)
    
    def test_format_task3_output_with_special_characters(self):
        """Test Task 3 output with special characters in text."""
        # Create recognized text with quotes and commas
        text_content = 'Text with "quotes" and, commas'
        special_text = RecognizedText(
            detection=self.word_detection1,
            text=text_content,
            language="en",
            recognition_confidence=0.85,
            character_confidences=[0.9] * len(text_content)
        )
        
        output = self.formatter.format_task3_output([special_text])
        expected = '10,20,110,50,0.9500,"Text with ""quotes"" and, commas"'
        assert output == expected
    
    def test_format_task4_output(self):
        """Test Task 4 (Phrase Detection + Recognition) output formatting."""
        recognized_texts = [self.recognized_phrase1]
        output = self.formatter.format_task4_output(recognized_texts)
        
        expected = '10,20,230,50,0.9200,"London Bridge"'
        assert output == expected
    
    def test_format_bounding_box_coordinates(self):
        """Test bounding box coordinate formatting."""
        coords = self.formatter.format_bounding_box_coordinates(self.bbox1)
        expected = (10, 20, 110, 50)
        assert coords == expected
    
    def test_format_text_with_encoding(self):
        """Test text formatting with proper encoding."""
        # Test with regular text
        text = "London Bridge"
        formatted = self.formatter.format_text_with_encoding(text)
        assert formatted == "London Bridge"
        
        # Test with text containing quotes
        text_with_quotes = 'Text with "quotes"'
        formatted = self.formatter.format_text_with_encoding(text_with_quotes)
        assert formatted == 'Text with ""quotes""'
        
        # Test with Chinese characters
        chinese_text = "北京"
        formatted = self.formatter.format_text_with_encoding(chinese_text)
        assert formatted == "北京"
    
    def test_validate_task1_format_valid(self):
        """Test Task 1 format validation with valid input."""
        valid_output = "10,20,110,50,0.9500\n150,50,230,75,0.8700"
        assert self.formatter.validate_task1_format(valid_output) is True
        
        # Test empty output
        assert self.formatter.validate_task1_format("") is True
    
    def test_validate_task1_format_invalid(self):
        """Test Task 1 format validation with invalid input."""
        # Invalid number of fields
        invalid_output = "10,20,110,50"
        assert self.formatter.validate_task1_format(invalid_output) is False
        
        # Invalid coordinates (x2 <= x1)
        invalid_coords = "110,20,10,50,0.95"
        assert self.formatter.validate_task1_format(invalid_coords) is False
        
        # Invalid confidence (> 1.0)
        invalid_conf = "10,20,110,50,1.5"
        assert self.formatter.validate_task1_format(invalid_conf) is False
        
        # Non-numeric values
        invalid_numeric = "a,b,c,d,e"
        assert self.formatter.validate_task1_format(invalid_numeric) is False
    
    def test_validate_task3_format_valid(self):
        """Test Task 3 format validation with valid input."""
        valid_output = '10,20,110,50,0.9500,"London"\n150,50,230,75,0.8700,"Bridge"'
        assert self.formatter.validate_task3_format(valid_output) is True
        
        # Test with escaped quotes
        valid_escaped = '10,20,110,50,0.9500,"Text with ""quotes"""'
        assert self.formatter.validate_task3_format(valid_escaped) is True
    
    def test_validate_task3_format_invalid(self):
        """Test Task 3 format validation with invalid input."""
        # Missing quotes around text
        invalid_quotes = "10,20,110,50,0.9500,London"
        assert self.formatter.validate_task3_format(invalid_quotes) is False
        
        # Invalid number of fields
        invalid_fields = '10,20,110,50,0.9500'
        assert self.formatter.validate_task3_format(invalid_fields) is False
    
    def test_validate_competition_format(self):
        """Test general competition format validation."""
        valid_task1 = "10,20,110,50,0.9500"
        assert self.formatter.validate_competition_format('task1', valid_task1) is True
        
        valid_task3 = '10,20,110,50,0.9500,"London"'
        assert self.formatter.validate_competition_format('task3', valid_task3) is True
        
        # Test invalid task ID
        with pytest.raises(ValueError):
            self.formatter.validate_competition_format('invalid_task', valid_task1)
    
    def test_save_competition_output(self):
        """Test saving competition output to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "task1_output.txt"
            valid_output = "10,20,110,50,0.9500"
            
            self.formatter.save_competition_output('task1', valid_output, output_path)
            
            # Verify file was created and contains correct content
            assert output_path.exists()
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert content == valid_output
    
    def test_save_competition_output_invalid_format(self):
        """Test saving competition output with invalid format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "task1_output.txt"
            invalid_output = "invalid,format"
            
            with pytest.raises(ValueError):
                self.formatter.save_competition_output('task1', invalid_output, output_path)
    
    def test_format_processing_result(self):
        """Test formatting ProcessingResult for different tasks."""
        result = ProcessingResult(
            task_type='word_detection_recognition',
            detections=[self.word_detection1, self.word_detection2],
            recognized_texts=[self.recognized_word1, self.recognized_word2],
            linked_texts=[],
            processing_time=1.5,
            metrics={'precision': 0.95, 'recall': 0.87}
        )
        
        # Test Task 1 formatting
        task1_output = self.formatter.format_processing_result(result, 'task1')
        expected_task1 = "10,20,110,50,0.9500\n150,50,230,75,0.8700"
        assert task1_output == expected_task1
        
        # Test Task 3 formatting
        task3_output = self.formatter.format_processing_result(result, 'task3')
        expected_task3 = '10,20,110,50,0.9500,"London"\n150,50,230,75,0.8700,"Bridge"'
        assert task3_output == expected_task3
    
    def test_get_task_info(self):
        """Test getting task information."""
        task1_info = self.formatter.get_task_info('task1')
        expected_info = {
            'name': 'Word Detection',
            'output_format': 'bbox_only',
            'file_extension': '.txt'
        }
        assert task1_info == expected_info
        
        # Test invalid task ID
        with pytest.raises(ValueError):
            self.formatter.get_task_info('invalid_task')
    
    def test_get_supported_tasks(self):
        """Test getting list of supported tasks."""
        tasks = self.formatter.get_supported_tasks()
        expected_tasks = ['task1', 'task2', 'task3', 'task4']
        assert tasks == expected_tasks
    
    def test_escape_text_for_output(self):
        """Test text escaping for CSV output."""
        # Test regular text
        regular_text = "London Bridge"
        escaped = self.formatter._escape_text_for_output(regular_text)
        assert escaped == "London Bridge"
        
        # Test text with quotes
        quoted_text = 'Text with "quotes"'
        escaped = self.formatter._escape_text_for_output(quoted_text)
        assert escaped == 'Text with ""quotes""'
        
        # Test text with multiple quotes
        multi_quotes = 'Text with "multiple" "quotes"'
        escaped = self.formatter._escape_text_for_output(multi_quotes)
        assert escaped == 'Text with ""multiple"" ""quotes""'
    
    def test_parse_csv_line(self):
        """Test CSV line parsing."""
        # Test simple line
        simple_line = "10,20,110,50,0.95"
        parsed = self.formatter._parse_csv_line(simple_line)
        assert parsed == ["10", "20", "110", "50", "0.95"]
        
        # Test line with quoted text
        quoted_line = '10,20,110,50,0.95,"London Bridge"'
        parsed = self.formatter._parse_csv_line(quoted_line)
        assert parsed == ["10", "20", "110", "50", "0.95", "London Bridge"]
        
        # Test line with escaped quotes
        escaped_line = '10,20,110,50,0.95,"Text with ""quotes"""'
        parsed = self.formatter._parse_csv_line(escaped_line)
        assert parsed == ["10", "20", "110", "50", "0.95", 'Text with "quotes"']


if __name__ == "__main__":
    pytest.main([__file__])