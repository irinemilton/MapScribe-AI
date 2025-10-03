"""
ICDAR 2025 competition output formatter for historical map text processing.
Implements format compliance for all four competition tasks.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import csv
from ..models.data_structures import Detection, RecognizedText, BoundingBox, ProcessingResult


class ICDAR2025Formatter:
    """
    Formatter for ICDAR 2025 Historical Map Text Detection, Recognition, and Linking challenge.
    Supports all four competition tasks with proper output formatting.
    """
    
    def __init__(self):
        """Initialize the formatter with task-specific configurations."""
        self.task_configs = {
            'task1': {
                'name': 'Word Detection',
                'output_format': 'bbox_only',
                'file_extension': '.txt'
            },
            'task2': {
                'name': 'Phrase Detection', 
                'output_format': 'bbox_only',
                'file_extension': '.txt'
            },
            'task3': {
                'name': 'Word Detection + Recognition',
                'output_format': 'bbox_with_text',
                'file_extension': '.txt'
            },
            'task4': {
                'name': 'Phrase Detection + Recognition',
                'output_format': 'bbox_with_text', 
                'file_extension': '.txt'
            }
        }
    
    def format_task1_output(self, detections: List[Detection], image_name: str = "") -> str:
        """
        Format Task 1 (Word Detection) output according to ICDAR 2025 specifications.
        
        Args:
            detections: List of word-level detections
            image_name: Name of the processed image
            
        Returns:
            Formatted output string for Task 1
        """
        if not detections:
            return ""
        
        # Filter for word-level detections only
        word_detections = [d for d in detections if d.detection_type == 'word']
        
        output_lines = []
        for detection in word_detections:
            bbox = detection.bbox
            # ICDAR format: x1,y1,x2,y2,confidence
            x1, y1 = bbox.x, bbox.y
            x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height
            confidence = detection.confidence
            
            line = f"{x1},{y1},{x2},{y2},{confidence:.4f}"
            output_lines.append(line)
        
        return "\n".join(output_lines)
    
    def format_task2_output(self, detections: List[Detection], image_name: str = "") -> str:
        """
        Format Task 2 (Phrase Detection) output according to ICDAR 2025 specifications.
        
        Args:
            detections: List of phrase-level detections
            image_name: Name of the processed image
            
        Returns:
            Formatted output string for Task 2
        """
        if not detections:
            return ""
        
        # Filter for phrase-level detections only
        phrase_detections = [d for d in detections if d.detection_type == 'phrase']
        
        output_lines = []
        for detection in phrase_detections:
            bbox = detection.bbox
            # ICDAR format: x1,y1,x2,y2,confidence
            x1, y1 = bbox.x, bbox.y
            x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height
            confidence = detection.confidence
            
            line = f"{x1},{y1},{x2},{y2},{confidence:.4f}"
            output_lines.append(line)
        
        return "\n".join(output_lines)
    
    def format_task3_output(self, recognized_texts: List[RecognizedText], image_name: str = "") -> str:
        """
        Format Task 3 (Word Detection + Recognition) output according to ICDAR 2025 specifications.
        
        Args:
            recognized_texts: List of recognized word-level text
            image_name: Name of the processed image
            
        Returns:
            Formatted output string for Task 3
        """
        if not recognized_texts:
            return ""
        
        # Filter for word-level recognition only
        word_texts = [rt for rt in recognized_texts if rt.detection.detection_type == 'word']
        
        output_lines = []
        for recognized_text in word_texts:
            bbox = recognized_text.detection.bbox
            # ICDAR format: x1,y1,x2,y2,confidence,"text"
            x1, y1 = bbox.x, bbox.y
            x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height
            confidence = recognized_text.detection.confidence
            text = self._escape_text_for_output(recognized_text.text)
            
            line = f'{x1},{y1},{x2},{y2},{confidence:.4f},"{text}"'
            output_lines.append(line)
        
        return "\n".join(output_lines)
    
    def format_task4_output(self, recognized_texts: List[RecognizedText], image_name: str = "") -> str:
        """
        Format Task 4 (Phrase Detection + Recognition) output according to ICDAR 2025 specifications.
        
        Args:
            recognized_texts: List of recognized phrase-level text
            image_name: Name of the processed image
            
        Returns:
            Formatted output string for Task 4
        """
        if not recognized_texts:
            return ""
        
        # Filter for phrase-level recognition only
        phrase_texts = [rt for rt in recognized_texts if rt.detection.detection_type == 'phrase']
        
        output_lines = []
        for recognized_text in phrase_texts:
            bbox = recognized_text.detection.bbox
            # ICDAR format: x1,y1,x2,y2,confidence,"text"
            x1, y1 = bbox.x, bbox.y
            x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height
            confidence = recognized_text.detection.confidence
            text = self._escape_text_for_output(recognized_text.text)
            
            line = f'{x1},{y1},{x2},{y2},{confidence:.4f},"{text}"'
            output_lines.append(line)
        
        return "\n".join(output_lines)
    
    def format_bounding_box_coordinates(self, bbox: BoundingBox) -> Tuple[int, int, int, int]:
        """
        Format bounding box coordinates according to ICDAR 2025 specifications.
        
        Args:
            bbox: BoundingBox object
            
        Returns:
            Tuple of (x1, y1, x2, y2) coordinates
        """
        x1, y1 = bbox.x, bbox.y
        x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height
        return (x1, y1, x2, y2)
    
    def format_text_with_encoding(self, text: str, encoding: str = 'utf-8') -> str:
        """
        Format text output with proper encoding for competition submission.
        
        Args:
            text: Input text string
            encoding: Text encoding (default: utf-8)
            
        Returns:
            Properly encoded text string
        """
        try:
            # Ensure proper UTF-8 encoding for multilingual text
            encoded_text = text.encode(encoding).decode(encoding)
            return self._escape_text_for_output(encoded_text)
        except UnicodeError:
            # Fallback to ASCII with replacement characters
            return text.encode('ascii', errors='replace').decode('ascii')
    
    def validate_task1_format(self, output: str) -> bool:
        """
        Validate Task 1 output format compliance.
        
        Args:
            output: Formatted output string
            
        Returns:
            True if format is valid, False otherwise
        """
        if not output.strip():
            return True  # Empty output is valid
        
        lines = output.strip().split('\n')
        for line in lines:
            parts = line.split(',')
            if len(parts) != 5:
                return False
            
            try:
                # Validate coordinates and confidence
                x1, y1, x2, y2, conf = map(float, parts)
                if x2 <= x1 or y2 <= y1:
                    return False
                if conf < 0.0 or conf > 1.0:
                    return False
            except ValueError:
                return False
        
        return True
    
    def validate_task2_format(self, output: str) -> bool:
        """
        Validate Task 2 output format compliance.
        
        Args:
            output: Formatted output string
            
        Returns:
            True if format is valid, False otherwise
        """
        # Task 2 has same format as Task 1
        return self.validate_task1_format(output)
    
    def validate_task3_format(self, output: str) -> bool:
        """
        Validate Task 3 output format compliance.
        
        Args:
            output: Formatted output string
            
        Returns:
            True if format is valid, False otherwise
        """
        if not output.strip():
            return True  # Empty output is valid
        
        lines = output.strip().split('\n')
        for line in lines:
            # Check if line has proper quote structure for text field
            if not (line.count('"') >= 2 and line.endswith('"')):
                return False
            
            # Split by comma, but preserve quoted text
            parts = self._parse_csv_line(line)
            if len(parts) != 6:
                return False
            
            try:
                # Validate coordinates and confidence
                x1, y1, x2, y2, conf = map(float, parts[:5])
                if x2 <= x1 or y2 <= y1:
                    return False
                if conf < 0.0 or conf > 1.0:
                    return False
                
                # Text field validation - should be the 6th field (already unquoted by CSV parser)
                text_field = parts[5]
                # Text field should not be empty
                if not text_field:
                    return False
                    
            except (ValueError, IndexError):
                return False
        
        return True
    
    def validate_task4_format(self, output: str) -> bool:
        """
        Validate Task 4 output format compliance.
        
        Args:
            output: Formatted output string
            
        Returns:
            True if format is valid, False otherwise
        """
        # Task 4 has same format as Task 3
        return self.validate_task3_format(output)
    
    def validate_competition_format(self, task_id: str, output: str) -> bool:
        """
        Validate output format for specified competition task.
        
        Args:
            task_id: Task identifier ('task1', 'task2', 'task3', 'task4')
            output: Formatted output string
            
        Returns:
            True if format is valid, False otherwise
        """
        validation_methods = {
            'task1': self.validate_task1_format,
            'task2': self.validate_task2_format,
            'task3': self.validate_task3_format,
            'task4': self.validate_task4_format
        }
        
        if task_id not in validation_methods:
            raise ValueError(f"Invalid task ID: {task_id}")
        
        return validation_methods[task_id](output)
    
    def save_competition_output(self, task_id: str, output: str, output_path: Path, image_name: str = "") -> None:
        """
        Save formatted output to file according to competition requirements.
        
        Args:
            task_id: Task identifier ('task1', 'task2', 'task3', 'task4')
            output: Formatted output string
            output_path: Path to save the output file
            image_name: Name of the processed image
        """
        if task_id not in self.task_configs:
            raise ValueError(f"Invalid task ID: {task_id}")
        
        # Validate format before saving
        if not self.validate_competition_format(task_id, output):
            raise ValueError(f"Output format validation failed for {task_id}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with proper encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)
    
    def format_processing_result(self, result: ProcessingResult, task_id: str, image_name: str = "") -> str:
        """
        Format a ProcessingResult object for the specified competition task.
        
        Args:
            result: ProcessingResult object containing all processing outputs
            task_id: Task identifier ('task1', 'task2', 'task3', 'task4')
            image_name: Name of the processed image
            
        Returns:
            Formatted output string for the specified task
        """
        if task_id == 'task1':
            return self.format_task1_output(result.detections, image_name)
        elif task_id == 'task2':
            return self.format_task2_output(result.detections, image_name)
        elif task_id == 'task3':
            return self.format_task3_output(result.recognized_texts, image_name)
        elif task_id == 'task4':
            return self.format_task4_output(result.recognized_texts, image_name)
        else:
            raise ValueError(f"Invalid task ID: {task_id}")
    
    def _escape_text_for_output(self, text: str) -> str:
        """
        Escape text for proper CSV output formatting.
        
        Args:
            text: Input text string
            
        Returns:
            Escaped text string
        """
        # Escape double quotes by doubling them
        escaped = text.replace('"', '""')
        return escaped
    
    def _parse_csv_line(self, line: str) -> List[str]:
        """
        Parse a CSV line while preserving quoted text fields.
        
        Args:
            line: CSV line string
            
        Returns:
            List of parsed fields
        """
        import csv
        import io
        
        # Use CSV reader to properly parse quoted fields
        reader = csv.reader(io.StringIO(line))
        try:
            return next(reader)
        except StopIteration:
            return []
    
    def get_task_info(self, task_id: str) -> Dict[str, Any]:
        """
        Get information about a specific competition task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Dictionary containing task information
        """
        if task_id not in self.task_configs:
            raise ValueError(f"Invalid task ID: {task_id}")
        
        return self.task_configs[task_id].copy()
    
    def get_supported_tasks(self) -> List[str]:
        """
        Get list of supported competition tasks.
        
        Returns:
            List of task identifiers
        """
        return list(self.task_configs.keys())