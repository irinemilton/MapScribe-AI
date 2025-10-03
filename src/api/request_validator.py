"""
Request validation for the REST API.
"""

from typing import Set, Dict, Any, Optional
from werkzeug.datastructures import FileStorage
import re


class RequestValidator:
    """Validates API requests and parameters."""
    
    def __init__(self):
        """Initialize the validator."""
        self.valid_tasks = {'task1', 'task2', 'task3', 'task4'}
        self.valid_formats = {'txt', 'json'}
        self.valid_metrics = {'precision', 'recall', 'f1', 'iou', 'accuracy'}
    
    def validate_task(self, task: str) -> bool:
        """
        Validate competition task identifier.
        
        Args:
            task: Task identifier
            
        Returns:
            True if valid, False otherwise
        """
        return task in self.valid_tasks
    
    def validate_file(self, file: FileStorage, allowed_extensions: Set[str]) -> bool:
        """
        Validate uploaded file.
        
        Args:
            file: Uploaded file
            allowed_extensions: Set of allowed file extensions
            
        Returns:
            True if valid, False otherwise
        """
        if not file or not file.filename:
            return False
        
        # Check file extension
        if '.' not in file.filename:
            return False
        
        extension = file.filename.rsplit('.', 1)[1].lower()
        return extension in allowed_extensions
    
    def validate_confidence_threshold(self, confidence: float) -> bool:
        """
        Validate confidence threshold value.
        
        Args:
            confidence: Confidence threshold
            
        Returns:
            True if valid, False otherwise
        """
        return 0.0 <= confidence <= 1.0
    
    def validate_format(self, format_type: str) -> bool:
        """
        Validate output format.
        
        Args:
            format_type: Output format
            
        Returns:
            True if valid, False otherwise
        """
        return format_type in self.valid_formats
    
    def validate_metrics(self, metrics: list) -> bool:
        """
        Validate evaluation metrics list.
        
        Args:
            metrics: List of metric names
            
        Returns:
            True if valid, False otherwise
        """
        if not metrics:
            return False
        
        return all(metric in self.valid_metrics for metric in metrics)
    
    def validate_config_updates(self, config_updates: Dict[str, Any]) -> bool:
        """
        Validate configuration updates.
        
        Args:
            config_updates: Dictionary of configuration updates
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(config_updates, dict):
            return False
        
        # Define valid configuration keys and their types
        valid_config_keys = {
            'detection.confidence_threshold': (float, int),
            'detection.nms_threshold': (float, int),
            'detection.model_path': str,
            'recognition.confidence_threshold': (float, int),
            'recognition.beam_width': int,
            'recognition.languages': list,
            'preprocessing.enhance_contrast': bool,
            'preprocessing.noise_reduction': bool,
            'system.device': str,
            'system.num_workers': int
        }
        
        for key, value in config_updates.items():
            if key not in valid_config_keys:
                return False
            
            expected_type = valid_config_keys[key]
            if isinstance(expected_type, tuple):
                if not isinstance(value, expected_type):
                    return False
            else:
                if not isinstance(value, expected_type):
                    return False
        
        return True
    
    def validate_batch_size(self, batch_size: int, max_batch_size: int) -> bool:
        """
        Validate batch size.
        
        Args:
            batch_size: Number of files in batch
            max_batch_size: Maximum allowed batch size
            
        Returns:
            True if valid, False otherwise
        """
        return 1 <= batch_size <= max_batch_size
    
    def validate_result_id(self, result_id: str) -> bool:
        """
        Validate result ID format.
        
        Args:
            result_id: Result identifier
            
        Returns:
            True if valid, False otherwise
        """
        # UUID format validation
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        
        # Also allow batch result IDs (UUID_index format)
        batch_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_\d+$',
            re.IGNORECASE
        )
        
        return bool(uuid_pattern.match(result_id) or batch_pattern.match(result_id))
    
    def validate_evaluation_data(self, results_data: list, ground_truth_data: list) -> bool:
        """
        Validate evaluation data format.
        
        Args:
            results_data: List of result entries
            ground_truth_data: List of ground truth entries
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(results_data, list) or not isinstance(ground_truth_data, list):
            return False
        
        if not results_data or not ground_truth_data:
            return False
        
        # Validate result entries
        for result in results_data:
            if not isinstance(result, dict):
                return False
            
            required_fields = ['bbox', 'confidence']
            if not all(field in result for field in required_fields):
                return False
            
            # Validate bbox format
            bbox = result['bbox']
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return False
            
            # Validate confidence
            confidence = result['confidence']
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                return False
        
        # Similar validation for ground truth
        for gt in ground_truth_data:
            if not isinstance(gt, dict):
                return False
            
            if 'bbox' not in gt:
                return False
            
            bbox = gt['bbox']
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return False
        
        return True
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe storage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]
        
        # Remove or replace dangerous characters
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            max_name_length = 255 - len(ext) - 1 if ext else 255
            filename = name[:max_name_length] + ('.' + ext if ext else '')
        
        return filename
    
    def validate_processing_params(self, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate processing parameters and return error messages.
        
        Args:
            params: Processing parameters
            
        Returns:
            Dictionary of validation errors (empty if valid)
        """
        errors = {}
        
        # Validate confidence threshold
        if 'confidence_threshold' in params:
            confidence = params['confidence_threshold']
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                errors['confidence_threshold'] = 'Must be a number between 0 and 1'
        
        # Validate format
        if 'format' in params:
            format_type = params['format']
            if not self.validate_format(format_type):
                errors['format'] = f'Must be one of: {", ".join(self.valid_formats)}'
        
        # Validate save_intermediate
        if 'save_intermediate' in params:
            save_intermediate = params['save_intermediate']
            if not isinstance(save_intermediate, bool):
                errors['save_intermediate'] = 'Must be a boolean value'
        
        return errors