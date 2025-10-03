"""
Competition validation module for ICDAR 2025 Historical Map Text Processing.
Validates output formats, requirements compliance, and submission preparation.
"""

import re
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from dataclasses import dataclass

from ..models.data_structures import ProcessingResult, Detection, RecognizedText


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]


class ICDAR2025Validator:
    """
    Validator for ICDAR 2025 competition requirements and output formats.
    Ensures compliance with all competition specifications.
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.logger = logging.getLogger(__name__)
        
        # Competition task definitions
        self.task_definitions = {
            'task1': {
                'name': 'Word Detection',
                'output_format': 'x,y,width,height,confidence',
                'file_extension': '.txt',
                'requires_text': False,
                'detection_level': 'word'
            },
            'task2': {
                'name': 'Phrase Detection',
                'output_format': 'x,y,width,height,confidence',
                'file_extension': '.txt',
                'requires_text': False,
                'detection_level': 'phrase'
            },
            'task3': {
                'name': 'Word Detection + Recognition',
                'output_format': 'x,y,width,height,confidence,text',
                'file_extension': '.txt',
                'requires_text': True,
                'detection_level': 'word'
            },
            'task4': {
                'name': 'Phrase Detection + Recognition',
                'output_format': 'x,y,width,height,confidence,text',
                'file_extension': '.txt',
                'requires_text': True,
                'detection_level': 'phrase'
            }
        }
        
        # Validation thresholds
        self.min_confidence = 0.0
        self.max_confidence = 1.0
        self.min_bbox_size = 1
        self.max_text_length = 1000
    
    def validate_output_format(self, output_content: str, task: str) -> ValidationResult:
        """
        Validate output format compliance for a specific task.
        
        Args:
            output_content: Content of the output file
            task: Competition task (task1, task2, task3, task4)
            
        Returns:
            ValidationResult: Validation result with errors and warnings
        """
        errors = []
        warnings = []
        details = {}
        
        try:
            if task not in self.task_definitions:
                errors.append(f"Invalid task: {task}")
                return ValidationResult(False, errors, warnings, details)
            
            task_def = self.task_definitions[task]
            lines = output_content.strip().split('\n')
            
            if not lines or (len(lines) == 1 and not lines[0].strip()):
                warnings.append("Output file is empty")
                details['line_count'] = 0
                return ValidationResult(True, errors, warnings, details)
            
            details['line_count'] = len(lines)
            valid_lines = 0
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Validate line format
                line_validation = self._validate_output_line(line, task_def, line_num)
                if line_validation.errors:
                    errors.extend(line_validation.errors)
                if line_validation.warnings:
                    warnings.extend(line_validation.warnings)
                
                if line_validation.is_valid:
                    valid_lines += 1
            
            details['valid_lines'] = valid_lines
            details['invalid_lines'] = len(lines) - valid_lines
            
            # Check if any valid lines exist
            if valid_lines == 0 and len(lines) > 0:
                errors.append("No valid output lines found")
            
            is_valid = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            is_valid = False
        
        return ValidationResult(is_valid, errors, warnings, details)
    
    def validate_processing_result(self, result: ProcessingResult, task: str) -> ValidationResult:
        """
        Validate a ProcessingResult object for competition compliance.
        
        Args:
            result: ProcessingResult to validate
            task: Competition task
            
        Returns:
            ValidationResult: Validation result
        """
        errors = []
        warnings = []
        details = {}
        
        try:
            # Validate task type
            if result.task_type != task:
                errors.append(f"Task type mismatch: expected {task}, got {result.task_type}")
            
            # Validate detections
            if not result.detections:
                warnings.append("No detections found")
            else:
                detection_validation = self._validate_detections(result.detections, task)
                errors.extend(detection_validation.errors)
                warnings.extend(detection_validation.warnings)
                details.update(detection_validation.details)
            
            # Validate recognition results for tasks 3 and 4
            if task in ['task3', 'task4']:
                if not result.recognized_texts:
                    warnings.append("No recognized texts found for recognition task")
                else:
                    recognition_validation = self._validate_recognized_texts(result.recognized_texts)
                    errors.extend(recognition_validation.errors)
                    warnings.extend(recognition_validation.warnings)
                    details.update(recognition_validation.details)
            
            # Validate processing metrics
            if result.processing_time <= 0:
                warnings.append("Invalid processing time")
            
            details['detection_count'] = len(result.detections)
            details['recognition_count'] = len(result.recognized_texts)
            details['processing_time'] = result.processing_time
            
            is_valid = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Result validation error: {str(e)}")
            is_valid = False
        
        return ValidationResult(is_valid, errors, warnings, details)
    
    def validate_submission_package(self, submission_dir: Path) -> ValidationResult:
        """
        Validate a complete submission package.
        
        Args:
            submission_dir: Directory containing submission files
            
        Returns:
            ValidationResult: Validation result for entire submission
        """
        errors = []
        warnings = []
        details = {}
        
        try:
            if not submission_dir.exists():
                errors.append(f"Submission directory not found: {submission_dir}")
                return ValidationResult(False, errors, warnings, details)
            
            # Check for required files and structure
            structure_validation = self._validate_submission_structure(submission_dir)
            errors.extend(structure_validation.errors)
            warnings.extend(structure_validation.warnings)
            details.update(structure_validation.details)
            
            # Validate individual result files
            result_files = list(submission_dir.glob("*.txt"))
            details['result_file_count'] = len(result_files)
            
            file_validations = {}
            for result_file in result_files:
                file_validation = self._validate_result_file(result_file)
                file_validations[result_file.name] = file_validation
                
                if file_validation.errors:
                    errors.extend([f"{result_file.name}: {error}" for error in file_validation.errors])
                if file_validation.warnings:
                    warnings.extend([f"{result_file.name}: {warning}" for warning in file_validation.warnings])
            
            details['file_validations'] = file_validations
            
            is_valid = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Submission validation error: {str(e)}")
            is_valid = False
        
        return ValidationResult(is_valid, errors, warnings, details)
    
    def generate_validation_report(self, validation_results: Dict[str, ValidationResult], 
                                 output_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            validation_results: Dictionary of validation results
            output_path: Optional path to save the report
            
        Returns:
            Report content as string
        """
        report_lines = []
        report_lines.append("ICDAR 2025 Competition Validation Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")
        
        # Summary
        total_validations = len(validation_results)
        valid_count = sum(1 for result in validation_results.values() if result.is_valid)
        invalid_count = total_validations - valid_count
        
        report_lines.append("SUMMARY")
        report_lines.append("-" * 20)
        report_lines.append(f"Total validations: {total_validations}")
        report_lines.append(f"Valid: {valid_count}")
        report_lines.append(f"Invalid: {invalid_count}")
        report_lines.append(f"Success rate: {valid_count/total_validations*100:.1f}%")
        report_lines.append("")
        
        # Detailed results
        report_lines.append("DETAILED RESULTS")
        report_lines.append("-" * 20)
        
        for name, result in validation_results.items():
            report_lines.append(f"\n{name}:")
            report_lines.append(f"  Status: {'VALID' if result.is_valid else 'INVALID'}")
            
            if result.errors:
                report_lines.append("  Errors:")
                for error in result.errors:
                    report_lines.append(f"    - {error}")
            
            if result.warnings:
                report_lines.append("  Warnings:")
                for warning in result.warnings:
                    report_lines.append(f"    - {warning}")
            
            if result.details:
                report_lines.append("  Details:")
                for key, value in result.details.items():
                    report_lines.append(f"    {key}: {value}")
        
        report_content = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.info(f"Validation report saved to: {output_path}")
        
        return report_content
    
    def _validate_output_line(self, line: str, task_def: Dict[str, Any], line_num: int) -> ValidationResult:
        """Validate a single output line."""
        errors = []
        warnings = []
        details = {}
        
        try:
            parts = line.split(',')
            expected_parts = 6 if task_def['requires_text'] else 5
            
            if len(parts) != expected_parts:
                errors.append(f"Line {line_num}: Expected {expected_parts} parts, got {len(parts)}")
                return ValidationResult(False, errors, warnings, details)
            
            # Validate bounding box coordinates
            try:
                x, y, width, height = map(float, parts[:4])
                
                if x < 0 or y < 0:
                    errors.append(f"Line {line_num}: Negative coordinates not allowed")
                
                if width < self.min_bbox_size or height < self.min_bbox_size:
                    errors.append(f"Line {line_num}: Bounding box too small")
                
                details['bbox'] = {'x': x, 'y': y, 'width': width, 'height': height}
                
            except ValueError:
                errors.append(f"Line {line_num}: Invalid bounding box coordinates")
            
            # Validate confidence score
            try:
                confidence = float(parts[4])
                
                if not (self.min_confidence <= confidence <= self.max_confidence):
                    errors.append(f"Line {line_num}: Confidence {confidence} out of range [0,1]")
                
                details['confidence'] = confidence
                
            except ValueError:
                errors.append(f"Line {line_num}: Invalid confidence score")
            
            # Validate text (for recognition tasks)
            if task_def['requires_text']:
                text = parts[5].strip()
                
                if not text:
                    warnings.append(f"Line {line_num}: Empty text field")
                elif len(text) > self.max_text_length:
                    warnings.append(f"Line {line_num}: Text too long ({len(text)} chars)")
                
                details['text'] = text
                details['text_length'] = len(text)
            
            is_valid = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Line {line_num}: Parsing error - {str(e)}")
            is_valid = False
        
        return ValidationResult(is_valid, errors, warnings, details)
    
    def _validate_detections(self, detections: List[Detection], task: str) -> ValidationResult:
        """Validate detection objects."""
        errors = []
        warnings = []
        details = {}
        
        task_def = self.task_definitions[task]
        expected_type = task_def['detection_level']
        
        valid_detections = 0
        confidence_scores = []
        
        for i, detection in enumerate(detections):
            # Validate detection type
            if detection.detection_type != expected_type:
                warnings.append(f"Detection {i}: Expected {expected_type}, got {detection.detection_type}")
            
            # Validate bounding box
            bbox = detection.bbox
            if bbox.width < self.min_bbox_size or bbox.height < self.min_bbox_size:
                errors.append(f"Detection {i}: Bounding box too small")
            
            if bbox.x < 0 or bbox.y < 0:
                errors.append(f"Detection {i}: Negative coordinates")
            
            # Validate confidence
            if not (self.min_confidence <= detection.confidence <= self.max_confidence):
                errors.append(f"Detection {i}: Invalid confidence {detection.confidence}")
            else:
                confidence_scores.append(detection.confidence)
                valid_detections += 1
        
        details['valid_detections'] = valid_detections
        details['total_detections'] = len(detections)
        
        if confidence_scores:
            details['avg_confidence'] = sum(confidence_scores) / len(confidence_scores)
            details['min_confidence'] = min(confidence_scores)
            details['max_confidence'] = max(confidence_scores)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, details)
    
    def _validate_recognized_texts(self, recognized_texts: List[RecognizedText]) -> ValidationResult:
        """Validate recognized text objects."""
        errors = []
        warnings = []
        details = {}
        
        valid_texts = 0
        text_lengths = []
        recognition_confidences = []
        
        for i, recognized_text in enumerate(recognized_texts):
            # Validate text content
            if not recognized_text.text.strip():
                warnings.append(f"Recognition {i}: Empty text")
            else:
                text_lengths.append(len(recognized_text.text))
                valid_texts += 1
            
            # Validate recognition confidence
            if not (self.min_confidence <= recognized_text.recognition_confidence <= self.max_confidence):
                errors.append(f"Recognition {i}: Invalid confidence {recognized_text.recognition_confidence}")
            else:
                recognition_confidences.append(recognized_text.recognition_confidence)
            
            # Validate language
            if not recognized_text.language:
                warnings.append(f"Recognition {i}: Missing language information")
        
        details['valid_texts'] = valid_texts
        details['total_texts'] = len(recognized_texts)
        
        if text_lengths:
            details['avg_text_length'] = sum(text_lengths) / len(text_lengths)
            details['max_text_length'] = max(text_lengths)
        
        if recognition_confidences:
            details['avg_recognition_confidence'] = sum(recognition_confidences) / len(recognition_confidences)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, details)
    
    def _validate_submission_structure(self, submission_dir: Path) -> ValidationResult:
        """Validate submission directory structure."""
        errors = []
        warnings = []
        details = {}
        
        # Check for required files
        required_files = ['README.md', 'results']
        optional_files = ['metadata.json', 'model_info.txt']
        
        existing_files = [f.name for f in submission_dir.iterdir()]
        details['existing_files'] = existing_files
        
        # Check required files
        missing_required = []
        for required_file in required_files:
            if required_file not in existing_files:
                missing_required.append(required_file)
        
        if missing_required:
            errors.extend([f"Missing required file: {f}" for f in missing_required])
        
        # Check for results directory or files
        results_dir = submission_dir / 'results'
        result_files = list(submission_dir.glob('*.txt'))
        
        if not results_dir.exists() and not result_files:
            errors.append("No results directory or result files found")
        
        details['has_results_dir'] = results_dir.exists()
        details['result_file_count'] = len(result_files)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, details)
    
    def _validate_result_file(self, result_file: Path) -> ValidationResult:
        """Validate individual result file."""
        errors = []
        warnings = []
        details = {}
        
        try:
            # Determine task from filename
            task = self._extract_task_from_filename(result_file.name)
            if not task:
                warnings.append("Could not determine task from filename")
                task = 'task1'  # Default assumption
            
            # Read and validate content
            with open(result_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            validation_result = self.validate_output_format(content, task)
            errors.extend(validation_result.errors)
            warnings.extend(validation_result.warnings)
            details.update(validation_result.details)
            
            details['file_size'] = result_file.stat().st_size
            details['detected_task'] = task
            
        except Exception as e:
            errors.append(f"File validation error: {str(e)}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, details)
    
    def _extract_task_from_filename(self, filename: str) -> Optional[str]:
        """Extract task identifier from filename."""
        # Look for task patterns in filename
        task_patterns = [
            r'task[_-]?([1-4])',
            r't([1-4])',
            r'(\d)\.txt$'
        ]
        
        for pattern in task_patterns:
            match = re.search(pattern, filename.lower())
            if match:
                task_num = match.group(1)
                return f'task{task_num}'
        
        return None