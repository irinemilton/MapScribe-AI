"""
Response formatting for the REST API.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from ..models.data_structures import ProcessingResult, Detection, RecognizedText


class ResponseFormatter:
    """Formats API responses in consistent structure."""
    
    def __init__(self):
        """Initialize the response formatter."""
        pass
    
    def format_processing_response(self, result_id: str, task: str, filename: str,
                                 processing_result: ProcessingResult,
                                 result_file_path: Path) -> Dict[str, Any]:
        """
        Format single image processing response.
        
        Args:
            result_id: Unique result identifier
            task: Competition task
            filename: Original filename
            processing_result: Processing result object
            result_file_path: Path to saved result file
            
        Returns:
            Formatted response dictionary
        """
        response = {
            'result_id': result_id,
            'task': task,
            'task_name': self._get_task_name(task),
            'filename': filename,
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_result.processing_time,
            'results': {
                'detections': self._format_detections(processing_result.detections),
                'recognized_texts': self._format_recognized_texts(processing_result.recognized_texts),
                'statistics': {
                    'detection_count': len(processing_result.detections),
                    'recognition_count': len(processing_result.recognized_texts),
                    'average_confidence': self._calculate_average_confidence(processing_result)
                }
            },
            'download_url': f'/results/{result_id}/download',
            'result_url': f'/results/{result_id}'
        }
        
        # Add metrics if available
        if processing_result.metrics:
            response['metrics'] = processing_result.metrics
        
        return response
    
    def format_batch_response(self, batch_id: str, task: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format batch processing response.
        
        Args:
            batch_id: Batch identifier
            task: Competition task
            results: List of individual results
            
        Returns:
            Formatted batch response
        """
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        response = {
            'batch_id': batch_id,
            'task': task,
            'task_name': self._get_task_name(task),
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files': len(results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'success_rate': len(successful_results) / len(results) if results else 0,
                'total_processing_time': sum(r.get('processing_time', 0) for r in successful_results),
                'average_processing_time': (
                    sum(r.get('processing_time', 0) for r in successful_results) / len(successful_results)
                    if successful_results else 0
                )
            },
            'statistics': {
                'total_detections': sum(r.get('detection_count', 0) for r in successful_results),
                'total_recognitions': sum(r.get('recognition_count', 0) for r in successful_results),
                'average_detections_per_image': (
                    sum(r.get('detection_count', 0) for r in successful_results) / len(successful_results)
                    if successful_results else 0
                ),
                'average_recognitions_per_image': (
                    sum(r.get('recognition_count', 0) for r in successful_results) / len(successful_results)
                    if successful_results else 0
                )
            },
            'results': results
        }
        
        return response
    
    def format_error_response(self, error_message: str, error_code: str = None,
                            details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Format error response.
        
        Args:
            error_message: Error message
            error_code: Optional error code
            details: Optional error details
            
        Returns:
            Formatted error response
        """
        response = {
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'status': 'error'
        }
        
        if error_code:
            response['error_code'] = error_code
        
        if details:
            response['details'] = details
        
        return response
    
    def format_evaluation_response(self, task: str, metrics: Dict[str, float],
                                 total_predictions: int, total_ground_truth: int) -> Dict[str, Any]:
        """
        Format evaluation response.
        
        Args:
            task: Competition task
            metrics: Evaluation metrics
            total_predictions: Number of predictions
            total_ground_truth: Number of ground truth entries
            
        Returns:
            Formatted evaluation response
        """
        response = {
            'task': task,
            'task_name': self._get_task_name(task),
            'evaluation': {
                'metrics': metrics,
                'summary': {
                    'total_predictions': total_predictions,
                    'total_ground_truth': total_ground_truth,
                    'evaluation_timestamp': datetime.now().isoformat()
                }
            },
            'status': 'completed'
        }
        
        return response
    
    def format_model_info_response(self, models: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Format model information response.
        
        Args:
            models: Dictionary of model information
            
        Returns:
            Formatted model info response
        """
        response = {
            'models': models,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return response
    
    def format_config_response(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format configuration response.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Formatted configuration response
        """
        response = {
            'configuration': config,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return response
    
    def format_health_response(self, status: str, services: Dict[str, str],
                             version: str = '1.0.0') -> Dict[str, Any]:
        """
        Format health check response.
        
        Args:
            status: Overall health status
            services: Service status dictionary
            version: API version
            
        Returns:
            Formatted health response
        """
        response = {
            'status': status,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'services': services
        }
        
        return response
    
    def _format_detections(self, detections: List[Detection]) -> List[Dict[str, Any]]:
        """
        Format detection objects for API response.
        
        Args:
            detections: List of detection objects
            
        Returns:
            List of formatted detection dictionaries
        """
        formatted_detections = []
        
        for detection in detections:
            formatted_detection = {
                'bbox': {
                    'x': detection.bbox.x,
                    'y': detection.bbox.y,
                    'width': detection.bbox.width,
                    'height': detection.bbox.height
                },
                'confidence': detection.confidence,
                'type': detection.detection_type,
                'coordinates': list(detection.image_coordinates)
            }
            formatted_detections.append(formatted_detection)
        
        return formatted_detections
    
    def _format_recognized_texts(self, recognized_texts: List[RecognizedText]) -> List[Dict[str, Any]]:
        """
        Format recognized text objects for API response.
        
        Args:
            recognized_texts: List of recognized text objects
            
        Returns:
            List of formatted recognized text dictionaries
        """
        formatted_texts = []
        
        for recognized_text in recognized_texts:
            formatted_text = {
                'text': recognized_text.text,
                'language': recognized_text.language,
                'confidence': recognized_text.recognition_confidence,
                'bbox': {
                    'x': recognized_text.detection.bbox.x,
                    'y': recognized_text.detection.bbox.y,
                    'width': recognized_text.detection.bbox.width,
                    'height': recognized_text.detection.bbox.height
                },
                'character_confidences': recognized_text.character_confidences
            }
            formatted_texts.append(formatted_text)
        
        return formatted_texts
    
    def _calculate_average_confidence(self, processing_result: ProcessingResult) -> float:
        """
        Calculate average confidence from processing result.
        
        Args:
            processing_result: Processing result object
            
        Returns:
            Average confidence score
        """
        confidences = []
        
        # Add detection confidences
        for detection in processing_result.detections:
            confidences.append(detection.confidence)
        
        # Add recognition confidences
        for recognized_text in processing_result.recognized_texts:
            confidences.append(recognized_text.recognition_confidence)
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _get_task_name(self, task: str) -> str:
        """
        Get human-readable task name.
        
        Args:
            task: Task identifier
            
        Returns:
            Human-readable task name
        """
        task_names = {
            'task1': 'Word Detection',
            'task2': 'Phrase Detection',
            'task3': 'Word Detection + Recognition',
            'task4': 'Phrase Detection + Recognition'
        }
        
        return task_names.get(task, task)
    
    def format_success_response(self, message: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Format generic success response.
        
        Args:
            message: Success message
            data: Optional additional data
            
        Returns:
            Formatted success response
        """
        response = {
            'message': message,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
        
        if data:
            response['data'] = data
        
        return response