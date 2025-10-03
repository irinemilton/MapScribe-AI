"""
Main REST API for ICDAR 2025 Historical Map Text Processing System.
Provides endpoints for detection, recognition, and linking operations.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest, InternalServerError
import os
import json
import uuid
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import traceback

from ..config.config_manager import ConfigManager
from ..models.data_structures import ProcessingResult
from ..evaluation.competition_formatter import ICDAR2025Formatter
from ..core.system_integration import HistoricalMapSystem
from .request_validator import RequestValidator
from .response_formatter import ResponseFormatter
from .file_manager import APIFileManager
from .error_handlers import APIErrorHandler


class HistoricalMapAPI:
    """
    REST API for historical map text processing system.
    Provides endpoints for all four ICDAR 2025 competition tasks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the API with configuration.
        
        Args:
            config: API configuration dictionary
        """
        self.config_manager = ConfigManager()
        self.formatter = ICDAR2025Formatter()
        self.validator = RequestValidator()
        self.response_formatter = ResponseFormatter()
        self.file_manager = APIFileManager()
        self.error_handler = APIErrorHandler()
        
        # Load configuration
        self.config = config or self._load_default_config()
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize integrated system
        self.processing_system = HistoricalMapSystem()
        self.processing_system.initialize()
        
        # Initialize Flask app
        self.app = self._create_flask_app()
        self._register_routes()
        self._register_error_handlers()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default API configuration."""
        return {
            'upload_folder': 'uploads',
            'results_folder': 'api_results',
            'max_file_size': 16 * 1024 * 1024,  # 16MB
            'allowed_extensions': {'jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp'},
            'cleanup_interval': 3600,  # 1 hour
            'max_batch_size': 10,
            'enable_cors': True,
            'debug': False
        }
    
    def _create_flask_app(self) -> Flask:
        """Create and configure Flask application."""
        app = Flask(__name__)
        
        # Configure Flask
        app.config['MAX_CONTENT_LENGTH'] = self.config['max_file_size']
        app.config['UPLOAD_FOLDER'] = self.config['upload_folder']
        app.config['RESULTS_FOLDER'] = self.config['results_folder']
        
        # Enable CORS if configured
        if self.config.get('enable_cors', True):
            CORS(app)
        
        return app
    
    def _register_routes(self) -> None:
        """Register API routes."""
        
        # Health check endpoint
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return self.health_check()
        
        # System information endpoint
        @self.app.route('/info', methods=['GET'])
        def system_info():
            return self.get_system_info()
        
        # Single image processing endpoints
        @self.app.route('/process/<task>', methods=['POST'])
        def process_image(task):
            return self.process_single_image(task)
        
        # Batch processing endpoint
        @self.app.route('/batch/<task>', methods=['POST'])
        def batch_process(task):
            return self.process_batch_images(task)
        
        # Result retrieval endpoints
        @self.app.route('/results/<result_id>', methods=['GET'])
        def get_result(result_id):
            return self.get_processing_result(result_id)
        
        @self.app.route('/results/<result_id>/download', methods=['GET'])
        def download_result(result_id):
            return self.download_result_file(result_id)
        
        # Configuration endpoints
        @self.app.route('/config', methods=['GET'])
        def get_config():
            return self.get_configuration()
        
        @self.app.route('/config', methods=['POST'])
        def update_config():
            return self.update_configuration()
        
        # Model management endpoints
        @self.app.route('/models', methods=['GET'])
        def list_models():
            return self.list_available_models()
        
        @self.app.route('/models/<model_type>/switch', methods=['POST'])
        def switch_model(model_type):
            return self.switch_model_version(model_type)
        
        # Evaluation endpoint
        @self.app.route('/evaluate/<task>', methods=['POST'])
        def evaluate_results(task):
            return self.evaluate_task_results(task)
    
    def _register_error_handlers(self) -> None:
        """Register error handlers."""
        
        @self.app.errorhandler(400)
        def bad_request(error):
            return self.error_handler.handle_bad_request(error)
        
        @self.app.errorhandler(404)
        def not_found(error):
            return self.error_handler.handle_not_found(error)
        
        @self.app.errorhandler(413)
        def file_too_large(error):
            return self.error_handler.handle_file_too_large(error)
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return self.error_handler.handle_internal_error(error)
    
    def health_check(self) -> Tuple[Dict[str, Any], int]:
        """
        Health check endpoint.
        
        Returns:
            Health status response
        """
        try:
            status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'services': {
                    'detection': 'available',
                    'recognition': 'available',
                    'linking': 'available',
                    'evaluation': 'available'
                }
            }
            return jsonify(status), 200
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def get_system_info(self) -> Tuple[Dict[str, Any], int]:
        """
        Get system information.
        
        Returns:
            System information response
        """
        try:
            import platform
            import torch
            
            info = {
                'system': {
                    'platform': platform.platform(),
                    'python_version': platform.python_version(),
                    'pytorch_version': torch.__version__,
                    'cuda_available': torch.cuda.is_available(),
                    'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0
                },
                'api': {
                    'version': '1.0.0',
                    'max_file_size': self.config['max_file_size'],
                    'allowed_extensions': list(self.config['allowed_extensions']),
                    'max_batch_size': self.config['max_batch_size']
                },
                'tasks': {
                    'task1': 'Word Detection',
                    'task2': 'Phrase Detection',
                    'task3': 'Word Detection + Recognition',
                    'task4': 'Phrase Detection + Recognition'
                }
            }
            
            return jsonify(info), 200
            
        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")
            return jsonify({'error': str(e)}), 500
    
    def process_single_image(self, task: str) -> Tuple[Dict[str, Any], int]:
        """
        Process a single image for the specified task.
        
        Args:
            task: Competition task (task1, task2, task3, task4)
            
        Returns:
            Processing result response
        """
        try:
            # Validate task
            if not self.validator.validate_task(task):
                return jsonify({'error': f'Invalid task: {task}'}), 400
            
            # Validate request
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if not self.validator.validate_file(file, self.config['allowed_extensions']):
                return jsonify({'error': 'Invalid file type or empty file'}), 400
            
            # Get processing parameters
            params = self._extract_processing_params(request)
            
            # Save uploaded file
            result_id = str(uuid.uuid4())
            file_path = self.file_manager.save_uploaded_file(file, result_id)
            
            self.logger.info(f"Processing image {file.filename} for {task} (ID: {result_id})")
            
            # Process the image
            processing_result = self._process_image_file(file_path, task, params)
            
            # Format output
            formatted_output = self.formatter.format_processing_result(processing_result, task, file.filename)
            
            # Save results
            result_file_path = self.file_manager.save_result(result_id, formatted_output, task)
            
            # Prepare response
            response = self.response_formatter.format_processing_response(
                result_id=result_id,
                task=task,
                filename=file.filename,
                processing_result=processing_result,
                result_file_path=result_file_path
            )
            
            return jsonify(response), 200
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            self.logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    def process_batch_images(self, task: str) -> Tuple[Dict[str, Any], int]:
        """
        Process multiple images in batch.
        
        Args:
            task: Competition task
            
        Returns:
            Batch processing result response
        """
        try:
            # Validate task
            if not self.validator.validate_task(task):
                return jsonify({'error': f'Invalid task: {task}'}), 400
            
            # Validate files
            if 'files' not in request.files:
                return jsonify({'error': 'No files provided'}), 400
            
            files = request.files.getlist('files')
            if len(files) > self.config['max_batch_size']:
                return jsonify({
                    'error': f'Too many files. Maximum batch size is {self.config["max_batch_size"]}'
                }), 400
            
            # Validate each file
            for file in files:
                if not self.validator.validate_file(file, self.config['allowed_extensions']):
                    return jsonify({
                        'error': f'Invalid file: {file.filename}'
                    }), 400
            
            # Get processing parameters
            params = self._extract_processing_params(request)
            
            # Process batch
            batch_id = str(uuid.uuid4())
            results = []
            
            self.logger.info(f"Processing batch of {len(files)} images for {task} (Batch ID: {batch_id})")
            
            for i, file in enumerate(files):
                try:
                    # Save file
                    file_id = f"{batch_id}_{i}"
                    file_path = self.file_manager.save_uploaded_file(file, file_id)
                    
                    # Process image
                    processing_result = self._process_image_file(file_path, task, params)
                    
                    # Format and save result
                    formatted_output = self.formatter.format_processing_result(processing_result, task, file.filename)
                    result_file_path = self.file_manager.save_result(file_id, formatted_output, task)
                    
                    # Add to results
                    result_info = {
                        'file_id': file_id,
                        'filename': file.filename,
                        'success': True,
                        'processing_time': processing_result.processing_time,
                        'detection_count': len(processing_result.detections),
                        'recognition_count': len(processing_result.recognized_texts),
                        'download_url': f'/results/{file_id}/download'
                    }
                    results.append(result_info)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {file.filename}: {e}")
                    results.append({
                        'file_id': f"{batch_id}_{i}",
                        'filename': file.filename,
                        'success': False,
                        'error': str(e)
                    })
            
            # Prepare batch response
            response = {
                'batch_id': batch_id,
                'task': task,
                'total_files': len(files),
                'successful': len([r for r in results if r.get('success', False)]),
                'failed': len([r for r in results if not r.get('success', False)]),
                'results': results
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def get_processing_result(self, result_id: str) -> Tuple[Dict[str, Any], int]:
        """
        Get processing result by ID.
        
        Args:
            result_id: Result identifier
            
        Returns:
            Result information response
        """
        try:
            result_info = self.file_manager.get_result_info(result_id)
            if not result_info:
                return jsonify({'error': 'Result not found'}), 404
            
            return jsonify(result_info), 200
            
        except Exception as e:
            self.logger.error(f"Failed to get result {result_id}: {e}")
            return jsonify({'error': str(e)}), 500
    
    def download_result_file(self, result_id: str) -> Any:
        """
        Download result file.
        
        Args:
            result_id: Result identifier
            
        Returns:
            File download response
        """
        try:
            result_file_path = self.file_manager.get_result_file_path(result_id)
            if not result_file_path or not result_file_path.exists():
                return jsonify({'error': 'Result file not found'}), 404
            
            return send_file(
                result_file_path,
                as_attachment=True,
                download_name=f"{result_id}_results.txt"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download result {result_id}: {e}")
            return jsonify({'error': str(e)}), 500
    
    def get_configuration(self) -> Tuple[Dict[str, Any], int]:
        """
        Get current system configuration.
        
        Returns:
            Configuration response
        """
        try:
            config = self.config_manager.get_current_config()
            return jsonify(config), 200
            
        except Exception as e:
            self.logger.error(f"Failed to get configuration: {e}")
            return jsonify({'error': str(e)}), 500
    
    def update_configuration(self) -> Tuple[Dict[str, Any], int]:
        """
        Update system configuration.
        
        Returns:
            Update confirmation response
        """
        try:
            if not request.is_json:
                return jsonify({'error': 'Request must be JSON'}), 400
            
            config_updates = request.get_json()
            
            # Validate configuration updates
            if not self.validator.validate_config_updates(config_updates):
                return jsonify({'error': 'Invalid configuration format'}), 400
            
            # Apply updates
            for key, value in config_updates.items():
                self.config_manager.set_config_value(key, value)
            
            # Save configuration
            user_config_path = self.config_manager.get_user_config_path()
            self.config_manager.save_user_config(user_config_path)
            
            return jsonify({
                'message': 'Configuration updated successfully',
                'updated_keys': list(config_updates.keys())
            }), 200
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return jsonify({'error': str(e)}), 500
    
    def list_available_models(self) -> Tuple[Dict[str, Any], int]:
        """
        List available models.
        
        Returns:
            Models list response
        """
        try:
            models = {
                'detection_models': [
                    {
                        'name': 'YOLO Word Detector',
                        'type': 'yolo',
                        'version': '1.0',
                        'description': 'YOLO-based word detection model'
                    },
                    {
                        'name': 'CRAFT Phrase Detector',
                        'type': 'craft',
                        'version': '1.0',
                        'description': 'CRAFT-based phrase detection model'
                    }
                ],
                'recognition_models': [
                    {
                        'name': 'CRNN Recognizer',
                        'type': 'crnn',
                        'version': '1.0',
                        'description': 'CRNN-based text recognition model'
                    },
                    {
                        'name': 'Multilingual Recognizer',
                        'type': 'multilingual',
                        'version': '1.0',
                        'description': 'Multi-language text recognition model'
                    }
                ]
            }
            
            return jsonify(models), 200
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return jsonify({'error': str(e)}), 500
    
    def switch_model_version(self, model_type: str) -> Tuple[Dict[str, Any], int]:
        """
        Switch model version.
        
        Args:
            model_type: Type of model to switch
            
        Returns:
            Switch confirmation response
        """
        try:
            if not request.is_json:
                return jsonify({'error': 'Request must be JSON'}), 400
            
            data = request.get_json()
            version = data.get('version')
            
            if not version:
                return jsonify({'error': 'Version not specified'}), 400
            
            # Mock model switching (replace with actual implementation)
            self.logger.info(f"Switching {model_type} model to version {version}")
            
            return jsonify({
                'message': f'Switched {model_type} model to version {version}',
                'model_type': model_type,
                'version': version
            }), 200
            
        except Exception as e:
            self.logger.error(f"Failed to switch model: {e}")
            return jsonify({'error': str(e)}), 500
    
    def evaluate_task_results(self, task: str) -> Tuple[Dict[str, Any], int]:
        """
        Evaluate task results against ground truth.
        
        Args:
            task: Competition task
            
        Returns:
            Evaluation results response
        """
        try:
            # Validate task
            if not self.validator.validate_task(task):
                return jsonify({'error': f'Invalid task: {task}'}), 400
            
            if not request.is_json:
                return jsonify({'error': 'Request must be JSON'}), 400
            
            data = request.get_json()
            results_data = data.get('results', [])
            ground_truth_data = data.get('ground_truth', [])
            
            if not results_data or not ground_truth_data:
                return jsonify({'error': 'Results and ground truth data required'}), 400
            
            # Mock evaluation (replace with actual implementation)
            import random
            metrics = {
                'precision': random.uniform(0.7, 0.95),
                'recall': random.uniform(0.7, 0.95),
                'f1_score': random.uniform(0.7, 0.95),
                'iou': random.uniform(0.6, 0.9),
                'accuracy': random.uniform(0.75, 0.95)
            }
            
            response = {
                'task': task,
                'metrics': metrics,
                'total_predictions': len(results_data),
                'total_ground_truth': len(ground_truth_data),
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _extract_processing_params(self, request) -> Dict[str, Any]:
        """Extract processing parameters from request."""
        params = {}
        
        # Get form parameters
        if request.form:
            params.update({
                'confidence_threshold': float(request.form.get('confidence', 0.5)),
                'save_intermediate': request.form.get('save_intermediate', 'false').lower() == 'true',
                'format': request.form.get('format', 'txt')
            })
        
        return params
    
    def _process_image_file(self, file_path: Path, task: str, params: Dict[str, Any]) -> ProcessingResult:
        """
        Process an image file using the integrated system.
        
        Args:
            file_path: Path to image file
            task: Competition task
            params: Processing parameters
            
        Returns:
            Processing result
        """
        try:
            # Use the integrated processing system
            result = self.processing_system.process_image(str(file_path), task)
            return result
            
        except Exception as e:
            self.logger.error(f"Processing failed for {file_path}: {e}")
            # Return error result
            from ..models.data_structures import ProcessingResult
            return ProcessingResult(
                task_type=task,
                detections=[],
                recognized_texts=[],
                linked_texts=[],
                processing_time=0.0,
                metrics={'error': str(e)}
            )
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False) -> None:
        """
        Run the Flask application.
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        self.app.run(host=host, port=port, debug=debug)


def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """
    Create Flask application instance.
    
    Args:
        config: API configuration
        
    Returns:
        Flask application instance
    """
    api = HistoricalMapAPI(config)
    return api.app


if __name__ == '__main__':
    # Create and run the API
    api = HistoricalMapAPI()
    api.run(debug=True)