"""
System integration module for the Historical Map Text Processing System.
Wires together all pipeline components with proper error handling and configuration management.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from dataclasses import asdict

from ..config.config_manager import ConfigManager
from ..config.base_config import SystemConfig
from ..models.data_structures import ProcessingResult, ProcessedImage, Detection, RecognizedText, LinkedText
from ..processing.image_processor import BasicImageProcessor
from ..processing.historical_map_processor import HistoricalMapProcessor
from ..detection.yolo_detector import YOLODetector
from ..detection.phrase_detector import PhraseDetector
from ..recognition.crnn_recognizer import CRNNRecognizer
from ..recognition.multilingual_recognizer import MultilingualRecognizer
from ..linking.text_linker import TextLinker
from ..evaluation.competition_formatter import ICDAR2025Formatter
from ..evaluation.competition_evaluator import CompetitionEvaluator
from ..models.model_manager import ModelManager
from ..core.error_handling import ErrorHandler
from ..core.graceful_degradation import GracefulDegradation
from ..core.monitoring import SystemMonitor
from ..core.performance_optimization import PerformanceOptimizer
from .system_interface import HistoricalMapProcessorInterface


class HistoricalMapSystem(HistoricalMapProcessorInterface):
    """
    Main system integration class that coordinates all components
    of the historical map text processing pipeline.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the integrated system.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.logger = self._setup_logging()
        self.config_manager = ConfigManager()
        self.error_handler = ErrorHandler()
        self.graceful_degradation = GracefulDegradation()
        self.monitor = SystemMonitor()
        self.performance_optimizer = PerformanceOptimizer()
        
        # Load configuration
        self.config = self._load_configuration(config_path)
        
        # Initialize components
        self._initialize_components()
        
        # System state
        self._initialized = False
        self._processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_time': 0.0
        }
        
        self.logger.info("Historical Map System initialized")
    
    def initialize(self, config: Optional[SystemConfig] = None) -> None:
        """
        Initialize the system with configuration.
        
        Args:
            config: System configuration (optional, uses loaded config if not provided)
        """
        try:
            if config:
                self.config = config
            
            # Initialize all components with configuration
            self._initialize_processing_pipeline()
            self._initialize_models()
            self._initialize_evaluation_components()
            
            # Start monitoring
            self.monitor.start_monitoring()
            
            # Apply performance optimizations
            self.performance_optimizer.optimize_system(self.config)
            
            self._initialized = True
            self.logger.info("System initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            self.error_handler.handle_system_error(e)
            raise
    
    def process_image(self, image_path: str, task_type: str) -> ProcessingResult:
        """
        Process a single historical map image.
        
        Args:
            image_path: Path to input image
            task_type: Type of processing task (task1, task2, task3, task4)
            
        Returns:
            ProcessingResult: Complete processing results
        """
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        start_time = time.time()
        image_path = Path(image_path)
        
        try:
            self.logger.info(f"Processing image: {image_path} for {task_type}")
            
            # Validate inputs
            self._validate_inputs(image_path, task_type)
            
            # Start monitoring for this processing session
            session_id = self.monitor.start_processing_session(str(image_path), task_type)
            
            # Load and preprocess image
            processed_image = self._preprocess_image(image_path)
            
            # Execute processing pipeline based on task type
            result = self._execute_processing_pipeline(processed_image, task_type, str(image_path))
            
            # Post-process results
            result = self._post_process_results(result, task_type)
            
            # Update processing time
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            # Update statistics
            self._update_processing_stats(True, processing_time)
            
            # End monitoring session
            self.monitor.end_processing_session(session_id, True, processing_time)
            
            self.logger.info(f"Successfully processed {image_path} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Failed to process {image_path}: {e}")
            
            # Update statistics
            self._update_processing_stats(False, processing_time)
            
            # Try graceful degradation
            degraded_result = self.graceful_degradation.handle_processing_failure(
                str(image_path), task_type, e
            )
            
            if degraded_result:
                self.logger.warning(f"Returning degraded result for {image_path}")
                return degraded_result
            
            # Handle error and re-raise
            self.error_handler.handle_processing_error(e, str(image_path), task_type)
            raise
    
    def process_batch(self, image_paths: List[str], task_type: str) -> List[ProcessingResult]:
        """
        Process multiple historical map images.
        
        Args:
            image_paths: List of paths to input images
            task_type: Type of processing task
            
        Returns:
            List of processing results
        """
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        self.logger.info(f"Starting batch processing: {len(image_paths)} images for {task_type}")
        
        results = []
        successful = 0
        failed = 0
        
        # Apply batch optimizations
        self.performance_optimizer.optimize_for_batch_processing(len(image_paths))
        
        for i, image_path in enumerate(image_paths):
            try:
                self.logger.info(f"Processing batch item {i+1}/{len(image_paths)}: {image_path}")
                
                result = self.process_image(image_path, task_type)
                results.append(result)
                successful += 1
                
            except Exception as e:
                self.logger.error(f"Failed to process batch item {image_path}: {e}")
                
                # Create error result
                error_result = ProcessingResult(
                    task_type=task_type,
                    detections=[],
                    recognized_texts=[],
                    linked_texts=[],
                    processing_time=0.0,
                    metrics={'error': str(e)}
                )
                results.append(error_result)
                failed += 1
        
        self.logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
        return results
    
    def evaluate_results(self, results: List[ProcessingResult], ground_truth_path: str) -> Dict[str, float]:
        """
        Evaluate processing results against ground truth.
        
        Args:
            results: List of processing results
            ground_truth_path: Path to ground truth annotations
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            self.logger.info(f"Evaluating {len(results)} results against ground truth")
            
            # Use competition evaluator
            metrics = self.evaluator.evaluate_batch_results(results, ground_truth_path)
            
            self.logger.info(f"Evaluation completed: {len(metrics)} metrics calculated")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            self.error_handler.handle_evaluation_error(e)
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and statistics.
        
        Returns:
            System status dictionary
        """
        return {
            'initialized': self._initialized,
            'configuration': asdict(self.config) if self.config else None,
            'processing_stats': self._processing_stats.copy(),
            'monitoring_data': self.monitor.get_current_metrics(),
            'model_info': self.model_manager.get_all_model_metadata(),
            'performance_stats': self.performance_optimizer.get_performance_stats()
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the system."""
        try:
            self.logger.info("Shutting down system...")
            
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            # Cleanup models
            if hasattr(self, 'model_manager'):
                self.model_manager.cleanup()
            
            # Reset state
            self._initialized = False
            
            self.logger.info("System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def _load_configuration(self, config_path: Optional[str]) -> SystemConfig:
        """Load system configuration."""
        try:
            if config_path:
                return self.config_manager.load_config(config_path)
            else:
                return self.config_manager.get_default_config()
        except Exception as e:
            self.logger.warning(f"Failed to load configuration: {e}. Using defaults.")
            return self.config_manager.get_default_config()
    
    def _initialize_components(self) -> None:
        """Initialize all system components."""
        try:
            # Initialize core components
            self.image_processor = BasicImageProcessor()
            self.historical_processor = HistoricalMapProcessor()
            self.formatter = ICDAR2025Formatter()
            self.evaluator = CompetitionEvaluator()
            self.model_manager = ModelManager()
            
            self.logger.info("Core components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _initialize_processing_pipeline(self) -> None:
        """Initialize the processing pipeline components."""
        try:
            # Initialize detectors
            self.word_detector = YOLODetector()
            self.phrase_detector = PhraseDetector()
            
            # Initialize recognizers
            self.crnn_recognizer = CRNNRecognizer()
            self.multilingual_recognizer = MultilingualRecognizer()
            
            # Initialize linker
            self.text_linker = TextLinker()
            
            self.logger.info("Processing pipeline initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize processing pipeline: {e}")
            raise
    
    def _initialize_models(self) -> None:
        """Initialize and load models."""
        try:
            # Load detection models
            if self.config.detection.model_path:
                self.model_manager.load_detection_model(
                    self.config.detection.model_path,
                    asdict(self.config.detection)
                )
            
            # Load recognition models
            if self.config.recognition.model_path:
                self.model_manager.load_recognition_model(
                    self.config.recognition.model_path,
                    asdict(self.config.recognition)
                )
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load some models: {e}")
            # Continue with default models
    
    def _initialize_evaluation_components(self) -> None:
        """Initialize evaluation components."""
        try:
            # Configure evaluator with system config
            self.evaluator.configure(asdict(self.config.evaluation))
            
            self.logger.info("Evaluation components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize evaluation components: {e}")
            raise
    
    def _validate_inputs(self, image_path: Path, task_type: str) -> None:
        """Validate input parameters."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not image_path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")
        
        valid_tasks = ['task1', 'task2', 'task3', 'task4']
        if task_type not in valid_tasks:
            raise ValueError(f"Invalid task type: {task_type}. Must be one of {valid_tasks}")
        
        # Check file format
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        if image_path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
    
    def _preprocess_image(self, image_path: Path) -> ProcessedImage:
        """Preprocess the input image."""
        try:
            # Load image
            image = self.image_processor.load_image(str(image_path))
            
            # Apply historical map specific preprocessing
            processed_image = self.historical_processor.preprocess_image(
                image, asdict(self.config.preprocessing)
            )
            
            return processed_image
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def _execute_processing_pipeline(self, processed_image: ProcessedImage, 
                                   task_type: str, image_path: str) -> ProcessingResult:
        """Execute the main processing pipeline."""
        try:
            # Initialize result
            result = ProcessingResult(
                task_type=task_type,
                detections=[],
                recognized_texts=[],
                linked_texts=[],
                processing_time=0.0,
                metrics={}
            )
            
            # Step 1: Text Detection
            if task_type in ['task1', 'task3']:
                # Word-level detection
                detections = self.word_detector.detect_words(processed_image)
            else:
                # Phrase-level detection
                detections = self.phrase_detector.detect_phrases(processed_image)
            
            result.detections = detections
            
            # Step 2: Text Recognition (for tasks 3 and 4)
            if task_type in ['task3', 'task4']:
                recognized_texts = []
                
                for detection in detections:
                    try:
                        # Extract text region
                        text_region = self._extract_text_region(processed_image, detection)
                        
                        # Choose recognizer based on language detection
                        if self._is_chinese_text(text_region):
                            recognized_text = self.multilingual_recognizer.recognize_text(
                                text_region, detection
                            )
                        else:
                            recognized_text = self.crnn_recognizer.recognize_text(
                                text_region, detection
                            )
                        
                        recognized_texts.append(recognized_text)
                        
                    except Exception as e:
                        self.logger.warning(f"Recognition failed for detection: {e}")
                        continue
                
                result.recognized_texts = recognized_texts
            
            # Step 3: Text Linking (optional enhancement)
            if result.recognized_texts:
                try:
                    linked_texts = self.text_linker.link_to_coordinates(
                        result.recognized_texts, processed_image.image
                    )
                    result.linked_texts = linked_texts
                except Exception as e:
                    self.logger.warning(f"Text linking failed: {e}")
            
            # Calculate metrics
            result.metrics = self._calculate_processing_metrics(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Processing pipeline failed: {e}")
            raise
    
    def _extract_text_region(self, processed_image: ProcessedImage, detection: Detection) -> np.ndarray:
        """Extract text region from image based on detection."""
        bbox = detection.bbox
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        
        # Ensure coordinates are within image bounds
        img_h, img_w = processed_image.image.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        return processed_image.image[y:y+h, x:x+w]
    
    def _is_chinese_text(self, text_region: np.ndarray) -> bool:
        """Determine if text region likely contains Chinese characters."""
        # Simple heuristic - in practice, this would use a more sophisticated method
        # For now, return False to use Latin recognizer
        return False
    
    def _post_process_results(self, result: ProcessingResult, task_type: str) -> ProcessingResult:
        """Post-process results for quality improvement."""
        try:
            # Apply confidence filtering
            min_confidence = self.config.evaluation.confidence_threshold
            
            # Filter detections
            result.detections = [
                d for d in result.detections 
                if d.confidence >= min_confidence
            ]
            
            # Filter recognized texts
            result.recognized_texts = [
                r for r in result.recognized_texts 
                if r.recognition_confidence >= min_confidence
            ]
            
            # Update linked texts accordingly
            if result.linked_texts:
                valid_recognition_ids = {id(r) for r in result.recognized_texts}
                result.linked_texts = [
                    l for l in result.linked_texts 
                    if id(l.recognized_text) in valid_recognition_ids
                ]
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Post-processing failed: {e}")
            return result
    
    def _calculate_processing_metrics(self, result: ProcessingResult) -> Dict[str, float]:
        """Calculate processing metrics."""
        metrics = {}
        
        if result.detections:
            confidences = [d.confidence for d in result.detections]
            metrics['detection_confidence_avg'] = sum(confidences) / len(confidences)
            metrics['detection_confidence_min'] = min(confidences)
            metrics['detection_confidence_max'] = max(confidences)
            metrics['detection_count'] = len(result.detections)
        
        if result.recognized_texts:
            rec_confidences = [r.recognition_confidence for r in result.recognized_texts]
            metrics['recognition_confidence_avg'] = sum(rec_confidences) / len(rec_confidences)
            metrics['recognition_confidence_min'] = min(rec_confidences)
            metrics['recognition_confidence_max'] = max(rec_confidences)
            metrics['recognition_count'] = len(result.recognized_texts)
        
        if result.linked_texts:
            metrics['linking_count'] = len(result.linked_texts)
        
        return metrics
    
    def _update_processing_stats(self, success: bool, processing_time: float) -> None:
        """Update processing statistics."""
        self._processing_stats['total_processed'] += 1
        self._processing_stats['total_time'] += processing_time
        
        if success:
            self._processing_stats['successful'] += 1
        else:
            self._processing_stats['failed'] += 1
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)


# Factory function for easy system creation
def create_system(config_path: Optional[str] = None) -> HistoricalMapSystem:
    """
    Create and initialize a Historical Map System instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Initialized HistoricalMapSystem instance
    """
    system = HistoricalMapSystem(config_path)
    system.initialize()
    return system