"""
Demonstration of error handling and performance optimization features.

This example shows how to use the comprehensive error handling system,
graceful degradation, and performance monitoring in the historical map
text processing system.
"""

import logging
import time
import numpy as np
from pathlib import Path
from typing import List

# Import error handling components
from src.core.error_handling import (
    ProcessingError, DetectionError, RecognitionError, ModelLoadingError,
    ErrorCategory, ErrorSeverity, ErrorContext, error_manager,
    handle_errors, RetryMechanism
)

# Import graceful degradation components
from src.core.graceful_degradation import (
    degradation_manager, fallback_model_manager, CircuitBreaker,
    with_graceful_degradation
)

# Import monitoring components
from src.core.monitoring import (
    monitoring_system, monitor_performance
)

# Import performance optimization components
from src.core.performance_optimization import (
    BatchProcessor, BatchConfig, MemoryManager, PerformanceProfiler,
    optimize_memory, batch_process
)

# Import system components
from src.models.data_structures import ProcessedImage, Detection, BoundingBox


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockTextDetector:
    """Mock text detector for demonstration purposes."""
    
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.model_loaded = False
    
    @monitor_performance("mock_detector")
    @with_graceful_degradation("detection")
    def detect_words(self, image: ProcessedImage) -> List[Detection]:
        """Mock word detection with optional failure."""
        if self.should_fail:
            raise DetectionError("Mock detection failure for demonstration")
        
        if not self.model_loaded:
            raise DetectionError("Model not loaded")
        
        # Create mock detections
        detections = []
        for i in range(3):
            bbox = BoundingBox(
                x=i * 100,
                y=50,
                width=80,
                height=30,
                confidence=0.8 + i * 0.05
            )
            detection = Detection(
                bbox=bbox,
                detection_type='word',
                confidence=bbox.confidence,
                image_coordinates=(bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)
            )
            detections.append(detection)
        
        return detections
    
    @handle_errors(error_types=[ModelLoadingError], fallback_result=False)
    def load_model(self, model_path: str) -> bool:
        """Mock model loading with error handling."""
        if not Path(model_path).exists():
            raise ModelLoadingError(f"Model not found: {model_path}", model_path)
        
        # Simulate loading time
        time.sleep(0.1)
        self.model_loaded = True
        return True


class MockTextRecognizer:
    """Mock text recognizer for demonstration purposes."""
    
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
    
    @monitor_performance("mock_recognizer")
    @with_graceful_degradation("recognition")
    def recognize_text(self, text_region: np.ndarray) -> str:
        """Mock text recognition with optional failure."""
        if self.should_fail:
            raise RecognitionError("Mock recognition failure for demonstration")
        
        # Simulate processing time
        time.sleep(0.05)
        return f"recognized_text_{int(time.time() * 1000) % 1000}"


def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    logger.info("=== Error Handling Demonstration ===")
    
    # 1. Basic error handling with decorators
    @handle_errors(fallback_result="fallback_result")
    def function_that_might_fail(should_fail: bool = False):
        if should_fail:
            raise ValueError("Intentional failure for demonstration")
        return "success"
    
    # Test successful execution
    result = function_that_might_fail(False)
    logger.info(f"Successful execution result: {result}")
    
    # Test error handling
    result = function_that_might_fail(True)
    logger.info(f"Error handling result: {result}")
    
    # 2. Retry mechanism demonstration
    retry_mechanism = RetryMechanism(max_attempts=3, base_delay=0.1)
    
    @retry_mechanism
    def flaky_function():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise ConnectionError("Simulated network error")
        return "success after retries"
    
    try:
        result = flaky_function()
        logger.info(f"Retry mechanism result: {result}")
    except Exception as e:
        logger.error(f"Function failed after all retries: {e}")
    
    # 3. Circuit breaker demonstration
    circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
    
    @circuit_breaker
    def unreliable_service():
        raise ConnectionError("Service unavailable")
    
    # Trigger circuit breaker
    for i in range(4):
        try:
            unreliable_service()
        except Exception as e:
            logger.info(f"Attempt {i+1}: {type(e).__name__}: {e}")
    
    # 4. Error statistics
    stats = error_manager.get_statistics()
    logger.info(f"Error statistics: {stats}")


def demonstrate_graceful_degradation():
    """Demonstrate graceful degradation capabilities."""
    logger.info("\n=== Graceful Degradation Demonstration ===")
    
    # 1. Register fallback models
    fallback_model_manager.register_fallback_models(
        'detection',
        ['/path/to/primary_model.pth', '/path/to/backup_model.pth', '/path/to/simple_model.pth']
    )
    
    # 2. Simulate component health degradation
    degradation_manager.update_component_health('detection', 0.3)  # Low health
    degradation_manager.record_performance('detection', 8.0)  # Slow performance
    
    # Check if degradation should be triggered
    should_degrade = degradation_manager.should_degrade('detection')
    logger.info(f"Should degrade detection component: {should_degrade}")
    
    if should_degrade:
        fallback_method = degradation_manager.get_fallback_method('detection')
        logger.info(f"Using fallback method: {fallback_method}")
    
    # 3. Get system health status
    health_status = degradation_manager.get_system_health()
    logger.info(f"System health: {health_status}")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    logger.info("\n=== Performance Monitoring Demonstration ===")
    
    # 1. Start performance monitoring
    from src.core.performance_optimization import performance_profiler
    performance_profiler.start_monitoring(interval_seconds=0.5)
    
    # 2. Simulate some operations
    detector = MockTextDetector()
    recognizer = MockTextRecognizer()
    
    # Create mock image
    mock_image = ProcessedImage(
        image=np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8),
        original_path="mock_image.jpg",
        preprocessing_applied=["resize", "normalize"]
    )
    
    # Simulate detection operations
    try:
        detector.load_model("nonexistent_model.pth")  # This will fail
    except:
        pass
    
    # Create a real model file for successful loading
    test_model_path = Path("test_model.pth")
    test_model_path.touch()
    
    try:
        detector.load_model(str(test_model_path))
        detections = detector.detect_words(mock_image)
        logger.info(f"Detected {len(detections)} words")
        
        # Simulate recognition
        for detection in detections:
            text_region = np.random.randint(0, 255, (30, 80, 3), dtype=np.uint8)
            recognized_text = recognizer.recognize_text(text_region)
            logger.info(f"Recognized: {recognized_text}")
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
    
    finally:
        test_model_path.unlink(missing_ok=True)
    
    # 3. Get monitoring statistics
    time.sleep(1)  # Let monitoring collect some data
    performance_profiler.stop_monitoring()
    
    system_status = monitoring_system.get_system_status()
    logger.info(f"System status: {system_status}")


def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities."""
    logger.info("\n=== Batch Processing Demonstration ===")
    
    # 1. Create batch processor
    config = BatchConfig(
        batch_size=5,
        max_workers=2,
        memory_limit_mb=512,
        enable_progress_tracking=True
    )
    processor = BatchProcessor(config)
    
    # 2. Define processing function
    def process_item(item):
        # Simulate processing time
        time.sleep(0.01)
        if item == 7:  # Simulate occasional failure
            raise ValueError(f"Processing failed for item {item}")
        return item * 2
    
    # 3. Process batch of items
    items = list(range(20))
    results, metrics = processor.process_batch(items, process_item)
    
    logger.info(f"Batch processing results:")
    logger.info(f"  Total items: {metrics.total_items}")
    logger.info(f"  Processed: {metrics.processed_items}")
    logger.info(f"  Failed: {metrics.failed_items}")
    logger.info(f"  Duration: {metrics.duration:.2f}s")
    logger.info(f"  Throughput: {metrics.throughput:.2f} items/sec")
    logger.info(f"  Success rate: {metrics.success_rate:.1f}%")
    
    # 4. Demonstrate batch processing decorator
    @batch_process(batch_size=3, max_workers=2)
    def square_numbers(number):
        return number ** 2
    
    numbers = list(range(10))
    squared_results = square_numbers(numbers)
    logger.info(f"Squared numbers: {squared_results}")


def demonstrate_memory_management():
    """Demonstrate memory management capabilities."""
    logger.info("\n=== Memory Management Demonstration ===")
    
    # 1. Create memory manager
    memory_manager = MemoryManager(memory_limit_mb=100)  # Low limit for demo
    
    # 2. Check current memory usage
    current_usage = memory_manager.get_memory_usage()
    logger.info(f"Current memory usage: {current_usage:.2f} MB")
    
    # 3. Get detailed memory statistics
    memory_stats = memory_manager.get_memory_stats()
    logger.info(f"Memory statistics: {memory_stats}")
    
    # 4. Demonstrate memory optimization decorator
    @optimize_memory()
    def memory_intensive_function():
        # Simulate memory-intensive operation
        large_array = np.random.random((1000, 1000))
        return large_array.sum()
    
    result = memory_intensive_function()
    logger.info(f"Memory-intensive function result: {result}")
    
    # 5. Force memory optimization
    optimization_stats = memory_manager.optimize_memory_usage()
    logger.info(f"Memory optimization stats: {optimization_stats}")


def demonstrate_integrated_pipeline():
    """Demonstrate integrated pipeline with all features."""
    logger.info("\n=== Integrated Pipeline Demonstration ===")
    
    # Create components with error handling
    detector = MockTextDetector(should_fail=False)
    recognizer = MockTextRecognizer(should_fail=False)
    
    # Create test model file
    test_model_path = Path("integrated_test_model.pth")
    test_model_path.touch()
    
    try:
        # Load model with error handling
        detector.load_model(str(test_model_path))
        
        # Create batch of mock images
        images = []
        for i in range(10):
            image = ProcessedImage(
                image=np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8),
                original_path=f"image_{i}.jpg",
                preprocessing_applied=["resize", "normalize"]
            )
            images.append(image)
        
        # Process images in batch with monitoring and error handling
        @monitor_performance("integrated_pipeline")
        @with_graceful_degradation("pipeline")
        @optimize_memory()
        def process_image_pipeline(image: ProcessedImage):
            # Detect text
            detections = detector.detect_words(image)
            
            # Recognize text
            results = []
            for detection in detections:
                text_region = np.random.randint(0, 255, (30, 80, 3), dtype=np.uint8)
                recognized_text = recognizer.recognize_text(text_region)
                results.append({
                    'detection': detection,
                    'text': recognized_text
                })
            
            return results
        
        # Process batch
        config = BatchConfig(batch_size=3, max_workers=2)
        processor = BatchProcessor(config)
        
        all_results, batch_metrics = processor.process_batch(images, process_image_pipeline)
        
        logger.info(f"Integrated pipeline completed:")
        logger.info(f"  Processed {batch_metrics.processed_items} images")
        logger.info(f"  Success rate: {batch_metrics.success_rate:.1f}%")
        logger.info(f"  Throughput: {batch_metrics.throughput:.2f} images/sec")
        
        # Get final system status
        final_status = monitoring_system.get_system_status()
        logger.info(f"Final system status: {final_status['health']['status']}")
        
    finally:
        test_model_path.unlink(missing_ok=True)


def main():
    """Run all demonstrations."""
    logger.info("Starting Error Handling and Performance Optimization Demo")
    
    try:
        demonstrate_error_handling()
        demonstrate_graceful_degradation()
        demonstrate_performance_monitoring()
        demonstrate_batch_processing()
        demonstrate_memory_management()
        demonstrate_integrated_pipeline()
        
        logger.info("\n=== Demo Completed Successfully ===")
        
        # Export final metrics
        monitoring_system.export_metrics("demo_metrics.json")
        logger.info("Metrics exported to demo_metrics.json")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()