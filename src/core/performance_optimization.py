"""
Performance optimization utilities for historical map text processing.

This module provides batch processing, memory management, and performance
monitoring tools to improve system throughput and efficiency.
"""

import gc
import logging
import multiprocessing
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from queue import Queue, Empty
import numpy as np

from .monitoring import monitoring_system, monitor_performance
from .error_handling import ProcessingError, ErrorCategory, handle_errors


@dataclass
class BatchConfig:
    """Configuration for batch processing operations."""
    batch_size: int = 32
    max_workers: int = 4
    use_multiprocessing: bool = False
    memory_limit_mb: int = 1024
    timeout_seconds: float = 300.0
    enable_progress_tracking: bool = True


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    @property
    def duration(self) -> float:
        """Get operation duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def throughput(self) -> float:
        """Get processing throughput (items per second)."""
        duration = self.duration
        return self.processed_items / duration if duration > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        total = self.processed_items + self.failed_items
        return (self.processed_items / total * 100) if total > 0 else 0.0


class MemoryManager:
    """Manages memory usage and prevents out-of-memory errors."""
    
    def __init__(self, memory_limit_mb: int = 1024):
        """
        Initialize memory manager.
        
        Args:
            memory_limit_mb: Memory limit in megabytes
        """
        self.memory_limit_mb = memory_limit_mb
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
            return 0.0
    
    def check_memory_limit(self) -> bool:
        """
        Check if memory usage is within limits.
        
        Returns:
            True if within limits, False otherwise
        """
        current_usage = self.get_memory_usage()
        return current_usage <= self.memory_limit_mb
    
    def force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""
        with self._lock:
            gc.collect()
            self.logger.debug("Forced garbage collection")
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage and return statistics.
        
        Returns:
            Dictionary with memory optimization results
        """
        before_usage = self.get_memory_usage()
        
        # Force garbage collection
        self.force_garbage_collection()
        
        after_usage = self.get_memory_usage()
        freed_mb = before_usage - after_usage
        
        stats = {
            'before_usage_mb': before_usage,
            'after_usage_mb': after_usage,
            'freed_mb': freed_mb,
            'memory_limit_mb': self.memory_limit_mb,
            'within_limit': after_usage <= self.memory_limit_mb
        }
        
        self.logger.info(f"Memory optimization: freed {freed_mb:.2f} MB")
        return stats
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get detailed memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / (1024 * 1024),
                'limit_mb': self.memory_limit_mb
            }
        except ImportError:
            return {
                'rss_mb': 0.0,
                'vms_mb': 0.0,
                'percent': 0.0,
                'available_mb': 0.0,
                'limit_mb': self.memory_limit_mb
            }


class BatchProcessor:
    """Processes items in batches for improved performance."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Initialize batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self.memory_manager = MemoryManager(self.config.memory_limit_mb)
        self.logger = logging.getLogger(__name__)
        self._metrics = PerformanceMetrics()
    
    @monitor_performance("batch_processor")
    def process_batch(
        self,
        items: List[Any],
        process_function: Callable[[Any], Any],
        **kwargs
    ) -> Tuple[List[Any], PerformanceMetrics]:
        """
        Process a batch of items.
        
        Args:
            items: List of items to process
            process_function: Function to apply to each item
            **kwargs: Additional arguments for process_function
            
        Returns:
            Tuple of (results, metrics)
        """
        self._metrics = PerformanceMetrics(total_items=len(items))
        results = []
        
        try:
            # Process items in chunks
            for chunk_start in range(0, len(items), self.config.batch_size):
                chunk_end = min(chunk_start + self.config.batch_size, len(items))
                chunk = items[chunk_start:chunk_end]
                
                # Check memory before processing chunk
                if not self.memory_manager.check_memory_limit():
                    self.memory_manager.optimize_memory_usage()
                
                # Process chunk
                chunk_results = self._process_chunk(chunk, process_function, **kwargs)
                results.extend(chunk_results)
                
                # Update metrics
                self._metrics.processed_items += len([r for r in chunk_results if r is not None])
                self._metrics.failed_items += len([r for r in chunk_results if r is None])
                
                # Log progress
                if self.config.enable_progress_tracking:
                    progress = (chunk_end / len(items)) * 100
                    self.logger.info(f"Batch progress: {progress:.1f}% ({chunk_end}/{len(items)})")
        
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise ProcessingError(
                f"Batch processing error: {str(e)}",
                ErrorCategory.SYSTEM
            )
        
        finally:
            self._metrics.end_time = time.time()
            self._metrics.memory_usage_mb = self.memory_manager.get_memory_usage()
        
        return results, self._metrics
    
    def _process_chunk(
        self,
        chunk: List[Any],
        process_function: Callable[[Any], Any],
        **kwargs
    ) -> List[Any]:
        """
        Process a single chunk of items.
        
        Args:
            chunk: Chunk of items to process
            process_function: Function to apply to each item
            **kwargs: Additional arguments for process_function
            
        Returns:
            List of processing results
        """
        if self.config.use_multiprocessing:
            return self._process_chunk_multiprocessing(chunk, process_function, **kwargs)
        else:
            return self._process_chunk_threading(chunk, process_function, **kwargs)
    
    def _process_chunk_threading(
        self,
        chunk: List[Any],
        process_function: Callable[[Any], Any],
        **kwargs
    ) -> List[Any]:
        """Process chunk using threading."""
        results = [None] * len(chunk)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._safe_process_item, item, process_function, **kwargs): i
                for i, item in enumerate(chunk)
            }
            
            # Collect results
            for future in as_completed(future_to_index, timeout=self.config.timeout_seconds):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    self.logger.error(f"Item processing failed at index {index}: {e}")
                    results[index] = None
        
        return results
    
    def _process_chunk_multiprocessing(
        self,
        chunk: List[Any],
        process_function: Callable[[Any], Any],
        **kwargs
    ) -> List[Any]:
        """Process chunk using multiprocessing."""
        results = [None] * len(chunk)
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._safe_process_item, item, process_function, **kwargs): i
                for i, item in enumerate(chunk)
            }
            
            # Collect results
            for future in as_completed(future_to_index, timeout=self.config.timeout_seconds):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    self.logger.error(f"Item processing failed at index {index}: {e}")
                    results[index] = None
        
        return results
    
    @staticmethod
    def _safe_process_item(item: Any, process_function: Callable, **kwargs) -> Any:
        """
        Safely process a single item with error handling.
        
        Args:
            item: Item to process
            process_function: Function to apply
            **kwargs: Additional arguments
            
        Returns:
            Processing result or None if failed
        """
        try:
            return process_function(item, **kwargs)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to process item: {e}")
            return None
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current processing metrics."""
        return self._metrics


class ImageBatchProcessor(BatchProcessor):
    """Specialized batch processor for image processing operations."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize image batch processor."""
        super().__init__(config)
        self.image_cache: Dict[str, np.ndarray] = {}
        self.cache_size_limit = 100  # Maximum number of cached images
    
    @monitor_performance("image_batch_processor")
    def process_image_batch(
        self,
        image_paths: List[str],
        process_function: Callable[[np.ndarray], Any],
        preload_images: bool = True,
        **kwargs
    ) -> Tuple[List[Any], PerformanceMetrics]:
        """
        Process a batch of images.
        
        Args:
            image_paths: List of image file paths
            process_function: Function to apply to each image
            preload_images: Whether to preload images into memory
            **kwargs: Additional arguments for process_function
            
        Returns:
            Tuple of (results, metrics)
        """
        if preload_images:
            self._preload_images(image_paths)
        
        # Create processing items with image data
        items = []
        for path in image_paths:
            if path in self.image_cache:
                items.append((path, self.image_cache[path]))
            else:
                items.append((path, None))
        
        # Process with image-specific function
        def image_process_wrapper(item):
            path, image = item
            if image is None:
                image = self._load_image(path)
            return process_function(image, **kwargs)
        
        return self.process_batch(items, image_process_wrapper)
    
    def _preload_images(self, image_paths: List[str]) -> None:
        """
        Preload images into memory cache.
        
        Args:
            image_paths: List of image paths to preload
        """
        self.logger.info(f"Preloading {len(image_paths)} images")
        
        for path in image_paths:
            if len(self.image_cache) >= self.cache_size_limit:
                # Remove oldest cached image
                oldest_key = next(iter(self.image_cache))
                del self.image_cache[oldest_key]
            
            if path not in self.image_cache:
                try:
                    image = self._load_image(path)
                    self.image_cache[path] = image
                except Exception as e:
                    self.logger.error(f"Failed to preload image {path}: {e}")
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image as numpy array
        """
        try:
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            return image
        except ImportError:
            # Fallback to PIL if OpenCV not available
            from PIL import Image
            pil_image = Image.open(image_path)
            return np.array(pil_image)
    
    def clear_cache(self) -> None:
        """Clear image cache to free memory."""
        self.image_cache.clear()
        self.memory_manager.force_garbage_collection()
        self.logger.info("Image cache cleared")


class PerformanceProfiler:
    """Profiles system performance and identifies optimization opportunities."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.operation_times: Dict[str, List[float]] = {}
        self.memory_snapshots: List[Tuple[float, float]] = []  # (timestamp, memory_mb)
        self.cpu_snapshots: List[Tuple[float, float]] = []     # (timestamp, cpu_percent)
        self.logger = logging.getLogger(__name__)
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, interval_seconds: float = 1.0) -> None:
        """
        Start continuous performance monitoring.
        
        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous performance monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            timestamp = time.time()
            
            # Record memory usage
            try:
                import psutil
                memory_mb = psutil.virtual_memory().used / (1024 * 1024)
                cpu_percent = psutil.cpu_percent()
                
                self.memory_snapshots.append((timestamp, memory_mb))
                self.cpu_snapshots.append((timestamp, cpu_percent))
                
                # Keep only last 1000 snapshots
                if len(self.memory_snapshots) > 1000:
                    self.memory_snapshots = self.memory_snapshots[-1000:]
                if len(self.cpu_snapshots) > 1000:
                    self.cpu_snapshots = self.cpu_snapshots[-1000:]
                
            except ImportError:
                pass  # psutil not available
            
            time.sleep(interval)
    
    def profile_operation(self, operation_name: str):
        """
        Decorator to profile an operation.
        
        Args:
            operation_name: Name of the operation to profile
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record successful execution
                    duration = time.time() - start_time
                    self._record_operation_time(operation_name, duration)
                    
                    end_memory = self._get_memory_usage()
                    memory_delta = end_memory - start_memory
                    
                    self.logger.debug(
                        f"Operation {operation_name}: {duration:.3f}s, "
                        f"memory delta: {memory_delta:.2f}MB"
                    )
                    
                    return result
                    
                except Exception as e:
                    # Record failed execution
                    duration = time.time() - start_time
                    self._record_operation_time(f"{operation_name}_failed", duration)
                    raise
            
            return wrapper
        return decorator
    
    def _record_operation_time(self, operation: str, duration: float) -> None:
        """Record operation execution time."""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        
        self.operation_times[operation].append(duration)
        
        # Keep only last 1000 measurements per operation
        if len(self.operation_times[operation]) > 1000:
            self.operation_times[operation] = self.operation_times[operation][-1000:]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary with performance analysis
        """
        report = {
            'timestamp': time.time(),
            'operations': {},
            'memory_analysis': self._analyze_memory_usage(),
            'cpu_analysis': self._analyze_cpu_usage(),
            'bottlenecks': self._identify_bottlenecks(),
            'recommendations': self._generate_recommendations()
        }
        
        # Analyze operation performance
        for operation, times in self.operation_times.items():
            if times:
                report['operations'][operation] = {
                    'count': len(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_time': sum(times),
                    'p95_time': self._percentile(times, 95),
                    'p99_time': self._percentile(times, 99)
                }
        
        return report
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if not self.memory_snapshots:
            return {'status': 'no_data'}
        
        memory_values = [snapshot[1] for snapshot in self.memory_snapshots]
        
        return {
            'current_mb': memory_values[-1] if memory_values else 0,
            'avg_mb': sum(memory_values) / len(memory_values),
            'min_mb': min(memory_values),
            'max_mb': max(memory_values),
            'trend': self._calculate_trend(memory_values),
            'stability': self._calculate_stability(memory_values)
        }
    
    def _analyze_cpu_usage(self) -> Dict[str, Any]:
        """Analyze CPU usage patterns."""
        if not self.cpu_snapshots:
            return {'status': 'no_data'}
        
        cpu_values = [snapshot[1] for snapshot in self.cpu_snapshots]
        
        return {
            'current_percent': cpu_values[-1] if cpu_values else 0,
            'avg_percent': sum(cpu_values) / len(cpu_values),
            'min_percent': min(cpu_values),
            'max_percent': max(cpu_values),
            'trend': self._calculate_trend(cpu_values),
            'stability': self._calculate_stability(cpu_values)
        }
    
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        for operation, times in self.operation_times.items():
            if len(times) < 5:  # Need sufficient data
                continue
            
            avg_time = sum(times) / len(times)
            p95_time = self._percentile(times, 95)
            
            # Check for high variance (potential bottleneck)
            if p95_time > avg_time * 2:
                bottlenecks.append({
                    'operation': operation,
                    'avg_time': avg_time,
                    'p95_time': p95_time,
                    'variance_ratio': p95_time / avg_time,
                    'severity': 'high' if p95_time > avg_time * 5 else 'medium'
                })
        
        # Sort by severity
        bottlenecks.sort(key=lambda x: x['variance_ratio'], reverse=True)
        return bottlenecks
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Memory recommendations
        memory_analysis = self._analyze_memory_usage()
        if memory_analysis.get('trend', 0) > 0.1:
            recommendations.append("Memory usage is trending upward - consider memory optimization")
        
        # CPU recommendations
        cpu_analysis = self._analyze_cpu_usage()
        if cpu_analysis.get('avg_percent', 0) > 80:
            recommendations.append("High CPU usage detected - consider parallel processing")
        
        # Operation recommendations
        bottlenecks = self._identify_bottlenecks()
        if bottlenecks:
            top_bottleneck = bottlenecks[0]
            recommendations.append(
                f"Operation '{top_bottleneck['operation']}' shows high variance - "
                f"investigate for optimization opportunities"
            )
        
        return recommendations
    
    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    @staticmethod
    def _calculate_trend(values: List[float]) -> float:
        """Calculate trend (slope) of values over time."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        # Linear regression slope
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope
    
    @staticmethod
    def _calculate_stability(values: List[float]) -> float:
        """Calculate stability (inverse of coefficient of variation)."""
        if not values or len(values) < 2:
            return 1.0
        
        mean_val = sum(values) / len(values)
        if mean_val == 0:
            return 1.0
        
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        cv = std_dev / mean_val
        
        # Return stability as inverse of coefficient of variation
        return 1.0 / (1.0 + cv)


# Global instances
memory_manager = MemoryManager()
performance_profiler = PerformanceProfiler()


def optimize_memory():
    """Decorator to optimize memory usage for a function."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Check memory before execution
            if not memory_manager.check_memory_limit():
                memory_manager.optimize_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Force garbage collection after execution
                memory_manager.force_garbage_collection()
        
        return wrapper
    return decorator


def batch_process(batch_size: int = 32, max_workers: int = 4):
    """
    Decorator to add batch processing to a function.
    
    Args:
        batch_size: Size of each batch
        max_workers: Maximum number of worker threads/processes
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(items: List[Any], *args, **kwargs):
            config = BatchConfig(
                batch_size=batch_size,
                max_workers=max_workers
            )
            processor = BatchProcessor(config)
            
            def process_item(item):
                return func(item, *args, **kwargs)
            
            results, metrics = processor.process_batch(items, process_item)
            
            # Log performance metrics
            logging.getLogger(__name__).info(
                f"Batch processing completed: {metrics.processed_items}/{metrics.total_items} items, "
                f"throughput: {metrics.throughput:.2f} items/sec"
            )
            
            return results
        
        return wrapper
    return decorator