"""
Tests for the performance optimization system.
"""

import pytest
import time
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.core.performance_optimization import (
    BatchConfig, PerformanceMetrics, MemoryManager, BatchProcessor,
    ImageBatchProcessor, PerformanceProfiler, optimize_memory, batch_process
)


class TestBatchConfig:
    """Test BatchConfig functionality."""
    
    def test_default_config(self):
        """Test default batch configuration."""
        config = BatchConfig()
        
        assert config.batch_size == 32
        assert config.max_workers == 4
        assert config.use_multiprocessing is False
        assert config.memory_limit_mb == 1024
        assert config.timeout_seconds == 300.0
        assert config.enable_progress_tracking is True
    
    def test_custom_config(self):
        """Test custom batch configuration."""
        config = BatchConfig(
            batch_size=64,
            max_workers=8,
            use_multiprocessing=True,
            memory_limit_mb=2048,
            timeout_seconds=600.0,
            enable_progress_tracking=False
        )
        
        assert config.batch_size == 64
        assert config.max_workers == 8
        assert config.use_multiprocessing is True
        assert config.memory_limit_mb == 2048
        assert config.timeout_seconds == 600.0
        assert config.enable_progress_tracking is False


class TestPerformanceMetrics:
    """Test PerformanceMetrics functionality."""
    
    def test_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            total_items=100,
            processed_items=80,
            failed_items=5
        )
        
        assert metrics.total_items == 100
        assert metrics.processed_items == 80
        assert metrics.failed_items == 5
        assert metrics.start_time > 0
        assert metrics.end_time is None
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        metrics = PerformanceMetrics()
        start_time = time.time()
        metrics.start_time = start_time
        
        # Test with end_time set
        metrics.end_time = start_time + 5.0
        assert abs(metrics.duration - 5.0) < 0.1
        
        # Test without end_time (should use current time)
        metrics.end_time = None
        time.sleep(0.1)
        assert metrics.duration > 0.1
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        metrics = PerformanceMetrics(processed_items=100)
        metrics.start_time = time.time() - 10.0  # 10 seconds ago
        metrics.end_time = time.time()
        
        throughput = metrics.throughput
        assert 9.0 < throughput < 11.0  # Approximately 10 items/sec
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = PerformanceMetrics(
            processed_items=80,
            failed_items=20
        )
        
        assert metrics.success_rate == 80.0
        
        # Test with no items
        empty_metrics = PerformanceMetrics()
        assert empty_metrics.success_rate == 0.0


class TestMemoryManager:
    """Test MemoryManager functionality."""
    
    def test_memory_manager_creation(self):
        """Test creating memory manager."""
        manager = MemoryManager(memory_limit_mb=512)
        
        assert manager.memory_limit_mb == 512
        assert manager.memory_limit_bytes == 512 * 1024 * 1024
    
    @patch('src.core.performance_optimization.psutil')
    def test_get_memory_usage(self, mock_psutil):
        """Test getting memory usage."""
        manager = MemoryManager()
        
        # Mock memory info
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100 MB
        mock_psutil.Process.return_value = mock_process
        
        usage = manager.get_memory_usage()
        assert usage == 100.0
    
    @patch('src.core.performance_optimization.psutil')
    def test_get_memory_usage_no_psutil(self, mock_psutil):
        """Test memory usage without psutil."""
        mock_psutil.side_effect = ImportError("psutil not available")
        
        manager = MemoryManager()
        usage = manager.get_memory_usage()
        assert usage == 0.0
    
    def test_check_memory_limit(self):
        """Test memory limit checking."""
        manager = MemoryManager(memory_limit_mb=100)
        
        with patch.object(manager, 'get_memory_usage', return_value=50.0):
            assert manager.check_memory_limit() is True
        
        with patch.object(manager, 'get_memory_usage', return_value=150.0):
            assert manager.check_memory_limit() is False
    
    def test_force_garbage_collection(self):
        """Test forcing garbage collection."""
        manager = MemoryManager()
        
        with patch('gc.collect') as mock_gc:
            manager.force_garbage_collection()
            mock_gc.assert_called_once()
    
    def test_optimize_memory_usage(self):
        """Test memory optimization."""
        manager = MemoryManager()
        
        with patch.object(manager, 'get_memory_usage', side_effect=[200.0, 150.0]):
            with patch.object(manager, 'force_garbage_collection'):
                stats = manager.optimize_memory_usage()
                
                assert stats['before_usage_mb'] == 200.0
                assert stats['after_usage_mb'] == 150.0
                assert stats['freed_mb'] == 50.0
    
    @patch('src.core.performance_optimization.psutil')
    def test_get_memory_stats(self, mock_psutil):
        """Test getting detailed memory statistics."""
        manager = MemoryManager()
        
        # Mock psutil objects
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process.memory_info.return_value.vms = 200 * 1024 * 1024
        mock_process.memory_percent.return_value = 10.0
        
        mock_virtual_memory = Mock()
        mock_virtual_memory.available = 500 * 1024 * 1024
        
        mock_psutil.Process.return_value = mock_process
        mock_psutil.virtual_memory.return_value = mock_virtual_memory
        
        stats = manager.get_memory_stats()
        
        assert stats['rss_mb'] == 100.0
        assert stats['vms_mb'] == 200.0
        assert stats['percent'] == 10.0
        assert stats['available_mb'] == 500.0


class TestBatchProcessor:
    """Test BatchProcessor functionality."""
    
    def test_batch_processor_creation(self):
        """Test creating batch processor."""
        config = BatchConfig(batch_size=16, max_workers=2)
        processor = BatchProcessor(config)
        
        assert processor.config.batch_size == 16
        assert processor.config.max_workers == 2
    
    def test_process_batch_simple(self):
        """Test simple batch processing."""
        processor = BatchProcessor(BatchConfig(batch_size=5, max_workers=1))
        
        def square_function(x):
            return x * x
        
        items = list(range(10))
        results, metrics = processor.process_batch(items, square_function)
        
        expected = [x * x for x in items]
        assert results == expected
        assert metrics.total_items == 10
        assert metrics.processed_items == 10
        assert metrics.failed_items == 0
    
    def test_process_batch_with_failures(self):
        """Test batch processing with some failures."""
        processor = BatchProcessor(BatchConfig(batch_size=5, max_workers=1))
        
        def failing_function(x):
            if x == 5:
                raise ValueError("Intentional failure")
            return x * 2
        
        items = list(range(10))
        results, metrics = processor.process_batch(items, failing_function)
        
        # Should have 9 successful results and 1 None (failure)
        assert len(results) == 10
        assert results[5] is None  # Failed item
        assert metrics.processed_items == 9
        assert metrics.failed_items == 1
    
    def test_process_batch_threading(self):
        """Test batch processing with threading."""
        config = BatchConfig(batch_size=5, max_workers=2, use_multiprocessing=False)
        processor = BatchProcessor(config)
        
        def slow_function(x):
            time.sleep(0.01)  # Small delay
            return x + 1
        
        items = list(range(10))
        results, metrics = processor.process_batch(items, slow_function)
        
        expected = [x + 1 for x in items]
        assert results == expected
        assert metrics.processed_items == 10
    
    def test_memory_limit_checking(self):
        """Test memory limit checking during processing."""
        processor = BatchProcessor(BatchConfig(batch_size=5))
        
        # Mock memory manager to simulate high memory usage
        with patch.object(processor.memory_manager, 'check_memory_limit', return_value=False):
            with patch.object(processor.memory_manager, 'optimize_memory_usage') as mock_optimize:
                
                def simple_function(x):
                    return x
                
                items = list(range(10))
                results, metrics = processor.process_batch(items, simple_function)
                
                # Should have called memory optimization
                assert mock_optimize.call_count >= 1
    
    def test_safe_process_item(self):
        """Test safe item processing."""
        def working_function(x):
            return x * 2
        
        def failing_function(x):
            raise ValueError("Test error")
        
        # Test successful processing
        result = BatchProcessor._safe_process_item(5, working_function)
        assert result == 10
        
        # Test failed processing
        result = BatchProcessor._safe_process_item(5, failing_function)
        assert result is None


class TestImageBatchProcessor:
    """Test ImageBatchProcessor functionality."""
    
    def test_image_processor_creation(self):
        """Test creating image batch processor."""
        processor = ImageBatchProcessor()
        
        assert processor.image_cache == {}
        assert processor.cache_size_limit == 100
    
    def test_load_image_opencv(self):
        """Test loading image with OpenCV."""
        processor = ImageBatchProcessor()
        
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create a simple test image
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            with patch('cv2.imread', return_value=test_image):
                loaded_image = processor._load_image(temp_path)
                assert loaded_image.shape == (100, 100, 3)
        
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_image_pil_fallback(self):
        """Test loading image with PIL fallback."""
        processor = ImageBatchProcessor()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            # Mock cv2 import error to trigger PIL fallback
            with patch('src.core.performance_optimization.cv2', side_effect=ImportError):
                with patch('PIL.Image.open') as mock_open:
                    mock_image = Mock()
                    mock_image.__array__ = Mock(return_value=np.zeros((100, 100, 3)))
                    mock_open.return_value = mock_image
                    
                    with patch('numpy.array', return_value=np.zeros((100, 100, 3))):
                        loaded_image = processor._load_image(temp_path)
                        assert loaded_image.shape == (100, 100, 3)
        
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_preload_images(self):
        """Test preloading images into cache."""
        processor = ImageBatchProcessor()
        
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        
        with patch.object(processor, '_load_image', return_value=test_image):
            image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
            processor._preload_images(image_paths)
            
            assert len(processor.image_cache) == 3
            for path in image_paths:
                assert path in processor.image_cache
    
    def test_cache_size_limit(self):
        """Test image cache size limit."""
        processor = ImageBatchProcessor()
        processor.cache_size_limit = 2  # Small limit for testing
        
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        
        with patch.object(processor, '_load_image', return_value=test_image):
            image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
            processor._preload_images(image_paths)
            
            # Should only keep last 2 images
            assert len(processor.image_cache) == 2
            assert 'image1.jpg' not in processor.image_cache
            assert 'image2.jpg' in processor.image_cache
            assert 'image3.jpg' in processor.image_cache
    
    def test_clear_cache(self):
        """Test clearing image cache."""
        processor = ImageBatchProcessor()
        
        # Add some items to cache
        processor.image_cache['test1.jpg'] = np.zeros((50, 50, 3))
        processor.image_cache['test2.jpg'] = np.zeros((50, 50, 3))
        
        processor.clear_cache()
        
        assert len(processor.image_cache) == 0


class TestPerformanceProfiler:
    """Test PerformanceProfiler functionality."""
    
    def test_profiler_creation(self):
        """Test creating performance profiler."""
        profiler = PerformanceProfiler()
        
        assert profiler.operation_times == {}
        assert profiler.memory_snapshots == []
        assert profiler.cpu_snapshots == []
        assert profiler._monitoring_active is False
    
    def test_profile_operation_decorator(self):
        """Test operation profiling decorator."""
        profiler = PerformanceProfiler()
        
        @profiler.profile_operation("test_operation")
        def test_function(x):
            time.sleep(0.01)
            return x * 2
        
        result = test_function(5)
        
        assert result == 10
        assert "test_operation" in profiler.operation_times
        assert len(profiler.operation_times["test_operation"]) == 1
        assert profiler.operation_times["test_operation"][0] > 0.01
    
    def test_profile_operation_with_error(self):
        """Test profiling operation that raises an error."""
        profiler = PerformanceProfiler()
        
        @profiler.profile_operation("failing_operation")
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        # Should record failed operation
        assert "failing_operation_failed" in profiler.operation_times
    
    def test_record_operation_time(self):
        """Test recording operation times."""
        profiler = PerformanceProfiler()
        
        profiler._record_operation_time("test_op", 1.5)
        profiler._record_operation_time("test_op", 2.0)
        
        assert len(profiler.operation_times["test_op"]) == 2
        assert profiler.operation_times["test_op"] == [1.5, 2.0]
    
    def test_operation_times_limit(self):
        """Test operation times size limit."""
        profiler = PerformanceProfiler()
        
        # Record more than 1000 measurements
        for i in range(1100):
            profiler._record_operation_time("test_op", float(i))
        
        # Should keep only last 1000
        assert len(profiler.operation_times["test_op"]) == 1000
        assert profiler.operation_times["test_op"][0] == 100.0
    
    @patch('src.core.performance_optimization.psutil')
    def test_get_memory_usage(self, mock_psutil):
        """Test getting memory usage."""
        profiler = PerformanceProfiler()
        
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 200 * 1024 * 1024
        mock_psutil.Process.return_value = mock_process
        
        usage = profiler._get_memory_usage()
        assert usage == 200.0
    
    def test_percentile_calculation(self):
        """Test percentile calculation."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        p50 = PerformanceProfiler._percentile(values, 50)
        p95 = PerformanceProfiler._percentile(values, 95)
        
        assert p50 == 5
        assert p95 == 9
    
    def test_calculate_trend(self):
        """Test trend calculation."""
        # Increasing trend
        increasing_values = [1, 2, 3, 4, 5]
        trend = PerformanceProfiler._calculate_trend(increasing_values)
        assert trend > 0
        
        # Decreasing trend
        decreasing_values = [5, 4, 3, 2, 1]
        trend = PerformanceProfiler._calculate_trend(decreasing_values)
        assert trend < 0
        
        # Flat trend
        flat_values = [3, 3, 3, 3, 3]
        trend = PerformanceProfiler._calculate_trend(flat_values)
        assert abs(trend) < 0.1
    
    def test_calculate_stability(self):
        """Test stability calculation."""
        # Stable values
        stable_values = [10, 10.1, 9.9, 10.05, 9.95]
        stability = PerformanceProfiler._calculate_stability(stable_values)
        assert stability > 0.8
        
        # Unstable values
        unstable_values = [1, 10, 2, 15, 3]
        stability = PerformanceProfiler._calculate_stability(unstable_values)
        assert stability < 0.5
    
    def test_get_performance_report(self):
        """Test generating performance report."""
        profiler = PerformanceProfiler()
        
        # Add some test data
        profiler.operation_times["test_op"] = [1.0, 2.0, 3.0, 4.0, 5.0]
        profiler.memory_snapshots = [(time.time(), 100.0), (time.time(), 110.0)]
        profiler.cpu_snapshots = [(time.time(), 50.0), (time.time(), 60.0)]
        
        report = profiler.get_performance_report()
        
        assert 'timestamp' in report
        assert 'operations' in report
        assert 'memory_analysis' in report
        assert 'cpu_analysis' in report
        assert 'bottlenecks' in report
        assert 'recommendations' in report
        
        # Check operation analysis
        assert 'test_op' in report['operations']
        op_stats = report['operations']['test_op']
        assert op_stats['count'] == 5
        assert op_stats['avg_time'] == 3.0
        assert op_stats['min_time'] == 1.0
        assert op_stats['max_time'] == 5.0


class TestDecorators:
    """Test decorator functions."""
    
    def test_optimize_memory_decorator(self):
        """Test optimize_memory decorator."""
        @optimize_memory()
        def test_function():
            return "success"
        
        with patch('src.core.performance_optimization.memory_manager') as mock_manager:
            mock_manager.check_memory_limit.return_value = True
            
            result = test_function()
            
            assert result == "success"
            mock_manager.check_memory_limit.assert_called_once()
            mock_manager.force_garbage_collection.assert_called_once()
    
    def test_optimize_memory_with_limit_exceeded(self):
        """Test optimize_memory decorator when memory limit is exceeded."""
        @optimize_memory()
        def test_function():
            return "success"
        
        with patch('src.core.performance_optimization.memory_manager') as mock_manager:
            mock_manager.check_memory_limit.return_value = False
            
            result = test_function()
            
            assert result == "success"
            mock_manager.optimize_memory_usage.assert_called_once()
    
    def test_batch_process_decorator(self):
        """Test batch_process decorator."""
        @batch_process(batch_size=5, max_workers=2)
        def process_item(item):
            return item * 2
        
        items = list(range(10))
        results = process_item(items)
        
        expected = [x * 2 for x in items]
        assert results == expected


if __name__ == "__main__":
    pytest.main([__file__])