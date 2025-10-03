"""
Tests for the monitoring system.
"""

import pytest
import json
import time
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path

from src.core.monitoring import (
    Metric, HealthCheck, MetricsCollector, HealthMonitor, ErrorTracker,
    PerformanceProfiler, MonitoringSystem, MetricType, monitor_performance
)
from src.core.error_handling import ProcessingError, ErrorCategory, ErrorSeverity


class TestMetric:
    """Test Metric data class."""
    
    def test_metric_creation(self):
        """Test creating a metric."""
        metric = Metric(
            name="test_metric",
            value=42.0,
            metric_type=MetricType.GAUGE,
            timestamp=time.time(),
            labels={"component": "test"}
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.metric_type == MetricType.GAUGE
        assert metric.labels == {"component": "test"}
    
    def test_metric_to_dict(self):
        """Test converting metric to dictionary."""
        timestamp = time.time()
        metric = Metric(
            name="test_metric",
            value=42.0,
            metric_type=MetricType.COUNTER,
            timestamp=timestamp,
            labels={"key": "value"}
        )
        
        result = metric.to_dict()
        
        assert result == {
            'name': "test_metric",
            'value': 42.0,
            'type': "counter",
            'timestamp': timestamp,
            'labels': {"key": "value"}
        }


class TestHealthCheck:
    """Test HealthCheck data class."""
    
    def test_health_check_creation(self):
        """Test creating a health check."""
        timestamp = time.time()
        check = HealthCheck(
            component="test_component",
            status="healthy",
            message="All systems operational",
            timestamp=timestamp,
            details={"uptime": 3600}
        )
        
        assert check.component == "test_component"
        assert check.status == "healthy"
        assert check.message == "All systems operational"
        assert check.timestamp == timestamp
        assert check.details == {"uptime": 3600}
    
    def test_health_check_to_dict(self):
        """Test converting health check to dictionary."""
        timestamp = time.time()
        check = HealthCheck(
            component="test",
            status="degraded",
            message="Performance issues",
            timestamp=timestamp,
            details={"cpu": 90}
        )
        
        result = check.to_dict()
        
        expected = {
            'component': "test",
            'status': "degraded",
            'message': "Performance issues",
            'timestamp': timestamp,
            'details': {"cpu": 90}
        }
        
        assert result == expected


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    def test_increment_counter(self):
        """Test incrementing counter metrics."""
        collector = MetricsCollector()
        
        collector.increment_counter("test_counter", 5)
        collector.increment_counter("test_counter", 3)
        
        assert collector.get_counter_value("test_counter") == 8
        assert len(collector.metrics) == 2
    
    def test_increment_counter_with_labels(self):
        """Test incrementing counter with labels."""
        collector = MetricsCollector()
        
        labels1 = {"method": "GET"}
        labels2 = {"method": "POST"}
        
        collector.increment_counter("requests", 1, labels1)
        collector.increment_counter("requests", 2, labels2)
        collector.increment_counter("requests", 1, labels1)
        
        assert collector.get_counter_value("requests", labels1) == 2
        assert collector.get_counter_value("requests", labels2) == 2
    
    def test_set_gauge(self):
        """Test setting gauge metrics."""
        collector = MetricsCollector()
        
        collector.set_gauge("temperature", 25.5)
        collector.set_gauge("temperature", 26.0)
        
        assert collector.get_gauge_value("temperature") == 26.0
        assert len(collector.metrics) == 2
    
    def test_record_histogram(self):
        """Test recording histogram metrics."""
        collector = MetricsCollector()
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            collector.record_histogram("response_time", value)
        
        stats = collector.get_histogram_stats("response_time")
        
        assert stats['count'] == 5
        assert stats['sum'] == 15.0
        assert stats['avg'] == 3.0
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
    
    def test_record_timer(self):
        """Test recording timer metrics."""
        collector = MetricsCollector()
        
        durations = [0.1, 0.2, 0.15, 0.3, 0.25]
        for duration in durations:
            collector.record_timer("execution_time", duration)
        
        stats = collector.get_timer_stats("execution_time")
        
        assert stats['count'] == 5
        assert stats['sum'] == 1.0
        assert stats['avg'] == 0.2
        assert stats['min'] == 0.1
        assert stats['max'] == 0.3
    
    def test_histogram_size_limit(self):
        """Test histogram size limit."""
        collector = MetricsCollector()
        
        # Record more than 1000 values
        for i in range(1100):
            collector.record_histogram("test_histogram", float(i))
        
        # Should keep only last 1000
        assert len(collector.histograms["test_histogram"]) == 1000
        assert collector.histograms["test_histogram"][0] == 100.0  # First kept value
    
    def test_get_all_metrics(self):
        """Test getting all metrics."""
        collector = MetricsCollector()
        
        collector.increment_counter("counter1", 1)
        collector.set_gauge("gauge1", 42.0)
        collector.record_histogram("hist1", 1.5)
        
        all_metrics = collector.get_all_metrics()
        
        assert len(all_metrics) == 3
        assert all(isinstance(metric, dict) for metric in all_metrics)
        assert all('name' in metric for metric in all_metrics)
    
    def test_clear_metrics(self):
        """Test clearing all metrics."""
        collector = MetricsCollector()
        
        collector.increment_counter("test", 1)
        collector.set_gauge("test", 1.0)
        
        collector.clear_metrics()
        
        assert len(collector.metrics) == 0
        assert len(collector.counters) == 0
        assert len(collector.gauges) == 0


class TestHealthMonitor:
    """Test HealthMonitor functionality."""
    
    def test_register_health_check(self):
        """Test registering health check functions."""
        monitor = HealthMonitor()
        
        def test_check():
            return HealthCheck(
                component="test",
                status="healthy",
                message="OK",
                timestamp=time.time(),
                details={}
            )
        
        monitor.register_health_check("test_component", test_check)
        
        assert "test_component" in monitor.check_functions
    
    def test_run_health_check(self):
        """Test running a health check."""
        monitor = HealthMonitor()
        
        def test_check():
            return HealthCheck(
                component="test",
                status="healthy",
                message="All good",
                timestamp=time.time(),
                details={"version": "1.0"}
            )
        
        monitor.register_health_check("test_component", test_check)
        result = monitor.run_health_check("test_component")
        
        assert result is not None
        assert result.component == "test"
        assert result.status == "healthy"
        assert result.message == "All good"
    
    def test_run_health_check_with_error(self):
        """Test running a health check that raises an error."""
        monitor = HealthMonitor()
        
        def failing_check():
            raise Exception("Check failed")
        
        monitor.register_health_check("failing_component", failing_check)
        result = monitor.run_health_check("failing_component")
        
        assert result is not None
        assert result.component == "failing_component"
        assert result.status == "unhealthy"
        assert "Check failed" in result.message
    
    def test_run_all_health_checks(self):
        """Test running all registered health checks."""
        monitor = HealthMonitor()
        
        def check1():
            return HealthCheck("comp1", "healthy", "OK", time.time(), {})
        
        def check2():
            return HealthCheck("comp2", "degraded", "Slow", time.time(), {})
        
        monitor.register_health_check("comp1", check1)
        monitor.register_health_check("comp2", check2)
        
        results = monitor.run_all_health_checks()
        
        assert len(results) == 2
        assert "comp1" in results
        assert "comp2" in results
        assert results["comp1"].status == "healthy"
        assert results["comp2"].status == "degraded"
    
    def test_get_system_health(self):
        """Test getting overall system health."""
        monitor = HealthMonitor()
        
        # Manually add some health checks
        monitor.health_checks = {
            "comp1": HealthCheck("comp1", "healthy", "OK", time.time(), {}),
            "comp2": HealthCheck("comp2", "degraded", "Slow", time.time(), {}),
            "comp3": HealthCheck("comp3", "unhealthy", "Failed", time.time(), {})
        }
        
        health = monitor.get_system_health()
        
        assert health['status'] == 'unhealthy'  # Has unhealthy components
        assert health['healthy_components'] == 1
        assert health['degraded_components'] == 1
        assert health['unhealthy_components'] == 1
        assert health['total_components'] == 3


class TestErrorTracker:
    """Test ErrorTracker functionality."""
    
    def test_track_error(self):
        """Test tracking errors."""
        tracker = ErrorTracker()
        
        error = ProcessingError(
            "Test error",
            ErrorCategory.DETECTION,
            ErrorSeverity.HIGH
        )
        
        tracker.track_error(error)
        
        assert len(tracker.errors) == 1
        assert tracker.error_counts['detection'] == 1
        assert tracker.error_counts['detection_high'] == 1
        assert tracker.error_counts['total'] == 1
    
    def test_get_error_rate(self):
        """Test calculating error rates."""
        tracker = ErrorTracker()
        
        # Track some errors
        for _ in range(5):
            error = ProcessingError("Test", ErrorCategory.DETECTION)
            tracker.track_error(error)
        
        # Calculate rate for 1 second window
        rate = tracker.get_error_rate(time_window=1)
        assert rate == 5.0  # 5 errors per second
        
        # Calculate rate for specific category
        rate = tracker.get_error_rate("detection", time_window=1)
        assert rate == 5.0
    
    def test_get_error_distribution(self):
        """Test getting error distribution."""
        tracker = ErrorTracker()
        
        # Track various errors
        errors = [
            ProcessingError("Error 1", ErrorCategory.DETECTION, ErrorSeverity.HIGH),
            ProcessingError("Error 2", ErrorCategory.DETECTION, ErrorSeverity.LOW),
            ProcessingError("Error 3", ErrorCategory.RECOGNITION, ErrorSeverity.MEDIUM),
        ]
        
        for error in errors:
            tracker.track_error(error)
        
        distribution = tracker.get_error_distribution()
        
        assert distribution['total_errors'] == 3
        assert distribution['by_category']['detection'] == 2
        assert distribution['by_category']['recognition'] == 1
        assert distribution['by_severity']['high'] == 1
        assert distribution['by_severity']['medium'] == 1
        assert distribution['by_severity']['low'] == 1
    
    def test_get_error_trends(self):
        """Test getting error trends."""
        tracker = ErrorTracker()
        
        # Add some errors with different timestamps
        base_time = time.time() - 100
        for i in range(10):
            error = ProcessingError("Test", ErrorCategory.DETECTION)
            error.context.timestamp = base_time + i * 10
            tracker.track_error(error)
        
        trends = tracker.get_error_trends(time_buckets=5)
        
        assert 'detection' in trends
        assert 'total' in trends
        assert len(trends['detection']) == 5
        assert sum(trends['total']) == 10


class TestPerformanceProfiler:
    """Test PerformanceProfiler functionality."""
    
    def test_start_end_operation(self):
        """Test starting and ending operations."""
        profiler = PerformanceProfiler()
        
        profiler.start_operation("test_op")
        time.sleep(0.01)  # Small delay
        duration = profiler.end_operation("test_op", "test_component")
        
        assert duration is not None
        assert duration > 0
        assert len(profiler.execution_times["test_component"]) == 1
    
    def test_end_nonexistent_operation(self):
        """Test ending an operation that wasn't started."""
        profiler = PerformanceProfiler()
        
        duration = profiler.end_operation("nonexistent", "test")
        
        assert duration is None
    
    def test_record_memory_usage(self):
        """Test recording memory usage."""
        profiler = PerformanceProfiler()
        
        profiler.record_memory_usage("test_component", 128.5)
        profiler.record_memory_usage("test_component", 256.0)
        
        assert len(profiler.memory_usage["test_component"]) == 2
        assert profiler.memory_usage["test_component"] == [128.5, 256.0]
    
    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        profiler = PerformanceProfiler()
        
        # Add some execution times
        profiler.execution_times["test"] = [1.0, 2.0, 3.0]
        profiler.memory_usage["test"] = [100.0, 200.0, 150.0]
        
        stats = profiler.get_performance_stats("test")
        
        assert stats['component'] == "test"
        assert stats['execution_time']['count'] == 3
        assert stats['execution_time']['avg'] == 2.0
        assert stats['execution_time']['min'] == 1.0
        assert stats['execution_time']['max'] == 3.0
        assert stats['memory_usage']['avg'] == 150.0
    
    def test_get_bottlenecks(self):
        """Test identifying performance bottlenecks."""
        profiler = PerformanceProfiler()
        
        # Add execution times with outliers
        normal_times = [1.0] * 15  # 15 normal executions
        outlier_times = [10.0] * 5  # 5 slow executions
        profiler.execution_times["slow_component"] = normal_times + outlier_times
        
        bottlenecks = profiler.get_bottlenecks(threshold_percentile=0.9)
        
        assert len(bottlenecks) > 0
        assert bottlenecks[0]['component'] == "slow_component"
        assert bottlenecks[0]['outlier_ratio'] > 2.0
    
    def test_execution_times_limit(self):
        """Test execution times size limit."""
        profiler = PerformanceProfiler()
        
        # Record more than 1000 measurements
        for i in range(1100):
            profiler.execution_times["test"].append(float(i))
        
        # Should keep only last 1000
        assert len(profiler.execution_times["test"]) == 1000
        assert profiler.execution_times["test"][0] == 100.0  # First kept value


class TestMonitoringSystem:
    """Test MonitoringSystem integration."""
    
    def test_system_initialization(self):
        """Test monitoring system initialization."""
        system = MonitoringSystem()
        
        assert system.metrics_collector is not None
        assert system.health_monitor is not None
        assert system.error_tracker is not None
        assert system.performance_profiler is not None
    
    def test_get_system_status(self):
        """Test getting comprehensive system status."""
        system = MonitoringSystem()
        
        # Add some test data
        system.metrics_collector.increment_counter("test", 1)
        system.error_tracker.track_error(
            ProcessingError("Test", ErrorCategory.DETECTION)
        )
        
        status = system.get_system_status()
        
        assert 'timestamp' in status
        assert 'health' in status
        assert 'errors' in status
        assert 'performance' in status
        assert 'bottlenecks' in status
        assert 'metrics_summary' in status
    
    def test_export_metrics(self):
        """Test exporting metrics to file."""
        system = MonitoringSystem()
        
        # Add some test data
        system.metrics_collector.increment_counter("test", 1)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            export_file = f.name
        
        try:
            system.export_metrics(export_file)
            
            # Verify file was created and contains valid JSON
            assert Path(export_file).exists()
            
            with open(export_file, 'r') as f:
                data = json.load(f)
            
            assert 'timestamp' in data
            assert 'metrics_summary' in data
            
        finally:
            Path(export_file).unlink(missing_ok=True)
    
    def test_clear_all_data(self):
        """Test clearing all monitoring data."""
        system = MonitoringSystem()
        
        # Add some test data
        system.metrics_collector.increment_counter("test", 1)
        system.error_tracker.track_error(
            ProcessingError("Test", ErrorCategory.DETECTION)
        )
        system.performance_profiler.execution_times["test"] = [1.0, 2.0]
        
        system.clear_all_data()
        
        assert len(system.metrics_collector.metrics) == 0
        assert len(system.error_tracker.errors) == 0
        assert len(system.performance_profiler.execution_times) == 0


class TestMonitorPerformanceDecorator:
    """Test monitor_performance decorator."""
    
    def test_successful_execution(self):
        """Test decorator with successful execution."""
        @monitor_performance("test_component")
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
    
    def test_execution_with_error(self):
        """Test decorator with execution error."""
        @monitor_performance("test_component")
        def failing_function():
            raise ValueError("Function failed")
        
        with pytest.raises(ValueError):
            failing_function()
    
    def test_metrics_collection(self):
        """Test that decorator collects metrics."""
        # Clear existing metrics
        from src.core.monitoring import monitoring_system
        monitoring_system.clear_all_data()
        
        @monitor_performance("test_component")
        def test_function():
            time.sleep(0.01)  # Small delay
            return "success"
        
        test_function()
        
        # Check that metrics were collected
        counters = monitoring_system.metrics_collector.counters
        assert any("test_component_operations_total" in key for key in counters.keys())
        assert any("test_component_operations_success" in key for key in counters.keys())


if __name__ == "__main__":
    pytest.main([__file__])