"""
Monitoring and logging system for error tracking and system health.

This module provides comprehensive monitoring capabilities including
error tracking, performance metrics, and system health monitoring.
"""

import json
import logging
import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum

from .error_handling import ProcessingError, ErrorCategory, ErrorSeverity


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Represents a single metric measurement."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp,
            'labels': self.labels
        }


@dataclass
class HealthCheck:
    """Represents a health check result."""
    component: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: str
    timestamp: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health check to dictionary."""
        return asdict(self)


class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_metrics: Maximum number of metrics to keep in memory
        """
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Value to increment by
            labels: Optional labels for the metric
        """
        with self.lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
            
            metric = Metric(
                name=name,
                value=self.counters[key],
                metric_type=MetricType.COUNTER,
                timestamp=time.time(),
                labels=labels or {}
            )
            self.metrics.append(metric)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric value.
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels for the metric
        """
        with self.lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
            
            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                timestamp=time.time(),
                labels=labels or {}
            )
            self.metrics.append(metric)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record a histogram metric value.
        
        Args:
            name: Metric name
            value: Value to record
            labels: Optional labels for the metric
        """
        with self.lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            
            # Keep only last 1000 values per histogram
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            
            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                timestamp=time.time(),
                labels=labels or {}
            )
            self.metrics.append(metric)
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record a timer metric value.
        
        Args:
            name: Metric name
            duration: Duration in seconds
            labels: Optional labels for the metric
        """
        with self.lock:
            key = self._make_key(name, labels)
            self.timers[key].append(duration)
            
            # Keep only last 1000 values per timer
            if len(self.timers[key]) > 1000:
                self.timers[key] = self.timers[key][-1000:]
            
            metric = Metric(
                name=name,
                value=duration,
                metric_type=MetricType.TIMER,
                timestamp=time.time(),
                labels=labels or {}
            )
            self.metrics.append(metric)
    
    def get_counter_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value."""
        key = self._make_key(name, labels)
        return self.counters.get(key, 0)
    
    def get_gauge_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current gauge value."""
        key = self._make_key(name, labels)
        return self.gauges.get(key)
    
    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        key = self._make_key(name, labels)
        values = self.histograms.get(key, [])
        
        if not values:
            return {'count': 0, 'sum': 0.0, 'avg': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'count': len(values),
            'sum': sum(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }
    
    def get_timer_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get timer statistics."""
        key = self._make_key(name, labels)
        values = self.timers.get(key, [])
        
        if not values:
            return {'count': 0, 'sum': 0.0, 'avg': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'count': len(values),
            'sum': sum(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }
    
    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Get all collected metrics."""
        with self.lock:
            return [metric.to_dict() for metric in self.metrics]
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        with self.lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for metric storage."""
        if not labels:
            return name
        
        label_str = ','.join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"


class HealthMonitor:
    """Monitors system health and component status."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.health_checks: Dict[str, HealthCheck] = {}
        self.check_functions: Dict[str, Callable[[], HealthCheck]] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def register_health_check(self, component: str, check_function: Callable[[], HealthCheck]) -> None:
        """
        Register a health check function for a component.
        
        Args:
            component: Component name
            check_function: Function that returns HealthCheck result
        """
        self.check_functions[component] = check_function
        self.logger.info(f"Registered health check for {component}")
    
    def run_health_check(self, component: str) -> Optional[HealthCheck]:
        """
        Run health check for a specific component.
        
        Args:
            component: Component name
            
        Returns:
            HealthCheck result or None if component not registered
        """
        if component not in self.check_functions:
            return None
        
        try:
            check_result = self.check_functions[component]()
            with self.lock:
                self.health_checks[component] = check_result
            return check_result
        except Exception as e:
            self.logger.error(f"Health check failed for {component}: {e}")
            error_check = HealthCheck(
                component=component,
                status='unhealthy',
                message=f"Health check error: {str(e)}",
                timestamp=time.time(),
                details={'error': str(e)}
            )
            with self.lock:
                self.health_checks[component] = error_check
            return error_check
    
    def run_all_health_checks(self) -> Dict[str, HealthCheck]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary of component names to HealthCheck results
        """
        results = {}
        for component in self.check_functions:
            results[component] = self.run_health_check(component)
        return results
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Dictionary with system health information
        """
        with self.lock:
            health_checks = self.health_checks.copy()
        
        if not health_checks:
            return {
                'status': 'unknown',
                'healthy_components': 0,
                'degraded_components': 0,
                'unhealthy_components': 0,
                'total_components': 0,
                'components': {}
            }
        
        healthy = sum(1 for check in health_checks.values() if check.status == 'healthy')
        degraded = sum(1 for check in health_checks.values() if check.status == 'degraded')
        unhealthy = sum(1 for check in health_checks.values() if check.status == 'unhealthy')
        total = len(health_checks)
        
        # Determine overall status
        if unhealthy > 0:
            overall_status = 'unhealthy'
        elif degraded > 0:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
        
        return {
            'status': overall_status,
            'healthy_components': healthy,
            'degraded_components': degraded,
            'unhealthy_components': unhealthy,
            'total_components': total,
            'components': {name: check.to_dict() for name, check in health_checks.items()}
        }


class ErrorTracker:
    """Tracks and analyzes error patterns."""
    
    def __init__(self, max_errors: int = 1000):
        """
        Initialize error tracker.
        
        Args:
            max_errors: Maximum number of errors to keep in memory
        """
        self.max_errors = max_errors
        self.errors: deque = deque(maxlen=max_errors)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_rates: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def track_error(self, error: ProcessingError) -> None:
        """
        Track a processing error.
        
        Args:
            error: Processing error to track
        """
        with self.lock:
            error_data = {
                'error_id': error.error_id,
                'category': error.category.value,
                'severity': error.severity.value,
                'message': error.message,
                'timestamp': error.context.timestamp,
                'component': error.context.component,
                'operation': error.context.operation,
                'recoverable': error.recoverable
            }
            
            self.errors.append(error_data)
            
            # Update error counts
            category_key = error.category.value
            severity_key = f"{error.category.value}_{error.severity.value}"
            
            self.error_counts[category_key] += 1
            self.error_counts[severity_key] += 1
            self.error_counts['total'] += 1
    
    def get_error_rate(self, category: Optional[str] = None, time_window: int = 300) -> float:
        """
        Calculate error rate for a category within a time window.
        
        Args:
            category: Error category (None for all errors)
            time_window: Time window in seconds
            
        Returns:
            Error rate (errors per second)
        """
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self.lock:
            relevant_errors = [
                error for error in self.errors
                if error['timestamp'] >= cutoff_time and (
                    category is None or error['category'] == category
                )
            ]
        
        return len(relevant_errors) / time_window if time_window > 0 else 0.0
    
    def get_error_distribution(self) -> Dict[str, Any]:
        """
        Get error distribution statistics.
        
        Returns:
            Dictionary with error distribution data
        """
        with self.lock:
            total_errors = len(self.errors)
            
            if total_errors == 0:
                return {
                    'total_errors': 0,
                    'by_category': {},
                    'by_severity': {},
                    'by_component': {},
                    'recent_errors': []
                }
            
            # Count by category
            by_category = defaultdict(int)
            by_severity = defaultdict(int)
            by_component = defaultdict(int)
            
            for error in self.errors:
                by_category[error['category']] += 1
                by_severity[error['severity']] += 1
                by_component[error['component']] += 1
            
            # Get recent errors (last 10)
            recent_errors = list(self.errors)[-10:]
            
            return {
                'total_errors': total_errors,
                'by_category': dict(by_category),
                'by_severity': dict(by_severity),
                'by_component': dict(by_component),
                'recent_errors': recent_errors
            }
    
    def get_error_trends(self, time_buckets: int = 10) -> Dict[str, List[int]]:
        """
        Get error trends over time buckets.
        
        Args:
            time_buckets: Number of time buckets to analyze
            
        Returns:
            Dictionary with error trends by category
        """
        if not self.errors:
            return {}
        
        current_time = time.time()
        oldest_time = min(error['timestamp'] for error in self.errors)
        time_span = current_time - oldest_time
        bucket_size = time_span / time_buckets if time_buckets > 0 else 1
        
        trends = defaultdict(lambda: [0] * time_buckets)
        
        with self.lock:
            for error in self.errors:
                bucket_index = min(
                    int((error['timestamp'] - oldest_time) / bucket_size),
                    time_buckets - 1
                )
                trends[error['category']][bucket_index] += 1
                trends['total'][bucket_index] += 1
        
        return dict(trends)


class PerformanceProfiler:
    """Profiles system performance and identifies bottlenecks."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.execution_times: Dict[str, List[float]] = defaultdict(list)
        self.memory_usage: Dict[str, List[float]] = defaultdict(list)
        self.active_operations: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def start_operation(self, operation_id: str) -> None:
        """
        Start timing an operation.
        
        Args:
            operation_id: Unique identifier for the operation
        """
        with self.lock:
            self.active_operations[operation_id] = time.time()
    
    def end_operation(self, operation_id: str, component: str = "unknown") -> Optional[float]:
        """
        End timing an operation and record the duration.
        
        Args:
            operation_id: Unique identifier for the operation
            component: Component name for categorization
            
        Returns:
            Operation duration in seconds or None if not found
        """
        with self.lock:
            if operation_id not in self.active_operations:
                self.logger.warning(f"Operation {operation_id} not found in active operations")
                return None
            
            start_time = self.active_operations.pop(operation_id)
            duration = time.time() - start_time
            
            self.execution_times[component].append(duration)
            
            # Keep only last 1000 measurements per component
            if len(self.execution_times[component]) > 1000:
                self.execution_times[component] = self.execution_times[component][-1000:]
            
            return duration
    
    def record_memory_usage(self, component: str, memory_mb: float) -> None:
        """
        Record memory usage for a component.
        
        Args:
            component: Component name
            memory_mb: Memory usage in megabytes
        """
        with self.lock:
            self.memory_usage[component].append(memory_mb)
            
            # Keep only last 1000 measurements per component
            if len(self.memory_usage[component]) > 1000:
                self.memory_usage[component] = self.memory_usage[component][-1000:]
    
    def get_performance_stats(self, component: str) -> Dict[str, Any]:
        """
        Get performance statistics for a component.
        
        Args:
            component: Component name
            
        Returns:
            Dictionary with performance statistics
        """
        with self.lock:
            execution_times = self.execution_times.get(component, [])
            memory_usage = self.memory_usage.get(component, [])
        
        stats = {'component': component}
        
        # Execution time statistics
        if execution_times:
            stats['execution_time'] = {
                'count': len(execution_times),
                'avg': sum(execution_times) / len(execution_times),
                'min': min(execution_times),
                'max': max(execution_times),
                'recent_avg': sum(execution_times[-10:]) / min(len(execution_times), 10)
            }
        else:
            stats['execution_time'] = None
        
        # Memory usage statistics
        if memory_usage:
            stats['memory_usage'] = {
                'count': len(memory_usage),
                'avg': sum(memory_usage) / len(memory_usage),
                'min': min(memory_usage),
                'max': max(memory_usage),
                'recent_avg': sum(memory_usage[-10:]) / min(len(memory_usage), 10)
            }
        else:
            stats['memory_usage'] = None
        
        return stats
    
    def get_bottlenecks(self, threshold_percentile: float = 0.95) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks.
        
        Args:
            threshold_percentile: Percentile threshold for identifying bottlenecks
            
        Returns:
            List of bottleneck information
        """
        bottlenecks = []
        
        with self.lock:
            for component, times in self.execution_times.items():
                if len(times) < 10:  # Need sufficient data
                    continue
                
                sorted_times = sorted(times)
                threshold_index = int(len(sorted_times) * threshold_percentile)
                threshold_time = sorted_times[threshold_index]
                avg_time = sum(times) / len(times)
                
                if threshold_time > avg_time * 2:  # Significant outliers
                    bottlenecks.append({
                        'component': component,
                        'avg_time': avg_time,
                        'threshold_time': threshold_time,
                        'outlier_ratio': threshold_time / avg_time,
                        'sample_count': len(times)
                    })
        
        # Sort by outlier ratio (worst first)
        bottlenecks.sort(key=lambda x: x['outlier_ratio'], reverse=True)
        return bottlenecks


class MonitoringSystem:
    """Central monitoring system that coordinates all monitoring components."""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize monitoring system.
        
        Args:
            log_file: Optional log file path
        """
        self.metrics_collector = MetricsCollector()
        self.health_monitor = HealthMonitor()
        self.error_tracker = ErrorTracker()
        self.performance_profiler = PerformanceProfiler()
        
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        
        # Setup logging
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dictionary with complete system status
        """
        return {
            'timestamp': time.time(),
            'health': self.health_monitor.get_system_health(),
            'errors': self.error_tracker.get_error_distribution(),
            'performance': {
                component: self.performance_profiler.get_performance_stats(component)
                for component in self.performance_profiler.execution_times.keys()
            },
            'bottlenecks': self.performance_profiler.get_bottlenecks(),
            'metrics_summary': {
                'total_metrics': len(self.metrics_collector.metrics),
                'counters': len(self.metrics_collector.counters),
                'gauges': len(self.metrics_collector.gauges),
                'histograms': len(self.metrics_collector.histograms),
                'timers': len(self.metrics_collector.timers)
            }
        }
    
    def export_metrics(self, file_path: str) -> None:
        """
        Export all metrics to a file.
        
        Args:
            file_path: Path to export file
        """
        status = self.get_system_status()
        
        with open(file_path, 'w') as f:
            json.dump(status, f, indent=2, default=str)
        
        self.logger.info(f"Metrics exported to {file_path}")
    
    def clear_all_data(self) -> None:
        """Clear all monitoring data."""
        self.metrics_collector.clear_metrics()
        self.error_tracker.errors.clear()
        self.error_tracker.error_counts.clear()
        self.performance_profiler.execution_times.clear()
        self.performance_profiler.memory_usage.clear()
        self.performance_profiler.active_operations.clear()
        
        self.logger.info("All monitoring data cleared")


# Global monitoring system instance
monitoring_system = MonitoringSystem()


def monitor_performance(component: str):
    """
    Decorator to monitor function performance.
    
    Args:
        component: Component name for monitoring
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            operation_id = f"{component}_{func.__name__}_{int(time.time() * 1000000)}"
            
            # Start monitoring
            monitoring_system.performance_profiler.start_operation(operation_id)
            monitoring_system.metrics_collector.increment_counter(
                f"{component}_operations_total",
                labels={'operation': func.__name__}
            )
            
            try:
                result = func(*args, **kwargs)
                
                # Record success
                monitoring_system.metrics_collector.increment_counter(
                    f"{component}_operations_success",
                    labels={'operation': func.__name__}
                )
                
                return result
                
            except Exception as e:
                # Record error
                monitoring_system.metrics_collector.increment_counter(
                    f"{component}_operations_error",
                    labels={'operation': func.__name__, 'error_type': type(e).__name__}
                )
                raise
                
            finally:
                # End monitoring
                duration = monitoring_system.performance_profiler.end_operation(operation_id, component)
                if duration is not None:
                    monitoring_system.metrics_collector.record_timer(
                        f"{component}_operation_duration",
                        duration,
                        labels={'operation': func.__name__}
                    )
        
        return wrapper
    return decorator