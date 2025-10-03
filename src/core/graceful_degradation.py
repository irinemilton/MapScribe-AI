"""
Graceful degradation utilities for handling system failures.

This module provides mechanisms to maintain system functionality
even when components fail or perform poorly.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .error_handling import ProcessingError, ErrorCategory, ErrorSeverity


class DegradationLevel(Enum):
    """Levels of system degradation."""
    NONE = "none"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class DegradationStrategy:
    """Configuration for degradation behavior."""
    level: DegradationLevel
    fallback_method: str
    quality_threshold: float
    performance_threshold: float
    enabled: bool = True


class GracefulDegradationManager:
    """Manages graceful degradation strategies for system components."""
    
    def __init__(self):
        """Initialize degradation manager."""
        self.strategies: Dict[str, DegradationStrategy] = {}
        self.component_health: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Default degradation strategies
        self._setup_default_strategies()
    
    def _setup_default_strategies(self) -> None:
        """Setup default degradation strategies for common components."""
        self.strategies.update({
            'detection': DegradationStrategy(
                level=DegradationLevel.MODERATE,
                fallback_method='simple_detection',
                quality_threshold=0.5,
                performance_threshold=5.0
            ),
            'recognition': DegradationStrategy(
                level=DegradationLevel.MINIMAL,
                fallback_method='basic_ocr',
                quality_threshold=0.6,
                performance_threshold=3.0
            ),
            'preprocessing': DegradationStrategy(
                level=DegradationLevel.MINIMAL,
                fallback_method='basic_preprocessing',
                quality_threshold=0.7,
                performance_threshold=2.0
            ),
            'model_loading': DegradationStrategy(
                level=DegradationLevel.SEVERE,
                fallback_method='backup_model',
                quality_threshold=0.3,
                performance_threshold=10.0
            )
        })
    
    def register_strategy(self, component: str, strategy: DegradationStrategy) -> None:
        """
        Register degradation strategy for a component.
        
        Args:
            component: Component name
            strategy: Degradation strategy configuration
        """
        self.strategies[component] = strategy
        self.logger.info(f"Registered degradation strategy for {component}")
    
    def update_component_health(self, component: str, health_score: float) -> None:
        """
        Update health score for a component.
        
        Args:
            component: Component name
            health_score: Health score between 0.0 and 1.0
        """
        if not 0.0 <= health_score <= 1.0:
            raise ValueError("Health score must be between 0.0 and 1.0")
        
        self.component_health[component] = health_score
        
        # Check if degradation is needed
        if component in self.strategies:
            strategy = self.strategies[component]
            if health_score < strategy.quality_threshold:
                self._trigger_degradation(component, health_score)
    
    def record_performance(self, component: str, execution_time: float) -> None:
        """
        Record performance metrics for a component.
        
        Args:
            component: Component name
            execution_time: Execution time in seconds
        """
        if component not in self.performance_history:
            self.performance_history[component] = []
        
        self.performance_history[component].append(execution_time)
        
        # Keep only last 100 measurements
        if len(self.performance_history[component]) > 100:
            self.performance_history[component] = self.performance_history[component][-100:]
        
        # Check if performance degradation is needed
        if component in self.strategies:
            strategy = self.strategies[component]
            avg_time = sum(self.performance_history[component]) / len(self.performance_history[component])
            
            if avg_time > strategy.performance_threshold:
                self._trigger_performance_degradation(component, avg_time)
    
    def _trigger_degradation(self, component: str, health_score: float) -> None:
        """
        Trigger degradation for a component based on health score.
        
        Args:
            component: Component name
            health_score: Current health score
        """
        strategy = self.strategies[component]
        
        if not strategy.enabled:
            return
        
        self.logger.warning(
            f"Triggering degradation for {component} "
            f"(health: {health_score:.3f}, threshold: {strategy.quality_threshold:.3f})"
        )
        
        # Determine degradation level based on health score
        if health_score < 0.2:
            level = DegradationLevel.CRITICAL
        elif health_score < 0.4:
            level = DegradationLevel.SEVERE
        elif health_score < 0.6:
            level = DegradationLevel.MODERATE
        else:
            level = DegradationLevel.MINIMAL
        
        # Update strategy level
        strategy.level = level
    
    def _trigger_performance_degradation(self, component: str, avg_time: float) -> None:
        """
        Trigger degradation for a component based on performance.
        
        Args:
            component: Component name
            avg_time: Average execution time
        """
        strategy = self.strategies[component]
        
        if not strategy.enabled:
            return
        
        self.logger.warning(
            f"Triggering performance degradation for {component} "
            f"(avg_time: {avg_time:.3f}s, threshold: {strategy.performance_threshold:.3f}s)"
        )
        
        # Adjust degradation level based on performance
        if strategy.level == DegradationLevel.NONE:
            strategy.level = DegradationLevel.MINIMAL
        elif strategy.level == DegradationLevel.MINIMAL:
            strategy.level = DegradationLevel.MODERATE
    
    def should_degrade(self, component: str) -> bool:
        """
        Check if a component should use degraded functionality.
        
        Args:
            component: Component name
            
        Returns:
            True if component should degrade
        """
        if component not in self.strategies:
            return False
        
        strategy = self.strategies[component]
        return strategy.enabled and strategy.level != DegradationLevel.NONE
    
    def get_degradation_level(self, component: str) -> DegradationLevel:
        """
        Get current degradation level for a component.
        
        Args:
            component: Component name
            
        Returns:
            Current degradation level
        """
        if component not in self.strategies:
            return DegradationLevel.NONE
        
        return self.strategies[component].level
    
    def get_fallback_method(self, component: str) -> Optional[str]:
        """
        Get fallback method for a component.
        
        Args:
            component: Component name
            
        Returns:
            Fallback method name or None
        """
        if component not in self.strategies:
            return None
        
        return self.strategies[component].fallback_method
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Dictionary with system health information
        """
        total_components = len(self.component_health)
        if total_components == 0:
            return {'overall_health': 1.0, 'status': 'healthy', 'components': {}}
        
        overall_health = sum(self.component_health.values()) / total_components
        
        # Determine system status
        if overall_health >= 0.8:
            status = 'healthy'
        elif overall_health >= 0.6:
            status = 'degraded'
        elif overall_health >= 0.4:
            status = 'impaired'
        else:
            status = 'critical'
        
        # Count degraded components
        degraded_components = sum(
            1 for component in self.strategies
            if self.should_degrade(component)
        )
        
        return {
            'overall_health': overall_health,
            'status': status,
            'total_components': total_components,
            'degraded_components': degraded_components,
            'components': self.component_health.copy(),
            'degradation_levels': {
                component: strategy.level.value
                for component, strategy in self.strategies.items()
            }
        }


class FallbackModelManager:
    """Manages fallback models for graceful degradation."""
    
    def __init__(self):
        """Initialize fallback model manager."""
        self.fallback_models: Dict[str, List[str]] = {}
        self.model_performance: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_fallback_models(self, model_type: str, model_paths: List[str]) -> None:
        """
        Register fallback models for a model type.
        
        Args:
            model_type: Type of model (e.g., 'detection', 'recognition')
            model_paths: List of model paths in order of preference
        """
        self.fallback_models[model_type] = model_paths
        self.logger.info(f"Registered {len(model_paths)} fallback models for {model_type}")
    
    def get_fallback_model(self, model_type: str, failed_model: Optional[str] = None) -> Optional[str]:
        """
        Get next fallback model for a given type.
        
        Args:
            model_type: Type of model
            failed_model: Path of failed model to skip
            
        Returns:
            Path to fallback model or None
        """
        if model_type not in self.fallback_models:
            return None
        
        fallback_list = self.fallback_models[model_type]
        
        # If no specific failed model, return first fallback
        if failed_model is None:
            return fallback_list[0] if fallback_list else None
        
        # Find next model after the failed one
        try:
            failed_index = fallback_list.index(failed_model)
            if failed_index + 1 < len(fallback_list):
                return fallback_list[failed_index + 1]
        except ValueError:
            # Failed model not in list, return first fallback
            return fallback_list[0] if fallback_list else None
        
        return None
    
    def record_model_performance(self, model_path: str, performance_score: float) -> None:
        """
        Record performance score for a model.
        
        Args:
            model_path: Path to model
            performance_score: Performance score between 0.0 and 1.0
        """
        self.model_performance[model_path] = performance_score
    
    def get_best_performing_model(self, model_type: str) -> Optional[str]:
        """
        Get best performing model for a type based on recorded performance.
        
        Args:
            model_type: Type of model
            
        Returns:
            Path to best performing model or None
        """
        if model_type not in self.fallback_models:
            return None
        
        models = self.fallback_models[model_type]
        
        # Filter models with recorded performance
        models_with_performance = [
            (model, self.model_performance.get(model, 0.0))
            for model in models
            if model in self.model_performance
        ]
        
        if not models_with_performance:
            return models[0] if models else None
        
        # Return model with highest performance
        best_model = max(models_with_performance, key=lambda x: x[1])
        return best_model[0]


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to add circuit breaker functionality.
        
        Args:
            func: Function to protect with circuit breaker
            
        Returns:
            Protected function
        """
        def wrapper(*args, **kwargs):
            if self.state == 'open':
                if self._should_attempt_reset():
                    self.state = 'half-open'
                    self.logger.info(f"Circuit breaker half-open for {func.__name__}")
                else:
                    raise ProcessingError(
                        f"Circuit breaker open for {func.__name__}",
                        ErrorCategory.SYSTEM,
                        ErrorSeverity.HIGH,
                        recoverable=False
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful execution."""
        if self.state == 'half-open':
            self.state = 'closed'
            self.logger.info("Circuit breaker closed after successful recovery")
        
        self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            self.logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


class ResourceMonitor:
    """Monitors system resources and triggers degradation when needed."""
    
    def __init__(self):
        """Initialize resource monitor."""
        self.memory_threshold = 0.85  # 85% memory usage
        self.cpu_threshold = 0.90     # 90% CPU usage
        self.disk_threshold = 0.95    # 95% disk usage
        self.logger = logging.getLogger(__name__)
    
    def check_memory_usage(self) -> float:
        """
        Check current memory usage.
        
        Returns:
            Memory usage as percentage (0.0 to 1.0)
        """
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
            return 0.0
    
    def check_cpu_usage(self) -> float:
        """
        Check current CPU usage.
        
        Returns:
            CPU usage as percentage (0.0 to 1.0)
        """
        try:
            import psutil
            return psutil.cpu_percent(interval=1) / 100.0
        except ImportError:
            self.logger.warning("psutil not available for CPU monitoring")
            return 0.0
    
    def should_trigger_degradation(self) -> Tuple[bool, str]:
        """
        Check if resource usage should trigger degradation.
        
        Returns:
            Tuple of (should_degrade, reason)
        """
        memory_usage = self.check_memory_usage()
        if memory_usage > self.memory_threshold:
            return True, f"High memory usage: {memory_usage:.1%}"
        
        cpu_usage = self.check_cpu_usage()
        if cpu_usage > self.cpu_threshold:
            return True, f"High CPU usage: {cpu_usage:.1%}"
        
        return False, ""
    
    def get_resource_status(self) -> Dict[str, Any]:
        """
        Get current resource status.
        
        Returns:
            Dictionary with resource information
        """
        memory_usage = self.check_memory_usage()
        cpu_usage = self.check_cpu_usage()
        
        return {
            'memory_usage': memory_usage,
            'cpu_usage': cpu_usage,
            'memory_critical': memory_usage > self.memory_threshold,
            'cpu_critical': cpu_usage > self.cpu_threshold,
            'overall_status': 'critical' if (
                memory_usage > self.memory_threshold or cpu_usage > self.cpu_threshold
            ) else 'normal'
        }


# Global instances
degradation_manager = GracefulDegradationManager()
fallback_model_manager = FallbackModelManager()
resource_monitor = ResourceMonitor()


def with_graceful_degradation(component: str):
    """
    Decorator to add graceful degradation to a function.
    
    Args:
        component: Component name for degradation tracking
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Check if degradation is needed
                if degradation_manager.should_degrade(component):
                    fallback_method = degradation_manager.get_fallback_method(component)
                    if fallback_method:
                        # Use fallback method if available
                        # This would need to be implemented based on specific component needs
                        pass
                
                result = func(*args, **kwargs)
                
                # Record successful execution
                execution_time = time.time() - start_time
                degradation_manager.record_performance(component, execution_time)
                degradation_manager.update_component_health(component, 1.0)
                
                return result
                
            except Exception as e:
                # Record failure
                execution_time = time.time() - start_time
                degradation_manager.record_performance(component, execution_time)
                degradation_manager.update_component_health(component, 0.0)
                
                raise
        
        return wrapper
    return decorator