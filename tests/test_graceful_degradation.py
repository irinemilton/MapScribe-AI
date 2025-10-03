"""
Tests for the graceful degradation system.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.core.graceful_degradation import (
    DegradationLevel, DegradationStrategy, GracefulDegradationManager,
    FallbackModelManager, CircuitBreaker, ResourceMonitor,
    with_graceful_degradation
)
from src.core.error_handling import ProcessingError, ErrorCategory


class TestDegradationStrategy:
    """Test DegradationStrategy functionality."""
    
    def test_strategy_creation(self):
        """Test creating a degradation strategy."""
        strategy = DegradationStrategy(
            level=DegradationLevel.MODERATE,
            fallback_method='simple_detection',
            quality_threshold=0.5,
            performance_threshold=5.0,
            enabled=True
        )
        
        assert strategy.level == DegradationLevel.MODERATE
        assert strategy.fallback_method == 'simple_detection'
        assert strategy.quality_threshold == 0.5
        assert strategy.performance_threshold == 5.0
        assert strategy.enabled is True


class TestGracefulDegradationManager:
    """Test GracefulDegradationManager functionality."""
    
    def test_default_strategies(self):
        """Test default degradation strategies are set up."""
        manager = GracefulDegradationManager()
        
        assert 'detection' in manager.strategies
        assert 'recognition' in manager.strategies
        assert 'preprocessing' in manager.strategies
        assert 'model_loading' in manager.strategies
    
    def test_register_strategy(self):
        """Test registering a custom strategy."""
        manager = GracefulDegradationManager()
        
        custom_strategy = DegradationStrategy(
            level=DegradationLevel.SEVERE,
            fallback_method='custom_fallback',
            quality_threshold=0.3,
            performance_threshold=10.0
        )
        
        manager.register_strategy('custom_component', custom_strategy)
        
        assert 'custom_component' in manager.strategies
        assert manager.strategies['custom_component'] == custom_strategy
    
    def test_update_component_health(self):
        """Test updating component health scores."""
        manager = GracefulDegradationManager()
        
        # Test valid health score
        manager.update_component_health('detection', 0.8)
        assert manager.component_health['detection'] == 0.8
        
        # Test invalid health score
        with pytest.raises(ValueError):
            manager.update_component_health('detection', 1.5)
        
        with pytest.raises(ValueError):
            manager.update_component_health('detection', -0.1)
    
    def test_health_based_degradation_trigger(self):
        """Test degradation triggering based on health scores."""
        manager = GracefulDegradationManager()
        
        # Set low health score to trigger degradation
        manager.update_component_health('detection', 0.3)
        
        strategy = manager.strategies['detection']
        assert strategy.level != DegradationLevel.NONE
    
    def test_record_performance(self):
        """Test recording performance metrics."""
        manager = GracefulDegradationManager()
        
        # Record performance
        manager.record_performance('detection', 2.5)
        
        assert 'detection' in manager.performance_history
        assert manager.performance_history['detection'] == [2.5]
    
    def test_performance_based_degradation_trigger(self):
        """Test degradation triggering based on performance."""
        manager = GracefulDegradationManager()
        
        # Record slow performance to trigger degradation
        for _ in range(10):
            manager.record_performance('detection', 10.0)  # Exceeds threshold
        
        strategy = manager.strategies['detection']
        # Should have triggered some level of degradation
        assert strategy.level != DegradationLevel.NONE
    
    def test_performance_history_limit(self):
        """Test performance history size limit."""
        manager = GracefulDegradationManager()
        
        # Record more than 100 measurements
        for i in range(150):
            manager.record_performance('detection', i)
        
        # Should keep only last 100
        assert len(manager.performance_history['detection']) == 100
        assert manager.performance_history['detection'][0] == 50  # First kept measurement
    
    def test_should_degrade(self):
        """Test checking if component should degrade."""
        manager = GracefulDegradationManager()
        
        # Initially should not degrade
        assert manager.should_degrade('detection') is False
        
        # Trigger degradation
        manager.update_component_health('detection', 0.3)
        
        # Now should degrade
        assert manager.should_degrade('detection') is True
        
        # Test unknown component
        assert manager.should_degrade('unknown') is False
    
    def test_get_degradation_level(self):
        """Test getting degradation level."""
        manager = GracefulDegradationManager()
        
        # Default level
        assert manager.get_degradation_level('detection') == DegradationLevel.MODERATE
        
        # Unknown component
        assert manager.get_degradation_level('unknown') == DegradationLevel.NONE
    
    def test_get_fallback_method(self):
        """Test getting fallback method."""
        manager = GracefulDegradationManager()
        
        # Known component
        assert manager.get_fallback_method('detection') == 'simple_detection'
        
        # Unknown component
        assert manager.get_fallback_method('unknown') is None
    
    def test_get_system_health(self):
        """Test getting overall system health."""
        manager = GracefulDegradationManager()
        
        # Set various health scores
        manager.update_component_health('detection', 0.8)
        manager.update_component_health('recognition', 0.6)
        manager.update_component_health('preprocessing', 0.4)
        
        health = manager.get_system_health()
        
        assert health['overall_health'] == 0.6  # Average
        assert health['status'] == 'degraded'
        assert health['total_components'] == 3
        assert 'components' in health
        assert 'degradation_levels' in health


class TestFallbackModelManager:
    """Test FallbackModelManager functionality."""
    
    def test_register_fallback_models(self):
        """Test registering fallback models."""
        manager = FallbackModelManager()
        
        models = ['/path/to/model1.pth', '/path/to/model2.pth']
        manager.register_fallback_models('detection', models)
        
        assert 'detection' in manager.fallback_models
        assert manager.fallback_models['detection'] == models
    
    def test_get_fallback_model(self):
        """Test getting fallback models."""
        manager = FallbackModelManager()
        
        models = ['/path/to/model1.pth', '/path/to/model2.pth', '/path/to/model3.pth']
        manager.register_fallback_models('detection', models)
        
        # Get first fallback
        fallback = manager.get_fallback_model('detection')
        assert fallback == '/path/to/model1.pth'
        
        # Get fallback after specific failure
        fallback = manager.get_fallback_model('detection', '/path/to/model1.pth')
        assert fallback == '/path/to/model2.pth'
        
        # Get fallback after last model fails
        fallback = manager.get_fallback_model('detection', '/path/to/model3.pth')
        assert fallback is None
        
        # Unknown model type
        fallback = manager.get_fallback_model('unknown')
        assert fallback is None
    
    def test_record_model_performance(self):
        """Test recording model performance."""
        manager = FallbackModelManager()
        
        manager.record_model_performance('/path/to/model.pth', 0.85)
        
        assert '/path/to/model.pth' in manager.model_performance
        assert manager.model_performance['/path/to/model.pth'] == 0.85
    
    def test_get_best_performing_model(self):
        """Test getting best performing model."""
        manager = FallbackModelManager()
        
        models = ['/path/to/model1.pth', '/path/to/model2.pth', '/path/to/model3.pth']
        manager.register_fallback_models('detection', models)
        
        # Record performance for some models
        manager.record_model_performance('/path/to/model1.pth', 0.7)
        manager.record_model_performance('/path/to/model2.pth', 0.9)
        manager.record_model_performance('/path/to/model3.pth', 0.6)
        
        best = manager.get_best_performing_model('detection')
        assert best == '/path/to/model2.pth'
        
        # Test with no performance data
        manager_empty = FallbackModelManager()
        manager_empty.register_fallback_models('detection', models)
        best = manager_empty.get_best_performing_model('detection')
        assert best == '/path/to/model1.pth'  # First model as default


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""
    
    def test_successful_execution(self):
        """Test circuit breaker with successful execution."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        
        @breaker
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
        assert breaker.state == 'closed'
        assert breaker.failure_count == 0
    
    def test_failure_threshold(self):
        """Test circuit breaker opening after failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        
        @breaker
        def failing_function():
            raise ValueError("Function failed")
        
        # Fail 3 times to reach threshold
        for _ in range(3):
            with pytest.raises(ValueError):
                failing_function()
        
        assert breaker.state == 'open'
        assert breaker.failure_count == 3
        
        # Next call should raise ProcessingError (circuit open)
        with pytest.raises(ProcessingError):
            failing_function()
    
    def test_recovery_attempt(self):
        """Test circuit breaker recovery attempt."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        
        call_count = 0
        
        @breaker
        def sometimes_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("Initial failures")
            return "recovered"
        
        # Fail twice to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                sometimes_failing_function()
        
        assert breaker.state == 'open'
        
        # Wait for recovery timeout
        time.sleep(0.02)
        
        # Next call should attempt recovery
        result = sometimes_failing_function()
        assert result == "recovered"
        assert breaker.state == 'closed'
        assert breaker.failure_count == 0
    
    def test_half_open_state(self):
        """Test circuit breaker half-open state."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        
        @breaker
        def failing_function():
            raise ValueError("Still failing")
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                failing_function()
        
        assert breaker.state == 'open'
        
        # Wait for recovery timeout
        time.sleep(0.02)
        
        # Next call should go to half-open, then fail and go back to open
        with pytest.raises(ValueError):
            failing_function()
        
        assert breaker.state == 'open'
    
    def test_non_expected_exception(self):
        """Test circuit breaker with non-expected exceptions."""
        breaker = CircuitBreaker(failure_threshold=3, expected_exception=ValueError)
        
        @breaker
        def function_with_other_error():
            raise TypeError("Not the expected error type")
        
        # Should not trigger circuit breaker
        with pytest.raises(TypeError):
            function_with_other_error()
        
        assert breaker.state == 'closed'
        assert breaker.failure_count == 0


class TestResourceMonitor:
    """Test ResourceMonitor functionality."""
    
    @patch('src.core.graceful_degradation.psutil')
    def test_check_memory_usage(self, mock_psutil):
        """Test memory usage checking."""
        monitor = ResourceMonitor()
        
        # Mock memory usage
        mock_memory = Mock()
        mock_memory.percent = 75.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        usage = monitor.check_memory_usage()
        assert usage == 0.75
    
    @patch('src.core.graceful_degradation.psutil')
    def test_check_memory_usage_no_psutil(self, mock_psutil):
        """Test memory usage checking without psutil."""
        mock_psutil.side_effect = ImportError("psutil not available")
        
        monitor = ResourceMonitor()
        usage = monitor.check_memory_usage()
        assert usage == 0.0
    
    @patch('src.core.graceful_degradation.psutil')
    def test_check_cpu_usage(self, mock_psutil):
        """Test CPU usage checking."""
        monitor = ResourceMonitor()
        
        # Mock CPU usage
        mock_psutil.cpu_percent.return_value = 80.0
        
        usage = monitor.check_cpu_usage()
        assert usage == 0.80
    
    def test_should_trigger_degradation(self):
        """Test degradation triggering based on resource usage."""
        monitor = ResourceMonitor()
        monitor.memory_threshold = 0.8
        monitor.cpu_threshold = 0.9
        
        # Mock high memory usage
        with patch.object(monitor, 'check_memory_usage', return_value=0.9):
            with patch.object(monitor, 'check_cpu_usage', return_value=0.5):
                should_degrade, reason = monitor.should_trigger_degradation()
                assert should_degrade is True
                assert "memory usage" in reason
        
        # Mock high CPU usage
        with patch.object(monitor, 'check_memory_usage', return_value=0.5):
            with patch.object(monitor, 'check_cpu_usage', return_value=0.95):
                should_degrade, reason = monitor.should_trigger_degradation()
                assert should_degrade is True
                assert "CPU usage" in reason
        
        # Mock normal usage
        with patch.object(monitor, 'check_memory_usage', return_value=0.5):
            with patch.object(monitor, 'check_cpu_usage', return_value=0.5):
                should_degrade, reason = monitor.should_trigger_degradation()
                assert should_degrade is False
                assert reason == ""
    
    def test_get_resource_status(self):
        """Test getting resource status."""
        monitor = ResourceMonitor()
        
        with patch.object(monitor, 'check_memory_usage', return_value=0.7):
            with patch.object(monitor, 'check_cpu_usage', return_value=0.6):
                status = monitor.get_resource_status()
                
                assert status['memory_usage'] == 0.7
                assert status['cpu_usage'] == 0.6
                assert status['memory_critical'] is False
                assert status['cpu_critical'] is False
                assert status['overall_status'] == 'normal'


class TestWithGracefulDegradationDecorator:
    """Test with_graceful_degradation decorator."""
    
    def test_successful_execution(self):
        """Test decorator with successful execution."""
        @with_graceful_degradation('test_component')
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
    
    def test_execution_with_failure(self):
        """Test decorator with execution failure."""
        @with_graceful_degradation('test_component')
        def failing_function():
            raise ValueError("Function failed")
        
        with pytest.raises(ValueError):
            failing_function()


if __name__ == "__main__":
    pytest.main([__file__])