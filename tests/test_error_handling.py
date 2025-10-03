"""
Tests for the error handling system.
"""

import pytest
import time
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.core.error_handling import (
    ProcessingError, ModelLoadingError, DetectionError, RecognitionError,
    ErrorCategory, ErrorSeverity, ErrorContext, ErrorHandler,
    ModelLoadingErrorHandler, DetectionErrorHandler, RecognitionErrorHandler,
    RetryMechanism, ErrorLogger, ErrorHandlerManager, handle_errors
)


class TestProcessingError:
    """Test ProcessingError and its subclasses."""
    
    def test_processing_error_creation(self):
        """Test creating a ProcessingError."""
        context = ErrorContext(component="test", operation="test_op")
        error = ProcessingError(
            "Test error",
            ErrorCategory.DETECTION,
            ErrorSeverity.HIGH,
            context,
            recoverable=True
        )
        
        assert error.message == "Test error"
        assert error.category == ErrorCategory.DETECTION
        assert error.severity == ErrorSeverity.HIGH
        assert error.context == context
        assert error.recoverable is True
        assert error.error_id.startswith("detection_")
    
    def test_model_loading_error(self):
        """Test ModelLoadingError creation."""
        error = ModelLoadingError("Model not found", "/path/to/model")
        
        assert error.category == ErrorCategory.MODEL_LOADING
        assert error.severity == ErrorSeverity.HIGH
        assert error.model_path == "/path/to/model"
    
    def test_detection_error(self):
        """Test DetectionError creation."""
        error = DetectionError("Detection failed")
        
        assert error.category == ErrorCategory.DETECTION
        assert error.severity == ErrorSeverity.MEDIUM
    
    def test_recognition_error(self):
        """Test RecognitionError creation."""
        error = RecognitionError("Recognition failed")
        
        assert error.category == ErrorCategory.RECOGNITION
        assert error.severity == ErrorSeverity.MEDIUM


class TestErrorHandlers:
    """Test error handler implementations."""
    
    def test_model_loading_error_handler(self):
        """Test ModelLoadingErrorHandler."""
        fallback_models = {
            'detection': '/fallback/detection.pth',
            'recognition': '/fallback/recognition.pth'
        }
        handler = ModelLoadingErrorHandler(fallback_models)
        
        # Test can_handle
        model_error = ModelLoadingError("Model failed", "/path/to/detection.pth")
        other_error = DetectionError("Not a model error")
        
        assert handler.can_handle(model_error) is True
        assert handler.can_handle(other_error) is False
        
        # Test handle with existing fallback
        with patch('pathlib.Path.exists', return_value=True):
            result = handler.handle(model_error)
            assert result == '/fallback/detection.pth'
        
        # Test handle with non-existing fallback
        with patch('pathlib.Path.exists', return_value=False):
            result = handler.handle(model_error)
            assert result is None
    
    def test_detection_error_handler(self):
        """Test DetectionErrorHandler."""
        handler = DetectionErrorHandler()
        
        # Test can_handle
        detection_error = DetectionError("Detection failed")
        other_error = RecognitionError("Not a detection error")
        
        assert handler.can_handle(detection_error) is True
        assert handler.can_handle(other_error) is False
        
        # Test handle
        result = handler.handle(detection_error)
        assert result == []
    
    def test_recognition_error_handler(self):
        """Test RecognitionErrorHandler."""
        handler = RecognitionErrorHandler()
        
        # Test can_handle
        recognition_error = RecognitionError("Recognition failed")
        other_error = DetectionError("Not a recognition error")
        
        assert handler.can_handle(recognition_error) is True
        assert handler.can_handle(other_error) is False
        
        # Test handle
        result = handler.handle(recognition_error)
        assert result == "[RECOGNITION_FAILED]"


class TestRetryMechanism:
    """Test retry mechanism functionality."""
    
    def test_successful_execution(self):
        """Test retry mechanism with successful execution."""
        retry = RetryMechanism(max_attempts=3)
        
        @retry
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
    
    def test_retry_on_retryable_error(self):
        """Test retry mechanism with retryable errors."""
        retry = RetryMechanism(max_attempts=3, base_delay=0.01)
        call_count = 0
        
        @retry
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert call_count == 3
    
    def test_non_retryable_error(self):
        """Test retry mechanism with non-retryable errors."""
        retry = RetryMechanism(max_attempts=3, retryable_errors=[ConnectionError])
        
        @retry
        def failing_function():
            raise ValueError("Not retryable")
        
        with pytest.raises(ValueError):
            failing_function()
    
    def test_max_attempts_exceeded(self):
        """Test retry mechanism when max attempts are exceeded."""
        retry = RetryMechanism(max_attempts=2, base_delay=0.01)
        
        @retry
        def always_failing_function():
            raise ConnectionError("Always fails")
        
        with pytest.raises(ConnectionError):
            always_failing_function()
    
    def test_processing_error_retry(self):
        """Test retry mechanism with ProcessingError."""
        retry = RetryMechanism(max_attempts=3, base_delay=0.01)
        call_count = 0
        
        @retry
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ProcessingError(
                    "Recoverable error",
                    ErrorCategory.DETECTION,
                    recoverable=True
                )
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert call_count == 3
    
    def test_non_recoverable_processing_error(self):
        """Test retry mechanism with non-recoverable ProcessingError."""
        retry = RetryMechanism(max_attempts=3)
        
        @retry
        def failing_function():
            raise ProcessingError(
                "Non-recoverable error",
                ErrorCategory.CONFIGURATION,
                recoverable=False
            )
        
        with pytest.raises(ProcessingError):
            failing_function()


class TestErrorLogger:
    """Test error logging functionality."""
    
    def test_error_logging(self):
        """Test basic error logging."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            log_file = f.name
        
        try:
            logger = ErrorLogger(log_file)
            
            error = ProcessingError(
                "Test error",
                ErrorCategory.DETECTION,
                ErrorSeverity.HIGH
            )
            
            logger.log_error(error)
            
            # Check error was recorded
            assert len(logger.error_history) == 1
            assert logger.error_counts['detection_high'] == 1
            
            # Check log file was created
            assert Path(log_file).exists()
            
        finally:
            # Close any open handlers before deleting
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)
            Path(log_file).unlink(missing_ok=True)
    
    def test_error_statistics(self):
        """Test error statistics calculation."""
        logger = ErrorLogger()
        
        # Log multiple errors
        errors = [
            ProcessingError("Error 1", ErrorCategory.DETECTION, ErrorSeverity.HIGH),
            ProcessingError("Error 2", ErrorCategory.DETECTION, ErrorSeverity.LOW),
            ProcessingError("Error 3", ErrorCategory.RECOGNITION, ErrorSeverity.MEDIUM),
        ]
        
        for error in errors:
            logger.log_error(error)
        
        stats = logger.get_error_statistics()
        
        assert stats['total_errors'] == 3
        assert stats['category_stats']['detection']['count'] == 2
        assert stats['category_stats']['recognition']['count'] == 1
        assert stats['severity_stats']['high']['count'] == 1
        assert stats['severity_stats']['medium']['count'] == 1
        assert stats['severity_stats']['low']['count'] == 1
    
    def test_error_history_limit(self):
        """Test error history size limit."""
        logger = ErrorLogger()
        
        # Log more than 1000 errors
        for i in range(1100):
            error = ProcessingError(f"Error {i}", ErrorCategory.DETECTION)
            logger.log_error(error)
        
        # Should keep only last 1000
        assert len(logger.error_history) == 1000
        assert logger.error_history[-1]['message'] == "Error 1099"


class TestErrorHandlerManager:
    """Test error handler manager functionality."""
    
    def test_handler_registration(self):
        """Test registering error handlers."""
        manager = ErrorHandlerManager()
        handler = DetectionErrorHandler()
        
        manager.register_handler(handler)
        
        assert handler in manager.handlers
    
    def test_error_handling_routing(self):
        """Test error routing to appropriate handler."""
        manager = ErrorHandlerManager()
        detection_handler = DetectionErrorHandler()
        recognition_handler = RecognitionErrorHandler()
        
        manager.register_handler(detection_handler)
        manager.register_handler(recognition_handler)
        
        # Test detection error routing
        detection_error = DetectionError("Detection failed")
        result = manager.handle_error(detection_error)
        assert result == []
        
        # Test recognition error routing
        recognition_error = RecognitionError("Recognition failed")
        result = manager.handle_error(recognition_error)
        assert result == "[RECOGNITION_FAILED]"
    
    def test_fallback_handler(self):
        """Test fallback handler for unhandled errors."""
        manager = ErrorHandlerManager()
        
        # No handlers registered
        detection_error = DetectionError("Unhandled error")
        result = manager.handle_error(detection_error)
        assert result == []  # Fallback for detection errors
        
        recognition_error = RecognitionError("Unhandled error")
        result = manager.handle_error(recognition_error)
        assert result == ""  # Fallback for recognition errors
    
    def test_handler_failure(self):
        """Test handling when a handler fails."""
        manager = ErrorHandlerManager()
        
        # Create a failing handler
        failing_handler = Mock(spec=ErrorHandler)
        failing_handler.can_handle.return_value = True
        failing_handler.handle.side_effect = Exception("Handler failed")
        
        manager.register_handler(failing_handler)
        
        # Should fall back to default handler
        detection_error = DetectionError("Test error")
        result = manager.handle_error(detection_error)
        assert result == []  # Fallback result
    
    def test_statistics(self):
        """Test error statistics from manager."""
        manager = ErrorHandlerManager()
        
        error = ProcessingError("Test error", ErrorCategory.DETECTION)
        manager.handle_error(error)
        
        stats = manager.get_statistics()
        assert stats['total_errors'] == 1


class TestHandleErrorsDecorator:
    """Test handle_errors decorator functionality."""
    
    def test_successful_execution(self):
        """Test decorator with successful execution."""
        @handle_errors()
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
    
    def test_error_handling_with_fallback(self):
        """Test decorator with error and fallback result."""
        @handle_errors(fallback_result="fallback")
        def failing_function():
            raise ValueError("Function failed")
        
        result = failing_function()
        assert result == "fallback"
    
    def test_specific_error_types(self):
        """Test decorator with specific error types."""
        @handle_errors(error_types=[ValueError], fallback_result="handled")
        def function_with_specific_error():
            raise ValueError("Specific error")
        
        result = function_with_specific_error()
        assert result == "handled"
        
        # Test with non-matching error type
        @handle_errors(error_types=[ValueError], fallback_result="handled")
        def function_with_other_error():
            raise TypeError("Other error")
        
        with pytest.raises(TypeError):
            function_with_other_error()
    
    def test_no_logging(self):
        """Test decorator with logging disabled."""
        @handle_errors(log_errors=False, fallback_result="no_log")
        def failing_function():
            raise ValueError("Not logged")
        
        result = failing_function()
        assert result == "no_log"


class TestErrorContext:
    """Test ErrorContext functionality."""
    
    def test_error_context_creation(self):
        """Test creating ErrorContext."""
        context = ErrorContext(
            component="test_component",
            operation="test_operation",
            input_data={"key": "value"}
        )
        
        assert context.component == "test_component"
        assert context.operation == "test_operation"
        assert context.input_data == {"key": "value"}
        assert context.timestamp > 0
        assert context.stack_trace is not None
    
    def test_automatic_stack_trace(self):
        """Test automatic stack trace capture."""
        context = ErrorContext()
        
        assert context.stack_trace != ""
        assert isinstance(context.stack_trace, (str, list))


if __name__ == "__main__":
    pytest.main([__file__])