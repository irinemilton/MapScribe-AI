"""
Comprehensive error handling system for historical map text processing.

This module provides error classes, handlers, retry mechanisms, and logging
utilities for robust system operation.
"""

import logging
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import functools


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better classification and handling."""
    MODEL_LOADING = "model_loading"
    DETECTION = "detection"
    RECOGNITION = "recognition"
    PREPROCESSING = "preprocessing"
    VALIDATION = "validation"
    IO = "io"
    MEMORY = "memory"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Context information for error tracking and debugging."""
    timestamp: float = field(default_factory=time.time)
    component: str = ""
    operation: str = ""
    input_data: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    stack_trace: str = ""
    
    def __post_init__(self):
        """Capture stack trace if not provided."""
        if not self.stack_trace:
            self.stack_trace = traceback.format_stack()


class ProcessingError(Exception):
    """Base exception class for all processing errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        recoverable: bool = True
    ):
        """
        Initialize processing error.
        
        Args:
            message: Error description
            category: Error category
            severity: Error severity level
            context: Additional error context
            recoverable: Whether error is recoverable
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.recoverable = recoverable
        self.error_id = f"{category.value}_{int(time.time())}"


class ModelLoadingError(ProcessingError):
    """Error during model loading operations."""
    
    def __init__(self, message: str, model_path: str, **kwargs):
        super().__init__(
            message,
            ErrorCategory.MODEL_LOADING,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.model_path = model_path


class DetectionError(ProcessingError):
    """Error during text detection operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            ErrorCategory.DETECTION,
            **kwargs
        )


class RecognitionError(ProcessingError):
    """Error during text recognition operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            ErrorCategory.RECOGNITION,
            **kwargs
        )


class PreprocessingError(ProcessingError):
    """Error during image preprocessing operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            ErrorCategory.PREPROCESSING,
            **kwargs
        )


class ValidationError(ProcessingError):
    """Error during data validation operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )


class MemoryError(ProcessingError):
    """Error related to memory management."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            ErrorCategory.MEMORY,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ConfigurationError(ProcessingError):
    """Error in system configuration."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs
        )


class ErrorHandler(ABC):
    """Abstract base class for error handlers."""
    
    @abstractmethod
    def can_handle(self, error: ProcessingError) -> bool:
        """
        Check if this handler can handle the given error.
        
        Args:
            error: Error to check
            
        Returns:
            True if handler can process this error
        """
        pass
    
    @abstractmethod
    def handle(self, error: ProcessingError) -> Any:
        """
        Handle the given error.
        
        Args:
            error: Error to handle
            
        Returns:
            Recovery result or None
        """
        pass


class ModelLoadingErrorHandler(ErrorHandler):
    """Handler for model loading errors with fallback mechanisms."""
    
    def __init__(self, fallback_models: Optional[Dict[str, str]] = None):
        """
        Initialize model loading error handler.
        
        Args:
            fallback_models: Dictionary of fallback model paths by type
        """
        self.fallback_models = fallback_models or {}
        self.logger = logging.getLogger(__name__)
    
    def can_handle(self, error: ProcessingError) -> bool:
        """Check if this is a model loading error."""
        return isinstance(error, ModelLoadingError)
    
    def handle(self, error: ModelLoadingError) -> Optional[str]:
        """
        Handle model loading error by trying fallback models.
        
        Args:
            error: Model loading error
            
        Returns:
            Path to fallback model if available
        """
        self.logger.warning(f"Model loading failed: {error.message}")
        
        # Try to determine model type from path
        model_type = self._determine_model_type(error.model_path)
        
        if model_type and model_type in self.fallback_models:
            fallback_path = self.fallback_models[model_type]
            if Path(fallback_path).exists():
                self.logger.info(f"Using fallback model: {fallback_path}")
                return fallback_path
        
        # Log error for monitoring
        self.logger.error(f"No fallback available for model: {error.model_path}")
        return None
    
    def _determine_model_type(self, model_path: str) -> Optional[str]:
        """Determine model type from path."""
        path_lower = model_path.lower()
        if 'detection' in path_lower or 'detect' in path_lower:
            return 'detection'
        elif 'recognition' in path_lower or 'recognize' in path_lower:
            return 'recognition'
        elif 'yolo' in path_lower:
            return 'detection'
        elif 'crnn' in path_lower or 'ocr' in path_lower:
            return 'recognition'
        return None


class DetectionErrorHandler(ErrorHandler):
    """Handler for detection errors with graceful degradation."""
    
    def __init__(self):
        """Initialize detection error handler."""
        self.logger = logging.getLogger(__name__)
    
    def can_handle(self, error: ProcessingError) -> bool:
        """Check if this is a detection error."""
        return isinstance(error, DetectionError)
    
    def handle(self, error: DetectionError) -> List:
        """
        Handle detection error by returning empty results.
        
        Args:
            error: Detection error
            
        Returns:
            Empty detection list
        """
        self.logger.warning(f"Detection failed: {error.message}")
        
        # Return empty detection list to allow pipeline to continue
        return []


class RecognitionErrorHandler(ErrorHandler):
    """Handler for recognition errors with fallback strategies."""
    
    def __init__(self):
        """Initialize recognition error handler."""
        self.logger = logging.getLogger(__name__)
    
    def can_handle(self, error: ProcessingError) -> bool:
        """Check if this is a recognition error."""
        return isinstance(error, RecognitionError)
    
    def handle(self, error: RecognitionError) -> str:
        """
        Handle recognition error by returning placeholder text.
        
        Args:
            error: Recognition error
            
        Returns:
            Placeholder text string
        """
        self.logger.warning(f"Recognition failed: {error.message}")
        
        # Return placeholder text to maintain pipeline flow
        return "[RECOGNITION_FAILED]"


class RetryMechanism:
    """Implements retry logic for transient failures."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        retryable_errors: Optional[List[Type[Exception]]] = None
    ):
        """
        Initialize retry mechanism.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries
            backoff_factor: Exponential backoff factor
            retryable_errors: List of exception types that should trigger retry
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.retryable_errors = retryable_errors or [
            ConnectionError,
            TimeoutError,
            MemoryError,
            IOError
        ]
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to add retry logic to a function.
        
        Args:
            func: Function to wrap with retry logic
            
        Returns:
            Wrapped function with retry capability
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if error is retryable
                    if not self._is_retryable(e):
                        self.logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise
                    
                    if attempt < self.max_attempts - 1:
                        delay = min(
                            self.base_delay * (self.backoff_factor ** attempt),
                            self.max_delay
                        )
                        self.logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        time.sleep(delay)
                    else:
                        self.logger.error(
                            f"All {self.max_attempts} attempts failed for {func.__name__}"
                        )
            
            # All attempts failed, raise the last exception
            raise last_exception
        
        return wrapper
    
    def _is_retryable(self, error: Exception) -> bool:
        """
        Check if an error should trigger a retry.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is retryable
        """
        # Check if it's a ProcessingError with recoverable flag
        if isinstance(error, ProcessingError):
            return error.recoverable
        
        # Check if it's in the list of retryable error types
        return any(isinstance(error, error_type) for error_type in self.retryable_errors)


class ErrorLogger:
    """Centralized error logging and monitoring system."""
    
    def __init__(self, log_file: Optional[str] = None, log_level: int = logging.INFO):
        """
        Initialize error logger.
        
        Args:
            log_file: Path to log file (optional)
            log_level: Logging level
        """
        self.logger = logging.getLogger("error_handler")
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Error statistics
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []
    
    def log_error(self, error: ProcessingError) -> None:
        """
        Log error with full context information.
        
        Args:
            error: Processing error to log
        """
        # Update error counts
        error_key = f"{error.category.value}_{error.severity.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Create error record
        error_record = {
            'error_id': error.error_id,
            'timestamp': error.context.timestamp,
            'category': error.category.value,
            'severity': error.severity.value,
            'message': error.message,
            'component': error.context.component,
            'operation': error.context.operation,
            'recoverable': error.recoverable
        }
        
        # Add to history
        self.error_history.append(error_record)
        
        # Keep only last 1000 errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # Log based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"[{error.error_id}] {error.message}")
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"[{error.error_id}] {error.message}")
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"[{error.error_id}] {error.message}")
        else:
            self.logger.info(f"[{error.error_id}] {error.message}")
        
        # Log context if available
        if error.context.input_data:
            self.logger.debug(f"Input data: {error.context.input_data}")
        
        if error.context.system_state:
            self.logger.debug(f"System state: {error.context.system_state}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics and metrics.
        
        Returns:
            Dictionary with error statistics
        """
        total_errors = sum(self.error_counts.values())
        
        # Calculate error rates by category
        category_stats = {}
        for category in ErrorCategory:
            category_count = sum(
                count for key, count in self.error_counts.items()
                if key.startswith(category.value)
            )
            category_stats[category.value] = {
                'count': category_count,
                'rate': category_count / total_errors if total_errors > 0 else 0.0
            }
        
        # Calculate error rates by severity
        severity_stats = {}
        for severity in ErrorSeverity:
            severity_count = sum(
                count for key, count in self.error_counts.items()
                if key.endswith(severity.value)
            )
            severity_stats[severity.value] = {
                'count': severity_count,
                'rate': severity_count / total_errors if total_errors > 0 else 0.0
            }
        
        return {
            'total_errors': total_errors,
            'error_counts': self.error_counts.copy(),
            'category_stats': category_stats,
            'severity_stats': severity_stats,
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }
    
    def clear_statistics(self) -> None:
        """Clear error statistics and history."""
        self.error_counts.clear()
        self.error_history.clear()


class ErrorHandlerManager:
    """Manages multiple error handlers and routes errors appropriately."""
    
    def __init__(self):
        """Initialize error handler manager."""
        self.handlers: List[ErrorHandler] = []
        self.logger = ErrorLogger()
        self.fallback_enabled = True
    
    def register_handler(self, handler: ErrorHandler) -> None:
        """
        Register an error handler.
        
        Args:
            handler: Error handler to register
        """
        self.handlers.append(handler)
    
    def handle_error(self, error: ProcessingError) -> Any:
        """
        Handle error using appropriate handler.
        
        Args:
            error: Error to handle
            
        Returns:
            Recovery result from handler
        """
        # Log the error
        self.logger.log_error(error)
        
        # Find appropriate handler
        for handler in self.handlers:
            if handler.can_handle(error):
                try:
                    result = handler.handle(error)
                    self.logger.logger.info(f"Error handled successfully by {handler.__class__.__name__}")
                    return result
                except Exception as handler_error:
                    self.logger.logger.error(f"Handler {handler.__class__.__name__} failed: {handler_error}")
        
        # No handler found or all handlers failed
        if self.fallback_enabled:
            return self._fallback_handler(error)
        else:
            raise error
    
    def _fallback_handler(self, error: ProcessingError) -> Any:
        """
        Fallback handler for unhandled errors.
        
        Args:
            error: Unhandled error
            
        Returns:
            Default fallback result
        """
        self.logger.logger.warning(f"Using fallback handler for {error.category.value} error")
        
        # Return appropriate default values based on error category
        if error.category == ErrorCategory.DETECTION:
            return []  # Empty detection list
        elif error.category == ErrorCategory.RECOGNITION:
            return ""  # Empty text
        elif error.category == ErrorCategory.PREPROCESSING:
            return None  # No processed image
        else:
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return self.logger.get_error_statistics()
    
    def clear_statistics(self) -> None:
        """Clear error statistics."""
        self.logger.clear_statistics()


# Global error handler manager instance
error_manager = ErrorHandlerManager()

# Register default handlers
error_manager.register_handler(ModelLoadingErrorHandler())
error_manager.register_handler(DetectionErrorHandler())
error_manager.register_handler(RecognitionErrorHandler())


def handle_errors(
    error_types: Optional[List[Type[Exception]]] = None,
    fallback_result: Any = None,
    log_errors: bool = True
) -> Callable:
    """
    Decorator for automatic error handling.
    
    Args:
        error_types: List of exception types to handle
        fallback_result: Result to return on error
        log_errors: Whether to log errors
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_types and not any(isinstance(e, et) for et in error_types):
                    raise
                
                if log_errors:
                    context = ErrorContext(
                        component=func.__module__,
                        operation=func.__name__,
                        input_data={'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
                    )
                    
                    if isinstance(e, ProcessingError):
                        error_manager.handle_error(e)
                    else:
                        # Convert to ProcessingError
                        processing_error = ProcessingError(
                            message=str(e),
                            category=ErrorCategory.SYSTEM,
                            context=context
                        )
                        error_manager.handle_error(processing_error)
                
                return fallback_result
        
        return wrapper
    return decorator