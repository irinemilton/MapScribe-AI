"""
Error handling for the REST API.
"""

from flask import jsonify
from typing import Dict, Any, Tuple
import logging
import traceback
from datetime import datetime


class APIErrorHandler:
    """Handles API errors and formats error responses."""
    
    def __init__(self):
        """Initialize the error handler."""
        self.logger = logging.getLogger(__name__)
    
    def handle_bad_request(self, error) -> Tuple[Dict[str, Any], int]:
        """
        Handle 400 Bad Request errors.
        
        Args:
            error: Flask error object
            
        Returns:
            Error response tuple
        """
        self.logger.warning(f"Bad request: {error}")
        
        response = {
            'error': 'Bad Request',
            'message': 'The request was invalid or malformed',
            'status_code': 400,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add specific error details if available
        if hasattr(error, 'description') and error.description:
            response['details'] = error.description
        
        return jsonify(response), 400
    
    def handle_not_found(self, error) -> Tuple[Dict[str, Any], int]:
        """
        Handle 404 Not Found errors.
        
        Args:
            error: Flask error object
            
        Returns:
            Error response tuple
        """
        self.logger.warning(f"Resource not found: {error}")
        
        response = {
            'error': 'Not Found',
            'message': 'The requested resource was not found',
            'status_code': 404,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 404
    
    def handle_file_too_large(self, error) -> Tuple[Dict[str, Any], int]:
        """
        Handle 413 Request Entity Too Large errors.
        
        Args:
            error: Flask error object
            
        Returns:
            Error response tuple
        """
        self.logger.warning(f"File too large: {error}")
        
        response = {
            'error': 'File Too Large',
            'message': 'The uploaded file exceeds the maximum allowed size',
            'status_code': 413,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 413
    
    def handle_internal_error(self, error) -> Tuple[Dict[str, Any], int]:
        """
        Handle 500 Internal Server Error.
        
        Args:
            error: Flask error object
            
        Returns:
            Error response tuple
        """
        self.logger.error(f"Internal server error: {error}")
        self.logger.error(traceback.format_exc())
        
        response = {
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred while processing the request',
            'status_code': 500,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 500
    
    def handle_validation_error(self, validation_errors: Dict[str, str]) -> Tuple[Dict[str, Any], int]:
        """
        Handle validation errors.
        
        Args:
            validation_errors: Dictionary of validation error messages
            
        Returns:
            Error response tuple
        """
        self.logger.warning(f"Validation errors: {validation_errors}")
        
        response = {
            'error': 'Validation Error',
            'message': 'The request contains invalid parameters',
            'validation_errors': validation_errors,
            'status_code': 422,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 422
    
    def handle_processing_error(self, error_message: str, details: Dict[str, Any] = None) -> Tuple[Dict[str, Any], int]:
        """
        Handle processing errors.
        
        Args:
            error_message: Error message
            details: Optional error details
            
        Returns:
            Error response tuple
        """
        self.logger.error(f"Processing error: {error_message}")
        
        response = {
            'error': 'Processing Error',
            'message': error_message,
            'status_code': 500,
            'timestamp': datetime.now().isoformat()
        }
        
        if details:
            response['details'] = details
        
        return jsonify(response), 500
    
    def handle_model_error(self, error_message: str) -> Tuple[Dict[str, Any], int]:
        """
        Handle model loading or inference errors.
        
        Args:
            error_message: Error message
            
        Returns:
            Error response tuple
        """
        self.logger.error(f"Model error: {error_message}")
        
        response = {
            'error': 'Model Error',
            'message': f'Model operation failed: {error_message}',
            'status_code': 503,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 503
    
    def handle_configuration_error(self, error_message: str) -> Tuple[Dict[str, Any], int]:
        """
        Handle configuration errors.
        
        Args:
            error_message: Error message
            
        Returns:
            Error response tuple
        """
        self.logger.error(f"Configuration error: {error_message}")
        
        response = {
            'error': 'Configuration Error',
            'message': f'Configuration operation failed: {error_message}',
            'status_code': 400,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 400
    
    def handle_rate_limit_error(self, retry_after: int = None) -> Tuple[Dict[str, Any], int]:
        """
        Handle rate limiting errors.
        
        Args:
            retry_after: Seconds to wait before retrying
            
        Returns:
            Error response tuple
        """
        self.logger.warning("Rate limit exceeded")
        
        response = {
            'error': 'Rate Limit Exceeded',
            'message': 'Too many requests. Please try again later.',
            'status_code': 429,
            'timestamp': datetime.now().isoformat()
        }
        
        if retry_after:
            response['retry_after'] = retry_after
        
        return jsonify(response), 429
    
    def handle_authentication_error(self, error_message: str = None) -> Tuple[Dict[str, Any], int]:
        """
        Handle authentication errors.
        
        Args:
            error_message: Optional custom error message
            
        Returns:
            Error response tuple
        """
        self.logger.warning("Authentication failed")
        
        response = {
            'error': 'Authentication Required',
            'message': error_message or 'Authentication is required to access this resource',
            'status_code': 401,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 401
    
    def handle_authorization_error(self, error_message: str = None) -> Tuple[Dict[str, Any], int]:
        """
        Handle authorization errors.
        
        Args:
            error_message: Optional custom error message
            
        Returns:
            Error response tuple
        """
        self.logger.warning("Authorization failed")
        
        response = {
            'error': 'Insufficient Permissions',
            'message': error_message or 'You do not have permission to access this resource',
            'status_code': 403,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 403
    
    def handle_timeout_error(self, operation: str = None) -> Tuple[Dict[str, Any], int]:
        """
        Handle timeout errors.
        
        Args:
            operation: Optional operation that timed out
            
        Returns:
            Error response tuple
        """
        self.logger.error(f"Operation timed out: {operation}")
        
        response = {
            'error': 'Request Timeout',
            'message': f'The operation timed out: {operation}' if operation else 'The request timed out',
            'status_code': 408,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 408
    
    def handle_service_unavailable(self, service: str = None) -> Tuple[Dict[str, Any], int]:
        """
        Handle service unavailable errors.
        
        Args:
            service: Optional service name that is unavailable
            
        Returns:
            Error response tuple
        """
        self.logger.error(f"Service unavailable: {service}")
        
        response = {
            'error': 'Service Unavailable',
            'message': f'Service is temporarily unavailable: {service}' if service else 'Service is temporarily unavailable',
            'status_code': 503,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 503