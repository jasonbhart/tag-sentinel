"""Error handling and resilience for dataLayer integrity system.

This module provides comprehensive error handling, graceful degradation,
circuit breaker patterns, and resilience mechanisms for all dataLayer
operations including capture, validation, and processing.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Type
from functools import wraps
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import traceback

logger = logging.getLogger(__name__)








class ErrorSeverity(Enum):
    """Error severity levels for dataLayer operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComponentType(Enum):
    """Types of dataLayer components for error tracking."""
    CAPTURE = "capture"
    VALIDATION = "validation"
    REDACTION = "redaction"
    AGGREGATION = "aggregation"
    INTEGRATION = "integration"
    CONFIGURATION = "configuration"


@dataclass
class ErrorContext:
    """Context information for an error occurrence."""
    component: ComponentType
    operation: str
    page_url: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary."""
        return {
            'component': self.component.value,
            'operation': self.operation,
            'page_url': self.page_url,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }


class DataLayerError:
    """Comprehensive error information for dataLayer operations."""

    def __init__(
        self,
        message_or_error_type: str = None,
        context_or_message: Union[Dict[str, Any], str, ErrorContext] = None,
        severity: Optional[ErrorSeverity] = None,
        context: Optional[ErrorContext] = None,
        exception: Optional[Exception] = None,
        stack_trace: Optional[str] = None,
        recovery_action: Optional[str] = None,
        impact_assessment: Optional[str] = None,
        *,
        error_type: Optional[str] = None,
        message: Optional[str] = None
    ):
        """Initialize DataLayerError with multiple constructor patterns.

        Supports both:
        1. DataLayerError("Test error", {"context": "test"})  # backwards compatible
        2. DataLayerError(error_type="error", message="msg", ...)  # full constructor
        """
        # Handle keyword-only arguments first
        if error_type is not None and message is not None:
            # Keyword constructor: DataLayerError(error_type="error", message="msg", ...)
            self.error_type = error_type
            self.message = message
            self.severity = severity or ErrorSeverity.MEDIUM
            self.context = context or ErrorContext(
                component=ComponentType.CAPTURE,
                operation="generic_operation"
            )
        elif message_or_error_type is None and error_type is not None:
            # Partial keyword constructor
            self.error_type = error_type
            self.message = message or "Unknown error"
            self.severity = severity or ErrorSeverity.MEDIUM
            self.context = context or ErrorContext(
                component=ComponentType.CAPTURE,
                operation="generic_operation"
            )
        # Determine which positional constructor pattern is being used
        elif isinstance(context_or_message, dict):
            # Backwards compatible: DataLayerError(message, context_dict)
            self.message = message_or_error_type
            self.error_type = "generic_error"
            self.severity = severity or ErrorSeverity.MEDIUM

            # For backward compatibility, also store the dict directly as context
            # Tests expect error.context to be the original dict
            self.context = context_or_message

            # Also create ErrorContext for internal use
            self._error_context = ErrorContext(
                component=ComponentType.CAPTURE,
                operation="generic_operation"
            )
            self._error_context.metadata = context_or_message

        elif isinstance(context_or_message, str):
            # Full constructor: DataLayerError(error_type, message, ...)
            self.error_type = message_or_error_type
            self.message = context_or_message
            self.severity = severity or ErrorSeverity.MEDIUM
            self.context = context or ErrorContext(
                component=ComponentType.CAPTURE,
                operation="generic_operation"
            )
        elif context_or_message is None and severity is None and context is None:
            # Simple message constructor: DataLayerError("message")
            self.message = message_or_error_type
            self.error_type = "generic_error"
            self.severity = ErrorSeverity.MEDIUM
            self.context = ErrorContext(
                component=ComponentType.CAPTURE,
                operation="generic_operation"
            )
        else:
            # Legacy dataclass-style: DataLayerError(error_type, message, severity, context, ...)
            self.error_type = message_or_error_type
            self.message = context_or_message if isinstance(context_or_message, str) else "Unknown error"
            self.severity = severity or ErrorSeverity.MEDIUM
            self.context = context or ErrorContext(
                component=ComponentType.CAPTURE,
                operation="generic_operation"
            )

        self.exception = exception
        self.stack_trace = stack_trace
        self.recovery_action = recovery_action
        self.impact_assessment = impact_assessment

        # Generate stack trace if exception provided
        if self.exception and not self.stack_trace:
            self.stack_trace = ''.join(traceback.format_exception(
                type(self.exception), self.exception, self.exception.__traceback__
            ))

    def __str__(self) -> str:
        """String representation of the error."""
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            'error_type': self.error_type,
            'message': self.message,
            'severity': self.severity.value,
            'context': self._error_context.to_dict() if hasattr(self, '_error_context') and self._error_context else (self.context.to_dict() if hasattr(self.context, 'to_dict') else self.context),
            'exception_type': type(self.exception).__name__ if self.exception else None,
            'stack_trace': self.stack_trace,
            'recovery_action': self.recovery_action,
            'impact_assessment': self.impact_assessment
        }

    @classmethod
    def create_simple(
        cls,
        message: str,
        error_type: str = "generic_error",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        component: ComponentType = ComponentType.CAPTURE
    ) -> 'DataLayerError':
        """Create a simple DataLayerError with minimal required information.

        Args:
            message: Error message
            error_type: Type of error (defaults to "generic_error")
            severity: Error severity (defaults to MEDIUM)
            component: Component where error occurred (defaults to CAPTURE)

        Returns:
            DataLayerError instance
        """
        context = ErrorContext(
            component=component,
            operation="test_operation"
        )

        return cls(
            error_type,
            message,
            severity,
            context
        )


class CaptureError(Exception, DataLayerError):
    """Error during capture operations."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        # Initialize Exception with message
        Exception.__init__(self, message)
        # Initialize DataLayerError with backward-compatible constructor
        DataLayerError.__init__(self, message, context or {})
        # Override error_type and create proper ErrorContext for internal use
        self.error_type = "CaptureError"
        self._error_context = ErrorContext(component=ComponentType.CAPTURE, operation="capture")
        if context:
            self._error_context.metadata.update(context)


class ValidationError(Exception, DataLayerError):
    """Error during validation operations."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        # Initialize Exception with message
        Exception.__init__(self, message)
        # Initialize DataLayerError with backward-compatible constructor
        DataLayerError.__init__(self, message, context or {})
        # Override error_type and create proper ErrorContext for internal use
        self.error_type = "ValidationError"
        self._error_context = ErrorContext(component=ComponentType.VALIDATION, operation="validation")
        if context:
            self._error_context.metadata.update(context)


class RedactionError(Exception, DataLayerError):
    """Error during redaction operations."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        # Initialize Exception with message
        Exception.__init__(self, message)
        # Initialize DataLayerError with backward-compatible constructor
        DataLayerError.__init__(self, message, context or {})
        # Override error_type and create proper ErrorContext for internal use
        self.error_type = "RedactionError"
        self._error_context = ErrorContext(component=ComponentType.REDACTION, operation="redaction")
        if context:
            self._error_context.metadata.update(context)


class DataLayerErrorHandler:
    """Centralized error handling for dataLayer operations."""
    
    def __init__(self, max_error_history: int = 1000):
        """Initialize error handler.
        
        Args:
            max_error_history: Maximum number of errors to keep in history
        """
        self.max_error_history = max_error_history
        self.error_history: List[DataLayerError] = []
        self.error_callbacks: Dict[ComponentType, List[Callable]] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        
        # Error statistics
        self.error_counts: Dict[str, int] = {}
        self.component_error_rates: Dict[ComponentType, float] = {}

        # Track total operations for error rate calculation
        self._total_operations = 0

        # Circuit breakers by component
        self._circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        
        # Initialize recovery strategies
        self._register_default_recovery_strategies()
    
    def register_error_callback(
        self,
        component: ComponentType,
        callback: Callable[[DataLayerError], None]
    ) -> None:
        """Register callback for specific component errors.
        
        Args:
            component: Component type to listen for
            callback: Callback function to execute on error
        """
        if component not in self.error_callbacks:
            self.error_callbacks[component] = []
        self.error_callbacks[component].append(callback)
    
    def register_recovery_strategy(
        self,
        error_type: str,
        strategy: Callable[[DataLayerError], Any]
    ) -> None:
        """Register recovery strategy for specific error types.
        
        Args:
            error_type: Type of error to handle
            strategy: Recovery strategy function
        """
        self.recovery_strategies[error_type] = strategy
    
    def handle_error(
        self,
        error,  # Can be DataLayerError or regular Exception
        component: Optional['ComponentType'] = None,
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True
    ) -> Optional[Any]:
        """Handle a dataLayer error with logging and potential recovery.

        Args:
            error: DataLayer error to handle (DataLayerError or Exception)
            component: Component type where error occurred (optional)
            context: Additional context for the error (optional)
            attempt_recovery: Whether to attempt automatic recovery

        Returns:
            Recovery result if successful, None otherwise
        """
        # Convert regular Exception to DataLayerError if needed
        if not isinstance(error, DataLayerError):
            error = DataLayerError.create_simple(
                message=str(error),
                error_type=type(error).__name__,
                component=component or ComponentType.CAPTURE
            )

        # Update error context if provided
        if component is not None and hasattr(error, 'context'):
            if hasattr(error.context, 'component'):
                error.context.component = component
        if context is not None and hasattr(error, 'context'):
            if hasattr(error.context, 'metadata'):
                error.context.metadata.update(context)

        # Log error based on severity
        self._log_error(error)

        # Add to error history
        self._record_error(error)

        # Increment total operations counter
        self._total_operations += 1

        # Execute callbacks
        self._execute_error_callbacks(error)

        # Attempt recovery if enabled
        recovery_result = None
        if attempt_recovery:
            recovery_result = self._attempt_recovery(error)

        return recovery_result
    
    def _log_error(self, error: DataLayerError) -> None:
        """Log error with appropriate level based on severity."""
        log_data = error.to_dict()
        # Remove 'message' key to avoid conflict with LogRecord.message
        log_data.pop('message', None)

        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL DataLayer Error: {error.message}", extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH DataLayer Error: {error.message}", extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM DataLayer Error: {error.message}", extra=log_data)
        else:
            logger.info(f"LOW DataLayer Error: {error.message}", extra=log_data)
    
    def _record_error(self, error: DataLayerError) -> None:
        """Record error in history and update statistics."""
        # Add to history with size limit
        self.error_history.append(error)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
        
        # Update error counts
        # Handle both ErrorContext objects and backward-compatible dict contexts
        if hasattr(error, '_error_context') and error._error_context:
            component = error._error_context.component
        elif hasattr(error.context, 'component'):
            component = error.context.component
        else:
            # Fallback for dict contexts
            component = ComponentType.CAPTURE

        error_key = f"{component.value}:{error.error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Update component error rates (simplified) - component already determined above
        if component not in self.component_error_rates:
            self.component_error_rates[component] = 0.0
        self.component_error_rates[component] += 1.0
    
    def _execute_error_callbacks(self, error: DataLayerError) -> None:
        """Execute registered callbacks for the error's component."""
        # Handle both ErrorContext objects and backward-compatible dict contexts
        if hasattr(error, '_error_context') and error._error_context:
            component = error._error_context.component
        elif hasattr(error.context, 'component'):
            component = error.context.component
        else:
            # Fallback for dict contexts
            component = ComponentType.CAPTURE

        callbacks = self.error_callbacks.get(component, [])
        for callback in callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")
    
    def _attempt_recovery(self, error: DataLayerError) -> Optional[Any]:
        """Attempt to recover from error using registered strategies."""
        strategy = self.recovery_strategies.get(error.error_type)
        if not strategy:
            # Try generic recovery based on component
            # Handle both ErrorContext objects and backward-compatible dict contexts
            if hasattr(error, '_error_context') and error._error_context:
                component = error._error_context.component
            elif hasattr(error.context, 'component'):
                component = error.context.component
            else:
                # Fallback for dict contexts
                component = ComponentType.CAPTURE

            strategy = self.recovery_strategies.get(f"generic_{component.value}")
        
        if strategy:
            try:
                logger.info(f"Attempting recovery for {error.error_type}")
                result = strategy(error)
                logger.info(f"Recovery successful for {error.error_type}")
                return result
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {error.error_type}: {recovery_error}")
        
        return None
    
    def _register_default_recovery_strategies(self) -> None:
        """Register default recovery strategies for common error types."""
        
        # Capture failures - try alternative methods
        self.recovery_strategies["capture_timeout"] = self._recover_capture_timeout
        self.recovery_strategies["capture_javascript_error"] = self._recover_javascript_error
        
        # Validation failures - provide default/empty results
        self.recovery_strategies["schema_validation_failed"] = self._recover_validation_failure
        
        # Generic component recoveries
        self.recovery_strategies["generic_capture"] = self._recover_generic_capture
        self.recovery_strategies["generic_validation"] = self._recover_generic_validation
    
    def _recover_capture_timeout(self, error: DataLayerError) -> Dict[str, Any]:
        """Recovery strategy for capture timeouts."""
        return {
            "success": False,
            "fallback_data": {},
            "recovery_method": "timeout_fallback",
            "message": "Captured minimal data due to timeout"
        }
    
    def _recover_javascript_error(self, error: DataLayerError) -> Dict[str, Any]:
        """Recovery strategy for JavaScript errors during capture."""
        return {
            "success": False,
            "fallback_data": {"error": "JavaScript execution failed"},
            "recovery_method": "javascript_error_fallback",
            "message": "Used fallback data due to JavaScript error"
        }
    
    def _recover_validation_failure(self, error: DataLayerError) -> List[Dict[str, Any]]:
        """Recovery strategy for validation failures."""
        return [{
            "severity": "info",
            "message": "Validation failed, continuing without validation",
            "recovery_method": "skip_validation"
        }]
    
    def _recover_generic_capture(self, error: DataLayerError) -> Dict[str, Any]:
        """Generic recovery for capture component failures."""
        return {
            "success": False,
            "data": {},
            "recovery_method": "generic_capture_fallback"
        }
    
    def _recover_generic_validation(self, error: DataLayerError) -> List[Any]:
        """Generic recovery for validation component failures."""
        return []
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics.
        
        Returns:
            Error statistics dictionary
        """
        if not self.error_history:
            return {"total_errors": 0, "component_breakdown": {}, "severity_breakdown": {}}
        
        # Calculate statistics
        total_errors = len(self.error_history)
        recent_errors = [
            e for e in self.error_history 
            if e.context.timestamp > datetime.utcnow() - timedelta(hours=24)
        ]
        
        # Component breakdown
        component_counts = {}
        for error in self.error_history:
            component = error.context.component.value
            component_counts[component] = component_counts.get(component, 0) + 1
        
        # Severity breakdown
        severity_counts = {}
        for error in self.error_history:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_errors": total_errors,
            "recent_errors_24h": len(recent_errors),
            "component_breakdown": component_counts,
            "severity_breakdown": severity_counts,
            "error_rate_trend": self._calculate_error_trend(),
            "most_common_errors": self._get_most_common_errors()
        }
    
    def _calculate_error_trend(self) -> str:
        """Calculate error rate trend over time."""
        if len(self.error_history) < 10:
            return "insufficient_data"
        
        recent_half = self.error_history[-5:]
        earlier_half = self.error_history[-10:-5]
        
        recent_rate = len(recent_half) / 5
        earlier_rate = len(earlier_half) / 5
        
        if recent_rate > earlier_rate * 1.5:
            return "increasing"
        elif recent_rate < earlier_rate * 0.5:
            return "decreasing"
        else:
            return "stable"
    
    def _get_most_common_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most common error types."""
        error_type_counts = {}
        for error in self.error_history:
            # Handle both ErrorContext objects and dict contexts
            if hasattr(error, '_error_context') and error._error_context:
                component = error._error_context.component
            elif hasattr(error.context, 'component'):
                component = error.context.component
            else:
                # Fallback for dict contexts
                component = ComponentType.CAPTURE

            key = f"{component.value}:{error.error_type}"
            error_type_counts[key] = error_type_counts.get(key, 0) + 1
        
        sorted_errors = sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"error_type": error_type, "count": count}
            for error_type, count in sorted_errors[:limit]
        ]

    def get_error_rate(self) -> float:
        """Calculate the error rate as a ratio of errors to total operations.

        Returns:
            Error rate as a float between 0.0 and 1.0
        """
        if self._total_operations == 0:
            return 0.0
        return len(self.error_history) / self._total_operations

    def get_error_summary(self) -> Dict[str, Any]:
        """Get a comprehensive error summary.

        Returns:
            Dictionary with error summary statistics
        """
        by_component = {}
        by_severity = {}
        by_category = {}

        for error in self.error_history:
            # Count by component - handle both ErrorContext objects and dict contexts
            if hasattr(error, '_error_context') and error._error_context:
                component = error._error_context.component
            elif hasattr(error.context, 'component'):
                component = error.context.component
            else:
                # Fallback for dict contexts
                component = ComponentType.CAPTURE

            by_component[component] = by_component.get(component, 0) + 1

            # Count by severity
            severity = error.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1

            # Count by category (based on error type)
            category = self._categorize_error(error)
            by_category[category] = by_category.get(category, 0) + 1

        return {
            "total_errors": len(self.error_history),
            "by_component": by_component,
            "by_severity": by_severity,
            "by_category": by_category,
            "error_rate": self.get_error_rate(),
            "most_common": self._get_most_common_errors(3)
        }

    def _categorize_error(self, error: 'DataLayerError') -> str:
        """Categorize an error based on its type and message.

        Args:
            error: DataLayerError to categorize

        Returns:
            Error category string
        """
        error_type = error.error_type.lower()
        message = error.message.lower()

        if 'timeout' in error_type or 'timeout' in message:
            return 'timeout'
        elif 'validation' in error_type or 'schema' in message:
            return 'validation'
        elif 'network' in error_type or 'connection' in message:
            return 'network'
        elif 'permission' in error_type or 'auth' in message:
            return 'permission'
        else:
            return 'other'

    def get_circuit_breaker(self, component: 'ComponentType') -> 'CircuitBreaker':
        """Get or create a circuit breaker for a specific component.

        Args:
            component: Component type to get circuit breaker for

        Returns:
            CircuitBreaker instance for the component
        """
        component_key = component.value if hasattr(component, 'value') else str(component)

        if component_key not in self._circuit_breakers:
            self._circuit_breakers[component_key] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                expected_exception=Exception
            )

        return self._circuit_breakers[component_key]

    def get_health_status(self) -> Dict[str, Any]:
        """Get the health status of the error handler.

        Returns:
            Dictionary with health status information
        """
        total_errors = len(self.error_history)
        error_rate = self.get_error_rate()

        # Determine health based on error rate
        if error_rate == 0.0:
            health = "healthy"
        elif error_rate < 0.1:  # Less than 10% error rate
            health = "degraded"
        else:
            health = "unhealthy"

        return {
            "status": health,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "circuit_breakers": {
                key: {
                    "state": breaker.state.name if hasattr(breaker, 'state') else "unknown"
                }
                for key, breaker in self._circuit_breakers.items()
            }
        }


class CircuitBreaker:
    """Circuit breaker pattern for dataLayer operations."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2,
        expected_exception: Type[Exception] = Exception
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            success_threshold: Number of successes to close circuit from half-open
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def __call__(self, func):
        """Decorator to apply circuit breaker to a function."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half_open"
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is open")
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def record_success(self) -> None:
        """Record a successful operation."""
        self.success_count += 1
        if self.state == "half_open":
            if self.success_count >= self.success_threshold:
                self.state = "closed"
                self.failure_count = 0
                # Don't reset success_count - keep cumulative track

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == "closed" and self.failure_count >= self.failure_threshold:
            self.state = "open"
        elif self.state == "half_open":
            self.state = "open"
            self.success_count = 0  # Reset success count only when going back to open from half-open

    def can_execute(self) -> bool:
        """Check if operations can be executed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.success_count = 0
                return True
            return False
        elif self.state == "half_open":
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "last_failure_time": self.last_failure_time
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


def resilient_operation(
    component: ComponentType,
    operation: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    fallback_result: Any = None,
    error_handler: Optional[DataLayerErrorHandler] = None,
    error_callback: Optional[Callable] = None
):
    """Decorator for resilient dataLayer operations with retry logic.

    Args:
        component: Component type for error tracking
        operation: Operation name
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        fallback_result: Result to return if all retries fail
        error_handler: Error handler instance
        error_callback: Optional callback function to call on errors
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                context = ErrorContext(
                    component=component,
                    operation=operation,
                    max_retries=max_retries
                )

                # Extract page URL if available in arguments
                for arg in args:
                    if hasattr(arg, 'url'):
                        context.page_url = arg.url
                        break
                for value in kwargs.values():
                    if isinstance(value, str) and value.startswith('http'):
                        context.page_url = value
                        break

                last_exception = None

                for attempt in range(max_retries + 1):
                    context.retry_count = attempt

                    try:
                        return await func(*args, **kwargs)

                    except Exception as e:
                        last_exception = e

                        # Create error object
                        error = DataLayerError(
                            error_type=type(e).__name__,
                            message=str(e),
                            severity=ErrorSeverity.MEDIUM if attempt < max_retries else ErrorSeverity.HIGH,
                            context=context,
                            exception=e,
                            recovery_action=f"Retry {attempt + 1}/{max_retries}" if attempt < max_retries else "Use fallback",
                            impact_assessment="Operation failed, attempting recovery"
                        )

                        # Call error callback if provided
                        if error_callback:
                            try:
                                error_callback(e, context)
                            except Exception as callback_error:
                                logger.error(f"Error callback failed: {callback_error}")

                        # Handle error
                        if error_handler:
                            recovery_result = error_handler.handle_error(error, attempt_recovery=True)
                            if recovery_result is not None:
                                return recovery_result
                        else:
                            logger.warning(f"Resilient operation {operation} failed (attempt {attempt + 1}): {e}")

                        # Wait before retry (except on last attempt)
                        if attempt < max_retries:
                            await asyncio.sleep(retry_delay * (attempt + 1))  # Progressive backoff

                # All retries exhausted
                if fallback_result is not None:
                    logger.info(f"Using fallback result for {operation} after {max_retries + 1} attempts")
                    return fallback_result
                else:
                    # Re-raise the last exception
                    raise last_exception

            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                context = ErrorContext(
                    component=component,
                    operation=operation,
                    max_retries=max_retries
                )

                # Extract page URL if available in arguments
                for arg in args:
                    if hasattr(arg, 'url'):
                        context.page_url = arg.url
                        break
                for value in kwargs.values():
                    if isinstance(value, str) and value.startswith('http'):
                        context.page_url = value
                        break

                last_exception = None

                for attempt in range(max_retries + 1):
                    context.retry_count = attempt

                    try:
                        return func(*args, **kwargs)

                    except Exception as e:
                        last_exception = e

                        # Create error object
                        error = DataLayerError(
                            error_type=type(e).__name__,
                            message=str(e),
                            severity=ErrorSeverity.MEDIUM if attempt < max_retries else ErrorSeverity.HIGH,
                            context=context,
                            exception=e,
                            recovery_action=f"Retry {attempt + 1}/{max_retries}" if attempt < max_retries else "Use fallback",
                            impact_assessment="Operation failed, attempting recovery"
                        )

                        # Call error callback if provided
                        if error_callback:
                            try:
                                error_callback(e, context)
                            except Exception as callback_error:
                                logger.error(f"Error callback failed: {callback_error}")

                        # Handle error
                        if error_handler:
                            recovery_result = error_handler.handle_error(error, attempt_recovery=True)
                            if recovery_result is not None:
                                return recovery_result
                        else:
                            logger.warning(f"Resilient operation {operation} failed (attempt {attempt + 1}): {e}")

                        # Wait before retry (except on last attempt)
                        if attempt < max_retries:
                            import time
                            time.sleep(retry_delay * (attempt + 1))  # Progressive backoff

                # All retries exhausted
                if fallback_result is not None:
                    logger.info(f"Using fallback result for {operation} after {max_retries + 1} attempts")
                    return fallback_result
                else:
                    # Re-raise the last exception
                    raise last_exception

            return sync_wrapper
    return decorator


class GracefulDegradationContext:
    """Context manager for graceful degradation results."""

    def __init__(self, exception: Optional[Exception] = None, fallback_result: Any = None):
        self.exception = exception
        self.fallback_result = fallback_result
        self.has_error = exception is not None
        # Additional attributes expected by tests
        self.result = None
        self.error_occurred = False


class graceful_degradation:
    """Context manager for graceful degradation of operations.

    Supports multiple usage patterns:
    - with graceful_degradation(fallback_value="fallback") as degraded:
    - with graceful_degradation() as degraded:
    - with graceful_degradation(fallback_func=lambda: "fallback") as degraded:
    """

    def __init__(
        self,
        fallback_value: Any = None,
        fallback_func: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
        expected_exceptions: Optional[tuple] = None,
        operation_name: str = "unknown_operation",
        component: ComponentType = ComponentType.CAPTURE,
        error_handler: Optional[DataLayerErrorHandler] = None
    ):
        self.fallback_value = fallback_value
        self.fallback_func = fallback_func
        self.error_callback = error_callback
        self.expected_exceptions = expected_exceptions or (Exception,)
        self.operation_name = operation_name
        self.component = component
        self.error_handler = error_handler
        self.context_result = None

    def __enter__(self) -> GracefulDegradationContext:
        """Enter the context manager."""
        self.context_result = GracefulDegradationContext()
        return self.context_result

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager with error handling."""
        if exc_type is None:
            # No exception occurred - operation was successful
            # The context.result will remain None, indicating successful completion
            return False

        if not issubclass(exc_type, self.expected_exceptions):
            # Not an expected exception, let it propagate
            return False

        # Handle the expected exception
        if self.error_callback:
            try:
                self.error_callback(exc_value)
            except Exception as callback_error:
                logger.error(f"Error callback failed: {callback_error}")

        # Create error for handler if provided
        if self.error_handler:
            context = ErrorContext(component=self.component, operation=self.operation_name)
            error = DataLayerError(
                error_type=type(exc_value).__name__,
                message=str(exc_value),
                severity=ErrorSeverity.MEDIUM,
                context=context,
                exception=exc_value,
                recovery_action="Graceful degradation",
                impact_assessment="Operation degraded but system continues"
            )
            self.error_handler.handle_error(error, attempt_recovery=True)

        logger.warning(f"Graceful degradation for {self.operation_name}: {exc_value}")

        # Determine fallback result
        if self.fallback_func:
            try:
                fallback_result = self.fallback_func()
            except Exception as fallback_error:
                logger.error(f"Fallback function failed: {fallback_error}")
                fallback_result = None
        else:
            fallback_result = self.fallback_value

        # Update context with error information
        if self.context_result:
            self.context_result.error_occurred = True
            self.context_result.exception = exc_value

        # Set the fallback result in context if available
        if self.context_result:
            self.context_result.result = fallback_result

        if fallback_result is None:
            # No fallback provided, but still suppress exception for graceful degradation
            # The context will have result=None and error_occurred=True
            return True

        return True


class ResilientDataLayerService:
    """Wrapper service that adds resilience to dataLayer operations."""
    
    def __init__(self, base_service: Any, error_handler: Optional[DataLayerErrorHandler] = None):
        """Initialize resilient service wrapper.
        
        Args:
            base_service: Base dataLayer service to wrap
            error_handler: Error handler instance
        """
        self.base_service = base_service
        self.error_handler = error_handler or DataLayerErrorHandler()
        self.circuit_breakers = {}
        
        # Initialize circuit breakers for critical operations
        self.circuit_breakers['capture'] = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
        self.circuit_breakers['validation'] = CircuitBreaker(
            failure_threshold=10,
            recovery_timeout=30
        )
    
    @resilient_operation(
        component=ComponentType.CAPTURE,
        operation="capture_and_validate",
        max_retries=2,
        fallback_result={"success": False, "data": {}, "issues": []}
    )
    async def capture_and_validate(self, *args, **kwargs):
        """Resilient capture and validation with circuit breaker."""
        circuit_breaker = self.circuit_breakers['capture']
        return await circuit_breaker(self.base_service.capture_and_validate)(*args, **kwargs)
    
    @resilient_operation(
        component=ComponentType.AGGREGATION,
        operation="process_multiple_pages",
        max_retries=1,
        fallback_result=[]
    )
    async def process_multiple_pages(self, *args, **kwargs):
        """Resilient multi-page processing."""
        return await self.base_service.process_multiple_pages(*args, **kwargs)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the resilient service.
        
        Returns:
            Health status information
        """
        circuit_breaker_status = {}
        for name, cb in self.circuit_breakers.items():
            circuit_breaker_status[name] = {
                'state': cb.state,
                'failure_count': cb.failure_count,
                'last_failure': cb.last_failure_time
            }
        
        return {
            'circuit_breakers': circuit_breaker_status,
            'error_statistics': self.error_handler.get_error_statistics(),
            'overall_health': self._calculate_overall_health()
        }
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall system health based on circuit breakers and errors."""
        # Check circuit breaker states
        open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb.state == "open")
        
        if open_breakers > 1:
            return "critical"
        elif open_breakers == 1:
            return "degraded"
        
        # Check error rates
        error_stats = self.error_handler.get_error_statistics()
        recent_errors = error_stats.get('recent_errors_24h', 0)
        
        if recent_errors > 50:
            return "degraded"
        elif recent_errors > 100:
            return "critical"
        else:
            return "healthy"


# Global error handler instance
global_error_handler = DataLayerErrorHandler()


def get_global_error_handler() -> DataLayerErrorHandler:
    """Get the global error handler instance."""
    return global_error_handler


def create_resilient_service(base_service: Any) -> ResilientDataLayerService:
    """Create a resilient wrapper for a dataLayer service."""
    return ResilientDataLayerService(base_service, global_error_handler)