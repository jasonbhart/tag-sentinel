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


@dataclass
class DataLayerError:
    """Comprehensive error information for dataLayer operations."""
    error_type: str
    message: str
    severity: ErrorSeverity
    context: ErrorContext
    exception: Optional[Exception] = None
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None
    impact_assessment: Optional[str] = None
    
    def __post_init__(self):
        """Generate stack trace if exception provided."""
        if self.exception and not self.stack_trace:
            self.stack_trace = ''.join(traceback.format_exception(
                type(self.exception), self.exception, self.exception.__traceback__
            ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            'error_type': self.error_type,
            'message': self.message,
            'severity': self.severity.value,
            'context': self.context.to_dict(),
            'exception_type': type(self.exception).__name__ if self.exception else None,
            'stack_trace': self.stack_trace,
            'recovery_action': self.recovery_action,
            'impact_assessment': self.impact_assessment
        }


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
        error: DataLayerError,
        attempt_recovery: bool = True
    ) -> Optional[Any]:
        """Handle a dataLayer error with logging and potential recovery.
        
        Args:
            error: DataLayer error to handle
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            Recovery result if successful, None otherwise
        """
        # Log error based on severity
        self._log_error(error)
        
        # Add to error history
        self._record_error(error)
        
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
        error_key = f"{error.context.component.value}:{error.error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Update component error rates (simplified)
        component = error.context.component
        if component not in self.component_error_rates:
            self.component_error_rates[component] = 0.0
        self.component_error_rates[component] += 1.0
    
    def _execute_error_callbacks(self, error: DataLayerError) -> None:
        """Execute registered callbacks for the error's component."""
        callbacks = self.error_callbacks.get(error.context.component, [])
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
            strategy = self.recovery_strategies.get(f"generic_{error.context.component.value}")
        
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
            key = f"{error.context.component.value}:{error.error_type}"
            error_type_counts[key] = error_type_counts.get(key, 0) + 1
        
        sorted_errors = sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"error_type": error_type, "count": count}
            for error_type, count in sorted_errors[:limit]
        ]


class CircuitBreaker:
    """Circuit breaker pattern for dataLayer operations."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
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


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


def resilient_operation(
    component: ComponentType,
    operation: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    fallback_result: Any = None,
    error_handler: Optional[DataLayerErrorHandler] = None
):
    """Decorator for resilient dataLayer operations with retry logic.
    
    Args:
        component: Component type for error tracking
        operation: Operation name
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        fallback_result: Result to return if all retries fail
        error_handler: Error handler instance
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
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
        
        return wrapper
    return decorator


@asynccontextmanager
async def graceful_degradation(
    operation_name: str,
    component: ComponentType,
    fallback_result: Any = None,
    error_handler: Optional[DataLayerErrorHandler] = None
):
    """Context manager for graceful degradation of operations.
    
    Args:
        operation_name: Name of the operation
        component: Component type
        fallback_result: Result to provide if operation fails
        error_handler: Error handler instance
    """
    try:
        yield
    except Exception as e:
        context = ErrorContext(component=component, operation=operation_name)
        error = DataLayerError(
            error_type=type(e).__name__,
            message=str(e),
            severity=ErrorSeverity.MEDIUM,
            context=context,
            exception=e,
            recovery_action="Graceful degradation",
            impact_assessment="Operation degraded but system continues"
        )
        
        if error_handler:
            recovery_result = error_handler.handle_error(error, attempt_recovery=True)
            if recovery_result is not None:
                yield recovery_result
                return
        
        logger.warning(f"Graceful degradation for {operation_name}: {e}")
        
        if fallback_result is not None:
            yield fallback_result
        else:
            # Re-raise if no fallback provided
            raise


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