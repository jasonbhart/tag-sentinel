"""Error handling framework for analytics detectors.

Provides comprehensive error handling, logging, and graceful degradation
for detector components to ensure system resilience.
"""

import logging
import traceback
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field

from .base import DetectResult, DetectorNote, NoteSeverity, NoteCategory


# Configure logging
logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Severity levels for detector errors."""
    CRITICAL = "critical"    # System-breaking errors
    HIGH = "high"           # Major feature failures
    MEDIUM = "medium"       # Degraded functionality
    LOW = "low"            # Minor issues, warnings
    INFO = "info"          # Informational messages


class ErrorCategory(str, Enum):
    """Categories for error classification."""
    NETWORK = "network"                # Network request failures
    PARSING = "parsing"               # Data parsing errors
    VALIDATION = "validation"         # Validation failures
    CONFIGURATION = "configuration"   # Config-related errors
    TIMEOUT = "timeout"               # Timeout errors
    MEMORY = "memory"                 # Memory/resource errors
    EXTERNAL_API = "external_api"     # External API failures
    DATA_QUALITY = "data_quality"     # Data quality issues
    PERFORMANCE = "performance"       # Performance degradation
    UNKNOWN = "unknown"               # Unclassified errors


class DetectorError(BaseModel):
    """Structured error information for detector operations."""
    
    error_id: str = Field(description="Unique error identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    severity: ErrorSeverity = Field(description="Error severity level")
    category: ErrorCategory = Field(description="Error category")
    
    detector_name: str = Field(description="Detector that encountered the error")
    operation: str = Field(description="Operation that failed")
    message: str = Field(description="Human-readable error message")
    
    # Context information
    page_url: Optional[str] = Field(default=None, description="Page URL where error occurred")
    request_url: Optional[str] = Field(default=None, description="Request URL that caused error")
    
    # Technical details
    exception_type: Optional[str] = Field(default=None, description="Exception class name")
    exception_message: Optional[str] = Field(default=None, description="Exception message")
    stack_trace: Optional[str] = Field(default=None, description="Stack trace")
    
    # Additional context
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")
    
    # Recovery information
    recoverable: bool = Field(default=True, description="Whether error is recoverable")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    
    @classmethod
    def from_exception(cls, 
                      e: Exception, 
                      detector_name: str,
                      operation: str,
                      severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                      category: ErrorCategory = ErrorCategory.UNKNOWN,
                      **context) -> "DetectorError":
        """Create DetectorError from exception.
        
        Args:
            e: Exception that occurred
            detector_name: Name of detector
            operation: Operation that failed
            severity: Error severity level
            category: Error category
            **context: Additional context information
            
        Returns:
            DetectorError instance
        """
        error_id = f"{detector_name}_{operation}_{datetime.utcnow().timestamp()}"
        
        return cls(
            error_id=error_id,
            severity=severity,
            category=category,
            detector_name=detector_name,
            operation=operation,
            message=f"{operation} failed: {str(e)}",
            exception_type=type(e).__name__,
            exception_message=str(e),
            stack_trace=traceback.format_exc(),
            context=context,
            recoverable=cls._is_recoverable_error(e)
        )
    
    @staticmethod
    def _is_recoverable_error(e: Exception) -> bool:
        """Determine if error is recoverable."""
        # Network errors are typically recoverable
        if "timeout" in str(e).lower() or "connection" in str(e).lower():
            return True
        
        # Parsing errors for individual requests are recoverable
        if isinstance(e, (ValueError, KeyError)):
            return True
        
        # Memory errors are not recoverable
        if isinstance(e, MemoryError):
            return False
        
        # Default to recoverable
        return True
    
    def to_detector_note(self) -> DetectorNote:
        """Convert to DetectorNote for result reporting."""
        # Map error severity to note severity
        severity_map = {
            ErrorSeverity.CRITICAL: NoteSeverity.ERROR,
            ErrorSeverity.HIGH: NoteSeverity.ERROR,
            ErrorSeverity.MEDIUM: NoteSeverity.WARNING,
            ErrorSeverity.LOW: NoteSeverity.WARNING,
            ErrorSeverity.INFO: NoteSeverity.INFO
        }
        
        # Map error category to note category
        category_map = {
            ErrorCategory.NETWORK: NoteCategory.PERFORMANCE,
            ErrorCategory.PARSING: NoteCategory.VALIDATION,
            ErrorCategory.VALIDATION: NoteCategory.VALIDATION,
            ErrorCategory.CONFIGURATION: NoteCategory.CONFIGURATION,
            ErrorCategory.TIMEOUT: NoteCategory.PERFORMANCE,
            ErrorCategory.EXTERNAL_API: NoteCategory.VALIDATION,
            ErrorCategory.DATA_QUALITY: NoteCategory.DATA_QUALITY,
            ErrorCategory.PERFORMANCE: NoteCategory.PERFORMANCE,
        }
        
        return DetectorNote(
            severity=severity_map.get(self.severity, NoteSeverity.WARNING),
            category=category_map.get(self.category, NoteCategory.VALIDATION),
            message=self.message,
            page_url=self.page_url or "",
            detector_name=self.detector_name,
            details={
                "error_id": self.error_id,
                "operation": self.operation,
                "error_category": self.category,
                "error_severity": self.severity,
                "exception_type": self.exception_type,
                "recoverable": self.recoverable,
                "retry_count": self.retry_count,
                **self.context
            }
        )


class ErrorCollector:
    """Collects and manages errors during detector execution."""
    
    def __init__(self, detector_name: str):
        self.detector_name = detector_name
        self.errors: List[DetectorError] = []
        self.warnings: List[DetectorError] = []
        self.info: List[DetectorError] = []
    
    def add_error(self, error: DetectorError) -> None:
        """Add error to collection."""
        if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.errors.append(error)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.warnings.append(error)
        else:
            self.info.append(error)
        
        # Log error
        logger.error(f"[{self.detector_name}] {error.message}", extra={
            "error_id": error.error_id,
            "severity": error.severity,
            "category": error.category,
            "operation": error.operation
        })
    
    def add_from_exception(self, 
                          e: Exception,
                          operation: str,
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                          category: ErrorCategory = ErrorCategory.UNKNOWN,
                          **context) -> DetectorError:
        """Add error from exception and return it."""
        error = DetectorError.from_exception(
            e, self.detector_name, operation, severity, category, **context
        )
        self.add_error(error)
        return error
    
    def has_critical_errors(self) -> bool:
        """Check if any critical errors were collected."""
        return any(error.severity == ErrorSeverity.CRITICAL for error in self.errors)
    
    def has_blocking_errors(self) -> bool:
        """Check if any blocking (non-recoverable) errors were collected."""
        return any(not error.recoverable for error in self.errors)
    
    def add_to_result(self, result: DetectResult) -> None:
        """Add collected errors to DetectResult."""
        all_errors = self.errors + self.warnings + self.info
        
        for error in all_errors:
            note = error.to_detector_note()
            result.add_note(note)


class ResilientDetector:
    """Mixin class providing error handling and resilience features."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_collector: Optional[ErrorCollector] = None
    
    def _create_error_collector(self) -> ErrorCollector:
        """Create error collector for this detector instance."""
        detector_name = getattr(self, 'name', self.__class__.__name__)
        return ErrorCollector(detector_name)
    
    @contextmanager
    def error_context(self, operation: str, **context):
        """Context manager for error handling within operations."""
        if self.error_collector is None:
            self.error_collector = self._create_error_collector()
        
        try:
            yield self.error_collector
        except Exception as e:
            # Determine severity and category based on exception type
            severity, category = self._classify_exception(e)
            
            self.error_collector.add_from_exception(
                e, operation, severity, category, **context
            )
    
    def _classify_exception(self, e: Exception) -> tuple[ErrorSeverity, ErrorCategory]:
        """Classify exception to determine severity and category."""
        exception_type = type(e).__name__
        message = str(e).lower()
        
        # Network-related errors
        if any(term in message for term in ["timeout", "connection", "network", "dns"]):
            return ErrorSeverity.MEDIUM, ErrorCategory.NETWORK
        
        # Memory errors are critical
        if isinstance(e, MemoryError):
            return ErrorSeverity.CRITICAL, ErrorCategory.MEMORY
        
        # Parsing errors
        if isinstance(e, (ValueError, json.JSONDecodeError)):
            return ErrorSeverity.LOW, ErrorCategory.PARSING
        
        # Key/attribute errors (usually data quality issues)
        if isinstance(e, (KeyError, AttributeError)):
            return ErrorSeverity.LOW, ErrorCategory.DATA_QUALITY
        
        # Validation errors
        if "validation" in message:
            return ErrorSeverity.MEDIUM, ErrorCategory.VALIDATION
        
        # HTTP errors
        if "http" in message or any(code in message for code in ["404", "500", "503"]):
            return ErrorSeverity.MEDIUM, ErrorCategory.EXTERNAL_API
        
        # Default classification
        return ErrorSeverity.MEDIUM, ErrorCategory.UNKNOWN


F = TypeVar('F', bound=Callable[..., Any])


def resilient_operation(operation_name: str,
                       severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                       category: ErrorCategory = ErrorCategory.UNKNOWN,
                       default_return: Any = None,
                       log_success: bool = False) -> Callable[[F], F]:
    """Decorator for making operations resilient to errors.
    
    Args:
        operation_name: Name of the operation for logging
        severity: Default error severity
        category: Default error category  
        default_return: Value to return on error
        log_success: Whether to log successful operations
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if log_success:
                    logger.debug(f"Operation {operation_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Operation {operation_name} failed: {e}")
                
                # If this is being called on a ResilientDetector instance,
                # add error to collector
                if args and hasattr(args[0], 'error_collector'):
                    detector = args[0]
                    if detector.error_collector is None:
                        detector.error_collector = detector._create_error_collector()
                    
                    detector.error_collector.add_from_exception(
                        e, operation_name, severity, category
                    )
                
                return default_return
        
        return wrapper
    return decorator


def safe_request_processing(requests: List[Any],
                           processor: Callable[[Any], Any],
                           max_failures: Optional[int] = None,
                           error_collector: Optional[ErrorCollector] = None) -> List[Any]:
    """Safely process a list of requests with error handling.
    
    Args:
        requests: List of requests to process
        processor: Function to process each request
        max_failures: Maximum failures before stopping (None = no limit)
        error_collector: Optional error collector
        
    Returns:
        List of successfully processed results
    """
    results = []
    failures = 0
    
    for i, request in enumerate(requests):
        try:
            result = processor(request)
            if result is not None:
                results.append(result)
        except Exception as e:
            failures += 1
            
            if error_collector:
                error_collector.add_from_exception(
                    e, f"process_request_{i}",
                    severity=ErrorSeverity.LOW,
                    category=ErrorCategory.PARSING,
                    request_index=i,
                    total_requests=len(requests)
                )
            else:
                logger.warning(f"Failed to process request {i}: {e}")
            
            # Stop if too many failures
            if max_failures and failures >= max_failures:
                logger.warning(f"Stopping processing after {failures} failures")
                break
    
    return results


def with_timeout(timeout_seconds: float,
                operation_name: str = "operation",
                default_return: Any = None) -> Callable[[F], F]:
    """Decorator to add timeout protection to operations.
    
    Args:
        timeout_seconds: Timeout in seconds
        operation_name: Name for logging
        default_return: Value to return on timeout
        
    Returns:
        Decorated function with timeout protection
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"{operation_name} timed out after {timeout_seconds}s")
            
            # Set timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
                return result
            except TimeoutError as e:
                logger.warning(str(e))
                return default_return
            finally:
                signal.alarm(0)  # Cancel alarm
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


# Import json for the _classify_exception method
import json