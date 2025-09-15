"""Unit tests for DataLayer error handling system."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Callable
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.audit.datalayer.error_handling import (
    DataLayerErrorHandler,
    CircuitBreaker,
    CircuitBreakerOpenError,
    ComponentType,
    ErrorSeverity,
    DataLayerError,
    ErrorContext,
    ResilientDataLayerService
)

# Define missing classes for tests
class CaptureError(Exception):
    """Capture error for testing."""
    def __init__(self, message, context=None):
        super().__init__(message)
        self.context = context or {}

class ValidationError(Exception):
    """Validation error for testing."""
    def __init__(self, message, context=None):
        super().__init__(message)
        self.context = context or {}

class RedactionError(Exception):
    """Redaction error for testing."""
    def __init__(self, message, context=None):
        super().__init__(message)
        self.context = context or {}

# Define missing enums for tests
class ErrorCategory:
    TIMEOUT = "timeout"
    EXECUTION = "execution"
    NETWORK = "network"
    VALIDATION = "validation"

class CircuitBreakerState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class RecoveryStrategy:
    RETRY = "retry"
    FAIL_FAST = "fail_fast"


class TestDataLayerErrorHandler:
    """Test cases for DataLayerErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = DataLayerErrorHandler()
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        assert len(self.error_handler.error_history) == 0
        assert len(self.error_handler.error_callbacks) == 0
        assert len(self.error_handler.error_counts) == 0
    
    def test_register_error_callback(self):
        """Test registering error callbacks."""
        callback_called = False

        def test_callback(error):
            nonlocal callback_called
            callback_called = True

        self.error_handler.register_error_callback(ComponentType.CAPTURE, test_callback)

        # Should have registered callback
        assert ComponentType.CAPTURE in self.error_handler.error_callbacks
        assert len(self.error_handler.error_callbacks[ComponentType.CAPTURE]) == 1
    
    def test_register_recovery_strategy(self):
        """Test registering recovery strategies."""
        def test_recovery(error):
            return {"recovered": True}

        self.error_handler.register_recovery_strategy("capture_timeout", test_recovery)

        # Should have registered strategy
        assert "capture_timeout" in self.error_handler.recovery_strategies
    
    def test_handle_error_with_recovery(self):
        """Test error handling with recovery strategy."""
        # Create DataLayerError with proper structure
        error_context = ErrorContext(
            component=ComponentType.CAPTURE,
            operation="test_capture"
        )

        data_layer_error = DataLayerError(
            error_type="capture_error",
            message="Test capture error",
            severity=ErrorSeverity.MEDIUM,
            context=error_context,
            exception=CaptureError("Test capture error")
        )

        # Register a recovery strategy
        def mock_recovery(error):
            return {"recovered": True}

        self.error_handler.register_recovery_strategy("capture_error", mock_recovery)

        result = self.error_handler.handle_error(data_layer_error, attempt_recovery=True)

        assert result is not None
        assert result["recovered"] is True
        assert len(self.error_handler.error_history) == 1
    
    def test_handle_error_without_recovery(self):
        """Test error handling without recovery."""
        # Create DataLayerError with proper structure
        error_context = ErrorContext(
            component=ComponentType.VALIDATION,
            operation="test_validation"
        )

        data_layer_error = DataLayerError(
            error_type="validation_error",
            message="Test validation error",
            severity=ErrorSeverity.HIGH,
            context=error_context,
            exception=ValidationError("Test validation error")
        )

        result = self.error_handler.handle_error(data_layer_error, attempt_recovery=False)

        assert result is None  # No recovery attempted
        assert len(self.error_handler.error_history) == 1

        # Check error was recorded
        recorded_error = self.error_handler.error_history[0]
        assert recorded_error.context.component == ComponentType.VALIDATION
        assert recorded_error.error_type == "validation_error"
    
    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        # Initially no errors
        assert self.error_handler.get_error_rate() == 0.0
        
        # Add some errors
        for i in range(10):
            error = DataLayerError(f"Error {i}")
            self.error_handler.handle_error(error, ComponentType.CAPTURE, {"attempt": i})
        
        # Add some successes (mock by setting total operations)
        self.error_handler._total_operations = 20
        
        error_rate = self.error_handler.get_error_rate()
        assert error_rate == 0.5  # 10 errors out of 20 operations
    
    def test_get_error_summary(self):
        """Test getting error summary."""
        # Add diverse errors
        errors = [
            (CaptureError("Timeout"), ComponentType.CAPTURE),
            (ValidationError("Schema error"), ComponentType.VALIDATION),
            (RedactionError("Pattern error"), ComponentType.REDACTION),
            (CaptureError("Another timeout"), ComponentType.CAPTURE)
        ]
        
        for error, component in errors:
            self.error_handler.handle_error(error, component, {})
        
        summary = self.error_handler.get_error_summary()
        
        assert summary["total_errors"] == 4
        assert ComponentType.CAPTURE in summary["by_component"]
        assert ComponentType.VALIDATION in summary["by_component"]
        assert summary["by_component"][ComponentType.CAPTURE] == 2  # Two capture errors
        
        # Should have category breakdown
        assert ErrorCategory.TIMEOUT in summary["by_category"]
        assert ErrorCategory.VALIDATION in summary["by_category"]
    
    def test_circuit_breaker_integration(self):
        """Test integration with circuit breakers."""
        # Get circuit breaker for component
        circuit_breaker = self.error_handler.get_circuit_breaker(ComponentType.CAPTURE)
        
        assert circuit_breaker is not None
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        
        # Should reuse same circuit breaker
        same_breaker = self.error_handler.get_circuit_breaker(ComponentType.CAPTURE)
        assert same_breaker is circuit_breaker
    
    def test_error_history_retention(self):
        """Test error history retention limits."""
        # Add many errors to test history retention
        for i in range(150):  # More than max_history_size (100)
            error = DataLayerError(f"Error {i}")
            self.error_handler.handle_error(error, ComponentType.CAPTURE, {})
        
        # Should not exceed max history size
        assert len(self.error_handler.error_history) <= 100
        
        # Should keep most recent errors
        latest_error = self.error_handler.error_history[-1]
        assert "Error 149" in latest_error["error_message"]
    
    def test_health_check(self):
        """Test health check functionality."""
        # Initially healthy
        health = self.error_handler.get_health_status()
        assert health["status"] == "healthy"
        
        # Add some errors
        for i in range(5):
            error = DataLayerError(f"Error {i}")
            self.error_handler.handle_error(error, ComponentType.CAPTURE, {})
        
        self.error_handler._total_operations = 10  # 50% error rate
        
        health = self.error_handler.get_health_status()
        # Health status should reflect error rate
        assert health["status"] in ["degraded", "unhealthy"]
        assert health["error_rate"] == 0.5


class TestCircuitBreaker:
    """Test cases for CircuitBreaker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            success_threshold=2
        )
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        assert self.circuit_breaker.state == CircuitBreakerState.CLOSED
        assert self.circuit_breaker.failure_count == 0
        assert self.circuit_breaker.success_count == 0
    
    def test_record_success_in_closed_state(self):
        """Test recording success in closed state."""
        self.circuit_breaker.record_success()
        
        assert self.circuit_breaker.state == CircuitBreakerState.CLOSED
        assert self.circuit_breaker.failure_count == 0
    
    def test_record_failure_opens_circuit(self):
        """Test that failures open the circuit."""
        # Record failures up to threshold
        for i in range(3):
            self.circuit_breaker.record_failure()
        
        # Should open circuit
        assert self.circuit_breaker.state == CircuitBreakerState.OPEN
        assert self.circuit_breaker.failure_count == 3
    
    def test_circuit_opens_after_threshold(self):
        """Test circuit opening after failure threshold."""
        # Record failures below threshold
        self.circuit_breaker.record_failure()
        self.circuit_breaker.record_failure()
        assert self.circuit_breaker.state == CircuitBreakerState.CLOSED
        
        # One more failure should open circuit
        self.circuit_breaker.record_failure()
        assert self.circuit_breaker.state == CircuitBreakerState.OPEN
    
    def test_can_execute_in_different_states(self):
        """Test can_execute in different states."""
        # Closed state - should allow execution
        assert self.circuit_breaker.can_execute() is True
        
        # Open circuit
        for i in range(3):
            self.circuit_breaker.record_failure()
        
        # Open state - should not allow execution
        assert self.circuit_breaker.can_execute() is False
    
    def test_circuit_recovery_to_half_open(self):
        """Test circuit recovery to half-open state."""
        # Open the circuit
        for i in range(3):
            self.circuit_breaker.record_failure()
        
        assert self.circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        time.sleep(1.1)  # Slightly more than recovery_timeout
        
        # Should allow execution (half-open)
        assert self.circuit_breaker.can_execute() is True
        assert self.circuit_breaker.state == CircuitBreakerState.HALF_OPEN
    
    def test_half_open_to_closed_recovery(self):
        """Test recovery from half-open to closed."""
        # Get to half-open state
        for i in range(3):
            self.circuit_breaker.record_failure()
        
        time.sleep(1.1)
        self.circuit_breaker.can_execute()  # Transitions to half-open
        
        # Record successful operations
        self.circuit_breaker.record_success()
        self.circuit_breaker.record_success()
        
        # Should close circuit
        assert self.circuit_breaker.state == CircuitBreakerState.CLOSED
        assert self.circuit_breaker.failure_count == 0
    
    def test_half_open_to_open_on_failure(self):
        """Test half-open returning to open on failure."""
        # Get to half-open state
        for i in range(3):
            self.circuit_breaker.record_failure()
        
        time.sleep(1.1)
        self.circuit_breaker.can_execute()  # Transitions to half-open
        
        # Record failure in half-open
        self.circuit_breaker.record_failure()
        
        # Should return to open
        assert self.circuit_breaker.state == CircuitBreakerState.OPEN
        assert not self.circuit_breaker.can_execute()
    
    def test_get_circuit_stats(self):
        """Test getting circuit breaker statistics."""
        # Record some operations
        self.circuit_breaker.record_success()
        self.circuit_breaker.record_failure()
        self.circuit_breaker.record_success()
        
        stats = self.circuit_breaker.get_stats()
        
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 1
        assert stats["success_count"] == 2
        assert stats["failure_threshold"] == 3


class TestResilientOperationDecorator:
    """Test cases for resilient_operation decorator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.call_count = 0
        self.success_after = None
    
    def test_successful_operation(self):
        """Test decorator with successful operation."""
        @resilient_operation(ComponentType.CAPTURE, "test_operation")
        def successful_operation():
            self.call_count += 1
            return "success"
        
        result = successful_operation()
        
        assert result == "success"
        assert self.call_count == 1
    
    def test_operation_with_retries(self):
        """Test decorator with retries on failure."""
        @resilient_operation(ComponentType.CAPTURE, "test_operation", max_retries=2)
        def failing_operation():
            self.call_count += 1
            if self.call_count < 3:  # Fail first 2 times
                raise CaptureError("Temporary failure")
            return "success"
        
        result = failing_operation()
        
        assert result == "success"
        assert self.call_count == 3  # Initial + 2 retries
    
    def test_operation_max_retries_exceeded(self):
        """Test decorator when max retries exceeded."""
        @resilient_operation(ComponentType.CAPTURE, "test_operation", max_retries=2)
        def always_failing_operation():
            self.call_count += 1
            raise CaptureError("Persistent failure")
        
        with pytest.raises(CaptureError):
            always_failing_operation()
        
        assert self.call_count == 3  # Initial + 2 retries
    
    @pytest.mark.asyncio
    async def test_async_operation_with_retries(self):
        """Test decorator with async operations."""
        @resilient_operation(ComponentType.VALIDATION, "async_test", max_retries=1)
        async def async_failing_operation():
            self.call_count += 1
            if self.call_count < 2:
                raise ValidationError("Async failure")
            return "async_success"
        
        result = await async_failing_operation()
        
        assert result == "async_success"
        assert self.call_count == 2
    
    def test_circuit_breaker_integration(self):
        """Test decorator integration with circuit breaker."""
        @resilient_operation(ComponentType.CAPTURE, "test_operation", max_retries=0)
        def operation_with_circuit_breaker():
            self.call_count += 1
            raise CaptureError("Operation failure")
        
        # Should fail multiple times to open circuit
        for i in range(5):
            try:
                operation_with_circuit_breaker()
            except CaptureError:
                pass
        
        # Circuit should be open, preventing further execution
        # This depends on the circuit breaker configuration in the decorator
        assert self.call_count >= 3  # At least threshold calls made
    
    def test_error_callback_invocation(self):
        """Test error callback is invoked on failure."""
        callback_called = False
        callback_error = None
        
        def error_callback(error, context):
            nonlocal callback_called, callback_error
            callback_called = True
            callback_error = error
        
        @resilient_operation(
            ComponentType.REDACTION,
            "test_operation",
            max_retries=0,
            error_callback=error_callback
        )
        def failing_operation():
            raise RedactionError("Test error")
        
        with pytest.raises(RedactionError):
            failing_operation()
        
        assert callback_called
        assert isinstance(callback_error, RedactionError)


class TestGracefulDegradation:
    """Test cases for graceful degradation context manager."""
    
    def test_graceful_degradation_success(self):
        """Test graceful degradation with successful operation."""
        with graceful_degradation(fallback_value="fallback"):
            result = "success"
        
        # Should not use fallback
        assert result == "success"
    
    def test_graceful_degradation_with_exception(self):
        """Test graceful degradation with exception."""
        with graceful_degradation(fallback_value="fallback") as degraded:
            raise ValueError("Test error")
        
        # Should use fallback value
        assert degraded.result == "fallback"
        assert degraded.error_occurred is True
    
    def test_graceful_degradation_with_callback(self):
        """Test graceful degradation with error callback."""
        callback_called = False
        callback_error = None
        
        def error_callback(error):
            nonlocal callback_called, callback_error
            callback_called = True
            callback_error = error
        
        with graceful_degradation(
            fallback_value="fallback",
            error_callback=error_callback
        ) as degraded:
            raise ValueError("Test error")
        
        assert callback_called
        assert isinstance(callback_error, ValueError)
        assert degraded.result == "fallback"
    
    def test_graceful_degradation_with_fallback_function(self):
        """Test graceful degradation with fallback function."""
        def fallback_func():
            return "computed_fallback"
        
        with graceful_degradation(fallback_func=fallback_func) as degraded:
            raise ValueError("Test error")
        
        assert degraded.result == "computed_fallback"
    
    def test_graceful_degradation_no_fallback(self):
        """Test graceful degradation without fallback."""
        with graceful_degradation() as degraded:
            raise ValueError("Test error")
        
        assert degraded.result is None
        assert degraded.error_occurred is True
    
    def test_graceful_degradation_specific_exceptions(self):
        """Test graceful degradation with specific exception types."""
        # Should catch ValueError but not TypeError
        with pytest.raises(TypeError):
            with graceful_degradation(
                fallback_value="fallback",
                exception_types=(ValueError,)
            ):
                raise TypeError("This should not be caught")
    
    def test_graceful_degradation_context_info(self):
        """Test graceful degradation provides error context."""
        with graceful_degradation(fallback_value="fallback") as degraded:
            raise ValueError("Context test error")
        
        assert degraded.error_occurred is True
        assert degraded.exception is not None
        assert isinstance(degraded.exception, ValueError)
        assert "Context test error" in str(degraded.exception)


class TestErrorTypes:
    """Test cases for custom error types."""
    
    def test_data_layer_error(self):
        """Test DataLayerError base class."""
        error = DataLayerError("Test error", {"context": "test"})
        
        assert str(error) == "Test error"
        assert error.context == {"context": "test"}
    
    def test_capture_error(self):
        """Test CaptureError specific error."""
        error = CaptureError("Capture failed", {"url": "https://example.com"})
        
        assert isinstance(error, DataLayerError)
        assert str(error) == "Capture failed"
        assert error.context["url"] == "https://example.com"
    
    def test_validation_error(self):
        """Test ValidationError specific error."""
        error = ValidationError("Validation failed", {"schema": "test_schema"})
        
        assert isinstance(error, DataLayerError)
        assert str(error) == "Validation failed"
        assert error.context["schema"] == "test_schema"
    
    def test_redaction_error(self):
        """Test RedactionError specific error."""
        error = RedactionError("Redaction failed", {"pattern": "test_pattern"})
        
        assert isinstance(error, DataLayerError)
        assert str(error) == "Redaction failed"
        assert error.context["pattern"] == "test_pattern"


class TestErrorHandlingIntegration:
    """Integration tests for error handling system."""
    
    def test_end_to_end_error_handling(self):
        """Test complete error handling workflow."""
        error_handler = DataLayerErrorHandler()
        
        # Simulate various error scenarios
        errors_and_components = [
            (CaptureError("Timeout error"), ComponentType.CAPTURE),
            (ValidationError("Schema error"), ComponentType.VALIDATION),
            (RedactionError("Pattern error"), ComponentType.REDACTION),
            (CaptureError("Network error"), ComponentType.CAPTURE),
        ]
        
        for error, component in errors_and_components:
            error_handler.handle_error(error, component, {"url": "https://example.com"})
        
        # Check comprehensive error tracking
        summary = error_handler.get_error_summary()
        health = error_handler.get_health_status()
        
        assert summary["total_errors"] == 4
        assert len(summary["by_component"]) == 3  # Three different components
        assert health["total_errors"] == 4
        
        # Circuit breakers should be created for each component
        capture_cb = error_handler.get_circuit_breaker(ComponentType.CAPTURE)
        validation_cb = error_handler.get_circuit_breaker(ComponentType.VALIDATION)
        
        assert capture_cb is not None
        assert validation_cb is not None
        assert capture_cb is not validation_cb  # Different instances


if __name__ == "__main__":
    pytest.main([__file__, "-v"])