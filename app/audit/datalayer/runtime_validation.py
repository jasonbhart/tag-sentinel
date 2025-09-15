"""Runtime type validation utilities for DataLayer integrity system.

This module provides decorators and utilities for runtime type validation
to complement static type hints and Pydantic model validation.
"""

import functools
import inspect
import sys
from typing import Any, Dict, List, Union, get_type_hints, get_origin, get_args
from collections.abc import Callable

from .models import (
    DataLayerSnapshot,
    DLContext,
    DLResult,
    ValidationIssue,
    VariablePresence,
    EventFrequency,
    DLAggregate,
    RedactionMethod,
    ValidationSeverity
)


class TypeValidationError(Exception):
    """Raised when runtime type validation fails."""
    
    def __init__(self, param_name: str, expected_type: Any, actual_type: type, actual_value: Any = None):
        self.param_name = param_name
        self.expected_type = expected_type
        self.actual_type = actual_type
        self.actual_value = actual_value
        
        super().__init__(
            f"Parameter '{param_name}' expected {self._format_type(expected_type)}, "
            f"got {actual_type.__name__}"
        )
    
    def _format_type(self, type_hint: Any) -> str:
        """Format type hint for human-readable error message."""
        if hasattr(type_hint, '__name__'):
            return type_hint.__name__
        elif str(type_hint).startswith('typing.'):
            return str(type_hint).replace('typing.', '')
        else:
            return str(type_hint)


def validate_types(*, enable_return_validation: bool = True) -> Callable:
    """Decorator for runtime type validation of function parameters and return values.

    Args:
        enable_return_validation: Whether to validate return type (default: True)

    Usage:
        @validate_types()
        def process_snapshot(snapshot: DataLayerSnapshot, config: DLContext) -> DLResult:
            # Function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        # Get type hints once at decoration time
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Bind arguments to parameters
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Validate each parameter
                for param_name, value in bound_args.arguments.items():
                    if param_name in type_hints:
                        expected_type = type_hints[param_name]
                        _validate_parameter(param_name, value, expected_type)

                # Call original function and await result
                result = await func(*args, **kwargs)

                # Validate return type if enabled and type hint exists
                if enable_return_validation and 'return' in type_hints:
                    return_type = type_hints['return']
                    _validate_parameter('return', result, return_type)

                return result

            return async_wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Bind arguments to parameters
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Validate each parameter
                for param_name, value in bound_args.arguments.items():
                    if param_name in type_hints:
                        expected_type = type_hints[param_name]
                        _validate_parameter(param_name, value, expected_type)

                # Call original function
                result = func(*args, **kwargs)

                # Validate return type if enabled and type hint exists
                if enable_return_validation and 'return' in type_hints:
                    return_type = type_hints['return']
                    _validate_parameter('return', result, return_type)

                return result

            return wrapper
    return decorator


def _validate_parameter(param_name: str, value: Any, expected_type: Any) -> None:
    """Validate a single parameter against its expected type."""
    if value is None:
        # Check if None is allowed (Union with None or Optional)
        if _is_optional_type(expected_type):
            return
        else:
            raise TypeValidationError(param_name, expected_type, type(None), value)
    
    # Handle Union types (including Optional)
    if _is_union_type(expected_type):
        union_args = get_args(expected_type)
        if any(_check_type_match(value, arg) for arg in union_args):
            return
        raise TypeValidationError(param_name, expected_type, type(value), value)
    
    # Handle List types
    if _is_list_type(expected_type):
        if not isinstance(value, list):
            raise TypeValidationError(param_name, expected_type, type(value), value)
        
        # Validate list element types if specified
        list_args = get_args(expected_type)
        if list_args:
            element_type = list_args[0]
            for i, item in enumerate(value):
                try:
                    _validate_parameter(f"{param_name}[{i}]", item, element_type)
                except TypeValidationError as e:
                    # Re-raise with more specific context
                    raise TypeValidationError(
                        f"{param_name}[{i}]", element_type, type(item), item
                    ) from e
        return
    
    # Handle Dict types
    if _is_dict_type(expected_type):
        if not isinstance(value, dict):
            raise TypeValidationError(param_name, expected_type, type(value), value)
        
        # Validate key/value types if specified
        dict_args = get_args(expected_type)
        if len(dict_args) >= 2:
            key_type, value_type = dict_args[0], dict_args[1]
            for k, v in value.items():
                _validate_parameter(f"{param_name} key", k, key_type)
                _validate_parameter(f"{param_name}[{k}]", v, value_type)
        return
    
    # Handle direct type checks
    if not _check_type_match(value, expected_type):
        raise TypeValidationError(param_name, expected_type, type(value), value)


def _check_type_match(value: Any, expected_type: Any) -> bool:
    """Check if a value matches an expected type."""
    # Handle Any type first (can't use isinstance with typing.Any)
    if expected_type is Any:
        return True

    # Handle basic types
    if isinstance(expected_type, type):
        return isinstance(value, expected_type)

    # Handle string type annotations (for forward references)
    if isinstance(expected_type, str):
        # For now, just check against known model names
        model_mapping = {
            'DataLayerSnapshot': DataLayerSnapshot,
            'DLContext': DLContext,
            'DLResult': DLResult,
            'ValidationIssue': ValidationIssue,
            'VariablePresence': VariablePresence,
            'EventFrequency': EventFrequency,
            'DLAggregate': DLAggregate,
            'RedactionMethod': RedactionMethod,
            'ValidationSeverity': ValidationSeverity,
        }
        if expected_type in model_mapping:
            return isinstance(value, model_mapping[expected_type])

    # For other complex types, be permissive
    return True


def _is_union_type(type_hint: Any) -> bool:
    """Check if type hint is a Union type."""
    origin = get_origin(type_hint)
    if origin is Union:
        return True
    
    # Check for Python 3.10+ union syntax (X | Y)
    if sys.version_info >= (3, 10):
        import types
        if hasattr(types, 'UnionType') and isinstance(type_hint, types.UnionType):
            return True
    
    # Check string representation for union types
    type_str = str(type_hint)
    return 'Union[' in type_str or ' | ' in type_str


def _is_optional_type(type_hint: Any) -> bool:
    """Check if type hint is Optional (Union with None)."""
    origin = get_origin(type_hint)
    
    # Handle standard Union[X, None] syntax
    if origin is Union:
        args = get_args(type_hint)
        return type(None) in args
    
    # Handle Python 3.10+ union syntax (X | None)
    if sys.version_info >= (3, 10):
        import types
        if hasattr(types, 'UnionType') and isinstance(type_hint, types.UnionType):
            args = get_args(type_hint)
            return type(None) in args
    
    # Fallback: check string representation
    type_str = str(type_hint)
    return 'None' in type_str and ('Union[' in type_str or ' | ' in type_str)
    
    return False


def _is_list_type(type_hint: Any) -> bool:
    """Check if type hint is a List type."""
    origin = get_origin(type_hint)
    return origin is list or origin is List


def _is_dict_type(type_hint: Any) -> bool:
    """Check if type hint is a Dict type."""
    origin = get_origin(type_hint)
    return origin is dict or origin is Dict


def validate_datalayer_snapshot(snapshot: Any) -> DataLayerSnapshot:
    """Validate and convert input to DataLayerSnapshot with comprehensive checks.
    
    Args:
        snapshot: Input that should be a DataLayerSnapshot
    
    Returns:
        DataLayerSnapshot: Validated snapshot
        
    Raises:
        TypeValidationError: If validation fails
        ValueError: If required fields are missing or invalid
    """
    if not isinstance(snapshot, DataLayerSnapshot):
        raise TypeValidationError('snapshot', DataLayerSnapshot, type(snapshot), snapshot)
    
    # Additional business logic validation
    if not snapshot.page_url:
        raise ValueError("DataLayerSnapshot.page_url cannot be empty")
    
    if snapshot.depth_reached and snapshot.depth_reached < 1:
        raise ValueError("DataLayerSnapshot.depth_reached must be positive")
    
    if snapshot.entries_captured < 0:
        raise ValueError("DataLayerSnapshot.entries_captured cannot be negative")
    
    return snapshot


def validate_dl_context(context: Any) -> DLContext:
    """Validate and convert input to DLContext with comprehensive checks.
    
    Args:
        context: Input that should be a DLContext
    
    Returns:
        DLContext: Validated context
        
    Raises:
        TypeValidationError: If validation fails
        ValueError: If configuration is invalid
    """
    if not isinstance(context, DLContext):
        raise TypeValidationError('context', DLContext, type(context), context)
    
    # Additional business logic validation
    if not context.env:
        raise ValueError("DLContext.env cannot be empty")
    
    if not context.data_layer_object:
        raise ValueError("DLContext.data_layer_object cannot be empty")
    
    if context.max_depth < 1 or context.max_depth > 20:
        raise ValueError("DLContext.max_depth must be between 1 and 20")
    
    if context.max_entries < 1:
        raise ValueError("DLContext.max_entries must be positive")
    
    if context.max_size_bytes and context.max_size_bytes < 1024:
        raise ValueError("DLContext.max_size_bytes must be at least 1024 bytes")
    
    return context


def validate_validation_issues(issues: Any) -> List[ValidationIssue]:
    """Validate list of ValidationIssue objects.
    
    Args:
        issues: Input that should be a list of ValidationIssue objects
    
    Returns:
        List[ValidationIssue]: Validated issues list
        
    Raises:
        TypeValidationError: If validation fails
    """
    if not isinstance(issues, list):
        raise TypeValidationError('issues', List[ValidationIssue], type(issues), issues)
    
    validated_issues = []
    for i, issue in enumerate(issues):
        if not isinstance(issue, ValidationIssue):
            raise TypeValidationError(
                f'issues[{i}]', ValidationIssue, type(issue), issue
            )
        
        # Additional validation
        if not issue.message.strip():
            raise ValueError(f"ValidationIssue[{i}].message cannot be empty")
        
        if not issue.path.strip():
            raise ValueError(f"ValidationIssue[{i}].path cannot be empty")
        
        validated_issues.append(issue)
    
    return validated_issues


# Convenience decorators for common DataLayer types
def validate_snapshot_input(func: Callable) -> Callable:
    """Decorator to validate DataLayerSnapshot input parameter."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Assume first argument is snapshot
        if args:
            args = list(args)
            args[0] = validate_datalayer_snapshot(args[0])
        elif 'snapshot' in kwargs:
            kwargs['snapshot'] = validate_datalayer_snapshot(kwargs['snapshot'])
        
        return func(*args, **kwargs)
    return wrapper


def validate_context_input(func: Callable) -> Callable:
    """Decorator to validate DLContext input parameter."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Look for context parameter
        if 'context' in kwargs:
            kwargs['context'] = validate_dl_context(kwargs['context'])
        elif len(args) >= 2:  # Assume second argument is context
            args = list(args)
            args[1] = validate_dl_context(args[1])
        
        return func(*args, **kwargs)
    return wrapper


def safe_execute(func: Callable, *args, **kwargs) -> tuple[Any, Exception | None]:
    """Safely execute a function with type validation, catching any validation errors.
    
    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        tuple: (result, error) where error is None if successful
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except (TypeValidationError, ValueError, TypeError) as e:
        return None, e
    except Exception as e:
        # Re-raise unexpected errors
        raise e


# Type validation utilities for common patterns
class ValidationContext:
    """Context manager for collecting validation errors instead of raising immediately."""
    
    def __init__(self):
        self.errors: List[Exception] = []
        self._suppress_errors = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is TypeValidationError and self._suppress_errors:
            self.errors.append(exc_val)
            return True  # Suppress the exception
        return False
    
    def validate(self, param_name: str, value: Any, expected_type: Any) -> None:
        """Validate a parameter and collect any errors."""
        try:
            _validate_parameter(param_name, value, expected_type)
        except TypeValidationError as e:
            self.errors.append(e)
    
    def has_errors(self) -> bool:
        """Check if any validation errors were collected."""
        return len(self.errors) > 0
    
    def get_error_summary(self) -> str:
        """Get a summary of all validation errors."""
        if not self.errors:
            return "No validation errors"
        
        error_messages = [str(error) for error in self.errors]
        return f"Validation errors ({len(self.errors)}): " + "; ".join(error_messages)