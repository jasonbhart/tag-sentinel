"""Base classes and interfaces for rule checks.

This module provides the foundation for all rule check implementations,
including abstract base classes, common utilities, and result structures.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Type
from enum import Enum

from pydantic import BaseModel, Field

from ..indexing import AuditIndexes, AuditQuery
from ..models import Severity, Failure, RuleResults


class CheckResult(BaseModel):
    """Result of a single check execution."""
    
    check_id: str = Field(description="Unique identifier for the check")
    check_name: str = Field(description="Human-readable name of the check")
    passed: bool = Field(description="Whether the check passed")
    severity: Severity = Field(description="Severity level of the check")
    
    # Result details
    message: str = Field(description="Summary message describing the result")
    details: Optional[str] = Field(
        default=None,
        description="Additional details about the result"
    )
    
    # Metrics and evidence
    found_count: int = Field(
        default=0,
        description="Number of items found by the check"
    )
    expected_count: Optional[int] = Field(
        default=None,
        description="Expected number of items (for count checks)"
    )
    evidence: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Evidence supporting the result"
    )
    
    # Execution metadata
    execution_time_ms: Optional[float] = Field(
        default=None,
        description="Time taken to execute the check in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the check was executed"
    )
    
    def to_failure(self) -> Optional[Failure]:
        """Convert to Failure if check failed."""
        if self.passed:
            return None
            
        return Failure(
            check_id=self.check_id,
            severity=self.severity,
            message=self.message,
            details=self.details,
            evidence=self.evidence
        )


class CheckContext(BaseModel):
    """Context information available to checks during execution."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    # Data access
    indexes: AuditIndexes = Field(description="Indexed audit data")
    query: AuditQuery = Field(description="Query interface for the data")
    
    # Rule configuration
    rule_id: str = Field(description="ID of the rule being executed")
    rule_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration parameters for the rule"
    )
    check_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration parameters for the specific check"
    )
    
    # Environment context
    environment: Optional[str] = Field(
        default=None,
        description="Current environment (dev, staging, prod, etc.)"
    )
    target_urls: List[str] = Field(
        default_factory=list,
        description="URLs that were audited"
    )
    
    # Execution context
    debug: bool = Field(
        default=False,
        description="Whether debug information should be collected"
    )
    timeout_ms: int = Field(
        default=30000,
        description="Maximum execution time for the check in milliseconds"
    )


class BaseCheck(ABC):
    """Abstract base class for all rule checks.
    
    All check implementations must inherit from this class and implement
    the execute method to perform their specific validation logic.
    """
    
    def __init__(self, check_id: str, name: str, description: str):
        self.check_id = check_id
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute the check and return the result.
        
        Args:
            context: Execution context with access to audit data and configuration
            
        Returns:
            CheckResult containing the outcome of the check
        """
        pass
    
    @abstractmethod
    def get_supported_config_keys(self) -> List[str]:
        """Get list of configuration keys supported by this check.
        
        Returns:
            List of configuration parameter names this check understands
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate check configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        supported_keys = set(self.get_supported_config_keys())
        provided_keys = set(config.keys())
        
        # Check for unsupported keys
        unsupported = provided_keys - supported_keys
        if unsupported:
            errors.append(f"Unsupported configuration keys: {', '.join(unsupported)}")
        
        return errors
    
    def _create_result(
        self,
        context: CheckContext,
        passed: bool,
        message: str,
        details: Optional[str] = None,
        found_count: int = 0,
        expected_count: Optional[int] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        severity: Optional[Severity] = None
    ) -> CheckResult:
        """Helper method to create a CheckResult."""
        return CheckResult(
            check_id=self.check_id,
            check_name=self.name,
            passed=passed,
            severity=severity or self._determine_severity(context),
            message=message,
            details=details,
            found_count=found_count,
            expected_count=expected_count,
            evidence=evidence or []
        )
    
    def _determine_severity(self, context: CheckContext) -> Severity:
        """Determine severity from context or use default."""
        # Try to get from rule config first, then check config, then default
        severity_str = (
            context.rule_config.get('severity') or
            context.check_config.get('severity') or
            'warning'
        )
        
        if isinstance(severity_str, str):
            try:
                return Severity(severity_str.lower())
            except ValueError:
                return Severity.WARNING
        elif isinstance(severity_str, Severity):
            return severity_str
        else:
            return Severity.WARNING
    
    def _extract_evidence(self, items: List[Any], max_items: int = 10) -> List[Dict[str, Any]]:
        """Extract evidence from found items for reporting.
        
        Args:
            items: List of items to extract evidence from
            max_items: Maximum number of items to include in evidence
            
        Returns:
            List of evidence dictionaries
        """
        evidence = []
        for item in items[:max_items]:
            if hasattr(item, 'dict'):
                # Pydantic model
                evidence.append(item.dict())
            elif hasattr(item, '__dict__'):
                # Regular object with __dict__
                evidence.append(item.__dict__)
            elif isinstance(item, dict):
                # Already a dictionary
                evidence.append(item)
            else:
                # Convert to string representation
                evidence.append({'value': str(item), 'type': type(item).__name__})
        
        return evidence


class CheckRegistry:
    """Registry for managing available check types."""
    
    def __init__(self):
        self._checks: Dict[str, Type[BaseCheck]] = {}
    
    def register(self, check_type: str, check_class: Type[BaseCheck]) -> None:
        """Register a check class.
        
        Args:
            check_type: Identifier for the check type
            check_class: Check class to register
        """
        if not issubclass(check_class, BaseCheck):
            raise ValueError(f"Check class must inherit from BaseCheck: {check_class}")
        
        self._checks[check_type] = check_class
    
    def get_check_class(self, check_type: str) -> Optional[Type[BaseCheck]]:
        """Get a registered check class.
        
        Args:
            check_type: Check type identifier
            
        Returns:
            Check class or None if not found
        """
        return self._checks.get(check_type)
    
    def list_check_types(self) -> List[str]:
        """List all registered check types.
        
        Returns:
            List of check type identifiers
        """
        return list(self._checks.keys())
    
    def create_check(
        self,
        check_type: str,
        check_id: str,
        name: str,
        description: str,
        **kwargs
    ) -> BaseCheck:
        """Create an instance of a check.
        
        Args:
            check_type: Type of check to create
            check_id: Unique identifier for the check instance
            name: Human-readable name
            description: Check description
            **kwargs: Additional arguments passed to the check constructor
            
        Returns:
            Check instance
            
        Raises:
            ValueError: If check type is not registered
        """
        check_class = self.get_check_class(check_type)
        if not check_class:
            raise ValueError(f"Unknown check type: {check_type}")
        
        return check_class(check_id, name, description, **kwargs)


# Global check registry
check_registry = CheckRegistry()


def register_check(check_type: str):
    """Decorator to register a check class.
    
    Args:
        check_type: Identifier for the check type
    """
    def decorator(check_class: Type[BaseCheck]) -> Type[BaseCheck]:
        check_registry.register(check_type, check_class)
        return check_class
    
    return decorator