"""Base classes and interfaces for alert dispatching.

This module provides the foundation for all alert dispatcher implementations,
including abstract base classes, templating support, and common utilities
for alert notification systems.
"""

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..models import RuleResults, Failure, Severity
from ..evaluator import RuleEvaluationResult


class AlertStatus(str, Enum):
    """Status of alert dispatch attempt."""
    PENDING = "pending"
    SENT = "sent" 
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertTrigger(str, Enum):
    """Alert trigger conditions."""
    ANY_FAILURE = "any_failure"
    CRITICAL_FAILURE = "critical_failure"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    CUSTOM_CONDITION = "custom_condition"


@dataclass
class AlertContext:
    """Context information for alert generation."""
    
    # Rule evaluation results
    rule_results: RuleResults
    evaluation_results: List[RuleEvaluationResult]
    
    # Alert configuration
    alert_config: Dict[str, Any]
    trigger_condition: AlertTrigger
    
    # Environment context
    environment: Optional[str] = None
    target_urls: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Alert metadata
    alert_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    @property
    def failed_rules(self) -> List[RuleEvaluationResult]:
        """Get only failed rule results."""
        return [r for r in self.evaluation_results if not r.passed]
    
    @property
    def critical_failures(self) -> List[Failure]:
        """Get only critical severity failures."""
        return [f for f in self.rule_results.failures if f.severity == Severity.CRITICAL]
    
    @property
    def alert_severity(self) -> AlertSeverity:
        """Determine alert severity based on failures."""
        if self.critical_failures:
            return AlertSeverity.CRITICAL
        elif self.rule_results.summary.warning_failures > 0:
            return AlertSeverity.HIGH
        elif self.rule_results.summary.failed_rules > 0:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW


class AlertPayload(BaseModel):
    """Structured alert payload for dispatch."""
    
    # Alert identification
    alert_id: str = Field(description="Unique alert identifier")
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for tracking related alerts"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Alert generation timestamp"
    )
    
    # Alert content
    title: str = Field(description="Alert title/subject")
    message: str = Field(description="Alert message body")
    severity: AlertSeverity = Field(description="Alert severity level")
    
    # Context information
    environment: Optional[str] = Field(
        default=None,
        description="Environment where issues occurred"
    )
    target_urls: List[str] = Field(
        default_factory=list,
        description="URLs that were audited"
    )
    
    # Rule evaluation summary
    total_rules: int = Field(description="Total rules evaluated")
    failed_rules: int = Field(description="Number of failed rules") 
    critical_failures: int = Field(description="Number of critical failures")
    warning_failures: int = Field(description="Number of warning failures")
    
    # Failure details
    failures: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed failure information"
    )
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional alert metadata"
    )


class AlertTemplate:
    """Template engine for alert formatting."""
    
    def __init__(self, template_config: Dict[str, Any]):
        self.templates = template_config.get('templates', {})
        self.formatters = template_config.get('formatters', {})
        self.variables = template_config.get('variables', {})
    
    def format_alert(self, context: AlertContext) -> AlertPayload:
        """Format alert using template configuration.
        
        Args:
            context: Alert context with evaluation results
            
        Returns:
            Formatted alert payload
        """
        # Build template variables
        template_vars = self._build_template_variables(context)
        
        # Format title and message
        title = self._format_template('title', template_vars, context)
        message = self._format_template('message', template_vars, context)
        
        # Create alert payload
        return AlertPayload(
            alert_id=context.alert_id or self._generate_alert_id(context),
            correlation_id=context.correlation_id,
            title=title,
            message=message,
            severity=context.alert_severity,
            environment=context.environment,
            target_urls=context.target_urls,
            total_rules=context.rule_results.summary.total_rules,
            failed_rules=context.rule_results.summary.failed_rules,
            critical_failures=context.rule_results.summary.critical_failures,
            warning_failures=context.rule_results.summary.warning_failures,
            failures=[self._format_failure(f) for f in context.rule_results.failures],
            metadata=self._build_metadata(context)
        )
    
    def _build_template_variables(self, context: AlertContext) -> Dict[str, Any]:
        """Build variables available to templates."""
        variables = {
            # Summary statistics
            'total_rules': context.rule_results.summary.total_rules,
            'passed_rules': context.rule_results.summary.passed_rules,
            'failed_rules': context.rule_results.summary.failed_rules,
            'total_failures': context.rule_results.summary.total_failures,
            'critical_failures': context.rule_results.summary.critical_failures,
            'warning_failures': context.rule_results.summary.warning_failures,
            'info_failures': context.rule_results.summary.info_failures,
            
            # Context information
            'environment': context.environment or 'unknown',
            'target_urls': context.target_urls,
            'timestamp': context.timestamp.isoformat(),
            'alert_severity': context.alert_severity.value,
            
            # Failure details
            'failures': context.rule_results.failures,
            'failed_rule_results': context.failed_rules,
            
            # Execution info
            'execution_time_ms': context.rule_results.summary.execution_time_ms,
            'evaluation_time': context.rule_results.evaluation_time.isoformat(),
        }
        
        # Add custom variables
        variables.update(self.variables)
        
        return variables
    
    def _format_template(self, template_name: str, variables: Dict[str, Any], context: AlertContext) -> str:
        """Format a specific template with variables."""
        template = self.templates.get(template_name, '')
        
        if not template:
            # Provide default templates
            if template_name == 'title':
                template = self._get_default_title_template(context)
            elif template_name == 'message':
                template = self._get_default_message_template(context)
        
        # Simple variable substitution
        formatted = template
        for var_name, var_value in variables.items():
            placeholder = f'{{{var_name}}}'
            if placeholder in formatted:
                formatted = formatted.replace(placeholder, str(var_value))
        
        return formatted
    
    def _get_default_title_template(self, context: AlertContext) -> str:
        """Get default title template."""
        if context.alert_severity == AlertSeverity.CRITICAL:
            return "🚨 CRITICAL: {failed_rules} rules failed in {environment}"
        elif context.alert_severity == AlertSeverity.HIGH:
            return "⚠️  WARNING: {failed_rules} rules failed in {environment}"
        else:
            return "ℹ️  INFO: {failed_rules} rules failed in {environment}"
    
    def _get_default_message_template(self, context: AlertContext) -> str:
        """Get default message template."""
        return """Rule evaluation completed with {failed_rules} failures.

Environment: {environment}
Total Rules: {total_rules}
Failed Rules: {failed_rules}
Critical Failures: {critical_failures}
Warning Failures: {warning_failures}

Evaluation completed at: {evaluation_time}
Execution time: {execution_time_ms}ms

Target URLs:
{target_urls}

For detailed failure information, please check the audit system."""
    
    def _format_failure(self, failure: Failure) -> Dict[str, Any]:
        """Format failure for inclusion in alert."""
        return {
            'check_id': failure.check_id,
            'severity': failure.severity.value,
            'message': failure.message,
            'details': failure.details,
            'evidence_count': len(failure.evidence) if failure.evidence else 0
        }
    
    def _build_metadata(self, context: AlertContext) -> Dict[str, Any]:
        """Build additional metadata for alert."""
        return {
            'trigger_condition': context.trigger_condition.value,
            'rule_count_by_severity': {
                'critical': context.rule_results.summary.critical_failures,
                'warning': context.rule_results.summary.warning_failures,
                'info': context.rule_results.summary.info_failures
            },
            'context_info': context.rule_results.context_info
        }
    
    def _generate_alert_id(self, context: AlertContext) -> str:
        """Generate unique alert ID."""
        import uuid
        base_id = f"{context.environment or 'unknown'}_{context.timestamp.strftime('%Y%m%d_%H%M%S')}"
        unique_id = str(uuid.uuid4())[:8]
        return f"{base_id}_{unique_id}"


class AlertDispatchResult(BaseModel):
    """Result of alert dispatch attempt."""
    
    dispatcher_type: str = Field(description="Type of dispatcher used")
    alert_id: str = Field(description="Alert identifier")
    status: AlertStatus = Field(description="Dispatch status")
    
    # Timing information
    dispatch_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="When dispatch was attempted"
    )
    response_time_ms: Optional[float] = Field(
        default=None,
        description="Response time in milliseconds"
    )
    
    # Result details
    success: bool = Field(description="Whether dispatch succeeded")
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if dispatch failed"
    )
    response_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Response data from dispatch endpoint"
    )
    
    # Retry information
    attempt_number: int = Field(default=1, description="Attempt number")
    retry_after: Optional[datetime] = Field(
        default=None,
        description="When to retry if failed"
    )


class BaseAlertDispatcher(ABC):
    """Abstract base class for alert dispatchers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dispatcher_type = self.__class__.__name__
        self.template = AlertTemplate(config.get('template', {}))
        self.enabled = config.get('enabled', True)
        
        # Retry configuration
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay_seconds = config.get('retry_delay_seconds', 60)
        
        # Filtering configuration
        self.severity_filter = set(config.get('severity_filter', []))
        self.environment_filter = set(config.get('environment_filter', []))
        
        # Rate limiting
        self.rate_limit_enabled = config.get('rate_limit_enabled', False)
        self.rate_limit_requests = config.get('rate_limit_requests', 10)
        self.rate_limit_window_seconds = config.get('rate_limit_window_seconds', 60)
    
    @abstractmethod
    async def dispatch(self, context: AlertContext) -> AlertDispatchResult:
        """Dispatch alert using specific implementation.
        
        Args:
            context: Alert context with evaluation results
            
        Returns:
            Dispatch result with status and details
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> List[str]:
        """Validate dispatcher configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        pass
    
    def should_dispatch(self, context: AlertContext) -> bool:
        """Check if alert should be dispatched based on filters.
        
        Args:
            context: Alert context to evaluate
            
        Returns:
            True if alert should be dispatched
        """
        if not self.enabled:
            return False
        
        # Check severity filter
        if self.severity_filter and context.alert_severity not in self.severity_filter:
            return False
        
        # Check environment filter
        if (self.environment_filter and 
            context.environment and 
            context.environment not in self.environment_filter):
            return False
        
        # Check trigger condition
        return self._evaluate_trigger_condition(context)
    
    def _evaluate_trigger_condition(self, context: AlertContext) -> bool:
        """Evaluate if trigger condition is met."""
        trigger = context.trigger_condition
        
        if trigger == AlertTrigger.ANY_FAILURE:
            return context.rule_results.summary.failed_rules > 0
        elif trigger == AlertTrigger.CRITICAL_FAILURE:
            return context.rule_results.summary.critical_failures > 0
        elif trigger == AlertTrigger.THRESHOLD_EXCEEDED:
            threshold = context.alert_config.get('failure_threshold', 1)
            return context.rule_results.summary.failed_rules >= threshold
        elif trigger == AlertTrigger.CUSTOM_CONDITION:
            # Custom condition evaluation would be implemented here
            return True
        
        return False
    
    def format_alert(self, context: AlertContext) -> AlertPayload:
        """Format alert using configured template.
        
        Args:
            context: Alert context with evaluation results
            
        Returns:
            Formatted alert payload
        """
        return self.template.format_alert(context)
    
    def _create_dispatch_result(
        self,
        alert_id: str,
        success: bool,
        response_time_ms: Optional[float] = None,
        error_message: Optional[str] = None,
        response_data: Optional[Dict[str, Any]] = None,
        attempt_number: int = 1
    ) -> AlertDispatchResult:
        """Create dispatch result helper."""
        return AlertDispatchResult(
            dispatcher_type=self.dispatcher_type,
            alert_id=alert_id,
            status=AlertStatus.SENT if success else AlertStatus.FAILED,
            success=success,
            response_time_ms=response_time_ms,
            error_message=error_message,
            response_data=response_data,
            attempt_number=attempt_number
        )


class AlertDispatcherRegistry:
    """Registry for managing alert dispatcher types."""
    
    def __init__(self):
        self._dispatchers: Dict[str, type] = {}
    
    def register(self, dispatcher_type: str, dispatcher_class: type) -> None:
        """Register an alert dispatcher class.
        
        Args:
            dispatcher_type: Type identifier for the dispatcher
            dispatcher_class: Dispatcher class to register
        """
        if not issubclass(dispatcher_class, BaseAlertDispatcher):
            raise ValueError(f"Dispatcher class must inherit from BaseAlertDispatcher: {dispatcher_class}")
        
        self._dispatchers[dispatcher_type] = dispatcher_class
    
    def get_dispatcher_class(self, dispatcher_type: str) -> Optional[type]:
        """Get registered dispatcher class.
        
        Args:
            dispatcher_type: Dispatcher type identifier
            
        Returns:
            Dispatcher class or None if not found
        """
        return self._dispatchers.get(dispatcher_type)
    
    def create_dispatcher(self, dispatcher_type: str, config: Dict[str, Any]) -> BaseAlertDispatcher:
        """Create dispatcher instance.
        
        Args:
            dispatcher_type: Type of dispatcher to create
            config: Configuration for the dispatcher
            
        Returns:
            Dispatcher instance
            
        Raises:
            ValueError: If dispatcher type is not registered
        """
        dispatcher_class = self.get_dispatcher_class(dispatcher_type)
        if not dispatcher_class:
            raise ValueError(f"Unknown dispatcher type: {dispatcher_type}")
        
        return dispatcher_class(config)
    
    def list_dispatcher_types(self) -> List[str]:
        """List all registered dispatcher types."""
        return list(self._dispatchers.keys())


# Global dispatcher registry
dispatcher_registry = AlertDispatcherRegistry()


def register_dispatcher(dispatcher_type: str):
    """Decorator to register an alert dispatcher class.
    
    Args:
        dispatcher_type: Type identifier for the dispatcher
    """
    def decorator(dispatcher_class: type) -> type:
        dispatcher_registry.register(dispatcher_type, dispatcher_class)
        return dispatcher_class
    
    return decorator