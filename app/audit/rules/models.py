"""Pydantic models for the Rule Engine system.

This module defines the core data models used by the Rules & Alerts system,
including rule definitions, evaluation results, failures, and alerting.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, AnyHttpUrl


class Severity(str, Enum):
    """Rule severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RuleScope(str, Enum):
    """Scope of rule evaluation."""
    PAGE = "page"      # Rule applies to individual pages
    RUN = "run"        # Rule applies to entire audit run


class CheckType(str, Enum):
    """Types of rule checks available."""
    # Concrete, registered check types
    REQUEST_PRESENT = "request_present"
    REQUEST_ABSENT = "request_absent"
    SCRIPT_PRESENT = "script_present"
    TAG_EVENT_PRESENT = "tag_event_present"
    COOKIE_PRESENT = "cookie_present"
    CONSOLE_MESSAGE_PRESENT = "console_message_present"
    DUPLICATE_REQUESTS = "duplicate_requests"
    SEQUENCE_ORDER = "sequence_order"
    RELATIVE_ORDER = "relative_order"  # Keep for backward compatibility
    COOKIE_POLICY = "cookie_policy"
    EXPRESSION = "expression"
    JSONPATH = "jsonpath"


class AlertChannelType(str, Enum):
    """Alert notification channel types."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"


class AppliesTo(BaseModel):
    """Rule scoping and filtering configuration."""
    
    scope: RuleScope = Field(
        default=RuleScope.PAGE,
        description="Scope level where rule applies"
    )
    environments: List[str] = Field(
        default_factory=list,
        description="Environments where rule applies (empty = all)"
    )
    scenario_ids: List[str] = Field(
        default_factory=list,
        description="Scenario IDs where rule applies (empty = all)"
    )
    url_include: List[str] = Field(
        default_factory=list,
        description="URL regex patterns to include (empty = all)"
    )
    url_exclude: List[str] = Field(
        default_factory=list,
        description="URL regex patterns to exclude"
    )
    urls: List[str] = Field(
        default_factory=list,
        description="Specific URLs where rule applies (empty = all)"
    )
    vendors: List[str] = Field(
        default_factory=list,
        description="Vendor/tag types to include (empty = all)"
    )

    @property
    def is_unrestricted(self) -> bool:
        """Check if rule applies to all contexts (no restrictions)."""
        return not any([
            self.environments,
            self.scenario_ids,
            self.url_include,
            self.url_exclude,
            self.urls,
            self.vendors
        ])


class CheckConfig(BaseModel):
    """Base configuration for rule checks."""
    
    type: CheckType = Field(description="Type of check to perform")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Check parameters as dictionary"
    )
    
    # Common check parameters
    vendor: Optional[str] = Field(
        default=None,
        description="Vendor/tag type to check (e.g., 'ga4', 'gtm')"
    )
    url_pattern: Optional[str] = Field(
        default=None,
        description="URL regex pattern to match"
    )
    
    # Threshold parameters
    min_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum required count"
    )
    max_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum allowed count"
    )
    
    # Time-based parameters
    time_window_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Time window in milliseconds for temporal checks"
    )
    timeout_seconds: int = Field(
        default=30,
        ge=0,
        description="Timeout in seconds for check execution"
    )
    
    # Retry configuration
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of retries for check execution"
    )
    
    # Check enable/disable
    enabled: bool = Field(
        default=True,
        description="Whether the check is enabled"
    )
    
    # Expression-based parameters
    expression: Optional[str] = Field(
        default=None,
        description="Expression to evaluate (for expression checks)"
    )
    
    # Additional configuration
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional check-specific configuration"
    )


class Rule(BaseModel):
    """Complete rule definition."""
    
    id: str = Field(description="Unique rule identifier")
    name: str = Field(description="Human-readable rule name")
    description: Optional[str] = Field(
        default=None,
        description="Detailed rule description"
    )
    
    # Rule behavior
    severity: Severity = Field(description="Rule severity level")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    
    # Rule scoping
    applies_to: AppliesTo = Field(
        default_factory=AppliesTo,
        description="Rule scoping configuration"
    )
    
    # Rule check configuration
    check: CheckConfig = Field(description="Check configuration")
    
    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When rule was created"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="When rule was last updated"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Rule tags for organization"
    )

    @field_validator('id')
    @classmethod
    def validate_rule_id(cls, v):
        """Validate rule ID format."""
        if not v or not isinstance(v, str):
            raise ValueError("Rule ID must be a non-empty string")
        
        # Check for valid identifier pattern
        import re
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', v):
            raise ValueError("Rule ID must start with letter and contain only letters, numbers, hyphens, and underscores")
        
        return v

    @property
    def is_critical(self) -> bool:
        """Check if rule has critical severity."""
        return self.severity == Severity.CRITICAL

    @property
    def scope(self) -> RuleScope:
        """Determine rule scope based on check type."""
        # Run-scoped checks operate across all pages in a crawl
        run_scoped_checks = {
            # Duplicate detection needs to compare across pages
            "duplicate_requests", "request_duplicates", "event_duplicates", "cookie_duplicates",
            # Temporal sequence checks that span pages
            "relative_order", "relative_timing", "sequence_order",
        }

        if self.check.type in run_scoped_checks:
            return RuleScope.RUN

        return RuleScope.PAGE


class Failure(BaseModel):
    """Rule evaluation failure."""
    
    # Primary identifiers
    check_id: str = Field(description="ID of check/rule that failed")
    severity: Severity = Field(description="Severity level of failure")
    
    # Failure details  
    message: str = Field(description="Human-readable failure message")
    details: Optional[str] = Field(default=None, description="Additional failure details")
    evidence: Optional[List[Any]] = Field(
        default=None,
        description="Evidence data supporting the failure"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Context information about the failure"
    )
    
    # Scope and location (optional for backward compatibility)
    scope: Optional[RuleScope] = Field(default=None, description="Scope where failure occurred")
    page_url: Optional[str] = Field(
        default=None,
        description="Page URL where failure occurred (for page-scoped failures)"
    )
    
    # UI integration
    links: Dict[str, str] = Field(
        default_factory=dict,
        description="Deep links to UI for investigation"
    )
    
    # Timing
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When failure was detected"
    )

    @field_validator('page_url', mode='before')
    @classmethod
    def validate_page_url(cls, v):
        """Validate page URL format if provided."""
        if v is None:
            return v
        
        if isinstance(v, str) and v:
            try:
                result = urlparse(v)
                if not result.scheme or not result.netloc:
                    raise ValueError("Invalid URL format")
            except Exception as e:
                raise ValueError(f"Invalid page URL: {e}")
        
        return v

    @property
    def is_critical(self) -> bool:
        """Check if failure is critical."""
        return self.severity == Severity.CRITICAL

    @property
    def host(self) -> Optional[str]:
        """Extract host from page URL if available."""
        if self.page_url:
            return urlparse(self.page_url).netloc
        return None


class RuleSummary(BaseModel):
    """Summary of rule evaluation results."""
    
    # Core counts
    total_rules: int = Field(default=0, description="Total number of rules evaluated")
    passed_rules: int = Field(default=0, description="Number of rules that passed")
    failed_rules: int = Field(default=0, description="Number of rules that failed")
    enabled_rules: int = Field(default=0, description="Number of enabled rules")
    
    # Failure counts by severity
    total_failures: int = Field(default=0, description="Total number of failures")
    info_failures: int = Field(default=0, description="Number of info-level failures")
    warning_failures: int = Field(default=0, description="Number of warning-level failures")
    critical_failures: int = Field(default=0, description="Number of critical-level failures")
    
    # Scope breakdown
    page_failures: int = Field(default=0, description="Page-scoped failures")
    run_failures: int = Field(default=0, description="Run-scoped failures")
    
    # Execution metadata
    execution_time_ms: float = Field(default=0.0, description="Total evaluation time in milliseconds")
    indeterminate_rules: int = Field(
        default=0,
        description="Rules that could not be evaluated due to missing data"
    )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_rules == 0:
            return 0.0
        return (self.passed_rules / self.total_rules) * 100.0
    
    @property 
    def failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        if self.total_rules == 0:
            return 0.0
        return (self.failed_rules / self.total_rules) * 100.0

    @property
    def has_critical_failures(self) -> bool:
        """Check if there are any critical failures."""
        return self.critical_failures > 0

    @property
    def has_failures(self) -> bool:
        """Check if there are any failures at all."""
        return self.total_failures > 0



class RuleResults(BaseModel):
    """Complete rule evaluation results for a run."""
    
    # Run identification
    run_id: Optional[str] = Field(default=None, description="Unique run identifier")
    evaluation_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="When evaluation was performed"
    )
    
    # Results
    summary: RuleSummary = Field(
        default_factory=RuleSummary,
        description="Evaluation summary"
    )
    failures: List[Failure] = Field(
        default_factory=list,
        description="All rule failures detected"
    )
    
    # Configuration
    rules_evaluated: List[str] = Field(
        default_factory=list,
        description="List of rule IDs that were evaluated"
    )
    environment: Optional[str] = Field(
        default=None,
        description="Environment where evaluation was performed"
    )
    context_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context information about the evaluation"
    )
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evaluation metadata"
    )

    def add_failure(self, failure: Failure) -> None:
        """Add a failure to the results."""
        self.failures.append(failure)
        
        # Update summary counts
        if failure.severity == Severity.INFO:
            self.summary.info_failures += 1
        elif failure.severity == Severity.WARNING:
            self.summary.warning_failures += 1
        elif failure.severity == Severity.CRITICAL:
            self.summary.critical_failures += 1
        
        # Update scope counts
        if failure.scope == RuleScope.PAGE:
            self.summary.page_failures += 1
        elif failure.scope == RuleScope.RUN:
            self.summary.run_failures += 1

    def get_failures_by_severity(self, severity: Severity) -> List[Failure]:
        """Get failures filtered by severity."""
        return [f for f in self.failures if f.severity == severity]

    def get_failures_by_scope(self, scope: RuleScope) -> List[Failure]:
        """Get failures filtered by scope."""
        return [f for f in self.failures if f.scope == scope]

    def get_failures_by_rule(self, rule_id: str) -> List[Failure]:
        """Get failures for a specific rule."""
        return [f for f in self.failures if f.check_id == rule_id]

    @property
    def critical_failures(self) -> List[Failure]:
        """Get all critical failures."""
        return self.get_failures_by_severity(Severity.CRITICAL)

    @property
    def should_block_deployment(self) -> bool:
        """Check if results should block deployment (has critical failures)."""
        return self.summary.has_critical_failures

    def export_summary(self) -> Dict[str, Any]:
        """Export summary for reporting."""
        return {
            'run_id': self.run_id,
            'evaluation_time': self.evaluation_time.isoformat(),
            'environment': self.environment,
            'total_rules': self.summary.total_rules,
            'enabled_rules': self.summary.enabled_rules,
            'total_failures': self.summary.total_failures,
            'critical_failures': self.summary.critical_failures,
            'warning_failures': self.summary.warning_failures,
            'info_failures': self.summary.info_failures,
            'success_rate': self.summary.success_rate,
            'should_block_deployment': self.should_block_deployment,
            'execution_time_ms': self.summary.execution_time_ms,
            'indeterminate_rules': self.summary.indeterminate_rules
        }


class AlertConfig(BaseModel):
    """Configuration for alert notifications."""
    
    enabled: bool = Field(default=True, description="Whether alerts are enabled")
    channels: List[AlertChannelType] = Field(
        default_factory=list,
        description="Types of alert channels"
    )
    
    # Channel configuration
    webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for webhook alerts"
    )
    webhook_secret: Optional[str] = Field(
        default=None,
        description="Secret for webhook HMAC signing"
    )
    
    email_recipients: List[str] = Field(
        default_factory=list,
        description="Email recipients for email alerts"
    )
    email_smtp_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="SMTP configuration for email alerts"
    )
    
    # Alert filtering
    min_severity: Severity = Field(
        default=Severity.CRITICAL,
        description="Minimum severity to trigger alerts"
    )
    environments: List[str] = Field(
        default_factory=list,
        description="Environments where alerts are enabled (empty = all)"
    )
    
    # Rate limiting
    throttle_minutes: int = Field(
        default=60,
        ge=0,
        description="Minutes to wait between duplicate alerts"
    )
    
    # Templates
    template: Dict[str, Any] = Field(
        default_factory=dict,
        description="Alert template configuration"
    )

    @field_validator('webhook_url')
    @classmethod
    def validate_webhook_url(cls, v):
        """Validate webhook URL format if provided."""
        if v is None:
            return v
        
        if isinstance(v, str) and v:
            try:
                result = urlparse(v)
                if not result.scheme or not result.netloc:
                    raise ValueError("Invalid webhook URL format")
                if result.scheme not in ['http', 'https']:
                    raise ValueError("Webhook URL must use HTTP or HTTPS")
            except Exception as e:
                raise ValueError(f"Invalid webhook URL: {e}")
        
        return v

    @property
    def requires_webhook_config(self) -> bool:
        """Check if webhook configuration is required."""
        return AlertChannelType.WEBHOOK in self.channels

    @property
    def requires_email_config(self) -> bool:
        """Check if email configuration is required."""
        return AlertChannelType.EMAIL in self.channels


class AlertPayload(BaseModel):
    """Payload for alert notifications."""
    
    # Alert metadata
    alert_id: str = Field(description="Unique alert identifier")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When alert was generated"
    )
    
    # Run context
    run_id: Optional[str] = Field(default=None, description="Run identifier")
    environment: Optional[str] = Field(default=None, description="Environment name")
    
    # Alert content
    summary: RuleSummary = Field(description="Rule evaluation summary")
    failures: List[Failure] = Field(description="Rule failures that triggered alert")
    
    # UI links
    dashboard_url: Optional[str] = Field(
        default=None,
        description="Link to dashboard for detailed view"
    )
    run_url: Optional[str] = Field(
        default=None,
        description="Direct link to run results"
    )
    
    # Signature (for webhook security)
    signature: Optional[str] = Field(
        default=None,
        description="HMAC signature for webhook verification"
    )

    @property
    def has_critical_failures(self) -> bool:
        """Check if alert includes critical failures."""
        return any(f.is_critical for f in self.failures)

    @property
    def failure_count_by_severity(self) -> Dict[str, int]:
        """Count failures by severity level."""
        counts = {"info": 0, "warning": 0, "critical": 0}
        for failure in self.failures:
            counts[failure.severity.value] += 1
        return counts

    def to_webhook_payload(self) -> Dict[str, Any]:
        """Convert to webhook-compatible payload."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "environment": self.environment,
            "summary": {
                "total_failures": self.summary.total_failures,
                "critical_failures": self.summary.critical_failures,
                "warning_failures": self.summary.warning_failures,
                "info_failures": self.summary.info_failures,
                "success_rate": self.summary.success_rate
            },
            "failures": [
                {
                    "rule_id": f.check_id,
                    "severity": f.severity.value,
                    "scope": f.scope.value if f.scope else None,
                    "page_url": f.page_url,
                    "message": f.message,
                    "timestamp": f.timestamp.isoformat()
                }
                for f in self.failures
            ],
            "links": {
                "dashboard": self.dashboard_url,
                "run_details": self.run_url
            }
        }