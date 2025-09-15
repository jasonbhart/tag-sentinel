"""Pydantic models for cookie and consent management.

This module defines comprehensive data models for Epic 5 - Cookies & Consent,
including enhanced cookie records, privacy scenarios, policy compliance,
and differential analysis between consent states.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Literal, Union, TYPE_CHECKING
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from .config import PrivacyConfiguration

# Re-export the base CookieRecord from existing capture models
from ..models.capture import CookieRecord as BaseCookieRecord


class CookieRecord(BaseCookieRecord):
    """Enhanced cookie record with privacy-specific fields.
    
    Extends the base CookieRecord with additional metadata needed
    for privacy analysis and consent management.
    """
    
    # Override size field to provide default for test compatibility
    size: int = Field(default=0, description="Total cookie size in bytes")
    
    # Privacy classification
    essential: Optional[bool] = Field(
        default=None,
        description="Whether cookie is essential for site functionality"
    )
    
    # Test compatibility field  
    first_party: Optional[bool] = Field(
        default=None,
        description="Writeable alias for first-party status (syncs with is_first_party)"
    )
    
    # Additional metadata for Epic 5
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (purpose, classification, etc.)"
    )
    
    # Timing information
    set_time: Optional[datetime] = Field(
        default=None,
        description="When cookie was first observed"
    )
    
    modified_time: Optional[datetime] = Field(
        default=None,
        description="When cookie was last modified"
    )
    
    def __init__(self, **data):
        """Initialize with first_party sync logic."""
        # Handle bidirectional sync between first_party and is_first_party
        if 'first_party' in data and 'is_first_party' not in data:
            data['is_first_party'] = data['first_party']
        elif 'is_first_party' in data and 'first_party' not in data:
            data['first_party'] = data['is_first_party']
        elif 'first_party' in data and 'is_first_party' in data:
            # Both provided, ensure they match
            if data['first_party'] != data['is_first_party']:
                data['is_first_party'] = data['first_party']  # first_party takes precedence
        
        super().__init__(**data)
    
    @property  
    def first_party_computed(self) -> bool:
        """Computed property for read-only access."""
        return self.is_first_party
    
    @property
    def scenario_id(self) -> Optional[str]:
        """Get scenario_id from metadata."""
        return self.metadata.get('scenario_id')
    
    @scenario_id.setter
    def scenario_id(self, value: Optional[str]):
        """Set scenario_id in metadata."""
        if value is not None:
            self.metadata['scenario_id'] = value
        elif 'scenario_id' in self.metadata:
            del self.metadata['scenario_id']
    
    @property
    def category(self) -> Optional[Any]:
        """Get category from metadata.classification."""
        classification = self.metadata.get('classification', {})
        return classification.get('category')
    
    @category.setter
    def category(self, value: Optional[Any]):
        """Set category in metadata.classification."""
        if 'classification' not in self.metadata:
            self.metadata['classification'] = {}
        if value is not None:
            self.metadata['classification']['category'] = value
        elif 'category' in self.metadata.get('classification', {}):
            del self.metadata['classification']['category']
    
    def update_first_party_status(self, is_first_party: bool):
        """Update first-party status and sync fields."""
        self.is_first_party = is_first_party
        self.first_party = is_first_party


class Scenario(BaseModel):
    """Privacy testing scenario configuration.
    
    Defines a specific privacy testing scenario including headers,
    CMP interactions, and expected behaviors.
    """
    
    id: str = Field(description="Unique scenario identifier")
    name: str = Field(description="Human-readable scenario name")
    description: str = Field(description="Detailed scenario description")
    
    # Request configuration
    request_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="HTTP headers to inject (e.g., Sec-GPC: 1)"
    )
    
    # CMP interaction steps
    steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="CMP interaction steps (selectors, actions, waits)"
    )
    
    # Configuration
    enabled: bool = Field(default=True, description="Whether scenario is enabled")
    
    # Expected behavior
    expected_cookie_reduction: Optional[float] = Field(
        default=None,
        description="Expected percentage reduction in cookies vs baseline"
    )
    
    @field_validator('id')
    @classmethod
    def validate_scenario_id(cls, v):
        """Validate scenario ID format."""
        if not v or not isinstance(v, str):
            raise ValueError("Scenario ID must be a non-empty string")
        # Allow alphanumeric, dashes, underscores
        if not all(c.isalnum() or c in '-_' for c in v):
            raise ValueError("Scenario ID can only contain alphanumeric characters, dashes, and underscores")
        return v


class PolicySeverity(str, Enum):
    """Severity levels for policy violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CookiePolicyIssue(BaseModel):
    """Cookie policy compliance violation.
    
    Represents a specific violation of privacy policy requirements,
    with details about the violation and suggested remediation.
    """
    
    # Cookie identification
    cookie_name: str = Field(description="Name of the problematic cookie")
    cookie_domain: str = Field(description="Domain of the problematic cookie")
    cookie_path: str = Field(default="/", description="Path of the problematic cookie")
    
    # Violation details
    attribute: str = Field(description="Cookie attribute that violates policy")
    expected: str = Field(description="Expected value according to policy")
    observed: str = Field(description="Actual observed value")
    
    # Issue metadata
    severity: PolicySeverity = Field(description="Severity of the policy violation")
    rule_id: str = Field(description="Policy rule identifier")
    message: str = Field(description="Human-readable violation description")
    
    # Context
    scenario_id: Optional[str] = Field(
        default=None,
        description="Scenario where violation was detected"
    )
    page_url: Optional[str] = Field(
        default=None,
        description="Page URL where violation occurred"
    )
    
    # Remediation
    remediation: Optional[str] = Field(
        default=None,
        description="Suggested remediation steps"
    )
    
    @property
    def cookie_key(self) -> str:
        """Unique identifier for the cookie."""
        return f"{self.cookie_name}@{self.cookie_domain}{self.cookie_path}"


class ConsentState(str, Enum):
    """Possible consent states for CMP interactions."""
    UNKNOWN = "unknown"
    ACCEPT_ALL = "accept_all"
    REJECT_ALL = "reject_all"
    CUSTOM = "custom"


class ScenarioCookieReport(BaseModel):
    """Complete cookie analysis for a specific scenario.
    
    Contains all cookies found in a scenario plus analysis results
    and policy compliance information.
    """
    
    # Scenario identification
    scenario_id: str = Field(description="Scenario identifier")
    scenario_name: str = Field(description="Human-readable scenario name")
    
    # Page context
    page_url: str = Field(description="URL that was analyzed")
    page_title: Optional[str] = Field(default=None, description="Page title")
    
    # Timestamp
    analysis_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="When analysis was performed"
    )
    
    # Cookie inventory
    cookies: List[CookieRecord] = Field(
        default_factory=list,
        description="All cookies found in this scenario"
    )
    
    # Policy analysis
    policy_issues: List[CookiePolicyIssue] = Field(
        default_factory=list,
        description="Policy violations found"
    )
    
    # Consent state (for CMP scenarios)
    consent_state: Optional[ConsentState] = Field(
        default=None,
        description="Consent state if CMP was used"
    )
    
    # Statistics
    total_cookies: int = Field(default=0, description="Total number of cookies")
    first_party_cookies: int = Field(default=0, description="Number of first-party cookies")
    third_party_cookies: int = Field(default=0, description="Number of third-party cookies")
    essential_cookies: int = Field(default=0, description="Number of essential cookies")
    
    # Error information
    errors: List[str] = Field(
        default_factory=list,
        description="Errors encountered during scenario execution"
    )
    
    def model_post_init(self, __context):
        """Calculate statistics after model creation."""
        if self.cookies:
            self.total_cookies = len(self.cookies)
            self.first_party_cookies = len([c for c in self.cookies if c.is_first_party])
            self.third_party_cookies = self.total_cookies - self.first_party_cookies
            self.essential_cookies = len([c for c in self.cookies if c.essential])
    
    @property
    def has_violations(self) -> bool:
        """Check if any policy violations were found."""
        return len(self.policy_issues) > 0
    
    @property
    def violation_count_by_severity(self) -> Dict[str, int]:
        """Count violations by severity level."""
        counts = {severity.value: 0 for severity in PolicySeverity}
        for issue in self.policy_issues:
            counts[issue.severity.value] += 1
        return counts


class CookieChangeType(str, Enum):
    """Types of changes between cookie scenarios."""
    ADDED = "added"
    REMOVED = "removed" 
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class CookieChange(BaseModel):
    """Represents a change in a cookie between scenarios."""
    
    cookie_key: str = Field(description="Unique cookie identifier (name@domain/path)")
    change_type: CookieChangeType = Field(description="Type of change")
    
    # Cookie details
    cookie_name: str = Field(description="Cookie name")
    cookie_domain: str = Field(description="Cookie domain")
    cookie_path: str = Field(default="/", description="Cookie path")
    
    # Change details (for modified cookies)
    attribute_changes: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Attribute-level changes {attribute: {from: old, to: new}}"
    )
    
    # Context
    baseline_value: Optional[Any] = Field(
        default=None,
        description="Value in baseline scenario"
    )
    variant_value: Optional[Any] = Field(
        default=None,
        description="Value in variant scenario"
    )


class CookieDiff(BaseModel):
    """Comprehensive difference analysis between two cookie scenarios.
    
    Provides detailed comparison between baseline and variant scenarios,
    including cookies added, removed, and modified.
    """
    
    # Scenario identification
    baseline_scenario: str = Field(description="Baseline scenario identifier")
    variant_scenario: str = Field(description="Variant scenario identifier")
    
    # Analysis metadata
    comparison_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="When comparison was performed"
    )
    page_url: str = Field(description="URL that was compared")
    
    # Change lists
    added_cookies: List[CookieRecord] = Field(
        default_factory=list,
        description="Cookies present in variant but not baseline"
    )
    
    removed_cookies: List[CookieRecord] = Field(
        default_factory=list,
        description="Cookies present in baseline but not variant"
    )
    
    modified_cookies: List[CookieChange] = Field(
        default_factory=list,
        description="Cookies present in both but with different attributes"
    )
    
    unchanged_cookies: List[CookieRecord] = Field(
        default_factory=list,
        description="Cookies identical in both scenarios"
    )
    
    # Summary statistics
    total_changes: int = Field(default=0, description="Total number of changes")
    reduction_percentage: float = Field(
        default=0.0,
        description="Percentage reduction in cookie count"
    )
    
    # Policy impact
    policy_improvement: bool = Field(
        default=False,
        description="Whether variant scenario improves policy compliance"
    )
    violations_resolved: int = Field(
        default=0,
        description="Number of policy violations resolved"
    )
    violations_introduced: int = Field(
        default=0,
        description="Number of new policy violations introduced"
    )
    
    def model_post_init(self, __context):
        """Calculate statistics after model creation."""
        self.total_changes = len(self.added_cookies) + len(self.removed_cookies) + len(self.modified_cookies)
    
    @property
    def has_changes(self) -> bool:
        """Check if any changes were detected."""
        return self.total_changes > 0
    
    @property
    def cookie_reduction(self) -> int:
        """Calculate net change in cookie count."""
        return len(self.removed_cookies) - len(self.added_cookies)
    
    @property
    def significant_reduction(self) -> bool:
        """Check if there was a significant reduction in cookies."""
        return self.reduction_percentage > 10.0  # Configurable threshold

    @property
    def scenario_a_id(self) -> str:
        """Backward compatibility alias for baseline_scenario."""
        return self.baseline_scenario

    @scenario_a_id.setter
    def scenario_a_id(self, value: str):
        """Backward compatibility setter for baseline_scenario."""
        self.baseline_scenario = value

    @property
    def scenario_b_id(self) -> str:
        """Backward compatibility alias for variant_scenario."""
        return self.variant_scenario

    @scenario_b_id.setter
    def scenario_b_id(self, value: str):
        """Backward compatibility setter for variant_scenario."""
        self.variant_scenario = value


class PrivacyConfig(BaseModel):
    """Configuration for privacy testing and policy compliance.
    
    Defines privacy testing parameters, policy requirements,
    and scenario configurations.
    """
    
    # GPC configuration
    gpc_enabled: bool = Field(default=True, description="Enable GPC simulation")
    gpc_header: str = Field(default="Sec-GPC: 1", description="GPC header to inject")
    
    # DNT configuration (legacy)
    dnt_enabled: bool = Field(default=False, description="Enable DNT simulation")
    dnt_header: str = Field(default="DNT: 1", description="DNT header to inject")
    
    # CMP configuration
    cmp_enabled: bool = Field(default=False, description="Enable CMP interactions")
    cmp_selectors: Dict[str, str] = Field(
        default_factory=dict,
        description="CSS selectors for CMP buttons"
    )
    cmp_wait_after_click: int = Field(
        default=2000,
        description="Milliseconds to wait after CMP interaction"
    )
    
    # Policy requirements
    policies: Dict[str, Any] = Field(
        default_factory=dict,
        description="Policy compliance requirements"
    )
    
    # Scenarios
    scenarios: List[Scenario] = Field(
        default_factory=list,
        description="Configured privacy scenarios"
    )
    
    # Analysis settings
    essential_cookie_patterns: List[str] = Field(
        default_factory=lambda: [
            r"session.*",
            r"csrf.*", 
            r"auth.*",
            r"login.*",
            r"security.*"
        ],
        description="Regex patterns for essential cookies"
    )
    
    def get_enabled_scenarios(self) -> List[Scenario]:
        """Get list of enabled scenarios."""
        return [scenario for scenario in self.scenarios if scenario.enabled]
    
    def get_scenario_by_id(self, scenario_id: str) -> Optional[Scenario]:
        """Get scenario by ID."""
        for scenario in self.scenarios:
            if scenario.id == scenario_id:
                return scenario
        return None


class PrivacyAnalysisResult(BaseModel):
    """Complete privacy analysis result for a page.
    
    Contains results from all scenarios, comparative analysis,
    and overall privacy assessment.
    """
    
    # Page context
    page_url: str = Field(description="URL that was analyzed")
    page_title: Optional[str] = Field(default=None, description="Page title")
    analysis_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="When analysis was performed"
    )
    
    # Scenario results
    scenario_reports: Dict[str, ScenarioCookieReport] = Field(
        default_factory=dict,
        description="Results for each scenario"
    )
    
    # Comparative analysis
    scenario_diffs: Dict[str, CookieDiff] = Field(
        default_factory=dict,
        description="Comparisons between scenarios"
    )
    
    # Overall assessment
    privacy_score: float = Field(
        default=0.0,
        description="Overall privacy compliance score (0-100)"
    )
    
    gpc_effectiveness: Optional[float] = Field(
        default=None,
        description="GPC signal effectiveness (percentage reduction)"
    )
    
    cmp_effectiveness: Optional[float] = Field(
        default=None,
        description="CMP consent effectiveness (percentage reduction)"
    )
    
    # Issues and recommendations
    critical_issues: List[CookiePolicyIssue] = Field(
        default_factory=list,
        description="Critical privacy issues found"
    )
    
    recommendations: List[str] = Field(
        default_factory=list,
        description="Privacy improvement recommendations"
    )
    
    # Configuration used  
    config: Optional[Any] = Field(
        default=None,
        description="Privacy configuration used for analysis (PrivacyConfiguration or PrivacyConfig)"
    )
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata and metrics"
    )
    
    @property
    def baseline_report(self) -> Optional[ScenarioCookieReport]:
        """Get baseline scenario report."""
        return self.scenario_reports.get('baseline')
    
    @property
    def gpc_report(self) -> Optional[ScenarioCookieReport]:
        """Get GPC scenario report."""
        return self.scenario_reports.get('gpc_on')
    
    @property
    def has_privacy_issues(self) -> bool:
        """Check if any privacy issues were found."""
        return len(self.critical_issues) > 0
    
    @property
    def analysis_timestamp(self) -> datetime:
        """Alias for analysis_time to maintain API compatibility."""
        return self.analysis_time

