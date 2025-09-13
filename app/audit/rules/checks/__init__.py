"""Rule checks for validating audit data.

This package provides a comprehensive set of check implementations for validating
various aspects of web analytics and tag implementation, including presence checks,
count validation, duplicate detection, temporal analysis, and privacy compliance.
"""

from .base import (
    # Core abstractions
    BaseCheck,
    CheckContext,
    CheckResult,
    CheckRegistry,
    
    # Registry and decorators
    check_registry,
    register_check,
)

from .presence import (
    # Presence/absence checks
    RequestPresentCheck,
    RequestAbsentCheck,
    CookiePresentCheck,
    TagEventPresentCheck,
    ConsoleMessagePresentCheck,
)

from .duplicates import (
    # Duplicate detection checks
    RequestDuplicateCheck,
    EventDuplicateCheck,
    CookieDuplicateCheck,
    DuplicateGroup,
)

from .temporal import (
    # Temporal and sequencing checks
    LoadTimingCheck,
    SequenceOrderCheck,
    RelativeTimingCheck,
    SequenceOrderType,
    TimingComparison,
    SequenceItem,
)

from .privacy import (
    # Privacy and compliance checks
    GDPRComplianceCheck,
    CCPAComplianceCheck,
    CookieSecurityCheck,
    PrivacyRegulation,
)

from .expressions import (
    # Expression-based checks
    ExpressionCheck,
    JSONPathCheck,
    SafeExpressionEvaluator,
    SimpleJSONPath,
    SafeExpressionError,
    JSONPathError,
)

__all__ = [
    # Core abstractions
    'BaseCheck',
    'CheckContext', 
    'CheckResult',
    'CheckRegistry',
    
    # Registry and decorators
    'check_registry',
    'register_check',
    
    # Presence/absence checks
    'RequestPresentCheck',
    'RequestAbsentCheck',
    'CookiePresentCheck',
    'TagEventPresentCheck',
    'ConsoleMessagePresentCheck',
    
    # Duplicate detection checks
    'RequestDuplicateCheck',
    'EventDuplicateCheck',
    'CookieDuplicateCheck',
    'DuplicateGroup',
    
    # Temporal and sequencing checks
    'LoadTimingCheck',
    'SequenceOrderCheck',
    'RelativeTimingCheck',
    'SequenceOrderType',
    'TimingComparison',
    'SequenceItem',
    
    # Privacy and compliance checks
    'GDPRComplianceCheck',
    'CCPAComplianceCheck',
    'CookieSecurityCheck',
    'PrivacyRegulation',
    
    # Expression-based checks
    'ExpressionCheck',
    'JSONPathCheck',
    'SafeExpressionEvaluator',
    'SimpleJSONPath',
    'SafeExpressionError',
    'JSONPathError',
]