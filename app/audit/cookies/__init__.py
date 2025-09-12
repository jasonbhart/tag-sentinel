"""Epic 5: Cookie and Consent Management for Privacy Testing.

This module provides comprehensive cookie and consent management capabilities
including privacy signal simulation, CMP automation, policy compliance
validation, and cross-scenario differential analysis.
"""

from .models import (
    CookieRecord,
    Scenario,
    CookiePolicyIssue,
    ScenarioCookieReport,
    CookieDiff,
    PrivacyAnalysisResult,
    ConsentState
)

from .config import (
    PrivacyConfiguration,
    get_privacy_config,
    load_privacy_config_from_file
)

from .service import (
    CookieConsentService,
    analyze_page_privacy,
    create_cookie_consent_service
)

from .collection import EnhancedCookieCollector
from .classification import CookieClassifier, CookieCategory
from .policy import PolicyComplianceEngine, ComplianceFramework
from .gpc import GPCSimulator
from .cmp import CMPDetector, ConsentAutomator, CMPPlatform
from .comparison import ScenarioComparator
from .orchestration import ScenarioOrchestrator, execute_privacy_scenarios

__all__ = [
    # Core Models
    "CookieRecord",
    "Scenario", 
    "CookiePolicyIssue",
    "ScenarioCookieReport",
    "CookieDiff",
    "PrivacyAnalysisResult",
    "ConsentState",
    
    # Configuration
    "PrivacyConfiguration",
    "get_privacy_config",
    "load_privacy_config_from_file",
    
    # Main Service
    "CookieConsentService", 
    "analyze_page_privacy",
    "create_cookie_consent_service",
    
    # Core Components
    "EnhancedCookieCollector",
    "CookieClassifier",
    "CookieCategory",
    "PolicyComplianceEngine", 
    "ComplianceFramework",
    "GPCSimulator",
    "CMPDetector",
    "ConsentAutomator",
    "CMPPlatform",
    "ScenarioComparator",
    "ScenarioOrchestrator",
    "execute_privacy_scenarios"
]