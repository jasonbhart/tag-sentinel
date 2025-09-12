"""Configuration management for privacy testing and cookie analysis.

This module provides configuration loading and validation for Epic 5 privacy features,
including GPC simulation, CMP interactions, and policy compliance requirements.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

import yaml
from pydantic import BaseModel, Field, field_validator

from .models import Scenario, PrivacyConfig

logger = logging.getLogger(__name__)


class CMPConfig(BaseModel):
    """Consent Management Platform configuration."""
    
    enabled: bool = Field(default=False, description="Enable CMP interactions")
    
    # Selector configuration
    selectors: Dict[str, str] = Field(
        default_factory=dict,
        description="CSS selectors for CMP elements"
    )
    
    # Interaction timing
    wait_for_modal_ms: int = Field(
        default=3000,
        description="Time to wait for CMP modal to appear"
    )
    wait_after_click_ms: int = Field(
        default=2000, 
        description="Time to wait after CMP button clicks"
    )
    max_interaction_attempts: int = Field(
        default=3,
        description="Maximum attempts for CMP interactions"
    )
    
    # Debugging
    screenshot_interactions: bool = Field(
        default=True,
        description="Capture screenshots during CMP interactions"
    )


class GPCConfig(BaseModel):
    """Global Privacy Control configuration."""
    
    enabled: bool = Field(default=True, description="Enable GPC simulation")
    header: str = Field(default="Sec-GPC: 1", description="GPC header to inject")
    simulate_javascript_api: bool = Field(
        default=True,
        description="Simulate navigator.globalPrivacyControl API"
    )


class DNTConfig(BaseModel):
    """Do Not Track configuration."""
    
    enabled: bool = Field(default=False, description="Enable DNT simulation")
    header: str = Field(default="DNT: 1", description="DNT header to inject")


class PolicyRule(BaseModel):
    """Privacy policy rule definition."""
    
    id: str = Field(description="Rule identifier")
    name: str = Field(description="Human-readable rule name") 
    description: str = Field(description="Rule description")
    severity: str = Field(description="Rule severity (low, medium, high, critical)")
    condition: str = Field(description="Rule condition expression")
    attribute: str = Field(description="Cookie attribute being validated")
    expected: str = Field(description="Expected attribute value")
    remediation: Optional[str] = Field(
        default=None,
        description="Remediation guidance"
    )


class ClassificationConfig(BaseModel):
    """Cookie classification configuration."""
    
    essential_patterns: List[str] = Field(
        default_factory=list,
        description="Regex patterns for essential cookies"
    )
    analytics_patterns: List[str] = Field(
        default_factory=list,
        description="Regex patterns for analytics cookies"
    )
    analytics_domains: List[str] = Field(
        default_factory=list,
        description="Known analytics domains"
    )


class PolicyRequirements(BaseModel):
    """Privacy policy requirements."""
    
    secure_required: bool = Field(
        default=True,
        description="Require Secure flag on HTTPS sites"
    )
    same_site_default: str = Field(
        default="Lax",
        description="Default SameSite requirement"
    )
    http_only_sessions: bool = Field(
        default=True, 
        description="Require HttpOnly for session cookies"
    )
    third_party_limit: int = Field(
        default=10,
        description="Warning threshold for third-party cookies"
    )
    analytics_consent_required: bool = Field(
        default=True,
        description="Analytics cookies require consent"
    )
    validate_domain_scope: bool = Field(
        default=True,
        description="Validate cookie domain scope"
    )
    allow_subdomain_cookies: bool = Field(
        default=True,
        description="Allow cookies for subdomains"
    )


class PrivacyConfiguration(BaseModel):
    """Complete privacy testing configuration.
    
    Loads and validates privacy testing configuration from YAML files,
    with support for environment-specific overrides.
    """
    
    # Core privacy settings
    gpc: GPCConfig = Field(default_factory=GPCConfig)
    dnt: DNTConfig = Field(default_factory=DNTConfig) 
    cmp: CMPConfig = Field(default_factory=CMPConfig)
    
    # Policy requirements
    policies: PolicyRequirements = Field(default_factory=PolicyRequirements)
    
    # Scenarios
    scenarios: List[Scenario] = Field(default_factory=list)
    
    # Classification rules
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    
    # Policy rules
    policy_rules: List[PolicyRule] = Field(default_factory=list)
    
    # Environment
    environment: str = Field(default="development", description="Current environment")
    
    def get_enabled_scenarios(self) -> List[Scenario]:
        """Get list of enabled scenarios."""
        return [scenario for scenario in self.scenarios if scenario.enabled]
    
    def get_scenario_by_id(self, scenario_id: str) -> Optional[Scenario]:
        """Get scenario by ID."""
        for scenario in self.scenarios:
            if scenario.id == scenario_id:
                return scenario
        return None
    
    def get_cmp_selector(self, selector_name: str) -> Optional[str]:
        """Get CMP selector by name."""
        return self.cmp.selectors.get(selector_name)
    
    def is_essential_cookie(self, cookie_name: str) -> bool:
        """Check if cookie matches essential patterns."""
        for pattern in self.classification.essential_patterns:
            try:
                if re.match(pattern, cookie_name, re.IGNORECASE):
                    return True
            except re.error:
                logger.warning(f"Invalid regex pattern: {pattern}")
        return False
    
    def is_analytics_cookie(self, cookie_name: str) -> bool:
        """Check if cookie matches analytics patterns."""
        for pattern in self.classification.analytics_patterns:
            try:
                if re.match(pattern, cookie_name, re.IGNORECASE):
                    return True
            except re.error:
                logger.warning(f"Invalid regex pattern: {pattern}")
        return False
    
    def is_analytics_domain(self, domain: str) -> bool:
        """Check if domain is a known analytics domain."""
        clean_domain = domain.lstrip('.')
        
        # Direct match
        if clean_domain in self.classification.analytics_domains:
            return True
            
        # Check subdomains
        for analytics_domain in self.classification.analytics_domains:
            if clean_domain.endswith(f'.{analytics_domain}'):
                return True
                
        return False
    
    def validate_scenario_config(self) -> List[str]:
        """Validate scenario configuration and return issues."""
        issues = []
        
        scenario_ids = [s.id for s in self.scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            issues.append("Duplicate scenario IDs found")
        
        # Check baseline scenario exists
        if not any(s.id == 'baseline' for s in self.scenarios):
            issues.append("Baseline scenario is required")
        
        # Check CMP scenarios have required selectors if enabled
        for scenario in self.scenarios:
            if scenario.id.startswith('cmp_') and scenario.enabled:
                if not self.cmp.enabled:
                    issues.append(f"Scenario {scenario.id} requires CMP to be enabled")
                    
                # Check if required selectors exist in steps
                for step in scenario.steps:
                    if step.get('type') == 'click':
                        selector = step.get('selector')
                        if selector and selector in self.cmp.selectors:
                            if not self.cmp.selectors[selector]:
                                issues.append(f"Scenario {scenario.id} references undefined selector: {selector}")
        
        return issues


class PrivacyConfigLoader:
    """Loads privacy configuration from YAML files with environment support."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize config loader.
        
        Args:
            config_dir: Directory containing config files. Defaults to project config/ dir.
        """
        if config_dir is None:
            # Default to project config directory
            project_root = Path(__file__).parent.parent.parent.parent
            config_dir = project_root / "config"
        
        self.config_dir = Path(config_dir)
        self.environment = os.getenv('PRIVACY_ENV', 'development')
    
    def load_config(self, environment: Optional[str] = None) -> PrivacyConfiguration:
        """Load privacy configuration for specified environment.
        
        Args:
            environment: Environment name. Defaults to PRIVACY_ENV or 'development'.
            
        Returns:
            Loaded and validated privacy configuration.
            
        Raises:
            FileNotFoundError: If config files are not found.
            ValueError: If configuration is invalid.
        """
        env = environment or self.environment
        
        # Load base configuration
        base_config_path = self.config_dir / "privacy.yaml"
        if not base_config_path.exists():
            raise FileNotFoundError(f"Base privacy config not found: {base_config_path}")
        
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Load environment-specific overrides
        env_config_path = self.config_dir / f"privacy.{env}.yaml"
        if env_config_path.exists():
            logger.info(f"Loading environment config: {env_config_path}")
            with open(env_config_path, 'r', encoding='utf-8') as f:
                env_config = yaml.safe_load(f)
            
            # Merge environment config (deep merge)
            config_data = self._deep_merge(config_data, env_config)
        
        # Extract environment-specific defaults if present
        if env in config_data:
            env_defaults = config_data.pop(env)
            config_data = self._deep_merge(config_data, env_defaults)
        
        # Parse scenarios
        scenarios = []
        if 'scenarios' in config_data:
            for scenario_data in config_data['scenarios']:
                scenarios.append(Scenario(**scenario_data))
        
        # Parse policy rules
        policy_rules = []
        if 'policy_rules' in config_data:
            for rule_data in config_data['policy_rules']:
                policy_rules.append(PolicyRule(**rule_data))
        
        # Build configuration
        privacy_config = {
            'environment': env,
            'scenarios': scenarios,
            'policy_rules': policy_rules
        }
        
        # Parse privacy section
        if 'privacy' in config_data:
            privacy_section = config_data['privacy']
            
            if 'gpc' in privacy_section:
                privacy_config['gpc'] = GPCConfig(**privacy_section['gpc'])
            
            if 'dnt' in privacy_section:
                privacy_config['dnt'] = DNTConfig(**privacy_section['dnt'])
                
            if 'cmp' in privacy_section:
                privacy_config['cmp'] = CMPConfig(**privacy_section['cmp'])
                
            if 'policies' in privacy_section:
                privacy_config['policies'] = PolicyRequirements(**privacy_section['policies'])
        
        # Parse classification section
        if 'classification' in config_data:
            privacy_config['classification'] = ClassificationConfig(**config_data['classification'])
        
        try:
            config = PrivacyConfiguration(**privacy_config)
            
            # Validate configuration
            issues = config.validate_scenario_config()
            if issues:
                logger.warning(f"Configuration validation issues: {issues}")
            
            logger.info(f"Loaded privacy configuration for environment: {env}")
            logger.info(f"Enabled scenarios: {[s.id for s in config.get_enabled_scenarios()]}")
            
            return config
            
        except Exception as e:
            raise ValueError(f"Invalid privacy configuration: {e}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def create_default_config(self) -> PrivacyConfiguration:
        """Create a default privacy configuration for testing."""
        return PrivacyConfiguration(
            environment="test",
            scenarios=[
                Scenario(
                    id="baseline",
                    name="Baseline",
                    description="No privacy signals",
                    enabled=True
                ),
                Scenario(
                    id="gpc_on",
                    name="GPC Enabled", 
                    description="Global Privacy Control enabled",
                    enabled=True,
                    request_headers={"Sec-GPC": "1"}
                )
            ]
        )


# Global configuration instance
_config_loader = PrivacyConfigLoader()
_config_cache: Optional[PrivacyConfiguration] = None


def get_privacy_config(environment: Optional[str] = None, force_reload: bool = False) -> PrivacyConfiguration:
    """Get privacy configuration for the specified environment.
    
    Args:
        environment: Environment name. If None, uses PRIVACY_ENV or 'development'.
        force_reload: Force reload from files, ignoring cache.
        
    Returns:
        Privacy configuration instance.
    """
    global _config_cache
    
    if force_reload or _config_cache is None:
        try:
            _config_cache = _config_loader.load_config(environment)
        except FileNotFoundError:
            logger.warning("Privacy config file not found, using default configuration")
            _config_cache = _config_loader.create_default_config()
        except Exception as e:
            logger.error(f"Failed to load privacy config: {e}")
            logger.warning("Using default configuration")
            _config_cache = _config_loader.create_default_config()
    
    return _config_cache


def load_privacy_config_from_file(config_path: Path) -> PrivacyConfiguration:
    """Load privacy configuration from a specific file.
    
    Args:
        config_path: Path to the privacy config file.
        
    Returns:
        Privacy configuration instance.
    """
    config_dir = config_path.parent
    loader = PrivacyConfigLoader(config_dir)
    return loader.load_config()


def validate_privacy_config(config_path: Path) -> List[str]:
    """Validate a privacy configuration file.
    
    Args:
        config_path: Path to the privacy config file.
        
    Returns:
        List of validation issues (empty if valid).
    """
    try:
        config = load_privacy_config_from_file(config_path)
        return config.validate_scenario_config()
    except Exception as e:
        return [f"Configuration error: {e}"]