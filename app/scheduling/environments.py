"""Environment configuration system with hierarchical inheritance.

This module provides environment-specific configuration management with support for:
- Global defaults
- Site-specific overrides
- Environment-specific configurations
- Configuration validation and testing
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import yaml
import logging

from pydantic import BaseModel, Field, field_validator, model_validator


logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass


class EnvironmentConfig(BaseModel):
    """Environment-specific configuration with validation."""

    # Analytics and tracking IDs
    ga4_measurement_id: Optional[str] = Field(
        default=None,
        description="Google Analytics 4 Measurement ID (G-XXXXXXXXXX)"
    )

    gtm_container_id: Optional[str] = Field(
        default=None,
        description="Google Tag Manager Container ID (GTM-XXXXXXX)"
    )

    facebook_pixel_id: Optional[str] = Field(
        default=None,
        description="Facebook Pixel ID"
    )

    # Run configuration
    timeout_ms: int = Field(
        default=30000,
        ge=5000,
        le=300000,  # 5 minutes max
        description="Page timeout in milliseconds"
    )

    max_pages: int = Field(
        default=500,
        ge=1,
        le=100000,
        description="Maximum pages to crawl"
    )

    max_concurrent: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum concurrent requests"
    )

    requests_per_second: float = Field(
        default=2.0,
        gt=0.0,
        le=100.0,
        description="Rate limit for requests"
    )

    # Expected elements for validation
    expected_cookies: List[str] = Field(
        default_factory=list,
        description="List of expected cookie names"
    )

    expected_tags: List[str] = Field(
        default_factory=list,
        description="List of expected tag/script names"
    )

    expected_events: List[str] = Field(
        default_factory=list,
        description="List of expected analytics events"
    )

    # Environment-specific URLs
    base_urls: List[str] = Field(
        default_factory=list,
        description="Base URLs for this environment"
    )

    # Custom configuration
    custom_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom environment-specific parameters"
    )

    @field_validator('ga4_measurement_id')
    @classmethod
    def validate_ga4_id(cls, v):
        """Validate GA4 measurement ID format."""
        if v is not None and not re.match(r'^G-[A-Z0-9]{10}$', v):
            raise ValueError("GA4 measurement ID must match format G-XXXXXXXXXX")
        return v

    @field_validator('gtm_container_id')
    @classmethod
    def validate_gtm_id(cls, v):
        """Validate GTM container ID format."""
        if v is not None and not re.match(r'^GTM-[A-Z0-9]{5,8}$', v):
            raise ValueError("GTM container ID must match format GTM-XXXXXXX")
        return v

    @field_validator('facebook_pixel_id')
    @classmethod
    def validate_facebook_pixel_id(cls, v):
        """Validate Facebook Pixel ID format."""
        if v is not None and not re.match(r'^\d{15,16}$', v):
            raise ValueError("Facebook Pixel ID must be 15-16 digits")
        return v

    @field_validator('base_urls')
    @classmethod
    def validate_base_urls(cls, v):
        """Validate base URL formats."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP address
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )

        for url in v:
            if not url_pattern.match(url):
                raise ValueError(f"Invalid URL format: {url}")
        return v


class SiteEnvironmentConfig(BaseModel):
    """Site-specific environment configuration."""

    site_id: str = Field(description="Site identifier")

    environments: Dict[str, EnvironmentConfig] = Field(
        default_factory=dict,
        description="Environment configurations"
    )

    defaults: EnvironmentConfig = Field(
        default_factory=EnvironmentConfig,
        description="Default configuration for all environments"
    )

    @field_validator('site_id')
    @classmethod
    def validate_site_id(cls, v):
        """Validate site ID format."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("site_id must contain only alphanumeric characters, underscores, and hyphens")
        return v


class GlobalEnvironmentConfig(BaseModel):
    """Global environment configuration with inheritance."""

    version: str = Field(default="1.0", description="Configuration version")

    # Global defaults applied to all sites and environments
    defaults: EnvironmentConfig = Field(
        default_factory=EnvironmentConfig,
        description="Global default configuration"
    )

    # Environment-specific defaults (override global defaults)
    environment_defaults: Dict[str, EnvironmentConfig] = Field(
        default_factory=dict,
        description="Environment-specific default configurations"
    )

    # Site-specific configurations
    sites: Dict[str, SiteEnvironmentConfig] = Field(
        default_factory=dict,
        description="Site-specific configurations"
    )

    # Environment validation rules
    required_ids: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "production": ["ga4_measurement_id", "gtm_container_id"],
            "staging": ["ga4_measurement_id", "gtm_container_id"],
        },
        description="Required tracking IDs per environment"
    )

    @model_validator(mode='after')
    def validate_environment_consistency(self):
        """Validate that all sites define consistent environments."""
        all_environments = set()
        for site_config in self.sites.values():
            all_environments.update(site_config.environments.keys())

        # Check that required environments exist
        required_envs = {'production', 'staging'}
        missing_envs = required_envs - all_environments
        if missing_envs:
            logger.warning(f"Missing standard environments: {missing_envs}")

        return self


@dataclass
class ResolvedEnvironmentConfig:
    """Fully resolved configuration for a site-environment combination."""

    site_id: str
    environment: str
    config: EnvironmentConfig
    inheritance_chain: List[str] = field(default_factory=list)

    def get_audit_params(self) -> Dict[str, Any]:
        """Get audit run parameters from resolved configuration."""
        params = {
            'timeout_ms': self.config.timeout_ms,
            'max_pages': self.config.max_pages,
            'max_concurrent': self.config.max_concurrent,
            'requests_per_second': self.config.requests_per_second,
            'base_urls': self.config.base_urls,
        }

        # Add tracking IDs if present
        if self.config.ga4_measurement_id:
            params['ga4_measurement_id'] = self.config.ga4_measurement_id
        if self.config.gtm_container_id:
            params['gtm_container_id'] = self.config.gtm_container_id
        if self.config.facebook_pixel_id:
            params['facebook_pixel_id'] = self.config.facebook_pixel_id

        # Add custom parameters
        params.update(self.config.custom_params)

        return params


class EnvironmentConfigManager:
    """Manager for environment configuration with inheritance resolution."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the configuration manager.

        Args:
            config_path: Path to configuration file. If None, uses default location.
        """
        if config_path is None:
            # Default to config/environments.yaml in project root
            project_root = Path(__file__).parents[2]  # Go up from app/scheduling/
            config_path = project_root / "config" / "environments.yaml"

        self.config_path = Path(config_path)
        self._global_config: Optional[GlobalEnvironmentConfig] = None

    def load_config(self) -> GlobalEnvironmentConfig:
        """Load configuration from file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return GlobalEnvironmentConfig()

        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            if not isinstance(config_data, dict):
                raise ConfigValidationError("Configuration file must contain a YAML dictionary")

            self._global_config = GlobalEnvironmentConfig(**config_data)
            logger.info(f"Loaded environment configuration from: {self.config_path}")
            return self._global_config

        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Failed to parse YAML: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Failed to load configuration: {e}")

    def save_config(self, config: GlobalEnvironmentConfig) -> None:
        """Save configuration to file."""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_path, 'w') as f:
                config_dict = config.model_dump(exclude_none=True)
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Saved environment configuration to: {self.config_path}")
            self._global_config = config

        except Exception as e:
            raise ConfigValidationError(f"Failed to save configuration: {e}")

    def get_config(self) -> GlobalEnvironmentConfig:
        """Get current configuration, loading if necessary."""
        if self._global_config is None:
            return self.load_config()
        return self._global_config

    def resolve_environment_config(
        self,
        site_id: str,
        environment: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> ResolvedEnvironmentConfig:
        """Resolve configuration for a site-environment combination with inheritance.

        Inheritance order (later overrides earlier):
        1. Global defaults
        2. Environment defaults
        3. Site defaults
        4. Site-environment specific
        5. Manual overrides

        Args:
            site_id: Site identifier
            environment: Environment name
            overrides: Manual configuration overrides

        Returns:
            Fully resolved configuration

        Raises:
            ConfigValidationError: If configuration is invalid
        """
        global_config = self.get_config()
        inheritance_chain = []

        # Start with global defaults
        config_data = global_config.defaults.model_dump()
        inheritance_chain.append("global_defaults")

        # Apply environment defaults
        if environment in global_config.environment_defaults:
            env_defaults = global_config.environment_defaults[environment].model_dump()
            config_data = self._deep_merge(config_data, env_defaults)
            inheritance_chain.append(f"environment_defaults.{environment}")

        # Apply site defaults
        site_config = global_config.sites.get(site_id)
        if site_config:
            site_defaults = site_config.defaults.model_dump()
            config_data = self._deep_merge(config_data, site_defaults)
            inheritance_chain.append(f"sites.{site_id}.defaults")

            # Apply site-environment specific config
            if environment in site_config.environments:
                site_env_config = site_config.environments[environment].model_dump()
                config_data = self._deep_merge(config_data, site_env_config)
                inheritance_chain.append(f"sites.{site_id}.environments.{environment}")

        # Apply manual overrides
        if overrides:
            config_data = self._deep_merge(config_data, overrides)
            inheritance_chain.append("manual_overrides")

        # Create resolved configuration
        try:
            resolved_config = EnvironmentConfig(**config_data)
        except Exception as e:
            raise ConfigValidationError(
                f"Failed to resolve configuration for {site_id}:{environment}: {e}"
            )

        # Validate required IDs
        self._validate_required_ids(site_id, environment, resolved_config, global_config)

        return ResolvedEnvironmentConfig(
            site_id=site_id,
            environment=environment,
            config=resolved_config,
            inheritance_chain=inheritance_chain
        )

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _validate_required_ids(
        self,
        site_id: str,
        environment: str,
        config: EnvironmentConfig,
        global_config: GlobalEnvironmentConfig
    ) -> None:
        """Validate that required tracking IDs are present."""
        required_ids = global_config.required_ids.get(environment, [])

        for id_field in required_ids:
            value = getattr(config, id_field, None)
            if not value:
                raise ConfigValidationError(
                    f"Required tracking ID '{id_field}' missing for {site_id}:{environment}"
                )

    def list_sites(self) -> List[str]:
        """List all configured sites."""
        return list(self.get_config().sites.keys())

    def list_environments(self, site_id: Optional[str] = None) -> List[str]:
        """List environments for a site or all environments."""
        global_config = self.get_config()

        if site_id:
            site_config = global_config.sites.get(site_id)
            if site_config:
                return list(site_config.environments.keys())
            return []
        else:
            # Return all unique environments across all sites
            environments = set(global_config.environment_defaults.keys())
            for site_config in global_config.sites.values():
                environments.update(site_config.environments.keys())
            return sorted(environments)

    def validate_configuration(self) -> List[str]:
        """Validate the entire configuration and return any errors."""
        errors = []
        global_config = self.get_config()

        for site_id, site_config in global_config.sites.items():
            for environment in site_config.environments.keys():
                try:
                    self.resolve_environment_config(site_id, environment)
                except ConfigValidationError as e:
                    errors.append(f"{site_id}:{environment} - {str(e)}")

        return errors

    def test_configuration(
        self,
        site_id: str,
        environment: str,
        print_details: bool = False
    ) -> bool:
        """Test configuration resolution for a site-environment combination."""
        try:
            resolved = self.resolve_environment_config(site_id, environment)

            if print_details:
                print(f"✅ Configuration resolved for {site_id}:{environment}")
                print(f"Inheritance chain: {' → '.join(resolved.inheritance_chain)}")
                print(f"GA4 ID: {resolved.config.ga4_measurement_id}")
                print(f"GTM ID: {resolved.config.gtm_container_id}")
                print(f"Max pages: {resolved.config.max_pages}")
                print(f"Timeout: {resolved.config.timeout_ms}ms")

            return True
        except ConfigValidationError as e:
            if print_details:
                print(f"❌ Configuration error for {site_id}:{environment}: {e}")
            return False


def create_default_environment_config() -> Dict[str, Any]:
    """Create a default environment configuration template."""
    return {
        "version": "1.0",
        "defaults": {
            "timeout_ms": 30000,
            "max_pages": 500,
            "max_concurrent": 5,
            "requests_per_second": 2.0,
            "expected_cookies": [],
            "expected_tags": [],
            "expected_events": [],
            "base_urls": [],
            "custom_params": {}
        },
        "environment_defaults": {
            "production": {
                "max_pages": 1000,
                "requests_per_second": 1.5,
            },
            "staging": {
                "max_pages": 200,
                "requests_per_second": 3.0,
            }
        },
        "sites": {
            "example": {
                "site_id": "example",
                "defaults": {
                    "base_urls": ["https://example.com"],
                    "expected_cookies": ["session_id", "user_prefs"]
                },
                "environments": {
                    "production": {
                        "ga4_measurement_id": "G-XXXXXXXXXX",
                        "gtm_container_id": "GTM-XXXXXXX",
                        "base_urls": ["https://example.com"]
                    },
                    "staging": {
                        "ga4_measurement_id": "G-YYYYYYYYYY",
                        "gtm_container_id": "GTM-YYYYYYY",
                        "base_urls": ["https://staging.example.com"]
                    }
                }
            }
        },
        "required_ids": {
            "production": ["ga4_measurement_id", "gtm_container_id"],
            "staging": ["ga4_measurement_id", "gtm_container_id"]
        }
    }