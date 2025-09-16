"""Configuration system for analytics detectors.

This module provides configuration management for detector settings,
including YAML loading, validation, and environment-specific overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator

from ..models.capture import PageResult
from .base import DetectorRegistry


class EndpointConfig(BaseModel):
    """Configuration for analytics endpoint matching."""
    
    pattern: str = Field(description="URL pattern to match")
    enabled: bool = Field(default=True, description="Whether this endpoint is enabled")
    timeout_ms: Optional[int] = Field(
        default=None, 
        description="Override timeout for this endpoint"
    )
    max_retries: Optional[int] = Field(
        default=None,
        description="Override retry count for this endpoint"
    )


class GA4Config(BaseModel):
    """Configuration for GA4 detector."""
    
    enabled: bool = Field(default=True, description="Enable GA4 detection")
    
    endpoints: List[EndpointConfig] = Field(
        default_factory=lambda: [
            EndpointConfig(pattern="https://www.google-analytics.com/mp/collect"),
            EndpointConfig(pattern="https://region1.google-analytics.com/mp/collect"),
            EndpointConfig(pattern="https://*.google-analytics.com/mp/collect"),
            EndpointConfig(pattern="https://www.google-analytics.com/g/collect")
        ],
        description="Analytics endpoints to detect"
    )
    
    mp_debug: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": False,  # Non-prod only
            "timeout_ms": 5000,
            "max_validation_requests": 10
        },
        description="Measurement Protocol debug validation settings"
    )
    
    parameter_extraction: Dict[str, Any] = Field(
        default_factory=lambda: {
            "extract_client_info": True,
            "extract_page_info": True, 
            "extract_ecommerce": True,
            "extract_custom_dimensions": True,
            "max_parameters": 100,
            "sanitize_sensitive": True
        },
        description="Parameter extraction settings"
    )


class GTMConfig(BaseModel):
    """Configuration for GTM detector."""
    
    enabled: bool = Field(default=True, description="Enable GTM detection")
    
    container_validation: bool = Field(
        default=True,
        description="Validate container ID format"
    )
    
    validate_datalayer: bool = Field(
        default=True,
        description="Validate dataLayer presence and structure"
    )

    expected_container_ids: List[str] = Field(
        default_factory=list,
        description="Expected GTM container IDs (empty list means no validation)"
    )
    
    datalayer_extraction: Dict[str, Any] = Field(
        default_factory=lambda: {
            "check_html_responses": True,
            "check_js_responses": True,
            "max_event_extraction": 50,
            "validate_structure": True
        },
        description="DataLayer extraction settings"
    )
    
    performance_thresholds: Dict[str, Any] = Field(
        default_factory=lambda: {
            "slow_load_ms": 3000,
            "max_container_count": 5
        },
        description="Performance warning thresholds"
    )


class DuplicateConfig(BaseModel):
    """Configuration for duplicate detection."""
    
    enabled: bool = Field(default=True, description="Enable duplicate detection")
    
    window_ms: int = Field(
        default=4000,
        ge=100,
        le=60000,
        description="Time window for duplicate detection in milliseconds"
    )
    
    similarity_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for duplicate detection"
    )
    
    hash_algorithm: str = Field(
        default="md5",
        description="Hashing algorithm for event canonicalization"
    )
    
    ignore_parameters: List[str] = Field(
        default_factory=lambda: ["timestamp", "cb", "random", "_"],
        description="Parameters to ignore during duplicate detection"
    )


class SequencingConfig(BaseModel):
    """Configuration for tag sequencing analysis."""
    
    enabled: bool = Field(default=True, description="Enable sequencing analysis")
    
    rules: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "name": "GTM before GA4",
                "first": {"vendor": "gtm", "pattern": "container_load"},
                "second": {"vendor": "ga4", "pattern": "page_view"},
                "max_delay_ms": 2000,
                "severity": "warning"
            }
        ],
        description="Sequencing rules to validate"
    )
    
    timing_tolerance_ms: int = Field(
        default=100,
        ge=0,
        le=5000,
        description="Timing tolerance for sequence validation"
    )


class DetectorConfig(BaseModel):
    """Root configuration for all detectors."""
    
    # Environment settings
    environment: str = Field(
        default="production",
        description="Environment name (affects debug settings)"
    )
    
    site_domain: Optional[str] = Field(
        default=None,
        description="Primary site domain being audited"
    )
    
    # Global settings
    global_settings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_processing_time_ms": 5000,
            "max_events_per_page": 1000,
            "enable_debug": False,
            "enable_external_validation": False
        },
        description="Global detector settings"
    )
    
    # Individual detector configurations
    ga4: GA4Config = Field(default_factory=GA4Config)
    gtm: GTMConfig = Field(default_factory=GTMConfig) 
    duplicates: DuplicateConfig = Field(default_factory=DuplicateConfig)
    sequencing: SequencingConfig = Field(default_factory=SequencingConfig)
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment name."""
        allowed_envs = ['development', 'staging', 'production', 'test']
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of: {allowed_envs}")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def get_detector_config(self, detector_name: str) -> Dict[str, Any]:
        """Get configuration for a specific detector.
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            Configuration dictionary for the detector
        """
        detector_configs = {
            "GA4Detector": self.ga4.dict(),
            "GTMDetector": self.gtm.dict(),
            "DuplicateAnalyzer": self.duplicates.dict(),
            "SequencingAnalyzer": self.sequencing.dict()
        }
        
        config = detector_configs.get(detector_name, {})
        
        # Add global settings
        config.update({
            "environment": self.environment,
            "site_domain": self.site_domain,
            "is_production": self.is_production,
            **self.global_settings
        })
        
        return config


class ConfigManager:
    """Manages detector configuration loading and validation."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[DetectorConfig] = None
        self._environment_overrides: Dict[str, Any] = {}
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> DetectorConfig:
        """Load configuration from file.
        
        Args:
            config_path: Optional override for config file path
            
        Returns:
            Loaded and validated configuration
        """
        if config_path:
            self.config_path = Path(config_path)
        
        config_data = {}
        
        # Load from file if specified
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
            except Exception as e:
                raise ConfigurationError(f"Failed to load config from {self.config_path}: {e}")
        
        # Apply environment overrides
        config_data.update(self._environment_overrides)
        
        # Load environment variables
        env_config = self._load_environment_variables()
        self._merge_config(config_data, env_config)
        
        # Validate and create config object
        try:
            self._config = DetectorConfig(**config_data)
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {e}")
        
        return self._config
    
    def get_config(self) -> DetectorConfig:
        """Get current configuration, loading default if needed."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def set_environment_override(self, key: str, value: Any) -> None:
        """Set environment-specific configuration override.
        
        Args:
            key: Configuration key to override
            value: Override value
        """
        self._environment_overrides[key] = value
    
    def configure_registry(self, registry: DetectorRegistry) -> None:
        """Configure detector registry with current settings.
        
        Args:
            registry: Detector registry to configure
        """
        config = self.get_config()
        
        # Configure individual detectors based on enabled state
        detector_configs = {
            "GA4Detector": config.ga4.enabled,
            "GTMDetector": config.gtm.enabled,
            "DuplicateAnalyzer": config.duplicates.enabled,
            "SequencingAnalyzer": config.sequencing.enabled
        }
        
        for detector_name, enabled in detector_configs.items():
            registry.set_enabled(detector_name, enabled)
    
    def validate_config(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate configuration data without loading.
        
        Args:
            config_data: Configuration data to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        try:
            DetectorConfig(**config_data)
        except Exception as e:
            errors.append(str(e))
        
        return errors
    
    def create_default_config(self, output_path: Union[str, Path]) -> None:
        """Create default configuration file.
        
        Args:
            output_path: Path where to write the default config
        """
        default_config = DetectorConfig()
        config_dict = default_config.dict()
        
        # Add comments and structure for YAML output
        structured_config = self._add_config_comments(config_dict)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(structured_config, f, default_flow_style=False, indent=2)
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Environment name
        if env_env := os.getenv('TAG_SENTINEL_ENVIRONMENT'):
            env_config['environment'] = env_env
        
        # Site domain
        if env_domain := os.getenv('TAG_SENTINEL_SITE_DOMAIN'):
            env_config['site_domain'] = env_domain
        
        # GA4 MP Debug (non-prod only)
        if os.getenv('TAG_SENTINEL_GA4_MP_DEBUG') == 'true':
            env_config.setdefault('ga4', {})['mp_debug'] = {'enabled': True}
        
        # Debug mode
        if os.getenv('TAG_SENTINEL_DEBUG') == 'true':
            env_config.setdefault('global_settings', {})['enable_debug'] = True
        
        return env_config
    
    def _merge_config(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> None:
        """Deep merge configuration dictionaries."""
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def _add_config_comments(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add structure and comments for YAML output."""
        # This is a simplified version - a full implementation would
        # add proper YAML comments and structure
        return config_dict


class ConfigurationError(Exception):
    """Configuration-related errors."""
    pass


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> DetectorConfig:
    """Get current detector configuration."""
    return config_manager.get_config()


def load_config(config_path: Union[str, Path]) -> DetectorConfig:
    """Load configuration from specified path."""
    return config_manager.load_config(config_path)


def configure_detectors(registry: DetectorRegistry, config_path: Optional[Union[str, Path]] = None) -> DetectorConfig:
    """Configure detector registry with settings from config file.
    
    Args:
        registry: Detector registry to configure
        config_path: Optional path to configuration file
        
    Returns:
        Loaded configuration
    """
    if config_path:
        config = config_manager.load_config(config_path)
    else:
        config = config_manager.get_config()
    
    config_manager.configure_registry(registry)
    return config