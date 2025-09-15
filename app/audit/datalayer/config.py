"""Configuration system for dataLayer integrity auditing.

This module provides configuration management for dataLayer capture, validation,
and processing settings, including YAML loading, site-specific overrides, and
environment-specific configurations.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field, field_validator

from .models import RedactionMethod, ValidationSeverity


class RedactionRuleConfig(BaseModel):
    """Configuration for a single redaction rule."""
    
    path: str = Field(description="JSON Pointer path or glob pattern")
    method: RedactionMethod = Field(
        default=RedactionMethod.HASH,
        description="Redaction method to apply"
    )
    reason: str | None = Field(
        default=None,
        description="Reason for redaction (for audit trail)"
    )


class SchemaConfig(BaseModel):
    """Configuration for schema validation."""
    
    enabled: bool = Field(default=True, description="Enable schema validation")
    schema_path: str | None = Field(
        default=None,
        description="Path to JSON/YAML schema file"
    )
    cache_schemas: bool = Field(
        default=True,
        description="Cache loaded schemas for performance"
    )
    
    # Severity mapping for different validation rule types
    severity_mapping: Dict[str, ValidationSeverity] = Field(
        default_factory=lambda: {
            "required": ValidationSeverity.CRITICAL,
            "type": ValidationSeverity.WARNING, 
            "format": ValidationSeverity.WARNING,
            "enum": ValidationSeverity.WARNING,
            "pattern": ValidationSeverity.WARNING,
            "minimum": ValidationSeverity.WARNING,
            "maximum": ValidationSeverity.WARNING,
            "minLength": ValidationSeverity.WARNING,
            "maxLength": ValidationSeverity.WARNING,
            "uniqueItems": ValidationSeverity.WARNING,
            "additionalProperties": ValidationSeverity.INFO,
            "dependencies": ValidationSeverity.WARNING,
            "if": ValidationSeverity.WARNING
        },
        description="Mapping of JSON Schema keywords to severity levels"
    )
    
    # Schema loading options
    allow_yaml_schemas: bool = Field(
        default=True,
        description="Allow loading schemas from YAML files"
    )
    resolve_references: bool = Field(
        default=True,
        description="Resolve $ref references in schemas"
    )
    strict_mode: bool = Field(
        default=False,
        description="Enable strict validation mode"
    )


class CaptureConfig(BaseModel):
    """Configuration for dataLayer capture operations."""
    
    enabled: bool = Field(default=True, description="Enable dataLayer capture")
    
    # Object identification
    object_name: str = Field(
        default="dataLayer",
        description="Name of the global object to capture"
    )
    fallback_objects: List[str] = Field(
        default_factory=lambda: ["digitalData", "_satellite", "utag_data"],
        description="Fallback object names to try if primary object missing"
    )
    
    # Capture limits
    max_depth: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum nesting depth to capture"
    )
    max_entries: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Maximum number of entries to capture"
    )
    max_size_bytes: int = Field(
        default=1048576,  # 1MB
        ge=1024,
        description="Maximum capture size in bytes"
    )
    
    # JavaScript execution
    execution_timeout_ms: int = Field(
        default=5000,
        ge=100,
        le=30000,
        description="Timeout for JavaScript execution in milliseconds"
    )
    safe_mode: bool = Field(
        default=True,
        description="Enable safe mode for JavaScript execution"
    )
    
    # Normalization options
    normalize_pushes: bool = Field(
        default=True,
        description="Normalize push-based dataLayers to latest state + events"
    )
    extract_events: bool = Field(
        default=True,
        description="Extract events from dataLayer pushes"
    )
    event_detection_patterns: List[str] = Field(
        default_factory=lambda: ["event", "eventName", "type", "action"],
        description="Patterns to identify event pushes"
    )
    
    # Performance options
    async_capture: bool = Field(
        default=True,
        description="Use asynchronous capture when possible"
    )
    batch_processing: bool = Field(
        default=False,
        description="Enable batch processing for multiple pages"
    )


class RedactionConfig(BaseModel):
    """Configuration for sensitive data redaction."""
    
    enabled: bool = Field(default=True, description="Enable data redaction")
    
    # Default redaction settings
    default_method: RedactionMethod = Field(
        default=RedactionMethod.HASH,
        description="Default redaction method"
    )
    
    # Redaction rules
    rules: List[RedactionRuleConfig] = Field(
        default_factory=lambda: [
            RedactionRuleConfig(
                path="/user/email",
                method=RedactionMethod.HASH,
                reason="PII protection"
            ),
            RedactionRuleConfig(
                path="/user/id",
                method=RedactionMethod.HASH,
                reason="User identifier"
            ),
            RedactionRuleConfig(
                path="/**/ssn",
                method=RedactionMethod.REMOVE,
                reason="Sensitive personal data"
            ),
            RedactionRuleConfig(
                path="/**/password",
                method=RedactionMethod.REMOVE,
                reason="Security credential"
            ),
            RedactionRuleConfig(
                path="/payment/**",
                method=RedactionMethod.REMOVE,
                reason="Payment information"
            )
        ],
        description="List of redaction rules to apply"
    )
    
    # Pattern-based redaction
    pattern_detection: bool = Field(
        default=True,
        description="Enable automatic pattern-based redaction"
    )
    
    patterns: Dict[str, str] = Field(
        default_factory=lambda: {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
        },
        description="Regular expression patterns for automatic detection"
    )
    
    # Advanced pattern detection settings
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for pattern detection (0.0-1.0)"
    )
    
    # Audit trail
    keep_audit_trail: bool = Field(
        default=True,
        description="Keep audit trail of redacted data"
    )
    audit_trail_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days to retain redaction audit trail"
    )


class AggregationConfig(BaseModel):
    """Configuration for run-level data aggregation."""
    
    enabled: bool = Field(default=True, description="Enable data aggregation")
    
    # Variable tracking
    track_variable_presence: bool = Field(
        default=True,
        description="Track variable presence across pages"
    )
    max_example_values: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum example values to store per variable"
    )
    
    # Event tracking
    track_event_frequency: bool = Field(
        default=True,
        description="Track event frequency and patterns"
    )
    max_example_events: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum example events to store per type"
    )
    
    # Performance tracking
    track_performance_metrics: bool = Field(
        default=True,
        description="Track processing performance metrics"
    )
    
    # Incremental updates
    incremental_aggregation: bool = Field(
        default=True,
        description="Use incremental aggregation for large runs"
    )
    aggregation_batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Batch size for incremental aggregation"
    )


class PerformanceConfig(BaseModel):
    """Configuration for performance optimization."""
    
    # Concurrency
    max_concurrent_captures: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum concurrent dataLayer captures"
    )
    
    # Caching
    enable_caching: bool = Field(
        default=True,
        description="Enable result caching"
    )
    cache_ttl_seconds: int = Field(
        default=300,  # 5 minutes
        ge=60,
        le=3600,
        description="Cache time-to-live in seconds"
    )
    
    # Memory management
    max_memory_usage_mb: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Maximum memory usage in MB"
    )
    
    # Timeouts
    page_timeout_ms: int = Field(
        default=30000,
        ge=5000,
        le=60000,
        description="Page processing timeout in milliseconds"
    )
    
    # Optimization flags
    optimize_large_objects: bool = Field(
        default=True,
        description="Enable optimizations for large dataLayer objects"
    )
    streaming_processing: bool = Field(
        default=False,
        description="Enable streaming processing for memory efficiency"
    )


class DataLayerConfig(BaseModel):
    """Root configuration for dataLayer integrity system."""
    
    # Environment settings
    environment: str = Field(
        default="production",
        description="Environment name (affects debug and validation settings)"
    )
    
    site_domain: str | None = Field(
        default=None,
        description="Primary site domain being audited"
    )
    
    # Global settings
    global_settings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enable_debug": False,
            "enable_detailed_logging": False,
            "max_processing_time_ms": 10000,
            "enable_monitoring": True,
            "fail_fast": False
        },
        description="Global dataLayer processing settings"
    )
    
    # Component configurations
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)
    validation: SchemaConfig = Field(default_factory=SchemaConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    # Site-specific overrides
    site_overrides: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Site-specific configuration overrides"
    )
    
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
    
    @property
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self.global_settings.get("enable_debug", False)
    
    def get_site_config(self, site_domain: str) -> 'DataLayerConfig':
        """Get configuration with site-specific overrides applied.
        
        Args:
            site_domain: Domain to get configuration for
            
        Returns:
            Configuration with site overrides applied
        """
        if site_domain not in self.site_overrides:
            return self
        
        # Create a copy of current config
        config_dict = self.dict()
        
        # Apply site-specific overrides
        site_config = self.site_overrides[site_domain]
        self._deep_merge(config_dict, site_config)
        
        # Create new config instance with overrides
        return DataLayerConfig(**config_dict)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Deep merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


class DataLayerConfigManager:
    """Manages dataLayer configuration loading and validation."""
    
    def __init__(self, config_path: str | Path | None = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: DataLayerConfig | None = None
        self._environment_overrides: Dict[str, Any] = {}
        self._site_configs: Dict[str, DataLayerConfig] = {}
    
    def load_config(self, config_path: str | Path | None = None) -> DataLayerConfig:
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
                raise DataLayerConfigurationError(f"Failed to load config from {self.config_path}: {e}")
        
        # Apply environment overrides
        config_data.update(self._environment_overrides)
        
        # Load environment variables
        env_config = self._load_environment_variables()
        self._merge_config(config_data, env_config)
        
        # Validate and create config object
        try:
            self._config = DataLayerConfig(**config_data)
        except Exception as e:
            raise DataLayerConfigurationError(f"Invalid dataLayer configuration: {e}")
        
        return self._config
    
    def get_config(self) -> DataLayerConfig:
        """Get current configuration, loading default if needed."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def get_site_config(self, site_domain: str) -> DataLayerConfig:
        """Get configuration for specific site with overrides.
        
        Args:
            site_domain: Site domain to get configuration for
            
        Returns:
            Site-specific configuration
        """
        if site_domain not in self._site_configs:
            base_config = self.get_config()
            self._site_configs[site_domain] = base_config.get_site_config(site_domain)
        
        return self._site_configs[site_domain]
    
    def set_environment_override(self, key: str, value: Any) -> None:
        """Set environment-specific configuration override.
        
        Args:
            key: Configuration key to override
            value: Override value
        """
        self._environment_overrides[key] = value
    
    def validate_config(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate configuration data without loading.
        
        Args:
            config_data: Configuration data to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        try:
            DataLayerConfig(**config_data)
        except Exception as e:
            errors.append(str(e))
        
        return errors
    
    def create_default_config(self, output_path: str | Path) -> None:
        """Create default configuration file.
        
        Args:
            output_path: Path where to write the default config
        """
        default_config = DataLayerConfig()
        config_dict = default_config.dict()
        
        # Add example site overrides
        config_dict['site_overrides'] = {
            'ecommerce.example.com': {
                'capture': {
                    'object_name': 'digitalData',
                    'max_depth': 8
                },
                'redaction': {
                    'rules': [
                        {
                            'path': '/user/email',
                            'method': 'hash',
                            'reason': 'PII protection'
                        },
                        {
                            'path': '/cart/*/price',
                            'method': 'mask',
                            'reason': 'Commercial sensitivity'
                        }
                    ]
                },
                'validation': {
                    'schema_path': 'schemas/ecommerce-datalayer.json'
                }
            },
            'blog.example.com': {
                'capture': {
                    'max_entries': 100,
                    'extract_events': False
                },
                'aggregation': {
                    'track_event_frequency': False
                }
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Environment name
        if env_env := os.getenv('TAG_SENTINEL_ENVIRONMENT'):
            env_config['environment'] = env_env
        
        # Site domain
        if env_domain := os.getenv('TAG_SENTINEL_SITE_DOMAIN'):
            env_config['site_domain'] = env_domain
        
        # DataLayer object name
        if env_object := os.getenv('TAG_SENTINEL_DATALAYER_OBJECT'):
            env_config.setdefault('capture', {})['object_name'] = env_object
        
        # Debug mode
        if os.getenv('TAG_SENTINEL_DEBUG') == 'true':
            env_config.setdefault('global_settings', {})['enable_debug'] = True
        
        # Schema validation
        if os.getenv('TAG_SENTINEL_SCHEMA_VALIDATION') == 'false':
            env_config.setdefault('validation', {})['enabled'] = False
        
        # Redaction
        if os.getenv('TAG_SENTINEL_REDACTION') == 'false':
            env_config.setdefault('redaction', {})['enabled'] = False
        
        # Performance settings
        if env_concurrency := os.getenv('TAG_SENTINEL_MAX_CONCURRENT'):
            try:
                env_config.setdefault('performance', {})['max_concurrent_captures'] = int(env_concurrency)
            except ValueError:
                pass
        
        return env_config
    
    def _merge_config(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> None:
        """Deep merge configuration dictionaries."""
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value


class DataLayerConfigurationError(Exception):
    """DataLayer configuration-related errors."""
    pass


# Global configuration manager instance
datalayer_config_manager = DataLayerConfigManager()


def get_datalayer_config() -> DataLayerConfig:
    """Get current dataLayer configuration."""
    return datalayer_config_manager.get_config()


def load_datalayer_config(config_path: str | Path) -> DataLayerConfig:
    """Load dataLayer configuration from specified path."""
    return datalayer_config_manager.load_config(config_path)


def get_site_datalayer_config(site_domain: str) -> DataLayerConfig:
    """Get dataLayer configuration for specific site with overrides."""
    return datalayer_config_manager.get_site_config(site_domain)


class ConfigurationValidator:
    """Validates dataLayer configuration with detailed error reporting."""
    
    def __init__(self):
        """Initialize configuration validator."""
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
    
    def validate_comprehensive(self, config: DataLayerConfig) -> Dict[str, Any]:
        """Perform comprehensive configuration validation.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result with errors, warnings, and recommendations
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        # Validate capture configuration
        self._validate_capture_config(config.capture)
        
        # Validate schema configuration
        self._validate_schema_config(config.validation)
        
        # Validate redaction configuration
        self._validate_redaction_config(config.redaction)
        
        # Validate aggregation configuration
        self._validate_aggregation_config(config.aggregation)
        
        # Cross-configuration validation
        self._validate_cross_config_consistency(config)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(config)
        
        return {
            'valid': len(self.validation_errors) == 0,
            'errors': self.validation_errors.copy(),
            'warnings': self.validation_warnings.copy(),
            'recommendations': recommendations,
            'config_score': self._calculate_config_score()
        }
    
    def _validate_capture_config(self, capture: CaptureConfig) -> None:
        """Validate capture configuration."""
        # Depth validation
        if capture.max_depth > 10:
            self.validation_warnings.append(
                f"max_depth ({capture.max_depth}) is very high - may impact performance"
            )
        elif capture.max_depth < 3:
            self.validation_warnings.append(
                f"max_depth ({capture.max_depth}) is low - may miss nested data"
            )
        
        # Size validation
        if capture.max_size_bytes > 10 * 1024 * 1024:  # 10MB
            self.validation_warnings.append(
                f"max_size_bytes ({capture.max_size_bytes}) is very large - may cause memory issues"
            )
        
        # Object name validation
        if not capture.object_name:
            self.validation_errors.append("object_name cannot be empty")
        
        # Fallback objects validation
        if not capture.fallback_objects:
            self.validation_warnings.append("No fallback objects configured - may miss data on some sites")
    
    def _validate_schema_config(self, schema: SchemaConfig) -> None:
        """Validate schema configuration."""
        if schema.enabled and not schema.schema_path:
            self.validation_warnings.append(
                "Schema validation enabled but no schema_path specified"
            )
        
        # Check if schema file exists
        if schema.schema_path and not Path(schema.schema_path).exists():
            self.validation_errors.append(
                f"Schema file not found: {schema.schema_path}"
            )
    
    def _validate_redaction_config(self, redaction: RedactionConfig) -> None:
        """Validate redaction configuration."""
        if redaction.enabled:
            # Pattern validation
            if redaction.pattern_detection and not redaction.patterns:
                self.validation_warnings.append(
                    "Pattern detection enabled but no patterns configured"
                )
            
            # Confidence threshold validation
            if not 0.0 <= redaction.confidence_threshold <= 1.0:
                self.validation_errors.append(
                    f"confidence_threshold ({redaction.confidence_threshold}) must be between 0.0 and 1.0"
                )
            
            # Rules validation
            if not redaction.rules:
                self.validation_warnings.append("No redaction rules configured")
    
    def _validate_aggregation_config(self, aggregation: AggregationConfig) -> None:
        """Validate aggregation configuration."""
        if aggregation.max_example_values > 20:
            self.validation_warnings.append(
                f"max_example_values ({aggregation.max_example_values}) is high - may use excessive memory"
            )
    
    def _validate_cross_config_consistency(self, config: DataLayerConfig) -> None:
        """Validate consistency across configuration sections."""
        # Check if redaction and schema validation are compatible
        if config.redaction.enabled and config.validation.enabled:
            if config.redaction.keep_audit_trail and config.validation.strict_mode:
                self.validation_warnings.append(
                    "Strict schema mode with redaction audit trail may expose sensitive data"
                )
    
    def _generate_recommendations(self, config: DataLayerConfig) -> List[str]:
        """Generate configuration recommendations."""
        recommendations = []
        
        # Performance recommendations
        if config.capture.max_depth > 8 and config.aggregation.enabled:
            recommendations.append(
                "Consider reducing max_depth for better performance with aggregation enabled"
            )
        
        # Security recommendations
        if not config.redaction.enabled:
            recommendations.append(
                "Enable redaction to protect sensitive data in dataLayer captures"
            )
        
        # Reliability recommendations
        if not config.capture.fallback_objects:
            recommendations.append(
                "Configure fallback objects for better data capture reliability"
            )
        
        return recommendations
    
    def _calculate_config_score(self) -> float:
        """Calculate configuration quality score (0-100)."""
        base_score = 100.0
        
        # Deduct for errors and warnings
        base_score -= len(self.validation_errors) * 20
        base_score -= len(self.validation_warnings) * 5
        
        return max(0.0, min(100.0, base_score))


class RuntimeConfigurationManager:
    """Manages runtime configuration updates and hot reloading."""
    
    def __init__(self, config_manager: DataLayerConfigManager):
        """Initialize runtime configuration manager.
        
        Args:
            config_manager: Base configuration manager
        """
        self.config_manager = config_manager
        self.runtime_overrides: Dict[str, Any] = {}
        self.config_watchers: List[Callable[[DataLayerConfig], None]] = []
        self.validation_enabled = True
        self.validator = ConfigurationValidator()
    
    def update_runtime_config(
        self,
        config_updates: Dict[str, Any],
        validate: bool = True
    ) -> Dict[str, Any]:
        """Update configuration at runtime.
        
        Args:
            config_updates: Configuration updates to apply
            validate: Whether to validate updates before applying
            
        Returns:
            Result of the update operation
        """
        if validate and self.validation_enabled:
            # Create temporary config with updates
            current_config = self.config_manager.get_config()
            temp_config_data = current_config.dict()
            self._merge_config_updates(temp_config_data, config_updates)
            
            try:
                temp_config = DataLayerConfig(**temp_config_data)
                validation_result = self.validator.validate_comprehensive(temp_config)
                
                if not validation_result['valid']:
                    return {
                        'success': False,
                        'message': 'Configuration validation failed',
                        'errors': validation_result['errors']
                    }
                    
            except Exception as e:
                return {
                    'success': False,
                    'message': f'Configuration update failed: {e}',
                    'errors': [str(e)]
                }
        
        # Apply updates
        try:
            self._merge_config_updates(self.runtime_overrides, config_updates)
            
            # Force config reload with runtime overrides
            self.config_manager._environment_overrides.update(self.runtime_overrides)
            new_config = self.config_manager.load_config()
            
            # Notify watchers
            self._notify_config_watchers(new_config)
            
            return {
                'success': True,
                'message': 'Configuration updated successfully',
                'config': new_config.dict()
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to apply configuration updates: {e}',
                'errors': [str(e)]
            }
    
    def register_config_watcher(
        self,
        callback: Callable[[DataLayerConfig], None]
    ) -> None:
        """Register a callback to be notified of configuration changes.
        
        Args:
            callback: Callback function to register
        """
        self.config_watchers.append(callback)
    
    def reset_runtime_overrides(self) -> None:
        """Reset all runtime configuration overrides."""
        self.runtime_overrides.clear()
        self.config_manager._environment_overrides.clear()
        
        # Reload base configuration
        self.config_manager.load_config()
        
        # Notify watchers
        new_config = self.config_manager.get_config()
        self._notify_config_watchers(new_config)
    
    def get_runtime_overrides(self) -> Dict[str, Any]:
        """Get current runtime configuration overrides.
        
        Returns:
            Dictionary of runtime overrides
        """
        return self.runtime_overrides.copy()
    
    def _merge_config_updates(
        self,
        base_config: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> None:
        """Merge configuration updates into base configuration."""
        for key, value in updates.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config_updates(base_config[key], value)
            else:
                base_config[key] = value
    
    def _notify_config_watchers(self, config: DataLayerConfig) -> None:
        """Notify all registered configuration watchers."""
        for watcher in self.config_watchers:
            try:
                watcher(config)
            except Exception as e:
                logger.error(f"Configuration watcher {watcher.__name__} failed: {e}")


class ConfigurationTemplateManager:
    """Manages configuration templates for different use cases."""
    
    def __init__(self):
        """Initialize configuration template manager."""
        self.templates: Dict[str, Dict[str, Any]] = {}
        self._load_builtin_templates()
    
    def _load_builtin_templates(self) -> None:
        """Load built-in configuration templates."""
        
        # Production template
        self.templates['production'] = {
            'capture': {
                'enabled': True,
                'max_depth': 6,
                'max_size_bytes': 1048576,  # 1MB
                'timeout_ms': 5000,
                'safe_execution': True,
                'monitor_performance': True
            },
            'schema': {
                'enabled': True,
                'cache_schemas': True,
                'strict_mode': False
            },
            'redaction': {
                'enabled': True,
                'pattern_detection': True,
                'confidence_threshold': 0.8,
                'keep_audit_trail': True
            },
            'aggregation': {
                'enabled': True,
                'max_example_values': 3,
                'track_variable_presence': True,
                'track_event_frequency': True
            }
        }
        
        # Development template
        self.templates['development'] = {
            'capture': {
                'enabled': True,
                'max_depth': 10,
                'max_size_bytes': 5242880,  # 5MB
                'timeout_ms': 10000,
                'safe_execution': False,
                'monitor_performance': False
            },
            'schema': {
                'enabled': True,
                'cache_schemas': False,
                'strict_mode': False
            },
            'redaction': {
                'enabled': False,
                'pattern_detection': False,
                'confidence_threshold': 0.5
            },
            'aggregation': {
                'enabled': True,
                'max_example_values': 10,
                'track_variable_presence': True,
                'track_event_frequency': True
            }
        }
        
        # Testing template
        self.templates['testing'] = {
            'capture': {
                'enabled': True,
                'max_depth': 8,
                'max_size_bytes': 2097152,  # 2MB
                'timeout_ms': 3000,
                'safe_execution': True,
                'monitor_performance': True
            },
            'schema': {
                'enabled': True,
                'cache_schemas': True,
                'strict_mode': True
            },
            'redaction': {
                'enabled': True,
                'pattern_detection': True,
                'confidence_threshold': 0.9,
                'keep_audit_trail': False
            },
            'aggregation': {
                'enabled': False
            }
        }
        
        # Minimal template
        self.templates['minimal'] = {
            'capture': {
                'enabled': True,
                'max_depth': 3,
                'max_size_bytes': 262144,  # 256KB
                'timeout_ms': 2000
            },
            'schema': {
                'enabled': False
            },
            'redaction': {
                'enabled': False
            },
            'aggregation': {
                'enabled': False
            }
        }
    
    def get_template(self, template_name: str) -> Dict[str, Any]:
        """Get configuration template by name.
        
        Args:
            template_name: Name of template to retrieve
            
        Returns:
            Template configuration dictionary
            
        Raises:
            ValueError: If template not found
        """
        if template_name not in self.templates:
            available = ', '.join(self.templates.keys())
            raise ValueError(f"Template '{template_name}' not found. Available: {available}")
        
        return self.templates[template_name].copy()
    
    def list_templates(self) -> List[str]:
        """Get list of available template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def add_template(self, name: str, template: Dict[str, Any]) -> None:
        """Add a custom configuration template.
        
        Args:
            name: Template name
            template: Template configuration
        """
        self.templates[name] = template.copy()
    
    def create_config_from_template(
        self,
        template_name: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> DataLayerConfig:
        """Create configuration from template with optional overrides.
        
        Args:
            template_name: Name of template to use
            overrides: Optional configuration overrides
            
        Returns:
            DataLayer configuration instance
        """
        template_config = self.get_template(template_name)
        
        if overrides:
            self._deep_merge_dict(template_config, overrides)
        
        return DataLayerConfig(**template_config)
    
    def _deep_merge_dict(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Deep merge dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(base[key], value)
            else:
                base[key] = value


# Global instances
config_validator = ConfigurationValidator()
template_manager = ConfigurationTemplateManager()
runtime_manager = RuntimeConfigurationManager(datalayer_config_manager)


def validate_datalayer_config(config: DataLayerConfig) -> Dict[str, Any]:
    """Validate a dataLayer configuration comprehensively."""
    return config_validator.validate_comprehensive(config)


def create_config_from_template(
    template_name: str,
    overrides: Optional[Dict[str, Any]] = None
) -> DataLayerConfig:
    """Create configuration from a template."""
    return template_manager.create_config_from_template(template_name, overrides)


def update_runtime_datalayer_config(config_updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update dataLayer configuration at runtime."""
    return runtime_manager.update_runtime_config(config_updates)


def create_default_datalayer_config(output_path: str | Path) -> None:
    """Create default dataLayer configuration file."""
    datalayer_config_manager.create_default_config(output_path)