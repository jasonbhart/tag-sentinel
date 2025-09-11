"""Configuration validation for DataLayer integrity system.

This module provides comprehensive validation for DataLayer configuration objects,
ensuring they are properly configured with valid values, dependencies, and constraints.
"""

import re
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
from datetime import datetime

from .models import RedactionMethod, ValidationSeverity
from .config import (
    DataLayerConfig, CaptureConfig, RedactionConfig, SchemaConfig,
    AggregationConfig, PerformanceConfig, RedactionRuleConfig
)

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Configuration validation errors."""
    
    def __init__(self, component: str, field: str, message: str, value: Any = None):
        self.component = component
        self.field = field
        self.message = message
        self.value = value
        
        super().__init__(f"Config validation error in {component}.{field}: {message}")


class ConfigValidationWarning:
    """Non-critical configuration validation issue."""
    
    def __init__(self, component: str, field: str, message: str, value: Any = None):
        self.component = component
        self.field = field
        self.message = message
        self.value = value
        
    def __str__(self) -> str:
        return f"Config warning in {self.component}.{self.field}: {self.message}"


class ConfigValidator:
    """Validates DataLayer configuration objects with comprehensive checks."""
    
    def __init__(self, strict_mode: bool = False):
        """Initialize validator.
        
        Args:
            strict_mode: Enable strict validation (warnings become errors)
        """
        self.strict_mode = strict_mode
        self.errors: List[ConfigValidationError] = []
        self.warnings: List[ConfigValidationWarning] = []
        
    def validate_config(self, config: DataLayerConfig) -> Tuple[bool, List[str]]:
        """Validate complete DataLayer configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        logger.info("Validating DataLayer configuration")
        
        # Clear previous results
        self.errors.clear()
        self.warnings.clear()
        
        # Validate each component
        self._validate_root_config(config)
        self._validate_capture_config(config.capture)
        self._validate_redaction_config(config.redaction)
        self._validate_schema_config(config.validation)
        self._validate_aggregation_config(config.aggregation)
        self._validate_performance_config(config.performance)
        
        # Cross-component validation
        self._validate_configuration_consistency(config)
        
        # Environmental validation
        self._validate_environment_settings(config)
        
        # Collect all error messages
        error_messages = []
        
        # Add actual errors
        for error in self.errors:
            error_messages.append(str(error))
        
        # In strict mode, warnings become errors
        if self.strict_mode:
            for warning in self.warnings:
                error_messages.append(f"STRICT: {warning}")
        else:
            # Log warnings
            for warning in self.warnings:
                logger.warning(str(warning))
        
        is_valid = len(self.errors) == 0 and (not self.strict_mode or len(self.warnings) == 0)
        
        logger.info(f"Configuration validation {'passed' if is_valid else 'failed'} "
                   f"with {len(self.errors)} errors and {len(self.warnings)} warnings")
        
        return is_valid, error_messages
    
    def _validate_root_config(self, config: DataLayerConfig) -> None:
        """Validate root configuration settings."""
        # Environment validation
        if not config.environment:
            self._add_error('root', 'environment', "Environment cannot be empty")
        
        # Global settings validation
        if not isinstance(config.global_settings, dict):
            self._add_error('root', 'global_settings', "Global settings must be a dictionary")
        else:
            self._validate_global_settings(config.global_settings)
        
        # Site domain validation
        if config.site_domain:
            if not self._is_valid_domain(config.site_domain):
                self._add_error('root', 'site_domain', f"Invalid domain format: {config.site_domain}")
    
    def _validate_capture_config(self, config: CaptureConfig) -> None:
        """Validate capture configuration."""
        # Object name validation
        if not config.object_name:
            self._add_error('capture', 'object_name', "Object name cannot be empty")
        elif not self._is_valid_js_identifier(config.object_name):
            self._add_error('capture', 'object_name', f"Invalid JavaScript identifier: {config.object_name}")
        
        # Fallback objects validation
        for i, fallback in enumerate(config.fallback_objects):
            if not self._is_valid_js_identifier(fallback):
                self._add_error('capture', f'fallback_objects[{i}]', f"Invalid JavaScript identifier: {fallback}")
        
        # Limits validation
        if config.max_depth > 10:
            self._add_warning('capture', 'max_depth', 
                            f"Max depth {config.max_depth} is very high, may cause performance issues")
        
        if config.max_entries > 1000:
            self._add_warning('capture', 'max_entries',
                            f"Max entries {config.max_entries} is very high, may cause memory issues")
        
        if config.max_size_bytes > 5 * 1024 * 1024:  # 5MB
            self._add_warning('capture', 'max_size_bytes',
                            f"Max size {config.max_size_bytes} bytes is very high, may cause memory issues")
        
        # Timeout validation
        if config.execution_timeout_ms < 1000:
            self._add_warning('capture', 'execution_timeout_ms',
                            "Execution timeout less than 1 second may be too short for complex pages")
        
        if config.execution_timeout_ms > 15000:
            self._add_warning('capture', 'execution_timeout_ms',
                            "Execution timeout greater than 15 seconds may cause blocking")
        
        # Event detection patterns validation
        for i, pattern in enumerate(config.event_detection_patterns):
            if not pattern or not isinstance(pattern, str):
                self._add_error('capture', f'event_detection_patterns[{i}]',
                              "Event detection pattern cannot be empty")
    
    def _validate_redaction_config(self, config: RedactionConfig) -> None:
        """Validate redaction configuration."""
        # Rules validation
        path_set = set()
        for i, rule in enumerate(config.rules):
            # Check for duplicate paths
            if rule.path in path_set:
                self._add_warning('redaction', f'rules[{i}].path',
                                f"Duplicate redaction path: {rule.path}")
            path_set.add(rule.path)
            
            # Validate JSON Pointer paths
            if not self._is_valid_json_pointer_or_glob(rule.path):
                self._add_error('redaction', f'rules[{i}].path',
                              f"Invalid JSON Pointer or glob pattern: {rule.path}")
            
            # Validate rule combinations
            if rule.method == RedactionMethod.TRUNCATE and '/password' in rule.path:
                self._add_warning('redaction', f'rules[{i}]',
                                "TRUNCATE method not recommended for passwords, consider REMOVE")
        
        # Patterns validation
        for name, pattern in config.patterns.items():
            try:
                re.compile(pattern)
            except re.error as e:
                self._add_error('redaction', f'patterns[{name}]',
                              f"Invalid regular expression: {e}")
        
        # Audit trail validation
        if config.keep_audit_trail and config.audit_trail_retention_days < 7:
            self._add_warning('redaction', 'audit_trail_retention_days',
                            "Audit trail retention less than 7 days may be insufficient for compliance")
    
    def _validate_schema_config(self, config: SchemaConfig) -> None:
        """Validate schema configuration."""
        # Schema path validation
        if config.schema_path:
            schema_path = Path(config.schema_path)
            if not schema_path.exists():
                self._add_error('validation', 'schema_path',
                              f"Schema file does not exist: {config.schema_path}")
            elif not schema_path.suffix.lower() in ['.json', '.yaml', '.yml']:
                if not config.allow_yaml_schemas and schema_path.suffix.lower() in ['.yaml', '.yml']:
                    self._add_error('validation', 'schema_path',
                                  "YAML schemas are disabled but schema file is YAML")
        
        # Severity mapping validation
        for keyword, severity in config.severity_mapping.items():
            if not isinstance(severity, ValidationSeverity):
                self._add_error('validation', f'severity_mapping[{keyword}]',
                              f"Invalid severity level: {severity}")
        
        # Required schema keywords check
        required_keywords = ['required', 'type']
        for keyword in required_keywords:
            if keyword not in config.severity_mapping:
                self._add_warning('validation', 'severity_mapping',
                                f"Missing severity mapping for important keyword: {keyword}")
    
    def _validate_aggregation_config(self, config: AggregationConfig) -> None:
        """Validate aggregation configuration."""
        # Example limits validation
        if config.max_example_values > 5:
            self._add_warning('aggregation', 'max_example_values',
                            "More than 5 example values may use excessive memory")
        
        if config.max_example_events > 5:
            self._add_warning('aggregation', 'max_example_events',
                            "More than 5 example events may use excessive memory")
    
    def _validate_performance_config(self, config: PerformanceConfig) -> None:
        """Validate performance configuration."""
        # Concurrency validation
        if config.max_concurrent_captures > 20:
            self._add_warning('performance', 'max_concurrent_captures',
                            "High concurrency may overwhelm target sites or cause blocking")
        
        if config.max_concurrent_captures < 1:
            self._add_error('performance', 'max_concurrent_captures',
                          "Must allow at least 1 concurrent capture")
        
        # Memory validation
        if config.max_memory_usage_mb > 1024:
            self._add_warning('performance', 'max_memory_usage_mb',
                            "High memory limit may cause system resource issues")
        
        if config.max_memory_usage_mb < 64:
            self._add_error('performance', 'max_memory_usage_mb',
                          "Memory limit too low, may cause processing failures")
        
        # Cache validation
        if config.cache_ttl_seconds < 60:
            self._add_warning('performance', 'cache_ttl_seconds',
                            "Very short cache TTL may reduce performance benefits")
        
        # Timeout validation
        if config.page_timeout_ms < 10000:
            self._add_warning('performance', 'page_timeout_ms',
                            "Page timeout less than 10 seconds may be too short for slow pages")
    
    def _validate_configuration_consistency(self, config: DataLayerConfig) -> None:
        """Validate consistency across configuration components."""
        # Redaction + Schema consistency
        if config.redaction.enabled and config.validation.enabled:
            if config.validation.strict_mode and len(config.redaction.rules) == 0:
                self._add_warning('consistency', 'redaction_schema',
                                "Strict schema validation with no redaction rules may expose sensitive data")
        
        # Performance + Capture consistency
        if config.performance.max_concurrent_captures > 1 and not config.capture.batch_processing:
            self._add_warning('consistency', 'performance_capture',
                            "High concurrency enabled but batch processing disabled")
        
        if config.capture.max_size_bytes > config.performance.max_memory_usage_mb * 1024 * 1024 // 2:
            self._add_warning('consistency', 'memory_limits',
                            "Capture size limit is large relative to memory limit")
        
        # Environment-specific consistency
        if config.is_production:
            if config.global_settings.get('enable_debug', False):
                self._add_warning('consistency', 'production_debug',
                                "Debug mode enabled in production environment")
            
            if not config.redaction.enabled:
                self._add_error('consistency', 'production_redaction',
                              "Redaction must be enabled in production environment")
    
    def _validate_environment_settings(self, config: DataLayerConfig) -> None:
        """Validate environment-specific settings."""
        env = config.environment
        
        # Development environment
        if env == 'development':
            if not config.global_settings.get('enable_debug', True):
                self._add_warning('environment', 'development',
                                "Debug mode typically enabled in development")
        
        # Production environment
        elif env == 'production':
            if config.capture.execution_timeout_ms > 10000:
                self._add_warning('environment', 'production',
                                "Long execution timeouts may impact user experience in production")
            
            if config.global_settings.get('enable_detailed_logging', True):
                self._add_warning('environment', 'production',
                                "Detailed logging may impact performance and log volume in production")
        
        # Test environment
        elif env == 'test':
            if not config.global_settings.get('fail_fast', True):
                self._add_warning('environment', 'test',
                                "Fail-fast mode typically enabled in test environment")
    
    def _validate_global_settings(self, settings: Dict[str, Any]) -> None:
        """Validate global settings dictionary."""
        # Check for required settings
        recommended_settings = [
            'enable_debug', 'enable_detailed_logging', 'max_processing_time_ms',
            'enable_monitoring', 'fail_fast'
        ]
        
        for setting in recommended_settings:
            if setting not in settings:
                self._add_warning('global_settings', setting,
                                f"Recommended global setting missing: {setting}")
        
        # Validate specific settings
        if 'max_processing_time_ms' in settings:
            max_time = settings['max_processing_time_ms']
            if not isinstance(max_time, int) or max_time < 1000:
                self._add_error('global_settings', 'max_processing_time_ms',
                              "Max processing time must be at least 1000ms")
    
    def _is_valid_domain(self, domain: str) -> bool:
        """Check if domain format is valid."""
        domain_pattern = r'^([a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(domain_pattern, domain))
    
    def _is_valid_js_identifier(self, identifier: str) -> bool:
        """Check if string is a valid JavaScript identifier."""
        if not identifier:
            return False
        
        # JavaScript identifier pattern
        js_pattern = r'^[a-zA-Z_$][a-zA-Z0-9_$]*$'
        if not re.match(js_pattern, identifier):
            return False
        
        # Check against reserved keywords
        reserved_keywords = {
            'abstract', 'arguments', 'await', 'boolean', 'break', 'byte', 'case',
            'catch', 'char', 'class', 'const', 'continue', 'debugger', 'default',
            'delete', 'do', 'double', 'else', 'enum', 'eval', 'export', 'extends',
            'false', 'final', 'finally', 'float', 'for', 'function', 'goto',
            'if', 'implements', 'import', 'in', 'instanceof', 'int', 'interface',
            'let', 'long', 'native', 'new', 'null', 'package', 'private',
            'protected', 'public', 'return', 'short', 'static', 'super', 'switch',
            'synchronized', 'this', 'throw', 'throws', 'transient', 'true', 'try',
            'typeof', 'var', 'void', 'volatile', 'while', 'with', 'yield'
        }
        
        return identifier not in reserved_keywords
    
    def _is_valid_json_pointer_or_glob(self, path: str) -> bool:
        """Check if path is valid JSON Pointer or glob pattern."""
        if not path:
            return False
        
        # Check for glob patterns first (can be anywhere in path)
        glob_chars = set('*?[]')
        has_glob = any(char in path for char in glob_chars)
        
        # If it has glob patterns, it's a glob - validate glob syntax
        if has_glob:
            # Basic glob validation - just ensure it's not obviously malformed
            return True
        
        # Otherwise, it should be a JSON Pointer (must start with /)
        if not path.startswith('/'):
            return False
        
        # Validate JSON Pointer format
        try:
            # Basic JSON Pointer validation
            parts = path[1:].split('/')
            for part in parts:
                # Check for proper escaping (~0 for ~ and ~1 for /)
                if '~' in part:
                    # Validate that ~ is properly escaped
                    if not re.search(r'~[01]', part):
                        return False
                    # Check for invalid escape sequences
                    if re.search(r'~[^01]', part):
                        return False
            return True
        except:
            return False
    
    def _add_error(self, component: str, field: str, message: str, value: Any = None) -> None:
        """Add configuration error."""
        error = ConfigValidationError(component, field, message, value)
        self.errors.append(error)
    
    def _add_warning(self, component: str, field: str, message: str, value: Any = None) -> None:
        """Add configuration warning."""
        warning = ConfigValidationWarning(component, field, message, value)
        self.warnings.append(warning)


def validate_config_from_file(config_path: str | Path, strict_mode: bool = False) -> Tuple[DataLayerConfig | None, List[str]]:
    """Load and validate configuration from YAML/JSON file.
    
    Args:
        config_path: Path to configuration file
        strict_mode: Enable strict validation
        
    Returns:
        Tuple of (config_object_or_None, error_messages)
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        return None, [f"Configuration file does not exist: {config_path}"]
    
    try:
        # Load configuration file
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.json':
                import json
                config_data = json.load(f)
            else:
                import yaml
                config_data = yaml.safe_load(f)
        
        # Create configuration object
        config = DataLayerConfig(**config_data)
        
        # Validate configuration
        validator = ConfigValidator(strict_mode=strict_mode)
        is_valid, error_messages = validator.validate_config(config)
        
        if is_valid:
            return config, []
        else:
            return config, error_messages
            
    except Exception as e:
        return None, [f"Failed to load configuration: {e}"]


def create_sample_config() -> DataLayerConfig:
    """Create a sample configuration for testing/documentation.
    
    Returns:
        Sample DataLayerConfig with recommended settings
    """
    return DataLayerConfig(
        environment="development",
        site_domain="example.com",
        global_settings={
            "enable_debug": True,
            "enable_detailed_logging": True,
            "max_processing_time_ms": 5000,
            "enable_monitoring": True,
            "fail_fast": False
        },
        capture=CaptureConfig(
            object_name="dataLayer",
            max_depth=6,
            max_entries=500,
            execution_timeout_ms=5000,
            safe_mode=True
        ),
        redaction=RedactionConfig(
            enabled=True,
            keep_audit_trail=True,
            audit_trail_retention_days=30
        ),
        validation=SchemaConfig(
            enabled=True,
            cache_schemas=True,
            strict_mode=False
        ),
        aggregation=AggregationConfig(
            enabled=True,
            max_example_values=3,
            max_example_events=3
        ),
        performance=PerformanceConfig(
            max_concurrent_pages=5,
            enable_caching=True,
            cache_ttl_seconds=300,
            max_memory_usage_mb=512
        )
    )