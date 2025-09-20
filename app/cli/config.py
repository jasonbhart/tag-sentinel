"""Configuration system for Tag Sentinel CLI with proper precedence handling.

This module implements the configuration system specified in EPIC 10 with
support for multiple configuration sources and proper precedence:
CLI flags > environment variables > config files > defaults
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator

from ..audit.rules import Severity


class Environment(str, Enum):
    """Supported environment types."""
    DEVELOPMENT = "development"
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"
    PROD = "prod"

    @classmethod
    def normalize(cls, env: str) -> str:
        """Normalize environment string to standard form."""
        env_map = {
            "dev": "development",
            "prod": "production",
        }
        return env_map.get(env.lower(), env.lower())


class DebugConfig(BaseModel):
    """Debug configuration options."""
    headful: bool = Field(default=False, description="Run browser with GUI")
    devtools: bool = Field(default=False, description="Open browser developer tools")
    har: bool = Field(default=False, description="Generate HAR files")
    har_omit_content: bool = Field(default=False, description="Omit content from HAR files")
    screenshots: Optional[str] = Field(default=None, description="Screenshot capture mode")
    trace: bool = Field(default=False, description="Generate trace files")

    @validator('screenshots')
    def validate_screenshots(cls, v):
        if v is not None and v not in ['on_error', 'on_all', 'on_fail']:
            raise ValueError("screenshots must be one of: on_error, on_all, on_fail")
        return v


class ExecutionConfig(BaseModel):
    """Execution configuration options."""
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0, description="Page load timeout")
    max_concurrency: int = Field(default=3, ge=1, le=20, description="Maximum concurrent instances")
    max_pages: int = Field(default=500, ge=1, description="Maximum pages to audit")
    fail_fast: bool = Field(default=False, description="Stop on first critical failure")


class RuleConfig(BaseModel):
    """Rule evaluation configuration."""
    rules_file: Optional[Path] = Field(default=None, description="Path to rules configuration")
    scenario_id: Optional[str] = Field(default=None, description="Scenario identifier")
    severity_threshold: Severity = Field(default=Severity.INFO, description="Minimum severity")
    exit_on_critical: bool = Field(default=True, description="Exit with code 2 on critical failures")
    exit_on_warnings: bool = Field(default=False, description="Exit with code 1 on warnings")


class OutputConfig(BaseModel):
    """Output configuration options."""
    format: str = Field(default="json", description="Output format")
    output_file: Optional[Path] = Field(default=None, description="Output file path")
    output_dir: Optional[Path] = Field(default=None, description="Output directory")
    verbose: bool = Field(default=False, description="Verbose output")
    quiet: bool = Field(default=False, description="Quiet mode")
    json_output: bool = Field(default=False, description="Force JSON output")

    @validator('format')
    def validate_format(cls, v):
        if v not in ['json', 'yaml', 'text']:
            raise ValueError("format must be one of: json, yaml, text")
        return v


class AlertConfig(BaseModel):
    """Alert configuration options."""
    enable_alerts: bool = Field(default=False, description="Enable alert dispatching")
    alert_config_path: Optional[Path] = Field(default=None, description="Alert configuration file")


class InputConfig(BaseModel):
    """Input configuration options."""
    urls: List[str] = Field(default_factory=list, description="URLs to audit")
    seeds_file: Optional[Path] = Field(default=None, description="Seeds file path")
    sitemap_url: Optional[str] = Field(default=None, description="Sitemap URL")
    crawl_base_url: Optional[str] = Field(default=None, description="Base URL for crawling")


class CLIConfiguration(BaseModel):
    """Complete CLI configuration with all sections."""

    # Basic settings
    environment: str = Field(default="production", description="Target environment")

    # Configuration sections
    input: InputConfig = Field(default_factory=InputConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    rules: RuleConfig = Field(default_factory=RuleConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)

    # Metadata
    config_file_path: Optional[Path] = Field(default=None, description="Source config file")
    loaded_from: List[str] = Field(default_factory=list, description="Configuration sources")

    @validator('environment')
    def normalize_environment(cls, v):
        return Environment.normalize(v)


class ConfigurationLoader:
    """Loads and merges configuration from multiple sources with proper precedence."""

    # Environment variable prefix
    ENV_PREFIX = "OPENAUDIT_"

    # Default configuration file names (searched in order)
    DEFAULT_CONFIG_FILES = [
        "openaudit.yaml",
        "openaudit.yml",
        ".openaudit.yaml",
        ".openaudit.yml",
        "openaudit.json",
        ".openaudit.json"
    ]

    def __init__(self):
        self.loaded_sources: List[str] = []

    def load_configuration(
        self,
        config_file: Optional[Path] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
        search_paths: Optional[List[Path]] = None
    ) -> CLIConfiguration:
        """Load configuration with proper precedence.

        Precedence (highest to lowest):
        1. CLI overrides (flags)
        2. Environment variables
        3. Specified config file
        4. Auto-discovered config files
        5. Defaults

        Args:
            config_file: Explicitly specified config file
            cli_overrides: CLI flag overrides
            search_paths: Paths to search for config files

        Returns:
            Merged configuration
        """
        self.loaded_sources = []

        # Start with defaults
        config_data = {}
        self.loaded_sources.append("defaults")

        # Load from auto-discovered config files
        if not config_file:
            discovered_config = self._discover_config_file(search_paths or [Path.cwd()])
            if discovered_config:
                config_data = self._merge_config(config_data, discovered_config)
                self.loaded_sources.append(f"auto-discovered: {discovered_config['_source_file']}")

        # Load from explicitly specified config file
        if config_file:
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_file}")

            file_config = self._load_config_file(config_file)
            config_data = self._merge_config(config_data, file_config)
            self.loaded_sources.append(f"config file: {config_file}")

        # Load from environment variables
        env_config = self._load_environment_variables()
        if env_config:
            config_data = self._merge_config(config_data, env_config)
            self.loaded_sources.append("environment variables")

        # Apply CLI overrides
        if cli_overrides:
            config_data = self._merge_config(config_data, cli_overrides)
            self.loaded_sources.append("CLI flags")

        # Add metadata
        config_data["loaded_from"] = self.loaded_sources
        if config_file:
            config_data["config_file_path"] = config_file

        # Create and validate final configuration
        return CLIConfiguration(**config_data)

    def _discover_config_file(self, search_paths: List[Path]) -> Optional[Dict[str, Any]]:
        """Discover configuration file in search paths."""
        for search_path in search_paths:
            for config_filename in self.DEFAULT_CONFIG_FILES:
                config_path = search_path / config_filename
                if config_path.exists() and config_path.is_file():
                    config_data = self._load_config_file(config_path)
                    config_data["_source_file"] = str(config_path)
                    return config_data
        return None

    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from a file."""
        try:
            content = config_path.read_text(encoding='utf-8')

            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(content) or {}
            elif config_path.suffix.lower() == '.json':
                return json.loads(content)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {config_path}: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")

    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}

        # Map environment variables to configuration structure
        env_mapping = {
            f"{self.ENV_PREFIX}ENVIRONMENT": "environment",
            f"{self.ENV_PREFIX}TIMEOUT": "execution.timeout_seconds",
            f"{self.ENV_PREFIX}MAX_CONCURRENCY": "execution.max_concurrency",
            f"{self.ENV_PREFIX}MAX_PAGES": "execution.max_pages",
            f"{self.ENV_PREFIX}FAIL_FAST": "execution.fail_fast",
            f"{self.ENV_PREFIX}RULES_FILE": "rules.rules_file",
            f"{self.ENV_PREFIX}SCENARIO": "rules.scenario_id",
            f"{self.ENV_PREFIX}SEVERITY_THRESHOLD": "rules.severity_threshold",
            f"{self.ENV_PREFIX}EXIT_ON_CRITICAL": "rules.exit_on_critical",
            f"{self.ENV_PREFIX}EXIT_ON_WARNINGS": "rules.exit_on_warnings",
            f"{self.ENV_PREFIX}OUTPUT_FORMAT": "output.format",
            f"{self.ENV_PREFIX}OUTPUT_FILE": "output.output_file",
            f"{self.ENV_PREFIX}OUTPUT_DIR": "output.output_dir",
            f"{self.ENV_PREFIX}VERBOSE": "output.verbose",
            f"{self.ENV_PREFIX}QUIET": "output.quiet",
            f"{self.ENV_PREFIX}HEADFUL": "debug.headful",
            f"{self.ENV_PREFIX}DEVTOOLS": "debug.devtools",
            f"{self.ENV_PREFIX}HAR": "debug.har",
            f"{self.ENV_PREFIX}SCREENSHOTS": "debug.screenshots",
            f"{self.ENV_PREFIX}TRACE": "debug.trace",
            f"{self.ENV_PREFIX}ENABLE_ALERTS": "alerts.enable_alerts",
            f"{self.ENV_PREFIX}ALERT_CONFIG": "alerts.alert_config_path",
        }

        for env_var, config_path in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(env_value, config_path)
                self._set_nested_value(config, config_path, converted_value)

        return config

    def _convert_env_value(self, value: str, config_path: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean values
        if config_path.endswith(('.fail_fast', '.exit_on_critical', '.exit_on_warnings',
                                '.verbose', '.quiet', '.headful', '.devtools', '.har',
                                '.trace', '.enable_alerts')):
            return value.lower() in ('true', '1', 'yes', 'on')

        # Numeric values
        if config_path.endswith(('.timeout_seconds',)):
            return float(value)
        elif config_path.endswith(('.max_concurrency', '.max_pages')):
            return int(value)

        # Path values
        if config_path.endswith(('.rules_file', '.output_file', '.output_dir', '.alert_config_path')):
            return Path(value) if value else None

        # Severity enum
        if config_path.endswith('.severity_threshold'):
            severity_map = {
                'info': Severity.INFO,
                'warning': Severity.WARNING,
                'critical': Severity.CRITICAL
            }
            return severity_map.get(value.lower(), Severity.INFO)

        return value

    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set a nested configuration value using dot notation."""
        keys = path.split('.')
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries, with override taking precedence."""
        if not override:
            return base

        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value

        return result


def load_configuration(
    config_file: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    search_paths: Optional[List[Path]] = None
) -> CLIConfiguration:
    """Convenience function to load configuration.

    Args:
        config_file: Path to configuration file
        cli_overrides: CLI flag overrides
        search_paths: Paths to search for config files

    Returns:
        Loaded and merged configuration
    """
    loader = ConfigurationLoader()
    return loader.load_configuration(config_file, cli_overrides, search_paths)


def print_configuration(config: CLIConfiguration, format: str = "yaml") -> str:
    """Print configuration in specified format for debugging.

    Args:
        config: Configuration to print
        format: Output format (yaml, json)

    Returns:
        Formatted configuration string
    """
    # Use model_dump with mode="json" to get clean serialization without Python object tags
    config_dict = config.model_dump(
        mode="json",
        exclude={'loaded_from', 'config_file_path'},
        exclude_none=False
    )

    if format.lower() == "json":
        return json.dumps(config_dict, indent=2, default=str)
    else:
        # Use yaml.safe_dump to ensure clean YAML output
        return yaml.safe_dump(config_dict, default_flow_style=False, sort_keys=True)


def validate_configuration(config: CLIConfiguration) -> List[str]:
    """Validate configuration and return list of validation errors.

    Args:
        config: Configuration to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check input configuration
    input_count = sum([
        bool(config.input.urls),
        bool(config.input.seeds_file),
        bool(config.input.sitemap_url),
        bool(config.input.crawl_base_url)
    ])

    if input_count == 0:
        # Allow empty input when just validating configuration
        pass  # This will be caught by CLI validation if needed
    elif input_count > 1:
        errors.append("Multiple input sources specified - use only one")

    # Check file paths exist
    if config.input.seeds_file and not config.input.seeds_file.exists():
        errors.append(f"Seeds file not found: {config.input.seeds_file}")

    if config.rules.rules_file and not config.rules.rules_file.exists():
        errors.append(f"Rules file not found: {config.rules.rules_file}")

    if config.alerts.alert_config_path and not config.alerts.alert_config_path.exists():
        errors.append(f"Alert config file not found: {config.alerts.alert_config_path}")

    # Check output directory is writable
    if config.output.output_dir:
        try:
            config.output.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory {config.output.output_dir}: {e}")

    return errors