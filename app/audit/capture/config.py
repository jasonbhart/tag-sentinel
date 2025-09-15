"""Configuration system for browser capture.

This module provides configuration management for capture engine settings,
including YAML loading, validation, and environment-specific overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, field_validator

from .browser_factory import BrowserConfig, BrowserEngineType
from .engine import CaptureEngineConfig
from .page_session import PageSessionConfig, WaitStrategy


class CaptureConfig(BaseModel):
    """Root configuration for capture system."""

    environment: str = Field(default="production", description="Environment name")
    browser: Dict[str, Any] = Field(default_factory=dict, description="Browser configuration")
    engine: Dict[str, Any] = Field(default_factory=dict, description="Engine configuration")
    environments: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Environment-specific overrides"
    )

    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        valid_environments = {'production', 'staging', 'development', 'test'}
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v

    def get_browser_config(self) -> BrowserConfig:
        """Get browser configuration with environment overrides applied."""
        config = self.browser.copy()

        # Apply environment-specific overrides
        if self.environment in self.environments:
            env_config = self.environments[self.environment]
            if 'browser' in env_config:
                config.update(env_config['browser'])

        browser_config = BrowserConfig(
            engine=getattr(BrowserEngineType, config.get('engine', 'chromium').upper()),
            headless=config.get('headless', True),
            viewport={'width': config.get('window_width', 1366), 'height': config.get('window_height', 768)},
            user_agent=config.get('user_agent'),
            timezone=config.get('timezone'),
            locale=config.get('locale', 'en-US'),
            ignore_https_errors=config.get('ignore_https_errors', False),
            java_script_enabled=config.get('java_script_enabled', True),
        )

        return browser_config

    def get_engine_config(self) -> CaptureEngineConfig:
        """Get engine configuration with environment overrides applied."""
        config = self.engine.copy()

        # Apply environment-specific overrides
        if self.environment in self.environments:
            env_config = self.environments[self.environment]
            if 'engine' in env_config:
                config.update(env_config['engine'])

        # Convert artifacts_dir to Path if provided
        artifacts_dir = config.get('artifacts_dir')
        if artifacts_dir:
            artifacts_dir = Path(artifacts_dir)
        elif config.get('artifacts_enabled', False):
            # Use temp directory if artifacts enabled but no dir specified
            import tempfile
            artifacts_dir = Path(tempfile.gettempdir()) / "tag-sentinel-artifacts"

        # Get browser config with environment overrides applied
        browser_config = self.browser.copy()
        if self.environment in self.environments:
            env_config = self.environments[self.environment]
            if 'browser' in env_config:
                browser_config.update(env_config['browser'])

        return CaptureEngineConfig(
            browser_config=self.get_browser_config(),
            max_concurrent_pages=config.get('max_concurrent_pages', 5),
            page_timeout_ms=config.get('page_timeout_ms', 30000),
            default_wait_strategy=config.get('default_wait_strategy', WaitStrategy.NETWORKIDLE),
            default_wait_timeout_ms=config.get('default_wait_timeout_ms', 30000),
            enable_network_capture=config.get('enable_network_capture', True),
            enable_console_capture=config.get('enable_console_capture', True),
            enable_cookie_capture=config.get('enable_cookie_capture', True),
            redact_cookie_values=config.get('redact_cookie_values', True),
            filter_console_noise=config.get('filter_console_noise', True),
            artifacts_enabled=config.get('artifacts_enabled', False),
            artifacts_dir=artifacts_dir,
            take_screenshots=config.get('take_screenshots', False),
            screenshot_on_error=config.get('screenshot_on_error', True),
            enable_har=config.get('enable_har', False),
            enable_trace=config.get('enable_trace', False),
            retry_attempts=config.get('retry_attempts', 3),
            retry_delay_ms=config.get('retry_delay_ms', 1000),
            continue_on_error=config.get('continue_on_error', True),
            memory_limit_mb=config.get('memory_limit_mb'),
            cleanup_interval_pages=config.get('cleanup_interval_pages', 50),
            enable_batch_processing=config.get('enable_batch_processing', True),
            max_batch_size=config.get('max_batch_size', 10),
            enable_context_reuse=browser_config.get('enable_context_reuse', False),
        )


class CaptureConfigManager:
    """Manager for capture configuration loading and caching."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to capture config YAML file. Defaults to config/capture.yaml
        """
        if config_path is None:
            # Default to config/capture.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "capture.yaml"

        self.config_path = Path(config_path)
        self._config: Optional[CaptureConfig] = None
        self._loaded_env = None

    def load_config(self, force_reload: bool = False) -> CaptureConfig:
        """Load configuration from YAML file.

        Args:
            force_reload: Force reload even if already cached

        Returns:
            Loaded and validated configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
            ValueError: If configuration validation fails
        """
        current_env = os.environ.get('TAG_SENTINEL_ENV', 'production')

        if self._config is not None and not force_reload and current_env == self._loaded_env:
            return self._config

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # Override environment from env var if set
            if current_env != 'production':
                config_data['environment'] = current_env

            self._config = CaptureConfig(**config_data)
            self._loaded_env = current_env

            return self._config

        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {self.config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")

    @property
    def config(self) -> CaptureConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config

    @property
    def environment(self) -> str:
        """Get current environment."""
        return self.config.environment

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == 'production'


# Global config manager instance
_config_manager: Optional[CaptureConfigManager] = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> CaptureConfigManager:
    """Get global capture configuration manager.

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        Global CaptureConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = CaptureConfigManager(config_path)
    return _config_manager


def create_engine_from_config(config_path: Optional[Union[str, Path]] = None) -> CaptureEngineConfig:
    """Create engine configuration from YAML file.

    Args:
        config_path: Path to capture config YAML file

    Returns:
        Configured CaptureEngineConfig instance
    """
    config_manager = get_config(config_path)
    return config_manager.config.get_engine_config()