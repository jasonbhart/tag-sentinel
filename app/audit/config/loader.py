"""Configuration loader for crawl settings with YAML support and environment overrides.

This module provides functionality to load CrawlConfig from YAML files
with support for environment-specific overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from ..models.crawl import CrawlConfig


logger = logging.getLogger(__name__)


class ConfigLoadError(Exception):
    """Exception raised when configuration loading fails."""
    pass


def load_crawl_config(
    config_path: Optional[str] = None,
    environment: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> CrawlConfig:
    """Load CrawlConfig from YAML file with environment overrides.

    Args:
        config_path: Path to YAML config file. If None, uses default location.
        environment: Environment name for override selection. If None, uses ENV var.
        overrides: Additional configuration overrides to apply.

    Returns:
        Configured CrawlConfig instance.

    Raises:
        ConfigLoadError: If configuration loading or validation fails.

    Example:
        >>> config = load_crawl_config("config/crawl.yaml", environment="development")
        >>> config.max_pages
        500
    """
    if config_path is None:
        # Default to config/crawl.yaml in project root
        project_root = Path(__file__).parents[3]  # Go up from app/audit/config/
        config_path = project_root / "config" / "crawl.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigLoadError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"Failed to parse YAML config: {e}")
    except IOError as e:
        raise ConfigLoadError(f"Failed to read config file: {e}")

    if not isinstance(config_data, dict):
        raise ConfigLoadError("Config file must contain a YAML dictionary")

    # Determine environment from parameter or environment variable
    if environment is None:
        environment = os.getenv("TAG_SENTINEL_ENV", "production")

    # Apply environment-specific overrides
    if "environments" in config_data and environment in config_data["environments"]:
        env_overrides = config_data["environments"][environment]
        config_data = _deep_merge(config_data, env_overrides)
        logger.info(f"Applied environment overrides for: {environment}")

    # Remove the environments section as it's not part of CrawlConfig
    config_data.pop("environments", None)

    # Apply additional overrides if provided
    if overrides:
        config_data = _deep_merge(config_data, overrides)
        logger.debug("Applied additional configuration overrides")

    # Convert to CrawlConfig and validate
    try:
        return CrawlConfig(**config_data)
    except Exception as e:
        raise ConfigLoadError(f"Failed to create CrawlConfig: {e}")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base configuration dictionary.
        override: Override values to merge in.

    Returns:
        Merged configuration dictionary.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def create_default_crawl_config() -> Dict[str, Any]:
    """Create a default crawl configuration dictionary.

    Returns:
        Default configuration values suitable for YAML serialization.
    """
    return {
        "discovery_mode": "seeds",
        "seeds": [],
        "sitemap_url": None,
        "include_patterns": [],
        "exclude_patterns": [],
        "same_site_only": True,
        "max_pages": 500,
        "max_depth": None,
        "max_concurrency": 5,
        "requests_per_second": 2.0,
        "max_concurrent_per_host": 2,
        "page_timeout": 30,
        "navigation_timeout": 30,
        "max_retries": 3,
        "retry_delay": 1.0,
        "load_wait_strategy": "networkidle",
        "load_wait_timeout": 5,
        "load_wait_selector": None,
        "load_wait_js": None,
        "user_agent": None,
        "extra_headers": {},
        "environments": {
            "development": {
                "max_concurrency": 2,
                "requests_per_second": 1.0,
                "max_pages": 50,
            },
            "staging": {
                "max_pages": 1000,
                "requests_per_second": 3.0,
            },
            "test": {
                "max_concurrency": 1,
                "max_pages": 10,
                "requests_per_second": 5.0,
            }
        }
    }


def save_default_config(output_path: str) -> None:
    """Save default crawl configuration to YAML file.

    Args:
        output_path: Path where to save the configuration file.

    Raises:
        ConfigLoadError: If file writing fails.
    """
    config_data = create_default_crawl_config()

    try:
        with open(output_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved default crawl configuration to: {output_path}")
    except IOError as e:
        raise ConfigLoadError(f"Failed to save config file: {e}")