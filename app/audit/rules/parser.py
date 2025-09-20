"""YAML rule parsing and loading with validation and normalization.

This module provides robust YAML rule parsing with validation, environment variable
interpolation, regex compilation, and comprehensive error handling.
"""

import os
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from urllib.parse import urlparse
from copy import deepcopy

from .models import Rule, CheckConfig, AppliesTo, Severity, CheckType, RuleScope
from .schema import validate_rules_config, validate_rules_file, ValidationResult


class ParseError(Exception):
    """Exception raised during rule parsing."""
    
    def __init__(self, message: str, path: Optional[str] = None, line: Optional[int] = None):
        self.message = message
        self.path = path
        self.line = line
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format error message with path and line information."""
        parts = []
        if self.path:
            parts.append(f"in {self.path}")
        if self.line:
            parts.append(f"at line {self.line}")
        
        if parts:
            return f"{self.message} ({', '.join(parts)})"
        return self.message


class EnvironmentInterpolator:
    """Handles secure environment variable interpolation in YAML configurations."""
    
    # Regex pattern for environment variable substitution
    ENV_PATTERN = re.compile(r'\$\{env\.([A-Za-z_][A-Za-z0-9_]*)\}')
    
    @classmethod
    def interpolate(cls, value: Any, allow_missing: bool = False) -> Any:
        """Interpolate environment variables in configuration values.
        
        Args:
            value: Configuration value to interpolate
            allow_missing: If True, missing env vars are left as-is; if False, raises error
            
        Returns:
            Value with environment variables substituted
            
        Raises:
            ParseError: If required environment variable is missing and allow_missing=False
        """
        if isinstance(value, str):
            return cls._interpolate_string(value, allow_missing)
        elif isinstance(value, dict):
            return {k: cls.interpolate(v, allow_missing) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls.interpolate(item, allow_missing) for item in value]
        else:
            return value
    
    @classmethod
    def _interpolate_string(cls, text: str, allow_missing: bool) -> str:
        """Interpolate environment variables in a string value."""
        def replace_env_var(match):
            env_var = match.group(1)
            env_value = os.environ.get(env_var)
            
            if env_value is None:
                if allow_missing:
                    return match.group(0)  # Return original ${env.VAR} if missing
                else:
                    raise ParseError(f"Required environment variable '{env_var}' is not set")
            
            return env_value
        
        return cls.ENV_PATTERN.sub(replace_env_var, text)
    
    @classmethod
    def extract_env_vars(cls, config: Dict[str, Any]) -> List[str]:
        """Extract all environment variable references from configuration.
        
        Returns:
            List of environment variable names referenced in the config
        """
        env_vars = set()
        
        def _extract_from_value(value: Any):
            if isinstance(value, str):
                matches = cls.ENV_PATTERN.findall(value)
                env_vars.update(matches)
            elif isinstance(value, dict):
                for v in value.values():
                    _extract_from_value(v)
            elif isinstance(value, list):
                for item in value:
                    _extract_from_value(item)
        
        _extract_from_value(config)
        return sorted(list(env_vars))


class RegexCompiler:
    """Handles regex pattern compilation and caching."""
    
    def __init__(self):
        self._compiled_patterns: Dict[str, re.Pattern] = {}
    
    def compile_pattern(self, pattern: str, flags: int = 0) -> re.Pattern:
        """Compile and cache a regex pattern.
        
        Args:
            pattern: Regular expression pattern
            flags: Regex flags (e.g., re.IGNORECASE)
            
        Returns:
            Compiled regex pattern
            
        Raises:
            ParseError: If pattern is invalid
        """
        cache_key = f"{pattern}:{flags}"
        
        if cache_key in self._compiled_patterns:
            return self._compiled_patterns[cache_key]
        
        try:
            compiled_pattern = re.compile(pattern, flags)
            self._compiled_patterns[cache_key] = compiled_pattern
            return compiled_pattern
        except re.error as e:
            raise ParseError(f"Invalid regex pattern '{pattern}': {e}")
    
    def validate_pattern(self, pattern: str) -> bool:
        """Validate a regex pattern without compiling.
        
        Args:
            pattern: Regular expression pattern to validate
            
        Returns:
            True if pattern is valid, False otherwise
        """
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False
    
    def clear_cache(self) -> None:
        """Clear the compiled pattern cache."""
        self._compiled_patterns.clear()


class RuleParser:
    """Main class for parsing and loading YAML rule configurations."""
    
    def __init__(self, allow_missing_env_vars: bool = False):
        """Initialize the rule parser.
        
        Args:
            allow_missing_env_vars: If True, missing environment variables are left as-is
        """
        self.allow_missing_env_vars = allow_missing_env_vars
        self.regex_compiler = RegexCompiler()
        self._parsed_config: Optional[Dict[str, Any]] = None
        self._source_file: Optional[Path] = None
    
    def parse_file(self, file_path: Union[str, Path]) -> List[Rule]:
        """Parse rules from a YAML file.
        
        Args:
            file_path: Path to the YAML rules file
            
        Returns:
            List of parsed Rule objects
            
        Raises:
            ParseError: If parsing fails
        """
        file_path = Path(file_path)
        self._source_file = file_path
        
        # Validate file exists and is readable
        if not file_path.exists():
            raise ParseError(f"Rules file not found: {file_path}")
        
        if not file_path.is_file():
            raise ParseError(f"Path is not a file: {file_path}")
        
        try:
            content = file_path.read_text(encoding='utf-8')
            return self.parse_yaml(content)
        except Exception as e:
            if isinstance(e, ParseError):
                raise
            raise ParseError(f"Error reading file {file_path}: {e}")
    
    def parse_yaml(self, yaml_content: str) -> List[Rule]:
        """Parse rules from YAML content string.
        
        Args:
            yaml_content: YAML content as string
            
        Returns:
            List of parsed Rule objects
            
        Raises:
            ParseError: If parsing fails
        """
        # Parse YAML
        try:
            config = yaml.safe_load(yaml_content)
            if config is None:
                raise ParseError("YAML content is empty")
        except yaml.YAMLError as e:
            raise ParseError(f"YAML parsing error: {e}")
        
        return self.parse_config(config)
    
    def parse_config(self, config: Dict[str, Any]) -> List[Rule]:
        """Parse rules from a configuration dictionary.
        
        Args:
            config: Configuration dictionary (parsed YAML)
            
        Returns:
            List of parsed Rule objects
            
        Raises:
            ParseError: If parsing fails
        """
        self._parsed_config = config.copy()
        
        # Validate against schema first
        validation_result = validate_rules_config(config)
        if not validation_result.valid:
            error_messages = validation_result.get_detailed_errors()
            raise ParseError(f"Schema validation failed:\n" + "\n".join(f"  - {msg}" for msg in error_messages))
        
        # Interpolate environment variables
        try:
            config = EnvironmentInterpolator.interpolate(config, self.allow_missing_env_vars)
        except ParseError:
            raise  # Re-raise ParseError as-is
        except Exception as e:
            raise ParseError(f"Environment variable interpolation failed: {e}")
        
        # Extract defaults
        defaults = config.get('defaults', {})
        
        # Parse individual rules
        rules = []
        rule_configs = config.get('rules', [])
        
        for i, rule_config in enumerate(rule_configs):
            try:
                rule = self._parse_single_rule(rule_config, defaults)
                rules.append(rule)
            except ParseError as e:
                # Add context about which rule failed
                raise ParseError(f"Error in rule {i + 1} (id: {rule_config.get('id', 'unknown')}): {e.message}")
            except Exception as e:
                raise ParseError(f"Unexpected error parsing rule {i + 1}: {e}")
        
        return rules
    
    def parse_yaml_file(self, file_path: Path) -> List[Rule]:
        """Parse rules from a YAML file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            List of parsed Rule objects
            
        Raises:
            ParseError: If parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
            return self.parse_yaml(yaml_content)
        except FileNotFoundError:
            raise ParseError(f"Rule file not found: {file_path}")
        except PermissionError:
            raise ParseError(f"Permission denied reading file: {file_path}")
        except Exception as e:
            raise ParseError(f"Error reading file {file_path}: {e}")
    
    def _parse_single_rule(self, rule_config: Dict[str, Any], defaults: Dict[str, Any]) -> Rule:
        """Parse a single rule configuration.
        
        Args:
            rule_config: Individual rule configuration
            defaults: Default values to apply
            
        Returns:
            Parsed Rule object
        """
        # Apply defaults
        merged_config = {}
        merged_config.update(defaults)
        merged_config.update(rule_config)
        
        # Parse basic rule fields
        rule_id = merged_config['id']
        name = merged_config['name']
        description = merged_config.get('description')
        
        # Parse severity with validation
        severity_str = merged_config.get('severity', 'warning')
        try:
            severity = Severity(severity_str)
        except ValueError:
            raise ParseError(f"Invalid severity '{severity_str}' for rule '{rule_id}'")
        
        enabled = merged_config.get('enabled', True)
        tags = merged_config.get('tags', [])
        
        # Parse applies_to configuration
        applies_to_config = merged_config.get('applies_to', {})
        applies_to = self._parse_applies_to(applies_to_config, rule_id)
        
        # Parse check configuration
        check_config = merged_config['check']
        check = self._parse_check_config(check_config, rule_id)
        
        # Create timestamps
        created_at = datetime.utcnow()
        updated_at = None
        
        return Rule(
            id=rule_id,
            name=name,
            description=description,
            severity=severity,
            enabled=enabled,
            applies_to=applies_to,
            check=check,
            created_at=created_at,
            updated_at=updated_at,
            tags=tags
        )
    
    def _parse_applies_to(self, config: Dict[str, Any], rule_id: str) -> AppliesTo:
        """Parse rule scoping configuration.
        
        Args:
            config: applies_to configuration
            rule_id: Rule ID for error context
            
        Returns:
            AppliesTo object
        """
        # Validate URL patterns if provided
        url_include = config.get('url_include', [])
        url_exclude = config.get('url_exclude', [])
        
        for pattern in url_include + url_exclude:
            if not self.regex_compiler.validate_pattern(pattern):
                raise ParseError(f"Invalid URL regex pattern '{pattern}' in rule '{rule_id}'")
        
        return AppliesTo(
            scope=config.get('scope', RuleScope.PAGE),
            environments=config.get('environments', []),
            scenario_ids=config.get('scenario_ids', []),
            url_include=url_include,
            url_exclude=url_exclude,
            urls=config.get('urls', []),
            vendors=config.get('vendors', [])
        )
    
    def _parse_check_config(self, config: Dict[str, Any], rule_id: str) -> CheckConfig:
        """Parse check configuration.
        
        Args:
            config: Check configuration
            rule_id: Rule ID for error context
            
        Returns:
            CheckConfig object
        """
        # Normalize parameters -> config mapping for consistency
        normalized_config = config.copy()
        if 'parameters' in config:
            # Merge parameters into main config, with parameters taking precedence
            normalized_config.update(config['parameters'])
        
        # Parse and validate check type
        check_type_str = normalized_config['type']
        try:
            check_type = CheckType(check_type_str)
        except ValueError:
            raise ParseError(f"Invalid check type '{check_type_str}' for rule '{rule_id}'")
        
        # Validate URL pattern if provided
        url_pattern = normalized_config.get('url_pattern')
        if url_pattern and not self.regex_compiler.validate_pattern(url_pattern):
            raise ParseError(f"Invalid URL pattern '{url_pattern}' in check for rule '{rule_id}'")
        
        # Compile regex patterns for runtime use
        if url_pattern:
            self.regex_compiler.compile_pattern(url_pattern)
        
        # Preserve all check-specific configuration fields
        # Remove CheckConfig built-in fields to avoid duplication
        config_fields = {k: v for k, v in normalized_config.items()
                        if k not in {'type', 'vendor', 'url_pattern', 'min_count', 'max_count',
                                   'time_window_ms', 'expression', 'timeout_seconds', 'retry_count', 'enabled'}}

        # Build CheckConfig arguments, only including non-None values to preserve defaults
        check_config_args = {
            'type': check_type,
            'parameters': config.get('parameters', {}),  # Preserve original parameters
            'config': config_fields  # Include all other check-specific fields
        }

        # Only add optional fields if they have values
        if normalized_config.get('vendor') is not None:
            check_config_args['vendor'] = normalized_config['vendor']
        if url_pattern is not None:
            check_config_args['url_pattern'] = url_pattern
        if normalized_config.get('min_count') is not None:
            check_config_args['min_count'] = normalized_config['min_count']
        if normalized_config.get('max_count') is not None:
            check_config_args['max_count'] = normalized_config['max_count']
        if normalized_config.get('time_window_ms') is not None:
            check_config_args['time_window_ms'] = normalized_config['time_window_ms']
        if normalized_config.get('timeout_seconds') is not None:
            check_config_args['timeout_seconds'] = normalized_config['timeout_seconds']
        if normalized_config.get('retry_count') is not None:
            check_config_args['retry_count'] = normalized_config['retry_count']
        if normalized_config.get('enabled') is not None:
            check_config_args['enabled'] = normalized_config['enabled']
        if normalized_config.get('expression') is not None:
            check_config_args['expression'] = normalized_config['expression']

        return CheckConfig(**check_config_args)
    
    def get_parsed_config(self) -> Optional[Dict[str, Any]]:
        """Get the last parsed configuration dictionary.
        
        Returns:
            Parsed configuration dictionary or None if nothing has been parsed
        """
        return self._parsed_config
    
    def get_source_file(self) -> Optional[Path]:
        """Get the source file path for the last parsed configuration.
        
        Returns:
            Path to source file or None if parsed from string
        """
        return self._source_file
    
    def extract_environment_variables(self) -> List[str]:
        """Extract environment variables referenced in the last parsed config.
        
        Returns:
            List of environment variable names
        """
        if self._parsed_config is None:
            return []
        
        return EnvironmentInterpolator.extract_env_vars(self._parsed_config)
    
    def clear_cache(self) -> None:
        """Clear internal caches (compiled regexes)."""
        self.regex_compiler.clear_cache()


class ConfigurationManager:
    """Manages configuration inheritance, overrides, and environment-specific settings."""
    
    def __init__(self, base_path: Optional[Path] = None):
        self._base_config: Optional[Dict[str, Any]] = None
        self._environment_configs: Dict[str, Dict[str, Any]] = {}
        self._external_configs: List[Dict[str, Any]] = []
        self._base_path = Path(base_path) if base_path else None
        
        # Auto-load configuration if path is provided
        if self._base_path and self._base_path.exists():
            self._auto_load_configurations()
    
    def set_base_config(self, config: Dict[str, Any]) -> None:
        """Set the base configuration that others inherit from."""
        self._base_config = deepcopy(config)
    
    def add_environment_config(self, environment: str, config: Dict[str, Any]) -> None:
        """Add environment-specific configuration."""
        self._environment_configs[environment] = deepcopy(config)
    
    def add_external_config(self, config: Dict[str, Any]) -> None:
        """Add external configuration (from URLs, other files, etc.)."""
        self._external_configs.append(deepcopy(config))
    
    def get_merged_config(self, target_environment: Optional[str] = None) -> Dict[str, Any]:
        """Get fully merged configuration for a target environment.
        
        Merge order (later overrides earlier):
        1. Base config
        2. External configs (in order added)
        3. Environment-specific config
        
        Args:
            target_environment: Environment to build config for
            
        Returns:
            Merged configuration dictionary
        """
        if self._base_config is None:
            raise ValueError("Base configuration must be set before merging")
        
        # Start with base config
        merged = deepcopy(self._base_config)
        
        # Apply external configs in order
        for external_config in self._external_configs:
            merged = self._deep_merge_configs(merged, external_config)
        
        # Apply environment-specific config if specified
        if target_environment and target_environment in self._environment_configs:
            env_config = self._environment_configs[target_environment]
            merged = self._deep_merge_configs(merged, env_config)
        
        return merged
    
    def get_environment_rules(self, base_rules: List[Dict[str, Any]], 
                            target_environment: str) -> List[Dict[str, Any]]:
        """Get rules filtered and modified for a specific environment.
        
        Args:
            base_rules: Base rule configurations
            target_environment: Target environment name
            
        Returns:
            Rules filtered and modified for the environment
        """
        filtered_rules = []
        
        for rule_config in base_rules:
            # Check if rule applies to this environment
            applies_to = rule_config.get('applies_to', {})
            rule_environments = applies_to.get('environments', [])
            
            # If environments list is empty, rule applies to all environments
            # If environments list exists, rule must be included
            if not rule_environments or target_environment in rule_environments:
                # Create environment-specific version of the rule
                env_rule = self._adapt_rule_for_environment(rule_config, target_environment)
                filtered_rules.append(env_rule)
        
        return filtered_rules
    
    def _deep_merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    # Recursively merge nested dictionaries
                    result[key] = self._deep_merge_configs(result[key], value)
                elif isinstance(result[key], list) and isinstance(value, list):
                    # For lists, we have different strategies
                    if key == 'rules':
                        # For rules, merge by rule ID
                        result[key] = self._merge_rule_lists(result[key], value)
                    else:
                        # For other lists, override completely
                        result[key] = deepcopy(value)
                else:
                    # Override scalar values
                    result[key] = deepcopy(value)
            else:
                # Add new keys
                result[key] = deepcopy(value)
        
        return result
    
    def _merge_rule_lists(self, base_rules: List[Dict[str, Any]], 
                         override_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge two lists of rules, matching by rule ID.
        
        Args:
            base_rules: Base rules list
            override_rules: Override rules list
            
        Returns:
            Merged rules list
        """
        # Create lookup for base rules by ID
        base_by_id = {rule.get('id'): rule for rule in base_rules if rule.get('id')}
        result_rules = []
        
        # Process override rules
        override_ids = set()
        for override_rule in override_rules:
            rule_id = override_rule.get('id')
            if not rule_id:
                continue
                
            override_ids.add(rule_id)
            
            if rule_id in base_by_id:
                # Merge with existing rule
                merged_rule = self._deep_merge_configs(base_by_id[rule_id], override_rule)
                result_rules.append(merged_rule)
            else:
                # Add new rule
                result_rules.append(deepcopy(override_rule))
        
        # Add base rules that weren't overridden
        for base_rule in base_rules:
            rule_id = base_rule.get('id')
            if rule_id and rule_id not in override_ids:
                result_rules.append(deepcopy(base_rule))
        
        return result_rules
    
    def _adapt_rule_for_environment(self, rule_config: Dict[str, Any], 
                                  environment: str) -> Dict[str, Any]:
        """Adapt a rule configuration for a specific environment.
        
        Args:
            rule_config: Base rule configuration
            environment: Target environment
            
        Returns:
            Environment-adapted rule configuration
        """
        adapted = deepcopy(rule_config)
        
        # Look for environment-specific overrides within the rule
        env_overrides = adapted.get('environment_overrides', {})
        if environment in env_overrides:
            override_config = env_overrides[environment]
            adapted = self._deep_merge_configs(adapted, override_config)
            # Remove the environment_overrides section as it's no longer needed
            adapted.pop('environment_overrides', None)
        
        return adapted
    
    def validate_inheritance_chain(self) -> List[str]:
        """Validate the configuration inheritance chain for common issues.
        
        Returns:
            List of validation warnings
        """
        warnings = []
        
        if self._base_config is None:
            warnings.append("No base configuration set")
            return warnings
        
        # Check for conflicting rule IDs across environments
        all_rule_ids = set()
        duplicate_ids = set()
        
        for env_name, env_config in self._environment_configs.items():
            env_rules = env_config.get('rules', [])
            for rule in env_rules:
                rule_id = rule.get('id')
                if rule_id:
                    if rule_id in all_rule_ids:
                        duplicate_ids.add(rule_id)
                    all_rule_ids.add(rule_id)
        
        if duplicate_ids:
            warnings.append(f"Duplicate rule IDs across environments: {sorted(duplicate_ids)}")
        
        # Check for unused environment variables
        base_env_vars = EnvironmentInterpolator.extract_env_vars(self._base_config)
        for env_name, env_config in self._environment_configs.items():
            env_env_vars = EnvironmentInterpolator.extract_env_vars(env_config)
            unused_vars = set(env_env_vars) - set(base_env_vars)
            if unused_vars:
                warnings.append(f"Environment '{env_name}' defines unused variables: {sorted(unused_vars)}")
        
        return warnings
    
    def _auto_load_configurations(self) -> None:
        """Auto-load base and environment configurations from the base path."""
        if not self._base_path:
            return
            
        # Load base configuration if exists
        base_yaml = self._base_path / "base.yaml"
        if base_yaml.exists():
            try:
                with open(base_yaml, 'r', encoding='utf-8') as f:
                    base_config = yaml.safe_load(f.read())
                if base_config:
                    self.set_base_config(base_config)
            except Exception as e:
                # Silently ignore loading errors for auto-loading
                pass
        
        # Load environment-specific configurations
        for env_file in self._base_path.glob("*.yaml"):
            if env_file.name == "base.yaml":
                continue
            
            env_name = env_file.stem  # filename without extension
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    env_config = yaml.safe_load(f.read())
                if env_config:
                    self.add_environment_config(env_name, env_config)
            except Exception as e:
                # Silently ignore loading errors for auto-loading
                pass
    
    def load_environment_rules(self, environment: str) -> List[Rule]:
        """Load and parse rules for a specific environment.
        
        Args:
            environment: Environment name to load rules for
            
        Returns:
            List of parsed Rule objects for the environment
            
        Raises:
            ValueError: If no configuration is available
            ParseError: If parsing fails
        """
        if self._base_config is None:
            raise ValueError("No base configuration available. Load configuration first.")
        
        # Get merged config for the environment
        merged_config = self.get_merged_config(environment)
        
        # Parse rules from the merged configuration
        parser = RuleParser()
        rules = parser.parse_yaml(yaml.dump(merged_config))
        
        return rules


class AdvancedRuleParser(RuleParser):
    """Extended rule parser with advanced configuration management capabilities."""
    
    def __init__(self, allow_missing_env_vars: bool = False):
        super().__init__(allow_missing_env_vars)
        self.config_manager = ConfigurationManager()
        self._loaded_configs: List[str] = []  # Track loaded config sources
    
    def load_configuration_chain(self, configs: List[Union[str, Path, Dict[str, Any]]]) -> None:
        """Load a chain of configurations with inheritance.
        
        Args:
            configs: List of configuration sources (file paths, URLs, or config dicts)
                    First config becomes base, others are overlayed in order
        """
        if not configs:
            raise ParseError("At least one configuration must be provided")
        
        # Load base configuration
        base_config = self._load_single_config(configs[0])
        self.config_manager.set_base_config(base_config)
        
        # Load additional configurations as external configs
        for config_source in configs[1:]:
            config = self._load_single_config(config_source)
            self.config_manager.add_external_config(config)
    
    def parse_for_environment(self, target_environment: str) -> List[Rule]:
        """Parse rules for a specific environment with full inheritance.
        
        Args:
            target_environment: Environment to parse rules for
            
        Returns:
            List of parsed Rule objects for the environment
        """
        # Get merged configuration for the environment
        merged_config = self.config_manager.get_merged_config(target_environment)
        
        # Validate the merged configuration
        validation_result = validate_rules_config(merged_config)
        if not validation_result.valid:
            error_messages = validation_result.get_detailed_errors()
            raise ParseError(f"Merged configuration validation failed for environment '{target_environment}':\n" + 
                           "\n".join(f"  - {msg}" for msg in error_messages))
        
        # Apply environment variable interpolation
        try:
            merged_config = EnvironmentInterpolator.interpolate(merged_config, self.allow_missing_env_vars)
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"Environment variable interpolation failed for environment '{target_environment}': {e}")
        
        # Get environment-specific rules
        base_rules = merged_config.get('rules', [])
        env_rules = self.config_manager.get_environment_rules(base_rules, target_environment)
        
        # Parse rules
        defaults = merged_config.get('defaults', {})
        parsed_rules = []
        
        for i, rule_config in enumerate(env_rules):
            try:
                rule = self._parse_single_rule(rule_config, defaults)
                parsed_rules.append(rule)
            except ParseError as e:
                raise ParseError(f"Error in rule {i + 1} (id: {rule_config.get('id', 'unknown')}) "
                               f"for environment '{target_environment}': {e.message}")
        
        return parsed_rules
    
    def get_available_environments(self) -> List[str]:
        """Get list of available environments from loaded configurations.
        
        Returns:
            List of environment names
        """
        environments = set()
        
        # Check base config for environments section
        base_config = self.config_manager._base_config
        if base_config:
            environments.update(base_config.get('environments', {}).keys())
            
            # Also check rules for environment references
            for rule in base_config.get('rules', []):
                applies_to = rule.get('applies_to', {})
                rule_envs = applies_to.get('environments', [])
                environments.update(rule_envs)
        
        # Check environment-specific configs
        environments.update(self.config_manager._environment_configs.keys())
        
        return sorted(list(environments))
    
    def validate_configuration_chain(self) -> List[str]:
        """Validate the entire configuration chain for issues.
        
        Returns:
            List of validation warnings and errors
        """
        warnings = []
        
        # Validate inheritance chain
        inheritance_warnings = self.config_manager.validate_inheritance_chain()
        warnings.extend(inheritance_warnings)
        
        # Test parsing for all available environments
        available_envs = self.get_available_environments()
        
        for env in available_envs:
            try:
                rules = self.parse_for_environment(env)
                if not rules:
                    warnings.append(f"Environment '{env}' has no applicable rules")
            except ParseError as e:
                warnings.append(f"Environment '{env}' parsing failed: {e}")
        
        return warnings
    
    def _load_single_config(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load a single configuration from various sources.
        
        Args:
            source: Configuration source (file path, URL, or dict)
            
        Returns:
            Loaded configuration dictionary
        """
        if isinstance(source, dict):
            # Already a configuration dictionary
            self._loaded_configs.append("inline-config")
            return source
        
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            
            if source_path.exists():
                # Local file
                try:
                    content = source_path.read_text(encoding='utf-8')
                    config = yaml.safe_load(content)
                    self._loaded_configs.append(str(source_path))
                    return config
                except Exception as e:
                    raise ParseError(f"Error loading configuration from {source_path}: {e}")
            else:
                # Could be a URL - for now, treat as error
                raise ParseError(f"Configuration source not found: {source}")
        
        raise ParseError(f"Unsupported configuration source type: {type(source)}")
    
    def get_loaded_config_sources(self) -> List[str]:
        """Get list of configuration sources that were loaded.
        
        Returns:
            List of source identifiers
        """
        return self._loaded_configs.copy()


def load_rules_from_file(file_path: Union[str, Path], 
                        allow_missing_env_vars: bool = False) -> List[Rule]:
    """Convenience function to load rules from a file.
    
    Args:
        file_path: Path to YAML rules file
        allow_missing_env_vars: If True, missing environment variables are left as-is
        
    Returns:
        List of parsed Rule objects
        
    Raises:
        ParseError: If parsing fails
    """
    parser = RuleParser(allow_missing_env_vars=allow_missing_env_vars)
    return parser.parse_file(file_path)


def load_rules_from_yaml(yaml_content: str, 
                        allow_missing_env_vars: bool = False) -> List[Rule]:
    """Convenience function to load rules from YAML string.
    
    Args:
        yaml_content: YAML content as string
        allow_missing_env_vars: If True, missing environment variables are left as-is
        
    Returns:
        List of parsed Rule objects
        
    Raises:
        ParseError: If parsing fails
    """
    parser = RuleParser(allow_missing_env_vars=allow_missing_env_vars)
    return parser.parse_yaml(yaml_content)


def validate_and_load_rules(file_path: Union[str, Path]) -> Tuple[List[Rule], ValidationResult]:
    """Validate and load rules, returning both results and validation details.
    
    Args:
        file_path: Path to YAML rules file
        
    Returns:
        Tuple of (parsed rules, validation result)
    """
    # First validate the file
    validation_result = validate_rules_file(Path(file_path))
    
    if not validation_result.is_valid:
        return [], validation_result
    
    try:
        # If validation passes, parse the rules
        rules = load_rules_from_file(file_path)
        return rules, validation_result
    except ParseError:
        # If parsing fails after validation passes, create a new validation result
        # This shouldn't normally happen if schema is comprehensive
        validation_result = ValidationResult(
            valid=False,
            errors=[{
                'path': 'parsing',
                'message': 'Rule parsing failed after schema validation passed'
            }]
        )
        return [], validation_result


def load_rules_for_environment(config_sources: List[Union[str, Path, Dict[str, Any]]], 
                              environment: str,
                              allow_missing_env_vars: bool = False) -> List[Rule]:
    """Convenience function to load rules for a specific environment with inheritance.
    
    Args:
        config_sources: List of configuration sources (first is base, others overlay)
        environment: Target environment name
        allow_missing_env_vars: If True, missing environment variables are left as-is
        
    Returns:
        List of parsed Rule objects for the environment
        
    Raises:
        ParseError: If parsing fails
    """
    parser = AdvancedRuleParser(allow_missing_env_vars=allow_missing_env_vars)
    parser.load_configuration_chain(config_sources)
    return parser.parse_for_environment(environment)


def validate_configuration_chain(config_sources: List[Union[str, Path, Dict[str, Any]]]) -> List[str]:
    """Validate a configuration inheritance chain.
    
    Args:
        config_sources: List of configuration sources to validate
        
    Returns:
        List of validation warnings and errors
    """
    try:
        parser = AdvancedRuleParser(allow_missing_env_vars=True)
        parser.load_configuration_chain(config_sources)
        return parser.validate_configuration_chain()
    except Exception as e:
        return [f"Configuration chain validation failed: {e}"]


def get_available_environments(config_sources: List[Union[str, Path, Dict[str, Any]]]) -> List[str]:
    """Get available environments from configuration sources.
    
    Args:
        config_sources: List of configuration sources
        
    Returns:
        List of available environment names
    """
    try:
        parser = AdvancedRuleParser(allow_missing_env_vars=True)
        parser.load_configuration_chain(config_sources)
        return parser.get_available_environments()
    except Exception:
        return []


def create_environment_override_config(base_config_path: Union[str, Path],
                                     environment_overrides: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Create a configuration with environment-specific overrides.
    
    This is a utility function to help create configurations that work well
    with the inheritance system.
    
    Args:
        base_config_path: Path to base configuration file
        environment_overrides: Dict mapping environment names to override configs
        
    Returns:
        Configuration dictionary with environment overrides
        
    Example:
        overrides = {
            'staging': {
                'defaults': {'severity': 'warning'},
                'rules': [
                    {'id': 'ga4-pageview-present', 'enabled': False}
                ]
            }
        }
        config = create_environment_override_config('base.yaml', overrides)
    """
    # Load base configuration
    base_path = Path(base_config_path)
    if not base_path.exists():
        raise ParseError(f"Base configuration file not found: {base_path}")
    
    try:
        base_content = base_path.read_text(encoding='utf-8')
        base_config = yaml.safe_load(base_content)
    except Exception as e:
        raise ParseError(f"Error loading base configuration: {e}")
    
    # Create the override configuration
    override_config = {
        'version': base_config.get('version', '0.1'),
        'meta': {
            'name': f"Environment overrides for {base_config.get('meta', {}).get('name', 'rules')}",
            'description': 'Environment-specific rule overrides and configurations'
        }
    }
    
    # Add environment-specific sections
    for env_name, env_overrides in environment_overrides.items():
        # Add to environments section if it has environment variables
        if 'environments' in env_overrides:
            if 'environments' not in override_config:
                override_config['environments'] = {}
            override_config['environments'][env_name] = env_overrides['environments']
        
        # Add rules with environment filtering
        if 'rules' in env_overrides:
            if 'rules' not in override_config:
                override_config['rules'] = []
            
            for rule_override in env_overrides['rules']:
                # Ensure rules are scoped to the environment
                if 'applies_to' not in rule_override:
                    rule_override['applies_to'] = {}
                if 'environments' not in rule_override['applies_to']:
                    rule_override['applies_to']['environments'] = [env_name]
                elif env_name not in rule_override['applies_to']['environments']:
                    rule_override['applies_to']['environments'].append(env_name)
                
                override_config['rules'].append(rule_override)
        
        # Add defaults with environment-specific application
        if 'defaults' in env_overrides:
            override_config['defaults'] = env_overrides['defaults']
    
    return override_config