"""JSON Schema validation for dataLayer snapshots.

This module provides comprehensive JSON Schema validation with detailed error
reporting, severity mapping, and support for complex schema patterns including
references, conditionals, and custom formats.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
from urllib.parse import urlparse, urljoin
import re
from datetime import datetime

from .models import ValidationIssue, ValidationSeverity
from .config import SchemaConfig
from .runtime_validation import validate_types, validate_validation_issues

logger = logging.getLogger(__name__)

try:
    import jsonschema
    from jsonschema import Draft202012Validator, RefResolver, ValidationError as JSValidationError
    from jsonschema.exceptions import SchemaError
    HAS_JSONSCHEMA = True
except ImportError:
    logger.warning("jsonschema library not available - validation will be disabled")
    jsonschema = None
    Draft202012Validator = None
    RefResolver = None
    JSValidationError = None
    SchemaError = None
    HAS_JSONSCHEMA = False


class ValidationError(Exception):
    """Errors during schema validation."""
    pass


class SchemaValidationError(Exception):
    """Schema-specific validation errors."""
    pass


class SchemaLoader:
    """Loads and manages JSON/YAML schemas with caching and reference resolution."""
    
    def __init__(self, config: SchemaConfig | None = None):
        """Initialize schema loader.
        
        Args:
            config: Schema configuration
        """
        self.config = config or SchemaConfig()
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._resolver_cache: Dict[str, RefResolver] = {}
    
    def load_schema(self, schema_path: str | Path) -> Dict[str, Any]:
        """Load schema from file with caching.
        
        Args:
            schema_path: Path to schema file (.json or .yaml/.yml)
            
        Returns:
            Loaded schema dictionary
            
        Raises:
            SchemaValidationError: If schema loading fails
        """
        schema_path = Path(schema_path)
        schema_key = str(schema_path.absolute())
        
        # Check cache if caching is enabled
        if self.config.cache_schemas and schema_key in self._schema_cache:
            logger.debug(f"Using cached schema: {schema_path}")
            return self._schema_cache[schema_key]
        
        if not schema_path.exists():
            raise SchemaValidationError(f"Schema file not found: {schema_path}")
        
        try:
            logger.debug(f"Loading schema from: {schema_path}")
            
            # Load based on file extension
            with open(schema_path, 'r', encoding='utf-8') as f:
                if schema_path.suffix.lower() == '.json':
                    schema = json.load(f)
                elif schema_path.suffix.lower() in ['.yaml', '.yml']:
                    if not self.config.allow_yaml_schemas:
                        raise SchemaValidationError(
                            f"YAML schemas not allowed: {schema_path}"
                        )
                    schema = yaml.safe_load(f)
                else:
                    raise SchemaValidationError(
                        f"Unsupported schema format: {schema_path.suffix}"
                    )
            
            # Validate the schema itself
            self._validate_schema(schema)
            
            # Cache if enabled
            if self.config.cache_schemas:
                self._schema_cache[schema_key] = schema
            
            logger.debug(f"Successfully loaded schema: {schema_path}")
            return schema
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise SchemaValidationError(f"Invalid schema format in {schema_path}: {e}")
        except Exception as e:
            raise SchemaValidationError(f"Failed to load schema {schema_path}: {e}")
    
    def create_resolver(self, schema: Dict[str, Any], schema_path: Path | None = None):
        """Create JSON Schema reference resolver.
        
        Args:
            schema: Root schema
            schema_path: Path to schema file for resolving relative references
            
        Returns:
            Configured RefResolver
        """
        if not self.config.resolve_references or not HAS_JSONSCHEMA:
            return None
        
        # Create base URI for reference resolution
        if schema_path:
            base_uri = schema_path.as_uri()
        else:
            base_uri = ""
        
        # Check cache
        cache_key = f"{base_uri}:{id(schema)}"
        if cache_key in self._resolver_cache:
            return self._resolver_cache[cache_key]
        
        try:
            resolver = RefResolver(base_uri=base_uri, referrer=schema)
            
            # Cache resolver
            self._resolver_cache[cache_key] = resolver
            
            return resolver
            
        except Exception as e:
            logger.warning(f"Failed to create resolver: {e}")
            return None
    
    def _validate_schema(self, schema: Dict[str, Any]) -> None:
        """Validate that the schema itself is valid JSON Schema.
        
        Args:
            schema: Schema to validate
            
        Raises:
            SchemaValidationError: If schema is invalid
        """
        if not HAS_JSONSCHEMA:
            logger.warning("Cannot validate schema - jsonschema not available")
            return
        
        try:
            # Use Draft 2020-12 meta-schema for validation
            Draft202012Validator.check_schema(schema)
            
        except SchemaError as e:
            raise SchemaValidationError(f"Invalid JSON Schema: {e.message}")
    
    def clear_cache(self) -> None:
        """Clear schema and resolver caches."""
        self._schema_cache.clear()
        self._resolver_cache.clear()
        logger.debug("Schema cache cleared")


class Validator:
    """Validates dataLayer data against JSON schemas with detailed error reporting."""
    
    def __init__(self, config: SchemaConfig | None = None):
        """Initialize validator.
        
        Args:
            config: Schema validation configuration
        """
        self.config = config or SchemaConfig()
        self.schema_loader = SchemaLoader(config)
        self._validator_cache: Dict[str, Draft202012Validator] = {}
    
    def validate_data(
        self,
        data: Dict[str, Any],
        schema: str | Path | Dict[str, Any],
        page_url: str
    ) -> List[ValidationIssue]:
        """Validate data against a JSON schema.
        
        Args:
            data: Data to validate
            schema: Schema dictionary or path to schema file
            page_url: URL of the page being validated
            
        Returns:
            List of validation issues found
        """
        if not self.config.enabled:
            logger.debug("Schema validation disabled")
            return []
        
        if not HAS_JSONSCHEMA:
            logger.warning("jsonschema library not available - skipping validation")
            return []
        
        try:
            # Load schema if it's a path
            if isinstance(schema, (str, Path)):
                schema_dict = self.schema_loader.load_schema(schema)
                schema_path = Path(schema)
            else:
                schema_dict = schema
                schema_path = None
            
            # Create or get cached validator
            validator = self._get_validator(schema_dict, schema_path)
            
            # Perform validation
            validation_issues = []
            errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
            
            for error in errors:
                issue = self._create_validation_issue(error, page_url)
                validation_issues.append(issue)
            
            logger.debug(f"Validation complete: {len(validation_issues)} issues found")
            return validation_issues
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            # Return a critical validation issue for the error
            return [ValidationIssue(
                page_url=page_url,
                path="/",
                severity=ValidationSeverity.CRITICAL,
                message=f"Schema validation error: {e}",
                schema_rule="validation_error"
            )]
    
    @validate_types()
    def validate_snapshot(
        self,
        latest_data: Dict[str, Any],
        events_data: List[Dict[str, Any]],
        schema_path: str | Path,
        page_url: str
    ) -> List[ValidationIssue]:
        """Validate a complete dataLayer snapshot.
        
        Args:
            latest_data: Latest state data
            events_data: Events data
            schema_path: Path to schema file
            page_url: URL of the page
            
        Returns:
            Combined validation issues for all data
        """
        all_issues = []
        
        # Validate latest state
        latest_issues = self.validate_data(latest_data, schema_path, page_url)
        all_issues.extend(latest_issues)
        
        # Validate each event
        for i, event in enumerate(events_data):
            event_issues = self.validate_data(event, schema_path, page_url)
            
            # Adjust paths to indicate event context
            for issue in event_issues:
                issue.path = f"/events/{i}{issue.path}"
                issue.event_type = event.get('event') or event.get('eventName')
            
            all_issues.extend(event_issues)
        
        return all_issues
    
    def _get_validator(
        self,
        schema: Dict[str, Any],
        schema_path: Path | None = None
    ):
        """Get or create a JSON Schema validator with caching.
        
        Args:
            schema: Schema dictionary
            schema_path: Optional path for reference resolution
            
        Returns:
            Configured validator
        """
        # Create stable cache key based on schema content, not object id
        import hashlib
        schema_hash = hashlib.sha256(
            json.dumps(schema, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        cache_key = f"{schema_hash}:{schema_path}"
        
        if cache_key in self._validator_cache:
            logger.debug(f"Using cached validator for schema {schema_hash}")
            return self._validator_cache[cache_key]
        
        # Create resolver for references
        resolver = self.schema_loader.create_resolver(schema, schema_path)
        
        # Create validator
        if resolver:
            validator = Draft202012Validator(schema, resolver=resolver)
        else:
            validator = Draft202012Validator(schema)
        
        # Cache validator with size limit
        if len(self._validator_cache) > 100:  # Limit cache size
            # Remove oldest entries (simple LRU approximation)
            oldest_keys = list(self._validator_cache.keys())[:20]
            for key in oldest_keys:
                del self._validator_cache[key]
            logger.debug("Validator cache cleanup: removed 20 oldest entries")
        
        self._validator_cache[cache_key] = validator
        logger.debug(f"Cached new validator for schema {schema_hash}")
        
        return validator
    
    def _create_validation_issue(
        self,
        error,
        page_url: str
    ) -> ValidationIssue:
        """Create a ValidationIssue from a jsonschema ValidationError.
        
        Args:
            error: JSON Schema validation error
            page_url: URL of the page being validated
            
        Returns:
            ValidationIssue object
        """
        # Build JSON Pointer path
        path_parts = []
        for part in error.absolute_path:
            if isinstance(part, int):
                path_parts.append(str(part))
            else:
                path_parts.append(str(part))
        
        json_path = "/" + "/".join(path_parts) if path_parts else "/"
        
        # Determine severity based on schema rule
        severity = self._map_error_to_severity(error)
        
        # Extract variable name from path
        variable_name = None
        if len(path_parts) > 0:
            variable_name = path_parts[0]
        
        # Format expected/observed values
        expected = self._format_expected_value(error)
        observed = error.instance
        
        # Clean up observed value for display
        if isinstance(observed, str) and len(observed) > 100:
            observed = observed[:100] + "..."
        
        return ValidationIssue(
            page_url=page_url,
            path=json_path,
            severity=severity,
            message=error.message,
            schema_rule=error.validator,
            expected=expected,
            observed=observed,
            variable_name=variable_name
        )
    
    def _map_error_to_severity(self, error) -> ValidationSeverity:
        """Map JSON Schema validation error to severity level.
        
        Args:
            error: Validation error
            
        Returns:
            Appropriate severity level
        """
        rule_type = error.validator
        
        # Use configured severity mapping
        severity_str = self.config.severity_mapping.get(
            rule_type, 
            ValidationSeverity.WARNING
        )
        
        # Handle string to enum conversion
        if isinstance(severity_str, str):
            try:
                return ValidationSeverity(severity_str)
            except ValueError:
                return ValidationSeverity.WARNING
        
        return severity_str
    
    def _format_expected_value(self, error) -> str | None:
        """Format expected value from validation error.
        
        Args:
            error: Validation error
            
        Returns:
            Formatted expected value description
        """
        validator = error.validator
        schema = error.schema
        
        if validator == "type":
            expected_types = schema.get("type", [])
            if isinstance(expected_types, list):
                return f"type: {' or '.join(expected_types)}"
            else:
                return f"type: {expected_types}"
        
        elif validator == "required":
            missing_props = error.validator_value
            return f"required properties: {', '.join(missing_props)}"
        
        elif validator == "enum":
            allowed_values = schema.get("enum", [])
            return f"one of: {', '.join(map(str, allowed_values))}"
        
        elif validator == "pattern":
            pattern = schema.get("pattern", "")
            return f"pattern: {pattern}"
        
        elif validator in ["minimum", "maximum"]:
            return f"{validator}: {error.validator_value}"
        
        elif validator in ["minLength", "maxLength"]:
            return f"{validator}: {error.validator_value}"
        
        elif validator == "format":
            format_name = schema.get("format", "")
            return f"format: {format_name}"
        
        else:
            # Generic fallback
            return f"{validator}: {error.validator_value}"
    
    def get_validation_summary(
        self,
        issues: List[ValidationIssue]
    ) -> Dict[str, Any]:
        """Generate validation summary statistics.
        
        Args:
            issues: List of validation issues
            
        Returns:
            Summary statistics
        """
        if not issues:
            return {
                'total_issues': 0,
                'by_severity': {},
                'by_rule': {},
                'by_variable': {},
                'most_common_issues': []
            }
        
        # Count by severity
        severity_counts = {}
        for issue in issues:
            severity = str(issue.severity)
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by rule type
        rule_counts = {}
        for issue in issues:
            rule = issue.schema_rule or 'unknown'
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        # Count by variable
        variable_counts = {}
        for issue in issues:
            var_name = issue.variable_name or 'unknown'
            variable_counts[var_name] = variable_counts.get(var_name, 0) + 1
        
        # Find most common issues
        issue_patterns = {}
        for issue in issues:
            pattern = f"{issue.schema_rule}: {issue.message[:50]}..."
            issue_patterns[pattern] = issue_patterns.get(pattern, 0) + 1
        
        most_common = sorted(
            issue_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'total_issues': len(issues),
            'by_severity': severity_counts,
            'by_rule': rule_counts,
            'by_variable': variable_counts,
            'most_common_issues': [
                {'pattern': pattern, 'count': count}
                for pattern, count in most_common
            ]
        }
    
    def clear_cache(self) -> None:
        """Clear the validator cache."""
        cache_size = len(self._validator_cache)
        self._validator_cache.clear()
        logger.debug(f"Cleared validator cache ({cache_size} entries)")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the validator cache.
        
        Returns:
            Cache statistics
        """
        return {
            'size': len(self._validator_cache),
            'max_size': 100,
            'keys': list(self._validator_cache.keys())
        }


class SchemaManager:
    """Manages multiple schemas and validation operations."""
    
    def __init__(self, config: SchemaConfig | None = None):
        """Initialize schema manager.
        
        Args:
            config: Schema configuration
        """
        self.config = config or SchemaConfig()
        self.validator = Validator(config)
        self._schema_registry: Dict[str, Dict[str, Any]] = {}
    
    def register_schema(
        self,
        name: str,
        schema: str | Path | Dict[str, Any]
    ) -> None:
        """Register a schema for later use.
        
        Args:
            name: Schema identifier
            schema: Schema dictionary or path to schema file
        """
        if isinstance(schema, (str, Path)):
            schema_dict = self.validator.schema_loader.load_schema(schema)
        else:
            schema_dict = schema
        
        self._schema_registry[name] = schema_dict
        logger.debug(f"Registered schema: {name}")
    
    def validate_with_schema(
        self,
        data: Dict[str, Any],
        schema_name: str,
        page_url: str
    ) -> List[ValidationIssue]:
        """Validate data using a registered schema.
        
        Args:
            data: Data to validate
            schema_name: Name of registered schema
            page_url: URL of the page
            
        Returns:
            List of validation issues
        """
        if schema_name not in self._schema_registry:
            raise ValidationError(f"Schema '{schema_name}' not registered")
        
        schema = self._schema_registry[schema_name]
        return self.validator.validate_data(data, schema, page_url)
    
    def validate_with_site_schema(
        self,
        data: Dict[str, Any],
        site_domain: str,
        page_url: str,
        fallback_schema: str | None = None
    ) -> List[ValidationIssue]:
        """Validate data using site-specific schema with fallback.
        
        Args:
            data: Data to validate
            site_domain: Site domain to get schema for
            page_url: URL of the page
            fallback_schema: Fallback schema name if site schema not found
            
        Returns:
            List of validation issues
        """
        # Try site-specific schema first
        site_schema_name = f"site_{site_domain.replace('.', '_')}"
        
        if site_schema_name in self._schema_registry:
            return self.validate_with_schema(data, site_schema_name, page_url)
        
        # Try fallback schema
        if fallback_schema and fallback_schema in self._schema_registry:
            return self.validate_with_schema(data, fallback_schema, page_url)
        
        # No schema available
        logger.warning(f"No schema available for site {site_domain}")
        return []
    
    def get_registered_schemas(self) -> List[str]:
        """Get list of registered schema names.
        
        Returns:
            List of schema names
        """
        return list(self._schema_registry.keys())
    
    def clear_schemas(self) -> None:
        """Clear all registered schemas."""
        self._schema_registry.clear()
        self.validator.schema_loader.clear_cache()
        logger.debug("Schema registry cleared")