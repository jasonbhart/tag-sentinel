"""JSON Schema definitions and validation for YAML rule configurations.

This module provides versioned JSON Schema definitions for rule YAML files,
validation functions, and error reporting for rule configuration validation.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import jsonschema
from jsonschema import Draft7Validator, ValidationError

from .models import Severity, CheckType, AlertChannelType


class SchemaVersion:
    """Schema version constants."""
    V0_1 = "0.1"
    CURRENT = V0_1
    SUPPORTED = [V0_1]


class ValidationResult:
    """Validation result compatible with tests.

    Exposes:
    - valid: bool             -> overall validity
    - errors: List[str]       -> human-readable error messages
    - warnings: List[str]     -> non-fatal issues
    - schema_version: str|None
    """

    def __init__(
        self,
        valid: bool,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        schema_version: Optional[str] = None,
    ):
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.schema_version = schema_version

    @property
    def is_valid(self) -> bool:
        """Alias for valid property for test compatibility."""
        return self.valid

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def get_error_summary(self) -> str:
        if not self.has_errors:
            return "No validation errors"
        return (
            f"1 validation error: {self.errors[0]}"
            if len(self.errors) == 1
            else f"{len(self.errors)} validation errors found"
        )
    
    def get_detailed_errors(self) -> List[str]:
        """Get detailed error messages."""
        return self.errors.copy()


def get_schema_v0_1() -> Dict[str, Any]:
    """Get JSON Schema for rules YAML version 0.1."""
    
    # Define enum values from our models
    severity_values = [s.value for s in Severity]
    check_type_values = [ct.value for ct in CheckType]
    alert_channel_values = [ac.value for ac in AlertChannelType]
    
    return {
        "$schema": "https://json-schema.org/draft-07/schema#",
        "$id": "https://tag-sentinel.dev/schemas/rules/v0.1.json",
        "title": "Tag Sentinel Rules Configuration",
        "description": "Schema for Tag Sentinel YAML rule configuration files",
        "type": "object",
        "required": ["version", "rules"],
        # Allow additional properties to be present without failing validation
        "additionalProperties": True,
        "properties": {
            "version": {
                "type": "string",
                "const": "0.1",
                "description": "Schema version identifier"
            },
            "meta": {
                "type": "object",
                "description": "Metadata about the rule configuration",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Human-readable name for this rule set"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of the rule set purpose"
                    },
                    "author": {
                        "type": "string",
                        "description": "Author of the rule set"
                    },
                    "created": {
                        "type": "string",
                        "format": "date",
                        "description": "Creation date (YYYY-MM-DD)"
                    }
                }
            },
            "defaults": {
                "type": "object",
                "description": "Default values applied to all rules",
                "additionalProperties": False,
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": severity_values,
                        "default": "warning",
                        "description": "Default severity level for rules"
                    },
                    "enabled": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether rules are enabled by default"
                    },
                    "applies_to": {
                        "$ref": "#/$defs/applies_to",
                        "description": "Default scoping for all rules"
                    }
                }
            },
            "environments": {
                "type": "object",
                "description": "Environment-specific configuration values",
                "patternProperties": {
                    "^[a-zA-Z][a-zA-Z0-9_-]*$": {
                        "type": "object",
                        "description": "Environment configuration",
                        "additionalProperties": True
                    }
                }
            },
            "rules": {
                "type": "array",
                # Allow empty rules list
                "minItems": 0,
                "description": "List of rule definitions",
                "items": {
                    "$ref": "#/$defs/rule"
                }
            },
            "alerts": {
                "type": "object",
                "description": "Alert configuration",
                "additionalProperties": True,
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether alerts are enabled globally"
                    },
                    "channels": {
                        "type": "array",
                        "description": "Alert notification channels",
                        "items": {
                            "$ref": "#/$defs/alert_channel"
                        }
                    }
                }
            }
        },
        "$defs": {
            "applies_to": {
                "type": "object",
                "description": "Rule scoping configuration",
                "additionalProperties": True,
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["page", "run"],
                        "description": "Scope level where rule applies"
                    },
                    "environments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Environment names where rule applies"
                    },
                    "scenario_ids": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "Scenario IDs where rule applies"
                    },
                    "url_include": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URL regex patterns to include"
                    },
                    "url_exclude": {
                        "type": "array",
                        "items": {"type": "string"}, 
                        "description": "URL regex patterns to exclude"
                    },
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific URLs where rule applies"
                    },
                    "vendors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Vendor/tag types to include"
                    }
                }
            },
            "check_config": {
                "type": "object",
                "required": ["type"],
                # Allow additional properties and a "parameters" object per tests
                "additionalProperties": True,
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": check_type_values,
                        "description": "Type of check to perform"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Arbitrary parameter map for the check"
                    },
                    "vendor": {
                        "type": "string",
                        "description": "Vendor/tag type (e.g., 'ga4', 'gtm')"
                    },
                    "url_pattern": {
                        "type": "string",
                        "description": "URL regex pattern to match"
                    },
                    "min_count": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Minimum required count"
                    },
                    "max_count": {
                        "type": "integer", 
                        "minimum": 0,
                        "description": "Maximum allowed count"
                    },
                    "time_window_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Time window in milliseconds"
                    },
                    "expression": {
                        "type": "string",
                        "description": "Expression to evaluate"
                    },
                    "config": {
                        "type": "object",
                        "description": "Additional check-specific configuration"
                    }
                },
                "allOf": [
                    {
                        "if": {"properties": {"type": {"const": "request_present"}}},
                        "then": {
                            "anyOf": [
                                {"required": ["vendor"]},
                                {"required": ["url_pattern"]},
                                {
                                    "properties": {
                                        "parameters": {
                                            "type": "object",
                                            "anyOf": [
                                                {"required": ["vendor"]},
                                                {"required": ["url_pattern"]}
                                            ]
                                        }
                                    },
                                    "required": ["parameters"]
                                }
                            ]
                        }
                    },
                    {
                        "if": {"properties": {"type": {"const": "script_present"}}},
                        "then": {
                            "anyOf": [
                                {"required": ["url_pattern"]},
                                {
                                    "properties": {
                                        "parameters": {
                                            "type": "object",
                                            "required": ["url_pattern"]
                                        }
                                    },
                                    "required": ["parameters"]
                                }
                            ]
                        }
                    },
                    {
                        "if": {"properties": {"type": {"const": "duplicate_requests"}}},
                        "then": {
                            "anyOf": [
                                {
                                    "properties": {
                                        "max_count": {"minimum": 1},
                                        "time_window_ms": {"minimum": 100}
                                    }
                                },
                                {
                                    "properties": {
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "max_count": {"minimum": 1},
                                                "time_window_ms": {"minimum": 100}
                                            }
                                        }
                                    },
                                    "required": ["parameters"]
                                }
                            ]
                        }
                    },
                    {
                        "if": {"properties": {"type": {"const": "expression"}}},
                        "then": {
                            "anyOf": [
                                {"required": ["expression"]},
                                {
                                    "properties": {
                                        "parameters": {
                                            "type": "object",
                                            "required": ["expression"]
                                        }
                                    },
                                    "required": ["parameters"]
                                }
                            ]
                        }
                    }
                ]
            },
            "rule": {
                "type": "object",
                "required": ["id", "name", "check"],
                # Allow custom fields on rule objects
                "additionalProperties": True,
                "properties": {
                    "id": {
                        "type": "string",
                        "pattern": "^[a-zA-Z][a-zA-Z0-9_-]*$",
                        "description": "Unique rule identifier"
                    },
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Human-readable rule name"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed rule description"
                    },
                    "severity": {
                        "type": "string",
                        "enum": severity_values,
                        "description": "Rule severity level"
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "Whether rule is enabled"
                    },
                    "applies_to": {
                        "$ref": "#/$defs/applies_to",
                        "description": "Rule scoping configuration"
                    },
                    "check": {
                        "anyOf": [
                            {"$ref": "#/$defs/check_config"}
                        ],
                        "description": "Check configuration"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Rule tags for organization"
                    }
                }
            },
            "alert_channel": {
                "type": "object",
                "required": ["type"],
                "additionalProperties": False,
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": alert_channel_values,
                        "description": "Alert channel type"
                    },
                    "enabled": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether this channel is enabled"
                    },
                    "min_severity": {
                        "type": "string",
                        "enum": severity_values,
                        "default": "critical",
                        "description": "Minimum severity to trigger alerts"
                    },
                    "webhook_url": {
                        "type": "string",
                        "format": "uri",
                        "description": "Webhook URL for webhook alerts"
                    },
                    "webhook_secret": {
                        "type": "string",
                        "description": "Secret for webhook HMAC signing"
                    },
                    "email_recipients": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "format": "email"
                        },
                        "description": "Email recipients"
                    },
                    "email_smtp_config": {
                        "type": "object",
                        "description": "SMTP configuration"
                    },
                    "throttle_minutes": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 60,
                        "description": "Minutes between duplicate alerts"
                    }
                },
                "allOf": [
                    {
                        "if": {"properties": {"type": {"const": "webhook"}}},
                        "then": {"required": ["webhook_url"]}
                    },
                    {
                        "if": {"properties": {"type": {"const": "email"}}},
                        "then": {"required": ["email_recipients"]}
                    }
                ]
            }
        }
    }


def get_schema_by_version(version: str) -> Dict[str, Any]:
    """Get JSON Schema by version string."""
    if version == SchemaVersion.V0_1:
        return get_schema_v0_1()
    else:
        raise ValueError(f"Unsupported schema version: {version}. Supported versions: {SchemaVersion.SUPPORTED}")


def validate_rules_config(config: Dict[str, Any]) -> ValidationResult:
    """Validate a rules configuration dictionary against the appropriate schema.
    
    Args:
        config: The parsed YAML configuration as a dictionary
        
    Returns:
        ValidationResult with validation status and detailed error information
    """
    # Basic type validation first
    if not isinstance(config, dict):
        return ValidationResult(valid=False, errors=["Configuration must be a dictionary"], warnings=[])

    # Extract version from config
    version = config.get('version')
    if not version:
        return ValidationResult(valid=False, errors=["Version field is required"], warnings=[])
    
    # Check if version is supported
    if version not in SchemaVersion.SUPPORTED:
        return ValidationResult(valid=False, errors=[f'Unsupported version "{version}". Supported versions: {SchemaVersion.SUPPORTED}'], warnings=[])
    
    # Get schema for version
    try:
        schema = get_schema_by_version(version)
    except ValueError as e:
        return ValidationResult(valid=False, errors=[str(e)], warnings=[])
    
    # Validate against schema
    validator = Draft7Validator(schema)
    validation_errors: List[str] = []
    
    for error in validator.iter_errors(config):
        # Convert JSONPath to a readable path
        path_parts = []
        for part in error.absolute_path:
            if isinstance(part, int):
                path_parts.append(f"[{part}]")
            else:
                if path_parts:
                    path_parts.append(f".{part}")
                else:
                    path_parts.append(str(part))
        path = "".join(path_parts) if path_parts else "root"
        validation_errors.append(f"Error at '{path}': {error.message}")
    
    return ValidationResult(valid=len(validation_errors) == 0, errors=validation_errors, warnings=[], schema_version=version)


def validate_rules_yaml(yaml_content: str) -> ValidationResult:
    """Validate YAML rules content as string.
    
    Args:
        yaml_content: Raw YAML content as string
        
    Returns:
        ValidationResult with validation status and error details
    """
    import yaml
    
    try:
        # Parse YAML
        config = yaml.safe_load(yaml_content)
        if config is None:
            return ValidationResult(valid=False, errors=["YAML content is empty or invalid"], warnings=[])
        
        # Validate parsed config
        return validate_rules_config(config)
        
    except yaml.YAMLError as e:
        return ValidationResult(valid=False, errors=[f'YAML parsing error: {e}'], warnings=[])
    except Exception as e:
        return ValidationResult(valid=False, errors=[f'Unexpected validation error: {e}'], warnings=[])


def validate_rules_file(file_path: Path) -> ValidationResult:
    """Validate a YAML rules file.
    
    Args:
        file_path: Path to the YAML rules file
        
    Returns:
        ValidationResult with validation status and error details
    """
    try:
        # Check if file exists
        if not file_path.exists():
            return ValidationResult(valid=False, errors=[f'Rules file not found: {file_path}'], warnings=[])
        
        # Read and validate file content
        yaml_content = file_path.read_text(encoding='utf-8')
        result = validate_rules_yaml(yaml_content)
        
        # Add file context to errors
        # No mutation necessary for string-based errors
        
        return result
        
    except Exception as e:
        return ValidationResult(valid=False, errors=[f'Error reading rules file {file_path}: {e}'], warnings=[])


def generate_example_config() -> Dict[str, Any]:
    """Generate an example rules configuration following the schema."""
    return {
        "version": "0.1",
        "meta": {
            "name": "Example Analytics Rules",
            "description": "Example rule configuration for analytics auditing",
            "author": "Tag Sentinel",
            "created": "2025-01-01"
        },
        "defaults": {
            "severity": "warning",
            "enabled": True,
            "applies_to": {
                "environments": ["production"]
            }
        },
        "environments": {
            "production": {
                "ga4_measurement_id": "G-XXXXXXXXXX",
                "gtm_container_id": "GTM-XXXXXXX"
            },
            "staging": {
                "ga4_measurement_id": "G-YYYYYYYYYY", 
                "gtm_container_id": "GTM-YYYYYYY"
            }
        },
        "rules": [
            {
                "id": "gtm-script-present",
                "name": "GTM container script present",
                "description": "Verify GTM container script is present",
                "severity": "critical",
                "check": {
                    "type": "request_present",
                    "parameters": {
                        "url_pattern": "https://www\\.googletagmanager\\.com/gtm\\.js\\?id=GTM-",
                        "method": "GET"
                    },
                    "timeout_seconds": 30,
                    "retry_count": 0,
                    "enabled": True
                },
                "tags": ["gtm", "core"]
            },
            {
                "id": "no-duplicate-pageviews",
                "name": "No duplicate page_view events",
                "description": "Ensure page_view events are not duplicated",
                "severity": "warning",
                "check": {
                    "type": "duplicate_requests",
                    "parameters": {
                        "window_seconds": 5,
                        "event_name": "page_view"
                    }
                },
                "tags": ["analytics", "ga4", "quality"]
            }
        ],
        "alerts": {
            "enabled": True,
            "channels": [
                {
                    "type": "webhook",
                    "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                    "min_severity": "critical",
                    "throttle_minutes": 30
                },
                {
                    "type": "email",
                    "email_recipients": ["alerts@example.com"],
                    "min_severity": "warning",
                    "throttle_minutes": 60
                }
            ]
        }
    }


def export_schema_to_file(version: str, output_path: Path) -> None:
    """Export JSON Schema to a file.
    
    Args:
        version: Schema version to export
        output_path: Path where to save the schema file
    """
    schema = get_schema_by_version(version)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)


def export_example_to_file(output_path: Path) -> None:
    """Export example configuration to a YAML file.
    
    Args:
        output_path: Path where to save the example file
    """
    import yaml
    
    example_config = generate_example_config()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(example_config, f, default_flow_style=False, sort_keys=False, indent=2)
