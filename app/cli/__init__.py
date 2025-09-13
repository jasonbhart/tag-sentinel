"""CLI module for Tag Sentinel.

This package provides command-line interface functionality for running
audits, evaluating rules, and generating reports from the command line.
"""

from .runner import (
    # Exit codes
    ExitCode,
    
    # Main CLI runner
    CLIRunner,
    run_audit_with_rules,
    
    # Configuration
    CLIConfig,
    RuleEvaluationConfig,
)

__all__ = [
    # Exit codes
    'ExitCode',
    
    # Main CLI runner
    'CLIRunner',
    'run_audit_with_rules',
    
    # Configuration
    'CLIConfig',
    'RuleEvaluationConfig',
]