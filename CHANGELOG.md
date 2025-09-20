# Changelog

All notable changes to Tag Sentinel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-15

### Added
- **CLI Framework (EPIC 10)**: Complete command-line interface using Typer
  - Modern CLI with subcommands and rich help text
  - Configuration system with precedence (CLI flags > env vars > config files > defaults)
  - Progress tracking with real-time output and progress bars
  - Enhanced summary output with multiple formats (text, JSON, YAML)
  - Comprehensive exit code management for CI/CD integration
  - Package management and distribution setup

- **Configuration System**:
  - Support for YAML, JSON, and environment variable configuration
  - Auto-discovery of configuration files
  - Hierarchical configuration with proper precedence
  - Validation and error reporting

- **Progress Tracking**:
  - Real-time progress indicators with timestamps
  - Progress bars showing completion percentages
  - Detailed step breakdown with timing information
  - Support for both text and JSON output formats

- **Exit Code Management**:
  - Standardized exit codes for CI/CD integration
  - Proper error categorization (success, rule failures, critical failures, config errors)
  - Comprehensive error reporting in summaries

- **Enhanced Output**:
  - Text format with emoji indicators and clear structure
  - JSON format for machine processing
  - YAML format for human-readable structured output
  - Progress tracking integration in summaries

### CLI Commands
- `openaudit run`: Execute audits with comprehensive options
- `openaudit validate-config`: Validate configuration files
- `openaudit version`: Show version information

### Console Scripts
- `openaudit`: Primary CLI entry point
- `tag-sentinel`: Alternative CLI entry point

### Dependencies
- Typer for modern CLI framework
- Pydantic for configuration validation
- PyYAML for configuration file support
- Rich ecosystem for enhanced terminal output

### Documentation
- Comprehensive EXIT_CODES.md with CI/CD integration examples
- Updated README.md with CLI usage examples
- Detailed docstrings throughout the codebase

### Package Management
- Complete pyproject.toml configuration
- Support for building wheels and source distributions
- Proper entry point configuration
- MIT license with SPDX identifier

## [Unreleased]

### Planned
- Rule evaluation integration with the new CLI
- Alert dispatching through the CLI
- Additional output formats
- Plugin system for custom validators
- Performance optimizations