#!/usr/bin/env python3
"""Main CLI entry point for Tag Sentinel using Typer.

This module provides the primary command-line interface for Tag Sentinel,
implementing the specifications from EPIC 10. It uses Typer for modern
CLI features and integrates with the existing audit runner system.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Optional

import typer
from typing_extensions import Annotated

from .runner import CLIRunner, CLIConfig, ExitCode
from .config import CLIConfiguration, load_configuration, print_configuration, validate_configuration
from .progress import create_audit_progress_tracker, create_real_time_output
from ..audit.rules import Severity


# Create the main Typer app
app = typer.Typer(
    name="openaudit",
    help="Tag Sentinel - Open source web analytics auditing platform",
    add_completion=False,
    rich_markup_mode="rich"
)


def version_callback(value: bool):
    """Show version information."""
    if value:
        typer.echo("Tag Sentinel CLI v1.0.0")
        raise typer.Exit()


@app.callback()
def main():
    """
    Tag Sentinel - Open source web analytics auditing platform.

    A comprehensive CLI for running web analytics audits with support for
    GA4, GTM, cookie analysis, and custom rule evaluation.
    """
    pass


@app.command(name="version")
def show_version():
    """Show version information."""
    typer.echo("Tag Sentinel CLI v1.0.0")


@app.command()
def run(
    # Input configuration
    urls: Annotated[
        Optional[List[str]],
        typer.Argument(help="URLs to audit (can also use --seeds or --sitemap)")
    ] = None,

    env: Annotated[
        Optional[str],
        typer.Option("--env", "-e", help="Environment to audit (dev, staging, production)")
    ] = None,

    seeds: Annotated[
        Optional[Path],
        typer.Option("--seeds", help="Path to seeds file (one URL per line)")
    ] = None,

    sitemap: Annotated[
        Optional[str],
        typer.Option("--sitemap", help="URL to sitemap.xml file")
    ] = None,

    crawl: Annotated[
        Optional[str],
        typer.Option("--crawl", help="Base URL to crawl from")
    ] = None,

    max_pages: Annotated[
        Optional[int],
        typer.Option("--max-pages", help="Maximum pages to audit")
    ] = None,

    # Rule configuration
    rules: Annotated[
        Optional[Path],
        typer.Option("--rules", "-r", help="Path to rules configuration file")
    ] = None,

    # Output configuration
    out: Annotated[
        Optional[Path],
        typer.Option("--out", "-o", help="Output directory for results")
    ] = None,

    output_format: Annotated[
        Optional[str],
        typer.Option("--format", help="Output format")
    ] = None,

    # Browser options
    headful: Annotated[
        bool,
        typer.Option("--headful", help="Run browser with GUI (for debugging)")
    ] = False,

    devtools: Annotated[
        bool,
        typer.Option("--devtools", help="Open browser developer tools")
    ] = False,

    # Debug artifacts
    har: Annotated[
        bool,
        typer.Option("--har", help="Generate HAR files for network requests")
    ] = False,

    har_omit_content: Annotated[
        bool,
        typer.Option("--har-omit-content", help="Omit response content from HAR files")
    ] = False,

    screenshots: Annotated[
        Optional[str],
        typer.Option("--screenshots", help="When to capture screenshots")
    ] = None,

    trace: Annotated[
        bool,
        typer.Option("--trace", help="Generate trace files for debugging")
    ] = False,

    # Execution options
    timeout: Annotated[
        Optional[float],
        typer.Option("--timeout", help="Page load timeout in seconds")
    ] = None,

    concurrency: Annotated[
        Optional[int],
        typer.Option("--concurrency", help="Number of concurrent browser instances")
    ] = None,

    # Rule evaluation options
    scenario: Annotated[
        Optional[str],
        typer.Option("--scenario", "-s", help="Scenario identifier for rule filtering")
    ] = None,

    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast", help="Stop on first critical rule failure")
    ] = False,

    severity_threshold: Annotated[
        Optional[str],
        typer.Option("--severity-threshold", help="Minimum severity level to report")
    ] = None,

    # CI/CD options
    exit_on_critical: Annotated[
        bool,
        typer.Option("--exit-on-critical/--no-exit-on-critical", help="Exit with code 2 on critical failures")
    ] = True,

    exit_on_warnings: Annotated[
        bool,
        typer.Option("--exit-on-warnings/--no-exit-on-warnings", help="Exit with code 1 on warning failures")
    ] = False,

    # Alert options
    enable_alerts: Annotated[
        bool,
        typer.Option("--enable-alerts", help="Enable alert dispatching")
    ] = False,

    alert_config: Annotated[
        Optional[Path],
        typer.Option("--alert-config", help="Path to alert configuration file")
    ] = None,

    # Output options
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,

    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Quiet mode (minimal output)")
    ] = False,

    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output results as JSON")
    ] = False,

    # Configuration debugging
    print_config: Annotated[
        bool,
        typer.Option("--print-config", help="Print effective configuration and exit")
    ] = False,
):
    """
    Run an audit with specified parameters.

    Examples:

        # Basic audit of a single URL
        openaudit run https://example.com

        # Audit with rules and environment
        openaudit run --env staging --rules config/rules.yaml https://example.com

        # Multiple URLs with debug artifacts
        openaudit run --headful --screenshots on_error --har \\
            https://example.com https://staging.example.com

        # CI/CD integration
        openaudit run --env production --rules config/rules.yaml \\
            --seeds seeds.txt --max-pages 1000 --out runs/$(date +%Y-%m-%d)
    """

    # Validate seeds file exists if specified (but skip check for --print-config)
    if seeds and not print_config and not seeds.exists():
        typer.echo(f"❌ Seeds file not found: {seeds}", err=True)
        raise typer.Exit(code=ExitCode.CONFIG_ERROR.value)

    # Sitemap and crawl validation will be handled during URL loading

    # Check if we have any input source (but don't process yet to avoid validation conflicts)
    input_sources = [
        ("URLs", bool(urls)),
        ("--seeds", bool(seeds)),
        ("--sitemap", bool(sitemap)),
        ("--crawl", bool(crawl))
    ]
    active_sources = [name for name, active in input_sources if active]

    # Validate we don't have multiple input sources
    if len(active_sources) > 1:
        typer.echo(f"❌ Multiple input sources provided: {', '.join(active_sources)}. Please specify only one input source.", err=True)
        raise typer.Exit(code=ExitCode.CONFIG_ERROR.value)

    has_input = len(active_sources) > 0
    if not has_input and not print_config:
        typer.echo("❌ No URLs specified. Provide URLs as arguments, or use --seeds, --sitemap, or --crawl", err=True)
        raise typer.Exit(code=ExitCode.CONFIG_ERROR.value)

    # Helper function to normalize and validate severity
    def normalize_severity(severity_str: Optional[str]) -> Optional[Severity]:
        if severity_str is None:
            return None

        severity_map = {
            "info": Severity.INFO,
            "information": Severity.INFO,
            "warn": Severity.WARNING,
            "warning": Severity.WARNING,
            "critical": Severity.CRITICAL,
            "error": Severity.CRITICAL,
            "crit": Severity.CRITICAL
        }

        normalized = severity_str.lower().strip()
        if normalized not in severity_map:
            typer.echo(f"❌ Invalid severity threshold '{severity_str}'. Valid values: info, warning, critical", err=True)
            raise typer.Exit(code=ExitCode.CONFIG_ERROR.value)

        return severity_map[normalized]

    # Create output directory
    if out:
        out.mkdir(parents=True, exist_ok=True)
        output_file = out / f"results.{output_format}"
    else:
        output_file = None

    # Override output format if --json specified
    if json_output:
        output_format = "json"

    # Build CLI overrides dictionary - only include values that were explicitly provided
    cli_overrides = {}

    # Helper function to check if a flag was explicitly provided
    import sys
    def flag_was_provided(flag_name: str) -> bool:
        """Check if a boolean flag was explicitly provided in command line arguments."""
        # Check both affirmative and negated forms
        affirmative = flag_name in sys.argv
        negated = flag_name.replace("--", "--no-") in sys.argv
        return affirmative or negated

    # Basic settings
    if env is not None:
        cli_overrides["environment"] = env

    # Input configuration - replace entire input config to avoid conflicts
    if urls:
        cli_overrides["input"] = {
            "urls": urls,
            "seeds_file": None,
            "sitemap_url": None,
            "crawl_base_url": None
        }
    elif seeds:
        cli_overrides["input"] = {
            "urls": [],
            "seeds_file": seeds,
            "sitemap_url": None,
            "crawl_base_url": None
        }
    elif sitemap:
        cli_overrides["input"] = {
            "urls": [],
            "seeds_file": None,
            "sitemap_url": sitemap,
            "crawl_base_url": None
        }
    elif crawl:
        cli_overrides["input"] = {
            "urls": [],
            "seeds_file": None,
            "sitemap_url": None,
            "crawl_base_url": crawl
        }

    # Execution configuration - only set if provided
    execution_config = {}
    if timeout is not None:
        execution_config["timeout_seconds"] = timeout
    if concurrency is not None:
        execution_config["max_concurrency"] = concurrency
    if max_pages is not None:
        execution_config["max_pages"] = max_pages
    if flag_was_provided("--fail-fast"):
        execution_config["fail_fast"] = fail_fast
    if execution_config:
        cli_overrides["execution"] = execution_config

    # Rules configuration
    rules_config = {}
    severity_enum = normalize_severity(severity_threshold)
    if severity_enum is not None:
        rules_config["severity_threshold"] = severity_enum
    if flag_was_provided("--exit-on-critical"):
        rules_config["exit_on_critical"] = exit_on_critical
    if flag_was_provided("--exit-on-warnings"):
        rules_config["exit_on_warnings"] = exit_on_warnings
    if rules:
        rules_config["rules_file"] = rules
    if scenario:
        rules_config["scenario_id"] = scenario
    if rules_config:
        cli_overrides["rules"] = rules_config

    # Output configuration
    output_config = {}
    effective_format = "json" if json_output else output_format
    if effective_format is not None:
        output_config["format"] = effective_format
    if flag_was_provided("--verbose") or flag_was_provided("-v"):
        output_config["verbose"] = verbose
    if flag_was_provided("--quiet") or flag_was_provided("-q"):
        output_config["quiet"] = quiet
    if flag_was_provided("--json"):
        output_config["json_output"] = json_output
    if out:
        output_config["output_dir"] = out
    if output_config:
        cli_overrides["output"] = output_config

    # Debug configuration
    debug_config = {}
    if flag_was_provided("--headful"):
        debug_config["headful"] = headful
    if flag_was_provided("--devtools"):
        debug_config["devtools"] = devtools
    if flag_was_provided("--har"):
        debug_config["har"] = har
    if flag_was_provided("--har-omit-content"):
        debug_config["har_omit_content"] = har_omit_content
    if flag_was_provided("--trace"):
        debug_config["trace"] = trace
    if screenshots:
        debug_config["screenshots"] = screenshots
    if debug_config:
        cli_overrides["debug"] = debug_config

    # Alert configuration
    alert_config_dict = {}
    if flag_was_provided("--enable-alerts"):
        alert_config_dict["enable_alerts"] = enable_alerts
    if alert_config:
        alert_config_dict["alert_config_path"] = alert_config
    if alert_config_dict:
        cli_overrides["alerts"] = alert_config_dict

    # Load configuration with precedence
    try:
        # Look for config file specified via --rules or auto-discover
        config_file_path = None

        # TODO: Add support for explicit config file parameter
        # For now, let auto-discovery handle it

        full_config = load_configuration(
            config_file=config_file_path,
            cli_overrides=cli_overrides,
            search_paths=[Path.cwd()]
        )

        # Validate configuration (skip validation for --print-config)
        if not print_config:
            validation_errors = validate_configuration(full_config)
            if validation_errors:
                for error in validation_errors:
                    typer.echo(f"❌ Configuration error: {error}", err=True)
                raise typer.Exit(code=ExitCode.CONFIG_ERROR.value)

    except Exception as e:
        typer.echo(f"❌ Configuration error: {e}", err=True)
        raise typer.Exit(code=ExitCode.CONFIG_ERROR.value)

    # Process input URLs after configuration loading to handle all input types
    final_urls = []

    if full_config.input.urls:
        final_urls.extend(full_config.input.urls)
    elif full_config.input.seeds_file:
        # Read seeds file now that configuration is validated
        # Skip file reading only if file doesn't exist and we're in print-config mode
        if print_config and not Path(full_config.input.seeds_file).exists():
            # For print-config, show the configured path even if file doesn't exist
            pass
        else:
            try:
                with open(full_config.input.seeds_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            final_urls.append(line)
            except Exception as e:
                if print_config:
                    # For print-config, tolerate file read errors
                    pass
                else:
                    typer.echo(f"❌ Error reading seeds file {full_config.input.seeds_file}: {e}", err=True)
                    raise typer.Exit(code=ExitCode.CONFIG_ERROR.value)
    elif full_config.input.sitemap_url:
        # Load URLs from sitemap
        try:
            from .input import load_urls_from_sitemap

            if not quiet:
                typer.echo(f"🗺️  Loading URLs from sitemap: {full_config.input.sitemap_url}")

            final_urls = asyncio.run(load_urls_from_sitemap(
                sitemap_url=full_config.input.sitemap_url,
                max_urls=full_config.execution.max_pages,
                timeout=int(full_config.execution.timeout_seconds)
            ))

            if not quiet:
                typer.echo(f"📋 Loaded {len(final_urls)} URLs from sitemap")

        except Exception as e:
            if print_config:
                # For print-config, tolerate sitemap loading errors
                pass
            else:
                typer.echo(f"❌ Error loading sitemap {full_config.input.sitemap_url}: {e}", err=True)
                raise typer.Exit(code=ExitCode.CONFIG_ERROR.value)
    elif full_config.input.crawl_base_url:
        # Load URLs from crawling
        try:
            from .input import load_urls_from_crawl

            if not quiet:
                typer.echo(f"🕷️  Crawling URLs from base: {full_config.input.crawl_base_url}")

            final_urls = asyncio.run(load_urls_from_crawl(
                base_url=full_config.input.crawl_base_url,
                max_urls=full_config.execution.max_pages,
                max_depth=3,  # Default crawl depth
                max_concurrency=full_config.execution.max_concurrency,
                timeout=int(full_config.execution.timeout_seconds)
            ))

            if not quiet:
                typer.echo(f"🔍 Discovered {len(final_urls)} URLs from crawling")

        except Exception as e:
            if print_config:
                # For print-config, tolerate crawl errors
                pass
            else:
                typer.echo(f"❌ Error crawling from {full_config.input.crawl_base_url}: {e}", err=True)
                raise typer.Exit(code=ExitCode.CONFIG_ERROR.value)

    # Update the configuration with final URLs
    full_config.input.urls = final_urls

    # Final validation - ensure we have URLs unless just printing config
    if not final_urls and not print_config:
        typer.echo("❌ No URLs to process. Check your input configuration.", err=True)
        raise typer.Exit(code=ExitCode.CONFIG_ERROR.value)

    # Print configuration if requested
    if print_config:
        typer.echo("# Effective Configuration")
        typer.echo("# Loaded from: " + " -> ".join(full_config.loaded_from))
        typer.echo(print_configuration(full_config, "yaml"))
        raise typer.Exit()

    # Convert configuration to legacy CLIConfig format for the runner
    # TODO: Update runner to use new configuration system directly
    if full_config.output.output_dir:
        output_file = full_config.output.output_dir / f"results.{full_config.output.format}"
    else:
        output_file = None

    config = CLIConfig(
        urls=full_config.input.urls,
        rules_config_path=full_config.rules.rules_file,
        headless=not full_config.debug.headful,
        timeout_seconds=full_config.execution.timeout_seconds,
        max_concurrency=full_config.execution.max_concurrency,
        max_pages=full_config.execution.max_pages,
        environment=full_config.environment,
        scenario_id=full_config.rules.scenario_id,
        fail_fast=full_config.execution.fail_fast,
        severity_threshold=full_config.rules.severity_threshold,
        output_format=full_config.output.format,
        output_file=output_file,
        verbose=full_config.output.verbose,
        quiet=full_config.output.quiet,
        devtools=full_config.debug.devtools,
        har=full_config.debug.har,
        har_omit_content=full_config.debug.har_omit_content,
        screenshots=full_config.debug.screenshots,
        trace=full_config.debug.trace,
        enable_alerts=full_config.alerts.enable_alerts,
        alert_config_path=full_config.alerts.alert_config_path,
        exit_on_critical=full_config.rules.exit_on_critical,
        exit_on_warnings=full_config.rules.exit_on_warnings,
    )

    # Create progress tracking
    progress_tracker = create_audit_progress_tracker(
        quiet=full_config.output.quiet,
        verbose=full_config.output.verbose
    )

    real_time_output = create_real_time_output(
        format_type="json" if full_config.output.json_output else "text",
        quiet=full_config.output.quiet
    )

    # Run the audit with progress tracking
    runner = CLIRunner(config, progress_tracker=progress_tracker, real_time_output=real_time_output)

    try:
        exit_code = asyncio.run(runner.run())

        # The runner now handles enhanced summary output
        # Exit with the appropriate code
        sys.exit(exit_code.value)
    except KeyboardInterrupt:
        typer.echo("❌ Operation interrupted by user", err=True)
        progress_tracker.fail_step("audit", "Interrupted by user")
        raise typer.Exit(code=ExitCode.RUNTIME_ERROR.value)
    except Exception as e:
        typer.echo(f"❌ Runtime error: {e}", err=True)
        progress_tracker.fail_step("audit", str(e))
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=ExitCode.RUNTIME_ERROR.value)


@app.command()
def validate_config(
    config_file: Annotated[
        Path,
        typer.Argument(help="Path to configuration file to validate")
    ],

    environment: Annotated[
        Optional[str],
        typer.Option("--env", "-e", help="Environment context for validation")
    ] = None,

    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose validation output")
    ] = False,
):
    """
    Validate configuration files for syntax and completeness.

    This command validates rule configurations, alert configurations,
    and other configuration files without running an actual audit.
    """

    if not config_file.exists():
        typer.echo(f"❌ Configuration file not found: {config_file}", err=True)
        raise typer.Exit(code=ExitCode.CONFIG_ERROR.value)

    try:
        # Attempt to load and validate the configuration
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        elif config_file.suffix.lower() == '.json':
            import json
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        else:
            typer.echo(f"❌ Unsupported configuration file format: {config_file.suffix}", err=True)
            raise typer.Exit(code=ExitCode.CONFIG_ERROR.value)

        # Basic validation
        if not isinstance(config_data, dict):
            typer.echo(f"❌ Configuration file must contain a dictionary/object", err=True)
            raise typer.Exit(code=ExitCode.CONFIG_ERROR.value)

        # TODO: Add more sophisticated validation based on configuration type
        # For now, just validate that it's parseable

        typer.echo(f"✅ Configuration file {config_file} is valid")

        if verbose:
            typer.echo(f"   Environment: {environment or 'default'}")
            typer.echo(f"   Format: {config_file.suffix}")
            typer.echo(f"   Size: {config_file.stat().st_size} bytes")

    except Exception as e:
        typer.echo(f"❌ Configuration validation failed: {e}", err=True)
        raise typer.Exit(code=ExitCode.CONFIG_ERROR.value)


def cli_main():
    """Entry point for console script."""
    app()


if __name__ == "__main__":
    app()