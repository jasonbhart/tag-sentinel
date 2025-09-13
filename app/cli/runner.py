"""CLI runner for Tag Sentinel with rule evaluation and exit code mapping.

This module provides the main CLI interface for running audits with rule
evaluation, mapping rule severities to CI/CD-friendly exit codes, and
generating reports suitable for automated workflows.
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ..audit.capture.engine import create_capture_engine
from ..audit.rules import (
    # Rule evaluation
    evaluate_from_config,
    load_rules_from_file,
    
    # Models
    RuleResults,
    Severity,
    
    # Alert dispatching
    dispatcher_registry,
    AlertContext,
    AlertTrigger,
)
from ..audit.rules.indexing import build_audit_indexes


class ExitCode(IntEnum):
    """CLI exit codes for CI/CD integration.
    
    Maps rule evaluation results to standard exit codes that can be
    used in CI/CD pipelines for automated quality gates.
    """
    SUCCESS = 0           # All rules passed
    RULE_FAILURES = 1     # Some rules failed (info/warning)
    CRITICAL_FAILURES = 2 # Critical rule failures found
    CONFIG_ERROR = 3      # Configuration or setup error
    RUNTIME_ERROR = 4     # Runtime error during execution
    TIMEOUT_ERROR = 5     # Operation timed out


@dataclass
class CLIConfig:
    """Configuration for CLI execution."""
    
    # Input configuration
    urls: List[str]
    rules_config_path: Optional[Path] = None
    
    # Execution options
    headless: bool = True
    timeout_seconds: float = 30.0
    max_concurrency: int = 3
    
    # Rule evaluation options
    environment: Optional[str] = None
    fail_fast: bool = False
    severity_threshold: Severity = Severity.INFO
    
    # Output options
    output_format: str = "json"  # json, yaml, text
    output_file: Optional[Path] = None
    verbose: bool = False
    quiet: bool = False
    
    # Alert options
    enable_alerts: bool = False
    alert_config_path: Optional[Path] = None
    
    # CI/CD options
    exit_on_critical: bool = True
    exit_on_warnings: bool = False


class RuleEvaluationConfig(BaseModel):
    """Pydantic model for rule evaluation configuration."""
    
    # Rule evaluation settings
    environment: Optional[str] = Field(
        default=None,
        description="Environment context for rule evaluation"
    )
    fail_fast: bool = Field(
        default=False,
        description="Stop evaluation on first critical failure"
    )
    severity_threshold: Severity = Field(
        default=Severity.INFO,
        description="Minimum severity level to report"
    )
    
    # Alert settings
    enable_alerts: bool = Field(
        default=False,
        description="Enable alert dispatching"
    )
    alert_trigger: AlertTrigger = Field(
        default=AlertTrigger.ANY_FAILURE,
        description="Condition that triggers alerts"
    )
    
    # Execution settings
    parallel_evaluation: bool = Field(
        default=True,
        description="Enable parallel rule evaluation"
    )
    max_workers: int = Field(
        default=4,
        description="Maximum number of worker threads"
    )


class CLIRunner:
    """Main CLI runner for Tag Sentinel audits with rule evaluation."""
    
    def __init__(self, config: CLIConfig):
        self.config = config
        self.start_time: Optional[datetime] = None
        self.results: List[Dict[str, Any]] = []
    
    async def run(self) -> ExitCode:
        """Run the complete audit and rule evaluation process.
        
        Returns:
            Exit code based on rule evaluation results
        """
        try:
            self.start_time = datetime.utcnow()
            
            if not self.config.quiet:
                self._print_header()
            
            # Load rules configuration
            if not self.config.rules_config_path:
                if not self.config.quiet:
                    print("⚠️  No rules configuration provided - running capture only")
                return await self._run_capture_only()
            
            # Load and validate rules
            try:
                rules = load_rules_from_file(str(self.config.rules_config_path))
                if not self.config.quiet:
                    print(f"📋 Loaded {len(rules)} rules from {self.config.rules_config_path}")
            except Exception as e:
                self._print_error(f"Failed to load rules configuration: {e}")
                return ExitCode.CONFIG_ERROR
            
            # Run capture and evaluation
            return await self._run_with_rules(rules)
            
        except KeyboardInterrupt:
            self._print_error("Operation interrupted by user")
            return ExitCode.RUNTIME_ERROR
        except Exception as e:
            self._print_error(f"Runtime error: {e}")
            if self.config.verbose:
                import traceback
                traceback.print_exc()
            return ExitCode.RUNTIME_ERROR
    
    async def _run_capture_only(self) -> ExitCode:
        """Run capture without rule evaluation."""
        try:
            engine = create_capture_engine(headless=self.config.headless)
            
            async with engine.session():
                for i, url in enumerate(self.config.urls, 1):
                    if not self.config.quiet:
                        print(f"🔍 Capturing [{i}/{len(self.config.urls)}]: {url}")
                    
                    try:
                        result = await engine.capture_page(
                            url,
                            timeout=int(self.config.timeout_seconds * 1000)
                        )
                        
                        summary = {
                            "url": result.url,
                            "status": str(result.capture_status),
                            "requests": len(result.network_requests or []),
                            "console_logs": len(result.console_logs or []),
                            "cookies": len(result.cookies or []),
                            "timestamp": result.timestamp.isoformat() if result.timestamp else None
                        }
                        
                        self.results.append(summary)
                        
                        if self.config.verbose:
                            print(f"   ✅ Status: {result.capture_status}")
                            print(f"   📡 Requests: {summary['requests']}")
                            print(f"   🍪 Cookies: {summary['cookies']}")
                        
                    except Exception as e:
                        error_summary = {
                            "url": url,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        self.results.append(error_summary)
                        
                        if not self.config.quiet:
                            print(f"   ❌ Error: {e}")
            
            # Output results
            await self._output_results({"captures": self.results})
            
            if not self.config.quiet:
                self._print_summary(len(self.config.urls), 0, None)
            
            return ExitCode.SUCCESS
            
        except Exception as e:
            self._print_error(f"Capture failed: {e}")
            return ExitCode.RUNTIME_ERROR
    
    async def _run_with_rules(self, rules: List[Any]) -> ExitCode:
        """Run capture with rule evaluation.
        
        Args:
            rules: Loaded rule configurations
            
        Returns:
            Exit code based on rule evaluation results
        """
        try:
            # Create capture engine
            engine = create_capture_engine(headless=self.config.headless)
            
            # Capture all pages
            page_results = []
            async with engine.session():
                for i, url in enumerate(self.config.urls, 1):
                    if not self.config.quiet:
                        print(f"🔍 Capturing [{i}/{len(self.config.urls)}]: {url}")
                    
                    try:
                        result = await engine.capture_page(
                            url,
                            timeout=int(self.config.timeout_seconds * 1000)
                        )
                        page_results.append(result)
                        
                        if self.config.verbose:
                            print(f"   ✅ Status: {result.capture_status}")
                            print(f"   📡 Requests: {len(result.network_requests or [])}")
                        
                    except Exception as e:
                        if not self.config.quiet:
                            print(f"   ❌ Capture error: {e}")
                        continue
            
            if not page_results:
                self._print_error("No successful page captures")
                return ExitCode.RUNTIME_ERROR
            
            # Build audit indexes from captured data
            if not self.config.quiet:
                print(f"📊 Building audit indexes from {len(page_results)} page(s)...")
            
            # Build audit indexes from page results
            indexes = build_audit_indexes(page_results)
            
            # Evaluate rules
            if not self.config.quiet:
                print(f"⚖️  Evaluating {len(rules)} rules...")
            
            rule_evaluation_config = RuleEvaluationConfig(
                environment=self.config.environment,
                fail_fast=self.config.fail_fast,
                severity_threshold=self.config.severity_threshold,
                enable_alerts=self.config.enable_alerts
            )
            
            rule_results = evaluate_from_config(
                rules=rules,
                indexes=indexes,
                config=rule_evaluation_config.model_dump(),
                environment=self.config.environment
            )
            
            # Process results
            exit_code = self._determine_exit_code(rule_results)
            
            # Handle alerts if enabled
            if self.config.enable_alerts and rule_results.summary.failed_rules > 0:
                await self._dispatch_alerts(rule_results, page_results)
            
            # Output results
            await self._output_results({
                "captures": [self._summarize_page_result(r) for r in page_results],
                "rule_evaluation": self._summarize_rule_results(rule_results),
                "exit_code": exit_code.value
            })
            
            # Print summary
            if not self.config.quiet:
                self._print_rule_summary(rule_results)
                self._print_summary(len(page_results), len(rules), rule_results)
            
            return exit_code
            
        except Exception as e:
            self._print_error(f"Rule evaluation failed: {e}")
            if self.config.verbose:
                import traceback
                traceback.print_exc()
            return ExitCode.RUNTIME_ERROR
    
    def _convert_page_results_to_audit_data(self, page_results: List[Any]) -> Dict[str, Any]:
        """Convert page results to audit data format expected by indexing."""
        # This is a placeholder - would need to be implemented based on
        # the actual structure expected by build_audit_indexes
        return {
            "pages": page_results,
            "run_info": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "environment": self.config.environment,
                "total_pages": len(page_results)
            }
        }
    
    def _determine_exit_code(self, rule_results: RuleResults) -> ExitCode:
        """Determine appropriate exit code based on rule results.
        
        Args:
            rule_results: Results from rule evaluation
            
        Returns:
            Appropriate exit code for CI/CD integration
        """
        if rule_results.summary.critical_failures > 0:
            return ExitCode.CRITICAL_FAILURES
        
        if self.config.exit_on_warnings and rule_results.summary.warning_failures > 0:
            return ExitCode.RULE_FAILURES
        
        if rule_results.summary.failed_rules > 0:
            return ExitCode.RULE_FAILURES
        
        return ExitCode.SUCCESS
    
    async def _dispatch_alerts(self, rule_results: RuleResults, page_results: List[Any]):
        """Dispatch alerts for rule failures.
        
        Args:
            rule_results: Results from rule evaluation
            page_results: Captured page results
        """
        try:
            if not self.config.alert_config_path:
                return
            
            # Load alert configuration
            import yaml
            with open(self.config.alert_config_path, 'r') as f:
                alert_config = yaml.safe_load(f)
            
            # Create alert context
            context = AlertContext(
                rule_results=rule_results,
                evaluation_results=[],  # Would need actual evaluation results
                alert_config=alert_config,
                trigger_condition=AlertTrigger.ANY_FAILURE,
                environment=self.config.environment,
                target_urls=self.config.urls
            )
            
            # Dispatch alerts through configured dispatchers
            for dispatcher_config in alert_config.get('dispatchers', []):
                dispatcher_type = dispatcher_config.get('type')
                if dispatcher_type:
                    try:
                        dispatcher = dispatcher_registry.create_dispatcher(
                            dispatcher_type,
                            dispatcher_config
                        )
                        result = dispatcher.dispatch(context)
                        
                        if self.config.verbose:
                            print(f"   📤 Alert dispatched via {dispatcher_type}: {result.success}")
                    
                    except Exception as e:
                        if not self.config.quiet:
                            print(f"   ❌ Alert dispatch failed for {dispatcher_type}: {e}")
        
        except Exception as e:
            if not self.config.quiet:
                print(f"⚠️  Alert processing failed: {e}")
    
    def _summarize_page_result(self, result: Any) -> Dict[str, Any]:
        """Create summary of page capture result."""
        return {
            "url": getattr(result, 'url', 'unknown'),
            "status": str(getattr(result, 'capture_status', 'unknown')),
            "requests": len(getattr(result, 'network_requests', [])),
            "console_logs": len(getattr(result, 'console_logs', [])),
            "cookies": len(getattr(result, 'cookies', [])),
            "timestamp": getattr(result, 'timestamp', datetime.utcnow()).isoformat()
        }
    
    def _summarize_rule_results(self, rule_results: RuleResults) -> Dict[str, Any]:
        """Create summary of rule evaluation results."""
        return {
            "summary": {
                "total_rules": rule_results.summary.total_rules,
                "passed_rules": rule_results.summary.passed_rules,
                "failed_rules": rule_results.summary.failed_rules,
                "critical_failures": rule_results.summary.critical_failures,
                "warning_failures": rule_results.summary.warning_failures,
                "info_failures": rule_results.summary.info_failures,
                "execution_time_ms": rule_results.summary.execution_time_ms
            },
            "failures": [
                {
                    "check_id": f.check_id,
                    "severity": f.severity.value,
                    "message": f.message,
                    "evidence_count": len(f.evidence) if f.evidence else 0
                }
                for f in rule_results.failures[:10]  # Limit to first 10 for CLI output
            ],
            "evaluation_time": rule_results.evaluation_time.isoformat()
        }
    
    async def _output_results(self, data: Dict[str, Any]):
        """Output results in specified format.
        
        Args:
            data: Results data to output
        """
        if self.config.output_format.lower() == "json":
            output = json.dumps(data, indent=2, default=str)
        elif self.config.output_format.lower() == "yaml":
            import yaml
            output = yaml.dump(data, default_flow_style=False)
        else:
            # Text format
            output = self._format_text_output(data)
        
        if self.config.output_file:
            self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.config.output_file.write_text(output)
            
            if not self.config.quiet:
                print(f"📄 Results written to {self.config.output_file}")
        elif not self.config.quiet:
            print("\n" + output)
    
    def _format_text_output(self, data: Dict[str, Any]) -> str:
        """Format results as human-readable text."""
        lines = []
        
        # Capture summary
        if "captures" in data:
            lines.append("🔍 CAPTURE SUMMARY")
            lines.append("=" * 50)
            for capture in data["captures"]:
                lines.append(f"URL: {capture['url']}")
                lines.append(f"Status: {capture['status']}")
                lines.append(f"Requests: {capture['requests']}")
                lines.append(f"Cookies: {capture['cookies']}")
                lines.append("")
        
        # Rule evaluation summary
        if "rule_evaluation" in data:
            eval_data = data["rule_evaluation"]
            summary = eval_data["summary"]
            
            lines.append("⚖️  RULE EVALUATION SUMMARY")
            lines.append("=" * 50)
            lines.append(f"Total Rules: {summary['total_rules']}")
            lines.append(f"Passed: {summary['passed_rules']}")
            lines.append(f"Failed: {summary['failed_rules']}")
            lines.append(f"Critical Failures: {summary['critical_failures']}")
            lines.append(f"Warning Failures: {summary['warning_failures']}")
            lines.append(f"Execution Time: {summary['execution_time_ms']}ms")
            lines.append("")
            
            # Show failures
            if eval_data["failures"]:
                lines.append("❌ FAILURES")
                lines.append("-" * 20)
                for i, failure in enumerate(eval_data["failures"], 1):
                    lines.append(f"{i}. {failure['check_id']} ({failure['severity']})")
                    lines.append(f"   {failure['message']}")
                    if failure['evidence_count']:
                        lines.append(f"   Evidence: {failure['evidence_count']} items")
                    lines.append("")
        
        return "\n".join(lines)
    
    def _print_header(self):
        """Print CLI header."""
        print("🛡️  Tag Sentinel - Web Analytics Auditing")
        print("=" * 50)
    
    def _print_error(self, message: str):
        """Print error message to stderr."""
        if not self.config.quiet:
            print(f"❌ ERROR: {message}", file=sys.stderr)
    
    def _print_rule_summary(self, rule_results: RuleResults):
        """Print rule evaluation summary."""
        summary = rule_results.summary
        
        print(f"\n⚖️  RULE EVALUATION COMPLETED")
        print(f"   Total Rules: {summary.total_rules}")
        print(f"   ✅ Passed: {summary.passed_rules}")
        print(f"   ❌ Failed: {summary.failed_rules}")
        
        if summary.critical_failures > 0:
            print(f"   🚨 Critical: {summary.critical_failures}")
        if summary.warning_failures > 0:
            print(f"   ⚠️  Warning: {summary.warning_failures}")
        if summary.info_failures > 0:
            print(f"   ℹ️  Info: {summary.info_failures}")
        
        print(f"   ⏱️  Execution: {summary.execution_time_ms}ms")
    
    def _print_summary(self, pages_captured: int, rules_evaluated: int, rule_results: Optional[RuleResults]):
        """Print final summary."""
        elapsed = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        
        print(f"\n🏁 AUDIT COMPLETED")
        print(f"   Pages Captured: {pages_captured}")
        if rules_evaluated > 0:
            print(f"   Rules Evaluated: {rules_evaluated}")
            if rule_results:
                if rule_results.summary.failed_rules == 0:
                    print(f"   Result: ✅ All rules passed")
                elif rule_results.summary.critical_failures > 0:
                    print(f"   Result: 🚨 Critical failures detected")
                else:
                    print(f"   Result: ⚠️  Some rules failed")
        print(f"   Total Time: {elapsed:.1f}s")


def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="tag-sentinel",
        description="Web Analytics Auditing and Monitoring Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture a single page
  tag-sentinel https://example.com
  
  # Run with rules
  tag-sentinel --rules config/rules.yaml https://example.com
  
  # Multiple URLs with JSON output
  tag-sentinel --rules config/rules.yaml --output results.json \\
    https://example.com https://staging.example.com
  
  # CI/CD integration (exits with appropriate codes)
  tag-sentinel --rules config/rules.yaml --exit-on-warnings \\
    --environment production https://example.com
        """
    )
    
    # URLs (positional)
    parser.add_argument(
        "urls",
        nargs="+",
        help="URLs to audit"
    )
    
    # Rules configuration
    parser.add_argument(
        "--rules", "-r",
        type=Path,
        help="Path to rules configuration file (YAML)"
    )
    
    # Environment
    parser.add_argument(
        "--environment", "-e",
        help="Environment context (dev, staging, prod)"
    )
    
    # Browser options
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode (default)"
    )
    
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser with GUI (for debugging)"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Page load timeout in seconds (default: 30)"
    )
    
    # Rule evaluation options
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first critical failure"
    )
    
    parser.add_argument(
        "--severity-threshold",
        choices=["info", "warning", "critical"],
        default="info",
        help="Minimum severity level to report (default: info)"
    )
    
    # CI/CD options
    parser.add_argument(
        "--exit-on-critical",
        action="store_true",
        default=True,
        help="Exit with code 2 on critical failures (default)"
    )
    
    parser.add_argument(
        "--exit-on-warnings",
        action="store_true",
        help="Exit with code 1 on warning failures"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "yaml", "text"],
        default="json",
        help="Output format (default: json)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (minimal output)"
    )
    
    # Alert options
    parser.add_argument(
        "--enable-alerts",
        action="store_true",
        help="Enable alert dispatching"
    )
    
    parser.add_argument(
        "--alert-config",
        type=Path,
        help="Path to alert configuration file"
    )
    
    return parser


def parse_cli_args(args: Optional[List[str]] = None) -> CLIConfig:
    """Parse command line arguments into configuration.
    
    Args:
        args: Arguments to parse (None for sys.argv)
        
    Returns:
        Parsed CLI configuration
    """
    parser = create_cli_parser()
    parsed_args = parser.parse_args(args)
    
    # Convert severity string to enum
    severity_map = {
        "info": Severity.INFO,
        "warning": Severity.WARNING,
        "critical": Severity.CRITICAL
    }
    
    return CLIConfig(
        urls=parsed_args.urls,
        rules_config_path=parsed_args.rules,
        headless=not parsed_args.headed if hasattr(parsed_args, 'headed') else parsed_args.headless,
        timeout_seconds=parsed_args.timeout,
        environment=parsed_args.environment,
        fail_fast=parsed_args.fail_fast,
        severity_threshold=severity_map[parsed_args.severity_threshold],
        output_format=parsed_args.format,
        output_file=parsed_args.output,
        verbose=parsed_args.verbose,
        quiet=parsed_args.quiet,
        enable_alerts=parsed_args.enable_alerts,
        alert_config_path=parsed_args.alert_config,
        exit_on_critical=parsed_args.exit_on_critical,
        exit_on_warnings=parsed_args.exit_on_warnings,
    )


async def run_audit_with_rules(
    urls: List[str],
    rules_config_path: Optional[str] = None,
    environment: Optional[str] = None,
    **kwargs
) -> ExitCode:
    """Run audit with rule evaluation (programmatic interface).
    
    Args:
        urls: URLs to audit
        rules_config_path: Path to rules configuration file
        environment: Environment context
        **kwargs: Additional configuration options
        
    Returns:
        Exit code based on evaluation results
    """
    config = CLIConfig(
        urls=urls,
        rules_config_path=Path(rules_config_path) if rules_config_path else None,
        environment=environment,
        **kwargs
    )
    
    runner = CLIRunner(config)
    return await runner.run()


def main():
    """Main CLI entry point."""
    try:
        config = parse_cli_args()
        runner = CLIRunner(config)
        exit_code = asyncio.run(runner.run())
        sys.exit(exit_code.value)
    
    except KeyboardInterrupt:
        print("❌ Interrupted by user", file=sys.stderr)
        sys.exit(ExitCode.RUNTIME_ERROR.value)
    except Exception as e:
        print(f"❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(ExitCode.RUNTIME_ERROR.value)


def map_severity_to_exit_code(severity: Optional[Severity]) -> ExitCode:
    """Map rule failure severity to appropriate exit code.
    
    Args:
        severity: Highest severity from rule failures (None if no failures)
        
    Returns:
        Appropriate exit code for CI/CD integration
    """
    if severity is None:
        return ExitCode.SUCCESS
    
    if severity == Severity.CRITICAL:
        return ExitCode.CRITICAL_FAILURES
    
    # Warning and info both map to general rule failures
    return ExitCode.RULE_FAILURES


def load_audit_data():
    """Mock function for loading audit data - will be patched in tests."""
    raise NotImplementedError("This function should be mocked in tests")


def evaluate_rules_for_cli(args) -> tuple[ExitCode, RuleResults]:
    """Evaluate rules for CLI integration testing.
    
    Args:
        args: Mock CLI arguments object
        
    Returns:
        Tuple of (exit_code, rule_results)
    """
    # This is a simplified version for integration testing
    try:
        # Load rules
        from app.audit.rules.parser import RuleParser
        parser = RuleParser()
        rules = parser.parse_yaml_file(Path(args.rules_config))
        
        # Load audit data (would be provided by mock)
        audit_data = load_audit_data()  # Mock function
        
        # Build indexes - ensure audit_data is a list of PageResult objects
        from app.audit.rules.indexing import AuditIndexes, AuditQuery, build_audit_indexes
        if isinstance(audit_data, list):
            indexes = build_audit_indexes(audit_data)
        else:
            indexes = build_audit_indexes([audit_data])
        
        # Create evaluation context
        from app.audit.rules.evaluator import EvaluationContext, RuleEvaluationEngine
        context = EvaluationContext(
            indexes=indexes,
            query=AuditQuery(indexes),
            environment=args.environment,
            target_urls=args.target_urls,
            debug=args.debug,
            fail_fast=args.fail_fast,
            max_workers=args.max_workers
        )
        
        # Evaluate rules
        engine = RuleEvaluationEngine()
        results = engine.evaluate_rules(rules, context)
        
        # Determine exit code
        highest_severity = None
        if results.failures:
            severities = [f.severity for f in results.failures]
            if Severity.CRITICAL in severities:
                highest_severity = Severity.CRITICAL
            elif Severity.WARNING in severities:
                highest_severity = Severity.WARNING
            elif Severity.INFO in severities:
                highest_severity = Severity.INFO
        
        exit_code = map_severity_to_exit_code(highest_severity)
        return exit_code, results
        
    except Exception as e:
        # For testing, re-raise to see the actual error
        raise e


def load_audit_data():
    """Mock function for loading audit data in tests."""
    # This would be mocked in tests
    raise NotImplementedError("This function should be mocked in tests")


if __name__ == "__main__":
    main()