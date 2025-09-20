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
from .summary import SummaryReporter


class ExitCode(IntEnum):
    """CLI exit codes for CI/CD integration.

    Maps rule evaluation results to standard exit codes that can be
    used in CI/CD pipelines for automated quality gates.
    """
    SUCCESS = 0           # All rules passed, all pages captured successfully
    RULE_FAILURES = 1     # Some rules failed (info/warning) or some pages failed to capture
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
    max_pages: int = 500

    # Rule evaluation options
    environment: Optional[str] = None
    scenario_id: Optional[str] = None
    fail_fast: bool = False
    severity_threshold: Severity = Severity.INFO

    # Output options
    output_format: str = "json"  # json, yaml, text
    output_file: Optional[Path] = None
    verbose: bool = False
    quiet: bool = False

    # Debug options
    devtools: bool = False
    har: bool = False
    har_omit_content: bool = False
    screenshots: Optional[str] = None
    trace: bool = False

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

    def __init__(self, config: CLIConfig, progress_tracker=None, real_time_output=None):
        self.config = config
        self.start_time: Optional[datetime] = None
        self.results: List[Dict[str, Any]] = []
        self.progress_tracker = progress_tracker
        self.real_time_output = real_time_output

        # Create enhanced summary reporter
        # Use JSON format if --json flag was used, otherwise use text
        summary_format = "json" if config.output_format == "json" and config.output_file is None else "text"
        self.summary_reporter = SummaryReporter(
            output_format=summary_format,
            verbose=config.verbose,
            quiet=config.quiet
        )
    
    async def run(self) -> ExitCode:
        """Run the complete audit and rule evaluation process.

        Returns:
            Exit code based on rule evaluation results
        """
        try:
            self.start_time = datetime.utcnow()

            # Initialize summary reporter
            self.summary_reporter.set_start_time(self.start_time)
            self.summary_reporter.set_environment(self.config.environment)
            self.summary_reporter.set_target_urls(self.config.urls)

            # Start initialization progress
            if self.progress_tracker:
                self.progress_tracker.start_step("initialization")

            if not self.config.quiet:
                self._print_header()

            # Load rules configuration
            if not self.config.rules_config_path:
                if not self.config.quiet:
                    print("⚠️  No rules configuration provided - running capture only")

                if self.progress_tracker:
                    self.progress_tracker.complete_step("initialization", {"mode": "capture_only"})

                return await self._run_capture_only()

            # Load and validate rules
            try:
                rules = load_rules_from_file(str(self.config.rules_config_path))
                if not self.config.quiet:
                    print(f"📋 Loaded {len(rules)} rules from {self.config.rules_config_path}")

                if self.progress_tracker:
                    self.progress_tracker.complete_step("initialization", {
                        "mode": "with_rules",
                        "rule_count": len(rules)
                    })

            except Exception as e:
                self._print_error(f"Failed to load rules configuration: {e}")
                if self.progress_tracker:
                    self.progress_tracker.fail_step("initialization", str(e))
                return ExitCode.CONFIG_ERROR

            # Run capture and evaluation
            return await self._run_with_rules(rules)

        except KeyboardInterrupt:
            self._print_error("Operation interrupted by user")
            if self.progress_tracker:
                self.progress_tracker.fail_step("audit", "Interrupted by user")
            return ExitCode.RUNTIME_ERROR
        except Exception as e:
            self._print_error(f"Runtime error: {e}")
            if self.progress_tracker:
                self.progress_tracker.fail_step("audit", str(e))
            if self.config.verbose:
                import traceback
                traceback.print_exc()
            return ExitCode.RUNTIME_ERROR
    
    async def _run_capture_only(self) -> ExitCode:
        """Run capture without rule evaluation."""
        try:
            # Start page capture progress
            if self.progress_tracker:
                self.progress_tracker.start_step("page_capture")

            # Warn about unsupported har_omit_content feature
            if self.config.har and self.config.har_omit_content and not self.config.quiet:
                print("⚠️  --har-omit-content is not yet implemented and will be ignored")

            engine = create_capture_engine(
                headless=self.config.headless,
                devtools=self.config.devtools,
                max_concurrent_pages=self.config.max_concurrency,
                enable_artifacts=any([
                    self.config.har,
                    self.config.screenshots,
                    self.config.trace
                ]),
                artifacts_dir=self.config.output_file.parent if self.config.output_file else Path("."),
                enable_har=self.config.har,
                take_screenshots=(self.config.screenshots == "on_all"),
                screenshot_on_error=(self.config.screenshots in ["on_error", "on_fail"]),
                enable_trace=self.config.trace
                # TODO: Add har_omit_content support when Playwright recording allows content filtering
            )

            async with engine.session():
                successful_captures = 0
                failed_captures = 0

                # Respect max_pages limit
                urls_to_process = self.config.urls[:self.config.max_pages]

                for i, url in enumerate(urls_to_process, 1):
                    if not self.config.quiet:
                        print(f"🔍 Capturing [{i}/{len(urls_to_process)}]: {url}")

                    # Emit real-time event
                    if self.real_time_output:
                        self.real_time_output.emit_event("page_capture_start", {"url": url, "index": i, "total": len(urls_to_process)})

                    capture_start = time.time()

                    try:
                        result = await engine.capture_page(
                            url,
                            session_config_overrides={
                                'wait_timeout_ms': int(self.config.timeout_seconds * 1000)
                            }
                        )

                        capture_duration = int((time.time() - capture_start) * 1000)

                        summary = {
                            "url": result.url,
                            "status": str(result.capture_status),
                            "requests": len(result.network_requests or []),
                            "console_logs": len(result.console_logs or []),
                            "cookies": len(result.cookies or []),
                            "timestamp": result.capture_time.isoformat() if result.capture_time else None
                        }

                        self.results.append(summary)
                        successful_captures += 1

                        # Update progress
                        if self.progress_tracker:
                            progress = i / len(urls_to_process)
                            self.progress_tracker.update_step("page_capture", progress, {
                                "url": url,
                                "successful": successful_captures,
                                "failed": failed_captures,
                                "total": len(urls_to_process)
                            })

                        # Emit completion event
                        if self.real_time_output:
                            self.real_time_output.emit_event("page_capture_complete", {
                                "url": url,
                                "success": True,
                                "duration_ms": capture_duration,
                                "requests": summary["requests"],
                                "cookies": summary["cookies"]
                            })

                        if self.config.verbose:
                            print(f"   ✅ Status: {result.capture_status}")
                            print(f"   📡 Requests: {summary['requests']}")
                            print(f"   🍪 Cookies: {summary['cookies']}")

                    except Exception as e:
                        capture_duration = int((time.time() - capture_start) * 1000)
                        failed_captures += 1

                        error_summary = {
                            "url": url,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        self.results.append(error_summary)

                        # Update progress
                        if self.progress_tracker:
                            progress = i / len(urls_to_process)
                            self.progress_tracker.update_step("page_capture", progress, {
                                "url": url,
                                "successful": successful_captures,
                                "failed": failed_captures,
                                "total": len(urls_to_process)
                            })

                        # Emit failure event
                        if self.real_time_output:
                            self.real_time_output.emit_event("page_capture_complete", {
                                "url": url,
                                "success": False,
                                "duration_ms": capture_duration,
                                "error": str(e)
                            })

                        if not self.config.quiet:
                            print(f"   ❌ Error: {e}")

            # Complete page capture step
            if self.progress_tracker:
                self.progress_tracker.complete_step("page_capture", {
                    "total": len(urls_to_process),
                    "successful": successful_captures,
                    "failed": failed_captures
                })

            # Start output generation
            if self.progress_tracker:
                self.progress_tracker.start_step("output_generation")

            # Output results
            await self._output_results({"captures": self.results})

            if self.progress_tracker:
                self.progress_tracker.complete_step("output_generation")

            # Update summary reporter and print enhanced summary
            self.summary_reporter.set_end_time(datetime.utcnow())
            self.summary_reporter.set_page_capture_results(
                total=len(urls_to_process),
                successful=successful_captures,
                failed=failed_captures
            )
            self.summary_reporter.set_rule_evaluation_results(0, None)

            if self.progress_tracker:
                self.summary_reporter.set_progress_summary(self.progress_tracker.summary)

            # Add output file if configured
            if self.config.output_file:
                self.summary_reporter.add_output_file(self.config.output_file)

            # Determine exit code based on capture results
            exit_code = ExitCode.RULE_FAILURES if failed_captures > 0 else ExitCode.SUCCESS

            # Add exit code information to summary
            self._add_exit_code_to_summary(exit_code)
            self.summary_reporter.set_exit_code(exit_code.value)

            # Print enhanced summary
            if not self.config.quiet:
                self.summary_reporter.print_summary()

            return exit_code

        except Exception as e:
            self._print_error(f"Capture failed: {e}")
            if self.progress_tracker:
                # Determine which step failed
                current_step = self.progress_tracker.current_step
                if current_step:
                    self.progress_tracker.fail_step(current_step.name, str(e))
            return ExitCode.RUNTIME_ERROR
    
    async def _run_with_rules(self, rules: List[Any]) -> ExitCode:
        """Run capture with rule evaluation.

        Args:
            rules: Loaded rule configurations

        Returns:
            Exit code based on rule evaluation results
        """
        try:
            # Start page capture progress
            if self.progress_tracker:
                self.progress_tracker.start_step("page_capture")

            # Create capture engine
            # Warn about unsupported har_omit_content feature
            if self.config.har and self.config.har_omit_content and not self.config.quiet:
                print("⚠️  --har-omit-content is not yet implemented and will be ignored")

            engine = create_capture_engine(
                headless=self.config.headless,
                devtools=self.config.devtools,
                max_concurrent_pages=self.config.max_concurrency,
                enable_artifacts=any([
                    self.config.har,
                    self.config.screenshots,
                    self.config.trace
                ]),
                artifacts_dir=self.config.output_file.parent if self.config.output_file else Path("."),
                enable_har=self.config.har,
                take_screenshots=(self.config.screenshots == "on_all"),
                screenshot_on_error=(self.config.screenshots in ["on_error", "on_fail"]),
                enable_trace=self.config.trace
                # TODO: Add har_omit_content support when Playwright recording allows content filtering
            )

            # Capture all pages
            page_results = []
            successful_captures = 0
            failed_captures = 0

            async with engine.session():
                # Respect max_pages limit
                urls_to_process = self.config.urls[:self.config.max_pages]

                for i, url in enumerate(urls_to_process, 1):
                    if not self.config.quiet:
                        print(f"🔍 Capturing [{i}/{len(urls_to_process)}]: {url}")

                    # Emit real-time event
                    if self.real_time_output:
                        self.real_time_output.emit_event("page_capture_start", {"url": url, "index": i, "total": len(urls_to_process)})

                    capture_start = time.time()

                    try:
                        result = await engine.capture_page(
                            url,
                            session_config_overrides={
                                'wait_timeout_ms': int(self.config.timeout_seconds * 1000)
                            }
                        )
                        page_results.append(result)
                        successful_captures += 1

                        capture_duration = int((time.time() - capture_start) * 1000)

                        # Update progress
                        if self.progress_tracker:
                            progress = i / len(urls_to_process)
                            self.progress_tracker.update_step("page_capture", progress, {
                                "url": url,
                                "successful": successful_captures,
                                "failed": failed_captures,
                                "total": len(urls_to_process)
                            })

                        # Emit completion event
                        if self.real_time_output:
                            self.real_time_output.emit_event("page_capture_complete", {
                                "url": url,
                                "success": True,
                                "duration_ms": capture_duration,
                                "requests": len(result.network_requests or []),
                                "cookies": len(result.cookies or [])
                            })

                        if self.config.verbose:
                            print(f"   ✅ Status: {result.capture_status}")
                            print(f"   📡 Requests: {len(result.network_requests or [])}")

                    except Exception as e:
                        failed_captures += 1
                        capture_duration = int((time.time() - capture_start) * 1000)

                        # Update progress
                        if self.progress_tracker:
                            progress = i / len(urls_to_process)
                            self.progress_tracker.update_step("page_capture", progress, {
                                "url": url,
                                "successful": successful_captures,
                                "failed": failed_captures,
                                "total": len(urls_to_process)
                            })

                        # Emit failure event
                        if self.real_time_output:
                            self.real_time_output.emit_event("page_capture_complete", {
                                "url": url,
                                "success": False,
                                "duration_ms": capture_duration,
                                "error": str(e)
                            })

                        if not self.config.quiet:
                            print(f"   ❌ Capture error: {e}")
                        continue

            # Complete page capture step
            if self.progress_tracker:
                self.progress_tracker.complete_step("page_capture", {
                    "total": len(urls_to_process),
                    "successful": successful_captures,
                    "failed": failed_captures
                })

            if not page_results:
                self._print_error("No successful page captures")
                return ExitCode.RUNTIME_ERROR

            # Build audit indexes from captured data
            if not self.config.quiet:
                print(f"📊 Building audit indexes from {len(page_results)} page(s)...")

            # Build audit indexes from page results
            indexes = build_audit_indexes(page_results)

            # Start rule evaluation progress
            if self.progress_tracker:
                self.progress_tracker.start_step("rule_evaluation")

            # Evaluate rules
            if not self.config.quiet:
                print(f"⚖️  Evaluating {len(rules)} rules...")

            # Emit rule evaluation start event
            if self.real_time_output:
                self.real_time_output.emit_event("rule_evaluation_start", {
                    "rule_count": len(rules),
                    "page_count": len(page_results)
                })

            evaluation_start = time.time()

            # Map CLI configuration to evaluation context parameters
            evaluation_config = {
                'fail_fast': self.config.fail_fast,
                'timeout_seconds': int(self.config.timeout_seconds),
                'target_urls': self.config.urls,
                'debug': self.config.verbose,
            }

            # Add scenario_id if specified
            if self.config.scenario_id:
                evaluation_config['scenario_id'] = self.config.scenario_id

            # Map severity threshold to severity filter set
            if self.config.severity_threshold:
                severity_levels = {
                    Severity.INFO: {Severity.INFO, Severity.WARNING, Severity.CRITICAL},
                    Severity.WARNING: {Severity.WARNING, Severity.CRITICAL},
                    Severity.CRITICAL: {Severity.CRITICAL}
                }
                evaluation_config['severity_filter'] = severity_levels.get(
                    self.config.severity_threshold,
                    {Severity.INFO, Severity.WARNING, Severity.CRITICAL}
                )

            rule_results = evaluate_from_config(
                rules=rules,
                indexes=indexes,
                config=evaluation_config,
                environment=self.config.environment
            )

            evaluation_duration = int((time.time() - evaluation_start) * 1000)

            # Complete rule evaluation step
            if self.progress_tracker:
                self.progress_tracker.complete_step("rule_evaluation", {
                    "total_rules": rule_results.summary.total_rules,
                    "passed": rule_results.summary.passed_rules,
                    "failed": rule_results.summary.failed_rules,
                    "critical": rule_results.summary.critical_failures,
                    "duration_ms": evaluation_duration
                })

            # Emit rule evaluation completion event
            if self.real_time_output:
                self.real_time_output.emit_event("rule_evaluation_complete", {
                    "passed": rule_results.summary.passed_rules,
                    "failed": rule_results.summary.failed_rules,
                    "critical": rule_results.summary.critical_failures,
                    "duration_ms": evaluation_duration
                })

            # Process results
            exit_code = self._determine_exit_code(rule_results, failed_captures)

            # Handle alerts if enabled
            if self.config.enable_alerts and rule_results.summary.failed_rules > 0:
                if self.progress_tracker:
                    self.progress_tracker.start_step("alert_dispatch")

                await self._dispatch_alerts(rule_results, page_results)

                if self.progress_tracker:
                    self.progress_tracker.complete_step("alert_dispatch")

            # Start output generation
            if self.progress_tracker:
                self.progress_tracker.start_step("output_generation")

            # Output results
            await self._output_results({
                "captures": [self._summarize_page_result(r) for r in page_results],
                "rule_evaluation": self._summarize_rule_results(rule_results),
                "exit_code": exit_code.value
            })

            if self.progress_tracker:
                self.progress_tracker.complete_step("output_generation")

            # Update summary reporter
            self.summary_reporter.set_end_time(datetime.utcnow())
            self.summary_reporter.set_page_capture_results(
                total=len(urls_to_process),
                successful=successful_captures,
                failed=failed_captures
            )
            self.summary_reporter.set_rule_evaluation_results(len(rules), rule_results)

            if self.config.enable_alerts:
                # Count alert dispatches (this would need to be tracked during dispatch)
                self.summary_reporter.set_alert_results(0, 0)  # TODO: Track actual alert counts

            if self.progress_tracker:
                self.summary_reporter.set_progress_summary(self.progress_tracker.summary)

            # Add output file if configured
            if self.config.output_file:
                self.summary_reporter.add_output_file(self.config.output_file)

            # Add exit code information to summary
            self._add_exit_code_to_summary(exit_code)
            self.summary_reporter.set_exit_code(exit_code.value)

            # Print enhanced summary
            if not self.config.quiet:
                self.summary_reporter.print_summary()

            return exit_code

        except Exception as e:
            self._print_error(f"Rule evaluation failed: {e}")
            if self.progress_tracker:
                # Determine which step failed
                current_step = self.progress_tracker.current_step
                if current_step:
                    self.progress_tracker.fail_step(current_step.name, str(e))
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
    
    def _determine_exit_code(self, rule_results: RuleResults, failed_captures: int = 0) -> ExitCode:
        """Determine appropriate exit code based on rule results and capture failures.

        Args:
            rule_results: Results from rule evaluation
            failed_captures: Number of failed page captures

        Returns:
            Appropriate exit code for CI/CD integration
        """
        # Critical rule failures (only if exit_on_critical is enabled)
        if rule_results.summary.critical_failures > 0 and self.config.exit_on_critical:
            return ExitCode.CRITICAL_FAILURES

        # Page capture failures or rule failures
        if failed_captures > 0:
            return ExitCode.RULE_FAILURES

        # Warning failures (if configured to exit on warnings)
        if self.config.exit_on_warnings and rule_results.summary.warning_failures > 0:
            return ExitCode.RULE_FAILURES

        # Any rule failures (including critical failures when exit_on_critical is disabled)
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
            successful_dispatches = 0
            failed_dispatches = 0

            for dispatcher_config in alert_config.get('dispatchers', []):
                dispatcher_type = dispatcher_config.get('type')
                if dispatcher_type:
                    try:
                        dispatcher = dispatcher_registry.create_dispatcher(
                            dispatcher_type,
                            dispatcher_config
                        )
                        result = await dispatcher.dispatch(context)

                        if result.success:
                            successful_dispatches += 1
                        else:
                            failed_dispatches += 1

                        # Emit real-time event
                        if self.real_time_output:
                            self.real_time_output.emit_event("alert_dispatched", {
                                "dispatcher": dispatcher_type,
                                "success": result.success,
                                "message": result.message if hasattr(result, 'message') else None
                            })

                        if self.config.verbose:
                            print(f"   📤 Alert dispatched via {dispatcher_type}: {result.success}")

                    except Exception as e:
                        failed_dispatches += 1

                        # Emit failure event
                        if self.real_time_output:
                            self.real_time_output.emit_event("alert_dispatched", {
                                "dispatcher": dispatcher_type,
                                "success": False,
                                "error": str(e)
                            })

                        if not self.config.quiet:
                            print(f"   ❌ Alert dispatch failed for {dispatcher_type}: {e}")

        except Exception as e:
            if not self.config.quiet:
                print(f"⚠️  Alert processing failed: {e}")

    def _add_exit_code_to_summary(self, exit_code: ExitCode):
        """Add exit code information to the summary reporter."""
        # Add exit code as an error if non-zero for better visibility
        if exit_code != ExitCode.SUCCESS:
            exit_code_descriptions = {
                ExitCode.RULE_FAILURES: "Rule failures or page capture failures detected",
                ExitCode.CRITICAL_FAILURES: "Critical rule failures detected",
                ExitCode.CONFIG_ERROR: "Configuration error",
                ExitCode.RUNTIME_ERROR: "Runtime error during execution",
                ExitCode.TIMEOUT_ERROR: "Operation timed out"
            }
            description = exit_code_descriptions.get(exit_code, f"Unknown error (exit code {exit_code.value})")
            self.summary_reporter.add_error(f"Exit Code {exit_code.value}: {description}")
    
    def _summarize_page_result(self, result: Any) -> Dict[str, Any]:
        """Create summary of page capture result."""
        return {
            "url": getattr(result, 'url', 'unknown'),
            "status": str(getattr(result, 'capture_status', 'unknown')),
            "requests": len(getattr(result, 'network_requests', [])),
            "console_logs": len(getattr(result, 'console_logs', [])),
            "cookies": len(getattr(result, 'cookies', [])),
            "timestamp": getattr(result, 'capture_time', datetime.utcnow()).isoformat()
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

                # Handle error-only captures that don't have status/requests/cookies
                if "error" in capture:
                    lines.append(f"Error: {capture['error']}")
                    lines.append(f"Timestamp: {capture.get('timestamp', 'N/A')}")
                else:
                    lines.append(f"Status: {capture.get('status', 'N/A')}")
                    lines.append(f"Requests: {capture.get('requests', 0)}")
                    lines.append(f"Cookies: {capture.get('cookies', 0)}")
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

    # Scenario
    parser.add_argument(
        "--scenario", "-s",
        help="Scenario identifier for rule filtering"
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
        scenario_id=parsed_args.scenario,
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
        
        # Load audit data - simplified for testing
        # In a real implementation, this would load from file or capture system
        # For testing purposes, we'll create minimal audit data
        from app.audit.models.capture import PageResult, CaptureStatus

        audit_data = [
            PageResult(
                url="https://example.com",
                final_url="https://example.com",
                title="Example Page",
                capture_status=CaptureStatus.SUCCESS,
                network_requests=[],
                cookies=[],
                console_logs=[],
                errors=[],
                page_errors=[],
                metrics={"test": True}
            )
        ]
        
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


if __name__ == "__main__":
    main()