"""Enhanced summary output and formatting for CLI operations.

This module provides comprehensive summary reporting with multiple output formats,
detailed statistics, and formatted reports suitable for both human reading and
machine processing.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

import yaml

from ..audit.rules import RuleResults, Severity


class SummaryData:
    """Container for audit summary data."""

    def __init__(self):
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.pages_captured: int = 0
        self.pages_successful: int = 0
        self.pages_failed: int = 0
        self.rules_evaluated: int = 0
        self.rule_results: Optional[RuleResults] = None
        self.alerts_dispatched: int = 0
        self.alerts_successful: int = 0
        self.environment: Optional[str] = None
        self.target_urls: List[str] = []
        self.output_files: List[Path] = []
        self.errors: List[str] = []
        self.progress_summary: Optional[Dict[str, Any]] = None
        self.exit_code: Optional[int] = None

    @property
    def duration(self) -> Optional[timedelta]:
        """Get total operation duration."""
        if not self.start_time or not self.end_time:
            return None
        return self.end_time - self.start_time

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        duration = self.duration
        return duration.total_seconds() if duration else 0.0

    @property
    def success_rate(self) -> float:
        """Get page capture success rate."""
        if self.pages_captured == 0:
            return 0.0
        return self.pages_successful / self.pages_captured

    @property
    def has_rule_failures(self) -> bool:
        """Check if there were rule failures."""
        if not self.rule_results:
            return False
        return self.rule_results.summary.failed_rules > 0

    @property
    def has_critical_failures(self) -> bool:
        """Check if there were critical rule failures."""
        if not self.rule_results:
            return False
        return self.rule_results.summary.critical_failures > 0


class SummaryFormatter:
    """Formats summary data into various output formats."""

    def __init__(self, format_type: str = "text", verbose: bool = False):
        self.format_type = format_type.lower()
        self.verbose = verbose

    def format_summary(self, summary: SummaryData) -> str:
        """Format summary data into specified format."""
        if self.format_type == "json":
            return self._format_json(summary)
        elif self.format_type == "yaml":
            return self._format_yaml(summary)
        else:
            return self._format_text(summary)

    def _format_json(self, summary: SummaryData) -> str:
        """Format summary as JSON."""
        data = {
            "summary": {
                "start_time": summary.start_time.isoformat() if summary.start_time else None,
                "end_time": summary.end_time.isoformat() if summary.end_time else None,
                "duration_seconds": summary.duration_seconds,
                "environment": summary.environment,
                "target_urls": summary.target_urls,
            },
            "capture": {
                "total_pages": summary.pages_captured,
                "successful": summary.pages_successful,
                "failed": summary.pages_failed,
                "success_rate": round(summary.success_rate * 100, 1)
            },
            "rules": self._get_rule_summary_dict(summary),
            "alerts": {
                "dispatched": summary.alerts_dispatched,
                "successful": summary.alerts_successful
            },
            "output": {
                "files": [str(f) for f in summary.output_files]
            },
            "errors": summary.errors,
            "exit_code": summary.exit_code
        }

        if summary.progress_summary:
            data["progress"] = summary.progress_summary

        return json.dumps(data, indent=2, default=str)

    def _format_yaml(self, summary: SummaryData) -> str:
        """Format summary as YAML."""
        data = {
            "summary": {
                "start_time": summary.start_time.isoformat() if summary.start_time else None,
                "end_time": summary.end_time.isoformat() if summary.end_time else None,
                "duration_seconds": summary.duration_seconds,
                "environment": summary.environment,
                "target_urls": summary.target_urls,
            },
            "capture": {
                "total_pages": summary.pages_captured,
                "successful": summary.pages_successful,
                "failed": summary.pages_failed,
                "success_rate": round(summary.success_rate * 100, 1)
            },
            "rules": self._get_rule_summary_dict(summary),
            "alerts": {
                "dispatched": summary.alerts_dispatched,
                "successful": summary.alerts_successful
            },
            "output": {
                "files": [str(f) for f in summary.output_files]
            },
            "errors": summary.errors,
            "exit_code": summary.exit_code
        }

        if summary.progress_summary:
            data["progress"] = summary.progress_summary

        return yaml.dump(data, default_flow_style=False, sort_keys=True)

    def _format_text(self, summary: SummaryData) -> str:
        """Format summary as human-readable text."""
        lines = []

        # Header
        lines.append("🛡️  TAG SENTINEL AUDIT SUMMARY")
        lines.append("=" * 50)

        # Basic info
        if summary.environment:
            lines.append(f"Environment: {summary.environment}")

        if summary.start_time:
            lines.append(f"Start Time: {summary.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        lines.append(f"Duration: {summary.duration_seconds:.1f} seconds")
        lines.append("")

        # Page capture summary
        lines.append("📊 PAGE CAPTURE")
        lines.append("-" * 20)
        lines.append(f"Total Pages: {summary.pages_captured}")
        lines.append(f"✅ Successful: {summary.pages_successful}")

        if summary.pages_failed > 0:
            lines.append(f"❌ Failed: {summary.pages_failed}")

        lines.append(f"Success Rate: {summary.success_rate * 100:.1f}%")
        lines.append("")

        # Rule evaluation summary
        if summary.rules_evaluated > 0 and summary.rule_results:
            lines.extend(self._format_rule_summary_text(summary.rule_results))

        # Alerts summary
        if summary.alerts_dispatched > 0:
            lines.append("📤 ALERTS")
            lines.append("-" * 20)
            lines.append(f"Dispatched: {summary.alerts_dispatched}")
            lines.append(f"✅ Successful: {summary.alerts_successful}")

            if summary.alerts_dispatched > summary.alerts_successful:
                failed_alerts = summary.alerts_dispatched - summary.alerts_successful
                lines.append(f"❌ Failed: {failed_alerts}")

            lines.append("")

        # Progress summary (verbose mode)
        if self.verbose and summary.progress_summary:
            lines.extend(self._format_progress_summary_text(summary.progress_summary))

        # Output files
        if summary.output_files:
            lines.append("📄 OUTPUT FILES")
            lines.append("-" * 20)
            for output_file in summary.output_files:
                lines.append(f"• {output_file}")
            lines.append("")

        # Errors
        if summary.errors:
            lines.append("❌ ERRORS")
            lines.append("-" * 20)
            for error in summary.errors:
                lines.append(f"• {error}")
            lines.append("")

        # Final status
        lines.append("🏁 FINAL STATUS")
        lines.append("-" * 20)

        if summary.has_critical_failures:
            lines.append("🚨 CRITICAL FAILURES DETECTED")
        elif summary.has_rule_failures:
            lines.append("⚠️  RULE FAILURES DETECTED")
        elif summary.pages_failed > 0:
            lines.append("⚠️  SOME PAGE CAPTURES FAILED")
        else:
            lines.append("✅ ALL CHECKS PASSED")

        # Exit code information
        if summary.exit_code is not None:
            if summary.exit_code == 0:
                lines.append(f"Exit Code: {summary.exit_code} (SUCCESS)")
            else:
                lines.append(f"Exit Code: {summary.exit_code} (ERROR)")
        else:
            lines.append("Exit Code: Not set")

        return "\n".join(lines)

    def _get_rule_summary_dict(self, summary: SummaryData) -> Dict[str, Any]:
        """Get rule summary as dictionary."""
        if not summary.rule_results:
            return {
                "evaluated": False,
                "total": 0,
                "passed": 0,
                "failed": 0,
                "critical": 0,
                "warning": 0,
                "info": 0
            }

        return {
            "evaluated": True,
            "total": summary.rule_results.summary.total_rules,
            "passed": summary.rule_results.summary.passed_rules,
            "failed": summary.rule_results.summary.failed_rules,
            "critical": summary.rule_results.summary.critical_failures,
            "warning": summary.rule_results.summary.warning_failures,
            "info": summary.rule_results.summary.info_failures,
            "execution_time_ms": summary.rule_results.summary.execution_time_ms
        }

    def _format_rule_summary_text(self, rule_results: RuleResults) -> List[str]:
        """Format rule summary as text lines."""
        lines = []

        lines.append("⚖️  RULE EVALUATION")
        lines.append("-" * 20)
        lines.append(f"Total Rules: {rule_results.summary.total_rules}")
        lines.append(f"✅ Passed: {rule_results.summary.passed_rules}")
        lines.append(f"❌ Failed: {rule_results.summary.failed_rules}")

        if rule_results.summary.critical_failures > 0:
            lines.append(f"🚨 Critical: {rule_results.summary.critical_failures}")

        if rule_results.summary.warning_failures > 0:
            lines.append(f"⚠️  Warning: {rule_results.summary.warning_failures}")

        if rule_results.summary.info_failures > 0:
            lines.append(f"ℹ️  Info: {rule_results.summary.info_failures}")

        execution_time = rule_results.summary.execution_time_ms
        if execution_time < 1000:
            lines.append(f"⏱️  Execution: {execution_time}ms")
        else:
            lines.append(f"⏱️  Execution: {execution_time/1000:.1f}s")

        lines.append("")

        # Show top failures in verbose mode
        if self.verbose and rule_results.failures:
            lines.append("❌ TOP RULE FAILURES")
            lines.append("-" * 20)

            # Show up to 5 failures
            for i, failure in enumerate(rule_results.failures[:5], 1):
                severity_emoji = {
                    Severity.CRITICAL: "🚨",
                    Severity.WARNING: "⚠️",
                    Severity.INFO: "ℹ️"
                }.get(failure.severity, "❌")

                lines.append(f"{i}. {severity_emoji} {failure.check_id}")
                lines.append(f"   {failure.message}")

                if failure.evidence:
                    lines.append(f"   Evidence: {len(failure.evidence)} items")

                lines.append("")

            if len(rule_results.failures) > 5:
                remaining = len(rule_results.failures) - 5
                lines.append(f"... and {remaining} more failures")
                lines.append("")

        return lines

    def _format_progress_summary_text(self, progress_summary: Dict[str, Any]) -> List[str]:
        """Format progress summary as text lines."""
        lines = []

        lines.append("📈 PROGRESS DETAILS")
        lines.append("-" * 20)

        lines.append(f"Total Steps: {progress_summary.get('total_steps', 0)}")
        lines.append(f"✅ Completed: {progress_summary.get('completed', 0)}")

        if progress_summary.get('failed', 0) > 0:
            lines.append(f"❌ Failed: {progress_summary['failed']}")

        total_duration = progress_summary.get('total_duration_ms', 0)
        if total_duration < 1000:
            lines.append(f"⏱️  Total Time: {total_duration}ms")
        else:
            lines.append(f"⏱️  Total Time: {total_duration/1000:.1f}s")

        lines.append("")

        # Step details
        steps = progress_summary.get('steps', [])
        if steps:
            lines.append("📋 STEP BREAKDOWN")
            lines.append("-" * 20)

            for step in steps:
                name = step.get('name', 'unknown')
                state = step.get('state', 'unknown')
                duration_ms = step.get('duration_ms')

                state_emoji = {
                    'completed': '✅',
                    'failed': '❌',
                    'running': '🔄',
                    'pending': '⏳'
                }.get(state, '❓')

                duration_str = ""
                if duration_ms is not None:
                    if duration_ms < 1000:
                        duration_str = f" ({duration_ms}ms)"
                    else:
                        duration_str = f" ({duration_ms/1000:.1f}s)"

                lines.append(f"{state_emoji} {step.get('description', name)}{duration_str}")

                if step.get('error'):
                    lines.append(f"   Error: {step['error']}")

            lines.append("")

        return lines


class SummaryReporter:
    """Generates and outputs comprehensive audit summaries."""

    def __init__(self, output_format: str = "text", verbose: bool = False, quiet: bool = False):
        self.formatter = SummaryFormatter(output_format, verbose)
        self.quiet = quiet
        self.summary = SummaryData()

    def set_start_time(self, start_time: datetime):
        """Set audit start time."""
        self.summary.start_time = start_time

    def set_end_time(self, end_time: datetime):
        """Set audit end time."""
        self.summary.end_time = end_time

    def set_page_capture_results(self, total: int, successful: int, failed: int):
        """Set page capture results."""
        self.summary.pages_captured = total
        self.summary.pages_successful = successful
        self.summary.pages_failed = failed

    def set_rule_evaluation_results(self, rule_count: int, rule_results: Optional[RuleResults]):
        """Set rule evaluation results."""
        self.summary.rules_evaluated = rule_count
        self.summary.rule_results = rule_results

    def set_alert_results(self, dispatched: int, successful: int):
        """Set alert dispatch results."""
        self.summary.alerts_dispatched = dispatched
        self.summary.alerts_successful = successful

    def set_environment(self, environment: Optional[str]):
        """Set environment context."""
        self.summary.environment = environment

    def set_target_urls(self, urls: List[str]):
        """Set target URLs."""
        self.summary.target_urls = urls

    def add_output_file(self, file_path: Path):
        """Add output file to summary."""
        self.summary.output_files.append(file_path)

    def add_error(self, error: str):
        """Add error to summary."""
        self.summary.errors.append(error)

    def set_progress_summary(self, progress_summary: Dict[str, Any]):
        """Set progress tracking summary."""
        self.summary.progress_summary = progress_summary

    def set_exit_code(self, exit_code: int):
        """Set the operation exit code."""
        self.summary.exit_code = exit_code

    def generate_summary(self) -> str:
        """Generate formatted summary."""
        if not self.summary.end_time:
            self.summary.end_time = datetime.utcnow()

        return self.formatter.format_summary(self.summary)

    def print_summary(self, output_stream: TextIO = sys.stdout):
        """Print summary to output stream."""
        if self.quiet:
            return

        summary_text = self.generate_summary()
        print(summary_text, file=output_stream)

    def write_summary_file(self, file_path: Path):
        """Write summary to file."""
        summary_text = self.generate_summary()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(summary_text, encoding='utf-8')
        self.add_output_file(file_path)