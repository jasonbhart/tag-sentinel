"""Comprehensive reporting system for rule evaluation results.

This module provides flexible report generation capabilities with multiple
output formats, customization options, and actionable insights for rule
evaluation results. Supports HTML, JSON, YAML, CSV, and text formats.
"""

import csv
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
import yaml

from .models import RuleResults, Failure, Severity, RuleSummary
from .evaluator import RuleEvaluationResult


class ReportFormat(str, Enum):
    """Supported report output formats."""
    JSON = "json"
    YAML = "yaml"
    HTML = "html"
    TEXT = "text"
    CSV = "csv"
    MARKDOWN = "markdown"


class ReportLevel(str, Enum):
    """Report detail levels."""
    SUMMARY = "summary"      # High-level summary only
    STANDARD = "standard"    # Summary + key failures
    DETAILED = "detailed"    # All failures with evidence
    VERBOSE = "verbose"      # Maximum detail including debug info


class SortBy(str, Enum):
    """Sort options for report items."""
    SEVERITY = "severity"
    TIMESTAMP = "timestamp"
    CHECK_ID = "check_id"
    EVIDENCE_COUNT = "evidence_count"


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    # Output configuration
    format: ReportFormat = ReportFormat.HTML
    level: ReportLevel = ReportLevel.STANDARD
    output_file: Optional[Path] = None
    
    # Content filtering
    severity_filter: Optional[List[Severity]] = None
    check_id_filter: Optional[List[str]] = None
    max_failures: Optional[int] = 100
    
    # Sorting and organization
    sort_by: SortBy = SortBy.SEVERITY
    group_by_severity: bool = True
    show_passed_rules: bool = False
    
    # Customization
    title: Optional[str] = None
    description: Optional[str] = None
    include_metadata: bool = True
    include_execution_stats: bool = True
    
    # Formatting options
    pretty_print: bool = True
    include_timestamps: bool = True
    highlight_critical: bool = True
    
    # Template customization
    custom_template: Optional[str] = None
    template_variables: Dict[str, Any] = field(default_factory=dict)


class ReportData(BaseModel):
    """Structured data for report generation."""
    
    # Report metadata
    report_id: str = Field(description="Unique report identifier")
    generation_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report generation timestamp"
    )
    
    # Configuration context
    config: Dict[str, Any] = Field(description="Report configuration")
    environment: Optional[str] = Field(
        default=None,
        description="Environment context"
    )
    
    # Rule evaluation results
    rule_results: RuleResults = Field(description="Rule evaluation results")
    evaluation_results: List[RuleEvaluationResult] = Field(
        default_factory=list,
        description="Detailed evaluation results"
    )
    
    # Summary statistics
    summary: RuleSummary = Field(description="Evaluation summary")
    
    # Filtered and organized failures
    failures_by_severity: Dict[str, List[Failure]] = Field(
        default_factory=dict,
        description="Failures grouped by severity"
    )
    
    # Additional context
    target_urls: List[str] = Field(
        default_factory=list,
        description="URLs that were audited"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional report metadata"
    )


class BaseReportGenerator(ABC):
    """Abstract base class for report generators."""
    
    def __init__(self, config: ReportConfig):
        self.config = config
    
    @abstractmethod
    def generate(self, data: ReportData) -> str:
        """Generate report content.
        
        Args:
            data: Report data to render
            
        Returns:
            Generated report content
        """
        pass
    
    def _filter_failures(self, failures: List[Failure]) -> List[Failure]:
        """Apply filtering to failures based on configuration."""
        filtered = failures
        
        # Filter by severity
        if self.config.severity_filter:
            severity_set = set(self.config.severity_filter)
            filtered = [f for f in filtered if f.severity in severity_set]
        
        # Filter by check ID
        if self.config.check_id_filter:
            check_id_set = set(self.config.check_id_filter)
            filtered = [f for f in filtered if f.check_id in check_id_set]
        
        # Apply limit
        if self.config.max_failures:
            filtered = filtered[:self.config.max_failures]
        
        return filtered
    
    def _sort_failures(self, failures: List[Failure]) -> List[Failure]:
        """Sort failures based on configuration."""
        if self.config.sort_by == SortBy.SEVERITY:
            # Sort by severity (critical first)
            severity_order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
            return sorted(failures, key=lambda f: severity_order.get(f.severity, 3))
        elif self.config.sort_by == SortBy.CHECK_ID:
            return sorted(failures, key=lambda f: f.check_id)
        elif self.config.sort_by == SortBy.EVIDENCE_COUNT:
            return sorted(failures, key=lambda f: len(f.evidence) if f.evidence else 0, reverse=True)
        else:
            return failures
    
    def _group_failures_by_severity(self, failures: List[Failure]) -> Dict[str, List[Failure]]:
        """Group failures by severity level."""
        groups = {}
        for failure in failures:
            severity_key = failure.severity.value
            if severity_key not in groups:
                groups[severity_key] = []
            groups[severity_key].append(failure)
        return groups


class JSONReportGenerator(BaseReportGenerator):
    """JSON format report generator."""
    
    def generate(self, data: ReportData) -> str:
        """Generate JSON report."""
        report_dict = {
            "report_metadata": {
                "id": data.report_id,
                "generation_time": data.generation_time.isoformat(),
                "format": "json",
                "level": self.config.level.value,
                "environment": data.environment
            },
            "summary": {
                "total_rules": data.summary.total_rules,
                "passed_rules": data.summary.passed_rules,
                "failed_rules": data.summary.failed_rules,
                "critical_failures": data.summary.critical_failures,
                "warning_failures": data.summary.warning_failures,
                "info_failures": data.summary.info_failures,
                "execution_time_ms": data.summary.execution_time_ms
            },
            "failures": []
        }
        
        # Add failures based on detail level
        if self.config.level in [ReportLevel.STANDARD, ReportLevel.DETAILED, ReportLevel.VERBOSE]:
            filtered_failures = self._filter_failures(data.rule_results.failures)
            sorted_failures = self._sort_failures(filtered_failures)
            
            for failure in sorted_failures:
                failure_dict = {
                    "check_id": failure.check_id,
                    "severity": failure.severity.value,
                    "message": failure.message,
                    "details": failure.details
                }
                
                if self.config.level in [ReportLevel.DETAILED, ReportLevel.VERBOSE]:
                    failure_dict["evidence"] = failure.evidence or []
                    failure_dict["evidence_count"] = len(failure.evidence) if failure.evidence else 0
                
                if self.config.level == ReportLevel.VERBOSE:
                    failure_dict["context"] = failure.context or {}
                
                report_dict["failures"].append(failure_dict)
        
        # Add metadata if requested
        if self.config.include_metadata:
            report_dict["metadata"] = data.metadata
            report_dict["target_urls"] = data.target_urls
        
        # Add evaluation results for verbose level
        if self.config.level == ReportLevel.VERBOSE and data.evaluation_results:
            report_dict["evaluation_results"] = [
                {
                    "rule_id": result.rule.id,
                    "passed": result.passed,
                    "execution_time_ms": result.execution_time_ms,
                    "check_results": [
                        {
                            "check_id": cr.check_id,
                            "check_name": cr.check_name,
                            "passed": cr.passed,
                            "message": cr.message
                        }
                        for cr in result.check_results
                    ]
                }
                for result in data.evaluation_results
            ]
        
        if self.config.pretty_print:
            return json.dumps(report_dict, indent=2, default=str, ensure_ascii=False)
        else:
            return json.dumps(report_dict, default=str, ensure_ascii=False)


class HTMLReportGenerator(BaseReportGenerator):
    """HTML format report generator with styling."""
    
    def generate(self, data: ReportData) -> str:
        """Generate HTML report."""
        title = self.config.title or "Tag Sentinel Rule Evaluation Report"
        
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "    <meta charset='utf-8'>",
            f"    <title>{title}</title>",
            "    <meta name='viewport' content='width=device-width, initial-scale=1'>",
            self._get_html_styles(),
            "</head>",
            "<body>",
            "    <div class='container'>",
            self._generate_header(data, title),
            self._generate_summary_section(data),
            self._generate_failures_section(data),
            self._generate_footer(data),
            "    </div>",
            "</body>",
            "</html>"
        ]
        
        return "\n".join(html_parts)
    
    def _get_html_styles(self) -> str:
        """Get CSS styles for HTML report."""
        return """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
            color: #212529;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        .header .subtitle {
            opacity: 0.9;
            margin-top: 0.5rem;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .summary-card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #6c757d;
        }
        .summary-card.success { border-left-color: #28a745; }
        .summary-card.warning { border-left-color: #ffc107; }
        .summary-card.danger { border-left-color: #dc3545; }
        .summary-card .number {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .summary-card .label {
            color: #6c757d;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .section {
            background: white;
            border-radius: 8px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section h2 {
            margin-top: 0;
            color: #495057;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 0.5rem;
        }
        .failure-item {
            border: 1px solid #dee2e6;
            border-radius: 6px;
            margin-bottom: 1rem;
            overflow: hidden;
        }
        .failure-header {
            padding: 1rem;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .failure-header.critical { background-color: #f8d7da; color: #721c24; }
        .failure-header.warning { background-color: #fff3cd; color: #856404; }
        .failure-header.info { background-color: #d1ecf1; color: #0c5460; }
        .failure-body {
            padding: 1rem;
            background-color: #f8f9fa;
        }
        .severity-badge {
            font-size: 0.75rem;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
        }
        .severity-badge.critical { background-color: #dc3545; }
        .severity-badge.warning { background-color: #ffc107; color: #212529; }
        .severity-badge.info { background-color: #17a2b8; }
        .evidence-list {
            margin-top: 1rem;
            padding: 0;
            list-style: none;
        }
        .evidence-item {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }
        .footer {
            text-align: center;
            color: #6c757d;
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid #e9ecef;
        }
        .no-failures {
            text-align: center;
            color: #28a745;
            font-size: 1.2rem;
            margin: 2rem 0;
        }
        @media (max-width: 768px) {
            .container { padding: 10px; }
            .header { padding: 1rem; }
            .header h1 { font-size: 1.8rem; }
            .summary-grid { grid-template-columns: 1fr; }
        }
    </style>"""
    
    def _generate_header(self, data: ReportData, title: str) -> str:
        """Generate HTML header section."""
        subtitle_parts = []
        if data.environment:
            subtitle_parts.append(f"Environment: {data.environment}")
        if self.config.include_timestamps:
            subtitle_parts.append(f"Generated: {data.generation_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        subtitle = " | ".join(subtitle_parts) if subtitle_parts else ""
        
        return f"""
        <div class='header'>
            <h1>🛡️ {title}</h1>
            {f"<div class='subtitle'>{subtitle}</div>" if subtitle else ""}
        </div>"""
    
    def _generate_summary_section(self, data: ReportData) -> str:
        """Generate summary statistics section."""
        summary = data.summary
        
        # Determine overall status
        if summary.critical_failures > 0:
            status_class = "danger"
            status_icon = "🚨"
            status_text = "Critical Issues Found"
        elif summary.failed_rules > 0:
            status_class = "warning"
            status_icon = "⚠️"
            status_text = "Issues Found"
        else:
            status_class = "success"
            status_icon = "✅"
            status_text = "All Rules Passed"
        
        cards = [
            f"""
            <div class='summary-card {status_class}'>
                <div class='number'>{status_icon}</div>
                <div class='label'>{status_text}</div>
            </div>""",
            f"""
            <div class='summary-card'>
                <div class='number'>{summary.total_rules}</div>
                <div class='label'>Total Rules</div>
            </div>""",
            f"""
            <div class='summary-card success'>
                <div class='number'>{summary.passed_rules}</div>
                <div class='label'>Passed</div>
            </div>""",
            f"""
            <div class='summary-card danger'>
                <div class='number'>{summary.failed_rules}</div>
                <div class='label'>Failed</div>
            </div>"""
        ]
        
        if summary.critical_failures > 0:
            cards.append(f"""
            <div class='summary-card danger'>
                <div class='number'>{summary.critical_failures}</div>
                <div class='label'>Critical</div>
            </div>""")
        
        if summary.warning_failures > 0:
            cards.append(f"""
            <div class='summary-card warning'>
                <div class='number'>{summary.warning_failures}</div>
                <div class='label'>Warnings</div>
            </div>""")
        
        if self.config.include_execution_stats:
            cards.append(f"""
            <div class='summary-card'>
                <div class='number'>{summary.execution_time_ms}ms</div>
                <div class='label'>Execution Time</div>
            </div>""")
        
        return f"""
        <div class='section'>
            <h2>📊 Summary</h2>
            <div class='summary-grid'>
                {''.join(cards)}
            </div>
        </div>"""
    
    def _generate_failures_section(self, data: ReportData) -> str:
        """Generate failures section."""
        if self.config.level == ReportLevel.SUMMARY:
            return ""
        
        failures = self._filter_failures(data.rule_results.failures)
        
        if not failures:
            return """
        <div class='section'>
            <h2>✅ Results</h2>
            <div class='no-failures'>
                🎉 All rules passed successfully! No issues found.
            </div>
        </div>"""
        
        sorted_failures = self._sort_failures(failures)
        
        failure_items = []
        for failure in sorted_failures:
            severity_class = failure.severity.value.lower()
            severity_badge = f"<span class='severity-badge {severity_class}'>{failure.severity.value}</span>"
            
            evidence_html = ""
            if (self.config.level in [ReportLevel.DETAILED, ReportLevel.VERBOSE] and 
                failure.evidence):
                evidence_items = [
                    f"<li class='evidence-item'>{self._escape_html(str(evidence))}</li>"
                    for evidence in failure.evidence[:10]  # Limit evidence items
                ]
                evidence_count = len(failure.evidence)
                if evidence_count > 10:
                    evidence_items.append(f"<li class='evidence-item'><em>... and {evidence_count - 10} more items</em></li>")
                
                evidence_html = f"""
                <div>
                    <strong>Evidence ({evidence_count} items):</strong>
                    <ul class='evidence-list'>
                        {''.join(evidence_items)}
                    </ul>
                </div>"""
            
            failure_item = f"""
            <div class='failure-item'>
                <div class='failure-header {severity_class}'>
                    <span>{failure.check_id}</span>
                    {severity_badge}
                </div>
                <div class='failure-body'>
                    <p><strong>Message:</strong> {self._escape_html(failure.message)}</p>
                    {f"<p><strong>Details:</strong> {self._escape_html(failure.details)}</p>" if failure.details else ""}
                    {evidence_html}
                </div>
            </div>"""
            
            failure_items.append(failure_item)
        
        return f"""
        <div class='section'>
            <h2>❌ Failures ({len(failures)})</h2>
            {''.join(failure_items)}
        </div>"""
    
    def _generate_footer(self, data: ReportData) -> str:
        """Generate footer section."""
        return f"""
        <div class='footer'>
            <p>Generated by Tag Sentinel Rule Engine</p>
            <p>Report ID: {data.report_id}</p>
        </div>"""
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML characters in text."""
        if not text:
            return ""
        return (str(text)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#x27;"))


class TextReportGenerator(BaseReportGenerator):
    """Plain text format report generator."""
    
    def generate(self, data: ReportData) -> str:
        """Generate plain text report."""
        lines = []
        
        # Header
        title = self.config.title or "Tag Sentinel Rule Evaluation Report"
        lines.append("=" * len(title))
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")
        
        # Metadata
        if self.config.include_timestamps:
            lines.append(f"Generated: {data.generation_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        if data.environment:
            lines.append(f"Environment: {data.environment}")
        if data.target_urls:
            lines.append(f"Target URLs: {', '.join(data.target_urls[:3])}")
            if len(data.target_urls) > 3:
                lines.append(f"             ... and {len(data.target_urls) - 3} more")
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-------")
        summary = data.summary
        lines.append(f"Total Rules:      {summary.total_rules}")
        lines.append(f"Passed Rules:     {summary.passed_rules}")
        lines.append(f"Failed Rules:     {summary.failed_rules}")
        
        if summary.critical_failures > 0:
            lines.append(f"Critical Failures: {summary.critical_failures}")
        if summary.warning_failures > 0:
            lines.append(f"Warning Failures:  {summary.warning_failures}")
        if summary.info_failures > 0:
            lines.append(f"Info Failures:     {summary.info_failures}")
        
        if self.config.include_execution_stats:
            lines.append(f"Execution Time:   {summary.execution_time_ms}ms")
        
        lines.append("")
        
        # Overall status
        if summary.critical_failures > 0:
            lines.append("STATUS: 🚨 CRITICAL ISSUES FOUND")
        elif summary.failed_rules > 0:
            lines.append("STATUS: ⚠️  ISSUES FOUND")
        else:
            lines.append("STATUS: ✅ ALL RULES PASSED")
        lines.append("")
        
        # Failures (if not summary level)
        if self.config.level != ReportLevel.SUMMARY:
            failures = self._filter_failures(data.rule_results.failures)
            
            if failures:
                lines.append("FAILURES")
                lines.append("--------")
                
                sorted_failures = self._sort_failures(failures)
                
                for i, failure in enumerate(sorted_failures, 1):
                    severity_icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
                    icon = severity_icon.get(failure.severity.value.lower(), "❌")
                    
                    lines.append(f"{i}. {icon} {failure.check_id} ({failure.severity.value.upper()})")
                    lines.append(f"   Message: {failure.message}")
                    
                    if failure.details:
                        lines.append(f"   Details: {failure.details}")
                    
                    if (self.config.level in [ReportLevel.DETAILED, ReportLevel.VERBOSE] and 
                        failure.evidence):
                        lines.append(f"   Evidence ({len(failure.evidence)} items):")
                        for j, evidence in enumerate(failure.evidence[:5], 1):  # Limit to 5
                            lines.append(f"     {j}. {evidence}")
                        if len(failure.evidence) > 5:
                            lines.append(f"     ... and {len(failure.evidence) - 5} more items")
                    
                    lines.append("")
            else:
                lines.append("NO FAILURES")
                lines.append("-----------")
                lines.append("🎉 All rules passed successfully!")
                lines.append("")
        
        # Footer
        lines.append("-" * 50)
        lines.append("Generated by Tag Sentinel Rule Engine")
        lines.append(f"Report ID: {data.report_id}")
        
        return "\n".join(lines)


class CSVReportGenerator(BaseReportGenerator):
    """CSV format report generator."""
    
    def generate(self, data: ReportData) -> str:
        """Generate CSV report."""
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        headers = [
            "check_id",
            "severity", 
            "message",
            "details",
            "evidence_count"
        ]
        
        if self.config.include_timestamps:
            headers.append("timestamp")
        
        writer.writerow(headers)
        
        # Write failure data
        failures = self._filter_failures(data.rule_results.failures)
        sorted_failures = self._sort_failures(failures)
        
        for failure in sorted_failures:
            row = [
                failure.check_id,
                failure.severity.value,
                failure.message,
                failure.details or "",
                len(failure.evidence) if failure.evidence else 0
            ]
            
            if self.config.include_timestamps:
                row.append(data.generation_time.isoformat())
            
            writer.writerow(row)
        
        return output.getvalue()


class MarkdownReportGenerator(BaseReportGenerator):
    """Markdown format report generator."""
    
    def generate(self, data: ReportData) -> str:
        """Generate Markdown report."""
        lines = []
        
        # Header
        title = self.config.title or "Tag Sentinel Rule Evaluation Report"
        lines.append(f"# 🛡️ {title}")
        lines.append("")
        
        # Metadata
        if self.config.include_timestamps:
            lines.append(f"**Generated:** {data.generation_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        if data.environment:
            lines.append(f"**Environment:** {data.environment}")
        if data.target_urls:
            lines.append(f"**Target URLs:** {len(data.target_urls)} URLs")
        lines.append("")
        
        # Summary
        lines.append("## 📊 Summary")
        lines.append("")
        summary = data.summary
        
        # Status badge
        if summary.critical_failures > 0:
            lines.append("![Status](https://img.shields.io/badge/Status-Critical%20Issues-red)")
        elif summary.failed_rules > 0:
            lines.append("![Status](https://img.shields.io/badge/Status-Issues%20Found-yellow)")
        else:
            lines.append("![Status](https://img.shields.io/badge/Status-All%20Passed-green)")
        lines.append("")
        
        # Summary table
        lines.append("| Metric | Count |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Rules | {summary.total_rules} |")
        lines.append(f"| ✅ Passed | {summary.passed_rules} |")
        lines.append(f"| ❌ Failed | {summary.failed_rules} |")
        
        if summary.critical_failures > 0:
            lines.append(f"| 🚨 Critical | {summary.critical_failures} |")
        if summary.warning_failures > 0:
            lines.append(f"| ⚠️ Warnings | {summary.warning_failures} |")
        if summary.info_failures > 0:
            lines.append(f"| ℹ️ Info | {summary.info_failures} |")
        
        if self.config.include_execution_stats:
            lines.append(f"| ⏱️ Execution Time | {summary.execution_time_ms}ms |")
        lines.append("")
        
        # Failures
        if self.config.level != ReportLevel.SUMMARY:
            failures = self._filter_failures(data.rule_results.failures)
            
            if failures:
                lines.append("## ❌ Failures")
                lines.append("")
                
                sorted_failures = self._sort_failures(failures)
                
                for i, failure in enumerate(sorted_failures, 1):
                    severity_emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
                    emoji = severity_emoji.get(failure.severity.value.lower(), "❌")
                    
                    lines.append(f"### {i}. {emoji} {failure.check_id}")
                    lines.append("")
                    lines.append(f"**Severity:** {failure.severity.value.upper()}")
                    lines.append(f"**Message:** {failure.message}")
                    
                    if failure.details:
                        lines.append(f"**Details:** {failure.details}")
                    
                    if (self.config.level in [ReportLevel.DETAILED, ReportLevel.VERBOSE] and 
                        failure.evidence):
                        lines.append(f"**Evidence ({len(failure.evidence)} items):**")
                        lines.append("")
                        for j, evidence in enumerate(failure.evidence[:10], 1):  # Limit to 10
                            lines.append(f"{j}. `{evidence}`")
                        if len(failure.evidence) > 10:
                            lines.append(f"... and {len(failure.evidence) - 10} more items")
                    
                    lines.append("")
            else:
                lines.append("## ✅ Results")
                lines.append("")
                lines.append("🎉 **All rules passed successfully!** No issues found.")
                lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by Tag Sentinel Rule Engine*")
        lines.append(f"*Report ID: {data.report_id}*")
        
        return "\n".join(lines)


class ReportGenerator:
    """Main report generator that coordinates different format generators."""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self._generators = {
            ReportFormat.JSON: JSONReportGenerator(config),
            ReportFormat.HTML: HTMLReportGenerator(config),
            ReportFormat.TEXT: TextReportGenerator(config),
            ReportFormat.CSV: CSVReportGenerator(config),
            ReportFormat.MARKDOWN: MarkdownReportGenerator(config),
            ReportFormat.YAML: self._create_yaml_generator(config)
        }
    
    def _create_yaml_generator(self, config: ReportConfig):
        """Create YAML generator (wrapper around JSON)."""
        class YAMLGenerator(JSONReportGenerator):
            def generate(self, data: ReportData) -> str:
                json_content = super().generate(data)
                data_dict = json.loads(json_content)
                return yaml.dump(data_dict, default_flow_style=False, sort_keys=False)
        
        return YAMLGenerator(config)
    
    def generate_report(
        self,
        rule_results: RuleResults,
        evaluation_results: Optional[List[RuleEvaluationResult]] = None,
        environment: Optional[str] = None,
        target_urls: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate report from rule evaluation results.
        
        Args:
            rule_results: Rule evaluation results
            evaluation_results: Detailed evaluation results
            environment: Environment context
            target_urls: URLs that were audited
            metadata: Additional metadata
            
        Returns:
            Generated report content
        """
        # Generate unique report ID
        report_id = f"report_{int(time.time())}_{hash(str(rule_results)) % 10000:04d}"
        
        # Prepare report data
        data = ReportData(
            report_id=report_id,
            config=self.config.__dict__,
            environment=environment,
            rule_results=rule_results,
            evaluation_results=evaluation_results or [],
            summary=rule_results.summary,
            target_urls=target_urls or [],
            metadata=metadata or {}
        )
        
        # Group failures by severity if requested
        if self.config.group_by_severity:
            data.failures_by_severity = self._group_failures_by_severity(rule_results.failures)
        
        # Generate report using appropriate generator
        generator = self._generators.get(self.config.format)
        if not generator:
            raise ValueError(f"Unsupported report format: {self.config.format}")
        
        content = generator.generate(data)
        
        # Save to file if specified
        if self.config.output_file:
            self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.config.output_file.write_text(content, encoding='utf-8')
        
        return content
    
    def _group_failures_by_severity(self, failures: List[Failure]) -> Dict[str, List[Failure]]:
        """Group failures by severity level."""
        groups = {}
        for failure in failures:
            severity_key = failure.severity.value
            if severity_key not in groups:
                groups[severity_key] = []
            groups[severity_key].append(failure)
        return groups


# Convenience functions

def generate_report(
    rule_results: RuleResults,
    format: Union[ReportFormat, str] = ReportFormat.HTML,
    level: Union[ReportLevel, str] = ReportLevel.STANDARD,
    output_file: Optional[Union[Path, str]] = None,
    **kwargs
) -> str:
    """Generate a report from rule evaluation results.
    
    Args:
        rule_results: Rule evaluation results
        format: Report format (default: HTML)
        level: Report detail level (default: STANDARD)
        output_file: Optional output file path
        **kwargs: Additional configuration options
        
    Returns:
        Generated report content
    """
    config = ReportConfig(
        format=ReportFormat(format) if isinstance(format, str) else format,
        level=ReportLevel(level) if isinstance(level, str) else level,
        output_file=Path(output_file) if output_file else None,
        **kwargs
    )
    
    generator = ReportGenerator(config)
    return generator.generate_report(rule_results)


def generate_html_report(
    rule_results: RuleResults,
    title: Optional[str] = None,
    output_file: Optional[Union[Path, str]] = None,
    **kwargs
) -> str:
    """Generate HTML report (convenience function).
    
    Args:
        rule_results: Rule evaluation results
        title: Report title
        output_file: Optional output file path
        **kwargs: Additional configuration options
        
    Returns:
        Generated HTML report content
    """
    return generate_report(
        rule_results,
        format=ReportFormat.HTML,
        title=title,
        output_file=output_file,
        **kwargs
    )


def generate_json_report(
    rule_results: RuleResults,
    output_file: Optional[Union[Path, str]] = None,
    pretty_print: bool = True,
    **kwargs
) -> str:
    """Generate JSON report (convenience function).
    
    Args:
        rule_results: Rule evaluation results
        output_file: Optional output file path
        pretty_print: Format JSON with indentation
        **kwargs: Additional configuration options
        
    Returns:
        Generated JSON report content
    """
    return generate_report(
        rule_results,
        format=ReportFormat.JSON,
        output_file=output_file,
        pretty_print=pretty_print,
        **kwargs
    )