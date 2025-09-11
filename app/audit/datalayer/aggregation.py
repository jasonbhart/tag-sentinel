"""Run-level data aggregation for dataLayer integrity analysis.

This module provides comprehensive aggregation capabilities for analyzing dataLayer
data across multiple pages within a crawl run, including variable tracking, 
event frequency analysis, and validation issue categorization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Counter
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import json
import statistics

from .models import ValidationIssue, ValidationSeverity, DLAggregate
from .config import AggregationConfig
from .runtime_validation import validate_types

logger = logging.getLogger(__name__)


@dataclass
class VariableStats:
    """Statistics for a dataLayer variable."""
    name: str
    presence_count: int = 0
    total_pages: int = 0
    example_values: List[Any] = field(default_factory=list)
    value_types: Set[str] = field(default_factory=set)
    paths_seen: Set[str] = field(default_factory=set)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    
    @property
    def presence_rate(self) -> float:
        """Calculate presence rate as percentage."""
        if self.total_pages == 0:
            return 0.0
        return (self.presence_count / self.total_pages) * 100.0
    
    @property
    def is_consistent(self) -> bool:
        """Check if variable appears on all pages."""
        return self.presence_count == self.total_pages
    
    def add_value(self, value: Any, path: str, timestamp: Optional[datetime] = None) -> None:
        """Add a new observed value for this variable.
        
        Args:
            value: The value observed
            path: JSON Pointer path where value was found
            timestamp: When the value was observed
        """
        # Update counts
        self.presence_count += 1
        
        # Track value type
        self.value_types.add(type(value).__name__)
        
        # Track path
        self.paths_seen.add(path)
        
        # Track timestamps
        if timestamp:
            if not self.first_seen or timestamp < self.first_seen:
                self.first_seen = timestamp
            if not self.last_seen or timestamp > self.last_seen:
                self.last_seen = timestamp
        
        # Add to examples if we have room
        if len(self.example_values) < 10:  # Configurable limit
            if value not in self.example_values:
                self.example_values.append(value)


@dataclass
class EventStats:
    """Statistics for dataLayer events."""
    event_type: str
    frequency: int = 0
    pages_seen: Set[str] = field(default_factory=set)
    example_data: List[Dict[str, Any]] = field(default_factory=list)
    variable_patterns: Dict[str, int] = field(default_factory=dict)
    timing_stats: List[float] = field(default_factory=list)  # Time between events
    
    @property
    def unique_pages(self) -> int:
        """Number of unique pages where this event was seen."""
        return len(self.pages_seen)
    
    @property
    def avg_timing(self) -> Optional[float]:
        """Average time between events (in seconds)."""
        if len(self.timing_stats) < 2:
            return None
        return statistics.mean(self.timing_stats)
    
    def add_event(self, event_data: Dict[str, Any], page_url: str, timestamp: Optional[datetime] = None) -> None:
        """Add an observed event occurrence.
        
        Args:
            event_data: Event data dictionary
            page_url: URL where event was observed
            timestamp: When the event occurred
        """
        self.frequency += 1
        self.pages_seen.add(page_url)
        
        # Add to examples if we have room
        if len(self.example_data) < 5:  # Configurable limit
            self.example_data.append({
                'data': event_data.copy(),
                'page_url': page_url,
                'timestamp': timestamp.isoformat() if timestamp else None
            })
        
        # Track variable patterns in this event
        for key in event_data.keys():
            self.variable_patterns[key] = self.variable_patterns.get(key, 0) + 1


@dataclass
class ValidationIssuePattern:
    """Pattern analysis for validation issues."""
    issue_type: str
    severity: ValidationSeverity
    frequency: int = 0
    affected_variables: Set[str] = field(default_factory=set)
    affected_pages: Set[str] = field(default_factory=set)
    example_messages: List[str] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    
    @property
    def impact_score(self) -> float:
        """Calculate impact score based on severity and frequency."""
        severity_weights = {
            ValidationSeverity.CRITICAL: 10,
            ValidationSeverity.WARNING: 5,
            ValidationSeverity.INFO: 1
        }
        base_score = severity_weights.get(self.severity, 1)
        return base_score * self.frequency
    
    def add_issue(self, issue: ValidationIssue, timestamp: Optional[datetime] = None) -> None:
        """Add a validation issue to this pattern.
        
        Args:
            issue: ValidationIssue to add
            timestamp: When the issue was observed
        """
        self.frequency += 1
        self.affected_pages.add(issue.page_url)
        
        if issue.variable_name:
            self.affected_variables.add(issue.variable_name)
        
        # Track example messages
        if len(self.example_messages) < 5 and issue.message not in self.example_messages:
            self.example_messages.append(issue.message)
        
        # Update timestamps
        if timestamp:
            if not self.first_seen or timestamp < self.first_seen:
                self.first_seen = timestamp
            if not self.last_seen or timestamp > self.last_seen:
                self.last_seen = timestamp


class DataAggregator:
    """Aggregates dataLayer data across multiple pages for analysis."""
    
    def __init__(self, config: AggregationConfig | None = None):
        """Initialize aggregator with configuration.
        
        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregationConfig()
        
        # Variable tracking
        self.variable_stats: Dict[str, VariableStats] = {}
        self.total_pages_processed: int = 0
        
        # Event tracking  
        self.event_stats: Dict[str, EventStats] = {}
        
        # Validation issue tracking
        self.issue_patterns: Dict[str, ValidationIssuePattern] = {}
        
        # Run metadata
        self.run_started: Optional[datetime] = None
        self.run_completed: Optional[datetime] = None
        self.processed_urls: List[str] = []
        
    @validate_types()
    def process_page_data(
        self,
        page_url: str,
        latest_data: Dict[str, Any],
        events_data: List[Dict[str, Any]],
        validation_issues: List[ValidationIssue],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Process data from a single page.
        
        Args:
            page_url: URL of the page
            latest_data: Latest dataLayer state
            events_data: List of events
            validation_issues: Validation issues found
            timestamp: When the data was captured
        """
        if not timestamp:
            timestamp = datetime.utcnow()
            
        if not self.run_started:
            self.run_started = timestamp
        
        self.total_pages_processed += 1
        self.processed_urls.append(page_url)
        
        logger.debug(f"Processing aggregation data for {page_url}")
        
        # Process variable presence and examples
        if self.config.track_variable_presence:
            self._process_variables(latest_data, "", timestamp)
        
        # Process events
        if self.config.track_event_frequency:
            self._process_events(events_data, page_url, timestamp)
        
        # Process validation issues
        self._process_validation_issues(validation_issues, timestamp)
        
        self.run_completed = timestamp
    
    def _process_variables(
        self,
        data: Dict[str, Any],
        path_prefix: str = "",
        timestamp: Optional[datetime] = None
    ) -> None:
        """Process variables from data structure.
        
        Args:
            data: Data to process
            path_prefix: Current path prefix
            timestamp: Processing timestamp
        """
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path_prefix}/{key}" if path_prefix else f"/{key}"
                
                # Track this variable
                if key not in self.variable_stats:
                    self.variable_stats[key] = VariableStats(
                        name=key,
                        total_pages=self.total_pages_processed
                    )
                
                # Update total pages for existing variables
                self.variable_stats[key].total_pages = self.total_pages_processed
                
                # Add this occurrence
                self.variable_stats[key].add_value(value, current_path, timestamp)
                
                # Recursively process nested structures
                if isinstance(value, dict):
                    self._process_variables(value, current_path, timestamp)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            item_path = f"{current_path}/{i}"
                            self._process_variables(item, item_path, timestamp)
    
    def _process_events(
        self,
        events_data: List[Dict[str, Any]],
        page_url: str,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Process event data for aggregation.
        
        Args:
            events_data: List of events
            page_url: URL where events occurred
            timestamp: Processing timestamp
        """
        for event in events_data:
            # Determine event type
            event_type = (
                event.get('event') or 
                event.get('eventName') or 
                event.get('eventAction') or
                'unknown_event'
            )
            
            # Initialize or update event stats
            if event_type not in self.event_stats:
                self.event_stats[event_type] = EventStats(event_type=event_type)
            
            self.event_stats[event_type].add_event(event, page_url, timestamp)
    
    def _process_validation_issues(
        self,
        issues: List[ValidationIssue],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Process validation issues for pattern analysis.
        
        Args:
            issues: List of validation issues
            timestamp: Processing timestamp
        """
        for issue in issues:
            # Create pattern key
            pattern_key = f"{issue.schema_rule or 'unknown'}:{issue.severity.value}"
            
            # Initialize or update pattern
            if pattern_key not in self.issue_patterns:
                self.issue_patterns[pattern_key] = ValidationIssuePattern(
                    issue_type=issue.schema_rule or 'unknown',
                    severity=issue.severity
                )
            
            self.issue_patterns[pattern_key].add_issue(issue, timestamp)
    
    @validate_types()
    def generate_aggregate_report(self) -> DLAggregate:
        """Generate comprehensive aggregate report.
        
        Returns:
            Complete aggregation results
        """
        logger.debug("Generating aggregate report")
        
        # Variable analysis
        variable_summary = self._analyze_variables()
        
        # Event analysis
        event_summary = self._analyze_events()
        
        # Validation issue analysis
        issue_summary = self._analyze_validation_issues()
        
        # Overall statistics
        run_duration = None
        if self.run_started and self.run_completed:
            run_duration = (self.run_completed - self.run_started).total_seconds()
        
        return DLAggregate(
            total_pages=self.total_pages_processed,
            run_duration_seconds=run_duration,
            variables_found=len(self.variable_stats),
            events_found=len(self.event_stats),
            validation_issues=sum(p.frequency for p in self.issue_patterns.values()),
            variable_analysis=variable_summary,
            event_analysis=event_summary,
            validation_analysis=issue_summary,
            run_started=self.run_started,
            run_completed=self.run_completed,
            processed_urls=self.processed_urls.copy()
        )
    
    def _analyze_variables(self) -> Dict[str, Any]:
        """Analyze variable statistics.
        
        Returns:
            Variable analysis summary
        """
        if not self.variable_stats:
            return {'total_variables': 0}
        
        # Consistency analysis
        consistent_vars = [
            name for name, stats in self.variable_stats.items()
            if stats.is_consistent
        ]
        
        # Presence rate distribution
        presence_rates = [stats.presence_rate for stats in self.variable_stats.values()]
        
        # Most/least common variables
        variables_by_frequency = sorted(
            self.variable_stats.items(),
            key=lambda x: x[1].presence_count,
            reverse=True
        )
        
        return {
            'total_variables': len(self.variable_stats),
            'consistent_variables': len(consistent_vars),
            'consistent_rate': len(consistent_vars) / len(self.variable_stats) * 100,
            'avg_presence_rate': statistics.mean(presence_rates) if presence_rates else 0,
            'median_presence_rate': statistics.median(presence_rates) if presence_rates else 0,
            'most_common_variables': [
                {
                    'name': name,
                    'presence_count': stats.presence_count,
                    'presence_rate': stats.presence_rate,
                    'example_values': stats.example_values[:3],  # Top 3 examples
                    'value_types': list(stats.value_types)
                }
                for name, stats in variables_by_frequency[:10]
            ],
            'least_common_variables': [
                {
                    'name': name,
                    'presence_count': stats.presence_count,
                    'presence_rate': stats.presence_rate,
                    'paths_seen': list(stats.paths_seen)[:3]  # Top 3 paths
                }
                for name, stats in variables_by_frequency[-5:]
            ]
        }
    
    def _analyze_events(self) -> Dict[str, Any]:
        """Analyze event statistics.
        
        Returns:
            Event analysis summary
        """
        if not self.event_stats:
            return {'total_event_types': 0}
        
        # Sort events by frequency
        events_by_frequency = sorted(
            self.event_stats.items(),
            key=lambda x: x[1].frequency,
            reverse=True
        )
        
        # Calculate total events
        total_events = sum(stats.frequency for stats in self.event_stats.values())
        
        return {
            'total_event_types': len(self.event_stats),
            'total_events': total_events,
            'avg_events_per_page': total_events / self.total_pages_processed if self.total_pages_processed > 0 else 0,
            'most_frequent_events': [
                {
                    'event_type': name,
                    'frequency': stats.frequency,
                    'unique_pages': stats.unique_pages,
                    'avg_timing': stats.avg_timing,
                    'common_variables': sorted(
                        stats.variable_patterns.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                }
                for name, stats in events_by_frequency[:10]
            ]
        }
    
    def _analyze_validation_issues(self) -> Dict[str, Any]:
        """Analyze validation issue patterns.
        
        Returns:
            Validation issue analysis summary
        """
        if not self.issue_patterns:
            return {'total_issue_types': 0}
        
        # Sort by impact score
        patterns_by_impact = sorted(
            self.issue_patterns.items(),
            key=lambda x: x[1].impact_score,
            reverse=True
        )
        
        # Count by severity
        severity_counts = defaultdict(int)
        for pattern in self.issue_patterns.values():
            severity_counts[pattern.severity.value] += pattern.frequency
        
        # Total issues
        total_issues = sum(pattern.frequency for pattern in self.issue_patterns.values())
        
        return {
            'total_issue_types': len(self.issue_patterns),
            'total_issues': total_issues,
            'issues_per_page': total_issues / self.total_pages_processed if self.total_pages_processed > 0 else 0,
            'by_severity': dict(severity_counts),
            'highest_impact_issues': [
                {
                    'issue_type': pattern.issue_type,
                    'severity': pattern.severity.value,
                    'frequency': pattern.frequency,
                    'affected_pages': len(pattern.affected_pages),
                    'affected_variables': len(pattern.affected_variables),
                    'impact_score': pattern.impact_score,
                    'example_messages': pattern.example_messages[:2]
                }
                for _, pattern in patterns_by_impact[:10]
            ]
        }
    
    def get_variable_details(self, variable_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific variable.
        
        Args:
            variable_name: Name of variable to analyze
            
        Returns:
            Detailed variable information or None if not found
        """
        if variable_name not in self.variable_stats:
            return None
        
        stats = self.variable_stats[variable_name]
        
        return {
            'name': stats.name,
            'presence_count': stats.presence_count,
            'presence_rate': stats.presence_rate,
            'is_consistent': stats.is_consistent,
            'total_pages': stats.total_pages,
            'value_types': list(stats.value_types),
            'paths_seen': list(stats.paths_seen),
            'example_values': stats.example_values,
            'first_seen': stats.first_seen.isoformat() if stats.first_seen else None,
            'last_seen': stats.last_seen.isoformat() if stats.last_seen else None
        }
    
    def clear_data(self) -> None:
        """Clear all aggregated data."""
        self.variable_stats.clear()
        self.event_stats.clear()
        self.issue_patterns.clear()
        self.total_pages_processed = 0
        self.processed_urls.clear()
        self.run_started = None
        self.run_completed = None
        logger.debug("Aggregation data cleared")


class ValidationIssueAnalyzer:
    """Advanced validation issue analysis with categorization and trend detection."""
    
    def __init__(self, config: AggregationConfig | None = None):
        """Initialize validation issue analyzer.
        
        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregationConfig()
        self.issue_history: List[ValidationIssue] = []
        self.analysis_cache: Dict[str, Any] = {}
        self._last_analysis: Optional[datetime] = None
    
    @validate_types()
    def add_issues(self, issues: List[ValidationIssue], timestamp: Optional[datetime] = None) -> None:
        """Add validation issues for analysis.
        
        Args:
            issues: List of validation issues to add
            timestamp: When issues were captured
        """
        if not timestamp:
            timestamp = datetime.utcnow()
        
        # Add timestamp to each issue for trend analysis
        for issue in issues:
            if not hasattr(issue, '_analysis_timestamp'):
                issue._analysis_timestamp = timestamp
        
        self.issue_history.extend(issues)
        self._invalidate_cache()
    
    def analyze_issue_trends(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze trends in validation issues over time.
        
        Args:
            time_window_hours: Time window for trend analysis
            
        Returns:
            Trend analysis results
        """
        if not self.issue_history:
            return {'trends': [], 'summary': 'No issues to analyze'}
        
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        recent_issues = [
            issue for issue in self.issue_history
            if hasattr(issue, '_analysis_timestamp') and issue._analysis_timestamp >= cutoff_time
        ]
        
        if not recent_issues:
            return {'trends': [], 'summary': f'No issues in last {time_window_hours} hours'}
        
        # Group issues by hour
        hourly_counts = defaultdict(int)
        hourly_severity_counts = defaultdict(lambda: defaultdict(int))
        
        for issue in recent_issues:
            hour_key = issue._analysis_timestamp.strftime('%Y-%m-%d-%H')
            hourly_counts[hour_key] += 1
            hourly_severity_counts[hour_key][issue.severity.value] += 1
        
        # Calculate trend direction
        hours = sorted(hourly_counts.keys())
        if len(hours) >= 2:
            recent_avg = statistics.mean([hourly_counts[h] for h in hours[-3:]])
            earlier_avg = statistics.mean([hourly_counts[h] for h in hours[:-3]]) if len(hours) > 3 else hourly_counts[hours[0]]
            
            if recent_avg > earlier_avg * 1.2:
                trend_direction = "increasing"
            elif recent_avg < earlier_avg * 0.8:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "insufficient_data"
        
        return {
            'time_window_hours': time_window_hours,
            'total_issues': len(recent_issues),
            'trend_direction': trend_direction,
            'hourly_breakdown': dict(hourly_counts),
            'severity_trends': {
                hour: dict(severities) 
                for hour, severities in hourly_severity_counts.items()
            },
            'peak_hour': max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else None
        }
    
    def categorize_issues_by_impact(self) -> Dict[str, Any]:
        """Categorize validation issues by their business impact.
        
        Returns:
            Issue categorization by impact
        """
        if not self.issue_history:
            return {'categories': {}}
        
        # Impact categories based on severity and affected areas
        categories = {
            'critical_data_quality': [],
            'user_experience': [],
            'compliance_risk': [],
            'technical_debt': [],
            'minor_inconsistencies': []
        }
        
        # Categorization rules
        for issue in self.issue_history:
            # Critical data quality issues
            if (issue.severity == ValidationSeverity.CRITICAL or 
                any(keyword in issue.message.lower() for keyword in ['required', 'missing', 'null', 'undefined'])):
                categories['critical_data_quality'].append(issue)
                
            # User experience issues (tracking/analytics related)
            elif (issue.variable_name and 
                  any(keyword in issue.variable_name.lower() for keyword in ['user', 'customer', 'session', 'page'])):
                categories['user_experience'].append(issue)
                
            # Compliance risks (PII, sensitive data)
            elif (issue.variable_name and 
                  any(keyword in issue.variable_name.lower() for keyword in ['email', 'phone', 'ssn', 'payment', 'personal'])):
                categories['compliance_risk'].append(issue)
                
            # Technical debt (format, type issues)
            elif issue.schema_rule in ['format', 'type', 'pattern']:
                categories['technical_debt'].append(issue)
                
            # Everything else is minor
            else:
                categories['minor_inconsistencies'].append(issue)
        
        # Calculate category metrics
        category_metrics = {}
        total_issues = len(self.issue_history)
        
        for category, issues in categories.items():
            if issues:
                affected_pages = set(issue.page_url for issue in issues)
                affected_variables = set(issue.variable_name for issue in issues if issue.variable_name)
                
                category_metrics[category] = {
                    'count': len(issues),
                    'percentage': (len(issues) / total_issues) * 100,
                    'affected_pages': len(affected_pages),
                    'affected_variables': len(affected_variables),
                    'severity_breakdown': dict(Counter(issue.severity.value for issue in issues)),
                    'top_issues': [
                        {
                            'message': issue.message[:100],
                            'variable': issue.variable_name,
                            'page': issue.page_url,
                            'severity': issue.severity.value
                        }
                        for issue in sorted(issues, key=lambda x: x.severity.value == 'critical', reverse=True)[:3]
                    ]
                }
            else:
                category_metrics[category] = {
                    'count': 0,
                    'percentage': 0,
                    'affected_pages': 0,
                    'affected_variables': 0,
                    'severity_breakdown': {},
                    'top_issues': []
                }
        
        return {
            'total_issues': total_issues,
            'categories': category_metrics,
            'recommendation': self._generate_impact_recommendations(category_metrics)
        }
    
    def analyze_issue_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in validation issues for root cause identification.
        
        Returns:
            Pattern analysis results
        """
        if not self.issue_history:
            return {'patterns': []}
        
        # Pattern detection
        patterns = {
            'page_patterns': self._analyze_page_patterns(),
            'variable_patterns': self._analyze_variable_patterns(),
            'temporal_patterns': self._analyze_temporal_patterns(),
            'correlation_patterns': self._analyze_correlation_patterns()
        }
        
        # Generate insights
        insights = []
        
        # Page pattern insights
        if patterns['page_patterns']['problematic_pages']:
            insights.append({
                'type': 'page_concentration',
                'severity': 'high' if len(patterns['page_patterns']['problematic_pages']) < 5 else 'medium',
                'message': f"Issues concentrated on {len(patterns['page_patterns']['problematic_pages'])} pages",
                'detail': patterns['page_patterns']['problematic_pages'][:5]
            })
        
        # Variable pattern insights
        if patterns['variable_patterns']['problematic_variables']:
            insights.append({
                'type': 'variable_concentration',
                'severity': 'high' if len(patterns['variable_patterns']['problematic_variables']) < 10 else 'medium',
                'message': f"Issues concentrated in {len(patterns['variable_patterns']['problematic_variables'])} variables",
                'detail': patterns['variable_patterns']['problematic_variables'][:5]
            })
        
        # Temporal pattern insights
        if patterns['temporal_patterns']['peak_times']:
            insights.append({
                'type': 'temporal_clustering',
                'severity': 'medium',
                'message': f"Issues peak during: {', '.join(patterns['temporal_patterns']['peak_times'])}",
                'detail': patterns['temporal_patterns']
            })
        
        return {
            'patterns': patterns,
            'insights': insights,
            'recommendations': self._generate_pattern_recommendations(patterns, insights)
        }
    
    def _analyze_page_patterns(self) -> Dict[str, Any]:
        """Analyze validation issues by page patterns."""
        page_issue_counts = Counter(issue.page_url for issue in self.issue_history)
        total_pages = len(set(issue.page_url for issue in self.issue_history))
        
        # Identify problematic pages (pages with disproportionate issues)
        avg_issues_per_page = len(self.issue_history) / total_pages if total_pages > 0 else 0
        problematic_threshold = avg_issues_per_page * 2  # 2x average
        
        problematic_pages = [
            {'page': page, 'issue_count': count, 'ratio': count / avg_issues_per_page}
            for page, count in page_issue_counts.most_common()
            if count > problematic_threshold
        ]
        
        return {
            'total_pages': total_pages,
            'avg_issues_per_page': avg_issues_per_page,
            'problematic_pages': problematic_pages,
            'page_distribution': dict(page_issue_counts.most_common(10))
        }
    
    def _analyze_variable_patterns(self) -> Dict[str, Any]:
        """Analyze validation issues by variable patterns."""
        variable_issue_counts = Counter(
            issue.variable_name for issue in self.issue_history 
            if issue.variable_name
        )
        
        total_variables = len(variable_issue_counts)
        if total_variables == 0:
            return {'problematic_variables': [], 'variable_distribution': {}}
        
        # Identify problematic variables
        avg_issues_per_var = sum(variable_issue_counts.values()) / total_variables
        problematic_threshold = avg_issues_per_var * 2
        
        problematic_variables = [
            {'variable': var, 'issue_count': count, 'ratio': count / avg_issues_per_var}
            for var, count in variable_issue_counts.most_common()
            if count > problematic_threshold
        ]
        
        return {
            'total_variables': total_variables,
            'avg_issues_per_variable': avg_issues_per_var,
            'problematic_variables': problematic_variables,
            'variable_distribution': dict(variable_issue_counts.most_common(10))
        }
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in validation issues."""
        if not self.issue_history:
            return {'peak_times': []}
        
        # Group by hour of day
        hour_counts = defaultdict(int)
        day_counts = defaultdict(int)
        
        for issue in self.issue_history:
            if hasattr(issue, '_analysis_timestamp'):
                timestamp = issue._analysis_timestamp
                hour_counts[timestamp.hour] += 1
                day_counts[timestamp.strftime('%A')] += 1
        
        # Find peak hours and days
        peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_days = sorted(day_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        peak_times = []
        if peak_hours:
            peak_times.extend([f"{hour}:00" for hour, _ in peak_hours])
        if peak_days:
            peak_times.extend([day for day, _ in peak_days])
        
        return {
            'peak_times': peak_times,
            'hourly_distribution': dict(hour_counts),
            'daily_distribution': dict(day_counts)
        }
    
    def _analyze_correlation_patterns(self) -> Dict[str, Any]:
        """Analyze correlations between different types of issues."""
        # Group issues by rule type and severity combinations
        rule_severity_combos = Counter(
            f"{issue.schema_rule}:{issue.severity.value}"
            for issue in self.issue_history
            if issue.schema_rule
        )
        
        # Find frequently co-occurring issue types on the same pages
        page_issue_types = defaultdict(set)
        for issue in self.issue_history:
            if issue.schema_rule:
                page_issue_types[issue.page_url].add(issue.schema_rule)
        
        # Find rule combinations that appear together frequently
        common_combinations = defaultdict(int)
        for page, rule_types in page_issue_types.items():
            if len(rule_types) > 1:
                rule_list = sorted(rule_types)
                for i, rule1 in enumerate(rule_list):
                    for rule2 in rule_list[i+1:]:
                        common_combinations[f"{rule1}+{rule2}"] += 1
        
        return {
            'rule_severity_combinations': dict(rule_severity_combos.most_common(10)),
            'common_rule_combinations': dict(common_combinations.most_common(5))
        }
    
    def _generate_impact_recommendations(self, category_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on impact categorization."""
        recommendations = []
        
        # Critical data quality recommendations
        critical_count = category_metrics.get('critical_data_quality', {}).get('count', 0)
        if critical_count > 0:
            recommendations.append(
                f"URGENT: Address {critical_count} critical data quality issues that may break analytics tracking"
            )
        
        # Compliance risk recommendations
        compliance_count = category_metrics.get('compliance_risk', {}).get('count', 0)
        if compliance_count > 0:
            recommendations.append(
                f"HIGH PRIORITY: Review {compliance_count} compliance-related issues for privacy/security risks"
            )
        
        # User experience recommendations
        ux_count = category_metrics.get('user_experience', {}).get('count', 0)
        if ux_count > 5:  # Threshold for concern
            recommendations.append(
                f"MEDIUM PRIORITY: {ux_count} user experience issues may impact analytics accuracy"
            )
        
        return recommendations
    
    def _generate_pattern_recommendations(
        self,
        patterns: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on pattern analysis."""
        recommendations = []
        
        for insight in insights:
            if insight['type'] == 'page_concentration' and insight['severity'] == 'high':
                recommendations.append(
                    "Focus debugging efforts on the most problematic pages to maximize impact"
                )
            elif insight['type'] == 'variable_concentration':
                recommendations.append(
                    "Review data layer implementation for the most problematic variables"
                )
            elif insight['type'] == 'temporal_clustering':
                recommendations.append(
                    "Investigate if temporal issue patterns correlate with deployment or traffic patterns"
                )
        
        return recommendations
    
    def generate_issue_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation issue analysis report.
        
        Returns:
            Complete issue analysis report
        """
        cache_key = "full_report"
        
        # Check if we have a recent cached report
        if (cache_key in self.analysis_cache and 
            self._last_analysis and 
            datetime.utcnow() - self._last_analysis < timedelta(minutes=5)):
            return self.analysis_cache[cache_key]
        
        # Generate fresh analysis
        trend_analysis = self.analyze_issue_trends()
        impact_analysis = self.categorize_issues_by_impact()
        pattern_analysis = self.analyze_issue_patterns()
        
        report = {
            'summary': {
                'total_issues': len(self.issue_history),
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'data_quality_score': self._calculate_data_quality_score(impact_analysis)
            },
            'trends': trend_analysis,
            'impact_analysis': impact_analysis,
            'pattern_analysis': pattern_analysis,
            'actionable_recommendations': self._compile_all_recommendations(
                impact_analysis, pattern_analysis
            )
        }
        
        # Cache the report
        self.analysis_cache[cache_key] = report
        self._last_analysis = datetime.utcnow()
        
        return report
    
    def _calculate_data_quality_score(self, impact_analysis: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100).
        
        Args:
            impact_analysis: Impact analysis results
            
        Returns:
            Data quality score
        """
        if not impact_analysis.get('categories'):
            return 100.0  # Perfect score if no issues
        
        categories = impact_analysis['categories']
        total_issues = impact_analysis['total_issues']
        
        if total_issues == 0:
            return 100.0
        
        # Weight different categories
        weighted_issues = (
            categories.get('critical_data_quality', {}).get('count', 0) * 5 +
            categories.get('compliance_risk', {}).get('count', 0) * 4 +
            categories.get('user_experience', {}).get('count', 0) * 3 +
            categories.get('technical_debt', {}).get('count', 0) * 2 +
            categories.get('minor_inconsistencies', {}).get('count', 0) * 1
        )
        
        # Calculate score (higher weighted issues = lower score)
        max_possible_weighted = total_issues * 5  # All critical
        if max_possible_weighted == 0:
            return 100.0
        
        score = max(0, 100 - (weighted_issues / max_possible_weighted * 100))
        return round(score, 1)
    
    def _compile_all_recommendations(
        self,
        impact_analysis: Dict[str, Any],
        pattern_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compile all recommendations with priorities.
        
        Args:
            impact_analysis: Impact analysis results
            pattern_analysis: Pattern analysis results
            
        Returns:
            Prioritized recommendations list
        """
        recommendations = []
        
        # Impact-based recommendations
        for rec in impact_analysis.get('recommendation', []):
            recommendations.append({
                'type': 'impact_based',
                'priority': 'high' if 'URGENT' in rec else 'medium' if 'HIGH' in rec else 'low',
                'message': rec,
                'source': 'impact_analysis'
            })
        
        # Pattern-based recommendations
        for rec in pattern_analysis.get('recommendations', []):
            recommendations.append({
                'type': 'pattern_based',
                'priority': 'medium',
                'message': rec,
                'source': 'pattern_analysis'
            })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations
    
    def _invalidate_cache(self) -> None:
        """Invalidate analysis cache when new data is added."""
        self.analysis_cache.clear()
        self._last_analysis = None
    
    def clear_history(self) -> None:
        """Clear all issue history and analysis cache."""
        self.issue_history.clear()
        self._invalidate_cache()
        logger.debug("Validation issue analysis data cleared")