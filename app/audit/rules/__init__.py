"""Rules & Alerts system for Tag Sentinel.

This package provides a YAML-driven rule engine that evaluates captured artifacts,
generates alerts, and provides CI-friendly exit codes for automated quality gates.
"""

from .models import (
    # Enums
    Severity,
    RuleScope,
    CheckType,
    AlertChannelType,
    
    # Core models
    AppliesTo,
    CheckConfig,
    Rule,
    Failure,
    RuleSummary,
    RuleResults,
    AlertConfig,
    AlertPayload,
)

from .schema import (
    # Schema utilities
    SchemaVersion,
    ValidationResult,
    get_schema_v0_1,
    get_schema_by_version,
    validate_rules_config,
    validate_rules_yaml,
    validate_rules_file,
    generate_example_config,
    export_schema_to_file,
    export_example_to_file,
)

from .parser import (
    # Parser utilities
    ParseError,
    EnvironmentInterpolator,
    RegexCompiler,
    RuleParser,
    load_rules_from_file,
    load_rules_from_yaml,
    validate_and_load_rules,
    
    # Configuration management
    ConfigurationManager,
    AdvancedRuleParser,
    load_rules_for_environment,
    validate_configuration_chain,
    get_available_environments,
    create_environment_override_config,
)

from .indexing import (
    # Index models
    TimelineEntry,
    RequestIndex,
    CookieIndex,
    EventIndex,
    PageIndex,
    RunSummary,
    AuditIndexes,
    
    # Index builder
    IndexBuilder,
    build_audit_indexes,
    build_page_index,
    build_run_summary,
    
    # Query interface
    QueryFilter,
    QueryBuilder,
    QueryAggregator,
    AuditQuery,
)

from .checks import (
    # Core check abstractions
    BaseCheck,
    CheckContext,
    CheckResult,
    CheckRegistry,
    
    # Registry and decorators
    check_registry,
    register_check,
    
    # Presence/absence checks
    RequestPresentCheck,
    RequestAbsentCheck,
    CookiePresentCheck,
    TagEventPresentCheck,
    ConsoleMessagePresentCheck,
    
    # Duplicate detection checks
    RequestDuplicateCheck,
    EventDuplicateCheck,
    CookieDuplicateCheck,
    DuplicateGroup,
    
    # Temporal and sequencing checks
    LoadTimingCheck,
    SequenceOrderCheck,
    RelativeTimingCheck,
    SequenceOrderType,
    TimingComparison,
    SequenceItem,
    
    # Privacy and compliance checks
    GDPRComplianceCheck,
    CCPAComplianceCheck,
    CookieSecurityCheck,
    PrivacyRegulation,
    
    # Expression-based checks
    ExpressionCheck,
    JSONPathCheck,
    SafeExpressionEvaluator,
    SimpleJSONPath,
    SafeExpressionError,
    JSONPathError,
)

from .evaluator import (
    # Core evaluation engine
    RuleEvaluationEngine,
    RuleEvaluationOrchestrator,
    EvaluationContext,
    RuleEvaluationResult,
    
    # Convenience functions
    orchestrator,
    evaluate_rules,
    evaluate_from_config,
)

from .alerts import (
    # Alert status and severity enums
    AlertStatus,
    AlertSeverity,
    AlertTrigger,
    
    # Core alert classes
    AlertContext,
    AlertPayload,
    AlertTemplate,
    AlertDispatchResult,
    
    # Base dispatcher framework
    BaseAlertDispatcher,
    AlertDispatcherRegistry,
    
    # Registry and decorators
    dispatcher_registry,
    register_dispatcher,
)

from .reporting import (
    # Report configuration and formats
    ReportFormat,
    ReportLevel,
    ReportConfig,
    SortBy,
    
    # Report data models
    ReportData,
    
    # Report generators
    ReportGenerator,
    BaseReportGenerator,
    JSONReportGenerator,
    HTMLReportGenerator,
    TextReportGenerator,
    CSVReportGenerator,
    MarkdownReportGenerator,
    
    # Convenience functions
    generate_report,
    generate_html_report,
    generate_json_report,
)

from .metrics import (
    # Metric types and enums
    MetricType,
    AlertMetricType,
    RuleMetricType,
    SystemMetricType,
    
    # Core metrics classes
    MetricPoint,
    MetricSummary,
    MetricsCollector,
    
    # Specialized metrics
    RuleEngineMetrics,
    AlertMetrics,
    MetricsReporter,
    
    # Global accessors
    get_metrics_collector,
    get_rule_metrics,
    get_alert_metrics,
    create_metrics_reporter,
    
    # Utilities
    timer_metric,
    counter_metric,
)

__all__ = [
    # Enums
    'Severity',
    'RuleScope', 
    'CheckType',
    'AlertChannelType',
    
    # Core models
    'AppliesTo',
    'CheckConfig',
    'Rule',
    'Failure',
    'RuleSummary',
    'RuleResults',
    'AlertConfig',
    'AlertPayload',
    
    # Schema utilities
    'SchemaVersion',
    'ValidationResult',
    'get_schema_v0_1',
    'get_schema_by_version',
    'validate_rules_config',
    'validate_rules_yaml',
    'validate_rules_file',
    'generate_example_config',
    'export_schema_to_file',
    'export_example_to_file',
    
    # Parser utilities
    'ParseError',
    'EnvironmentInterpolator',
    'RegexCompiler',
    'RuleParser',
    'load_rules_from_file',
    'load_rules_from_yaml',
    'validate_and_load_rules',
    
    # Configuration management
    'ConfigurationManager',
    'AdvancedRuleParser',
    'load_rules_for_environment',
    'validate_configuration_chain',
    'get_available_environments',
    'create_environment_override_config',
    
    # Index models
    'TimelineEntry',
    'RequestIndex',
    'CookieIndex',
    'EventIndex',
    'PageIndex',
    'RunSummary',
    'AuditIndexes',
    
    # Index builder
    'IndexBuilder',
    'build_audit_indexes',
    'build_page_index',
    'build_run_summary',
    
    # Query interface
    'QueryFilter',
    'QueryBuilder',
    'QueryAggregator',
    'AuditQuery',
    
    # Core check abstractions
    'BaseCheck',
    'CheckContext',
    'CheckResult',
    'CheckRegistry',
    
    # Registry and decorators
    'check_registry',
    'register_check',
    
    # Presence/absence checks
    'RequestPresentCheck',
    'RequestAbsentCheck',
    'CookiePresentCheck',
    'TagEventPresentCheck',
    'ConsoleMessagePresentCheck',
    
    # Duplicate detection checks
    'RequestDuplicateCheck',
    'EventDuplicateCheck',
    'CookieDuplicateCheck',
    'DuplicateGroup',
    
    # Temporal and sequencing checks
    'LoadTimingCheck',
    'SequenceOrderCheck',
    'RelativeTimingCheck',
    'SequenceOrderType',
    'TimingComparison',
    'SequenceItem',
    
    # Privacy and compliance checks
    'GDPRComplianceCheck',
    'CCPAComplianceCheck',
    'CookieSecurityCheck',
    'PrivacyRegulation',
    
    # Expression-based checks
    'ExpressionCheck',
    'JSONPathCheck',
    'SafeExpressionEvaluator',
    'SimpleJSONPath',
    'SafeExpressionError',
    'JSONPathError',
    
    # Core evaluation engine
    'RuleEvaluationEngine',
    'RuleEvaluationOrchestrator',
    'EvaluationContext',
    'RuleEvaluationResult',
    
    # Convenience functions
    'orchestrator',
    'evaluate_rules',
    'evaluate_from_config',
    
    # Alert status and severity enums
    'AlertStatus',
    'AlertSeverity',
    'AlertTrigger',
    
    # Core alert classes
    'AlertContext',
    'AlertPayload',
    'AlertTemplate',
    'AlertDispatchResult',
    
    # Base dispatcher framework
    'BaseAlertDispatcher',
    'AlertDispatcherRegistry',
    
    # Registry and decorators
    'dispatcher_registry',
    'register_dispatcher',
    
    # Report configuration and formats
    'ReportFormat',
    'ReportLevel',
    'ReportConfig',
    'SortBy',
    
    # Report data models
    'ReportData',
    
    # Report generators
    'ReportGenerator',
    'BaseReportGenerator',
    'JSONReportGenerator',
    'HTMLReportGenerator',
    'TextReportGenerator',
    'CSVReportGenerator',
    'MarkdownReportGenerator',
    
    # Convenience functions
    'generate_report',
    'generate_html_report',
    'generate_json_report',
    
    # Metric types and enums
    'MetricType',
    'AlertMetricType',
    'RuleMetricType',
    'SystemMetricType',
    
    # Core metrics classes
    'MetricPoint',
    'MetricSummary',
    'MetricsCollector',
    
    # Specialized metrics
    'RuleEngineMetrics',
    'AlertMetrics',
    'MetricsReporter',
    
    # Global accessors
    'get_metrics_collector',
    'get_rule_metrics',
    'get_alert_metrics',
    'create_metrics_reporter',
    
    # Utilities
    'timer_metric',
    'counter_metric',
]

__version__ = "0.1.0"