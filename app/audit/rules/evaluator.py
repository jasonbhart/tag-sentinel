"""Core rule evaluation engine for orchestrating rule execution.

This module provides the main rule evaluation orchestrator that coordinates
rule filtering, check execution, result aggregation, and reporting across
all available check types and rule configurations. Performance optimized
for large-scale rule evaluation with parallel processing.
"""

import asyncio
import gc
import logging
import psutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from datetime import datetime
from functools import lru_cache
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from .models import Rule, RuleResults, RuleSummary, Failure, Severity
from .indexing import AuditIndexes, AuditQuery, PerformanceMonitor, BatchProcessor
from .checks import (
    BaseCheck, CheckContext, CheckResult, check_registry,
    # Import all check types for registration
    RequestPresentCheck, RequestAbsentCheck, CookiePresentCheck,
    TagEventPresentCheck, ConsoleMessagePresentCheck,
    RequestDuplicateCheck, EventDuplicateCheck, CookieDuplicateCheck,
    LoadTimingCheck, SequenceOrderCheck, RelativeTimingCheck,
    GDPRComplianceCheck, CCPAComplianceCheck, CookieSecurityCheck,
    ExpressionCheck, JSONPathCheck
)


logger = logging.getLogger(__name__)


class EvaluationMetrics(BaseModel):
    """Performance metrics for rule evaluation."""
    
    total_rules: int = 0
    evaluated_rules: int = 0
    failed_rules: int = 0
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_ms: float = 0
    
    parallel_workers_used: int = 0
    memory_usage_mb: float = 0
    cpu_usage_percent: float = 0
    
    # Per-rule timing
    min_rule_time_ms: float = 0
    max_rule_time_ms: float = 0
    avg_rule_time_ms: float = 0
    
    # Resource utilization
    peak_memory_mb: float = 0
    avg_cpu_percent: float = 0


@dataclass
class EvaluationContext:
    """Context for rule evaluation execution."""
    
    indexes: AuditIndexes
    query: AuditQuery
    environment: Optional[str] = None
    target_urls: List[str] = field(default_factory=list)
    debug: bool = False
    timeout_seconds: int = 300
    max_workers: Optional[int] = None
    
    # Evaluation options
    fail_fast: bool = False
    include_passed: bool = True
    severity_filter: Optional[Set[Severity]] = None
    rule_tag_filter: Optional[Set[str]] = None
    
    # Performance tracking
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_rules_evaluated: int = 0
    total_checks_executed: int = 0


@dataclass
class RuleEvaluationResult:
    """Result of evaluating a single rule."""
    
    rule: Rule
    check_results: List[CheckResult]
    passed: bool
    failures: List[Failure]
    execution_time_ms: float
    error: Optional[str] = None
    
    @property
    def severity(self) -> Severity:
        """Get the highest severity from failures."""
        if not self.failures:
            return Severity.INFO
        
        severity_order = {
            Severity.INFO: 0,
            Severity.WARNING: 1,
            Severity.CRITICAL: 2
        }
        
        return max(self.failures, key=lambda f: severity_order[f.severity]).severity


class OptimizedRuleEvaluationEngine:
    """High-performance rule evaluation engine with optimizations."""
    
    def __init__(self, enable_profiling: bool = False):
        self.check_cache: Dict[str, BaseCheck] = {}
        self.evaluation_hooks: Dict[str, List[Callable]] = defaultdict(list)
        self.performance_monitor = PerformanceMonitor()
        self.batch_processor = BatchProcessor()
        self.enable_profiling = enable_profiling
        self.metrics = EvaluationMetrics()
        
        # Performance tuning parameters
        self.optimal_batch_size = self._calculate_optimal_batch_size()
        self.max_parallel_workers = min(cpu_count(), 8)  # Reasonable upper bound
        
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on system resources."""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_cores = cpu_count()
        
        # Base batch size on memory and CPU cores
        if memory_gb >= 16:
            base_batch_size = 500
        elif memory_gb >= 8:
            base_batch_size = 250
        else:
            base_batch_size = 100
            
        # Adjust for CPU cores
        return min(base_batch_size, cpu_cores * 50)
    
    def _monitor_resource_usage(self):
        """Monitor system resource usage during evaluation."""
        if self.enable_profiling:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_percent = process.cpu_percent()
            
            self.metrics.memory_usage_mb = memory_mb
            self.metrics.cpu_usage_percent = cpu_percent
            
            # Track peaks
            self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, memory_mb)
            
    @lru_cache(maxsize=128)
    def _get_cached_check_instance(self, check_type: str) -> Optional[BaseCheck]:
        """Get cached check instance for performance."""
        return check_registry.get_check_class(check_type)


class RuleEvaluationEngine(OptimizedRuleEvaluationEngine):
    """Core rule evaluation orchestrator with backward compatibility."""
    
    def __init__(self):
        super().__init__()
        # Maintain backward compatibility
        self.check_cache: Dict[str, BaseCheck] = {}
        self.evaluation_hooks: Dict[str, List[Callable]] = defaultdict(list)
    
    def register_hook(self, hook_name: str, hook_func: Callable) -> None:
        """Register evaluation hook for extensibility.
        
        Args:
            hook_name: Name of hook (pre_evaluation, post_evaluation, etc.)
            hook_func: Function to call at hook point
        """
        self.evaluation_hooks[hook_name].append(hook_func)
    
    def evaluate_rules(
        self,
        rules: List[Rule],
        context: EvaluationContext
    ) -> RuleResults:
        """Evaluate a list of rules against audit data.
        
        Args:
            rules: List of rules to evaluate
            context: Evaluation context and options
            
        Returns:
            Aggregated rule evaluation results
        """
        context.start_time = datetime.utcnow()
        
        try:
            # Execute pre-evaluation hooks
            self._execute_hooks('pre_evaluation', rules, context)
            
            # Filter rules based on context
            applicable_rules = self._filter_rules(rules, context)
            context.total_rules_evaluated = len(applicable_rules)
            
            # Evaluate rules with optimized parallelization
            if context.max_workers and context.max_workers > 1:
                rule_results = self._evaluate_rules_optimized_parallel(applicable_rules, context)
            else:
                rule_results = self._evaluate_rules_sequential(applicable_rules, context)
            
            # Aggregate results
            aggregated_results = self._aggregate_results(rule_results, context)
            
            # Execute post-evaluation hooks
            self._execute_hooks('post_evaluation', aggregated_results, context)
            
            context.end_time = datetime.utcnow()
            return aggregated_results
            
        except Exception as e:
            context.end_time = datetime.utcnow()
            # Create error result
            return self._create_error_result(str(e), context)
    
    def evaluate_single_rule(
        self,
        rule: Rule,
        context: EvaluationContext
    ) -> RuleEvaluationResult:
        """Evaluate a single rule.
        
        Args:
            rule: Rule to evaluate
            context: Evaluation context
            
        Returns:
            Single rule evaluation result
        """
        start_time = time.time()
        
        try:
            # Check if rule applies to current context
            if not self._rule_applies(rule, context):
                return RuleEvaluationResult(
                    rule=rule,
                    check_results=[],
                    passed=True,
                    failures=[],
                    execution_time_ms=0,
                    error="Rule does not apply to current context"
                )
            
            # Create check instance
            check = self._get_or_create_check(rule)
            
            # Create check context
            check_context = self._create_check_context(rule, context)
            
            # Execute check
            check_result = check.execute(check_context)
            context.total_checks_executed += 1
            
            # Convert to rule result
            failures = [check_result.to_failure()] if not check_result.passed else []
            failures = [f for f in failures if f is not None]
            
            execution_time = (time.time() - start_time) * 1000
            
            return RuleEvaluationResult(
                rule=rule,
                check_results=[check_result],
                passed=check_result.passed,
                failures=failures,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return RuleEvaluationResult(
                rule=rule,
                check_results=[],
                passed=False,
                failures=[],
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    def _filter_rules(self, rules: List[Rule], context: EvaluationContext) -> List[Rule]:
        """Filter rules based on evaluation context."""
        filtered = []
        
        for rule in rules:
            # Check environment filter
            if (context.environment and 
                rule.applies_to.environments and 
                context.environment not in rule.applies_to.environments):
                continue
            
            # Check URL patterns
            if rule.applies_to.url_include or rule.applies_to.url_exclude:
                if not self._url_matches(context.target_urls, rule.applies_to):
                    continue
            
            # Check severity filter
            if context.severity_filter and rule.severity not in context.severity_filter:
                continue
            
            # Check tag filter
            if context.rule_tag_filter:
                rule_tags = set(rule.tags or [])
                if not rule_tags.intersection(context.rule_tag_filter):
                    continue
            
            filtered.append(rule)
        
        return filtered
    
    def _evaluate_rules_sequential(
        self,
        rules: List[Rule],
        context: EvaluationContext
    ) -> List[RuleEvaluationResult]:
        """Evaluate rules sequentially."""
        results = []
        
        for rule in rules:
            result = self.evaluate_single_rule(rule, context)
            results.append(result)
            
            # Fail fast if enabled and rule failed
            if context.fail_fast and not result.passed:
                break
        
        return results
    
    def _evaluate_rules_parallel(
        self,
        rules: List[Rule],
        context: EvaluationContext
    ) -> List[RuleEvaluationResult]:
        """Evaluate rules in parallel using thread pool."""
        results = []
        max_workers = context.max_workers or min(len(rules), 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all rules
            future_to_rule = {
                executor.submit(self.evaluate_single_rule, rule, context): rule
                for rule in rules
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_rule, timeout=context.timeout_seconds):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Fail fast if enabled and rule failed
                    if context.fail_fast and not result.passed:
                        # Cancel remaining futures
                        for f in future_to_rule:
                            if not f.done():
                                f.cancel()
                        break
                        
                except Exception as e:
                    rule = future_to_rule[future]
                    results.append(RuleEvaluationResult(
                        rule=rule,
                        check_results=[],
                        passed=False,
                        failures=[],
                        execution_time_ms=0,
                        error=f"Parallel execution error: {str(e)}"
                    ))
        
        return results
    
    def _evaluate_rules_optimized_parallel(
        self,
        rules: List[Rule],
        context: EvaluationContext
    ) -> List[RuleEvaluationResult]:
        """Optimized parallel evaluation with batching and resource monitoring."""
        if not rules:
            return []
        
        # Initialize metrics
        start_time = time.time()
        self.metrics.total_rules = len(rules)
        self.metrics.start_time = datetime.utcnow()
        
        # Determine optimal parallelization strategy
        total_rules = len(rules)
        max_workers = min(
            context.max_workers or self.max_parallel_workers,
            total_rules,
            cpu_count()
        )
        
        self.metrics.parallel_workers_used = max_workers
        
        # For small rule sets, use sequential evaluation
        if total_rules < 10 or max_workers == 1:
            return self._evaluate_rules_sequential(rules, context)
        
        # Split rules into batches for better load balancing
        batch_size = max(1, total_rules // max_workers)
        rule_batches = [
            rules[i:i + batch_size] 
            for i in range(0, total_rules, batch_size)
        ]
        
        results = []
        rule_times = []
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch evaluation jobs
            future_to_batch = {}
            for batch in rule_batches:
                future = executor.submit(self._evaluate_rule_batch, batch, context)
                future_to_batch[future] = batch
            
            # Collect results with monitoring
            for future in as_completed(future_to_batch, timeout=context.timeout_seconds):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    
                    # Monitor resource usage
                    self._monitor_resource_usage()
                    
                    # Collect timing metrics
                    for result in batch_results:
                        rule_times.append(result.execution_time_ms)
                    
                    # Fail fast if enabled and critical failure found
                    if context.fail_fast:
                        critical_failures = [
                            r for r in batch_results 
                            if not r.passed and any(
                                f.severity == Severity.CRITICAL 
                                for f in r.failures
                            )
                        ]
                        if critical_failures:
                            # Cancel remaining batches
                            for f in future_to_batch:
                                if not f.done():
                                    f.cancel()
                            break
                    
                except Exception as e:
                    batch = future_to_batch[future]
                    logger.error(f"Batch evaluation failed: {e}")
                    
                    # Create error results for failed batch
                    for rule in batch:
                        results.append(RuleEvaluationResult(
                            rule=rule,
                            check_results=[],
                            passed=False,
                            failures=[],
                            execution_time_ms=0,
                            error=f"Batch evaluation error: {str(e)}"
                        ))
        
        # Update metrics
        end_time = time.time()
        self.metrics.end_time = datetime.utcnow()
        self.metrics.total_duration_ms = (end_time - start_time) * 1000
        self.metrics.evaluated_rules = len(results)
        self.metrics.failed_rules = len([r for r in results if not r.passed])
        
        if rule_times:
            self.metrics.min_rule_time_ms = min(rule_times)
            self.metrics.max_rule_time_ms = max(rule_times)
            self.metrics.avg_rule_time_ms = sum(rule_times) / len(rule_times)
        
        # Trigger garbage collection after large evaluation
        if total_rules > 100:
            gc.collect()
        
        return results
    
    def _evaluate_rule_batch(
        self,
        rule_batch: List[Rule],
        context: EvaluationContext
    ) -> List[RuleEvaluationResult]:
        """Evaluate a batch of rules sequentially within a worker."""
        batch_results = []
        
        for rule in rule_batch:
            try:
                result = self.evaluate_single_rule(rule, context)
                batch_results.append(result)
                
                # Early exit for fail-fast mode
                if (context.fail_fast and not result.passed and 
                    any(f.severity == Severity.CRITICAL for f in result.failures)):
                    break
                    
            except Exception as e:
                batch_results.append(RuleEvaluationResult(
                    rule=rule,
                    check_results=[],
                    passed=False,
                    failures=[],
                    execution_time_ms=0,
                    error=f"Rule evaluation error: {str(e)}"
                ))
        
        return batch_results
    
    def get_evaluation_metrics(self) -> EvaluationMetrics:
        """Get performance metrics from the last evaluation."""
        return self.metrics
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = EvaluationMetrics()
    
    def _aggregate_results(
        self,
        rule_results: List[RuleEvaluationResult],
        context: EvaluationContext
    ) -> RuleResults:
        """Aggregate individual rule results into summary."""
        all_failures = []
        passed_count = 0
        failed_count = 0
        total_execution_time = 0
        
        severity_counts = {
            Severity.INFO: 0,
            Severity.WARNING: 0,
            Severity.CRITICAL: 0
        }
        
        for result in rule_results:
            if result.passed:
                passed_count += 1
            else:
                failed_count += 1
            
            all_failures.extend(result.failures)
            total_execution_time += result.execution_time_ms
            
            # Count by severity
            if result.failures:
                for failure in result.failures:
                    severity_counts[failure.severity] += 1
        
        # Create summary
        summary = RuleSummary(
            total_rules=len(rule_results),
            passed_rules=passed_count,
            failed_rules=failed_count,
            total_failures=len(all_failures),
            critical_failures=severity_counts[Severity.CRITICAL],
            warning_failures=severity_counts[Severity.WARNING],
            info_failures=severity_counts[Severity.INFO],
            execution_time_ms=total_execution_time
        )
        
        # Filter results if requested
        if not context.include_passed:
            rule_results = [r for r in rule_results if not r.passed]
        
        return RuleResults(
            summary=summary,
            failures=all_failures,
            evaluation_time=context.end_time or datetime.utcnow(),
            context_info={
                'environment': context.environment,
                'target_urls': context.target_urls,
                'total_checks_executed': context.total_checks_executed,
                'parallel_execution': context.max_workers is not None and context.max_workers > 1
            }
        )
    
    def _rule_applies(self, rule: Rule, context: EvaluationContext) -> bool:
        """Check if rule applies to current evaluation context."""
        applies_to = rule.applies_to
        
        # Check environment
        if applies_to.environments:
            if not context.environment or context.environment not in applies_to.environments:
                return False
        
        # Check URL patterns
        if applies_to.url_include or applies_to.url_exclude:
            if not self._url_matches(context.target_urls, applies_to):
                return False
        
        return True
    
    def _url_matches(self, urls: List[str], applies_to) -> bool:
        """Check if URLs match the applies_to criteria."""
        if not urls:
            return True
        
        # Check include patterns
        if applies_to.url_include:
            import re
            include_match = False
            for url in urls:
                for pattern in applies_to.url_include:
                    if re.search(pattern, url):
                        include_match = True
                        break
                if include_match:
                    break
            
            if not include_match:
                return False
        
        # Check exclude patterns
        if applies_to.url_exclude:
            import re
            for url in urls:
                for pattern in applies_to.url_exclude:
                    if re.search(pattern, url):
                        return False
        
        return True
    
    def _validate_concrete_check_type(self, check_type: str) -> str:
        """Validate and map check type aliases to concrete registered types."""
        # Map enum aliases to registered check types
        type_mapping = {
            # Duplicate checks
            "duplicate_requests": "request_duplicates",

            # Temporal checks
            "relative_order": "relative_timing",  # Keep for backward compatibility

            # Privacy checks - map generic to specific
            "cookie_policy": "cookie_security",  # Default to security check

            # Script presence - delegate to request_present with script filter
            "script_present": "request_present",
        }

        # Return mapped type or original if no mapping needed
        concrete_type = type_mapping.get(check_type, check_type)

        # Validate that the concrete type is actually registered
        if not check_registry.get_check_class(concrete_type):
            available_types = list(check_registry._checks.keys())
            raise ValueError(f"Unknown check type: {check_type} (mapped to {concrete_type}). Available types: {available_types}")

        return concrete_type
    
    def _get_or_create_check(self, rule: Rule) -> BaseCheck:
        """Get or create check instance for rule."""
        cache_key = f"{rule.check.type}:{rule.id}"

        if cache_key not in self.check_cache:
            # Validate concrete check type
            concrete_check_id = self._validate_concrete_check_type(rule.check.type)
            check_class = check_registry.get_check_class(concrete_check_id)
            if not check_class:
                raise ValueError(f"Unknown check type: {rule.check.type} (mapped to {concrete_check_id})")

            check_instance = check_class(
                check_id=rule.id,
                name=rule.name,
                description=rule.description
            )

            # Apply special configuration for aliased check types
            final_config = self._apply_alias_config_defaults(rule.check.type, rule.check.config)

            # Validate configuration
            config_errors = check_instance.validate_config(final_config)
            if config_errors:
                raise ValueError(f"Invalid check configuration: {'; '.join(config_errors)}")

            self.check_cache[cache_key] = check_instance

        return self.check_cache[cache_key]

    def _apply_alias_config_defaults(self, original_check_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default configuration for aliased check types."""
        final_config = dict(config) if config else {}

        # Apply defaults based on the original check type
        if original_check_type == "script_present":
            # Default to looking for script resources if not specified
            if 'resource_type' not in final_config:
                final_config['resource_type'] = 'script'

        return final_config
    
    def _create_check_context(self, rule: Rule, context: EvaluationContext) -> CheckContext:
        """Create check context for rule execution."""
        return CheckContext(
            indexes=context.indexes,
            query=context.query,
            rule_id=rule.id,
            rule_config={
                'severity': rule.severity.value,
                'tags': rule.tags or [],
                'environment': context.environment
            },
            check_config=self._apply_alias_config_defaults(rule.check.type, rule.check.config),
            environment=context.environment,
            target_urls=context.target_urls,
            debug=context.debug,
            timeout_ms=context.timeout_seconds * 1000
        )
    
    def _execute_hooks(self, hook_name: str, *args) -> None:
        """Execute registered hooks."""
        for hook_func in self.evaluation_hooks.get(hook_name, []):
            try:
                hook_func(*args)
            except Exception:
                # Ignore hook errors to avoid disrupting evaluation
                pass
    
    def _create_error_result(self, error_message: str, context: EvaluationContext) -> RuleResults:
        """Create error result for failed evaluations."""
        summary = RuleSummary(
            total_rules=0,
            passed_rules=0,
            failed_rules=0,
            total_failures=0,
            critical_failures=0,
            warning_failures=0,
            info_failures=0,
            execution_time_ms=0
        )
        
        return RuleResults(
            summary=summary,
            failures=[],
            evaluation_time=context.end_time or datetime.utcnow(),
            context_info={
                'error': error_message,
                'environment': context.environment,
                'target_urls': context.target_urls
            }
        )


class RuleEvaluationOrchestrator:
    """High-level orchestrator for rule evaluation workflows."""
    
    def __init__(self):
        self.engine = RuleEvaluationEngine()
        self.default_context = EvaluationContext(
            indexes=None,  # Must be provided
            query=None     # Must be provided
        )
    
    def evaluate_from_config(
        self,
        config_path: str,
        indexes: AuditIndexes,
        environment: Optional[str] = None,
        **kwargs
    ) -> RuleResults:
        """Evaluate rules from configuration file.
        
        Args:
            config_path: Path to rule configuration file
            indexes: Audit indexes to evaluate against
            environment: Target environment for rule filtering
            **kwargs: Additional evaluation options
            
        Returns:
            Rule evaluation results
        """
        from .parser import load_rules_from_file
        
        # Load rules from configuration
        rules = load_rules_from_file(config_path)
        
        # Create evaluation context
        context = EvaluationContext(
            indexes=indexes,
            query=AuditQuery(indexes),
            environment=environment,
            **kwargs
        )
        
        # Evaluate rules
        return self.engine.evaluate_rules(rules, context)
    
    def evaluate_rules_for_environment(
        self,
        base_config_path: str,
        environment_config_path: Optional[str],
        indexes: AuditIndexes,
        environment: str,
        **kwargs
    ) -> RuleResults:
        """Evaluate rules with environment-specific overrides.
        
        Args:
            base_config_path: Path to base rule configuration
            environment_config_path: Path to environment-specific overrides
            indexes: Audit indexes to evaluate against
            environment: Target environment
            **kwargs: Additional evaluation options
            
        Returns:
            Rule evaluation results
        """
        from .parser import load_rules_for_environment
        
        # Load rules with environment overrides
        rules = load_rules_for_environment(
            base_config_path,
            environment,
            environment_config_path
        )
        
        # Create evaluation context
        context = EvaluationContext(
            indexes=indexes,
            query=AuditQuery(indexes),
            environment=environment,
            **kwargs
        )
        
        # Evaluate rules
        return self.engine.evaluate_rules(rules, context)
    
    def create_context(self, **kwargs) -> EvaluationContext:
        """Create evaluation context with defaults."""
        context_kwargs = {
            'indexes': None,
            'query': None,
            'environment': None,
            'target_urls': [],
            'debug': False,
            'timeout_seconds': 300,
            'max_workers': None,
            'fail_fast': False,
            'include_passed': True,
            'severity_filter': None,
            'rule_tag_filter': None
        }
        context_kwargs.update(kwargs)
        
        return EvaluationContext(**context_kwargs)


# Global orchestrator instance
orchestrator = RuleEvaluationOrchestrator()


# Convenience functions
def evaluate_rules(
    rules: List[Rule],
    indexes: AuditIndexes,
    environment: Optional[str] = None,
    **kwargs
) -> RuleResults:
    """Evaluate rules against audit data.
    
    Args:
        rules: List of rules to evaluate
        indexes: Audit indexes to evaluate against
        environment: Target environment for rule filtering
        **kwargs: Additional evaluation options
        
    Returns:
        Rule evaluation results
    """
    context = EvaluationContext(
        indexes=indexes,
        query=AuditQuery(indexes),
        environment=environment,
        **kwargs
    )
    
    return orchestrator.engine.evaluate_rules(rules, context)


def evaluate_from_config(
    config_path: Optional[str] = None,
    indexes: Optional[AuditIndexes] = None,
    environment: Optional[str] = None,
    rules: Optional[List[Rule]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> RuleResults:
    """Evaluate rules from configuration file or direct parameters.
    
    Args:
        config_path: Path to rule configuration file (legacy)
        indexes: Audit indexes to evaluate against
        environment: Target environment for rule filtering
        rules: Pre-loaded rules (alternative to config_path)
        config: Configuration dict (alternative to loading from file)
        **kwargs: Additional evaluation options
        
    Returns:
        Rule evaluation results
    """
    if rules is not None and indexes is not None:
        # CLI-style call with rules and indexes directly
        engine = RuleEvaluationEngine()
        
        # Create evaluation context from config if provided
        context_params = {
            'indexes': indexes,
            'query': AuditQuery(indexes),
            'environment': environment,
        }
        
        # Merge any config parameters
        if config:
            context_params.update(config)
            
        context = EvaluationContext(**context_params)
        return engine.evaluate_rules(rules, context)
    
    elif config_path and indexes:
        # Legacy call with config_path
        return orchestrator.evaluate_from_config(
            config_path, indexes, environment, **kwargs
        )
    else:
        raise ValueError("Must provide either (rules, indexes) or (config_path, indexes)")


# Async wrapper for CLI compatibility  
async def evaluate_from_config_async(
    config_path: Optional[str] = None,
    indexes: Optional[AuditIndexes] = None,
    environment: Optional[str] = None,
    rules: Optional[List[Rule]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> RuleResults:
    """Async wrapper for evaluate_from_config for CLI compatibility.
    
    Args:
        config_path: Path to rule configuration file (legacy)
        indexes: Audit indexes to evaluate against
        environment: Target environment for rule filtering
        rules: Pre-loaded rules (alternative to config_path)
        config: Configuration dict (alternative to loading from file)
        **kwargs: Additional evaluation options
        
    Returns:
        Rule evaluation results
    """
    # Just call the sync version - rule evaluation is CPU-bound
    return evaluate_from_config(
        config_path=config_path,
        indexes=indexes,
        environment=environment,
        rules=rules,
        config=config,
        **kwargs
    )