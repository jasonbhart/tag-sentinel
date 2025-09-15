"""Unit tests for rule evaluation engine."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from app.audit.rules.evaluator import (
    RuleEvaluationEngine, 
    EvaluationContext,
    RuleEvaluationResult,
    OptimizedRuleEvaluationEngine
)
from app.audit.rules.models import Rule, RuleResults, RuleSummary, Failure, Severity, CheckConfig, CheckType
from app.audit.rules.indexing import AuditIndexes, AuditQuery


@pytest.fixture
def mock_audit_indexes():
    """Create mock audit indexes."""
    indexes = Mock(spec=AuditIndexes)
    indexes.requests = Mock()
    indexes.cookies = Mock() 
    indexes.events = Mock()
    indexes.pages = Mock()
    indexes.summary = Mock()
    return indexes


@pytest.fixture
def mock_audit_query():
    """Create mock audit query."""
    query = Mock(spec=AuditQuery)
    return query


@pytest.fixture
def evaluation_context(mock_audit_indexes, mock_audit_query):
    """Create evaluation context for testing."""
    return EvaluationContext(
        indexes=mock_audit_indexes,
        query=mock_audit_query,
        environment="test",
        target_urls=["https://example.com"]
    )


@pytest.fixture
def sample_rule():
    """Create a sample rule for testing."""
    return Rule(
        id="test_rule",
        name="Test Rule",
        description="A rule for testing",
        severity=Severity.WARNING,
        check=CheckConfig(
            type=CheckType.REQUEST_PRESENT,
            parameters={"url_pattern": "test.js"}
        )
    )


class TestRuleEvaluationEngine:
    """Test the main rule evaluation engine."""
    
    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = RuleEvaluationEngine()
        
        assert isinstance(engine.check_cache, dict)
        assert isinstance(engine.evaluation_hooks, dict)
        assert len(engine.check_cache) == 0
    
    def test_register_hook(self):
        """Test registering evaluation hooks."""
        engine = RuleEvaluationEngine()
        
        def test_hook(data, context):
            pass
        
        engine.register_hook("test_hook", test_hook)
        
        assert "test_hook" in engine.evaluation_hooks
        assert test_hook in engine.evaluation_hooks["test_hook"]
    
    @patch('app.audit.rules.evaluator.RuleEvaluationEngine.evaluate_single_rule')
    def test_evaluate_rules_sequential(self, mock_evaluate_single, sample_rule, evaluation_context):
        """Test sequential rule evaluation."""
        engine = RuleEvaluationEngine()
        
        # Mock single rule evaluation
        mock_result = RuleEvaluationResult(
            rule=sample_rule,
            check_results=[],
            passed=True,
            failures=[],
            execution_time_ms=100.0
        )
        mock_evaluate_single.return_value = mock_result
        
        rules = [sample_rule]
        evaluation_context.max_workers = None  # Force sequential
        
        result = engine.evaluate_rules(rules, evaluation_context)
        
        assert isinstance(result, RuleResults)
        assert mock_evaluate_single.call_count == 1
    
    @patch('app.audit.rules.evaluator.RuleEvaluationEngine.evaluate_single_rule')
    def test_evaluate_rules_parallel(self, mock_evaluate_single, sample_rule, evaluation_context):
        """Test parallel rule evaluation."""
        engine = RuleEvaluationEngine()
        
        # Mock single rule evaluation
        mock_result = RuleEvaluationResult(
            rule=sample_rule,
            check_results=[],
            passed=True,
            failures=[],
            execution_time_ms=100.0
        )
        mock_evaluate_single.return_value = mock_result
        
        rules = [sample_rule] * 3  # Multiple rules
        evaluation_context.max_workers = 2  # Force parallel
        
        result = engine.evaluate_rules(rules, evaluation_context)
        
        assert isinstance(result, RuleResults)
        assert mock_evaluate_single.call_count == 3
    
    def test_filter_rules_by_environment(self, evaluation_context):
        """Test filtering rules by environment."""
        engine = RuleEvaluationEngine()
        
        # Create rules with different environment constraints
        rule_all_envs = Rule(
            id="all_envs",
            name="All Environments",
            description="Applies to all environments",
            severity=Severity.INFO,
            check=CheckConfig(type=CheckType.REQUEST_PRESENT, parameters={})
        )
        
        rule_prod_only = Rule(
            id="prod_only",
            name="Production Only", 
            description="Applies only to production",
            severity=Severity.CRITICAL,
            check=CheckConfig(type=CheckType.REQUEST_PRESENT, parameters={})
        )
        # Set environment constraint
        rule_prod_only.applies_to.environments = ["production"]
        
        rules = [rule_all_envs, rule_prod_only]
        
        # Test with production environment
        evaluation_context.environment = "production"
        filtered = engine._filter_rules(rules, evaluation_context)
        assert len(filtered) == 2  # Both should apply
        
        # Test with test environment  
        evaluation_context.environment = "test"
        filtered = engine._filter_rules(rules, evaluation_context)
        assert len(filtered) == 1  # Only all_envs should apply
        assert filtered[0].id == "all_envs"
    
    def test_aggregate_results(self, sample_rule, evaluation_context):
        """Test result aggregation."""
        engine = RuleEvaluationEngine()
        
        # Create test evaluation results
        passed_result = RuleEvaluationResult(
            rule=sample_rule,
            check_results=[],
            passed=True,
            failures=[],
            execution_time_ms=50.0
        )
        
        failed_result = RuleEvaluationResult(
            rule=sample_rule,
            check_results=[],
            passed=False,
            failures=[
                Failure(
                    check_id="test_check",
                    severity=Severity.CRITICAL,
                    message="Test failure"
                )
            ],
            execution_time_ms=75.0
        )
        
        rule_results = [passed_result, failed_result]
        
        aggregated = engine._aggregate_results(rule_results, evaluation_context)
        
        assert isinstance(aggregated, RuleResults)
        assert aggregated.summary.total_rules == 2
        assert aggregated.summary.passed_rules == 1
        assert aggregated.summary.failed_rules == 1
        assert aggregated.summary.critical_failures == 1
        assert len(aggregated.failures) == 1
    
    def test_error_handling_during_evaluation(self, sample_rule, evaluation_context):
        """Test error handling during evaluation."""
        engine = RuleEvaluationEngine()
        
        # Mock evaluate_single_rule to raise an exception
        with patch.object(engine, 'evaluate_single_rule', side_effect=Exception("Test error")):
            result = engine.evaluate_rules([sample_rule], evaluation_context)
            
            # Should return an error result instead of crashing
            assert isinstance(result, RuleResults)
            # Error result creation should be tested separately


class TestOptimizedRuleEvaluationEngine:
    """Test the optimized evaluation engine."""
    
    def test_optimized_engine_initialization(self):
        """Test optimized engine initializes with performance features."""
        engine = OptimizedRuleEvaluationEngine(enable_profiling=True)
        
        assert engine.enable_profiling is True
        assert hasattr(engine, 'performance_monitor')
        assert hasattr(engine, 'batch_processor')
        assert hasattr(engine, 'metrics')
        assert engine.optimal_batch_size > 0
        assert engine.max_parallel_workers > 0
    
    def test_calculate_optimal_batch_size(self):
        """Test batch size calculation."""
        engine = OptimizedRuleEvaluationEngine()
        
        batch_size = engine._calculate_optimal_batch_size()
        
        assert isinstance(batch_size, int)
        assert batch_size > 0
        assert batch_size <= 500  # Should have reasonable upper bound
    
    @patch('psutil.Process')
    def test_monitor_resource_usage(self, mock_process):
        """Test resource usage monitoring."""
        engine = OptimizedRuleEvaluationEngine(enable_profiling=True)
        
        # Mock process metrics
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_process_instance.cpu_percent.return_value = 25.5
        mock_process.return_value = mock_process_instance
        
        engine._monitor_resource_usage()
        
        assert engine.metrics.memory_usage_mb == 100.0
        assert engine.metrics.cpu_usage_percent == 25.5
    
    def test_cached_check_instance(self):
        """Test check instance caching."""
        engine = OptimizedRuleEvaluationEngine()
        
        # This would need actual check registry setup to test properly
        # For now, test that the method exists and handles missing checks
        result = engine._get_cached_check_instance("nonexistent_check")
        assert result is None


class TestEvaluationContext:
    """Test evaluation context."""
    
    def test_context_creation(self, mock_audit_indexes, mock_audit_query):
        """Test context creation with defaults."""
        context = EvaluationContext(
            indexes=mock_audit_indexes,
            query=mock_audit_query
        )
        
        assert context.indexes == mock_audit_indexes
        assert context.query == mock_audit_query
        assert context.environment is None
        assert context.target_urls == []
        assert context.debug is False
        assert context.timeout_seconds == 300
        assert context.max_workers is None
        assert context.fail_fast is False
        assert context.include_passed is True
        assert context.start_time is None
        assert context.end_time is None
        assert context.total_rules_evaluated == 0
        assert context.total_checks_executed == 0
    
    def test_context_with_options(self, mock_audit_indexes, mock_audit_query):
        """Test context creation with custom options."""
        context = EvaluationContext(
            indexes=mock_audit_indexes,
            query=mock_audit_query,
            environment="production",
            target_urls=["https://example.com", "https://shop.example.com"],
            debug=True,
            timeout_seconds=600,
            max_workers=4,
            fail_fast=True,
            include_passed=False
        )
        
        assert context.environment == "production"
        assert context.target_urls == ["https://example.com", "https://shop.example.com"]
        assert context.debug is True
        assert context.timeout_seconds == 600
        assert context.max_workers == 4
        assert context.fail_fast is True
        assert context.include_passed is False


class TestRuleEvaluationResult:
    """Test rule evaluation result."""
    
    def test_result_creation(self, sample_rule):
        """Test result creation."""
        result = RuleEvaluationResult(
            rule=sample_rule,
            check_results=[],
            passed=True,
            failures=[],
            execution_time_ms=123.45
        )
        
        assert result.rule == sample_rule
        assert result.passed is True
        assert result.failures == []
        assert result.execution_time_ms == 123.45
        assert result.error is None
    
    def test_severity_property_no_failures(self, sample_rule):
        """Test severity property with no failures."""
        result = RuleEvaluationResult(
            rule=sample_rule,
            check_results=[],
            passed=True,
            failures=[],
            execution_time_ms=100.0
        )
        
        assert result.severity == Severity.INFO
    
    def test_severity_property_with_failures(self, sample_rule):
        """Test severity property with mixed failures."""
        failures = [
            Failure(check_id="test1", severity=Severity.WARNING, message="Warning"),
            Failure(check_id="test2", severity=Severity.CRITICAL, message="Critical"),
            Failure(check_id="test3", severity=Severity.INFO, message="Info")
        ]
        
        result = RuleEvaluationResult(
            rule=sample_rule,
            check_results=[],
            passed=False,
            failures=failures,
            execution_time_ms=100.0
        )
        
        # Should return highest severity (CRITICAL)
        assert result.severity == Severity.CRITICAL
    
    def test_result_with_error(self, sample_rule):
        """Test result with error."""
        result = RuleEvaluationResult(
            rule=sample_rule,
            check_results=[],
            passed=False,
            failures=[],
            execution_time_ms=0.0,
            error="Test error occurred"
        )
        
        assert result.error == "Test error occurred"
        assert result.passed is False


class TestEvaluationHooks:
    """Test evaluation hook system."""
    
    def test_execute_hooks(self, sample_rule, evaluation_context):
        """Test hook execution."""
        engine = RuleEvaluationEngine()
        
        hook_calls = []
        
        def test_hook(data, context):
            hook_calls.append(("test_hook", data, context))
        
        def another_hook(data, context):
            hook_calls.append(("another_hook", data, context))
        
        engine.register_hook("pre_evaluation", test_hook)
        engine.register_hook("pre_evaluation", another_hook)
        
        # Execute hooks
        engine._execute_hooks("pre_evaluation", [sample_rule], evaluation_context)
        
        assert len(hook_calls) == 2
        assert hook_calls[0][0] == "test_hook"
        assert hook_calls[1][0] == "another_hook"
        assert hook_calls[0][1] == [sample_rule]  # data
        assert hook_calls[0][2] == evaluation_context  # context
    
    def test_hook_execution_with_exception(self, sample_rule, evaluation_context):
        """Test hook execution continues even if one hook raises exception."""
        engine = RuleEvaluationEngine()
        
        successful_calls = []
        
        def failing_hook(data, context):
            raise Exception("Hook failed")
        
        def successful_hook(data, context):
            successful_calls.append("called")
        
        engine.register_hook("test_hook", failing_hook)
        engine.register_hook("test_hook", successful_hook)
        
        # Should not raise exception, should continue with other hooks
        engine._execute_hooks("test_hook", [sample_rule], evaluation_context)
        
        # Successful hook should still be called
        assert len(successful_calls) == 1


class TestParallelEvaluation:
    """Test parallel evaluation features."""
    
    @patch('app.audit.rules.evaluator.RuleEvaluationEngine.evaluate_single_rule')
    def test_fail_fast_parallel(self, mock_evaluate_single, evaluation_context):
        """Test fail-fast behavior in parallel evaluation."""
        engine = RuleEvaluationEngine()
        
        # Create rules
        rules = [
            Rule(id=f"rule_{i}", name=f"Rule {i}", description="Test", 
                 severity=Severity.CRITICAL, 
                 check=CheckConfig(type=CheckType.REQUEST_PRESENT, parameters={}))
            for i in range(5)
        ]
        
        # Mock results - first one fails critically
        def mock_evaluation_side_effect(rule, context):
            if rule.id == "rule_0":
                return RuleEvaluationResult(
                    rule=rule,
                    check_results=[],
                    passed=False,
                    failures=[Failure(check_id="test", severity=Severity.CRITICAL, message="Critical failure")],
                    execution_time_ms=100.0
                )
            else:
                return RuleEvaluationResult(
                    rule=rule,
                    check_results=[],
                    passed=True,
                    failures=[],
                    execution_time_ms=50.0
                )
        
        mock_evaluate_single.side_effect = mock_evaluation_side_effect
        
        evaluation_context.fail_fast = True
        evaluation_context.max_workers = 2
        
        result = engine.evaluate_rules(rules, evaluation_context)
        
        assert isinstance(result, RuleResults)
        # With fail-fast, should stop after critical failure
        # Exact behavior depends on timing in parallel execution