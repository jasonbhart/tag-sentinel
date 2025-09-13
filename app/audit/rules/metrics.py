"""Comprehensive monitoring and metrics collection for the rule engine.

This module provides detailed metrics collection, performance monitoring, and
operational insights for rule engine operations, alert delivery, and system health.
Designed for production monitoring and operational dashboards.
"""

import json
import logging
import time
import threading
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Deque
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import psutil
from pydantic import BaseModel, Field

from .models import RuleResults, Severity, Failure
from .evaluator import RuleEvaluationResult
from .alerts.base import AlertDispatchResult, AlertStatus


logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics collected."""
    COUNTER = "counter"          # Monotonically increasing counters
    GAUGE = "gauge"             # Point-in-time values
    HISTOGRAM = "histogram"     # Distribution of values
    TIMER = "timer"            # Duration measurements
    RATE = "rate"              # Rate calculations over time


class AlertMetricType(str, Enum):
    """Alert-specific metric types."""
    DISPATCH_SUCCESS = "alert.dispatch.success"
    DISPATCH_FAILURE = "alert.dispatch.failure" 
    DISPATCH_RETRY = "alert.dispatch.retry"
    DISPATCH_TIMEOUT = "alert.dispatch.timeout"
    WEBHOOK_DELIVERY = "alert.webhook.delivered"
    EMAIL_DELIVERY = "alert.email.delivered"


class RuleMetricType(str, Enum):
    """Rule evaluation metric types."""
    RULES_EVALUATED = "rules.evaluated"
    RULES_PASSED = "rules.passed"
    RULES_FAILED = "rules.failed"
    EVALUATION_TIME = "rules.evaluation_time"
    CHECK_EXECUTIONS = "rules.check_executions"
    CRITICAL_FAILURES = "rules.critical_failures"
    WARNING_FAILURES = "rules.warning_failures"


class SystemMetricType(str, Enum):
    """System resource metric types."""
    CPU_USAGE = "system.cpu_usage"
    MEMORY_USAGE = "system.memory_usage" 
    DISK_IO = "system.disk_io"
    NETWORK_IO = "system.network_io"
    THREAD_COUNT = "system.threads"


class MetricPoint(BaseModel):
    """A single metric data point."""
    
    name: str = Field(description="Metric name")
    value: Union[int, float] = Field(description="Metric value")
    metric_type: MetricType = Field(description="Type of metric")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When metric was recorded"
    )
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Metric labels/tags"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement"
    )


class MetricSummary(BaseModel):
    """Statistical summary of metric values."""
    
    count: int = Field(description="Number of data points")
    sum: float = Field(description="Sum of all values")
    min: float = Field(description="Minimum value")
    max: float = Field(description="Maximum value")
    mean: float = Field(description="Mean value")
    median: float = Field(description="Median value")
    p95: float = Field(description="95th percentile")
    p99: float = Field(description="99th percentile")


@dataclass
class MetricBuffer:
    """Thread-safe circular buffer for metrics."""
    
    max_size: int = 10000
    values: Deque[float] = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add(self, value: float):
        """Add value to buffer."""
        with self.lock:
            if len(self.values) >= self.max_size:
                self.values.popleft()
            self.values.append(value)
    
    def get_summary(self) -> Optional[MetricSummary]:
        """Get statistical summary of buffered values."""
        with self.lock:
            if not self.values:
                return None
            
            values_list = list(self.values)
            sorted_values = sorted(values_list)
            count = len(values_list)
            
            return MetricSummary(
                count=count,
                sum=sum(values_list),
                min=min(values_list),
                max=max(values_list),
                mean=mean(values_list),
                median=median(sorted_values),
                p95=sorted_values[int(count * 0.95)] if count > 0 else 0,
                p99=sorted_values[int(count * 0.99)] if count > 0 else 0
            )
    
    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.values.clear()


class MetricsCollector:
    """Central metrics collection and storage system."""
    
    def __init__(self, max_history_hours: int = 24):
        self.max_history_hours = max_history_hours
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self.metric_buffers: Dict[str, MetricBuffer] = {}
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = {}
        self.lock = threading.Lock()
        
        # Performance tracking
        self.start_time = datetime.now(timezone.utc)
        self.system_metrics_enabled = True
        
        # Background system metrics collection
        self._system_monitor_running = False
        self._system_monitor_thread: Optional[threading.Thread] = None
    
    def record_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Record counter metric (monotonically increasing)."""
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + value
        
        self._store_metric(MetricPoint(
            name=name,
            value=self.counters[name],
            metric_type=MetricType.COUNTER,
            labels=labels or {}
        ))
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record gauge metric (point-in-time value)."""
        with self.lock:
            self.gauges[name] = value
        
        self._store_metric(MetricPoint(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {}
        ))
    
    def record_timer(self, name: str, duration_ms: float, labels: Optional[Dict[str, str]] = None):
        """Record timer metric (duration measurement)."""
        with self.lock:
            if name not in self.timers:
                self.timers[name] = []
            self.timers[name].append(duration_ms)
            
            # Keep only recent timer values
            cutoff_time = time.time() - (3600 * self.max_history_hours)
            # For simplicity, just keep last 1000 measurements
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]
        
        # Also store in buffer for statistical analysis
        self._get_or_create_buffer(name).add(duration_ms)
        
        self._store_metric(MetricPoint(
            name=name,
            value=duration_ms,
            metric_type=MetricType.TIMER,
            labels=labels or {},
            unit="ms"
        ))
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record histogram metric (distribution of values)."""
        self._get_or_create_buffer(name).add(value)
        
        self._store_metric(MetricPoint(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            labels=labels or {}
        ))
    
    def _get_or_create_buffer(self, name: str) -> MetricBuffer:
        """Get or create metric buffer."""
        if name not in self.metric_buffers:
            self.metric_buffers[name] = MetricBuffer()
        return self.metric_buffers[name]
    
    def _store_metric(self, metric: MetricPoint):
        """Store metric point with automatic cleanup."""
        metric_name = metric.name
        
        with self.lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            
            self.metrics[metric_name].append(metric)
            
            # Clean up old metrics
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.max_history_hours)
            self.metrics[metric_name] = [
                m for m in self.metrics[metric_name] 
                if m.timestamp > cutoff_time
            ]
    
    def get_counter(self, name: str) -> int:
        """Get current counter value."""
        with self.lock:
            return self.counters.get(name, 0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Get current gauge value."""
        with self.lock:
            return self.gauges.get(name)
    
    def get_metric_summary(self, name: str) -> Optional[MetricSummary]:
        """Get statistical summary of metric."""
        if name in self.metric_buffers:
            return self.metric_buffers[name].get_summary()
        return None
    
    def get_metrics_by_type(self, metric_type: MetricType) -> List[MetricPoint]:
        """Get all metrics of a specific type."""
        all_metrics = []
        with self.lock:
            for metrics_list in self.metrics.values():
                all_metrics.extend([m for m in metrics_list if m.metric_type == metric_type])
        return sorted(all_metrics, key=lambda m: m.timestamp)
    
    def get_metrics_by_name(self, name: str) -> List[MetricPoint]:
        """Get all metrics with a specific name."""
        with self.lock:
            return list(self.metrics.get(name, []))
    
    def get_recent_metrics(self, minutes: int = 5) -> List[MetricPoint]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        recent_metrics = []
        
        with self.lock:
            for metrics_list in self.metrics.values():
                recent_metrics.extend([m for m in metrics_list if m.timestamp > cutoff_time])
        
        return sorted(recent_metrics, key=lambda m: m.timestamp)
    
    def start_system_monitoring(self, interval_seconds: int = 30):
        """Start background system metrics collection."""
        if self._system_monitor_running:
            return
        
        self._system_monitor_running = True
        self._system_monitor_thread = threading.Thread(
            target=self._collect_system_metrics,
            args=(interval_seconds,),
            daemon=True
        )
        self._system_monitor_thread.start()
        logger.info(f"Started system metrics collection (interval: {interval_seconds}s)")
    
    def stop_system_monitoring(self):
        """Stop background system metrics collection."""
        self._system_monitor_running = False
        if self._system_monitor_thread:
            self._system_monitor_thread.join(timeout=5)
    
    def _collect_system_metrics(self, interval_seconds: int):
        """Background thread for system metrics collection."""
        while self._system_monitor_running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_gauge(SystemMetricType.CPU_USAGE, cpu_percent, {"unit": "percent"})
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.record_gauge(SystemMetricType.MEMORY_USAGE, memory.percent, {"unit": "percent"})
                
                # Thread count
                process = psutil.Process()
                thread_count = process.num_threads()
                self.record_gauge(SystemMetricType.THREAD_COUNT, thread_count)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.record_counter("system.disk.read_bytes", disk_io.read_bytes)
                    self.record_counter("system.disk.write_bytes", disk_io.write_bytes)
                
                # Network I/O
                net_io = psutil.net_io_counters()
                if net_io:
                    self.record_counter("system.network.bytes_sent", net_io.bytes_sent)
                    self.record_counter("system.network.bytes_recv", net_io.bytes_recv)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.warning(f"System metrics collection error: {e}")
                time.sleep(interval_seconds)
    
    def export_metrics(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """Export all metrics in specified format."""
        if format.lower() == "json":
            return self._export_json()
        elif format.lower() == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self) -> str:
        """Export metrics as JSON."""
        export_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "collection_period_hours": self.max_history_hours,
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "summaries": {}
        }
        
        # Add metric summaries
        for name, buffer in self.metric_buffers.items():
            summary = buffer.get_summary()
            if summary:
                export_data["summaries"][name] = summary.dict()
        
        return json.dumps(export_data, indent=2, default=str)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # Gauges  
        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        # Timer summaries
        for name, buffer in self.metric_buffers.items():
            summary = buffer.get_summary()
            if summary:
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count {summary.count}")
                lines.append(f"{name}_sum {summary.sum}")
                lines.append(f"{name} {{quantile=\"0.5\"}} {summary.median}")
                lines.append(f"{name} {{quantile=\"0.95\"}} {summary.p95}")
                lines.append(f"{name} {{quantile=\"0.99\"}} {summary.p99}")
        
        return "\n".join(lines)
    
    def clear_metrics(self):
        """Clear all collected metrics."""
        with self.lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.timers.clear()
            
        for buffer in self.metric_buffers.values():
            buffer.clear()
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get system health check information."""
        recent_metrics = len(self.get_recent_metrics(5))
        
        return {
            "status": "healthy" if recent_metrics > 0 else "warning",
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            "metrics_collected": sum(len(metrics) for metrics in self.metrics.values()),
            "recent_metrics_5min": recent_metrics,
            "system_monitoring": self._system_monitor_running,
            "memory_usage_mb": psutil.Process().memory_info().rss / (1024 * 1024)
        }


class RuleEngineMetrics:
    """Specialized metrics collector for rule engine operations."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def record_rule_evaluation_start(self, total_rules: int, environment: Optional[str] = None):
        """Record start of rule evaluation."""
        labels = {"environment": environment or "unknown"}
        self.collector.record_gauge("rules.evaluation.started", total_rules, labels)
    
    def record_rule_evaluation_complete(
        self, 
        results: RuleResults,
        duration_ms: float,
        environment: Optional[str] = None
    ):
        """Record completion of rule evaluation."""
        labels = {"environment": environment or "unknown"}
        
        # Basic counters
        self.collector.record_counter(RuleMetricType.RULES_EVALUATED, results.summary.total_rules, labels)
        self.collector.record_counter(RuleMetricType.RULES_PASSED, results.summary.passed_rules, labels)
        self.collector.record_counter(RuleMetricType.RULES_FAILED, results.summary.failed_rules, labels)
        
        # Failure severity breakdown
        self.collector.record_counter(RuleMetricType.CRITICAL_FAILURES, results.summary.critical_failures, labels)
        self.collector.record_counter(RuleMetricType.WARNING_FAILURES, results.summary.warning_failures, labels)
        
        # Performance metrics
        self.collector.record_timer(RuleMetricType.EVALUATION_TIME, duration_ms, labels)
        
        # Success rate gauge
        success_rate = (results.summary.passed_rules / results.summary.total_rules * 100) if results.summary.total_rules > 0 else 0
        self.collector.record_gauge("rules.success_rate", success_rate, labels)
    
    def record_check_execution(self, check_type: str, duration_ms: float, success: bool):
        """Record individual check execution."""
        labels = {"check_type": check_type, "success": str(success).lower()}
        self.collector.record_counter(RuleMetricType.CHECK_EXECUTIONS, 1, labels)
        self.collector.record_timer(f"checks.{check_type}.duration", duration_ms, labels)
    
    def record_rule_failure(self, rule_id: str, check_type: str, severity: Severity):
        """Record specific rule failure."""
        labels = {
            "rule_id": rule_id,
            "check_type": check_type,
            "severity": severity.value
        }
        self.collector.record_counter("rules.failures", 1, labels)
    
    def record_parallel_execution(self, workers_used: int, total_duration_ms: float):
        """Record parallel execution metrics."""
        self.collector.record_gauge("rules.parallel.workers", workers_used)
        self.collector.record_timer("rules.parallel.total_duration", total_duration_ms)
    
    def get_rule_performance_summary(self) -> Dict[str, Any]:
        """Get rule engine performance summary."""
        eval_time_summary = self.collector.get_metric_summary(RuleMetricType.EVALUATION_TIME)
        check_exec_summary = self.collector.get_metric_summary(RuleMetricType.CHECK_EXECUTIONS)
        
        return {
            "total_evaluations": self.collector.get_counter(RuleMetricType.RULES_EVALUATED),
            "total_rules_passed": self.collector.get_counter(RuleMetricType.RULES_PASSED),
            "total_rules_failed": self.collector.get_counter(RuleMetricType.RULES_FAILED),
            "critical_failures": self.collector.get_counter(RuleMetricType.CRITICAL_FAILURES),
            "warning_failures": self.collector.get_counter(RuleMetricType.WARNING_FAILURES),
            "evaluation_time_stats": eval_time_summary.dict() if eval_time_summary else None,
            "check_execution_stats": check_exec_summary.dict() if check_exec_summary else None
        }


class AlertMetrics:
    """Specialized metrics collector for alert dispatching operations."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def record_alert_dispatch_start(self, dispatcher_type: str, alert_id: str):
        """Record start of alert dispatch."""
        labels = {"dispatcher_type": dispatcher_type}
        self.collector.record_counter("alerts.dispatch.started", 1, labels)
    
    def record_alert_dispatch_result(self, result: AlertDispatchResult):
        """Record alert dispatch result."""
        labels = {
            "dispatcher_type": result.dispatcher_type,
            "status": result.status.value,
            "success": str(result.success).lower()
        }
        
        # Dispatch outcome counters
        if result.success:
            self.collector.record_counter(AlertMetricType.DISPATCH_SUCCESS, 1, labels)
        else:
            self.collector.record_counter(AlertMetricType.DISPATCH_FAILURE, 1, labels)
        
        # Response time tracking
        if result.response_time_ms:
            self.collector.record_timer("alerts.dispatch.response_time", result.response_time_ms, labels)
        
        # Retry tracking
        if result.attempt_number > 1:
            self.collector.record_counter(AlertMetricType.DISPATCH_RETRY, 1, labels)
        
        # Specific channel metrics
        if result.dispatcher_type == "webhook":
            self._record_webhook_metrics(result)
        elif result.dispatcher_type == "email":
            self._record_email_metrics(result)
    
    def _record_webhook_metrics(self, result: AlertDispatchResult):
        """Record webhook-specific metrics."""
        labels = {"success": str(result.success).lower()}
        
        if result.success:
            self.collector.record_counter(AlertMetricType.WEBHOOK_DELIVERY, 1, labels)
        
        # HTTP status code tracking
        if result.response_data and "status_code" in result.response_data:
            status_code = result.response_data["status_code"]
            self.collector.record_counter(
                "alerts.webhook.http_status", 
                1, 
                {**labels, "status_code": str(status_code)}
            )
    
    def _record_email_metrics(self, result: AlertDispatchResult):
        """Record email-specific metrics."""
        labels = {"success": str(result.success).lower()}
        
        if result.success:
            self.collector.record_counter(AlertMetricType.EMAIL_DELIVERY, 1, labels)
        
        # Recipients count tracking
        if result.response_data and "recipients_count" in result.response_data:
            recipients = result.response_data["recipients_count"]
            self.collector.record_gauge("alerts.email.recipients", recipients, labels)
    
    def record_alert_timeout(self, dispatcher_type: str, timeout_seconds: int):
        """Record alert dispatch timeout."""
        labels = {"dispatcher_type": dispatcher_type}
        self.collector.record_counter(AlertMetricType.DISPATCH_TIMEOUT, 1, labels)
        self.collector.record_histogram("alerts.timeout.duration", timeout_seconds, labels)
    
    def get_alert_delivery_summary(self) -> Dict[str, Any]:
        """Get alert delivery performance summary."""
        return {
            "total_dispatches_started": self.collector.get_counter("alerts.dispatch.started"),
            "successful_dispatches": self.collector.get_counter(AlertMetricType.DISPATCH_SUCCESS),
            "failed_dispatches": self.collector.get_counter(AlertMetricType.DISPATCH_FAILURE),
            "retry_attempts": self.collector.get_counter(AlertMetricType.DISPATCH_RETRY),
            "timeouts": self.collector.get_counter(AlertMetricType.DISPATCH_TIMEOUT),
            "webhook_deliveries": self.collector.get_counter(AlertMetricType.WEBHOOK_DELIVERY),
            "email_deliveries": self.collector.get_counter(AlertMetricType.EMAIL_DELIVERY),
            "response_time_stats": self.collector.get_metric_summary("alerts.dispatch.response_time")
        }


class MetricsReporter:
    """Generate reports from collected metrics."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.rule_metrics = RuleEngineMetrics(collector)
        self.alert_metrics = AlertMetrics(collector)
    
    def generate_operational_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate comprehensive operational report."""
        return {
            "report_generated": datetime.now(timezone.utc).isoformat(),
            "time_period_hours": hours_back,
            "system_health": self.collector.get_health_check(),
            "rule_engine_performance": self.rule_metrics.get_rule_performance_summary(),
            "alert_delivery_performance": self.alert_metrics.get_alert_delivery_summary(),
            "system_resources": self._get_system_resource_summary(),
            "recommendations": self._generate_recommendations()
        }
    
    def _get_system_resource_summary(self) -> Dict[str, Any]:
        """Get system resource utilization summary."""
        cpu_summary = self.collector.get_metric_summary(SystemMetricType.CPU_USAGE)
        memory_summary = self.collector.get_metric_summary(SystemMetricType.MEMORY_USAGE)
        
        return {
            "cpu_utilization": cpu_summary.dict() if cpu_summary else None,
            "memory_utilization": memory_summary.dict() if memory_summary else None,
            "current_cpu_percent": self.collector.get_gauge(SystemMetricType.CPU_USAGE),
            "current_memory_percent": self.collector.get_gauge(SystemMetricType.MEMORY_USAGE),
            "current_thread_count": self.collector.get_gauge(SystemMetricType.THREAD_COUNT)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate operational recommendations based on metrics."""
        recommendations = []
        
        # CPU recommendations
        cpu_usage = self.collector.get_gauge(SystemMetricType.CPU_USAGE)
        if cpu_usage and cpu_usage > 80:
            recommendations.append("High CPU usage detected. Consider scaling or optimizing rule evaluation.")
        
        # Memory recommendations
        memory_usage = self.collector.get_gauge(SystemMetricType.MEMORY_USAGE)
        if memory_usage and memory_usage > 85:
            recommendations.append("High memory usage. Consider increasing available memory or optimizing data structures.")
        
        # Rule performance recommendations
        failed_dispatches = self.collector.get_counter(AlertMetricType.DISPATCH_FAILURE)
        total_dispatches = self.collector.get_counter("alerts.dispatch.started")
        if total_dispatches > 0 and (failed_dispatches / total_dispatches) > 0.1:
            recommendations.append("High alert dispatch failure rate. Check webhook/email configurations.")
        
        # Evaluation time recommendations
        eval_time_summary = self.collector.get_metric_summary(RuleMetricType.EVALUATION_TIME)
        if eval_time_summary and eval_time_summary.mean > 10000:  # 10 seconds
            recommendations.append("Slow rule evaluation detected. Consider parallel processing or rule optimization.")
        
        return recommendations
    
    def export_to_file(self, file_path: Union[str, Path], format: str = "json"):
        """Export metrics to file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            report = self.generate_operational_report()
            with open(path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format.lower() == "prometheus":
            content = self.collector.export_metrics("prometheus")
            with open(path, 'w') as f:
                f.write(content)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None
_global_rule_metrics: Optional[RuleEngineMetrics] = None
_global_alert_metrics: Optional[AlertMetrics] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
        _global_collector.start_system_monitoring()
    return _global_collector


def get_rule_metrics() -> RuleEngineMetrics:
    """Get global rule engine metrics instance."""
    global _global_rule_metrics
    if _global_rule_metrics is None:
        _global_rule_metrics = RuleEngineMetrics(get_metrics_collector())
    return _global_rule_metrics


def get_alert_metrics() -> AlertMetrics:
    """Get global alert metrics instance.""" 
    global _global_alert_metrics
    if _global_alert_metrics is None:
        _global_alert_metrics = AlertMetrics(get_metrics_collector())
    return _global_alert_metrics


def create_metrics_reporter() -> MetricsReporter:
    """Create metrics reporter with global collector."""
    return MetricsReporter(get_metrics_collector())


# Context managers for automatic metric recording

class timer_metric:
    """Context manager for automatic timer metrics."""
    
    def __init__(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time: Optional[float] = None
        self.collector = get_metrics_collector()
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.collector.record_timer(self.metric_name, duration_ms, self.labels)


def counter_metric(metric_name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
    """Decorator for automatic counter metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            try:
                result = func(*args, **kwargs)
                collector.record_counter(metric_name, value, labels)
                return result
            except Exception as e:
                error_labels = {**(labels or {}), "error": "true"}
                collector.record_counter(f"{metric_name}.errors", 1, error_labels)
                raise
        return wrapper
    return decorator