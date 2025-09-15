"""Metrics collection and monitoring for the scheduling system."""

import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import asyncio


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """A metric value with metadata."""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class TimerContext:
    """Context manager for timing operations."""
    metric_name: str
    collector: 'MetricsCollector'
    labels: Dict[str, str]
    start_time: Optional[float] = None

    def __enter__(self) -> 'TimerContext':
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.metric_name, duration, self.labels)


class MetricsCollector:
    """Collects and stores metrics for the scheduling system."""

    def __init__(self, retention_hours: int = 24, max_samples_per_metric: int = 10000):
        """Initialize metrics collector.

        Args:
            retention_hours: How long to keep metric samples
            max_samples_per_metric: Maximum samples to keep per metric
        """
        self.retention_hours = retention_hours
        self.max_samples_per_metric = max_samples_per_metric

        # Metric storage
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples_per_metric))
        self._timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples_per_metric))

        # Metric metadata
        self._metric_labels: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._metric_descriptions: Dict[str, str] = {}

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 3600  # 1 hour

    async def start(self) -> None:
        """Start the metrics collector."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Metrics collector started")

    async def stop(self) -> None:
        """Stop the metrics collector."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        logger.info("Metrics collector stopped")

    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None, description: Optional[str] = None) -> None:
        """Increment a counter metric."""
        metric_key = self._make_metric_key(name, labels)
        self._counters[metric_key] += value

        if labels:
            self._metric_labels[metric_key] = labels
        if description:
            self._metric_descriptions[name] = description

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, description: Optional[str] = None) -> None:
        """Set a gauge metric value."""
        metric_key = self._make_metric_key(name, labels)
        self._gauges[metric_key] = value

        if labels:
            self._metric_labels[metric_key] = labels
        if description:
            self._metric_descriptions[name] = description

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, description: Optional[str] = None) -> None:
        """Record a histogram value."""
        metric_key = self._make_metric_key(name, labels)
        now = datetime.now(timezone.utc)

        self._histograms[metric_key].append(MetricValue(value, now, labels or {}))

        if labels:
            self._metric_labels[metric_key] = labels
        if description:
            self._metric_descriptions[name] = description

    def record_timer(self, name: str, duration_seconds: float, labels: Optional[Dict[str, str]] = None, description: Optional[str] = None) -> None:
        """Record a timer duration."""
        metric_key = self._make_metric_key(name, labels)
        now = datetime.now(timezone.utc)

        self._timers[metric_key].append(MetricValue(duration_seconds, now, labels or {}))

        if labels:
            self._metric_labels[metric_key] = labels
        if description:
            self._metric_descriptions[name] = description

    def timer(self, name: str, labels: Optional[Dict[str, str]] = None) -> TimerContext:
        """Create a timer context manager."""
        return TimerContext(name, self, labels or {})

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        metric_key = self._make_metric_key(name, labels)
        return self._counters.get(metric_key, 0.0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current gauge value."""
        metric_key = self._make_metric_key(name, labels)
        return self._gauges.get(metric_key)

    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        metric_key = self._make_metric_key(name, labels)
        values = [mv.value for mv in self._histograms.get(metric_key, [])]

        if not values:
            return {}

        values.sort()
        count = len(values)

        return {
            'count': count,
            'sum': sum(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / count,
            'p50': values[int(count * 0.5)] if count > 0 else 0,
            'p90': values[int(count * 0.9)] if count > 0 else 0,
            'p95': values[int(count * 0.95)] if count > 0 else 0,
            'p99': values[int(count * 0.99)] if count > 0 else 0
        }

    def get_timer_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get timer statistics."""
        return self.get_histogram_stats(name, labels)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        metrics = {
            'counters': dict(self._counters),
            'gauges': dict(self._gauges),
            'histograms': {},
            'timers': {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        # Get histogram stats
        for metric_key in self._histograms:
            metrics['histograms'][metric_key] = self.get_histogram_stats(metric_key.split('|')[0], self._parse_labels(metric_key))

        # Get timer stats
        for metric_key in self._timers:
            metrics['timers'][metric_key] = self.get_timer_stats(metric_key.split('|')[0], self._parse_labels(metric_key))

        return metrics

    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Counters
        for metric_key, value in self._counters.items():
            name, labels = self._parse_metric_key(metric_key)
            labels_str = self._format_prometheus_labels(labels)
            if name in self._metric_descriptions:
                lines.append(f"# HELP {name} {self._metric_descriptions[name]}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name}{labels_str} {value}")

        # Gauges
        for metric_key, value in self._gauges.items():
            name, labels = self._parse_metric_key(metric_key)
            labels_str = self._format_prometheus_labels(labels)
            if name in self._metric_descriptions:
                lines.append(f"# HELP {name} {self._metric_descriptions[name]}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name}{labels_str} {value}")

        # Histograms (simplified)
        for metric_key in self._histograms:
            name, labels = self._parse_metric_key(metric_key)
            stats = self.get_histogram_stats(name, labels)
            labels_str = self._format_prometheus_labels(labels)

            if stats:
                if name in self._metric_descriptions:
                    lines.append(f"# HELP {name} {self._metric_descriptions[name]}")
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count{labels_str} {stats['count']}")
                lines.append(f"{name}_sum{labels_str} {stats['sum']}")

        return '\n'.join(lines) + '\n'

    def _make_metric_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a metric key with labels."""
        if not labels:
            return name

        label_parts = []
        for key, value in sorted(labels.items()):
            label_parts.append(f"{key}={value}")

        return f"{name}|{','.join(label_parts)}"

    def _parse_metric_key(self, metric_key: str) -> tuple[str, Dict[str, str]]:
        """Parse a metric key into name and labels."""
        if '|' not in metric_key:
            return metric_key, {}

        name, labels_str = metric_key.split('|', 1)
        labels = {}

        if labels_str:
            for label_pair in labels_str.split(','):
                if '=' in label_pair:
                    key, value = label_pair.split('=', 1)
                    labels[key] = value

        return name, labels

    def _parse_labels(self, metric_key: str) -> Dict[str, str]:
        """Parse labels from metric key."""
        _, labels = self._parse_metric_key(metric_key)
        return labels

    def _format_prometheus_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus export."""
        if not labels:
            return ""

        label_parts = []
        for key, value in sorted(labels.items()):
            # Escape quotes in values
            escaped_value = value.replace('"', '\\"')
            label_parts.append(f'{key}="{escaped_value}"')

        return '{' + ','.join(label_parts) + '}'

    async def _cleanup_loop(self) -> None:
        """Background task to clean up old metrics."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_old_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")

    async def _cleanup_old_metrics(self) -> None:
        """Remove old metric samples."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
        cleaned_count = 0

        # Clean histograms
        for metric_key, samples in self._histograms.items():
            original_len = len(samples)
            # Filter out old samples
            while samples and samples[0].timestamp < cutoff_time:
                samples.popleft()
            cleaned_count += original_len - len(samples)

        # Clean timers
        for metric_key, samples in self._timers.items():
            original_len = len(samples)
            # Filter out old samples
            while samples and samples[0].timestamp < cutoff_time:
                samples.popleft()
            cleaned_count += original_len - len(samples)

        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} old metric samples")


class SchedulingMetrics:
    """Pre-defined metrics for the scheduling system."""

    def __init__(self, collector: MetricsCollector):
        """Initialize scheduling metrics."""
        self.collector = collector

    def record_tick_duration(self, duration_ms: float) -> None:
        """Record engine tick duration."""
        self.collector.record_timer(
            'scheduling_tick_duration_seconds',
            duration_ms / 1000.0,
            description='Time taken for each scheduling engine tick'
        )

    def increment_schedules_processed(self, count: int = 1) -> None:
        """Increment schedules processed counter."""
        self.collector.increment_counter(
            'scheduling_schedules_processed_total',
            count,
            description='Total number of schedules processed'
        )

    def increment_runs_dispatched(self, priority: str = None) -> None:
        """Increment runs dispatched counter."""
        labels = {'priority': priority} if priority else {}
        self.collector.increment_counter(
            'scheduling_runs_dispatched_total',
            1.0,
            labels,
            description='Total number of runs dispatched'
        )

    def increment_runs_completed(self, status: str) -> None:
        """Increment runs completed counter."""
        self.collector.increment_counter(
            'scheduling_runs_completed_total',
            1.0,
            {'status': status},
            description='Total number of runs completed'
        )

    def increment_blackout_blocks(self) -> None:
        """Increment blackout blocks counter."""
        self.collector.increment_counter(
            'scheduling_blackout_blocks_total',
            description='Total number of runs blocked by blackout windows'
        )

    def increment_lock_conflicts(self) -> None:
        """Increment lock conflicts counter."""
        self.collector.increment_counter(
            'scheduling_lock_conflicts_total',
            description='Total number of runs blocked by concurrency limits'
        )

    def increment_catch_up_runs(self) -> None:
        """Increment catch-up runs counter."""
        self.collector.increment_counter(
            'scheduling_catch_up_runs_total',
            description='Total number of catch-up runs created'
        )

    def set_active_schedules(self, count: int) -> None:
        """Set number of active schedules."""
        self.collector.set_gauge(
            'scheduling_schedules_active',
            count,
            description='Number of currently active schedules'
        )

    def set_queue_depth(self, depth: int) -> None:
        """Set current queue depth."""
        self.collector.set_gauge(
            'scheduling_queue_depth',
            depth,
            description='Current number of runs in the dispatch queue'
        )

    def set_running_runs(self, count: int) -> None:
        """Set number of currently running runs."""
        self.collector.set_gauge(
            'scheduling_runs_running',
            count,
            description='Number of currently running audit runs'
        )

    def record_queue_wait_time(self, wait_time_seconds: float) -> None:
        """Record time a run spent in queue."""
        self.collector.record_histogram(
            'scheduling_queue_wait_time_seconds',
            wait_time_seconds,
            description='Time runs spend waiting in the dispatch queue'
        )

    def record_run_duration(self, duration_seconds: float, status: str) -> None:
        """Record run duration."""
        self.collector.record_histogram(
            'scheduling_run_duration_seconds',
            duration_seconds,
            {'status': status},
            description='Duration of audit runs'
        )

    def record_schedule_error(self, schedule_id: str, error_type: str) -> None:
        """Record a schedule error."""
        self.collector.increment_counter(
            'scheduling_schedule_errors_total',
            1.0,
            {'schedule_id': schedule_id, 'error_type': error_type},
            description='Total number of schedule processing errors'
        )


def create_metrics_collector(retention_hours: int = 24) -> MetricsCollector:
    """Create a metrics collector with default configuration."""
    return MetricsCollector(retention_hours=retention_hours)


def create_scheduling_metrics(collector: MetricsCollector) -> SchedulingMetrics:
    """Create scheduling metrics with the given collector."""
    return SchedulingMetrics(collector)