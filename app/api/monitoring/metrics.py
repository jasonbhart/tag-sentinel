"""Metrics collection and tracking for Tag Sentinel API.

This module provides comprehensive metrics collection including
request counts, response times, error rates, and custom business metrics.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import threading

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """A single metric value with metadata."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""

    # Collection settings
    enable_request_metrics: bool = True
    enable_response_time_metrics: bool = True
    enable_error_metrics: bool = True
    enable_custom_metrics: bool = True

    # Histogram settings
    histogram_buckets: List[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
    ])

    # Buffer settings
    max_buffer_size: int = 10000
    buffer_flush_interval: int = 60  # seconds

    # Aggregation settings
    aggregation_window_size: int = 300  # 5 minutes
    percentiles: List[float] = field(default_factory=lambda: [50.0, 90.0, 95.0, 99.0])

    # Labels
    default_labels: Dict[str, str] = field(default_factory=dict)
    include_endpoint_labels: bool = True
    include_method_labels: bool = True
    include_status_labels: bool = True

    # Export settings
    export_interval: int = 60  # seconds
    export_batch_size: int = 1000


class MetricsCollector:
    """Main metrics collector for the API."""

    def __init__(self, config: Optional[MetricsConfig] = None):
        """Initialize metrics collector.

        Args:
            config: Metrics collection configuration
        """
        self.config = config or MetricsConfig()

        # Thread-safe storage for metrics
        self._lock = threading.RLock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Buffered metrics for export
        self._metric_buffer: List[MetricValue] = []

        # Background tasks
        self._flush_task: Optional[asyncio.Task] = None
        self._aggregation_task: Optional[asyncio.Task] = None

        # Start background tasks
        self._start_background_tasks()

        logger.info("MetricsCollector initialized")

    def _start_background_tasks(self):
        """Start background tasks for metrics processing."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._flush_task = loop.create_task(self._periodic_flush())
                self._aggregation_task = loop.create_task(self._periodic_aggregation())
        except RuntimeError:
            # No event loop running yet - tasks will be started later
            pass

    async def _periodic_flush(self):
        """Periodically flush metrics buffer."""
        while True:
            try:
                await asyncio.sleep(self.config.buffer_flush_interval)
                await self.flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics flush task: {e}")

    async def _periodic_aggregation(self):
        """Periodically aggregate metrics."""
        while True:
            try:
                await asyncio.sleep(self.config.aggregation_window_size)
                self._aggregate_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics aggregation task: {e}")

    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name
            value: Amount to increment
            labels: Optional labels for the metric
        """
        if not self.config.enable_custom_metrics:
            return

        full_name = self._build_metric_name(name, labels)

        with self._lock:
            self._counters[full_name] += value

        self._buffer_metric(MetricValue(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            labels=labels or {}
        ))

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value.

        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels for the metric
        """
        if not self.config.enable_custom_metrics:
            return

        full_name = self._build_metric_name(name, labels)

        with self._lock:
            self._gauges[full_name] = value

        self._buffer_metric(MetricValue(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {}
        ))

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a value in a histogram.

        Args:
            name: Metric name
            value: Value to record
            labels: Optional labels for the metric
        """
        if not self.config.enable_custom_metrics:
            return

        full_name = self._build_metric_name(name, labels)

        with self._lock:
            histogram = self._histograms[full_name]
            histogram.append(value)

            # Keep only recent values to prevent memory growth
            if len(histogram) > self.config.max_buffer_size:
                histogram[:] = histogram[-self.config.max_buffer_size//2:]

        self._buffer_metric(MetricValue(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            labels=labels or {}
        ))

    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a timer value.

        Args:
            name: Metric name
            duration: Duration in seconds
            labels: Optional labels for the metric
        """
        if not self.config.enable_response_time_metrics:
            return

        full_name = self._build_metric_name(name, labels)

        with self._lock:
            self._timers[full_name].append(duration)

        self._buffer_metric(MetricValue(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            labels=labels or {}
        ))

    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations.

        Args:
            name: Metric name
            labels: Optional labels for the metric

        Returns:
            Timer context manager
        """
        return TimerContext(self, name, labels)

    def record_request(self, method: str, endpoint: str, status_code: int, duration: float) -> None:
        """Record an API request.

        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            duration: Request duration in seconds
        """
        labels = {}

        if self.config.include_method_labels:
            labels["method"] = method

        if self.config.include_endpoint_labels:
            labels["endpoint"] = endpoint

        if self.config.include_status_labels:
            labels["status_code"] = str(status_code)
            labels["status_class"] = f"{status_code // 100}xx"

        # Add default labels
        labels.update(self.config.default_labels)

        # Record request count
        if self.config.enable_request_metrics:
            self.increment_counter("api_requests_total", 1.0, labels)

        # Record response time
        if self.config.enable_response_time_metrics:
            self.record_timer("api_request_duration_seconds", duration, labels)

        # Record errors
        if self.config.enable_error_metrics and status_code >= 400:
            error_labels = labels.copy()
            error_labels["error_type"] = "client_error" if status_code < 500 else "server_error"
            self.increment_counter("api_errors_total", 1.0, error_labels)

    def _build_metric_name(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Build full metric name including labels."""
        if not labels:
            return name

        label_parts = [f"{k}={v}" for k, v in sorted(labels.items())]
        return f"{name}{{{','.join(label_parts)}}}"

    def _buffer_metric(self, metric: MetricValue) -> None:
        """Add metric to buffer for export."""
        with self._lock:
            self._metric_buffer.append(metric)

            # Prevent buffer overflow
            if len(self._metric_buffer) > self.config.max_buffer_size:
                self._metric_buffer = self._metric_buffer[-self.config.max_buffer_size//2:]

    def _aggregate_metrics(self) -> None:
        """Aggregate histogram and timer metrics."""
        with self._lock:
            # Aggregate histograms
            for name, values in self._histograms.items():
                if values:
                    self._create_histogram_aggregates(name, values)

            # Aggregate timers
            for name, values in self._timers.items():
                if values:
                    self._create_timer_aggregates(name, list(values))

    def _create_histogram_aggregates(self, name: str, values: List[float]) -> None:
        """Create aggregated metrics from histogram values."""
        if not values:
            return

        # Basic statistics
        self.set_gauge(f"{name}_count", len(values))
        self.set_gauge(f"{name}_sum", sum(values))
        self.set_gauge(f"{name}_mean", statistics.mean(values))

        if len(values) > 1:
            self.set_gauge(f"{name}_stddev", statistics.stdev(values))

        # Percentiles
        sorted_values = sorted(values)
        for percentile in self.config.percentiles:
            index = int((percentile / 100) * (len(sorted_values) - 1))
            value = sorted_values[index]
            self.set_gauge(f"{name}_p{int(percentile)}", value)

        # Histogram buckets
        for bucket in self.config.histogram_buckets:
            count = sum(1 for v in values if v <= bucket)
            self.set_gauge(f"{name}_bucket_le_{bucket}", count)

    def _create_timer_aggregates(self, name: str, values: List[float]) -> None:
        """Create aggregated metrics from timer values."""
        self._create_histogram_aggregates(name, values)

    async def flush_buffer(self) -> List[MetricValue]:
        """Flush metrics buffer and return buffered metrics.

        Returns:
            List of buffered metrics
        """
        with self._lock:
            buffered_metrics = self._metric_buffer.copy()
            self._metric_buffer.clear()

        logger.debug(f"Flushed {len(buffered_metrics)} metrics from buffer")
        return buffered_metrics

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values.

        Returns:
            Dictionary of current metrics
        """
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: list(v) for k, v in self._histograms.items()},
                "timers": {k: list(v) for k, v in self._timers.items()},
                "buffer_size": len(self._metric_buffer)
            }

    async def close(self) -> None:
        """Close metrics collector and clean up resources."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass

        logger.info("MetricsCollector closed")


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str, labels: Optional[Dict[str, str]] = None):
        """Initialize timer context.

        Args:
            collector: Metrics collector
            name: Metric name
            labels: Optional labels
        """
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record duration."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.name, duration, self.labels)


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def set_metrics_collector(collector: MetricsCollector) -> None:
    """Set global metrics collector instance."""
    global _global_collector
    _global_collector = collector


# Convenience functions for common metrics
def increment(name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
    """Increment a counter metric."""
    get_metrics_collector().increment_counter(name, value, labels)


def gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Set a gauge metric."""
    get_metrics_collector().set_gauge(name, value, labels)


def histogram(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Record a histogram value."""
    get_metrics_collector().record_histogram(name, value, labels)


def timer(name: str, labels: Optional[Dict[str, str]] = None):
    """Create a timer context manager."""
    return get_metrics_collector().timer(name, labels)