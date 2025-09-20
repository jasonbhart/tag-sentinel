"""Metrics exporters for Tag Sentinel API.

This module provides exporters for various monitoring systems
including Prometheus, CloudWatch, and custom formats.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, TextIO
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json

from .metrics import MetricValue, MetricType, MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for metrics exporters."""
    export_interval: int = 60  # seconds
    batch_size: int = 1000
    enable_labels: bool = True
    label_prefix: str = ""
    metric_prefix: str = "tag_sentinel_"
    timestamp_format: str = "unix"  # unix, iso, custom
    filter_patterns: List[str] = field(default_factory=list)


class MetricsExporter(ABC):
    """Abstract base class for metrics exporters."""

    def __init__(self, name: str, config: Optional[ExportConfig] = None):
        """Initialize metrics exporter.

        Args:
            name: Exporter name
            config: Export configuration
        """
        self.name = name
        self.config = config or ExportConfig()
        self.enabled = True
        self.last_export_time: Optional[float] = None

    @abstractmethod
    async def export_metrics(self, metrics: List[MetricValue]) -> bool:
        """Export metrics to destination.

        Args:
            metrics: List of metrics to export

        Returns:
            True if export was successful
        """
        pass

    def should_export_metric(self, metric: MetricValue) -> bool:
        """Check if metric should be exported based on filters.

        Args:
            metric: Metric to check

        Returns:
            True if metric should be exported
        """
        if not self.config.filter_patterns:
            return True

        metric_name = f"{self.config.metric_prefix}{metric.name}"

        for pattern in self.config.filter_patterns:
            if pattern.startswith("!"):
                # Exclusion pattern
                if metric_name.startswith(pattern[1:]):
                    return False
            else:
                # Inclusion pattern
                if not metric_name.startswith(pattern):
                    return False

        return True

    def format_metric_name(self, metric: MetricValue) -> str:
        """Format metric name for export.

        Args:
            metric: Metric to format

        Returns:
            Formatted metric name
        """
        name = metric.name
        if self.config.metric_prefix:
            name = f"{self.config.metric_prefix}{name}"
        return name

    def format_labels(self, labels: Dict[str, str]) -> Dict[str, str]:
        """Format labels for export.

        Args:
            labels: Original labels

        Returns:
            Formatted labels
        """
        if not self.config.enable_labels:
            return {}

        formatted = {}
        for key, value in labels.items():
            label_key = key
            if self.config.label_prefix:
                label_key = f"{self.config.label_prefix}{key}"
            formatted[label_key] = str(value)

        return formatted


class PrometheusExporter(MetricsExporter):
    """Exporter for Prometheus metrics format."""

    def __init__(
        self,
        name: str = "prometheus",
        output_file: Optional[str] = None,
        registry_file: str = "/tmp/tag_sentinel_metrics.prom",
        config: Optional[ExportConfig] = None
    ):
        """Initialize Prometheus exporter.

        Args:
            name: Exporter name
            output_file: Optional output file path
            registry_file: Registry file for metrics
            config: Export configuration
        """
        super().__init__(name, config)
        self.output_file = output_file
        self.registry_file = registry_file
        self.metric_families: Dict[str, Dict[str, Any]] = {}

    async def export_metrics(self, metrics: List[MetricValue]) -> bool:
        """Export metrics in Prometheus format."""
        try:
            # Group metrics by name and type
            grouped_metrics = self._group_metrics(metrics)

            # Generate Prometheus format
            output_lines = []
            for metric_name, metric_group in grouped_metrics.items():
                lines = self._format_metric_family(metric_name, metric_group)
                output_lines.extend(lines)

            # Write to file
            content = "\n".join(output_lines) + "\n"

            if self.output_file:
                async with aiofiles.open(self.output_file, 'w') as f:
                    await f.write(content)

            # Also write to registry file for Prometheus scraping
            async with aiofiles.open(self.registry_file, 'w') as f:
                await f.write(content)

            logger.debug(f"Exported {len(metrics)} metrics to Prometheus format")
            return True

        except Exception as e:
            logger.error(f"Error exporting to Prometheus: {e}")
            return False

    def _group_metrics(self, metrics: List[MetricValue]) -> Dict[str, List[MetricValue]]:
        """Group metrics by name for Prometheus export."""
        grouped = {}
        for metric in metrics:
            if not self.should_export_metric(metric):
                continue

            metric_name = self.format_metric_name(metric)
            if metric_name not in grouped:
                grouped[metric_name] = []
            grouped[metric_name].append(metric)

        return grouped

    def _format_metric_family(self, name: str, metrics: List[MetricValue]) -> List[str]:
        """Format a metric family for Prometheus."""
        if not metrics:
            return []

        lines = []
        metric_type = metrics[0].metric_type

        # Add HELP comment
        lines.append(f"# HELP {name} {name} metric")

        # Add TYPE comment
        prom_type = self._get_prometheus_type(metric_type)
        lines.append(f"# TYPE {name} {prom_type}")

        # Add metric values
        for metric in metrics:
            formatted_labels = self.format_labels(metric.labels)
            labels_str = ""

            if formatted_labels:
                label_pairs = [f'{k}="{v}"' for k, v in formatted_labels.items()]
                labels_str = "{" + ",".join(label_pairs) + "}"

            timestamp_str = ""
            if self.config.timestamp_format == "unix":
                timestamp_str = f" {int(metric.timestamp * 1000)}"  # Prometheus uses milliseconds

            lines.append(f"{name}{labels_str} {metric.value}{timestamp_str}")

        return lines

    def _get_prometheus_type(self, metric_type: MetricType) -> str:
        """Get Prometheus metric type string."""
        type_map = {
            MetricType.COUNTER: "counter",
            MetricType.GAUGE: "gauge",
            MetricType.HISTOGRAM: "histogram",
            MetricType.TIMER: "histogram"
        }
        return type_map.get(metric_type, "gauge")


class CloudWatchExporter(MetricsExporter):
    """Exporter for AWS CloudWatch metrics."""

    def __init__(
        self,
        name: str = "cloudwatch",
        namespace: str = "TagSentinel/API",
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        config: Optional[ExportConfig] = None
    ):
        """Initialize CloudWatch exporter.

        Args:
            name: Exporter name
            namespace: CloudWatch namespace
            region: AWS region
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            config: Export configuration
        """
        super().__init__(name, config)
        self.namespace = namespace
        self.region = region
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self._cloudwatch_client = None

    async def export_metrics(self, metrics: List[MetricValue]) -> bool:
        """Export metrics to CloudWatch."""
        try:
            # Import boto3 only when needed
            import boto3

            if not self._cloudwatch_client:
                session = boto3.Session(
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.region
                )
                self._cloudwatch_client = session.client('cloudwatch')

            # Convert metrics to CloudWatch format
            metric_data = []
            for metric in metrics:
                if not self.should_export_metric(metric):
                    continue

                cw_metric = self._convert_to_cloudwatch_metric(metric)
                metric_data.append(cw_metric)

                # CloudWatch has a limit of 20 metrics per put_metric_data call
                if len(metric_data) >= 20:
                    await self._send_cloudwatch_batch(metric_data)
                    metric_data = []

            # Send remaining metrics
            if metric_data:
                await self._send_cloudwatch_batch(metric_data)

            logger.debug(f"Exported {len(metrics)} metrics to CloudWatch")
            return True

        except ImportError:
            logger.error("boto3 package not available for CloudWatch export")
            return False
        except Exception as e:
            logger.error(f"Error exporting to CloudWatch: {e}")
            return False

    def _convert_to_cloudwatch_metric(self, metric: MetricValue) -> Dict[str, Any]:
        """Convert metric to CloudWatch format."""
        cw_metric = {
            'MetricName': self.format_metric_name(metric),
            'Value': metric.value,
            'Unit': self._get_cloudwatch_unit(metric.metric_type),
            'Timestamp': metric.timestamp
        }

        # Add dimensions (CloudWatch's version of labels)
        if metric.labels and self.config.enable_labels:
            dimensions = []
            formatted_labels = self.format_labels(metric.labels)
            for key, value in formatted_labels.items():
                dimensions.append({
                    'Name': key,
                    'Value': value
                })
            cw_metric['Dimensions'] = dimensions

        return cw_metric

    def _get_cloudwatch_unit(self, metric_type: MetricType) -> str:
        """Get CloudWatch unit for metric type."""
        unit_map = {
            MetricType.COUNTER: "Count",
            MetricType.GAUGE: "None",
            MetricType.HISTOGRAM: "None",
            MetricType.TIMER: "Seconds"
        }
        return unit_map.get(metric_type, "None")

    async def _send_cloudwatch_batch(self, metric_data: List[Dict[str, Any]]) -> None:
        """Send batch of metrics to CloudWatch."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._cloudwatch_client.put_metric_data,
            {'Namespace': self.namespace, 'MetricData': metric_data}
        )


class JSONExporter(MetricsExporter):
    """Exporter for JSON format metrics."""

    def __init__(
        self,
        name: str = "json",
        output_file: str = "/tmp/tag_sentinel_metrics.json",
        pretty_print: bool = True,
        config: Optional[ExportConfig] = None
    ):
        """Initialize JSON exporter.

        Args:
            name: Exporter name
            output_file: Output file path
            pretty_print: Whether to format JSON nicely
            config: Export configuration
        """
        super().__init__(name, config)
        self.output_file = output_file
        self.pretty_print = pretty_print

    async def export_metrics(self, metrics: List[MetricValue]) -> bool:
        """Export metrics in JSON format."""
        try:
            # Convert metrics to JSON-serializable format
            json_metrics = []
            for metric in metrics:
                if not self.should_export_metric(metric):
                    continue

                json_metric = {
                    "name": self.format_metric_name(metric),
                    "value": metric.value,
                    "type": metric.metric_type.value,
                    "timestamp": metric.timestamp,
                    "labels": self.format_labels(metric.labels)
                }
                json_metrics.append(json_metric)

            # Create export document
            export_doc = {
                "export_time": time.time(),
                "exporter": self.name,
                "metric_count": len(json_metrics),
                "metrics": json_metrics
            }

            # Write to file
            json_str = json.dumps(
                export_doc,
                indent=2 if self.pretty_print else None,
                separators=(',', ': ') if self.pretty_print else (',', ':')
            )

            async with aiofiles.open(self.output_file, 'w') as f:
                await f.write(json_str)

            logger.debug(f"Exported {len(json_metrics)} metrics to JSON")
            return True

        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return False


class CSVExporter(MetricsExporter):
    """Exporter for CSV format metrics."""

    def __init__(
        self,
        name: str = "csv",
        output_file: str = "/tmp/tag_sentinel_metrics.csv",
        include_headers: bool = True,
        config: Optional[ExportConfig] = None
    ):
        """Initialize CSV exporter.

        Args:
            name: Exporter name
            output_file: Output file path
            include_headers: Whether to include CSV headers
            config: Export configuration
        """
        super().__init__(name, config)
        self.output_file = output_file
        self.include_headers = include_headers

    async def export_metrics(self, metrics: List[MetricValue]) -> bool:
        """Export metrics in CSV format."""
        try:
            import csv
            import io

            # Create CSV content in memory
            output = io.StringIO()
            writer = csv.writer(output)

            # Write headers
            if self.include_headers:
                headers = ["name", "value", "type", "timestamp"]
                if self.config.enable_labels:
                    # Find all unique label keys
                    label_keys = set()
                    for metric in metrics:
                        if self.should_export_metric(metric):
                            label_keys.update(metric.labels.keys())
                    headers.extend(sorted(label_keys))
                writer.writerow(headers)

            # Write metrics
            for metric in metrics:
                if not self.should_export_metric(metric):
                    continue

                row = [
                    self.format_metric_name(metric),
                    metric.value,
                    metric.metric_type.value,
                    metric.timestamp
                ]

                # Add label values
                if self.config.enable_labels and self.include_headers:
                    formatted_labels = self.format_labels(metric.labels)
                    for label_key in sorted(label_keys):
                        row.append(formatted_labels.get(label_key, ""))

                writer.writerow(row)

            # Write to file
            csv_content = output.getvalue()
            async with aiofiles.open(self.output_file, 'w') as f:
                await f.write(csv_content)

            logger.debug(f"Exported {len(metrics)} metrics to CSV")
            return True

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False


class MetricsExportManager:
    """Manager for multiple metrics exporters."""

    def __init__(self, collector: Optional[MetricsCollector] = None):
        """Initialize export manager.

        Args:
            collector: Metrics collector to export from
        """
        self.collector = collector
        self.exporters: Dict[str, MetricsExporter] = {}
        self._export_task: Optional[asyncio.Task] = None

    def add_exporter(self, exporter: MetricsExporter) -> None:
        """Add metrics exporter.

        Args:
            exporter: Exporter to add
        """
        self.exporters[exporter.name] = exporter
        logger.info(f"Added metrics exporter: {exporter.name}")

    def remove_exporter(self, name: str) -> None:
        """Remove metrics exporter.

        Args:
            name: Exporter name to remove
        """
        if name in self.exporters:
            del self.exporters[name]
            logger.info(f"Removed metrics exporter: {name}")

    async def export_all(self) -> Dict[str, bool]:
        """Export metrics using all configured exporters.

        Returns:
            Dictionary mapping exporter names to success status
        """
        if not self.collector:
            logger.warning("No metrics collector configured")
            return {}

        # Get metrics from collector
        buffered_metrics = await self.collector.flush_buffer()

        if not buffered_metrics:
            logger.debug("No metrics to export")
            return {}

        # Export using all exporters
        results = {}
        for exporter in self.exporters.values():
            if not exporter.enabled:
                continue

            try:
                success = await exporter.export_metrics(buffered_metrics)
                results[exporter.name] = success

                if success:
                    exporter.last_export_time = time.time()

            except Exception as e:
                logger.error(f"Error in exporter {exporter.name}: {e}")
                results[exporter.name] = False

        return results

    def start_periodic_export(self, interval: int = 60) -> None:
        """Start periodic export of metrics.

        Args:
            interval: Export interval in seconds
        """
        if self._export_task is None:
            self._export_task = asyncio.create_task(self._export_loop(interval))
            logger.info(f"Started periodic metrics export with {interval}s interval")

    async def _export_loop(self, interval: int) -> None:
        """Background loop for periodic metric export."""
        while True:
            try:
                results = await self.export_all()
                successful = sum(1 for success in results.values() if success)
                total = len(results)
                logger.debug(f"Metrics export completed: {successful}/{total} exporters successful")

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics export loop: {e}")
                await asyncio.sleep(interval)

    async def close(self) -> None:
        """Close export manager and clean up resources."""
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass

        logger.info("MetricsExportManager closed")


# Try to import aiofiles for file operations
try:
    import aiofiles
except ImportError:
    # Fallback to synchronous file operations
    class aiofiles:
        @staticmethod
        def open(filename, mode='r'):
            return SyncFileContext(filename, mode)

    class SyncFileContext:
        def __init__(self, filename, mode):
            self.filename = filename
            self.mode = mode
            self.file = None

        async def __aenter__(self):
            self.file = open(self.filename, self.mode)
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self.file:
                self.file.close()

        async def write(self, data):
            self.file.write(data)

        async def read(self):
            return self.file.read()