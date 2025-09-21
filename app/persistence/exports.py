"""Export services for Tag Sentinel persistence layer.

This module provides streaming export functionality for audit data in JSON and CSV formats,
optimized for large datasets with memory-efficient processing.
"""

import csv
import json
from datetime import datetime
from enum import Enum
from io import StringIO
from typing import AsyncGenerator, Dict, Any, List, Optional, Union
from dataclasses import asdict

from .dao import AuditDAO
from .models import RequestLog, Cookie, RuleFailure, Artifact


class ExportFormat(str, Enum):
    """Supported export formats."""
    JSON = "json"
    NDJSON = "ndjson"  # Newline-delimited JSON
    CSV = "csv"


class ExportService:
    """Service for streaming exports of audit data."""

    def __init__(self, dao: AuditDAO):
        """Initialize export service with DAO."""
        self.dao = dao

    # ============= Request Log Exports =============

    async def export_request_logs(
        self,
        run_id: int,
        format: ExportFormat = ExportFormat.NDJSON,
        batch_size: int = 1000
    ) -> AsyncGenerator[str, None]:
        """Stream request logs for a run in specified format."""
        if format == ExportFormat.CSV:
            # Yield CSV header first
            yield self._get_request_log_csv_header()
        elif format == ExportFormat.JSON:
            # Start JSON array
            yield "[\n"
            first_batch = True

        async for logs in self.dao.stream_request_logs_for_run(run_id, batch_size):
            if format == ExportFormat.JSON:
                # JSON array format with proper streaming
                for i, log in enumerate(logs):
                    if not first_batch or i > 0:
                        yield ",\n"
                    yield json.dumps(self._serialize_request_log(log), indent=2)
                first_batch = False
            elif format == ExportFormat.NDJSON:
                # Newline-delimited JSON (streaming friendly)
                for log in logs:
                    yield json.dumps(self._serialize_request_log(log)) + "\n"
            elif format == ExportFormat.CSV:
                # CSV format
                yield self._serialize_request_logs_csv(logs)

        if format == ExportFormat.JSON:
            # Close JSON array
            yield "\n]"

    def _serialize_request_log(self, log: RequestLog) -> Dict[str, Any]:
        """Convert RequestLog to serializable dictionary."""
        return {
            'id': log.id,
            'page_result_id': log.page_result_id,
            'url': log.url,
            'method': log.method,
            'resource_type': log.resource_type,
            'status_code': log.status_code,
            'status_text': log.status_text,
            'request_headers': log.request_headers_json,
            'response_headers': log.response_headers_json,
            'timings': log.timings_json,
            'start_time': log.start_time.isoformat() if log.start_time else None,
            'end_time': log.end_time.isoformat() if log.end_time else None,
            'duration_ms': log.duration_ms,
            'sizes': log.sizes_json,
            'vendor_tags': log.vendor_tags_json,
            'success': log.success,
            'error_text': log.error_text,
            'protocol': log.protocol,
            'remote_address': log.remote_address,
            'host': log.host
        }

    def _get_request_log_csv_header(self) -> str:
        """Get CSV header for request logs."""
        headers = [
            'id', 'page_result_id', 'url', 'method', 'resource_type',
            'status_code', 'status_text', 'start_time', 'end_time', 'duration_ms',
            'success', 'error_text', 'protocol', 'remote_address', 'host',
            'request_headers_json', 'response_headers_json', 'timings_json',
            'sizes_json', 'vendor_tags_json'
        ]
        return ','.join(headers) + '\n'

    def _serialize_request_logs_csv(self, logs: List[RequestLog]) -> str:
        """Convert request logs to CSV format."""
        output = StringIO()
        writer = csv.writer(output)

        for log in logs:
            row = [
                log.id,
                log.page_result_id,
                log.url,
                log.method,
                log.resource_type,
                log.status_code,
                log.status_text,
                log.start_time.isoformat() if log.start_time else '',
                log.end_time.isoformat() if log.end_time else '',
                log.duration_ms,
                log.success,
                log.error_text or '',
                log.protocol or '',
                log.remote_address or '',
                log.host,
                json.dumps(log.request_headers_json) if log.request_headers_json else '',
                json.dumps(log.response_headers_json) if log.response_headers_json else '',
                json.dumps(log.timings_json) if log.timings_json else '',
                json.dumps(log.sizes_json) if log.sizes_json else '',
                json.dumps(log.vendor_tags_json) if log.vendor_tags_json else ''
            ]
            writer.writerow(row)

        content = output.getvalue()
        output.close()
        return content

    # ============= Cookie Exports =============

    async def export_cookies(
        self,
        run_id: int,
        format: ExportFormat = ExportFormat.NDJSON,
        batch_size: int = 1000,
        first_party_only: Optional[bool] = None
    ) -> AsyncGenerator[str, None]:
        """Stream cookies for a run in specified format."""
        if format == ExportFormat.CSV:
            yield self._get_cookie_csv_header()
        elif format == ExportFormat.JSON:
            # Start JSON array
            yield "[\n"
            first_batch = True

        async for cookies in self.dao.stream_cookies_for_run(run_id, batch_size):
            # Apply client-side filtering if needed
            if first_party_only is not None:
                cookies = [c for c in cookies if c.first_party == first_party_only]

            if format == ExportFormat.JSON and cookies:
                # JSON array format with proper streaming
                for i, cookie in enumerate(cookies):
                    if not first_batch or i > 0:
                        yield ",\n"
                    yield json.dumps(self._serialize_cookie(cookie), indent=2)
                first_batch = False
            elif format == ExportFormat.NDJSON:
                for cookie in cookies:
                    yield json.dumps(self._serialize_cookie(cookie)) + "\n"
            elif format == ExportFormat.CSV:
                yield self._serialize_cookies_csv(cookies)

        if format == ExportFormat.JSON:
            # Close JSON array
            yield "\n]"

    def _serialize_cookie(self, cookie: Cookie) -> Dict[str, Any]:
        """Convert Cookie to serializable dictionary."""
        return {
            'id': cookie.id,
            'page_result_id': cookie.page_result_id,
            'name': cookie.name,
            'domain': cookie.domain,
            'path': cookie.path,
            'expires': cookie.expires.isoformat() if cookie.expires else None,
            'max_age': cookie.max_age,
            'size': cookie.size,
            'secure': cookie.secure,
            'http_only': cookie.http_only,
            'same_site': cookie.same_site,
            'first_party': cookie.first_party,
            'essential': cookie.essential,
            'is_session': cookie.is_session,
            'value_redacted': cookie.value_redacted,
            'metadata': cookie.metadata_json,
            'set_time': cookie.set_time.isoformat() if cookie.set_time else None,
            'modified_time': cookie.modified_time.isoformat() if cookie.modified_time else None,
            'cookie_key': cookie.cookie_key
        }

    def _get_cookie_csv_header(self) -> str:
        """Get CSV header for cookies."""
        headers = [
            'id', 'page_result_id', 'name', 'domain', 'path', 'expires', 'max_age',
            'size', 'secure', 'http_only', 'same_site', 'first_party', 'essential',
            'is_session', 'value_redacted', 'set_time', 'modified_time',
            'cookie_key', 'metadata_json'
        ]
        return ','.join(headers) + '\n'

    def _serialize_cookies_csv(self, cookies: List[Cookie]) -> str:
        """Convert cookies to CSV format."""
        output = StringIO()
        writer = csv.writer(output)

        for cookie in cookies:
            row = [
                cookie.id,
                cookie.page_result_id,
                cookie.name,
                cookie.domain,
                cookie.path,
                cookie.expires.isoformat() if cookie.expires else '',
                cookie.max_age,
                cookie.size,
                cookie.secure,
                cookie.http_only,
                cookie.same_site or '',
                cookie.first_party,
                cookie.essential,
                cookie.is_session,
                cookie.value_redacted,
                cookie.set_time.isoformat() if cookie.set_time else '',
                cookie.modified_time.isoformat() if cookie.modified_time else '',
                cookie.cookie_key,
                json.dumps(cookie.metadata_json) if cookie.metadata_json else ''
            ]
            writer.writerow(row)

        content = output.getvalue()
        output.close()
        return content

    # ============= Tag Inventory Export =============

    async def export_tag_inventory(
        self,
        run_id: int,
        format: ExportFormat = ExportFormat.JSON
    ) -> str:
        """Export aggregated tag inventory for a run."""
        inventory = await self.dao.get_tag_inventory_for_run(run_id)

        if format == ExportFormat.CSV:
            return self._serialize_tag_inventory_csv(inventory)
        elif format == ExportFormat.NDJSON:
            return '\n'.join(json.dumps(tag) for tag in inventory) + '\n'
        else:
            return json.dumps(inventory, indent=2)

    def _serialize_tag_inventory_csv(self, inventory: List[Dict[str, Any]]) -> str:
        """Convert tag inventory to CSV format."""
        if not inventory:
            return "vendor,name,id,count,pages\n"

        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(['vendor', 'name', 'id', 'count', 'pages'])

        # Write data
        for tag in inventory:
            writer.writerow([
                tag.get('vendor', ''),
                tag.get('name', ''),
                tag.get('id', ''),
                tag.get('count', 0),
                tag.get('pages', 0)
            ])

        content = output.getvalue()
        output.close()
        return content

    # ============= Rule Failures Export =============

    async def export_rule_failures(
        self,
        run_id: int,
        format: ExportFormat = ExportFormat.NDJSON,
        severity_filter: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream rule failures for a run."""
        # Get all rule failures for the run
        failures = await self.dao.get_rule_failures_for_run(
            run_id=run_id,
            severity=severity_filter,
            limit=10000  # Reasonable limit for rule failures
        )

        if format == ExportFormat.CSV:
            yield self._get_rule_failure_csv_header()
        elif format == ExportFormat.JSON:
            yield "[\n"

        if format == ExportFormat.JSON:
            for i, failure in enumerate(failures):
                if i > 0:
                    yield ",\n"
                yield json.dumps(self._serialize_rule_failure(failure), indent=2)
            yield "\n]"
        elif format == ExportFormat.NDJSON:
            for failure in failures:
                yield json.dumps(self._serialize_rule_failure(failure)) + "\n"
        elif format == ExportFormat.CSV:
            yield self._serialize_rule_failures_csv(failures)

    def _serialize_rule_failure(self, failure: RuleFailure) -> Dict[str, Any]:
        """Convert RuleFailure to serializable dictionary."""
        return {
            'id': failure.id,
            'run_id': failure.run_id,
            'rule_id': failure.rule_id,
            'rule_name': failure.rule_name,
            'severity': failure.severity,
            'message': failure.message,
            'page_url': failure.page_url,
            'details': failure.details_json,
            'detected_at': failure.detected_at.isoformat() if failure.detected_at else None
        }

    def _get_rule_failure_csv_header(self) -> str:
        """Get CSV header for rule failures."""
        headers = [
            'id', 'run_id', 'rule_id', 'rule_name', 'severity', 'message',
            'page_url', 'detected_at', 'details_json'
        ]
        return ','.join(headers) + '\n'

    def _serialize_rule_failures_csv(self, failures: List[RuleFailure]) -> str:
        """Convert rule failures to CSV format."""
        output = StringIO()
        writer = csv.writer(output)

        for failure in failures:
            row = [
                failure.id,
                failure.run_id,
                failure.rule_id,
                failure.rule_name or '',
                failure.severity,
                failure.message,
                failure.page_url or '',
                failure.detected_at.isoformat() if failure.detected_at else '',
                json.dumps(failure.details_json) if failure.details_json else ''
            ]
            writer.writerow(row)

        content = output.getvalue()
        output.close()
        return content

    # ============= Export Utilities =============

    async def get_export_summary(self, run_id: int) -> Dict[str, Any]:
        """Get summary of available export data for a run."""
        stats = await self.dao.get_run_statistics(run_id)

        if not stats:
            return {}

        return {
            'run_id': run_id,
            'run_status': stats['status'],
            'export_timestamp': datetime.utcnow().isoformat(),
            'available_exports': {
                'request_logs': {
                    'count': stats['requests']['total'],
                    'formats': ['json', 'ndjson', 'csv']
                },
                'cookies': {
                    'count': stats['cookies']['total'],
                    'formats': ['json', 'ndjson', 'csv']
                },
                'tag_inventory': {
                    'count': 'varies',  # Aggregated data
                    'formats': ['json', 'ndjson', 'csv']
                },
                'rule_failures': {
                    'count': stats['rule_failures']['total'],
                    'formats': ['json', 'ndjson', 'csv']
                }
            },
            'run_statistics': stats
        }

    def get_content_type(self, format: ExportFormat) -> str:
        """Get appropriate Content-Type header for export format."""
        content_types = {
            ExportFormat.JSON: 'application/json',
            ExportFormat.NDJSON: 'application/x-ndjson',
            ExportFormat.CSV: 'text/csv'
        }
        return content_types.get(format, 'application/octet-stream')

    def get_file_extension(self, format: ExportFormat) -> str:
        """Get appropriate file extension for export format."""
        extensions = {
            ExportFormat.JSON: '.json',
            ExportFormat.NDJSON: '.ndjson',
            ExportFormat.CSV: '.csv'
        }
        return extensions.get(format, '.txt')