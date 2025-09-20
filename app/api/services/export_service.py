"""Export service layer for Tag Sentinel REST API.

This module implements the business logic for data transformation
and streaming export generation with memory efficiency.
"""

import json
import csv
import io
from typing import AsyncGenerator, Dict, List, Any, Optional, Literal
from datetime import datetime
import logging

from app.api.schemas.exports import (
    RequestLogExport,
    CookieExport,
    TagExport,
    DataLayerExport
)
from app.api.persistence.repositories import ExportDataRepository
from app.api.persistence.factory import RepositoryFactory

logger = logging.getLogger(__name__)


class ExportService:
    """Service for generating audit data exports in multiple formats."""

    def __init__(self, repository: Optional[ExportDataRepository] = None):
        """Initialize export service.

        Args:
            repository: Export data repository instance (defaults to in-memory)
        """
        self.repository = repository or RepositoryFactory.create_export_repository()
        logger.info(f"ExportService initialized with repository={type(self.repository).__name__}")

    async def export_requests(
        self,
        audit_id: str,
        format: Literal["json", "csv"] = "json",
        filters: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Export request logs in streaming format.

        Args:
            audit_id: Audit to export data for
            format: Export format (json/csv)
            filters: Optional filtering criteria

        Yields:
            Formatted data chunks for streaming response
        """
        logger.info(f"Starting request log export for audit {audit_id} in {format} format")

        # Get request data from repository
        requests = await self.repository.get_request_logs(audit_id, filters)

        if format == "csv":
            async for chunk in self._export_requests_csv(requests):
                yield chunk
        else:
            async for chunk in self._export_requests_json(requests):
                yield chunk

        logger.info(f"Completed request log export for audit {audit_id}")

    async def export_cookies(
        self,
        audit_id: str,
        format: Literal["json", "csv"] = "json",
        filters: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Export cookie inventory in streaming format.

        Args:
            audit_id: Audit to export data for
            format: Export format (json/csv)
            filters: Optional filtering criteria

        Yields:
            Formatted data chunks for streaming response
        """
        logger.info(f"Starting cookie export for audit {audit_id} in {format} format")

        # Get cookie data from repository
        cookies = await self.repository.get_cookies(audit_id, filters)

        if format == "csv":
            async for chunk in self._export_cookies_csv(cookies):
                yield chunk
        else:
            async for chunk in self._export_cookies_json(cookies):
                yield chunk

        logger.info(f"Completed cookie export for audit {audit_id}")

    async def export_tags(
        self,
        audit_id: str,
        format: Literal["json", "csv"] = "json",
        filters: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Export tag detection results in streaming format.

        Args:
            audit_id: Audit to export data for
            format: Export format (json/csv)
            filters: Optional filtering criteria

        Yields:
            Formatted data chunks for streaming response
        """
        logger.info(f"Starting tag export for audit {audit_id} in {format} format")

        # Get tag data from repository
        tags = await self.repository.get_tags(audit_id, filters)

        if format == "csv":
            async for chunk in self._export_tags_csv(tags):
                yield chunk
        else:
            async for chunk in self._export_tags_json(tags):
                yield chunk

        logger.info(f"Completed tag export for audit {audit_id}")

    async def export_data_layer(
        self,
        audit_id: str,
        format: Literal["json", "csv"] = "json",
        filters: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Export data layer snapshots in streaming format.

        Args:
            audit_id: Audit to export data for
            format: Export format (json/csv)
            filters: Optional filtering criteria

        Yields:
            Formatted data chunks for streaming response
        """
        logger.info(f"Starting data layer export for audit {audit_id} in {format} format")

        # Get data layer data from repository
        snapshots = await self.repository.get_data_layer_snapshots(audit_id, filters)

        if format == "csv":
            async for chunk in self._export_data_layer_csv(snapshots):
                yield chunk
        else:
            async for chunk in self._export_data_layer_json(snapshots):
                yield chunk

        logger.info(f"Completed data layer export for audit {audit_id}")

    # Private helper methods for format-specific export

    async def _export_requests_json(
        self,
        requests: List[RequestLogExport]
    ) -> AsyncGenerator[str, None]:
        """Export request logs as NDJSON (newline-delimited JSON)."""
        for request in requests:
            # Convert to JSON and yield as line
            json_line = request.model_dump_json() + "\n"
            yield json_line

    async def _export_requests_csv(
        self,
        requests: List[RequestLogExport]
    ) -> AsyncGenerator[str, None]:
        """Export request logs as CSV."""
        if not requests:
            return

        # Create CSV header
        fieldnames = list(requests[0].model_dump().keys())

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)

        # Write header
        writer.writeheader()
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        # Write data rows
        for request in requests:
            data = request.model_dump(mode='json')
            # Handle nested objects by converting to JSON strings
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    data[key] = json.dumps(value)
                elif value is None:
                    data[key] = ""

            writer.writerow(data)
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    async def _export_cookies_json(
        self,
        cookies: List[CookieExport]
    ) -> AsyncGenerator[str, None]:
        """Export cookies as NDJSON."""
        for cookie in cookies:
            json_line = cookie.model_dump_json() + "\n"
            yield json_line

    async def _export_cookies_csv(
        self,
        cookies: List[CookieExport]
    ) -> AsyncGenerator[str, None]:
        """Export cookies as CSV."""
        if not cookies:
            return

        fieldnames = list(cookies[0].model_dump().keys())

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)

        writer.writeheader()
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        for cookie in cookies:
            data = cookie.model_dump(mode='json')
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    data[key] = json.dumps(value)
                elif value is None:
                    data[key] = ""

            writer.writerow(data)
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    async def _export_tags_json(
        self,
        tags: List[TagExport]
    ) -> AsyncGenerator[str, None]:
        """Export tags as NDJSON."""
        for tag in tags:
            json_line = tag.model_dump_json() + "\n"
            yield json_line

    async def _export_tags_csv(
        self,
        tags: List[TagExport]
    ) -> AsyncGenerator[str, None]:
        """Export tags as CSV."""
        if not tags:
            return

        fieldnames = list(tags[0].model_dump().keys())

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)

        writer.writeheader()
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        for tag in tags:
            data = tag.model_dump(mode='json')
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    data[key] = json.dumps(value)
                elif value is None:
                    data[key] = ""

            writer.writerow(data)
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    async def _export_data_layer_json(
        self,
        snapshots: List[DataLayerExport]
    ) -> AsyncGenerator[str, None]:
        """Export data layer snapshots as NDJSON."""
        for snapshot in snapshots:
            json_line = snapshot.model_dump_json() + "\n"
            yield json_line

    async def _export_data_layer_csv(
        self,
        snapshots: List[DataLayerExport]
    ) -> AsyncGenerator[str, None]:
        """Export data layer snapshots as CSV."""
        if not snapshots:
            return

        fieldnames = list(snapshots[0].model_dump().keys())

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)

        writer.writeheader()
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        for snapshot in snapshots:
            data = snapshot.model_dump(mode='json')
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    data[key] = json.dumps(value)
                elif value is None:
                    data[key] = ""

            writer.writerow(data)
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)