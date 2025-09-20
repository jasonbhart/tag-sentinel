"""Unit tests for the ExportService class.

Tests the business logic for data transformation and streaming export generation
with memory efficiency and format validation.
"""

import pytest
import json
import csv
import io
from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, Mock

from app.api.services.export_service import ExportService
from app.api.schemas.exports import (
    RequestLogExport,
    CookieExport,
    TagExport,
    DataLayerExport
)
from app.api.persistence.repositories import InMemoryExportDataRepository


class TestExportService:
    """Test suite for ExportService."""

    @pytest.fixture
    def mock_repository(self):
        """Create a mock repository with test data."""
        repository = AsyncMock(spec=InMemoryExportDataRepository)

        # Mock request log data
        repository.get_request_logs.return_value = [
            RequestLogExport(
                id="req_001",
                audit_id="test_audit_001",
                page_url="https://example.com/",
                url="https://example.com/analytics.js",
                method="GET",
                resource_type="script",
                status="success",
                status_code=200,
                timestamp=datetime(2024, 1, 15, 10, 35, 0),
                response_time=245,
                is_analytics=True,
                analytics_vendor="Google Analytics"
            ),
            RequestLogExport(
                id="req_002",
                audit_id="test_audit_001",
                page_url="https://example.com/",
                url="https://example.com/page.html",
                method="GET",
                resource_type="document",
                status="success",
                status_code=200,
                timestamp=datetime(2024, 1, 15, 10, 35, 1),
                response_time=1200,
                is_analytics=False,
                analytics_vendor=None
            )
        ]

        # Mock cookie data
        repository.get_cookies.return_value = [
            CookieExport(
                audit_id="test_audit_001",
                page_url="https://example.com/",
                name="_ga",
                domain=".example.com",
                path="/",
                value="GA1.2.123456789.1234567890",
                secure=False,
                http_only=False,
                same_site="Lax",
                discovered_at=datetime(2024, 1, 15, 10, 35, 0),
                source="javascript",
                category="analytics",
                vendor="Google Analytics",
                is_essential=False
            ),
            CookieExport(
                audit_id="test_audit_001",
                page_url="https://example.com/",
                name="session_id",
                domain="example.com",
                path="/",
                value="sess_abc123def456",
                secure=True,
                http_only=True,
                same_site="Strict",
                discovered_at=datetime(2024, 1, 15, 10, 35, 1),
                source="http_header",
                category="functional",
                vendor=None,
                is_essential=True
            )
        ]

        # Mock tag data
        repository.get_tags.return_value = [
            TagExport(
                audit_id="test_audit_001",
                page_url="https://example.com/",
                tag_id="tag_001",
                vendor="Google Analytics 4",
                tag_type="pageview",
                implementation_method="gtm",
                detected_at=datetime(2024, 1, 15, 10, 35, 0),
                confidence_score=0.95,
                detection_method="script_analysis",
                has_errors=False,
                load_time=150
            ),
            TagExport(
                audit_id="test_audit_001",
                page_url="https://example.com/",
                tag_id="tag_002",
                vendor="Facebook Pixel",
                tag_type="conversion",
                implementation_method="script_tag",
                detected_at=datetime(2024, 1, 15, 10, 35, 1),
                confidence_score=0.87,
                detection_method="pixel_analysis",
                has_errors=True,
                load_time=890
            )
        ]

        # Mock data layer data
        repository.get_data_layer_snapshots.return_value = [
            DataLayerExport(
                audit_id="test_audit_001",
                page_url="https://example.com/",
                snapshot_id="dl_001",
                captured_at=datetime(2024, 1, 15, 10, 35, 0),
                trigger_event="page_load",
                data={"event": "page_view", "page_title": "Home"},
                total_properties=5,
                nested_levels=1,
                contains_pii=False,
                schema_valid=True
            ),
            DataLayerExport(
                audit_id="test_audit_001",
                page_url="https://example.com/checkout",
                snapshot_id="dl_002",
                captured_at=datetime(2024, 1, 15, 10, 35, 1),
                trigger_event="purchase",
                data={"event": "purchase", "value": 99.99, "currency": "USD"},
                total_properties=12,
                nested_levels=2,
                contains_pii=True,
                schema_valid=True
            )
        ]

        return repository

    @pytest.fixture
    def export_service(self, mock_repository):
        """Create an ExportService instance with mock repository."""
        return ExportService(repository=mock_repository)

    @pytest.mark.asyncio
    async def test_export_requests_json(self, export_service):
        """Test request log export in JSON format."""
        chunks = []
        async for chunk in export_service.export_requests("test_audit_001", "json"):
            chunks.append(chunk)

        # Should have one line per request
        assert len(chunks) == 2

        # Parse each JSON line
        for chunk in chunks:
            data = json.loads(chunk.strip())
            assert "audit_id" in data
            assert "id" in data
            assert "url" in data
            assert data["audit_id"] == "test_audit_001"

    @pytest.mark.asyncio
    async def test_export_requests_csv(self, export_service):
        """Test request log export in CSV format."""
        chunks = []
        async for chunk in export_service.export_requests("test_audit_001", "csv"):
            chunks.append(chunk)

        # Combine chunks and parse CSV
        csv_content = "".join(chunks)
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)

        # Should have 2 data rows
        assert len(rows) == 2

        # Verify headers and data
        assert "audit_id" in reader.fieldnames
        assert "url" in reader.fieldnames
        assert rows[0]["audit_id"] == "test_audit_001"
        assert "example.com" in rows[0]["url"]

    @pytest.mark.asyncio
    async def test_export_cookies_json(self, export_service):
        """Test cookie export in JSON format."""
        chunks = []
        async for chunk in export_service.export_cookies("test_audit_001", "json"):
            chunks.append(chunk)

        assert len(chunks) == 2

        # Verify JSON structure
        for chunk in chunks:
            data = json.loads(chunk.strip())
            assert "audit_id" in data
            assert "name" in data
            assert "domain" in data
            assert data["audit_id"] == "test_audit_001"

    @pytest.mark.asyncio
    async def test_export_cookies_csv(self, export_service):
        """Test cookie export in CSV format."""
        chunks = []
        async for chunk in export_service.export_cookies("test_audit_001", "csv"):
            chunks.append(chunk)

        csv_content = "".join(chunks)
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)

        assert len(rows) == 2
        assert "name" in reader.fieldnames
        assert "domain" in reader.fieldnames
        assert rows[0]["name"] in ["_ga", "session_id"]

    @pytest.mark.asyncio
    async def test_export_tags_json(self, export_service):
        """Test tag export in JSON format."""
        chunks = []
        async for chunk in export_service.export_tags("test_audit_001", "json"):
            chunks.append(chunk)

        assert len(chunks) == 2

        for chunk in chunks:
            data = json.loads(chunk.strip())
            assert "audit_id" in data
            assert "vendor" in data
            assert "tag_type" in data
            assert data["audit_id"] == "test_audit_001"

    @pytest.mark.asyncio
    async def test_export_data_layer_json(self, export_service):
        """Test data layer export in JSON format."""
        chunks = []
        async for chunk in export_service.export_data_layer("test_audit_001", "json"):
            chunks.append(chunk)

        assert len(chunks) == 2

        for chunk in chunks:
            data = json.loads(chunk.strip())
            assert "audit_id" in data
            assert "trigger_event" in data
            assert "total_properties" in data
            assert data["audit_id"] == "test_audit_001"

    @pytest.mark.asyncio
    async def test_export_with_filters(self, export_service, mock_repository):
        """Test export with filtering applied."""
        filters = {"status": "success", "analytics_only": True}

        chunks = []
        async for chunk in export_service.export_requests("test_audit_001", "json", filters):
            chunks.append(chunk)

        # Verify that filters were passed to repository
        mock_repository.get_request_logs.assert_called_once_with("test_audit_001", filters)

    @pytest.mark.asyncio
    async def test_export_empty_data(self, export_service):
        """Test export behavior with empty data."""
        # Mock empty data
        export_service.repository.get_request_logs.return_value = []

        chunks = []
        async for chunk in export_service.export_requests("empty_audit", "csv"):
            chunks.append(chunk)

        # Should handle empty data gracefully
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_csv_nested_data_handling(self, export_service):
        """Test CSV export properly handles nested objects."""
        chunks = []
        async for chunk in export_service.export_data_layer("test_audit_001", "csv"):
            chunks.append(chunk)

        csv_content = "".join(chunks)
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)

        # data should be JSON-encoded in CSV
        for row in rows:
            if "data" in row:
                # Should be valid JSON string
                data_content = json.loads(row["data"])
                assert isinstance(data_content, dict)

    @pytest.mark.asyncio
    async def test_csv_null_value_handling(self, export_service):
        """Test CSV export properly handles null values."""
        chunks = []
        async for chunk in export_service.export_requests("test_audit_001", "csv"):
            chunks.append(chunk)

        csv_content = "".join(chunks)
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)

        # Null values should be converted to empty strings
        for row in rows:
            for value in row.values():
                assert value != "None"  # Should not have string "None"

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, export_service):
        """Test that exports are streaming and memory-efficient."""
        chunk_count = 0
        total_size = 0

        async for chunk in export_service.export_requests("test_audit_001", "json"):
            chunk_count += 1
            total_size += len(chunk)
            # Each chunk should be reasonably small (one record)
            assert len(chunk) < 10000  # Reasonable single record size

        # Should produce chunks incrementally, not all at once
        assert chunk_count > 0
        assert total_size > 0

    @pytest.mark.asyncio
    async def test_export_format_validation(self, export_service):
        """Test that export formats are properly validated."""
        # JSON format should produce valid JSON lines
        chunks = []
        async for chunk in export_service.export_requests("test_audit_001", "json"):
            chunks.append(chunk)

        for chunk in chunks:
            # Each chunk should be valid JSON followed by newline
            assert chunk.endswith("\n")
            data = json.loads(chunk.strip())
            assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_repository_error_handling(self, export_service, mock_repository):
        """Test error handling when repository fails."""
        # Mock repository error
        mock_repository.get_request_logs.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            chunks = []
            async for chunk in export_service.export_requests("test_audit_001", "json"):
                chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_concurrent_exports(self, export_service):
        """Test that multiple concurrent exports work correctly."""
        import asyncio

        # Start multiple exports concurrently
        tasks = []
        for i in range(3):
            task = asyncio.create_task(
                self._collect_chunks(export_service.export_requests(f"audit_{i}", "json"))
            )
            tasks.append(task)

        # Wait for all exports to complete
        results = await asyncio.gather(*tasks)

        # Each should have produced chunks
        for chunks in results:
            assert len(chunks) > 0

    async def _collect_chunks(self, generator):
        """Helper method to collect chunks from async generator."""
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)
        return chunks

    @pytest.mark.asyncio
    async def test_large_dataset_handling(self, export_service, mock_repository):
        """Test handling of large datasets."""
        # Mock large dataset
        large_dataset = []
        for i in range(1000):
            large_dataset.append(RequestLogExport(
                id=f"req_{i:04d}",
                audit_id="large_audit",
                page_url="https://example.com/",
                url=f"https://example.com/request_{i}.js",
                method="GET",
                resource_type="script",
                status="success",
                status_code=200,
                timestamp=datetime(2024, 1, 15, 10, 35, 0),
                response_time=100 + i,
                is_analytics=True,
                analytics_vendor="Test Vendor"
            ))

        mock_repository.get_request_logs.return_value = large_dataset

        chunk_count = 0
        async for chunk in export_service.export_requests("large_audit", "json"):
            chunk_count += 1

        # Should process all records
        assert chunk_count == 1000