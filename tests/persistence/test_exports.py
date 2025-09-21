"""Tests for export functionality."""

import json
import pytest
from io import StringIO
from typing import List, Dict, Any

from app.persistence.dao import AuditDAO
from app.persistence.exports import ExportService, ExportFormat
from app.persistence.models import Run


class TestRequestLogExports:
    """Test request log export functionality."""

    async def test_export_request_logs_ndjson(
        self,
        dao: AuditDAO,
        populated_run: Run
    ):
        """Test exporting request logs in NDJSON format."""
        export_service = ExportService(dao)

        chunks = []
        async for chunk in export_service.export_request_logs(
            populated_run.id,
            format=ExportFormat.NDJSON,
            batch_size=2
        ):
            chunks.append(chunk)

        # Verify NDJSON format
        assert len(chunks) > 0

        # Parse each line as JSON
        lines = "".join(chunks).strip().split("\n")
        for line in lines:
            if line.strip():  # Skip empty lines
                log_data = json.loads(line)
                assert "id" in log_data
                assert "url" in log_data
                assert "method" in log_data

    async def test_export_request_logs_json(
        self,
        dao: AuditDAO,
        populated_run: Run
    ):
        """Test exporting request logs in JSON format."""
        export_service = ExportService(dao)

        chunks = []
        async for chunk in export_service.export_request_logs(
            populated_run.id,
            format=ExportFormat.JSON,
            batch_size=2
        ):
            chunks.append(chunk)

        # Verify JSON format
        assert len(chunks) > 0

        # Should be valid JSON array
        full_json = "".join(chunks)
        logs_data = json.loads(full_json)

        assert isinstance(logs_data, list)
        assert len(logs_data) > 0

        for log_data in logs_data:
            assert "id" in log_data
            assert "url" in log_data
            assert "method" in log_data

    async def test_export_request_logs_csv(
        self,
        dao: AuditDAO,
        populated_run: Run
    ):
        """Test exporting request logs in CSV format."""
        export_service = ExportService(dao)

        chunks = []
        async for chunk in export_service.export_request_logs(
            populated_run.id,
            format=ExportFormat.CSV,
            batch_size=2
        ):
            chunks.append(chunk)

        # Verify CSV format
        assert len(chunks) > 0

        csv_content = "".join(chunks)
        lines = csv_content.strip().split("\n")

        # First line should be header
        header = lines[0]
        assert "id" in header
        assert "url" in header
        assert "method" in header

        # Should have data rows
        assert len(lines) > 1


class TestCookieExports:
    """Test cookie export functionality."""

    async def test_export_cookies_ndjson(
        self,
        dao: AuditDAO,
        populated_run: Run
    ):
        """Test exporting cookies in NDJSON format."""
        export_service = ExportService(dao)

        chunks = []
        async for chunk in export_service.export_cookies(
            populated_run.id,
            format=ExportFormat.NDJSON,
            batch_size=3
        ):
            chunks.append(chunk)

        # Verify NDJSON format
        assert len(chunks) > 0

        lines = "".join(chunks).strip().split("\n")
        for line in lines:
            if line.strip():
                cookie_data = json.loads(line)
                assert "name" in cookie_data
                assert "domain" in cookie_data

    async def test_export_cookies_json(
        self,
        dao: AuditDAO,
        populated_run: Run
    ):
        """Test exporting cookies in JSON format."""
        export_service = ExportService(dao)

        chunks = []
        async for chunk in export_service.export_cookies(
            populated_run.id,
            format=ExportFormat.JSON,
            batch_size=3
        ):
            chunks.append(chunk)

        # Verify JSON format
        full_json = "".join(chunks)
        cookies_data = json.loads(full_json)

        assert isinstance(cookies_data, list)
        assert len(cookies_data) > 0

        for cookie_data in cookies_data:
            assert "name" in cookie_data
            assert "domain" in cookie_data

    async def test_export_cookies_with_filter(
        self,
        dao: AuditDAO,
        populated_run: Run
    ):
        """Test exporting cookies with first-party filter."""
        export_service = ExportService(dao)

        chunks = []
        async for chunk in export_service.export_cookies(
            populated_run.id,
            format=ExportFormat.JSON,
            first_party_only=True
        ):
            chunks.append(chunk)

        # Verify filtering works
        full_json = "".join(chunks)
        cookies_data = json.loads(full_json)

        for cookie_data in cookies_data:
            assert cookie_data["first_party"] is True

    async def test_export_cookies_empty_filtered_batches(
        self,
        dao: AuditDAO,
        populated_run: Run
    ):
        """Test that empty filtered batches don't break JSON format."""
        export_service = ExportService(dao)

        # Test with filter that might result in empty batches
        chunks = []
        async for chunk in export_service.export_cookies(
            populated_run.id,
            format=ExportFormat.JSON,
            first_party_only=False,  # This might filter out all cookies
            batch_size=1
        ):
            chunks.append(chunk)

        # Should still produce valid JSON even if some batches are empty
        full_json = "".join(chunks)
        cookies_data = json.loads(full_json)
        assert isinstance(cookies_data, list)


class TestRuleFailureExports:
    """Test rule failure export functionality."""

    async def test_export_rule_failures_with_data(
        self,
        dao: AuditDAO,
        sample_run: Run
    ):
        """Test exporting rule failures when data exists."""
        # Create some rule failures
        await dao.create_rule_failure(
            run_id=sample_run.id,
            rule_id="test_rule_1",
            rule_name="Test Rule 1",
            severity="error",
            message="Test error message"
        )
        await dao.create_rule_failure(
            run_id=sample_run.id,
            rule_id="test_rule_2",
            rule_name="Test Rule 2",
            severity="warning",
            message="Test warning message"
        )
        await dao.commit()

        export_service = ExportService(dao)

        chunks = []
        async for chunk in export_service.export_rule_failures(
            sample_run.id,
            format=ExportFormat.JSON
        ):
            chunks.append(chunk)

        # Verify JSON format
        full_json = "".join(chunks)
        failures_data = json.loads(full_json)

        assert isinstance(failures_data, list)
        assert len(failures_data) >= 2

        for failure_data in failures_data:
            assert "rule_id" in failure_data
            assert "severity" in failure_data
            assert "message" in failure_data

    async def test_export_rule_failures_empty(
        self,
        dao: AuditDAO,
        sample_run: Run
    ):
        """Test exporting rule failures when no data exists."""
        export_service = ExportService(dao)

        chunks = []
        async for chunk in export_service.export_rule_failures(
            sample_run.id,
            format=ExportFormat.JSON
        ):
            chunks.append(chunk)

        # Should produce empty JSON array
        full_json = "".join(chunks)
        failures_data = json.loads(full_json)

        assert isinstance(failures_data, list)
        assert len(failures_data) == 0


class TestTagInventoryExport:
    """Test tag inventory export functionality."""

    async def test_export_tag_inventory_json(
        self,
        dao: AuditDAO,
        populated_run: Run
    ):
        """Test exporting tag inventory in JSON format."""
        export_service = ExportService(dao)

        result = await export_service.export_tag_inventory(
            populated_run.id,
            format=ExportFormat.JSON
        )

        # Verify JSON format
        inventory_data = json.loads(result)

        assert isinstance(inventory_data, list)
        assert len(inventory_data) > 0

        for tag in inventory_data:
            assert "vendor" in tag
            assert "name" in tag
            assert "id" in tag
            assert "count" in tag
            assert "pages" in tag
            assert isinstance(tag["count"], int)
            assert isinstance(tag["pages"], int)
            assert tag["count"] > 0
            assert tag["pages"] > 0

    async def test_export_tag_inventory_csv(
        self,
        dao: AuditDAO,
        populated_run: Run
    ):
        """Test exporting tag inventory in CSV format."""
        export_service = ExportService(dao)

        result = await export_service.export_tag_inventory(
            populated_run.id,
            format=ExportFormat.CSV
        )

        # Verify CSV format
        lines = result.strip().split("\n")

        # First line should be header
        header = lines[0]
        assert "vendor" in header
        assert "name" in header
        assert "count" in header
        assert "pages" in header

        # Should have data rows
        assert len(lines) > 1

    async def test_export_tag_inventory_ndjson(
        self,
        dao: AuditDAO,
        populated_run: Run
    ):
        """Test exporting tag inventory in NDJSON format."""
        export_service = ExportService(dao)

        result = await export_service.export_tag_inventory(
            populated_run.id,
            format=ExportFormat.NDJSON
        )

        # Verify NDJSON format
        lines = result.strip().split("\n")
        for line in lines:
            if line.strip():
                tag_data = json.loads(line)
                assert "vendor" in tag_data
                assert "name" in tag_data


class TestExportSummary:
    """Test export summary functionality."""

    async def test_get_export_summary(
        self,
        dao: AuditDAO,
        populated_run: Run
    ):
        """Test getting export summary for a run."""
        export_service = ExportService(dao)

        summary = await export_service.get_export_summary(populated_run.id)

        assert "run_id" in summary
        assert "run_status" in summary
        assert "export_timestamp" in summary
        assert "available_exports" in summary
        assert "run_statistics" in summary

        exports = summary["available_exports"]
        assert "request_logs" in exports
        assert "cookies" in exports
        assert "tag_inventory" in exports
        assert "rule_failures" in exports

        # Verify each export type has count and formats
        for export_type in exports.values():
            assert "count" in export_type
            assert "formats" in export_type
            assert isinstance(export_type["formats"], list)

    async def test_get_export_summary_nonexistent_run(
        self,
        dao: AuditDAO
    ):
        """Test export summary for non-existent run."""
        export_service = ExportService(dao)

        summary = await export_service.get_export_summary(99999)

        assert summary == {}


class TestExportUtilities:
    """Test export utility functions."""

    def test_get_content_type(self):
        """Test content type detection."""
        export_service = ExportService(None)

        assert export_service.get_content_type(ExportFormat.JSON) == "application/json"
        assert export_service.get_content_type(ExportFormat.NDJSON) == "application/x-ndjson"
        assert export_service.get_content_type(ExportFormat.CSV) == "text/csv"

    def test_get_file_extension(self):
        """Test file extension detection."""
        export_service = ExportService(None)

        assert export_service.get_file_extension(ExportFormat.JSON) == ".json"
        assert export_service.get_file_extension(ExportFormat.NDJSON) == ".ndjson"
        assert export_service.get_file_extension(ExportFormat.CSV) == ".csv"