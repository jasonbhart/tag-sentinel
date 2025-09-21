"""Tests for DAO (Data Access Object) operations."""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any

from app.persistence.dao import AuditDAO
from app.persistence.models import Run, PageResult, RequestLog, Cookie


class TestRunOperations:
    """Test Run CRUD operations."""

    async def test_create_run(self, dao: AuditDAO, sample_run_data: Dict[str, Any]):
        """Test creating a new run."""
        run = await dao.create_run(**sample_run_data)

        assert run.id is not None
        assert run.name == sample_run_data["name"]
        assert run.environment == sample_run_data["environment"]
        assert run.start_url == sample_run_data["start_url"]
        assert run.status == sample_run_data["status"]

    async def test_get_run(self, dao: AuditDAO, sample_run: Run):
        """Test retrieving a run by ID."""
        retrieved_run = await dao.get_run(sample_run.id)

        assert retrieved_run is not None
        assert retrieved_run.id == sample_run.id
        assert retrieved_run.name == sample_run.name

    async def test_get_nonexistent_run(self, dao: AuditDAO):
        """Test retrieving a non-existent run."""
        run = await dao.get_run(99999)
        assert run is None

    async def test_update_run_status(self, dao: AuditDAO, sample_run: Run):
        """Test updating run status."""
        await dao.update_run_status(sample_run.id, "failed", "Test error")
        await dao.commit()

        updated_run = await dao.get_run(sample_run.id)
        assert updated_run.status == "failed"
        assert updated_run.error_message == "Test error"

    async def test_list_runs(self, dao: AuditDAO, sample_run: Run):
        """Test listing runs with pagination."""
        runs, total = await dao.list_runs(limit=10, offset=0)

        assert total >= 1
        assert len(runs) >= 1
        assert any(run.id == sample_run.id for run in runs)

    async def test_list_runs_by_environment(self, dao: AuditDAO, sample_run: Run):
        """Test filtering runs by environment."""
        runs, total = await dao.list_runs(environment="test")

        assert total >= 1
        assert all(run.environment == "test" for run in runs)


class TestPageResultOperations:
    """Test PageResult CRUD operations."""

    async def test_create_page_result(
        self,
        dao: AuditDAO,
        sample_run: Run,
        sample_page_result_data: Dict[str, Any]
    ):
        """Test creating a page result."""
        page_result = await dao.create_page_result(
            run_id=sample_run.id,
            **sample_page_result_data
        )

        assert page_result.id is not None
        assert page_result.run_id == sample_run.id
        assert page_result.url == sample_page_result_data["url"]
        assert page_result.status_code == sample_page_result_data["status_code"]

    async def test_get_page_results_for_run(
        self,
        dao: AuditDAO,
        sample_page_result: PageResult
    ):
        """Test retrieving page results for a run."""
        page_results = await dao.get_page_results_for_run(sample_page_result.run_id)

        assert len(page_results) >= 1
        assert any(pr.id == sample_page_result.id for pr in page_results)

    async def test_get_page_result_by_url(
        self,
        dao: AuditDAO,
        sample_page_result: PageResult
    ):
        """Test finding page result by URL."""
        found_page = await dao.get_page_result_by_url(
            sample_page_result.run_id,
            sample_page_result.url
        )

        assert found_page is not None
        assert found_page.id == sample_page_result.id


class TestRequestLogOperations:
    """Test RequestLog operations."""

    async def test_create_request_log(
        self,
        dao: AuditDAO,
        sample_page_result: PageResult,
        sample_request_log_data: Dict[str, Any]
    ):
        """Test creating a request log."""
        request_log = await dao.create_request_log(
            page_result_id=sample_page_result.id,
            **sample_request_log_data
        )

        assert request_log.id is not None
        assert request_log.page_result_id == sample_page_result.id
        assert request_log.url == sample_request_log_data["url"]
        assert request_log.vendor_tags_json == sample_request_log_data["vendor_tags_json"]

    async def test_stream_request_logs_for_run(
        self,
        dao: AuditDAO,
        populated_run: Run
    ):
        """Test streaming request logs for export."""
        batches = []
        async for batch in dao.stream_request_logs_for_run(populated_run.id, batch_size=2):
            batches.append(batch)

        assert len(batches) >= 1
        total_logs = sum(len(batch) for batch in batches)
        assert total_logs >= 6  # 3 pages * 2 requests per page


class TestCookieOperations:
    """Test Cookie operations."""

    async def test_create_cookie(
        self,
        dao: AuditDAO,
        sample_page_result: PageResult,
        sample_cookie_data: Dict[str, Any]
    ):
        """Test creating a cookie."""
        cookie = await dao.create_cookie(
            page_result_id=sample_page_result.id,
            **sample_cookie_data
        )

        assert cookie.id is not None
        assert cookie.page_result_id == sample_page_result.id
        assert cookie.name == sample_cookie_data["name"]
        assert cookie.first_party == sample_cookie_data["first_party"]

    async def test_stream_cookies_for_run(
        self,
        dao: AuditDAO,
        populated_run: Run
    ):
        """Test streaming cookies for export."""
        batches = []
        async for batch in dao.stream_cookies_for_run(populated_run.id, batch_size=3):
            batches.append(batch)

        assert len(batches) >= 1
        total_cookies = sum(len(batch) for batch in batches)
        assert total_cookies >= 6  # 3 pages * 2 cookies per page


class TestRuleFailureOperations:
    """Test RuleFailure operations."""

    async def test_create_rule_failure(
        self,
        dao: AuditDAO,
        sample_run: Run,
        sample_rule_failure_data: Dict[str, Any]
    ):
        """Test creating a rule failure."""
        rule_failure = await dao.create_rule_failure(
            run_id=sample_run.id,
            **sample_rule_failure_data
        )

        assert rule_failure.id is not None
        assert rule_failure.run_id == sample_run.id
        assert rule_failure.rule_id == sample_rule_failure_data["rule_id"]
        assert rule_failure.severity == sample_rule_failure_data["severity"]

    async def test_get_rule_failures_for_run(
        self,
        dao: AuditDAO,
        sample_run: Run,
        sample_rule_failure_data: Dict[str, Any]
    ):
        """Test retrieving rule failures for a run."""
        # Create a rule failure first
        await dao.create_rule_failure(
            run_id=sample_run.id,
            **sample_rule_failure_data
        )
        await dao.commit()

        failures = await dao.get_rule_failures_for_run(sample_run.id)

        assert len(failures) >= 1
        assert any(f.rule_id == sample_rule_failure_data["rule_id"] for f in failures)

    async def test_get_rule_failures_by_severity(
        self,
        dao: AuditDAO,
        sample_run: Run
    ):
        """Test filtering rule failures by severity."""
        # Create failures with different severities
        await dao.create_rule_failure(
            run_id=sample_run.id,
            rule_id="error_rule",
            rule_name="Error Rule",
            severity="error",
            message="Error message"
        )
        await dao.create_rule_failure(
            run_id=sample_run.id,
            rule_id="warning_rule",
            rule_name="Warning Rule",
            severity="warning",
            message="Warning message"
        )
        await dao.commit()

        error_failures = await dao.get_rule_failures_for_run(
            sample_run.id,
            severity="error"
        )

        assert len(error_failures) >= 1
        assert all(f.severity == "error" for f in error_failures)


class TestArtifactOperations:
    """Test Artifact operations."""

    async def test_create_artifact(
        self,
        dao: AuditDAO,
        sample_run: Run,
        sample_artifact_data: Dict[str, Any]
    ):
        """Test creating an artifact."""
        artifact = await dao.create_artifact(
            run_id=sample_run.id,
            **sample_artifact_data
        )

        assert artifact.id is not None
        assert artifact.run_id == sample_run.id
        assert artifact.type == sample_artifact_data["type"]
        assert artifact.path == sample_artifact_data["path"]

    async def test_get_artifacts_for_run(
        self,
        dao: AuditDAO,
        sample_run: Run,
        sample_artifact_data: Dict[str, Any]
    ):
        """Test retrieving artifacts for a run."""
        # Create an artifact first
        await dao.create_artifact(
            run_id=sample_run.id,
            **sample_artifact_data
        )
        await dao.commit()

        artifacts = await dao.get_artifacts_for_run(sample_run.id)

        assert len(artifacts) >= 1
        assert any(a.type == sample_artifact_data["type"] for a in artifacts)


class TestTagInventory:
    """Test tag inventory aggregation."""

    async def test_get_tag_inventory_for_run(
        self,
        dao: AuditDAO,
        populated_run: Run
    ):
        """Test tag inventory aggregation."""
        inventory = await dao.get_tag_inventory_for_run(populated_run.id)

        assert len(inventory) >= 1

        # Check structure of inventory items
        for tag in inventory:
            assert "vendor" in tag
            assert "name" in tag
            assert "id" in tag
            assert "count" in tag
            assert "pages" in tag
            assert isinstance(tag["count"], int)
            assert isinstance(tag["pages"], int)
            assert tag["count"] > 0
            assert tag["pages"] > 0  # This should now work with our fix


class TestRunStatistics:
    """Test run statistics aggregation."""

    async def test_get_run_statistics(
        self,
        dao: AuditDAO,
        populated_run: Run
    ):
        """Test run statistics calculation."""
        stats = await dao.get_run_statistics(populated_run.id)

        assert stats is not None
        assert "status" in stats
        assert "pages" in stats
        assert "requests" in stats
        assert "cookies" in stats
        assert "rule_failures" in stats

        # Check page stats
        page_stats = stats["pages"]
        assert page_stats["total"] >= 3
        assert page_stats["successful"] >= 0
        assert page_stats["failed"] >= 0

        # Check request stats
        request_stats = stats["requests"]
        assert request_stats["total"] >= 6
        assert request_stats["successful"] >= 0
        assert request_stats["failed"] >= 0

        # Check cookie stats
        cookie_stats = stats["cookies"]
        assert cookie_stats["total"] >= 6
        assert cookie_stats["first_party"] >= 0
        assert cookie_stats["third_party"] >= 0


class TestTransactionManagement:
    """Test transaction management."""

    async def test_commit_transaction(self, dao: AuditDAO, sample_run_data: Dict[str, Any]):
        """Test committing a transaction."""
        run = await dao.create_run(**sample_run_data)
        assert run.id is not None

        await dao.commit()

        # Verify data is persisted
        retrieved_run = await dao.get_run(run.id)
        assert retrieved_run is not None

    async def test_rollback_transaction(self, dao: AuditDAO, sample_run_data: Dict[str, Any]):
        """Test rolling back a transaction."""
        run = await dao.create_run(**sample_run_data)
        run_id = run.id
        assert run_id is not None

        await dao.rollback()

        # Verify data is not persisted after rollback
        retrieved_run = await dao.get_run(run_id)
        assert retrieved_run is None

    async def test_flush_without_commit(self, dao: AuditDAO, sample_run_data: Dict[str, Any]):
        """Test flushing changes without committing."""
        run = await dao.create_run(**sample_run_data)

        await dao.flush()

        # After flush, object should have ID but not be committed
        assert run.id is not None