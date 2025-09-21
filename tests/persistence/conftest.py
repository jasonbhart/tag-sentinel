"""Test configuration and fixtures for persistence layer tests."""

import asyncio
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.persistence.database import DatabaseConfig, Base
from app.persistence.dao import AuditDAO
from app.persistence.models import (
    Run, PageResult, RequestLog, Cookie, DataLayerSnapshot,
    RuleFailure, Artifact
)
from app.persistence.storage import LocalArtifactStore, ArtifactStore


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def test_db_config() -> AsyncGenerator[DatabaseConfig, None]:
    """Create test database configuration with in-memory SQLite."""
    # Use in-memory SQLite for fast tests
    config = DatabaseConfig(
        url="sqlite+aiosqlite:///:memory:",
        echo=False
    )

    # Create all tables
    async with config.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield config

    # Cleanup
    await config.close()


@pytest_asyncio.fixture
async def db_session(test_db_config: DatabaseConfig) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async with test_db_config.session() as session:
        yield session


@pytest_asyncio.fixture
async def dao(db_session: AsyncSession) -> AuditDAO:
    """Create DAO instance with test session."""
    return AuditDAO(db_session)


@pytest.fixture
def temp_artifact_dir() -> Path:
    """Create temporary directory for artifact storage tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
async def artifact_store(temp_artifact_dir: Path) -> ArtifactStore:
    """Create local artifact store for testing."""
    return LocalArtifactStore(base_path=temp_artifact_dir)


# ============= Test Data Fixtures =============

@pytest.fixture
def sample_run_data() -> Dict[str, Any]:
    """Sample run data for testing."""
    return {
        "name": "Test Run",
        "environment": "test",
        "start_url": "https://example.com",
        "status": "completed",
        "started_at": datetime.now(timezone.utc),
        "completed_at": datetime.now(timezone.utc),
        "config_json": {"max_pages": 10, "timeout": 30},
        "stats_json": {"pages_crawled": 5, "requests_made": 50}
    }


@pytest.fixture
def sample_page_result_data() -> Dict[str, Any]:
    """Sample page result data for testing."""
    return {
        "url": "https://example.com/page1",
        "final_url": "https://example.com/page1",
        "status_code": 200,
        "load_time_ms": 1500,
        "success": True,
        "crawled_at": datetime.now(timezone.utc),
        "content_hash": "abc123",
        "metadata_json": {"title": "Test Page"}
    }


@pytest.fixture
def sample_request_log_data() -> Dict[str, Any]:
    """Sample request log data for testing."""
    return {
        "url": "https://analytics.google.com/collect",
        "method": "POST",
        "resource_type": "xhr",
        "status_code": 200,
        "status_text": "OK",
        "start_time": datetime.now(timezone.utc),
        "end_time": datetime.now(timezone.utc),
        "duration_ms": 100,
        "success": True,
        "protocol": "h2",
        "host": "analytics.google.com",
        "request_headers_json": {"Content-Type": "application/json"},
        "response_headers_json": {"Server": "Google"},
        "timings_json": {"dns": 10, "connect": 20, "ssl": 30},
        "sizes_json": {"request": 1024, "response": 512},
        "vendor_tags_json": [{"vendor": "google", "name": "GA4", "id": "G-12345"}]
    }


@pytest.fixture
def sample_cookie_data() -> Dict[str, Any]:
    """Sample cookie data for testing."""
    return {
        "name": "_ga",
        "domain": ".example.com",
        "path": "/",
        "max_age": 63072000,
        "size": 27,
        "secure": True,
        "http_only": False,
        "same_site": "Lax",
        "first_party": True,
        "essential": False,
        "is_session": False,
        "value_redacted": True,
        "set_time": datetime.now(timezone.utc),
        "cookie_key": "_ga_.example.com",
        "metadata_json": {"vendor": "google", "purpose": "analytics"}
    }


@pytest.fixture
def sample_datalayer_snapshot_data() -> Dict[str, Any]:
    """Sample data layer snapshot data for testing."""
    return {
        "event_name": "page_view",
        "sequence_number": 1,
        "timestamp": datetime.now(timezone.utc),
        "data_json": {
            "event": "page_view",
            "page_title": "Test Page",
            "page_location": "https://example.com"
        },
        "redacted": False,
        "redaction_rules_json": []
    }


@pytest.fixture
def sample_rule_failure_data() -> Dict[str, Any]:
    """Sample rule failure data for testing."""
    return {
        "rule_id": "missing_ga4",
        "rule_name": "GA4 Tag Required",
        "severity": "error",
        "message": "GA4 tracking tag not found on page",
        "page_url": "https://example.com/page1",
        "detected_at": datetime.now(timezone.utc),
        "details_json": {"expected_tag": "G-12345", "found_tags": []}
    }


@pytest.fixture
def sample_artifact_data() -> Dict[str, Any]:
    """Sample artifact data for testing."""
    return {
        "type": "screenshot",
        "path": f"screenshots/{uuid4()}.png",
        "checksum": "sha256:abc123def456",
        "size_bytes": 1024000,
        "content_type": "image/png",
        "metadata_json": {"width": 1920, "height": 1080}
    }


# ============= Composite Test Data Fixtures =============

@pytest_asyncio.fixture
async def sample_run(dao: AuditDAO, sample_run_data: Dict[str, Any]) -> Run:
    """Create a sample run in the database."""
    run = await dao.create_run(**sample_run_data)
    await dao.commit()
    return run


@pytest_asyncio.fixture
async def sample_page_result(
    dao: AuditDAO,
    sample_run: Run,
    sample_page_result_data: Dict[str, Any]
) -> PageResult:
    """Create a sample page result in the database."""
    page_result = await dao.create_page_result(
        run_id=sample_run.id,
        **sample_page_result_data
    )
    await dao.commit()
    return page_result


@pytest_asyncio.fixture
async def sample_request_log(
    dao: AuditDAO,
    sample_page_result: PageResult,
    sample_request_log_data: Dict[str, Any]
) -> RequestLog:
    """Create a sample request log in the database."""
    request_log = await dao.create_request_log(
        page_result_id=sample_page_result.id,
        **sample_request_log_data
    )
    await dao.commit()
    return request_log


@pytest_asyncio.fixture
async def sample_cookie(
    dao: AuditDAO,
    sample_page_result: PageResult,
    sample_cookie_data: Dict[str, Any]
) -> Cookie:
    """Create a sample cookie in the database."""
    cookie = await dao.create_cookie(
        page_result_id=sample_page_result.id,
        **sample_cookie_data
    )
    await dao.commit()
    return cookie


@pytest_asyncio.fixture
async def populated_run(
    dao: AuditDAO,
    sample_run: Run
) -> Run:
    """Create a run with multiple pages, requests, and cookies for testing exports."""
    # Create multiple page results
    pages = []
    for i in range(3):
        page = await dao.create_page_result(
            run_id=sample_run.id,
            url=f"https://example.com/page{i+1}",
            final_url=f"https://example.com/page{i+1}",
            status_code=200,
            load_time_ms=1000 + i * 100,
            success=True,
            crawled_at=datetime.now(timezone.utc),
            content_hash=f"hash{i+1}",
            metadata_json={"title": f"Page {i+1}"}
        )
        pages.append(page)

    # Create request logs for each page
    for page in pages:
        for j in range(2):  # 2 requests per page
            await dao.create_request_log(
                page_result_id=page.id,
                url=f"https://analytics.google.com/collect?page={page.id}&req={j}",
                method="POST",
                resource_type="xhr",
                status_code=200,
                status_text="OK",
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                duration_ms=50 + j * 10,
                success=True,
                vendor_tags_json=[
                    {"vendor": "google", "name": "GA4", "id": f"G-{page.id}{j}"}
                ]
            )

        # Create cookies for each page
        for k in range(2):  # 2 cookies per page
            await dao.create_cookie(
                page_result_id=page.id,
                name=f"_cookie_{k}",
                domain=".example.com",
                path="/",
                max_age=3600,
                size=20 + k,
                secure=True,
                http_only=k == 0,
                first_party=True,
                essential=k == 0,
                is_session=False,
                value_redacted=True,
                cookie_key=f"_cookie_{k}_.example.com"
            )

    await dao.commit()
    return sample_run