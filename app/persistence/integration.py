"""Integration layer for connecting persistence to existing audit components."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from ..audit.models.crawl import CrawlConfig, CrawlMetrics
from ..audit.capture.models import CaptureResult, NetworkObservation
from ..audit.rules.models import RuleResults, RuleFailure as RuleFailureModel
from .dao import AuditDAO
from .models import Run, PageResult, RequestLog, Cookie, DataLayerSnapshot, RuleFailure, Artifact
from .config import PersistenceManager
from .storage import ArtifactStore


logger = logging.getLogger(__name__)


class PersistentRunner:
    """Wrapper that adds persistence to existing audit runners."""

    def __init__(
        self,
        persistence_manager: PersistenceManager,
        run_name: str,
        environment: str = "development",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize persistent runner.

        Args:
            persistence_manager: Persistence layer manager
            run_name: Name/description for this audit run
            environment: Environment being audited (dev/staging/prod)
            config: Configuration dictionary to store with run
        """
        self.persistence_manager = persistence_manager
        self.run_name = run_name
        self.environment = environment
        self.config = config or {}

        self._run: Optional[Run] = None
        self._page_results: Dict[str, PageResult] = {}
        self._dao: Optional[AuditDAO] = None
        self._session: Optional[AsyncSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        # Get database session and keep it alive for the runner lifecycle
        self._session = self.persistence_manager.database.session_factory()
        self._dao = AuditDAO(self._session)

        # Create audit run record
        self._run = await self._dao.create_run(
            site_id=self.run_name,  # Using run_name as site_id for now
            environment=self.environment,
            config=self.config
        )
        await self._dao.commit()

        logger.info(f"Created audit run {self._run.id}: {self.run_name}")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        try:
            if self._run and self._dao:
                error_message = None if exc_type is None else (str(exc_val) or exc_type.__name__)
                from ..persistence.models import RunStatus

                if exc_type:
                    # Clear any pending failed transaction first
                    try:
                        await self._dao.rollback()
                    except Exception as rollback_err:
                        logger.error(f"Failed to rollback transaction for run {self._run.id}: {rollback_err}")
                        return  # Nothing more we can do

                    # Now update status in fresh transaction
                    await self._dao.update_run_status(
                        self._run.id,
                        status=RunStatus.FAILED,
                        error_message=error_message
                    )
                    await self._dao.commit()
                    logger.info(f"Audit run {self._run.id} completed with status: FAILED")

                else:
                    # Happy path - just update status and commit
                    await self._dao.update_run_status(
                        self._run.id,
                        status=RunStatus.COMPLETED,
                        error_message=None
                    )
                    await self._dao.commit()
                    logger.info(f"Audit run {self._run.id} completed with status: COMPLETED")

        except Exception as status_error:
            # If we can't update the status, log the error but don't fail completely
            logger.error(f"Failed to update run status for {self._run.id if self._run else 'unknown'}: {status_error}")
            try:
                if self._dao:
                    await self._dao.rollback()
            except Exception:
                pass  # Ignore rollback errors at this point

        finally:
            # Always close the session
            if self._session:
                await self._session.close()
                self._session = None
                self._dao = None

    @property
    def run_id(self) -> Optional[int]:
        """Get the current run ID."""
        return self._run.id if self._run else None

    async def store_page_result(
        self,
        url: str,
        final_url: str,
        status_code: int,
        success: bool,
        load_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        content_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PageResult:
        """Store a page result."""
        if not self._dao or not self._run:
            raise RuntimeError("Persistent runner not properly initialized")

        from ..persistence.models import PageStatus

        page_result = await self._dao.create_page_result(
            run_id=self._run.id,
            url=url,
            status=PageStatus.SUCCESS if success else PageStatus.FAILED,
            final_url=final_url,
            title=metadata.get("page_title") if metadata else None,
            load_time_ms=load_time_ms,
            capture_error=error_message,
            metrics=metadata
        )

        self._page_results[url] = page_result
        return page_result

    async def store_capture_result(self, capture_result: CaptureResult) -> PageResult:
        """Store a complete capture result with all associated data."""
        if not self._dao or not self._run:
            raise RuntimeError("Persistent runner not properly initialized")

        # Store page result
        page_result = await self.store_page_result(
            url=capture_result.url,
            final_url=capture_result.final_url or capture_result.url,
            status_code=capture_result.status_code,
            success=capture_result.success,
            load_time_ms=capture_result.load_time_ms,
            error_message=capture_result.error_message,
            content_hash=capture_result.content_hash,
            metadata={
                "page_title": capture_result.page_title,
                "viewport": capture_result.viewport,
                "user_agent": capture_result.user_agent
            }
        )

        # Store network observations as request logs
        if capture_result.network_observations:
            await self._store_network_observations(
                page_result.id,
                capture_result.network_observations
            )

        # Store cookies
        if capture_result.cookies:
            await self._store_cookies(page_result.id, capture_result.cookies)

        # Store data layer snapshots
        if capture_result.datalayer_snapshots:
            await self._store_datalayer_snapshots(
                page_result.id,
                capture_result.datalayer_snapshots
            )

        # Store artifacts (screenshots, HAR files, etc.)
        if capture_result.artifacts:
            await self._store_artifacts(capture_result.artifacts)

        return page_result

    async def _store_network_observations(
        self,
        page_result_id: int,
        observations: List[NetworkObservation]
    ):
        """Store network observations as request logs."""
        request_logs_data = []
        for obs in observations:
            request_logs_data.append({
                "url": obs.url,
                "method": obs.method,
                "resource_type": obs.resource_type,
                "status_code": obs.status_code,
                "status_text": obs.status_text,
                "start_time": obs.start_time,
                "end_time": obs.end_time,
                # duration_ms is computed from start_time/end_time
                "success": obs.success,
                "error_text": obs.error_text,
                "protocol": obs.protocol,
                "remote_address": obs.remote_address,
                # host is computed from url
                "request_headers_json": obs.request_headers,
                "response_headers_json": obs.response_headers,
                "timings_json": obs.timings,
                "sizes_json": obs.sizes,
                "vendor_tags_json": obs.vendor_tags
            })

        if request_logs_data:
            await self._dao.bulk_create_request_logs(page_result_id, request_logs_data)

    async def _store_cookies(
        self,
        page_result_id: int,
        cookies: List[Dict[str, Any]]
    ):
        """Store cookies."""
        cookies_data = []
        for cookie_data in cookies:
            cookies_data.append({
                "name": cookie_data.get("name", ""),
                "domain": cookie_data.get("domain", ""),
                "path": cookie_data.get("path", "/"),
                "expires": cookie_data.get("expires"),
                "max_age": cookie_data.get("maxAge"),
                "size": cookie_data.get("size", 0),
                "secure": cookie_data.get("secure", False),
                "http_only": cookie_data.get("httpOnly", False),
                "same_site": cookie_data.get("sameSite"),
                "first_party": cookie_data.get("firstParty", True),
                "essential": cookie_data.get("essential", False),
                "is_session": cookie_data.get("session", False),
                "value_redacted": cookie_data.get("valueRedacted", True),
                "set_time": cookie_data.get("setTime"),
                "modified_time": cookie_data.get("modifiedTime"),
                # cookie_key is computed from name@domain+path
                "metadata_json": cookie_data.get("metadata")
            })

        if cookies_data:
            await self._dao.bulk_create_cookies(page_result_id, cookies_data)

    async def _store_datalayer_snapshots(
        self,
        page_result_id: int,
        snapshots: List[Dict[str, Any]]
    ):
        """Store data layer snapshots."""
        for snapshot in snapshots:
            # Calculate size if data is available
            data = snapshot.get("data", {})
            size_bytes = len(str(data)) if data else 0

            await self._dao.create_datalayer_snapshot(
                page_result_id=page_result_id,
                exists=bool(data),
                size_bytes=size_bytes,
                truncated=snapshot.get("truncated", False),
                sample_data=data,
                schema_valid=snapshot.get("schemaValid"),
                validation_errors=snapshot.get("validationErrors"),
                metadata={
                    "event": snapshot.get("event"),
                    "timestamp": snapshot.get("timestamp", datetime.now(timezone.utc)).isoformat() if isinstance(snapshot.get("timestamp"), datetime) else snapshot.get("timestamp"),
                    "redacted": snapshot.get("redacted", False),
                    "redaction_rules": snapshot.get("redactionRules", [])
                }
            )

    async def _store_artifacts(self, artifacts: List[Dict[str, Any]]):
        """Store artifacts."""
        for artifact_data in artifacts:
            # Store artifact in storage backend
            artifact_path = artifact_data.get("path", "")
            content = artifact_data.get("content", b"")

            if content and artifact_path:
                artifact_ref = await self.persistence_manager.artifact_store.put(
                    content=content,
                    path=artifact_path,
                    content_type=artifact_data.get("contentType"),
                    metadata=artifact_data.get("metadata", {})
                )

                # Store artifact record in database
                from ..persistence.models import ArtifactKind
                artifact_kind = artifact_data.get("type", "unknown")
                # Map to valid enum values
                if artifact_kind not in [k.value for k in ArtifactKind]:
                    artifact_kind = ArtifactKind.PAGE_SOURCE.value

                # Determine storage backend from persistence manager
                storage_backend = self.persistence_manager.config.storage_backend

                await self._dao.create_artifact(
                    run_id=self._run.id,
                    kind=artifact_kind,
                    path=artifact_ref.path,
                    checksum=artifact_ref.checksum,
                    size_bytes=artifact_ref.size_bytes,
                    content_type=artifact_ref.content_type,
                    storage_backend=storage_backend,
                    description=artifact_data.get("description")
                )

    async def store_rule_results(self, rule_results: RuleResults):
        """Store rule evaluation results."""
        if not self._dao or not self._run:
            raise RuntimeError("Persistent runner not properly initialized")

        # Store rule failures
        from ..persistence.models import SeverityLevel
        for failure in rule_results.failures:
            # Map severity to our enum
            severity_mapping = {
                "info": SeverityLevel.LOW,
                "warning": SeverityLevel.MEDIUM,
                "error": SeverityLevel.HIGH,
                "critical": SeverityLevel.CRITICAL
            }
            severity = severity_mapping.get(failure.severity.value.lower(), SeverityLevel.MEDIUM)

            await self._dao.create_rule_failure(
                run_id=self._run.id,
                rule_id=failure.rule_id,
                severity=severity,
                message=failure.message,
                rule_name=failure.rule_name,
                page_url=failure.page_url,
                details=failure.details
            )

    async def update_run_stats(self, stats: Dict[str, Any]):
        """Update run statistics."""
        if not self._dao or not self._run:
            raise RuntimeError("Persistent runner not properly initialized")

        # Get current run and update stats
        current_run = await self._dao.get_run_by_id(self._run.id)
        if current_run:
            current_run.summary_json = stats
            await self._dao.commit()

    async def get_run_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of the current run."""
        if not self._dao or not self._run:
            return None

        stats = await self._dao.get_run_statistics(self._run.id)

        # Handle status - it might be a string or enum depending on how it was loaded
        from ..persistence.models import RunStatus
        status_value = self._run.status
        if isinstance(status_value, str):
            status_value = RunStatus(status_value).value
        else:
            status_value = status_value.value

        return {
            "run_id": self._run.id,
            "name": self._run.site_id,  # Using site_id as name for now
            "site_id": self._run.site_id,
            "environment": self._run.environment,
            "status": status_value,
            "started_at": self._run.started_at.isoformat() if self._run.started_at else None,
            "finished_at": self._run.finished_at.isoformat() if self._run.finished_at else None,
            "statistics": stats
        }


class PersistenceIntegrator:
    """Helper class for integrating persistence into existing components."""

    @staticmethod
    def wrap_crawler_config(
        persistence_manager: PersistenceManager,
        crawl_config: CrawlConfig,
        run_name: str,
        environment: str = "development"
    ) -> Dict[str, Any]:
        """Convert CrawlConfig to persistence-friendly format."""
        return {
            "start_url": str(crawl_config.seeds[0]) if crawl_config.seeds else "",
            "max_pages": crawl_config.max_pages,
            "max_concurrency": crawl_config.max_concurrency,
            "requests_per_second": crawl_config.requests_per_second,
            "page_timeout": crawl_config.page_timeout,
            "discovery_mode": crawl_config.discovery_mode.value if crawl_config.discovery_mode else None,
            "sitemap_url": str(crawl_config.sitemap_url) if crawl_config.sitemap_url else None,
            "include_patterns": [str(p) for p in crawl_config.include_patterns],
            "exclude_patterns": [str(p) for p in crawl_config.exclude_patterns],
            "respect_robots": crawl_config.respect_robots
        }

    @staticmethod
    async def create_persistent_run(
        persistence_manager: PersistenceManager,
        run_name: str,
        environment: str,
        start_url: str,
        config: Optional[Dict[str, Any]] = None
    ) -> PersistentRunner:
        """Create a new persistent run."""
        return PersistentRunner(
            persistence_manager=persistence_manager,
            run_name=run_name,
            environment=environment,
            config=config or {}
        )

    @staticmethod
    def extract_crawler_metrics(metrics: CrawlMetrics) -> Dict[str, Any]:
        """Extract metrics from crawler for storage."""
        return {
            "urls_discovered": getattr(metrics, "urls_discovered", 0),
            "urls_processed": getattr(metrics, "urls_processed", 0),
            "urls_failed": getattr(metrics, "urls_failed", 0),
            "pages_per_second": getattr(metrics, "pages_per_second", 0.0),
            "total_duration_seconds": getattr(metrics, "total_duration_seconds", 0.0),
            "discovery_duration_seconds": getattr(metrics, "discovery_duration_seconds", 0.0),
            "processing_duration_seconds": getattr(metrics, "processing_duration_seconds", 0.0)
        }