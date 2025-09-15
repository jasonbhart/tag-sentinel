"""Integration with the Tag Sentinel audit runner system."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import uuid4

from .dispatch import RunDispatcherBackend, RunResult, RunStatus
from .models import RunRequest
from ..audit.crawler import Crawler
from ..audit.models.crawl import CrawlConfig
from app.audit.utils.url_normalizer import normalize

logger = logging.getLogger(__name__)


class AuditRunnerBackend(RunDispatcherBackend):
    """Real audit runner backend that integrates with the Tag Sentinel crawler."""

    def __init__(
        self,
        base_crawl_config: Optional[CrawlConfig] = None,
        max_concurrent_audits: int = 5,
        default_timeout_minutes: int = 60
    ):
        """Initialize the audit runner backend.

        Args:
            base_crawl_config: Base crawl configuration to use for all audits
            max_concurrent_audits: Maximum number of concurrent audit runs
            default_timeout_minutes: Default timeout for audit runs
        """
        self.base_crawl_config = base_crawl_config or self._create_default_config()
        self.max_concurrent_audits = max_concurrent_audits
        self.default_timeout_minutes = default_timeout_minutes

        # Track running audits
        self._running_audits: Dict[str, asyncio.Task] = {}
        self._audit_results: Dict[str, RunResult] = {}

        # Semaphore for concurrency control
        self._concurrency_semaphore = asyncio.Semaphore(max_concurrent_audits)

    async def dispatch_run(self, run_request: RunRequest) -> RunResult:
        """Dispatch an audit run using the Tag Sentinel crawler.

        Args:
            run_request: The run request to execute

        Returns:
            RunResult with initial status
        """
        logger.info(f"Dispatching audit run {run_request.id} for {run_request.site_id}")

        # Create initial result
        result = RunResult(
            run_id=run_request.id,
            status=RunStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            metadata={
                'site_id': run_request.site_id,
                'environment': run_request.environment,
                'schedule_id': run_request.schedule_id,
                'priority': run_request.priority,
                **run_request.metadata
            }
        )

        # Start the audit task
        audit_task = asyncio.create_task(
            self._run_audit(run_request, result)
        )
        self._running_audits[run_request.id] = audit_task
        self._audit_results[run_request.id] = result

        return result

    async def get_run_status(self, run_id: str) -> Optional[RunResult]:
        """Get the current status of an audit run.

        Args:
            run_id: The run ID to check

        Returns:
            Current RunResult or None if not found
        """
        return self._audit_results.get(run_id)

    async def cancel_run(self, run_id: str) -> bool:
        """Cancel a running audit.

        Args:
            run_id: The run ID to cancel

        Returns:
            True if cancelled, False if not found or already complete
        """
        if run_id not in self._running_audits:
            return False

        task = self._running_audits[run_id]
        if task.done():
            return False

        # Cancel the task
        task.cancel()

        # Update result
        if run_id in self._audit_results:
            result = self._audit_results[run_id]
            result.status = RunStatus.CANCELLED
            result.completed_at = datetime.now(timezone.utc)
            result.metadata['cancelled_by'] = 'user_request'

        logger.info(f"Cancelled audit run {run_id}")
        return True

    async def cleanup(self) -> None:
        """Clean up completed audit tasks."""
        completed_runs = []

        for run_id, task in self._running_audits.items():
            if task.done():
                completed_runs.append(run_id)

        for run_id in completed_runs:
            del self._running_audits[run_id]

        if completed_runs:
            logger.debug(f"Cleaned up {len(completed_runs)} completed audit tasks")

    async def get_active_runs(self) -> List[str]:
        """Get list of currently active run IDs."""
        return [
            run_id for run_id, task in self._running_audits.items()
            if not task.done()
        ]

    async def _run_audit(self, run_request: RunRequest, result: RunResult) -> None:
        """Execute an audit run using the crawler.

        Args:
            run_request: The run request to execute
            result: The result object to update
        """
        async with self._concurrency_semaphore:
            try:
                # Get environment-specific configuration
                crawl_config = await self._build_crawl_config(run_request)

                # Create crawler instance
                crawler = Crawler(crawl_config)

                # Set up progress tracking
                def progress_callback(progress_info: Dict[str, Any]) -> None:
                    result.metadata.update({
                        'progress': progress_info,
                        'last_updated': datetime.now(timezone.utc).isoformat()
                    })

                # Run the audit
                logger.info(f"Starting audit for {run_request.site_id} in {run_request.environment}")

                # Run the crawler to get PagePlans with timeout
                page_plans = []

                async def run_crawl():
                    async for page_plan in crawler.crawl():
                        page_plans.append(page_plan)
                        # Update progress
                        progress_callback({
                            'pages_discovered': len(page_plans),
                            'current_url': page_plan.url
                        })
                    return page_plans

                await asyncio.wait_for(
                    run_crawl(),
                    timeout=self.default_timeout_minutes * 60
                )

                # Convert to audit results format
                audit_results = {
                    'page_results': [{'url': pp.url, 'status': 'crawled'} for pp in page_plans],
                    'issues': [],  # Placeholder - real audit results would come from capture engine
                    'crawl_stats': {'pages_discovered': len(page_plans)}
                }

                # Update result with successful completion
                result.status = RunStatus.COMPLETED
                result.completed_at = datetime.now(timezone.utc)
                result.metadata.update({
                    'audit_results': audit_results,
                    'pages_crawled': len(audit_results.get('page_results', [])),
                    'issues_found': len(audit_results.get('issues', [])),
                    'completion_reason': 'success'
                })

                logger.info(f"Audit run {run_request.id} completed successfully")

            except asyncio.TimeoutError:
                result.status = RunStatus.TIMEOUT
                result.completed_at = datetime.now(timezone.utc)
                result.metadata['completion_reason'] = 'timeout'
                result.metadata['timeout_minutes'] = self.default_timeout_minutes
                logger.warning(f"Audit run {run_request.id} timed out after {self.default_timeout_minutes} minutes")

            except asyncio.CancelledError:
                result.status = RunStatus.CANCELLED
                result.completed_at = datetime.now(timezone.utc)
                result.metadata['completion_reason'] = 'cancelled'
                logger.info(f"Audit run {run_request.id} was cancelled")
                raise

            except Exception as e:
                result.status = RunStatus.FAILED
                result.completed_at = datetime.now(timezone.utc)
                result.metadata.update({
                    'completion_reason': 'error',
                    'error_message': str(e),
                    'error_type': type(e).__name__
                })
                logger.error(f"Audit run {run_request.id} failed: {e}", exc_info=True)

    async def _build_crawl_config(self, run_request: RunRequest) -> CrawlConfig:
        """Build crawl configuration for the audit run.

        Args:
            run_request: The run request

        Returns:
            Configured CrawlConfig
        """
        # Start with base configuration
        config = self.base_crawl_config

        # Environment-specific overrides
        if run_request.environment == 'production':
            # More conservative settings for production
            config.max_pages = min(config.max_pages, 100)
            config.requests_per_second = min(config.requests_per_second, 2.0)
        elif run_request.environment == 'development':
            # More aggressive settings for development
            config.max_pages = min(config.max_pages, 20)
            config.requests_per_second = min(config.requests_per_second, 5.0)

        # Site-specific settings
        if 'start_url' not in run_request.metadata:
            # Generate start URL from site ID
            start_url = f"https://{run_request.site_id}"
            if run_request.environment != 'production':
                # Use environment-specific URL
                start_url = f"https://{run_request.environment}.{run_request.site_id}"
        else:
            start_url = run_request.metadata['start_url']

        # Update config with run-specific settings
        config.seeds = [normalize(start_url)]

        # Apply metadata overrides
        if 'max_pages' in run_request.metadata:
            config.max_pages = int(run_request.metadata['max_pages'])

        if 'requests_per_second' in run_request.metadata:
            config.requests_per_second = float(run_request.metadata['requests_per_second'])

        # Set run identification
        config.metadata.update({
            'run_id': run_request.id,
            'schedule_id': run_request.schedule_id,
            'site_id': run_request.site_id,
            'environment': run_request.environment
        })

        return config

    def _create_default_config(self) -> CrawlConfig:
        """Create a default crawl configuration.

        Returns:
            Default CrawlConfig
        """
        return CrawlConfig(
            seeds=[],
            max_pages=50,
            max_depth=3,
            requests_per_second=2.0,
            max_concurrency=5,
            page_timeout=30,
            metadata={}
        )


def create_audit_runner_backend(
    max_concurrent_audits: int = 5,
    default_timeout_minutes: int = 60,
    base_config_overrides: Optional[Dict[str, Any]] = None
) -> AuditRunnerBackend:
    """Factory function to create an audit runner backend.

    Args:
        max_concurrent_audits: Maximum concurrent audit runs
        default_timeout_minutes: Default timeout for audits
        base_config_overrides: Overrides for base crawl configuration

    Returns:
        Configured AuditRunnerBackend
    """
    # Create base config with overrides
    base_config = CrawlConfig(
        seeds=[],
        max_pages=50,
        max_depth=3,
        requests_per_second=2.0,
        max_concurrency=5,
        page_timeout=30,
        metadata={}
    )

    if base_config_overrides:
        for key, value in base_config_overrides.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)

    return AuditRunnerBackend(
        base_crawl_config=base_config,
        max_concurrent_audits=max_concurrent_audits,
        default_timeout_minutes=default_timeout_minutes
    )
