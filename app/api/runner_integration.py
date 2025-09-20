"""Runner integration for Tag Sentinel API.

This module provides integration with the audit runner system for
dispatching audits and monitoring their execution status.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Protocol
from datetime import datetime
from abc import ABC, abstractmethod

from app.api.schemas.responses import AuditStatus

logger = logging.getLogger(__name__)


class AuditRunner(Protocol):
    """Protocol for audit runner implementations."""

    async def dispatch_audit(
        self,
        audit_id: str,
        site_id: str,
        env: str,
        params: Dict[str, Any]
    ) -> bool:
        """Dispatch an audit for execution.

        Args:
            audit_id: Unique audit identifier
            site_id: Site to audit
            env: Environment to audit
            params: Audit parameters

        Returns:
            True if successfully dispatched
        """
        ...

    async def get_audit_status(self, audit_id: str) -> Optional[AuditStatus]:
        """Get current status of an audit.

        Args:
            audit_id: Audit identifier

        Returns:
            Current audit status or None if not found
        """
        ...

    async def cancel_audit(self, audit_id: str) -> bool:
        """Cancel a running audit.

        Args:
            audit_id: Audit identifier to cancel

        Returns:
            True if successfully cancelled
        """
        ...

    async def get_audit_progress(self, audit_id: str) -> Optional[float]:
        """Get progress percentage for a running audit.

        Args:
            audit_id: Audit identifier

        Returns:
            Progress percentage (0.0-100.0) or None if not available
        """
        ...


class MockAuditRunner:
    """Mock audit runner for development and testing."""

    def __init__(self):
        """Initialize mock runner."""
        self._running_audits: Dict[str, Dict[str, Any]] = {}
        self._completed_audits: Dict[str, Dict[str, Any]] = {}
        logger.info("MockAuditRunner initialized")

    async def dispatch_audit(
        self,
        audit_id: str,
        site_id: str,
        env: str,
        params: Dict[str, Any]
    ) -> bool:
        """Dispatch an audit for mock execution."""
        logger.info(f"Dispatching mock audit {audit_id} for {site_id}/{env}")

        # Store audit in running state
        self._running_audits[audit_id] = {
            "site_id": site_id,
            "env": env,
            "params": params,
            "started_at": datetime.utcnow(),
            "progress": 0.0,
            "status": AuditStatus.RUNNING
        }

        # Simulate async audit execution
        asyncio.create_task(self._simulate_audit_execution(audit_id))

        return True

    async def get_audit_status(self, audit_id: str) -> Optional[AuditStatus]:
        """Get current status of a mock audit."""
        if audit_id in self._running_audits:
            return self._running_audits[audit_id]["status"]
        elif audit_id in self._completed_audits:
            return self._completed_audits[audit_id]["status"]
        return None

    async def cancel_audit(self, audit_id: str) -> bool:
        """Cancel a mock audit."""
        if audit_id in self._running_audits:
            audit_data = self._running_audits.pop(audit_id)
            audit_data["status"] = AuditStatus.FAILED
            audit_data["finished_at"] = datetime.utcnow()
            audit_data["error"] = "Cancelled by user"
            self._completed_audits[audit_id] = audit_data
            logger.info(f"Cancelled mock audit {audit_id}")
            return True
        return False

    async def get_audit_progress(self, audit_id: str) -> Optional[float]:
        """Get progress for a mock audit."""
        if audit_id in self._running_audits:
            return self._running_audits[audit_id]["progress"]
        elif audit_id in self._completed_audits:
            return 100.0
        return None

    async def _simulate_audit_execution(self, audit_id: str):
        """Simulate audit execution with progress updates."""
        try:
            # Simulate audit phases with progress
            phases = [
                ("Initializing", 10),
                ("Crawling pages", 40),
                ("Analyzing tags", 70),
                ("Generating report", 90),
                ("Finalizing", 100)
            ]

            for phase_name, progress in phases:
                if audit_id not in self._running_audits:
                    return  # Audit was cancelled

                self._running_audits[audit_id]["progress"] = progress
                self._running_audits[audit_id]["current_phase"] = phase_name

                logger.debug(f"Mock audit {audit_id}: {phase_name} ({progress}%)")

                # Simulate processing time
                await asyncio.sleep(2)

            # Complete the audit
            if audit_id in self._running_audits:
                audit_data = self._running_audits.pop(audit_id)
                audit_data["status"] = AuditStatus.COMPLETED
                audit_data["finished_at"] = datetime.utcnow()
                audit_data["progress"] = 100.0

                # Simulate some results
                audit_data["results"] = {
                    "pages_crawled": 15,
                    "tags_detected": 8,
                    "cookies_found": 12,
                    "failures": 0
                }

                self._completed_audits[audit_id] = audit_data
                logger.info(f"Completed mock audit {audit_id}")

        except Exception as e:
            # Handle simulation errors
            logger.error(f"Error simulating audit {audit_id}: {e}")
            if audit_id in self._running_audits:
                audit_data = self._running_audits.pop(audit_id)
                audit_data["status"] = AuditStatus.FAILED
                audit_data["finished_at"] = datetime.utcnow()
                audit_data["error"] = str(e)
                self._completed_audits[audit_id] = audit_data


class RunnerIntegrationService:
    """Service for integrating with audit runners."""

    def __init__(self, runner: Optional[AuditRunner] = None):
        """Initialize runner integration service.

        Args:
            runner: Audit runner implementation (defaults to mock)
        """
        self.runner = runner or MockAuditRunner()
        logger.info(f"RunnerIntegrationService initialized with runner={type(self.runner).__name__}")

    async def dispatch_audit(
        self,
        audit_id: str,
        site_id: str,
        env: str,
        params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Dispatch an audit to the runner.

        Args:
            audit_id: Unique audit identifier
            site_id: Site to audit
            env: Environment to audit
            params: Audit parameters

        Returns:
            True if successfully dispatched

        Raises:
            RuntimeError: If dispatch fails
        """
        try:
            params = params or {}
            logger.info(f"Dispatching audit {audit_id} to runner")

            success = await self.runner.dispatch_audit(audit_id, site_id, env, params)

            if success:
                logger.info(f"Successfully dispatched audit {audit_id}")
            else:
                logger.error(f"Failed to dispatch audit {audit_id}")
                raise RuntimeError(f"Failed to dispatch audit {audit_id}")

            return success

        except Exception as e:
            logger.error(f"Error dispatching audit {audit_id}: {e}")
            raise RuntimeError(f"Failed to dispatch audit: {str(e)}")

    async def get_audit_status(self, audit_id: str) -> Optional[AuditStatus]:
        """Get current audit status from runner.

        Args:
            audit_id: Audit identifier

        Returns:
            Current status or None if not found
        """
        try:
            status = await self.runner.get_audit_status(audit_id)
            logger.debug(f"Retrieved status for audit {audit_id}: {status}")
            return status

        except Exception as e:
            logger.error(f"Error getting status for audit {audit_id}: {e}")
            return None

    async def cancel_audit(self, audit_id: str) -> bool:
        """Cancel a running audit.

        Args:
            audit_id: Audit identifier

        Returns:
            True if successfully cancelled
        """
        try:
            success = await self.runner.cancel_audit(audit_id)
            if success:
                logger.info(f"Successfully cancelled audit {audit_id}")
            else:
                logger.warning(f"Failed to cancel audit {audit_id} (may not be running)")
            return success

        except Exception as e:
            logger.error(f"Error cancelling audit {audit_id}: {e}")
            return False

    async def get_audit_progress(self, audit_id: str) -> Optional[float]:
        """Get audit progress percentage.

        Args:
            audit_id: Audit identifier

        Returns:
            Progress percentage (0.0-100.0) or None
        """
        try:
            progress = await self.runner.get_audit_progress(audit_id)
            logger.debug(f"Retrieved progress for audit {audit_id}: {progress}%")
            return progress

        except Exception as e:
            logger.error(f"Error getting progress for audit {audit_id}: {e}")
            return None

    async def health_check(self) -> Dict[str, Any]:
        """Check runner health and connectivity.

        Returns:
            Health status information
        """
        try:
            # For mock runner, always healthy
            # For real runner, would check connectivity/status
            return {
                "status": "healthy",
                "runner_type": type(self.runner).__name__,
                "timestamp": datetime.utcnow().isoformat(),
                "capabilities": {
                    "dispatch": True,
                    "status_monitoring": True,
                    "progress_tracking": True,
                    "cancellation": True
                }
            }

        except Exception as e:
            logger.error(f"Runner health check failed: {e}")
            return {
                "status": "unhealthy",
                "runner_type": type(self.runner).__name__,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }