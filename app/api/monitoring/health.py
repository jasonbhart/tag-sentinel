"""Health checking system for Tag Sentinel API.

This module provides comprehensive health monitoring for API components
including database, external services, and system resources.
"""

import logging
import time
import asyncio
import psutil
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status values."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: float = field(default_factory=time.time)
    response_time: Optional[float] = None

    @property
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == HealthStatus.HEALTHY


@dataclass
class SystemHealth:
    """Overall system health status."""
    is_healthy: bool
    status: HealthStatus
    timestamp: float
    components: Dict[str, ComponentHealth]
    summary: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_components(cls, components: Dict[str, ComponentHealth]) -> "SystemHealth":
        """Create system health from component health checks."""
        timestamp = time.time()

        # Determine overall health
        if not components:
            return cls(
                is_healthy=True,
                status=HealthStatus.HEALTHY,
                timestamp=timestamp,
                components={}
            )

        # Check if any component is unhealthy
        unhealthy_components = [name for name, comp in components.items() if not comp.is_healthy]

        if not unhealthy_components:
            overall_status = HealthStatus.HEALTHY
            is_healthy = True
        elif len(unhealthy_components) == len(components):
            overall_status = HealthStatus.UNHEALTHY
            is_healthy = False
        else:
            overall_status = HealthStatus.DEGRADED
            is_healthy = False

        # Create summary
        summary = {
            "total_components": len(components),
            "healthy_components": len(components) - len(unhealthy_components),
            "unhealthy_components": len(unhealthy_components),
            "unhealthy_component_names": unhealthy_components
        }

        return cls(
            is_healthy=is_healthy,
            status=overall_status,
            timestamp=timestamp,
            components=components,
            summary=summary
        )


class HealthCheck(ABC):
    """Abstract base class for health checks."""

    def __init__(self, name: str, timeout: float = 5.0):
        """Initialize health check.

        Args:
            name: Name of the health check
            timeout: Timeout for the check in seconds
        """
        self.name = name
        self.timeout = timeout

    @abstractmethod
    async def check(self) -> ComponentHealth:
        """Perform the health check.

        Returns:
            ComponentHealth result
        """
        pass

    async def run_check(self) -> ComponentHealth:
        """Run health check with timeout and error handling."""
        start_time = time.time()

        try:
            # Run check with timeout
            result = await asyncio.wait_for(self.check(), timeout=self.timeout)
            result.response_time = time.time() - start_time
            return result

        except asyncio.TimeoutError:
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                response_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Health check {self.name} failed: {e}", exc_info=True)
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time=time.time() - start_time
            )


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity."""

    def __init__(self, name: str = "database", repository=None, timeout: float = 5.0):
        """Initialize database health check.

        Args:
            name: Name of the health check
            repository: Database repository to test
            timeout: Timeout for the check
        """
        super().__init__(name, timeout)
        self.repository = repository

    async def check(self) -> ComponentHealth:
        """Check database connectivity."""
        if not self.repository:
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="No database repository configured"
            )

        try:
            # Try a simple operation
            if hasattr(self.repository, 'health_check'):
                result = await self.repository.health_check()
                if result:
                    return ComponentHealth(
                        name=self.name,
                        status=HealthStatus.HEALTHY,
                        message="Database connection successful"
                    )

            # Fallback to listing audits
            audits = await self.repository.list_audits(limit=1)
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                details={"test_operation": "list_audits", "result_count": len(audits)}
            )

        except Exception as e:
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}"
            )


class SystemResourceHealthCheck(HealthCheck):
    """Health check for system resources (CPU, memory, disk)."""

    def __init__(
        self,
        name: str = "system_resources",
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0,
        disk_threshold: float = 90.0,
        timeout: float = 2.0
    ):
        """Initialize system resource health check.

        Args:
            name: Name of the health check
            cpu_threshold: CPU usage threshold percentage
            memory_threshold: Memory usage threshold percentage
            disk_threshold: Disk usage threshold percentage
            timeout: Timeout for the check
        """
        super().__init__(name, timeout)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    async def check(self) -> ComponentHealth:
        """Check system resource usage."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            resources = await loop.run_in_executor(None, self._get_system_resources)

            # Check thresholds
            issues = []
            status = HealthStatus.HEALTHY

            if resources["cpu_percent"] > self.cpu_threshold:
                issues.append(f"High CPU usage: {resources['cpu_percent']:.1f}%")
                status = HealthStatus.DEGRADED

            if resources["memory_percent"] > self.memory_threshold:
                issues.append(f"High memory usage: {resources['memory_percent']:.1f}%")
                status = HealthStatus.DEGRADED

            if resources["disk_percent"] > self.disk_threshold:
                issues.append(f"High disk usage: {resources['disk_percent']:.1f}%")
                status = HealthStatus.DEGRADED

            message = "System resources normal"
            if issues:
                message = "; ".join(issues)
                if len(issues) > 2 or any("High" in issue for issue in issues):
                    status = HealthStatus.UNHEALTHY

            return ComponentHealth(
                name=self.name,
                status=status,
                message=message,
                details=resources
            )

        except Exception as e:
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {str(e)}"
            )

    def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100

        # Process count
        process_count = len(psutil.pids())

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_percent": disk_percent,
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "process_count": process_count
        }


class ExternalServiceHealthCheck(HealthCheck):
    """Health check for external services."""

    def __init__(
        self,
        name: str,
        url: str,
        expected_status: int = 200,
        timeout: float = 10.0
    ):
        """Initialize external service health check.

        Args:
            name: Name of the service
            url: URL to check
            expected_status: Expected HTTP status code
            timeout: Request timeout
        """
        super().__init__(name, timeout)
        self.url = url
        self.expected_status = expected_status

    async def check(self) -> ComponentHealth:
        """Check external service availability."""
        try:
            import aiohttp

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(self.url) as response:
                    if response.status == self.expected_status:
                        return ComponentHealth(
                            name=self.name,
                            status=HealthStatus.HEALTHY,
                            message=f"Service responding with status {response.status}",
                            details={
                                "url": self.url,
                                "status_code": response.status,
                                "content_type": response.headers.get("content-type", "unknown")
                            }
                        )
                    else:
                        return ComponentHealth(
                            name=self.name,
                            status=HealthStatus.UNHEALTHY,
                            message=f"Unexpected status code: {response.status} (expected {self.expected_status})",
                            details={
                                "url": self.url,
                                "status_code": response.status
                            }
                        )

        except Exception as e:
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to connect to service: {str(e)}",
                details={"url": self.url}
            )


class CustomHealthCheck(HealthCheck):
    """Custom health check with user-defined check function."""

    def __init__(
        self,
        name: str,
        check_func: Callable[[], Awaitable[ComponentHealth]],
        timeout: float = 5.0
    ):
        """Initialize custom health check.

        Args:
            name: Name of the health check
            check_func: Async function that performs the check
            timeout: Timeout for the check
        """
        super().__init__(name, timeout)
        self.check_func = check_func

    async def check(self) -> ComponentHealth:
        """Run custom health check function."""
        return await self.check_func()


class HealthChecker:
    """Main health checker that manages multiple health checks."""

    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, HealthCheck] = {}
        self.last_check_time: Optional[float] = None
        self.last_check_result: Optional[SystemHealth] = None
        self.cache_duration = 30  # Cache results for 30 seconds

    def add_check(self, check: HealthCheck) -> None:
        """Add a health check.

        Args:
            check: Health check to add
        """
        self.checks[check.name] = check
        logger.info(f"Added health check: {check.name}")

    def remove_check(self, name: str) -> None:
        """Remove a health check.

        Args:
            name: Name of the health check to remove
        """
        if name in self.checks:
            del self.checks[name]
            logger.info(f"Removed health check: {name}")

    async def check_health(self, use_cache: bool = True) -> SystemHealth:
        """Perform all health checks.

        Args:
            use_cache: Whether to use cached results if available

        Returns:
            System health status
        """
        now = time.time()

        # Use cached result if available and recent
        if (use_cache and
            self.last_check_result and
            self.last_check_time and
            (now - self.last_check_time) < self.cache_duration):
            return self.last_check_result

        # Run all health checks concurrently
        tasks = []
        for check in self.checks.values():
            tasks.append(check.run_check())

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Build component health dict
            components = {}
            for check, result in zip(self.checks.values(), results):
                if isinstance(result, Exception):
                    components[check.name] = ComponentHealth(
                        name=check.name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed: {str(result)}"
                    )
                else:
                    components[check.name] = result
        else:
            components = {}

        # Create system health
        system_health = SystemHealth.from_components(components)

        # Cache result
        self.last_check_time = now
        self.last_check_result = system_health

        return system_health

    async def check_readiness(self) -> bool:
        """Check if system is ready to accept requests.

        Returns:
            True if system is ready
        """
        try:
            health = await self.check_health()

            # Consider system ready if no critical components are unhealthy
            critical_components = ["database"]  # Define critical components

            for name in critical_components:
                if name in health.components:
                    if health.components[name].status == HealthStatus.UNHEALTHY:
                        return False

            return True

        except Exception as e:
            logger.error(f"Error checking readiness: {e}")
            return False

    def get_component_health(self, name: str) -> Optional[ComponentHealth]:
        """Get health status of a specific component.

        Args:
            name: Component name

        Returns:
            Component health if available
        """
        if self.last_check_result and name in self.last_check_result.components:
            return self.last_check_result.components[name]
        return None


# Global health checker instance
_global_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    return _global_health_checker


def set_health_checker(checker: HealthChecker) -> None:
    """Set global health checker instance."""
    global _global_health_checker
    _global_health_checker = checker