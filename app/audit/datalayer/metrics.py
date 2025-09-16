"""DataLayer metrics collection and monitoring facade.

This module provides a simple interface for metrics collection,
delegating to DataLayerService for the actual implementation.
Aligns with TASK-4.16 metrics requirements.
"""

from typing import Dict, Any, Optional
from .service import DataLayerService


class DataLayerMetrics:
    """Facade for DataLayer metrics collection."""

    def __init__(self, service: DataLayerService):
        """Initialize metrics facade.

        Args:
            service: DataLayerService instance to delegate to
        """
        self.service = service

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary metrics.

        Returns:
            Dictionary containing processing statistics
        """
        return self.service.get_processing_summary()

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status.

        Returns:
            Dictionary containing health status and recommendations
        """
        return self.service.get_health_status()

    def get_resilience_metrics(self) -> Dict[str, Any]:
        """Get resilience and error recovery metrics.

        Returns:
            Dictionary containing resilience statistics
        """
        return self.service.get_resilience_metrics()

    def health_check(self) -> Dict[str, Any]:
        """Perform basic health check.

        Returns:
            Dictionary containing basic health status
        """
        return self.service.health_check()

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary metrics.

        Returns:
            Dictionary containing error statistics
        """
        return self.service.get_error_summary()


# Convenience functions for global access
_default_service: Optional[DataLayerService] = None


def set_default_service(service: DataLayerService) -> None:
    """Set the default service for global metrics access.

    Args:
        service: DataLayerService instance to use as default
    """
    global _default_service
    _default_service = service


def get_processing_summary() -> Dict[str, Any]:
    """Get processing summary from default service.

    Returns:
        Processing summary metrics

    Raises:
        RuntimeError: If no default service is set
    """
    if _default_service is None:
        raise RuntimeError("No default DataLayerService set. Call set_default_service() first.")
    return _default_service.get_processing_summary()


def get_health_status() -> Dict[str, Any]:
    """Get health status from default service.

    Returns:
        Health status metrics

    Raises:
        RuntimeError: If no default service is set
    """
    if _default_service is None:
        raise RuntimeError("No default DataLayerService set. Call set_default_service() first.")
    return _default_service.get_health_status()


def get_resilience_metrics() -> Dict[str, Any]:
    """Get resilience metrics from default service.

    Returns:
        Resilience metrics

    Raises:
        RuntimeError: If no default service is set
    """
    if _default_service is None:
        raise RuntimeError("No default DataLayerService set. Call set_default_service() first.")
    return _default_service.get_resilience_metrics()


def health_check() -> Dict[str, Any]:
    """Perform health check using default service.

    Returns:
        Health check results

    Raises:
        RuntimeError: If no default service is set
    """
    if _default_service is None:
        raise RuntimeError("No default DataLayerService set. Call set_default_service() first.")
    return _default_service.health_check()