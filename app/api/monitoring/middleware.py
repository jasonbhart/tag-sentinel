"""Monitoring middleware for Tag Sentinel API.

This module provides middleware for collecting metrics, health checks,
and request/response monitoring with minimal performance impact.
"""

import logging
import time
import asyncio
from typing import Callable, Optional, Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .metrics import MetricsCollector, get_metrics_collector
from .health import HealthChecker

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting API metrics."""

    def __init__(
        self,
        app,
        collector: Optional[MetricsCollector] = None,
        include_paths: Optional[list] = None,
        exclude_paths: Optional[list] = None
    ):
        """Initialize metrics middleware.

        Args:
            app: FastAPI application
            collector: Metrics collector (uses global if None)
            include_paths: Paths to include (all if None)
            exclude_paths: Paths to exclude from metrics
        """
        super().__init__(app)
        self.collector = collector or get_metrics_collector()

        # Path filtering
        self.include_paths = include_paths
        self.exclude_paths = exclude_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico"
        ]

        logger.info("MetricsMiddleware initialized")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for each request."""
        # Check if path should be monitored
        if not self._should_monitor_path(request.url.path):
            return await call_next(request)

        # Record request start
        start_time = time.time()
        request.state.start_time = start_time

        # Add request ID for tracing
        request_id = getattr(request.state, "request_id", None)
        if not request_id:
            import uuid
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id

        # Process request
        response = None
        status_code = 500
        error_occurred = False

        try:
            response = await call_next(request)
            status_code = response.status_code

        except Exception as e:
            error_occurred = True
            logger.error(f"Error in request {request_id}: {e}", exc_info=True)

            # Create error response
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "An internal error occurred",
                    "request_id": request_id
                }
            )
            status_code = 500

        finally:
            # Calculate duration
            duration = time.time() - start_time

            # Record metrics
            self._record_request_metrics(request, status_code, duration, error_occurred)

            # Add timing headers
            if response:
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Response-Time"] = f"{duration:.3f}"

        return response

    def _should_monitor_path(self, path: str) -> bool:
        """Check if path should be monitored."""
        # Check exclude list
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return False

        # Check include list
        if self.include_paths is not None:
            return any(path.startswith(included) for included in self.include_paths)

        return True

    def _record_request_metrics(
        self,
        request: Request,
        status_code: int,
        duration: float,
        error_occurred: bool
    ) -> None:
        """Record metrics for the request."""
        try:
            # Normalize endpoint for metrics
            endpoint = self._normalize_endpoint(request.url.path)

            # Record main request metrics
            self.collector.record_request(
                method=request.method,
                endpoint=endpoint,
                status_code=status_code,
                duration=duration
            )

            # Record additional metrics
            labels = {
                "method": request.method,
                "endpoint": endpoint,
                "status_code": str(status_code)
            }

            # Record request size if available
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    size = int(content_length)
                    self.collector.record_histogram("api_request_size_bytes", size, labels)
                except ValueError:
                    pass

            # Record concurrent requests
            active_requests = getattr(request.app.state, "active_requests", 0)
            self.collector.set_gauge("api_active_requests", active_requests)

            # Record error details if applicable
            if error_occurred:
                error_labels = labels.copy()
                error_labels["error_type"] = "server_error"
                self.collector.increment_counter("api_internal_errors_total", 1.0, error_labels)

        except Exception as e:
            logger.error(f"Error recording metrics: {e}")

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for metrics."""
        # Replace path parameters with placeholders
        import re

        # Common patterns for path parameters
        patterns = [
            (r"/audits/[^/]+", "/audits/{audit_id}"),
            (r"/audits/[^/]+/exports/[^/]+", "/audits/{audit_id}/exports/{export_id}"),
            (r"/users/[^/]+", "/users/{user_id}"),
            (r"/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", "/{uuid}"),
            (r"/\d+", "/{id}")
        ]

        normalized = path
        for pattern, replacement in patterns:
            normalized = re.sub(pattern, replacement, normalized)

        return normalized


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware for health check monitoring."""

    def __init__(
        self,
        app,
        health_checker: Optional[HealthChecker] = None,
        health_endpoint: str = "/health",
        ready_endpoint: str = "/ready"
    ):
        """Initialize health check middleware.

        Args:
            app: FastAPI application
            health_checker: Health checker instance
            health_endpoint: Endpoint for health checks
            ready_endpoint: Endpoint for readiness checks
        """
        super().__init__(app)
        self.health_checker = health_checker
        self.health_endpoint = health_endpoint
        self.ready_endpoint = ready_endpoint

        logger.info("HealthCheckMiddleware initialized")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle health check requests."""
        path = request.url.path

        # Handle health check endpoint
        if path == self.health_endpoint:
            return await self._handle_health_check()

        # Handle readiness check endpoint
        if path == self.ready_endpoint:
            return await self._handle_readiness_check()

        # Continue with normal request processing
        return await call_next(request)

    async def _handle_health_check(self) -> JSONResponse:
        """Handle health check request."""
        if not self.health_checker:
            return JSONResponse(
                status_code=200,
                content={"status": "ok", "message": "Service is running"}
            )

        try:
            health_status = await self.health_checker.check_health()

            status_code = 200 if health_status.is_healthy else 503

            return JSONResponse(
                status_code=status_code,
                content={
                    "status": "healthy" if health_status.is_healthy else "unhealthy",
                    "timestamp": health_status.timestamp,
                    "components": {
                        name: {
                            "status": "healthy" if comp.is_healthy else "unhealthy",
                            "message": comp.message,
                            "details": comp.details
                        }
                        for name, comp in health_status.components.items()
                    }
                }
            )

        except Exception as e:
            logger.error(f"Error in health check: {e}", exc_info=True)
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": f"Health check failed: {str(e)}"
                }
            )

    async def _handle_readiness_check(self) -> JSONResponse:
        """Handle readiness check request."""
        if not self.health_checker:
            return JSONResponse(
                status_code=200,
                content={"status": "ready", "message": "Service is ready"}
            )

        try:
            is_ready = await self.health_checker.check_readiness()

            status_code = 200 if is_ready else 503

            return JSONResponse(
                status_code=status_code,
                content={
                    "status": "ready" if is_ready else "not_ready",
                    "message": "Service readiness check"
                }
            )

        except Exception as e:
            logger.error(f"Error in readiness check: {e}", exc_info=True)
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": f"Readiness check failed: {str(e)}"
                }
            )


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking concurrent requests and resource usage."""

    def __init__(self, app, max_concurrent_requests: Optional[int] = None):
        """Initialize request tracking middleware.

        Args:
            app: FastAPI application
            max_concurrent_requests: Maximum allowed concurrent requests
        """
        super().__init__(app)
        self.max_concurrent_requests = max_concurrent_requests

        # Initialize state
        if not hasattr(app.state, "active_requests"):
            app.state.active_requests = 0
            app.state.request_lock = asyncio.Lock()

        logger.info("RequestTrackingMiddleware initialized")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track request lifecycle."""
        # Check concurrent request limit
        if self.max_concurrent_requests:
            async with request.app.state.request_lock:
                if request.app.state.active_requests >= self.max_concurrent_requests:
                    return JSONResponse(
                        status_code=503,
                        content={
                            "error": "service_unavailable",
                            "message": "Server is temporarily overloaded"
                        }
                    )

        # Increment active request count
        async with request.app.state.request_lock:
            request.app.state.active_requests += 1

        try:
            # Process request
            response = await call_next(request)
            return response

        finally:
            # Decrement active request count
            async with request.app.state.request_lock:
                request.app.state.active_requests = max(0, request.app.state.active_requests - 1)


class ErrorTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking and categorizing errors."""

    def __init__(self, app, collector: Optional[MetricsCollector] = None):
        """Initialize error tracking middleware.

        Args:
            app: FastAPI application
            collector: Metrics collector for error metrics
        """
        super().__init__(app)
        self.collector = collector or get_metrics_collector()

        logger.info("ErrorTrackingMiddleware initialized")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track errors and exceptions."""
        try:
            response = await call_next(request)

            # Track HTTP errors
            if response.status_code >= 400:
                self._record_http_error(request, response.status_code)

            return response

        except Exception as e:
            # Track unhandled exceptions
            self._record_exception(request, e)

            # Re-raise the exception to be handled by FastAPI
            raise

    def _record_http_error(self, request: Request, status_code: int) -> None:
        """Record HTTP error metrics."""
        try:
            endpoint = request.url.path
            method = request.method

            labels = {
                "endpoint": endpoint,
                "method": method,
                "status_code": str(status_code),
                "error_category": self._categorize_http_error(status_code)
            }

            self.collector.increment_counter("api_http_errors_total", 1.0, labels)

        except Exception as e:
            logger.error(f"Error recording HTTP error metric: {e}")

    def _record_exception(self, request: Request, exception: Exception) -> None:
        """Record exception metrics."""
        try:
            endpoint = request.url.path
            method = request.method
            exception_type = type(exception).__name__

            labels = {
                "endpoint": endpoint,
                "method": method,
                "exception_type": exception_type,
                "error_category": "exception"
            }

            self.collector.increment_counter("api_exceptions_total", 1.0, labels)

        except Exception as e:
            logger.error(f"Error recording exception metric: {e}")

    def _categorize_http_error(self, status_code: int) -> str:
        """Categorize HTTP error by status code."""
        if 400 <= status_code < 500:
            return "client_error"
        elif 500 <= status_code < 600:
            return "server_error"
        else:
            return "other_error"