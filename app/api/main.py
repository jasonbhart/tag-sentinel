"""FastAPI application for Tag Sentinel REST API.

This module configures the FastAPI application with middleware, error handling,
and OpenAPI documentation for the Tag Sentinel audit API.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict
import uuid

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.schemas import ErrorResponse, HealthResponse
from app.api.routes import audits_router, exports_router, artifacts_router


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application metadata
APP_VERSION = "1.0.0"
APP_TITLE = "Tag Sentinel API"
APP_DESCRIPTION = """
Tag Sentinel REST API provides comprehensive web analytics auditing and monitoring capabilities.

## Features

* **Audit Management**: Create, monitor, and manage web analytics audits
* **Real-time Status**: Track audit progress and status in real-time
* **Data Exports**: Export audit results in multiple formats (JSON, CSV, NDJSON)
* **Analytics Detection**: Detect and analyze GA4, GTM, and other analytics tags
* **Cookie Analysis**: Inventory and classify cookies for privacy compliance
* **Data Layer Validation**: Validate and monitor website data layer integrity

## Authentication

Authentication is currently disabled but the framework is ready for future API token
and OAuth integration.

## Rate Limiting

Rate limiting is not currently enforced but the framework is ready for future
implementation.

## Support

For issues and feedback, visit: https://github.com/your-org/tag-sentinel/issues
"""

# Global application state
app_start_time = datetime.utcnow()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    # Create FastAPI app with OpenAPI configuration
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        # API versioning handled by router prefix
        # Additional OpenAPI metadata
        contact={
            "name": "Tag Sentinel Team",
            "url": "https://github.com/your-org/tag-sentinel",
            "email": "support@example.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        servers=[
            {
                "url": "/api",
                "description": "Current API server"
            }
        ]
    )

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Total-Count", "X-Has-More"],
    )

    # Add request tracking middleware
    @app.middleware("http")
    async def add_request_id_and_logging(request: Request, call_next):
        """Add request ID and logging middleware."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Log request start
        start_time = time.time()
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_host": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
        )

        try:
            # Process request
            response = await call_next(request)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            # Log request completion
            duration = time.time() - start_time
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                }
            )

            return response

        except Exception as e:
            # Log request error
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)}",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "duration_ms": round(duration * 1000, 2),
                },
                exc_info=True
            )
            raise

    # Global exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with consistent error format."""
        request_id = getattr(request.state, "request_id", None)

        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=f"http_{exc.status_code}",
                message=str(exc.detail),
                request_id=request_id,
                timestamp=datetime.utcnow()
            ).model_dump(mode='json')
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors with detailed information."""
        request_id = getattr(request.state, "request_id", None)

        # Extract validation error details
        errors = []
        for error in exc.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"][1:])  # Skip 'body'
            errors.append({
                "field": field_path,
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })

        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error",
                message="Request validation failed",
                details={"validation_errors": errors},
                request_id=request_id,
                timestamp=datetime.utcnow()
            ).model_dump(mode='json')
        )

    @app.exception_handler(StarletteHTTPException)
    async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle Starlette HTTP exceptions."""
        request_id = getattr(request.state, "request_id", None)

        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=f"http_{exc.status_code}",
                message=str(exc.detail),
                request_id=request_id,
                timestamp=datetime.utcnow()
            ).model_dump(mode='json')
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        request_id = getattr(request.state, "request_id", None)

        logger.error(
            f"Unhandled exception in request {request_id}: {str(exc)}",
            exc_info=True
        )

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_server_error",
                message="An unexpected error occurred",
                request_id=request_id,
                timestamp=datetime.utcnow()
            ).model_dump(mode='json')
        )

    # Health check endpoint
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Health check",
        description="Returns the current health status of the API and its dependencies"
    )
    async def health_check():
        """Health check endpoint for monitoring and operational purposes."""
        # Calculate uptime
        uptime = (datetime.utcnow() - app_start_time).total_seconds()

        # Check service health (implement actual checks in production)
        services = {
            "database": "healthy",  # TODO: Implement actual database health check
            "cache": "healthy",     # TODO: Implement actual cache health check
            "audit_runner": "healthy",  # TODO: Implement audit runner health check
            "browser_engine": "healthy",  # TODO: Implement browser engine health check
        }

        # Determine overall status
        service_statuses = list(services.values())
        if all(status == "healthy" for status in service_statuses):
            overall_status = "healthy"
        elif any(status == "unhealthy" for status in service_statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"

        return HealthResponse(
            status=overall_status,
            version=APP_VERSION,
            timestamp=datetime.utcnow(),
            services=services,
            uptime_seconds=uptime
        )

    # Root redirect
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint that redirects to API documentation."""
        return JSONResponse(
            content={
                "message": "Tag Sentinel API",
                "version": APP_VERSION,
                "documentation": "/docs",
                "openapi": "/openapi.json"
            }
        )

    # Include API routers
    app.include_router(audits_router, prefix="/api")
    app.include_router(exports_router, prefix="/api")
    app.include_router(artifacts_router, prefix="/api")

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    # Development server configuration
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
    )