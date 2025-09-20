"""Artifact serving API routes for Tag Sentinel.

This module implements FastAPI routes for secure serving of audit artifacts
including HAR files, screenshots, and trace files with signed URL support.
"""

import logging
import os
import mimetypes
from typing import Optional, Any
from pathlib import Path
import hashlib
import time
from datetime import datetime
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.api.services.audit_service import AuditService, AuditNotFoundError
from app.api.routes.audits import get_audit_service
from app.api.schemas import ErrorResponse

logger = logging.getLogger(__name__)

# Create router with tags for OpenAPI documentation
router = APIRouter(
    prefix="/artifacts",
    tags=["Artifacts"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        403: {"model": ErrorResponse, "description": "Access Forbidden"},
        404: {"model": ErrorResponse, "description": "Artifact Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)

# Security scheme for signed URLs (optional)
security = HTTPBearer(auto_error=False)


class ArtifactService:
    """Service for managing artifact access and security."""

    def __init__(self, storage_path: str = "./artifacts", secret_key: str = "dev-secret"):
        """Initialize artifact service.

        Args:
            storage_path: Base path for artifact storage
            secret_key: Secret key for signing URLs (should be from config in prod)
        """
        self.storage_path = Path(storage_path)
        self.secret_key = secret_key
        self.url_expiry_seconds = 3600  # 1 hour default expiry

    def get_artifact_path(self, audit_id: str, artifact_type: str, filename: str) -> Path:
        """Get the full path to an artifact file.

        Args:
            audit_id: Audit identifier
            artifact_type: Type of artifact (har, screenshot, trace)
            filename: Name of the artifact file

        Returns:
            Full path to the artifact file
        """
        return self.storage_path / audit_id / artifact_type / filename

    def _sanitize_expires_in(self, expires_in: Any) -> int:
        """Sanitize expires_in value to a positive integer.

        Args:
            expires_in: Raw expiration value (int, float, string, etc.)

        Returns:
            Sanitized positive integer or default expiry seconds
        """
        try:
            # Handle None
            if expires_in is None:
                return self.url_expiry_seconds

            # Try direct integer conversion first to preserve precision
            try:
                value = int(expires_in)
            except (ValueError, TypeError):
                # Fall back to float conversion for cases like "3600.0"
                value = int(float(expires_in))

            # Ensure positive value and reasonable bounds
            # Max 10 years (315360000 seconds) to prevent datetime overflow
            if value <= 0:
                return self.url_expiry_seconds
            elif value > 315360000:  # 10 years
                return self.url_expiry_seconds
            else:
                return value

        except (ValueError, TypeError, OverflowError):
            # Any conversion error falls back to default
            return self.url_expiry_seconds

    def generate_signed_url(self, audit_id: str, artifact_type: str, filename: str, expires_in: Optional[int] = None) -> str:
        """Generate a signed URL for artifact access.

        Args:
            audit_id: Audit identifier
            artifact_type: Type of artifact
            filename: Name of the artifact file
            expires_in: Optional expiration time in seconds (defaults to service default)

        Returns:
            Signed URL with expiration
        """
        # Sanitize expiry_seconds to handle any input type safely
        expiry_seconds = self._sanitize_expires_in(expires_in)

        expires = int(time.time()) + expiry_seconds
        path = f"/api/artifacts/{audit_id}/{artifact_type}/{filename}"

        # Create signature
        message = f"{path}:{expires}"
        signature = hashlib.sha256(f"{message}:{self.secret_key}".encode()).hexdigest()

        return f"{path}?expires={expires}&signature={signature}"

    def verify_signed_url(self, path: str, expires: str, signature: str) -> bool:
        """Verify a signed URL.

        Args:
            path: URL path
            expires: Expiration timestamp
            signature: URL signature

        Returns:
            True if signature is valid and not expired
        """
        try:
            expires_int = int(expires)
            if time.time() > expires_int:
                return False  # Expired

            message = f"{path}:{expires}"
            expected_signature = hashlib.sha256(f"{message}:{self.secret_key}".encode()).hexdigest()
            return signature == expected_signature

        except (ValueError, TypeError):
            return False

    def get_content_type(self, filename: str) -> str:
        """Get content type for artifact file.

        Args:
            filename: Name of the file

        Returns:
            MIME content type
        """
        content_type, _ = mimetypes.guess_type(filename)

        # Override for common audit artifact types
        if filename.endswith('.har'):
            return 'application/json'
        elif filename.endswith('.trace'):
            return 'application/json'
        elif filename.endswith('.png'):
            return 'image/png'
        elif filename.endswith('.pdf'):
            return 'application/pdf'

        return content_type or 'application/octet-stream'


# Shared artifact service instance
_artifact_service_instance = None

def get_artifact_service() -> ArtifactService:
    """Dependency to provide artifact service instance."""
    global _artifact_service_instance
    if _artifact_service_instance is None:
        _artifact_service_instance = ArtifactService()
    return _artifact_service_instance


@router.get(
    "/{audit_id}/{artifact_type}/{filename}",
    summary="Download audit artifact",
    description="""
    Download audit artifacts including HAR files, screenshots, and trace files.

    ## Artifact Types

    - **har**: HTTP Archive (HAR) files containing detailed network activity
    - **screenshot**: Page screenshots in PNG format
    - **trace**: Browser trace files for performance analysis
    - **pdf**: PDF reports and summaries

    ## Security

    Artifact access can be controlled through:
    - **Signed URLs**: Time-limited access with cryptographic signatures
    - **Direct Access**: Immediate access for authorized users (future)

    ## Caching

    Artifacts include appropriate caching headers:
    - Long-term caching for immutable artifacts
    - ETags for efficient conditional requests
    - Proper content-type detection

    ## Examples

    ```bash
    # Download HAR file
    curl "http://localhost:8000/api/artifacts/audit123/har/network-trace.har"

    # Download screenshot with signed URL
    curl "http://localhost:8000/api/artifacts/audit123/screenshot/page-001.png?expires=1642252800&signature=abc123..."

    # Download trace file
    curl "http://localhost:8000/api/artifacts/audit123/trace/performance.trace"
    ```
    """,
    responses={
        200: {
            "description": "Artifact file download",
            "content": {
                "application/json": {"description": "HAR files"},
                "image/png": {"description": "Screenshot files"},
                "application/octet-stream": {"description": "Other artifact files"}
            }
        }
    }
)
async def download_artifact(
    audit_id: str,
    artifact_type: str,
    filename: str,
    http_request: Request,
    expires: Optional[str] = Query(None, description="URL expiration timestamp (for signed URLs)"),
    signature: Optional[str] = Query(None, description="URL signature (for signed URLs)"),
    audit_service: AuditService = Depends(get_audit_service),
    artifact_service: ArtifactService = Depends(get_artifact_service),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> FileResponse:
    """Download an artifact file for an audit."""
    request_id = getattr(http_request.state, "request_id", None)

    try:
        # Verify audit exists
        await audit_service.get_audit(audit_id)

        # Verify signed URL if provided
        if expires and signature:
            path = f"/api/artifacts/{audit_id}/{artifact_type}/{filename}"
            if not artifact_service.verify_signed_url(path, expires, signature):
                logger.warning(
                    f"Invalid signed URL for artifact {audit_id}/{artifact_type}/{filename}",
                    extra={"request_id": request_id, "audit_id": audit_id}
                )
                raise HTTPException(
                    status_code=403,
                    detail="Invalid or expired signed URL"
                )

        # Validate artifact type
        allowed_types = ["har", "screenshot", "trace", "pdf"]
        if artifact_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid artifact type. Must be one of: {', '.join(allowed_types)}"
            )

        # Get artifact file path
        artifact_path = artifact_service.get_artifact_path(audit_id, artifact_type, filename)

        # Check if file exists
        if not artifact_path.exists():
            logger.warning(
                f"Artifact file not found: {artifact_path}",
                extra={"request_id": request_id, "audit_id": audit_id}
            )
            raise HTTPException(
                status_code=404,
                detail=f"Artifact '{filename}' not found"
            )

        # Security check: ensure path is within allowed directory
        try:
            artifact_path.resolve().relative_to(artifact_service.storage_path.resolve())
        except ValueError:
            logger.error(
                f"Path traversal attempt: {artifact_path}",
                extra={"request_id": request_id, "audit_id": audit_id}
            )
            raise HTTPException(
                status_code=403,
                detail="Access denied"
            )

        # Get content type
        content_type = artifact_service.get_content_type(filename)

        # Set appropriate headers
        headers = {
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            "Content-Disposition": f'attachment; filename="{quote(filename)}"'
        }

        logger.info(
            f"Serving artifact {audit_id}/{artifact_type}/{filename}",
            extra={
                "request_id": request_id,
                "audit_id": audit_id,
                "artifact_type": artifact_type,
                "filename": filename,
                "content_type": content_type
            }
        )

        return FileResponse(
            path=artifact_path,
            media_type=content_type,
            headers=headers,
            filename=filename
        )

    except AuditNotFoundError:
        logger.warning(
            f"Audit {audit_id} not found for artifact access",
            extra={"request_id": request_id, "audit_id": audit_id}
        )
        raise HTTPException(
            status_code=404,
            detail=f"Audit '{audit_id}' not found"
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        logger.error(
            f"Failed to serve artifact {audit_id}/{artifact_type}/{filename}: {e}",
            extra={"request_id": request_id, "audit_id": audit_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to serve artifact"
        )


@router.post(
    "/{audit_id}/signed-urls",
    summary="Generate signed artifact URLs",
    description="""
    Generate signed URLs for secure artifact access with time-limited validity.

    ## Request Body

    ```json
    {
        "artifacts": [
            {"type": "har", "filename": "network-trace.har"},
            {"type": "screenshot", "filename": "page-001.png"}
        ],
        "expires_in": 3600
    }
    ```

    ## Response

    ```json
    {
        "urls": {
            "har/network-trace.har": "https://api.example.com/artifacts/audit123/har/network-trace.har?expires=1642252800&signature=abc123...",
            "screenshot/page-001.png": "https://api.example.com/artifacts/audit123/screenshot/page-001.png?expires=1642252800&signature=def456..."
        },
        "expires_at": "2024-01-15T12:00:00Z"
    }
    ```

    This endpoint is useful for:
    - Frontend applications needing temporary artifact access
    - Sharing audit results with external stakeholders
    - Implementing download links in email reports
    """,
    responses={
        200: {
            "description": "Signed URLs generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "urls": {
                            "har/network-trace.har": "/api/artifacts/audit123/har/network-trace.har?expires=1642252800&signature=abc123"
                        },
                        "expires_at": "2024-01-15T12:00:00Z"
                    }
                }
            }
        }
    }
)
async def generate_signed_urls(
    audit_id: str,
    request_body: dict,
    http_request: Request,
    audit_service: AuditService = Depends(get_audit_service),
    artifact_service: ArtifactService = Depends(get_artifact_service)
) -> dict:
    """Generate signed URLs for multiple artifacts."""
    request_id = getattr(http_request.state, "request_id", None)

    try:
        # Verify audit exists
        await audit_service.get_audit(audit_id)

        # Extract request parameters
        artifacts = request_body.get("artifacts", [])
        raw_expires_in = request_body.get("expires_in", 3600)

        if not artifacts:
            raise HTTPException(
                status_code=400,
                detail="No artifacts specified"
            )

        # Sanitize expires_in using the same logic as the service
        sanitized_expires_in = artifact_service._sanitize_expires_in(raw_expires_in)

        # Generate signed URLs
        urls = {}
        for artifact in artifacts:
            artifact_type = artifact.get("type")
            filename = artifact.get("filename")

            if not artifact_type or not filename:
                raise HTTPException(
                    status_code=400,
                    detail="Each artifact must specify 'type' and 'filename'"
                )

            # Verify artifact exists
            artifact_path = artifact_service.get_artifact_path(audit_id, artifact_type, filename)
            if not artifact_path.exists():
                logger.warning(
                    f"Artifact not found for signed URL: {artifact_path}",
                    extra={"request_id": request_id, "audit_id": audit_id}
                )
                continue  # Skip missing artifacts

            # Generate signed URL with sanitized expiration
            signed_url = artifact_service.generate_signed_url(audit_id, artifact_type, filename, sanitized_expires_in)
            urls[f"{artifact_type}/{filename}"] = signed_url

        # Calculate expiry time using sanitized value and format as ISO8601
        expires_timestamp = time.time() + sanitized_expires_in
        expires_at = datetime.utcfromtimestamp(expires_timestamp).isoformat() + "Z"

        logger.info(
            f"Generated {len(urls)} signed URLs for audit {audit_id}",
            extra={"request_id": request_id, "audit_id": audit_id}
        )

        return {
            "urls": urls,
            "expires_at": expires_at,
            "expires_in": sanitized_expires_in
        }

    except AuditNotFoundError:
        logger.warning(
            f"Audit {audit_id} not found for signed URL generation",
            extra={"request_id": request_id, "audit_id": audit_id}
        )
        raise HTTPException(
            status_code=404,
            detail=f"Audit '{audit_id}' not found"
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        logger.error(
            f"Failed to generate signed URLs for audit {audit_id}: {e}",
            extra={"request_id": request_id, "audit_id": audit_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate signed URLs"
        )