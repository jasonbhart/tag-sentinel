"""API routes for Tag Sentinel REST API.

This module exports all route modules for the Tag Sentinel API,
providing organized endpoints for audit management, exports, and system operations.
"""

from .audits import router as audits_router
from .exports import router as exports_router
from .artifacts import router as artifacts_router

__all__ = [
    "audits_router",
    "exports_router",
    "artifacts_router",
]