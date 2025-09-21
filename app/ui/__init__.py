"""Tag Sentinel Web UI package.

This package provides the web interface for Tag Sentinel, including
server-rendered pages, templates, and static assets.
"""

from .routes import ui_router

__all__ = ["ui_router"]