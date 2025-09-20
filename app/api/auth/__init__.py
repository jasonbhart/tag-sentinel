"""Authentication and authorization module for Tag Sentinel API.

This module provides authentication middleware, token management,
and authorization scaffolding ready for future security implementations.
"""

from .middleware import AuthenticationMiddleware
from .providers import AuthProvider, NoAuthProvider, APIKeyProvider, JWTProvider
from .models import User, AuthToken, Permission
from .dependencies import get_current_user, require_permissions
from .config import AuthConfig

__all__ = [
    "AuthenticationMiddleware",
    "AuthProvider",
    "NoAuthProvider",
    "APIKeyProvider",
    "JWTProvider",
    "User",
    "AuthToken",
    "Permission",
    "get_current_user",
    "require_permissions",
    "AuthConfig"
]