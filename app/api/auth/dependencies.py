"""Authentication dependencies for Tag Sentinel API.

This module provides FastAPI dependency functions for authentication
and authorization, integrating with the authentication middleware.
"""

import logging
from typing import List, Optional, Callable, Any
from functools import wraps

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .models import User, AuthContext, Permission, Role
from .exceptions import InsufficientPermissionsError, AuthenticationError

logger = logging.getLogger(__name__)

# Security scheme for dependency injection
security = HTTPBearer(auto_error=False)


async def get_auth_context(request: Request) -> AuthContext:
    """Get authentication context from request state.

    This dependency extracts the authentication context that was
    populated by the authentication middleware.

    Args:
        request: FastAPI request object

    Returns:
        Authentication context

    Raises:
        HTTPException: If authentication context is not available
    """
    auth_context = getattr(request.state, "auth", None)
    if auth_context is None:
        raise HTTPException(
            status_code=500,
            detail="Authentication context not available. Ensure authentication middleware is installed."
        )
    return auth_context


async def get_current_user(auth_context: AuthContext = Depends(get_auth_context)) -> User:
    """Get current authenticated user.

    Args:
        auth_context: Authentication context from middleware

    Returns:
        Current authenticated user

    Raises:
        HTTPException: If user is not authenticated
    """
    if not auth_context.is_authenticated or not auth_context.user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return auth_context.user


async def get_current_user_optional(auth_context: AuthContext = Depends(get_auth_context)) -> Optional[User]:
    """Get current authenticated user, returning None if not authenticated.

    Args:
        auth_context: Authentication context from middleware

    Returns:
        Current authenticated user or None
    """
    if auth_context.is_authenticated and auth_context.user:
        return auth_context.user
    return None


def require_permissions(*permissions: Permission) -> Callable:
    """Dependency factory to require specific permissions.

    Args:
        *permissions: Required permissions

    Returns:
        Dependency function that validates permissions

    Example:
        @app.get("/admin/users")
        async def get_users(user: User = Depends(require_permissions(Permission.ADMIN_USERS))):
            return users
    """
    async def check_permissions(user: User = Depends(get_current_user)) -> User:
        """Check if user has required permissions."""
        missing_permissions = []
        for permission in permissions:
            if not user.has_permission(permission):
                missing_permissions.append(permission.value)

        if missing_permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {missing_permissions}"
            )

        return user

    return check_permissions


def require_any_permission(*permissions: Permission) -> Callable:
    """Dependency factory to require any of the specified permissions.

    Args:
        *permissions: List of permissions (user needs at least one)

    Returns:
        Dependency function that validates permissions
    """
    async def check_any_permission(user: User = Depends(get_current_user)) -> User:
        """Check if user has any of the required permissions."""
        if not user.has_any_permission(list(permissions)):
            permission_names = [p.value for p in permissions]
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required any of: {permission_names}"
            )

        return user

    return check_any_permission


def require_roles(*roles: Role) -> Callable:
    """Dependency factory to require specific roles.

    Args:
        *roles: Required roles

    Returns:
        Dependency function that validates roles
    """
    async def check_roles(user: User = Depends(get_current_user)) -> User:
        """Check if user has required roles."""
        missing_roles = []
        for role in roles:
            if role not in user.roles:
                missing_roles.append(role.value)

        if missing_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient roles. Required: {missing_roles}"
            )

        return user

    return check_roles


def require_superuser() -> Callable:
    """Dependency to require superuser privileges.

    Returns:
        Dependency function that validates superuser status
    """
    async def check_superuser(user: User = Depends(get_current_user)) -> User:
        """Check if user is a superuser."""
        if not user.is_superuser:
            raise HTTPException(
                status_code=403,
                detail="Superuser privileges required"
            )

        return user

    return check_superuser


def require_active_user() -> Callable:
    """Dependency to require active user account.

    Returns:
        Dependency function that validates user is active
    """
    async def check_active_user(user: User = Depends(get_current_user)) -> User:
        """Check if user account is active."""
        if not user.is_active:
            raise HTTPException(
                status_code=403,
                detail="Active user account required"
            )

        return user

    return check_active_user


# Convenience dependency combinations

async def get_admin_user(user: User = Depends(require_roles(Role.ADMIN))) -> User:
    """Get current user requiring admin role."""
    return user


async def get_auditor_user(user: User = Depends(require_roles(Role.AUDITOR, Role.ANALYST, Role.ADMIN))) -> User:
    """Get current user requiring auditor, analyst, or admin role."""
    return user


async def get_analyst_user(user: User = Depends(require_roles(Role.ANALYST, Role.ADMIN))) -> User:
    """Get current user requiring analyst or admin role."""
    return user


# Permission-specific dependencies for common operations

async def require_audit_create(user: User = Depends(require_permissions(Permission.AUDIT_CREATE))) -> User:
    """Require audit creation permission."""
    return user


async def require_audit_read(user: User = Depends(require_permissions(Permission.AUDIT_READ))) -> User:
    """Require audit read permission."""
    return user


async def require_export_access(
    user: User = Depends(require_any_permission(
        Permission.EXPORT_REQUEST_LOGS,
        Permission.EXPORT_COOKIES,
        Permission.EXPORT_TAGS,
        Permission.EXPORT_DATA_LAYER
    ))
) -> User:
    """Require any export permission."""
    return user


async def require_artifact_access(user: User = Depends(require_permissions(Permission.ARTIFACT_READ))) -> User:
    """Require artifact read permission."""
    return user


# Decorators for function-level authorization

def requires_permission(permission: Permission):
    """Decorator to require permission for a function.

    Args:
        permission: Required permission

    Example:
        @requires_permission(Permission.AUDIT_CREATE)
        async def create_audit():
            pass
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract auth context from kwargs or request
            auth_context = None
            for key, value in kwargs.items():
                if isinstance(value, AuthContext):
                    auth_context = value
                    break

            if not auth_context or not auth_context.is_authenticated:
                raise HTTPException(status_code=401, detail="Authentication required")

            if not auth_context.user.has_permission(permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission '{permission.value}' required"
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator


def requires_role(role: Role):
    """Decorator to require role for a function.

    Args:
        role: Required role
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract auth context from kwargs
            auth_context = None
            for key, value in kwargs.items():
                if isinstance(value, AuthContext):
                    auth_context = value
                    break

            if not auth_context or not auth_context.is_authenticated:
                raise HTTPException(status_code=401, detail="Authentication required")

            if role not in auth_context.user.roles:
                raise HTTPException(
                    status_code=403,
                    detail=f"Role '{role.value}' required"
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Utility functions for manual authorization checks

def check_permission(user: User, permission: Permission) -> bool:
    """Check if user has specific permission.

    Args:
        user: User to check
        permission: Permission to check

    Returns:
        True if user has permission
    """
    return user.has_permission(permission)


def check_any_permission(user: User, permissions: List[Permission]) -> bool:
    """Check if user has any of the specified permissions.

    Args:
        user: User to check
        permissions: List of permissions to check

    Returns:
        True if user has any permission
    """
    return user.has_any_permission(permissions)


def check_role(user: User, role: Role) -> bool:
    """Check if user has specific role.

    Args:
        user: User to check
        role: Role to check

    Returns:
        True if user has role
    """
    return role in user.roles


def enforce_permission(user: User, permission: Permission) -> None:
    """Enforce permission requirement, raising exception if not met.

    Args:
        user: User to check
        permission: Required permission

    Raises:
        HTTPException: If user lacks permission
    """
    if not user.has_permission(permission):
        raise HTTPException(
            status_code=403,
            detail=f"Permission '{permission.value}' required"
        )


def enforce_role(user: User, role: Role) -> None:
    """Enforce role requirement, raising exception if not met.

    Args:
        user: User to check
        role: Required role

    Raises:
        HTTPException: If user lacks role
    """
    if role not in user.roles:
        raise HTTPException(
            status_code=403,
            detail=f"Role '{role.value}' required"
        )