"""Authentication and authorization models for Tag Sentinel API.

This module defines data models for users, tokens, permissions,
and related authentication/authorization structures.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """Available permissions in the system."""
    # Audit permissions
    AUDIT_CREATE = "audit:create"
    AUDIT_READ = "audit:read"
    AUDIT_UPDATE = "audit:update"
    AUDIT_DELETE = "audit:delete"
    AUDIT_LIST = "audit:list"

    # Export permissions
    EXPORT_REQUEST_LOGS = "export:request_logs"
    EXPORT_COOKIES = "export:cookies"
    EXPORT_TAGS = "export:tags"
    EXPORT_DATA_LAYER = "export:data_layer"

    # Artifact permissions
    ARTIFACT_READ = "artifact:read"
    ARTIFACT_GENERATE_SIGNED_URL = "artifact:generate_signed_url"

    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_TOKENS = "admin:tokens"
    ADMIN_SYSTEM = "admin:system"


class Role(str, Enum):
    """Predefined roles with associated permissions."""
    VIEWER = "viewer"
    AUDITOR = "auditor"
    ANALYST = "analyst"
    ADMIN = "admin"


# Role-based permission mappings
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.AUDIT_READ,
        Permission.AUDIT_LIST,
        Permission.ARTIFACT_READ,
    },
    Role.AUDITOR: {
        Permission.AUDIT_CREATE,
        Permission.AUDIT_READ,
        Permission.AUDIT_LIST,
        Permission.ARTIFACT_READ,
        Permission.ARTIFACT_GENERATE_SIGNED_URL,
    },
    Role.ANALYST: {
        Permission.AUDIT_CREATE,
        Permission.AUDIT_READ,
        Permission.AUDIT_LIST,
        Permission.EXPORT_REQUEST_LOGS,
        Permission.EXPORT_COOKIES,
        Permission.EXPORT_TAGS,
        Permission.EXPORT_DATA_LAYER,
        Permission.ARTIFACT_READ,
        Permission.ARTIFACT_GENERATE_SIGNED_URL,
    },
    Role.ADMIN: set(Permission),  # All permissions
}


@dataclass
class User:
    """User model for authentication and authorization."""
    id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    roles: Set[Role] = field(default_factory=set)
    permissions: Set[Permission] = field(default_factory=set)
    is_active: bool = True
    is_superuser: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

    def get_effective_permissions(self) -> Set[Permission]:
        """Get all effective permissions including role-based permissions."""
        effective_permissions = set(self.permissions)

        # Add permissions from roles
        for role in self.roles:
            if role in ROLE_PERMISSIONS:
                effective_permissions.update(ROLE_PERMISSIONS[role])

        # Superuser gets all permissions
        if self.is_superuser:
            effective_permissions.update(set(Permission))

        return effective_permissions

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        if not self.is_active:
            return False

        return permission in self.get_effective_permissions()

    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        if not self.is_active:
            return False

        effective_permissions = self.get_effective_permissions()
        return any(perm in effective_permissions for perm in permissions)

    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if user has all of the specified permissions."""
        if not self.is_active:
            return False

        effective_permissions = self.get_effective_permissions()
        return all(perm in effective_permissions for perm in permissions)

    def add_role(self, role: Role) -> None:
        """Add a role to the user."""
        self.roles.add(role)

    def remove_role(self, role: Role) -> None:
        """Remove a role from the user."""
        self.roles.discard(role)

    def add_permission(self, permission: Permission) -> None:
        """Add a direct permission to the user."""
        self.permissions.add(permission)

    def remove_permission(self, permission: Permission) -> None:
        """Remove a direct permission from the user."""
        self.permissions.discard(permission)


@dataclass
class AuthToken:
    """Authentication token model."""
    token: str
    user_id: str
    token_type: str = "bearer"  # bearer, api_key, etc.
    scopes: Set[str] = field(default_factory=set)
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """Check if token is valid (active and not expired)."""
        return self.is_active and not self.is_expired()

    def touch(self) -> None:
        """Update last_used timestamp."""
        self.last_used = datetime.utcnow()


class AuthTokenRequest(BaseModel):
    """Request model for creating authentication tokens."""
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1)
    scopes: Optional[List[str]] = Field(default=None, max_length=10)
    expires_in: Optional[int] = Field(default=3600, ge=60, le=2592000)  # 1 min to 30 days


class AuthTokenResponse(BaseModel):
    """Response model for authentication token creation."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    expires_at: datetime
    scopes: List[str] = Field(default_factory=list)


class UserInfo(BaseModel):
    """User information response model."""
    id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    is_active: bool = True
    is_superuser: bool = False
    created_at: datetime
    last_login: Optional[datetime] = None

    @classmethod
    def from_user(cls, user: User) -> "UserInfo":
        """Create UserInfo from User model."""
        return cls(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            roles=[role.value for role in user.roles],
            permissions=[perm.value for perm in user.get_effective_permissions()],
            is_active=user.is_active,
            is_superuser=user.is_superuser,
            created_at=user.created_at,
            last_login=user.last_login
        )


@dataclass
class AuthContext:
    """Authentication context for request processing."""
    user: Optional[User] = None
    token: Optional[AuthToken] = None
    is_authenticated: bool = False
    authentication_method: Optional[str] = None
    request_id: Optional[str] = None

    def has_permission(self, permission: Permission) -> bool:
        """Check if current context has permission."""
        if not self.is_authenticated or not self.user:
            return False
        return self.user.has_permission(permission)

    def require_permission(self, permission: Permission) -> None:
        """Require a specific permission, raise exception if not authorized."""
        if not self.has_permission(permission):
            from .exceptions import InsufficientPermissionsError
            raise InsufficientPermissionsError(
                f"Permission '{permission.value}' required",
                required_permission=permission.value
            )

    def require_any_permission(self, permissions: List[Permission]) -> None:
        """Require any of the specified permissions."""
        if not self.user or not self.user.has_any_permission(permissions):
            from .exceptions import InsufficientPermissionsError
            permission_names = [p.value for p in permissions]
            raise InsufficientPermissionsError(
                f"One of permissions {permission_names} required",
                required_permission=permission_names
            )