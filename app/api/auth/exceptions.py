"""Authentication and authorization exceptions for Tag Sentinel API.

This module defines custom exceptions for authentication and authorization
errors with proper error codes and details.
"""

from typing import Optional, Union, List


class AuthenticationError(Exception):
    """Base authentication error."""

    def __init__(
        self,
        message: str = "Authentication failed",
        error_code: str = "authentication_failed",
        details: Optional[dict] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class InvalidTokenError(AuthenticationError):
    """Raised when an authentication token is invalid."""

    def __init__(
        self,
        message: str = "Invalid authentication token",
        token_type: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="invalid_token",
            details={"token_type": token_type} if token_type else {}
        )


class ExpiredTokenError(AuthenticationError):
    """Raised when an authentication token is expired."""

    def __init__(
        self,
        message: str = "Authentication token has expired",
        expired_at: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="token_expired",
            details={"expired_at": expired_at} if expired_at else {}
        )


class InvalidCredentialsError(AuthenticationError):
    """Raised when user credentials are invalid."""

    def __init__(
        self,
        message: str = "Invalid username or password",
        username: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="invalid_credentials",
            details={"username": username} if username else {}
        )


class UserNotFoundError(AuthenticationError):
    """Raised when a user cannot be found."""

    def __init__(
        self,
        message: str = "User not found",
        username: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        details = {}
        if username:
            details["username"] = username
        if user_id:
            details["user_id"] = user_id

        super().__init__(
            message=message,
            error_code="user_not_found",
            details=details
        )


class UserInactiveError(AuthenticationError):
    """Raised when a user account is inactive."""

    def __init__(
        self,
        message: str = "User account is inactive",
        username: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="user_inactive",
            details={"username": username} if username else {}
        )


class AuthorizationError(Exception):
    """Base authorization error."""

    def __init__(
        self,
        message: str = "Authorization failed",
        error_code: str = "authorization_failed",
        details: Optional[dict] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class InsufficientPermissionsError(AuthorizationError):
    """Raised when user lacks required permissions."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_permission: Optional[Union[str, List[str]]] = None,
        user_permissions: Optional[List[str]] = None
    ):
        details = {}
        if required_permission:
            details["required_permission"] = required_permission
        if user_permissions:
            details["user_permissions"] = user_permissions

        super().__init__(
            message=message,
            error_code="insufficient_permissions",
            details=details
        )


class RoleRequiredError(AuthorizationError):
    """Raised when user lacks required role."""

    def __init__(
        self,
        message: str = "Required role not assigned",
        required_role: Optional[Union[str, List[str]]] = None,
        user_roles: Optional[List[str]] = None
    ):
        details = {}
        if required_role:
            details["required_role"] = required_role
        if user_roles:
            details["user_roles"] = user_roles

        super().__init__(
            message=message,
            error_code="role_required",
            details=details
        )


class TokenGenerationError(AuthenticationError):
    """Raised when token generation fails."""

    def __init__(
        self,
        message: str = "Failed to generate authentication token",
        reason: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="token_generation_failed",
            details={"reason": reason} if reason else {}
        )


class RateLimitExceededError(AuthenticationError):
    """Raised when authentication rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Authentication rate limit exceeded",
        retry_after: Optional[int] = None,
        limit_type: Optional[str] = None
    ):
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
        if limit_type:
            details["limit_type"] = limit_type

        super().__init__(
            message=message,
            error_code="rate_limit_exceeded",
            details=details
        )