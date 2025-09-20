"""Authentication middleware for Tag Sentinel API.

This module provides middleware for handling authentication across
all API requests with proper error handling and context management.
"""

import logging
from datetime import datetime
from typing import Optional, Callable, Any
from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.utils import get_authorization_scheme_param
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .models import User, AuthToken, AuthContext
from .providers import AuthProvider, create_auth_provider
from .config import AuthConfig
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    InvalidTokenError,
    ExpiredTokenError,
    UserInactiveError,
    RateLimitExceededError
)

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for handling authentication and authorization."""

    def __init__(
        self,
        app,
        config: Optional[AuthConfig] = None,
        provider: Optional[AuthProvider] = None
    ):
        """Initialize authentication middleware.

        Args:
            app: FastAPI application
            config: Authentication configuration
            provider: Authentication provider (created from config if None)
        """
        super().__init__(app)
        self.config = config or AuthConfig.from_env()
        self.provider = provider or create_auth_provider(self.config)

        # Paths that don't require authentication
        self.public_paths = {
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico"
        }

        # Paths that require authentication (can be empty for "auth all" behavior)
        self.protected_paths = set()

        logger.info(f"AuthenticationMiddleware initialized with provider: {type(self.provider).__name__}")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process authentication for each request."""
        try:
            # Create authentication context
            auth_context = AuthContext(request_id=getattr(request.state, "request_id", None))

            # Check if path requires authentication
            if self._is_public_path(request.url.path):
                # Public path - skip authentication
                request.state.auth = auth_context
                return await call_next(request)

            # Extract and validate authentication
            await self._authenticate_request(request, auth_context)

            # Store authentication context in request state
            request.state.auth = auth_context

            # Process request
            response = await call_next(request)

            # Add authentication headers if applicable
            self._add_auth_headers(response, auth_context)

            return response

        except AuthenticationError as e:
            return self._create_auth_error_response(e, 401)
        except AuthorizationError as e:
            return self._create_auth_error_response(e, 403)
        except RateLimitExceededError as e:
            return self._create_auth_error_response(e, 429)
        except Exception as e:
            logger.error(f"Unexpected error in authentication middleware: {e}", exc_info=True)
            return self._create_error_response(
                "Internal authentication error",
                500,
                "internal_auth_error"
            )

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public and doesn't require authentication."""
        # Exact match
        if path in self.public_paths:
            return True

        # Pattern matching for docs and static files
        if path.startswith("/docs") or path.startswith("/static"):
            return True

        # If no protected paths defined, all other paths require auth
        if not self.protected_paths:
            return False

        # Check if path matches protected patterns
        return not any(path.startswith(protected) for protected in self.protected_paths)

    async def _authenticate_request(self, request: Request, auth_context: AuthContext) -> None:
        """Authenticate the current request."""
        # Extract token from request
        token = self._extract_token(request)

        if not token:
            # No token provided for protected route
            raise InvalidTokenError("Authentication token required")

        try:
            # Authenticate token
            user, auth_token = await self.provider.authenticate_token(token)

            # Update authentication context
            auth_context.user = user
            auth_context.token = auth_token
            auth_context.is_authenticated = True
            auth_context.authentication_method = auth_token.token_type

            logger.debug(f"Authenticated user {user.username} with {auth_token.token_type}")

        except (InvalidTokenError, ExpiredTokenError, UserInactiveError) as e:
            logger.warning(f"Authentication failed: {e}")
            raise

    def _extract_token(self, request: Request) -> Optional[str]:
        """Extract authentication token from request."""
        # Try Authorization header first (Bearer token)
        authorization = request.headers.get("Authorization")
        if authorization:
            scheme, token = get_authorization_scheme_param(authorization)
            if scheme.lower() == "bearer":
                return token

        # Try API key from header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key

        # Try API key from query parameter (less secure, but convenient for some use cases)
        api_key = request.query_params.get("api_key")
        if api_key:
            return api_key

        return None

    def _add_auth_headers(self, response: Response, auth_context: AuthContext) -> None:
        """Add authentication-related headers to response."""
        if auth_context.is_authenticated and auth_context.user:
            response.headers["X-Authenticated-User"] = auth_context.user.username
            response.headers["X-Auth-Method"] = auth_context.authentication_method or "unknown"

    def _create_auth_error_response(self, error: Exception, status_code: int) -> JSONResponse:
        """Create authentication/authorization error response."""
        if hasattr(error, 'error_code'):
            error_code = error.error_code
            details = getattr(error, 'details', {})
        else:
            error_code = "authentication_failed"
            details = {}

        return self._create_error_response(str(error), status_code, error_code, details)

    def _create_error_response(
        self,
        message: str,
        status_code: int,
        error_code: str,
        details: Optional[dict] = None
    ) -> JSONResponse:
        """Create standardized error response."""
        content = {
            "error": error_code,
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }

        if details:
            content["details"] = details

        # Add WWW-Authenticate header for 401 responses
        headers = {}
        if status_code == 401:
            if self.config.provider.value == "jwt":
                headers["WWW-Authenticate"] = "Bearer"
            elif self.config.provider.value == "api_key":
                headers["WWW-Authenticate"] = "ApiKey"

        return JSONResponse(
            status_code=status_code,
            content=content,
            headers=headers
        )

    def add_public_path(self, path: str) -> None:
        """Add path to public paths that don't require authentication."""
        self.public_paths.add(path)

    def add_protected_path(self, path: str) -> None:
        """Add path to protected paths that require authentication."""
        self.protected_paths.add(path)

    def remove_public_path(self, path: str) -> None:
        """Remove path from public paths."""
        self.public_paths.discard(path)


class OptionalAuthenticationMiddleware(AuthenticationMiddleware):
    """Middleware that attempts authentication but doesn't require it.

    This middleware will populate authentication context if valid credentials
    are provided, but won't fail if no credentials are present.
    """

    async def _authenticate_request(self, request: Request, auth_context: AuthContext) -> None:
        """Attempt authentication without requiring it."""
        token = self._extract_token(request)

        if not token:
            # No token provided - that's OK for optional auth
            return

        try:
            # Try to authenticate token
            user, auth_token = await self.provider.authenticate_token(token)

            # Update authentication context
            auth_context.user = user
            auth_context.token = auth_token
            auth_context.is_authenticated = True
            auth_context.authentication_method = auth_token.token_type

            logger.debug(f"Optionally authenticated user {user.username}")

        except (InvalidTokenError, ExpiredTokenError, UserInactiveError) as e:
            # Authentication failed, but that's OK for optional auth
            logger.debug(f"Optional authentication failed: {e}")
            # Don't raise - just leave auth_context unauthenticated


# Security schemes for FastAPI dependency injection
bearer_security = HTTPBearer(auto_error=False)
api_key_security = HTTPBearer(auto_error=False)  # Can be customized for API key format


def create_auth_middleware(
    app,
    config: Optional[AuthConfig] = None,
    optional: bool = False
) -> AuthenticationMiddleware:
    """Factory function to create authentication middleware.

    Args:
        app: FastAPI application instance
        config: Authentication configuration
        optional: Whether authentication is optional

    Returns:
        Configured authentication middleware
    """
    if optional:
        return OptionalAuthenticationMiddleware(app, config)
    else:
        return AuthenticationMiddleware(app, config)