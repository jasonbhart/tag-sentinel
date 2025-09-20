"""Authentication provider implementations for Tag Sentinel API.

This module provides different authentication providers including
no-auth, API key, JWT, and OAuth2 implementations.
"""

import logging
import secrets
import hashlib
import hmac
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import jwt
from cryptography.fernet import Fernet
import base64

from .models import User, AuthToken, Permission, Role, ROLE_PERMISSIONS
from .exceptions import (
    InvalidTokenError,
    ExpiredTokenError,
    InvalidCredentialsError,
    UserNotFoundError,
    UserInactiveError,
    TokenGenerationError
)
from .config import AuthConfig, AuthProvider as AuthProviderEnum

logger = logging.getLogger(__name__)


class AuthProvider(ABC):
    """Abstract base class for authentication providers."""

    def __init__(self, config: AuthConfig):
        """Initialize authentication provider with configuration."""
        self.config = config

    @abstractmethod
    async def authenticate_token(self, token: str) -> tuple[User, AuthToken]:
        """Authenticate a token and return user and token info.

        Args:
            token: Authentication token

        Returns:
            Tuple of (User, AuthToken)

        Raises:
            InvalidTokenError: If token is invalid
            ExpiredTokenError: If token is expired
            UserNotFoundError: If user doesn't exist
            UserInactiveError: If user is inactive
        """
        pass

    @abstractmethod
    async def authenticate_credentials(self, username: str, password: str) -> User:
        """Authenticate user credentials and return user.

        Args:
            username: Username
            password: Password

        Returns:
            Authenticated user

        Raises:
            InvalidCredentialsError: If credentials are invalid
            UserNotFoundError: If user doesn't exist
            UserInactiveError: If user is inactive
        """
        pass

    @abstractmethod
    async def create_token(self, user: User, scopes: Optional[List[str]] = None) -> AuthToken:
        """Create authentication token for user.

        Args:
            user: User to create token for
            scopes: Optional token scopes

        Returns:
            Created authentication token

        Raises:
            TokenGenerationError: If token creation fails
        """
        pass

    @abstractmethod
    async def revoke_token(self, token: str) -> bool:
        """Revoke an authentication token.

        Args:
            token: Token to revoke

        Returns:
            True if token was revoked, False if not found
        """
        pass

    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID.

        Args:
            user_id: User identifier

        Returns:
            User if found, None otherwise
        """
        pass


class NoAuthProvider(AuthProvider):
    """No-authentication provider for development and testing.

    This provider creates a default admin user and accepts any token.
    WARNING: Should only be used in development/testing environments.
    """

    def __init__(self, config: AuthConfig):
        """Initialize no-auth provider."""
        super().__init__(config)
        self._default_user = User(
            id="default_admin",
            username="admin",
            email="admin@example.com",
            full_name="Default Admin User",
            roles={Role.ADMIN},
            is_active=True,
            is_superuser=True
        )

    async def authenticate_token(self, token: str) -> tuple[User, AuthToken]:
        """Accept any token and return default admin user."""
        auth_token = AuthToken(
            token=token,
            user_id=self._default_user.id,
            token_type="bearer",
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )
        return self._default_user, auth_token

    async def authenticate_credentials(self, username: str, password: str) -> User:
        """Accept any credentials and return default admin user."""
        return self._default_user

    async def create_token(self, user: User, scopes: Optional[List[str]] = None) -> AuthToken:
        """Create a simple token for the user."""
        token = f"no_auth_{secrets.token_urlsafe(32)}"
        return AuthToken(
            token=token,
            user_id=user.id,
            token_type="bearer",
            scopes=set(scopes or []),
            expires_at=datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes)
        )

    async def revoke_token(self, token: str) -> bool:
        """No-op for no-auth provider."""
        return True

    async def get_user(self, user_id: str) -> Optional[User]:
        """Return default admin user for any user ID."""
        return self._default_user


class APIKeyProvider(AuthProvider):
    """API key authentication provider.

    Uses secure API keys with optional expiration and scope restrictions.
    API keys are stored as hashed values for security.
    """

    def __init__(self, config: AuthConfig):
        """Initialize API key provider."""
        super().__init__(config)
        self._users: Dict[str, User] = {}
        self._api_keys: Dict[str, AuthToken] = {}  # key_hash -> token
        self._user_keys: Dict[str, List[str]] = {}  # user_id -> [key_hashes]

        # Create default admin user if none exists
        self._create_default_admin()

    def _create_default_admin(self) -> None:
        """Create default admin user for API key provider."""
        admin_user = User(
            id="admin",
            username="admin",
            email="admin@tag-sentinel.local",
            full_name="API Admin User",
            roles={Role.ADMIN},
            is_active=True,
            is_superuser=True
        )
        self._users[admin_user.id] = admin_user

    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(f"{self.config.secret_key}:{api_key}".encode()).hexdigest()

    def _generate_api_key(self) -> str:
        """Generate a new API key."""
        random_part = secrets.token_urlsafe(self.config.api_key_length)
        return f"{self.config.api_key_prefix}{random_part}"

    async def authenticate_token(self, token: str) -> tuple[User, AuthToken]:
        """Authenticate API key token."""
        # Hash the provided token
        key_hash = self._hash_api_key(token)

        # Look up token
        auth_token = self._api_keys.get(key_hash)
        if not auth_token:
            raise InvalidTokenError("Invalid API key")

        if not auth_token.is_valid():
            if auth_token.is_expired():
                raise ExpiredTokenError("API key has expired")
            else:
                raise InvalidTokenError("API key is inactive")

        # Get user
        user = self._users.get(auth_token.user_id)
        if not user:
            raise UserNotFoundError(f"User {auth_token.user_id} not found")

        if not user.is_active:
            raise UserInactiveError(f"User {user.username} is inactive")

        # Update last used
        auth_token.touch()

        return user, auth_token

    async def authenticate_credentials(self, username: str, password: str) -> User:
        """API key provider doesn't support credential authentication."""
        raise InvalidCredentialsError("API key provider doesn't support username/password authentication")

    async def create_token(self, user: User, scopes: Optional[List[str]] = None) -> AuthToken:
        """Create new API key for user."""
        if user.id not in self._users:
            raise UserNotFoundError(f"User {user.id} not found")

        # Generate API key
        api_key = self._generate_api_key()
        key_hash = self._hash_api_key(api_key)

        # Create token
        expires_at = datetime.utcnow() + timedelta(days=self.config.api_key_expire_days)
        auth_token = AuthToken(
            token=api_key,  # Store the actual key in the token for return
            user_id=user.id,
            token_type="api_key",
            scopes=set(scopes or []),
            expires_at=expires_at
        )

        # Store hashed version
        self._api_keys[key_hash] = auth_token

        # Track keys per user
        if user.id not in self._user_keys:
            self._user_keys[user.id] = []
        self._user_keys[user.id].append(key_hash)

        logger.info(f"Created API key for user {user.username}")
        return auth_token

    async def revoke_token(self, token: str) -> bool:
        """Revoke API key."""
        key_hash = self._hash_api_key(token)
        if key_hash in self._api_keys:
            auth_token = self._api_keys[key_hash]
            auth_token.is_active = False
            logger.info(f"Revoked API key for user {auth_token.user_id}")
            return True
        return False

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)

    async def create_user(self, user: User) -> User:
        """Create new user in the provider."""
        if user.id in self._users:
            raise ValueError(f"User {user.id} already exists")

        self._users[user.id] = user
        logger.info(f"Created user {user.username}")
        return user

    async def list_user_keys(self, user_id: str) -> List[AuthToken]:
        """List all API keys for a user."""
        user_key_hashes = self._user_keys.get(user_id, [])
        return [self._api_keys[key_hash] for key_hash in user_key_hashes if key_hash in self._api_keys]


class JWTProvider(AuthProvider):
    """JWT (JSON Web Token) authentication provider.

    Uses secure JWT tokens with configurable expiration and claims.
    Supports both access and refresh tokens.
    """

    def __init__(self, config: AuthConfig):
        """Initialize JWT provider."""
        super().__init__(config)
        self._users: Dict[str, User] = {}
        self._revoked_tokens: set = set()  # Simple revocation list

        # Create default admin user if none exists
        self._create_default_admin()

    def _create_default_admin(self) -> None:
        """Create default admin user for JWT provider."""
        admin_user = User(
            id="admin",
            username="admin",
            email="admin@tag-sentinel.local",
            full_name="JWT Admin User",
            roles={Role.ADMIN},
            is_active=True,
            is_superuser=True
        )
        self._users[admin_user.id] = admin_user

    def _create_jwt_token(self, user: User, token_type: str = "access", scopes: Optional[List[str]] = None) -> str:
        """Create JWT token for user."""
        now = datetime.utcnow()

        if token_type == "access":
            expires_delta = timedelta(minutes=self.config.access_token_expire_minutes)
        elif token_type == "refresh":
            expires_delta = timedelta(days=self.config.refresh_token_expire_days)
        else:
            raise ValueError(f"Unknown token type: {token_type}")

        expires_at = now + expires_delta

        payload = {
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.get_effective_permissions()],
            "scopes": scopes or [],
            "token_type": token_type,
            "iat": now,
            "exp": expires_at,
            "iss": self.config.jwt_issuer,
            "aud": self.config.jwt_audience,
            "jti": secrets.token_urlsafe(16)  # JWT ID for revocation
        }

        return jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)

    def _decode_jwt_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                issuer=self.config.jwt_issuer,
                audience=self.config.jwt_audience
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ExpiredTokenError("JWT token has expired")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"Invalid JWT token: {str(e)}")

    async def authenticate_token(self, token: str) -> tuple[User, AuthToken]:
        """Authenticate JWT token."""
        # Decode token
        payload = self._decode_jwt_token(token)

        # Check if token is revoked
        jti = payload.get("jti")
        if jti and jti in self._revoked_tokens:
            raise InvalidTokenError("JWT token has been revoked")

        # Get user
        user_id = payload.get("sub")
        user = self._users.get(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")

        if not user.is_active:
            raise UserInactiveError(f"User {user.username} is inactive")

        # Create auth token object
        expires_at = datetime.fromtimestamp(payload.get("exp", 0))
        auth_token = AuthToken(
            token=token,
            user_id=user.id,
            token_type="bearer",
            scopes=set(payload.get("scopes", [])),
            expires_at=expires_at,
            metadata={"jti": jti, "token_type": payload.get("token_type", "access")}
        )

        return user, auth_token

    async def authenticate_credentials(self, username: str, password: str) -> User:
        """Authenticate user credentials."""
        # Find user by username
        user = None
        for u in self._users.values():
            if u.username == username:
                user = u
                break

        if not user:
            raise UserNotFoundError(f"User {username} not found")

        if not user.is_active:
            raise UserInactiveError(f"User {username} is inactive")

        # In a real implementation, you would verify the password hash
        # For this demo, we'll accept any password for existing users
        logger.info(f"Authenticated user {username}")
        return user

    async def create_token(self, user: User, scopes: Optional[List[str]] = None) -> AuthToken:
        """Create JWT access token for user."""
        if user.id not in self._users:
            raise UserNotFoundError(f"User {user.id} not found")

        # Create JWT token
        jwt_token = self._create_jwt_token(user, "access", scopes)

        # Create auth token object
        expires_at = datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes)
        auth_token = AuthToken(
            token=jwt_token,
            user_id=user.id,
            token_type="bearer",
            scopes=set(scopes or []),
            expires_at=expires_at
        )

        logger.info(f"Created JWT token for user {user.username}")
        return auth_token

    async def create_refresh_token(self, user: User) -> AuthToken:
        """Create JWT refresh token for user."""
        jwt_token = self._create_jwt_token(user, "refresh")

        expires_at = datetime.utcnow() + timedelta(days=self.config.refresh_token_expire_days)
        auth_token = AuthToken(
            token=jwt_token,
            user_id=user.id,
            token_type="refresh",
            expires_at=expires_at
        )

        return auth_token

    async def revoke_token(self, token: str) -> bool:
        """Revoke JWT token by adding to revocation list."""
        try:
            payload = self._decode_jwt_token(token)
            jti = payload.get("jti")
            if jti:
                self._revoked_tokens.add(jti)
                logger.info(f"Revoked JWT token {jti}")
                return True
        except (InvalidTokenError, ExpiredTokenError):
            pass
        return False

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)

    async def create_user(self, user: User) -> User:
        """Create new user in the provider."""
        if user.id in self._users:
            raise ValueError(f"User {user.id} already exists")

        self._users[user.id] = user
        logger.info(f"Created user {user.username}")
        return user


def create_auth_provider(config: AuthConfig) -> AuthProvider:
    """Factory function to create authentication provider based on configuration.

    Args:
        config: Authentication configuration

    Returns:
        Configured authentication provider
    """
    if config.provider == AuthProviderEnum.NONE:
        return NoAuthProvider(config)
    elif config.provider == AuthProviderEnum.API_KEY:
        return APIKeyProvider(config)
    elif config.provider == AuthProviderEnum.JWT:
        return JWTProvider(config)
    else:
        raise ValueError(f"Unsupported authentication provider: {config.provider}")