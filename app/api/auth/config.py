"""Authentication configuration for Tag Sentinel API.

This module provides configuration management for authentication
providers, security settings, and auth-related parameters.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AuthProvider(str, Enum):
    """Available authentication provider types."""
    NONE = "none"  # No authentication (development only)
    API_KEY = "api_key"  # API key authentication
    JWT = "jwt"  # JWT token authentication
    OAUTH2 = "oauth2"  # OAuth2 authentication (future)


@dataclass
class AuthConfig:
    """Authentication configuration settings."""

    # Provider configuration
    provider: AuthProvider = AuthProvider.NONE
    secret_key: str = "dev-secret-key-change-in-production"
    algorithm: str = "HS256"

    # Token settings
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30
    api_key_expire_days: int = 365

    # Security settings
    password_min_length: int = 8
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    require_password_change_days: int = 90

    # Rate limiting
    auth_rate_limit_per_minute: int = 10
    token_rate_limit_per_hour: int = 100

    # API key settings
    api_key_prefix: str = "ts_"
    api_key_length: int = 32

    # JWT settings
    jwt_issuer: str = "tag-sentinel"
    jwt_audience: str = "tag-sentinel-api"

    # OAuth2 settings (future use)
    oauth2_providers: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # User management
    allow_user_registration: bool = False
    default_user_roles: List[str] = field(default_factory=lambda: ["viewer"])
    require_email_verification: bool = False

    # Session settings
    session_cookie_name: str = "tag_sentinel_session"
    session_cookie_secure: bool = True
    session_cookie_httponly: bool = True
    session_cookie_samesite: str = "strict"

    # Advanced settings
    enable_audit_logging: bool = True
    enable_brute_force_protection: bool = True
    enable_ip_whitelisting: bool = False
    allowed_ips: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "AuthConfig":
        """Create configuration from environment variables.

        Environment variables:
        - TAG_SENTINEL_AUTH_PROVIDER: Authentication provider type
        - TAG_SENTINEL_SECRET_KEY: Secret key for signing tokens
        - TAG_SENTINEL_JWT_ALGORITHM: JWT signing algorithm
        - TAG_SENTINEL_ACCESS_TOKEN_EXPIRE_MINUTES: Access token expiration
        - TAG_SENTINEL_API_KEY_EXPIRE_DAYS: API key expiration
        - TAG_SENTINEL_PASSWORD_MIN_LENGTH: Minimum password length
        - TAG_SENTINEL_MAX_LOGIN_ATTEMPTS: Maximum login attempts
        - TAG_SENTINEL_AUTH_RATE_LIMIT: Auth requests per minute
        - TAG_SENTINEL_ALLOW_USER_REGISTRATION: Allow user registration
        - TAG_SENTINEL_REQUIRE_EMAIL_VERIFICATION: Require email verification
        """
        provider_str = os.getenv("TAG_SENTINEL_AUTH_PROVIDER", "none").lower()
        try:
            provider = AuthProvider(provider_str)
        except ValueError:
            logger.warning(f"Invalid auth provider '{provider_str}', defaulting to none")
            provider = AuthProvider.NONE

        # Get secret key from environment or use default
        secret_key = os.getenv("TAG_SENTINEL_SECRET_KEY", "dev-secret-key-change-in-production")
        if secret_key == "dev-secret-key-change-in-production" and provider != AuthProvider.NONE:
            logger.warning("Using default secret key in production is not secure!")

        return cls(
            provider=provider,
            secret_key=secret_key,
            algorithm=os.getenv("TAG_SENTINEL_JWT_ALGORITHM", "HS256"),
            access_token_expire_minutes=int(os.getenv("TAG_SENTINEL_ACCESS_TOKEN_EXPIRE_MINUTES", "60")),
            api_key_expire_days=int(os.getenv("TAG_SENTINEL_API_KEY_EXPIRE_DAYS", "365")),
            password_min_length=int(os.getenv("TAG_SENTINEL_PASSWORD_MIN_LENGTH", "8")),
            max_login_attempts=int(os.getenv("TAG_SENTINEL_MAX_LOGIN_ATTEMPTS", "5")),
            auth_rate_limit_per_minute=int(os.getenv("TAG_SENTINEL_AUTH_RATE_LIMIT", "10")),
            allow_user_registration=os.getenv("TAG_SENTINEL_ALLOW_USER_REGISTRATION", "false").lower() == "true",
            require_email_verification=os.getenv("TAG_SENTINEL_REQUIRE_EMAIL_VERIFICATION", "false").lower() == "true",
            session_cookie_secure=os.getenv("TAG_SENTINEL_SESSION_COOKIE_SECURE", "true").lower() == "true",
            enable_brute_force_protection=os.getenv("TAG_SENTINEL_ENABLE_BRUTE_FORCE_PROTECTION", "true").lower() == "true",
            enable_ip_whitelisting=os.getenv("TAG_SENTINEL_ENABLE_IP_WHITELISTING", "false").lower() == "true",
            allowed_ips=os.getenv("TAG_SENTINEL_ALLOWED_IPS", "").split(",") if os.getenv("TAG_SENTINEL_ALLOWED_IPS") else []
        )

    @classmethod
    def for_development(cls) -> "AuthConfig":
        """Create configuration optimized for development."""
        return cls(
            provider=AuthProvider.NONE,
            secret_key="dev-secret-key",
            access_token_expire_minutes=480,  # 8 hours
            session_cookie_secure=False,  # Allow HTTP in development
            enable_brute_force_protection=False,  # Disable for easier testing
            auth_rate_limit_per_minute=100,  # Higher limit for development
        )

    @classmethod
    def for_testing(cls) -> "AuthConfig":
        """Create configuration optimized for testing."""
        return cls(
            provider=AuthProvider.NONE,
            secret_key="test-secret-key",
            access_token_expire_minutes=60,
            session_cookie_secure=False,
            enable_brute_force_protection=False,
            auth_rate_limit_per_minute=1000,  # Very high limit for tests
            enable_audit_logging=False,  # Disable audit logging in tests
        )

    @classmethod
    def for_production(
        cls,
        secret_key: str,
        provider: AuthProvider = AuthProvider.JWT
    ) -> "AuthConfig":
        """Create configuration for production deployment.

        Args:
            secret_key: Strong secret key for token signing
            provider: Authentication provider to use
        """
        if len(secret_key) < 32:
            raise ValueError("Production secret key must be at least 32 characters")

        return cls(
            provider=provider,
            secret_key=secret_key,
            access_token_expire_minutes=60,
            session_cookie_secure=True,
            enable_brute_force_protection=True,
            auth_rate_limit_per_minute=10,
            enable_audit_logging=True,
            require_email_verification=True,
            password_min_length=12,  # Stronger password requirement
        )

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors.

        Returns:
            List of validation messages
        """
        issues = []

        # Security validations
        if self.provider != AuthProvider.NONE:
            if self.secret_key == "dev-secret-key-change-in-production":
                issues.append("Default secret key should be changed in production")

            if len(self.secret_key) < 32:
                issues.append("Secret key should be at least 32 characters for security")

            if self.password_min_length < 8:
                issues.append("Password minimum length should be at least 8 characters")

        # Token expiration validations
        if self.access_token_expire_minutes > 1440:  # 24 hours
            issues.append("Access token expiration longer than 24 hours may be a security risk")

        # Rate limiting validations
        if self.auth_rate_limit_per_minute > 100:
            issues.append("High authentication rate limit may be vulnerable to brute force attacks")

        # Cookie security validations
        if not self.session_cookie_secure:
            issues.append("Session cookies should be secure in production")

        if not self.session_cookie_httponly:
            issues.append("Session cookies should be HTTP-only for security")

        return issues

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging/debugging.

        Returns:
            Dictionary with non-sensitive configuration values
        """
        return {
            "provider": self.provider.value,
            "algorithm": self.algorithm,
            "access_token_expire_minutes": self.access_token_expire_minutes,
            "password_min_length": self.password_min_length,
            "max_login_attempts": self.max_login_attempts,
            "auth_rate_limit_per_minute": self.auth_rate_limit_per_minute,
            "allow_user_registration": self.allow_user_registration,
            "require_email_verification": self.require_email_verification,
            "enable_brute_force_protection": self.enable_brute_force_protection,
            "enable_ip_whitelisting": self.enable_ip_whitelisting,
            "secret_key_length": len(self.secret_key),
            "has_oauth2_providers": len(self.oauth2_providers) > 0,
        }