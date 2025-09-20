"""API protection middleware for Tag Sentinel API.

This module provides middleware for rate limiting, abuse detection,
and general API protection with configurable policies.
"""

import logging
import time
from typing import Dict, List, Optional, Callable, Any, Set
from ipaddress import ip_address, ip_network
import re

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .rate_limiter import RateLimiter, RateLimitConfig, RateLimitExceededError, RateLimitScope
from ..auth.models import AuthContext

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for API rate limiting."""

    def __init__(
        self,
        app,
        default_config: Optional[RateLimitConfig] = None,
        storage=None
    ):
        """Initialize rate limit middleware.

        Args:
            app: FastAPI application
            default_config: Default rate limiting configuration
            storage: Storage backend for rate limiting data
        """
        super().__init__(app)

        self.default_config = default_config or RateLimitConfig(
            requests_per_window=100,
            window_seconds=60,
            scope=RateLimitScope.IP
        )

        self.rate_limiter = RateLimiter(self.default_config, storage)

        # Path-specific configurations
        self.path_configs: Dict[str, RateLimitConfig] = {}

        # Paths that bypass rate limiting
        self.exempt_paths: Set[str] = {
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        }

        # Different limits for authenticated vs unauthenticated users
        self.authenticated_config: Optional[RateLimitConfig] = None

        logger.info("RateLimitMiddleware initialized")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting to requests."""
        try:
            # Check if path is exempt
            if self._is_exempt_path(request.url.path):
                return await call_next(request)

            # Get rate limit configuration for this request
            config = self._get_rate_limit_config(request)

            # Generate rate limit key
            key = self._generate_rate_limit_key(request, config)

            # Check rate limit
            result = await self.rate_limiter.check_rate_limit(key, config_override=config)

            # Add rate limit headers to request state for later use
            request.state.rate_limit_result = result

            if not result.allowed:
                # Rate limit exceeded
                return self._create_rate_limit_response(result, config)

            # Process request
            response = await call_next(request)

            # Add rate limit headers to response
            if config.add_headers:
                for header_name, header_value in result.to_headers(config.header_prefix).items():
                    response.headers[header_name] = header_value

            return response

        except Exception as e:
            logger.error(f"Error in rate limit middleware: {e}", exc_info=True)
            # Don't block requests on middleware errors
            return await call_next(request)

    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from rate limiting."""
        return path in self.exempt_paths or path.startswith("/static")

    def _get_rate_limit_config(self, request: Request) -> RateLimitConfig:
        """Get rate limit configuration for request."""
        # Check for path-specific config
        for path_pattern, config in self.path_configs.items():
            if re.match(path_pattern, request.url.path):
                return config

        # Check if user is authenticated for different limits
        auth_context = getattr(request.state, "auth", None)
        if (auth_context and
            isinstance(auth_context, AuthContext) and
            auth_context.is_authenticated and
            self.authenticated_config):
            return self.authenticated_config

        return self.default_config

    def _generate_rate_limit_key(self, request: Request, config: RateLimitConfig) -> str:
        """Generate rate limit key for request."""
        request_info = {
            "client_ip": self._get_client_ip(request),
            "endpoint": request.url.path,
            "method": request.method
        }

        # Add user info if authenticated
        auth_context = getattr(request.state, "auth", None)
        if auth_context and isinstance(auth_context, AuthContext) and auth_context.is_authenticated:
            request_info["user_id"] = auth_context.user.id if auth_context.user else "unknown"

            # Add API key info if available
            if auth_context.token and auth_context.token.token_type == "api_key":
                # Use a hash of the token for the key to avoid storing sensitive data
                import hashlib
                token_hash = hashlib.sha256(auth_context.token.token.encode()).hexdigest()[:16]
                request_info["api_key"] = token_hash

        return self.rate_limiter.generate_key(request_info, config)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in case of multiple proxies
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"

    def _create_rate_limit_response(self, result, config: RateLimitConfig) -> JSONResponse:
        """Create rate limit exceeded response."""
        content = {
            "error": "rate_limit_exceeded",
            "message": config.error_message,
            "details": {
                "limit": result.limit,
                "remaining": result.remaining,
                "reset_time": result.reset_time
            }
        }

        headers = result.to_headers(config.header_prefix)

        return JSONResponse(
            status_code=429,
            content=content,
            headers=headers
        )

    def add_path_config(self, path_pattern: str, config: RateLimitConfig) -> None:
        """Add rate limit configuration for specific path pattern."""
        self.path_configs[path_pattern] = config

    def add_exempt_path(self, path: str) -> None:
        """Add path to rate limiting exemptions."""
        self.exempt_paths.add(path)

    def set_authenticated_config(self, config: RateLimitConfig) -> None:
        """Set different rate limit config for authenticated users."""
        self.authenticated_config = config


class APIProtectionMiddleware(BaseHTTPMiddleware):
    """Comprehensive API protection middleware."""

    def __init__(
        self,
        app,
        enable_ip_whitelist: bool = False,
        allowed_ips: Optional[List[str]] = None,
        enable_user_agent_filtering: bool = True,
        blocked_user_agents: Optional[List[str]] = None,
        enable_abuse_detection: bool = True,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        enable_security_headers: bool = True
    ):
        """Initialize API protection middleware.

        Args:
            app: FastAPI application
            enable_ip_whitelist: Enable IP address whitelisting
            allowed_ips: List of allowed IP addresses/networks
            enable_user_agent_filtering: Enable user agent filtering
            blocked_user_agents: List of blocked user agent patterns
            enable_abuse_detection: Enable abuse detection
            max_request_size: Maximum request size in bytes
            enable_security_headers: Add security headers to responses
        """
        super().__init__(app)

        self.enable_ip_whitelist = enable_ip_whitelist
        self.allowed_ips = self._parse_ip_list(allowed_ips or [])

        self.enable_user_agent_filtering = enable_user_agent_filtering
        self.blocked_user_agents = [re.compile(pattern, re.IGNORECASE)
                                   for pattern in (blocked_user_agents or [])]

        self.enable_abuse_detection = enable_abuse_detection
        self.max_request_size = max_request_size
        self.enable_security_headers = enable_security_headers

        # Abuse detection state
        self._suspicious_ips: Dict[str, Dict[str, Any]] = {}

        # Default blocked user agents (bots, scanners, etc.)
        default_blocked_patterns = [
            r".*bot.*",
            r".*crawler.*",
            r".*spider.*",
            r".*scraper.*",
            r".*scanner.*",
            r"sqlmap",
            r"nikto",
            r"nmap",
            r"masscan"
        ]

        if not blocked_user_agents:
            self.blocked_user_agents.extend([
                re.compile(pattern, re.IGNORECASE) for pattern in default_blocked_patterns
            ])

        logger.info("APIProtectionMiddleware initialized")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply API protection to requests."""
        try:
            # Check request size
            if await self._check_request_size(request):
                return self._create_error_response(
                    "Request too large",
                    413,
                    "request_too_large"
                )

            # Check IP whitelist
            if self.enable_ip_whitelist and not self._check_ip_whitelist(request):
                logger.warning(f"Blocked request from non-whitelisted IP: {self._get_client_ip(request)}")
                return self._create_error_response(
                    "Access denied",
                    403,
                    "ip_not_whitelisted"
                )

            # Check user agent
            if self.enable_user_agent_filtering and not self._check_user_agent(request):
                logger.warning(f"Blocked request with suspicious user agent: {request.headers.get('user-agent', 'unknown')}")
                return self._create_error_response(
                    "Access denied",
                    403,
                    "user_agent_blocked"
                )

            # Abuse detection
            if self.enable_abuse_detection:
                abuse_result = await self._check_abuse_patterns(request)
                if abuse_result:
                    return abuse_result

            # Process request
            response = await call_next(request)

            # Add security headers
            if self.enable_security_headers:
                self._add_security_headers(response)

            return response

        except Exception as e:
            logger.error(f"Error in API protection middleware: {e}", exc_info=True)
            return await call_next(request)

    def _parse_ip_list(self, ip_list: List[str]) -> List:
        """Parse IP addresses and networks."""
        parsed_ips = []
        for ip_str in ip_list:
            try:
                if "/" in ip_str:
                    # Network notation
                    parsed_ips.append(ip_network(ip_str, strict=False))
                else:
                    # Single IP
                    parsed_ips.append(ip_address(ip_str))
            except ValueError as e:
                logger.warning(f"Invalid IP address/network '{ip_str}': {e}")
        return parsed_ips

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _check_ip_whitelist(self, request: Request) -> bool:
        """Check if client IP is in whitelist."""
        if not self.allowed_ips:
            return True

        client_ip_str = self._get_client_ip(request)
        try:
            client_ip = ip_address(client_ip_str)

            for allowed in self.allowed_ips:
                if hasattr(allowed, 'network_address'):
                    # Network
                    if client_ip in allowed:
                        return True
                else:
                    # Single IP
                    if client_ip == allowed:
                        return True

            return False
        except ValueError:
            # Invalid IP format
            return False

    def _check_user_agent(self, request: Request) -> bool:
        """Check if user agent is allowed."""
        user_agent = request.headers.get("user-agent", "").lower()

        if not user_agent:
            # Block requests without user agent
            return False

        for pattern in self.blocked_user_agents:
            if pattern.search(user_agent):
                return False

        return True

    async def _check_request_size(self, request: Request) -> bool:
        """Check if request size exceeds limit."""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                return size > self.max_request_size
            except ValueError:
                pass
        return False

    async def _check_abuse_patterns(self, request: Request) -> Optional[Response]:
        """Check for abuse patterns."""
        client_ip = self._get_client_ip(request)
        now = time.time()

        # Initialize tracking for new IPs
        if client_ip not in self._suspicious_ips:
            self._suspicious_ips[client_ip] = {
                "requests": [],
                "error_count": 0,
                "last_error": 0
            }

        ip_data = self._suspicious_ips[client_ip]

        # Clean old request timestamps (older than 1 minute)
        ip_data["requests"] = [t for t in ip_data["requests"] if now - t < 60]

        # Add current request
        ip_data["requests"].append(now)

        # Check for rapid requests (more than 60 requests per minute from same IP)
        if len(ip_data["requests"]) > 60:
            logger.warning(f"Blocking IP {client_ip} for rapid requests: {len(ip_data['requests'])} requests/minute")
            return self._create_error_response(
                "Too many requests",
                429,
                "abuse_detected"
            )

        # Check for suspicious patterns (can be extended)
        path = request.url.path

        # Block common attack patterns
        suspicious_patterns = [
            r"\.php$",  # PHP files on non-PHP service
            r"\.asp$",  # ASP files
            r"\.jsp$",  # JSP files
            r"/wp-admin",  # WordPress admin
            r"/admin\.php",  # Admin PHP files
            r"\.env$",  # Environment files
            r"/\.git",  # Git directories
            r"script.*>",  # XSS attempts in path
            r"union.*select",  # SQL injection attempts
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                logger.warning(f"Blocked suspicious request from {client_ip}: {path}")
                return self._create_error_response(
                    "Suspicious request blocked",
                    403,
                    "suspicious_pattern"
                )

        return None

    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
        }

        for header, value in security_headers.items():
            if header not in response.headers:
                response.headers[header] = value

    def _create_error_response(self, message: str, status_code: int, error_code: str) -> JSONResponse:
        """Create error response."""
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error_code,
                "message": message,
                "timestamp": time.time()
            }
        )