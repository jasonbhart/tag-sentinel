"""Response and general optimization features for Tag Sentinel API.

This module provides response compression, static file optimization,
and other performance enhancements for production deployment.
"""

import logging
import time
import gzip
import brotli
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
import os

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for response compression."""
    enable_gzip: bool = True
    enable_brotli: bool = True
    gzip_level: int = 6
    brotli_quality: int = 4
    min_size: int = 500  # Minimum response size to compress
    max_size: int = 10 * 1024 * 1024  # Maximum response size to compress (10MB)

    # Content types to compress
    compressible_types: Set[str] = None

    def __post_init__(self):
        if self.compressible_types is None:
            self.compressible_types = {
                "application/json",
                "application/javascript",
                "application/xml",
                "text/plain",
                "text/html",
                "text/css",
                "text/javascript",
                "text/xml",
                "application/x-javascript",
                "application/xhtml+xml",
                "application/rss+xml",
                "application/atom+xml",
                "image/svg+xml"
            }


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware for compressing HTTP responses."""

    def __init__(self, app, config: Optional[CompressionConfig] = None):
        """Initialize compression middleware.

        Args:
            app: FastAPI application
            config: Compression configuration
        """
        super().__init__(app)
        self.config = config or CompressionConfig()

        logger.info("CompressionMiddleware initialized")

    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply compression to responses."""
        response = await call_next(request)

        # Check if response should be compressed
        if not self._should_compress(request, response):
            return response

        # Get accepted encodings from client
        accept_encoding = request.headers.get("accept-encoding", "").lower()

        # Try Brotli first (better compression)
        if self.config.enable_brotli and "br" in accept_encoding:
            return await self._compress_brotli(response)

        # Fall back to Gzip
        if self.config.enable_gzip and "gzip" in accept_encoding:
            return await self._compress_gzip(response)

        return response

    def _should_compress(self, request: Request, response: Response) -> bool:
        """Check if response should be compressed."""
        # Check if already compressed
        if "content-encoding" in response.headers:
            return False

        # Check content type
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type not in self.config.compressible_types:
            return False

        # Check content length
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length < self.config.min_size or length > self.config.max_size:
                    return False
            except ValueError:
                pass

        # Check if response has body
        if not hasattr(response, 'body') or not response.body:
            return False

        return True

    async def _compress_gzip(self, response: Response) -> Response:
        """Compress response with Gzip."""
        try:
            if hasattr(response, 'body') and response.body:
                compressed_body = gzip.compress(
                    response.body,
                    compresslevel=self.config.gzip_level
                )

                # Only use compression if it actually reduces size
                if len(compressed_body) < len(response.body):
                    response.body = compressed_body
                    response.headers["content-encoding"] = "gzip"
                    response.headers["content-length"] = str(len(compressed_body))
                    response.headers["vary"] = "Accept-Encoding"

        except Exception as e:
            logger.error(f"Error compressing response with gzip: {e}")

        return response

    async def _compress_brotli(self, response: Response) -> Response:
        """Compress response with Brotli."""
        try:
            if hasattr(response, 'body') and response.body:
                compressed_body = brotli.compress(
                    response.body,
                    quality=self.config.brotli_quality
                )

                # Only use compression if it actually reduces size
                if len(compressed_body) < len(response.body):
                    response.body = compressed_body
                    response.headers["content-encoding"] = "br"
                    response.headers["content-length"] = str(len(compressed_body))
                    response.headers["vary"] = "Accept-Encoding"

        except Exception as e:
            logger.error(f"Error compressing response with brotli: {e}")

        return response


@dataclass
class StaticFileConfig:
    """Configuration for static file optimization."""
    enable_caching: bool = True
    cache_max_age: int = 86400  # 1 day
    enable_etag: bool = True
    enable_last_modified: bool = True
    enable_compression: bool = True

    # File extensions to optimize
    optimizable_extensions: Set[str] = None

    def __post_init__(self):
        if self.optimizable_extensions is None:
            self.optimizable_extensions = {
                ".js", ".css", ".html", ".json", ".xml",
                ".svg", ".ico", ".woff", ".woff2", ".ttf"
            }


class StaticFileOptimizer:
    """Optimizer for static file serving."""

    def __init__(self, config: Optional[StaticFileConfig] = None):
        """Initialize static file optimizer.

        Args:
            config: Static file configuration
        """
        self.config = config or StaticFileConfig()
        self._file_cache: Dict[str, Dict[str, Any]] = {}

    def optimize_response(self, request: Request, file_path: str) -> Optional[Dict[str, str]]:
        """Optimize static file response headers.

        Args:
            request: HTTP request
            file_path: Path to static file

        Returns:
            Dictionary of headers to add to response
        """
        if not os.path.exists(file_path):
            return None

        headers = {}

        # Get file stats
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        file_mtime = file_stats.st_mtime

        # Check if file extension should be optimized
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.config.optimizable_extensions:
            return headers

        # Add caching headers
        if self.config.enable_caching:
            headers["cache-control"] = f"public, max-age={self.config.cache_max_age}"

        # Add ETag
        if self.config.enable_etag:
            etag = self._generate_etag(file_path, file_mtime, file_size)
            headers["etag"] = etag

            # Check if client has current version
            client_etag = request.headers.get("if-none-match")
            if client_etag == etag:
                headers["_status_code"] = 304  # Not Modified
                return headers

        # Add Last-Modified
        if self.config.enable_last_modified:
            last_modified = time.strftime(
                "%a, %d %b %Y %H:%M:%S GMT",
                time.gmtime(file_mtime)
            )
            headers["last-modified"] = last_modified

            # Check if client has current version
            if_modified_since = request.headers.get("if-modified-since")
            if if_modified_since == last_modified:
                headers["_status_code"] = 304  # Not Modified
                return headers

        return headers

    def _generate_etag(self, file_path: str, mtime: float, size: int) -> str:
        """Generate ETag for file."""
        import hashlib
        etag_data = f"{file_path}:{mtime}:{size}"
        etag_hash = hashlib.md5(etag_data.encode()).hexdigest()
        return f'"{etag_hash}"'


@dataclass
class ResponseOptimizationConfig:
    """Configuration for response optimization."""
    enable_compression: bool = True
    enable_static_optimization: bool = True
    enable_response_headers: bool = True
    remove_server_header: bool = True
    add_security_headers: bool = True


class ResponseOptimizer:
    """Main response optimizer that combines multiple optimization techniques."""

    def __init__(self, config: Optional[ResponseOptimizationConfig] = None):
        """Initialize response optimizer.

        Args:
            config: Response optimization configuration
        """
        self.config = config or ResponseOptimizationConfig()

        # Initialize sub-optimizers
        if self.config.enable_compression:
            self.compression_config = CompressionConfig()

        if self.config.enable_static_optimization:
            self.static_optimizer = StaticFileOptimizer()

        logger.info("ResponseOptimizer initialized")

    def optimize_headers(self, response: Response) -> None:
        """Add optimization headers to response.

        Args:
            response: Response to optimize
        """
        if not self.config.enable_response_headers:
            return

        # Remove server header for security
        if self.config.remove_server_header:
            response.headers.pop("server", None)

        # Add security headers
        if self.config.add_security_headers:
            self._add_security_headers(response)

        # Add performance hints
        self._add_performance_headers(response)

    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }

        for header, value in security_headers.items():
            if header not in response.headers:
                response.headers[header] = value

    def _add_performance_headers(self, response: Response) -> None:
        """Add performance-related headers to response."""
        # Add vary header for compression
        if "content-encoding" in response.headers:
            vary_header = response.headers.get("vary", "")
            if "Accept-Encoding" not in vary_header:
                if vary_header:
                    response.headers["vary"] = f"{vary_header}, Accept-Encoding"
                else:
                    response.headers["vary"] = "Accept-Encoding"


class ResponseOptimizationMiddleware(BaseHTTPMiddleware):
    """Middleware that applies multiple response optimizations."""

    def __init__(
        self,
        app,
        config: Optional[ResponseOptimizationConfig] = None,
        compression_config: Optional[CompressionConfig] = None,
        static_config: Optional[StaticFileConfig] = None
    ):
        """Initialize response optimization middleware.

        Args:
            app: FastAPI application
            config: Response optimization configuration
            compression_config: Compression configuration
            static_config: Static file configuration
        """
        super().__init__(app)
        self.config = config or ResponseOptimizationConfig()

        # Initialize components
        self.response_optimizer = ResponseOptimizer(self.config)

        if self.config.enable_compression:
            self.compression_middleware = CompressionMiddleware(
                app, compression_config or CompressionConfig()
            )

        logger.info("ResponseOptimizationMiddleware initialized")

    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply response optimizations."""
        # Get response
        response = await call_next(request)

        # Apply compression if enabled
        if self.config.enable_compression and hasattr(self, 'compression_middleware'):
            # Apply compression logic directly
            if self.compression_middleware._should_compress(request, response):
                accept_encoding = request.headers.get("accept-encoding", "").lower()

                if "br" in accept_encoding and self.compression_middleware.config.enable_brotli:
                    response = await self.compression_middleware._compress_brotli(response)
                elif "gzip" in accept_encoding and self.compression_middleware.config.enable_gzip:
                    response = await self.compression_middleware._compress_gzip(response)

        # Apply general optimizations
        self.response_optimizer.optimize_headers(response)

        return response


class DatabaseQueryOptimizer:
    """Optimizer for database queries."""

    def __init__(self):
        """Initialize database query optimizer."""
        self.query_cache: Dict[str, Any] = {}
        self.slow_query_threshold = 1.0  # seconds

    async def optimize_query(self, query_func, *args, **kwargs):
        """Optimize database query execution.

        Args:
            query_func: Database query function
            *args: Query arguments
            **kwargs: Query keyword arguments

        Returns:
            Query result
        """
        start_time = time.time()

        try:
            result = await query_func(*args, **kwargs)

            execution_time = time.time() - start_time

            # Log slow queries
            if execution_time > self.slow_query_threshold:
                logger.warning(f"Slow query detected: {query_func.__name__} took {execution_time:.2f}s")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query failed after {execution_time:.2f}s: {e}")
            raise


class MemoryOptimizer:
    """Memory usage optimization utilities."""

    @staticmethod
    def optimize_response_size(data: Any, max_size: int = 1024 * 1024) -> Any:
        """Optimize response data size.

        Args:
            data: Response data
            max_size: Maximum size in bytes

        Returns:
            Optimized data
        """
        try:
            import sys

            # Check current size
            current_size = sys.getsizeof(data)

            if current_size <= max_size:
                return data

            # If it's a list, truncate it
            if isinstance(data, list):
                # Estimate how many items we can keep
                if len(data) > 0:
                    item_size = sys.getsizeof(data[0])
                    max_items = max_size // item_size
                    return data[:max_items]

            # If it's a dict, remove some keys
            if isinstance(data, dict):
                total_size = 0
                optimized_data = {}

                for key, value in data.items():
                    item_size = sys.getsizeof(key) + sys.getsizeof(value)
                    if total_size + item_size <= max_size:
                        optimized_data[key] = value
                        total_size += item_size
                    else:
                        break

                return optimized_data

            return data

        except Exception as e:
            logger.error(f"Error optimizing response size: {e}")
            return data

    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """Get current memory usage statistics.

        Returns:
            Memory usage information
        """
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            return {
                "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                "percent": round(process.memory_percent(), 2),
                "available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2)
            }

        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}