"""Webhook alert dispatcher implementation.

This module provides webhook-based alert dispatching with HMAC signing,
retry logic, and comprehensive error handling for reliable alert delivery.
"""

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from .base import (
    BaseAlertDispatcher,
    AlertContext,
    AlertDispatchResult,
    AlertStatus,
    register_dispatcher
)


class WebhookConfig(BaseModel):
    """Configuration for webhook alert dispatcher."""
    
    # Required webhook settings
    url: str = Field(description="Webhook endpoint URL")
    
    # Security settings
    secret: Optional[str] = Field(
        default=None,
        description="HMAC secret for payload signing"
    )
    signature_header: str = Field(
        default="X-Signature-SHA256",
        description="HTTP header name for HMAC signature"
    )
    
    # HTTP settings
    method: str = Field(
        default="POST",
        description="HTTP method (POST, PUT, PATCH)"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers"
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="Request timeout in seconds"
    )
    
    # Retry configuration
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )
    retry_delay_seconds: float = Field(
        default=60.0,
        description="Initial retry delay in seconds"
    )
    retry_backoff_multiplier: float = Field(
        default=2.0,
        description="Backoff multiplier for exponential retry"
    )
    retry_max_delay_seconds: float = Field(
        default=900.0,
        description="Maximum retry delay (15 minutes)"
    )
    
    # Response validation
    expected_status_codes: List[int] = Field(
        default_factory=lambda: [200, 201, 202, 204],
        description="HTTP status codes considered successful"
    )
    verify_ssl: bool = Field(
        default=True,
        description="Verify SSL certificates"
    )


class WebhookPayload(BaseModel):
    """Structured webhook payload."""
    
    # Webhook metadata
    webhook_version: str = Field(
        default="1.0",
        description="Webhook payload version"
    )
    event_type: str = Field(
        default="alert.triggered",
        description="Event type identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp"
    )
    
    # Alert data (from AlertPayload)
    alert: Dict[str, Any] = Field(
        description="Alert payload data"
    )
    
    # Additional context
    source: str = Field(
        default="tag-sentinel",
        description="Source system identifier"
    )
    delivery_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Delivery metadata"
    )


@register_dispatcher("webhook")
class WebhookAlertDispatcher(BaseAlertDispatcher):
    """Webhook-based alert dispatcher with HMAC signing and retry logic."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        # Support both config dict and keyword arguments
        if config is None:
            config = {}
        
        # Merge keyword arguments into config
        if kwargs:
            webhook_config = config.get('webhook', {})
            webhook_config.update(kwargs)
            config = {**config, 'webhook': webhook_config}
        
        super().__init__(config)
        
        # Validate and parse webhook-specific configuration
        self.webhook_config = WebhookConfig(**config.get('webhook', {}))
        
        # Initialize HTTP client with timeouts
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=self.webhook_config.timeout_seconds,
                connect=10.0  # Connection timeout
            ),
            verify=self.webhook_config.verify_ssl,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
    
    def dispatch(self, context: AlertContext) -> AlertDispatchResult:
        """Dispatch alert via webhook with retry logic.
        
        Args:
            context: Alert context with evaluation results
            
        Returns:
            Dispatch result with status and details
        """
        # Check if dispatch should proceed
        if not self.should_dispatch(context):
            return self._create_dispatch_result(
                alert_id=context.alert_id or "unknown",
                success=False,
                error_message="Dispatch filtered out by configuration"
            )
        
        # Format alert payload
        alert_payload = self.format_alert(context)
        
        # Execute dispatch with retry logic
        return asyncio.run(self._dispatch_with_retry(alert_payload))
    
    async def _dispatch_with_retry(self, alert_payload) -> AlertDispatchResult:
        """Execute webhook dispatch with exponential backoff retry."""
        last_error = None
        retry_delay = self.webhook_config.retry_delay_seconds
        
        for attempt in range(self.webhook_config.max_retries + 1):
            try:
                start_time = time.time()
                
                # Prepare webhook payload
                webhook_payload = WebhookPayload(
                    alert=alert_payload.model_dump(),
                    delivery_info={
                        "attempt": attempt + 1,
                        "max_attempts": self.webhook_config.max_retries + 1,
                        "dispatcher_type": "webhook"
                    }
                )
                
                # Execute HTTP request
                result = await self._execute_webhook_request(webhook_payload)
                
                # Calculate response time
                response_time_ms = (time.time() - start_time) * 1000
                
                if result["success"]:
                    return self._create_dispatch_result(
                        alert_id=alert_payload.alert_id,
                        success=True,
                        response_time_ms=response_time_ms,
                        response_data=result.get("response_data"),
                        attempt_number=attempt + 1
                    )
                else:
                    last_error = result.get("error_message", "Unknown error")
                    
            except Exception as e:
                last_error = f"Webhook dispatch exception: {str(e)}"
            
            # If not the last attempt, wait before retry
            if attempt < self.webhook_config.max_retries:
                await asyncio.sleep(retry_delay)
                
                # Exponential backoff with jitter
                retry_delay = min(
                    retry_delay * self.webhook_config.retry_backoff_multiplier,
                    self.webhook_config.retry_max_delay_seconds
                )
        
        # All attempts failed
        return self._create_dispatch_result(
            alert_id=alert_payload.alert_id,
            success=False,
            error_message=f"Failed after {self.webhook_config.max_retries + 1} attempts. Last error: {last_error}",
            attempt_number=self.webhook_config.max_retries + 1
        )
    
    async def _execute_webhook_request(self, webhook_payload: WebhookPayload) -> Dict[str, Any]:
        """Execute the webhook HTTP request.
        
        Args:
            webhook_payload: Formatted webhook payload
            
        Returns:
            Result dictionary with success status and details
        """
        try:
            # Serialize payload
            payload_json = webhook_payload.model_dump_json()
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "tag-sentinel-webhook/1.0",
                **self.webhook_config.headers
            }
            
            # Add HMAC signature if secret is configured
            if self.webhook_config.secret:
                signature = self._generate_hmac_signature(
                    payload_json.encode('utf-8'),
                    self.webhook_config.secret
                )
                headers[self.webhook_config.signature_header] = f"sha256={signature}"
            
            # Execute request
            response = await self.client.request(
                method=self.webhook_config.method,
                url=self.webhook_config.url,
                content=payload_json,
                headers=headers
            )
            
            # Check if response is successful
            if response.status_code in self.webhook_config.expected_status_codes:
                response_data = {}
                try:
                    if response.headers.get("content-type", "").startswith("application/json"):
                        response_data = response.json()
                except Exception:
                    # Response parsing failed, but request was successful
                    pass
                
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "response_data": response_data
                }
            else:
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error_message": f"HTTP {response.status_code}: {response.text[:200]}"
                }
                
        except httpx.RequestError as e:
            return {
                "success": False,
                "error_message": f"Request error: {str(e)}"
            }
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "status_code": e.response.status_code,
                "error_message": f"HTTP error {e.response.status_code}: {e.response.text[:200]}"
            }
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Unexpected error: {str(e)}"
            }
    
    def _generate_hmac_signature(self, payload: bytes, secret: str) -> str:
        """Generate HMAC-SHA256 signature for payload.
        
        Args:
            payload: Raw payload bytes
            secret: HMAC secret key
            
        Returns:
            Hex-encoded HMAC signature
        """
        signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        )
        return signature.hexdigest()
    
    def validate_config(self) -> List[str]:
        """Validate webhook dispatcher configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate webhook URL
        try:
            parsed_url = urlparse(self.webhook_config.url)
            if not parsed_url.scheme or not parsed_url.netloc:
                errors.append("Invalid webhook URL format")
            elif parsed_url.scheme not in ['http', 'https']:
                errors.append("Webhook URL must use HTTP or HTTPS protocol")
        except Exception as e:
            errors.append(f"Webhook URL validation error: {str(e)}")
        
        # Validate HTTP method
        valid_methods = ['POST', 'PUT', 'PATCH']
        if self.webhook_config.method.upper() not in valid_methods:
            errors.append(f"HTTP method must be one of: {', '.join(valid_methods)}")
        
        # Validate retry configuration
        if self.webhook_config.max_retries < 0:
            errors.append("max_retries must be >= 0")
        
        if self.webhook_config.retry_delay_seconds <= 0:
            errors.append("retry_delay_seconds must be > 0")
        
        if self.webhook_config.retry_backoff_multiplier <= 1.0:
            errors.append("retry_backoff_multiplier must be > 1.0")
        
        # Validate timeout
        if self.webhook_config.timeout_seconds <= 0:
            errors.append("timeout_seconds must be > 0")
        
        # Validate expected status codes
        if not self.webhook_config.expected_status_codes:
            errors.append("expected_status_codes cannot be empty")
        
        for code in self.webhook_config.expected_status_codes:
            if not (100 <= code <= 599):
                errors.append(f"Invalid HTTP status code: {code}")
        
        return errors
    
    async def test_connectivity(self) -> Dict[str, Any]:
        """Test webhook connectivity and configuration.
        
        Returns:
            Test result with connectivity status and details
        """
        try:
            # Create a test payload
            test_payload = WebhookPayload(
                event_type="connectivity.test",
                alert={
                    "alert_id": "test-alert",
                    "title": "Webhook Connectivity Test",
                    "message": "This is a test message to verify webhook connectivity",
                    "severity": "low"
                },
                delivery_info={
                    "test": True,
                    "dispatcher_type": "webhook"
                }
            )
            
            # Execute test request
            result = await self._execute_webhook_request(test_payload)
            
            return {
                "success": result["success"],
                "status_code": result.get("status_code"),
                "error_message": result.get("error_message"),
                "timestamp": datetime.utcnow().isoformat(),
                "webhook_url": self.webhook_config.url
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Connectivity test failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
                "webhook_url": self.webhook_config.url
            }
    
    def __del__(self):
        """Cleanup HTTP client on destruction."""
        if hasattr(self, 'client'):
            asyncio.create_task(self.client.aclose())