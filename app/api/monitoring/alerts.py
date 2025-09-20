"""Alert management system for Tag Sentinel API.

This module provides alerting capabilities for monitoring metrics
and health checks with multiple notification channels.
"""

import logging
import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(str, Enum):
    """Alert status values."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """An alert notification."""
    id: str
    name: str
    message: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    resolved_at: Optional[float] = None
    acknowledged_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "message": self.message,
            "severity": self.severity.value,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "annotations": self.annotations,
            "resolved_at": self.resolved_at,
            "acknowledged_at": self.acknowledged_at
        }


@dataclass
class AlertRule:
    """Rule for triggering alerts."""
    name: str
    condition: Callable[[], bool]
    message_template: str
    severity: AlertSeverity = AlertSeverity.MEDIUM
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    cooldown_seconds: int = 300  # 5 minutes
    auto_resolve: bool = True
    last_triggered: Optional[float] = None


@dataclass
class AlertConfig:
    """Configuration for alert management."""
    enable_alerts: bool = True
    default_severity: AlertSeverity = AlertSeverity.MEDIUM
    max_active_alerts: int = 1000
    alert_retention_hours: int = 24
    batch_size: int = 10
    notification_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 60


class AlertChannel(ABC):
    """Abstract base class for alert notification channels."""

    def __init__(self, name: str, enabled: bool = True):
        """Initialize alert channel.

        Args:
            name: Channel name
            enabled: Whether channel is enabled
        """
        self.name = name
        self.enabled = enabled

    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert notification.

        Args:
            alert: Alert to send

        Returns:
            True if successful
        """
        pass

    async def send_alerts(self, alerts: List[Alert]) -> Dict[str, bool]:
        """Send multiple alerts.

        Args:
            alerts: List of alerts to send

        Returns:
            Dictionary mapping alert IDs to success status
        """
        results = {}
        for alert in alerts:
            try:
                success = await self.send_alert(alert)
                results[alert.id] = success
            except Exception as e:
                logger.error(f"Error sending alert {alert.id} via {self.name}: {e}")
                results[alert.id] = False

        return results


class LoggingChannel(AlertChannel):
    """Alert channel that logs alerts."""

    def __init__(self, name: str = "logging", log_level: str = "WARNING"):
        """Initialize logging channel.

        Args:
            name: Channel name
            log_level: Logging level for alerts
        """
        super().__init__(name)
        self.log_level = getattr(logging, log_level.upper(), logging.WARNING)

    async def send_alert(self, alert: Alert) -> bool:
        """Log alert message."""
        try:
            log_message = f"ALERT [{alert.severity.upper()}] {alert.name}: {alert.message}"
            if alert.labels:
                log_message += f" | Labels: {alert.labels}"

            logger.log(self.log_level, log_message)
            return True

        except Exception as e:
            logger.error(f"Error logging alert: {e}")
            return False


class EmailChannel(AlertChannel):
    """Alert channel for email notifications."""

    def __init__(
        self,
        name: str = "email",
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_email: str = "alerts@tagtsentinel.com",
        to_emails: List[str] = None,
        use_tls: bool = True
    ):
        """Initialize email channel.

        Args:
            name: Channel name
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: From email address
            to_emails: List of recipient email addresses
            use_tls: Whether to use TLS
        """
        super().__init__(name)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails or []
        self.use_tls = use_tls

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not self.to_emails:
            logger.warning("No email recipients configured")
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.name}"

            # Email body
            body = f"""
Alert: {alert.name}
Severity: {alert.severity.upper()}
Status: {alert.status.upper()}
Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(alert.timestamp))}

Message:
{alert.message}

Labels:
{json.dumps(alert.labels, indent=2)}

Annotations:
{json.dumps(alert.annotations, indent=2)}
"""

            msg.attach(MIMEText(body, 'plain'))

            # Send email in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_smtp_email, msg)

            return True

        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False

    def _send_smtp_email(self, msg):
        """Send email using SMTP (blocking operation)."""
        server = smtplib.SMTP(self.smtp_host, self.smtp_port)
        try:
            if self.use_tls:
                server.starttls()

            if self.username and self.password:
                server.login(self.username, self.password)

            server.send_message(msg)

        finally:
            server.quit()


class WebhookChannel(AlertChannel):
    """Alert channel for webhook notifications."""

    def __init__(
        self,
        name: str,
        webhook_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ):
        """Initialize webhook channel.

        Args:
            name: Channel name
            webhook_url: Webhook URL
            headers: Optional HTTP headers
            timeout: Request timeout
        """
        super().__init__(name)
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}
        self.timeout = timeout

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            import aiohttp

            payload = {
                "alert": alert.to_dict(),
                "webhook_name": self.name,
                "timestamp": time.time()
            }

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers
                ) as response:
                    if 200 <= response.status < 300:
                        return True
                    else:
                        logger.error(f"Webhook returned status {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
            return False


class SlackChannel(AlertChannel):
    """Alert channel for Slack notifications."""

    def __init__(
        self,
        name: str = "slack",
        webhook_url: str = "",
        channel: str = "#alerts",
        username: str = "Tag Sentinel"
    ):
        """Initialize Slack channel.

        Args:
            name: Channel name
            webhook_url: Slack webhook URL
            channel: Slack channel
            username: Bot username
        """
        super().__init__(name)
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        if not self.webhook_url:
            logger.warning("No Slack webhook URL configured")
            return False

        try:
            import aiohttp

            # Map severity to colors
            color_map = {
                AlertSeverity.CRITICAL: "#FF0000",
                AlertSeverity.HIGH: "#FF8C00",
                AlertSeverity.MEDIUM: "#FFD700",
                AlertSeverity.LOW: "#00FF00",
                AlertSeverity.INFO: "#0000FF"
            }

            # Create Slack message
            payload = {
                "channel": self.channel,
                "username": self.username,
                "attachments": [{
                    "color": color_map.get(alert.severity, "#808080"),
                    "title": f"{alert.severity.upper()}: {alert.name}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Status",
                            "value": alert.status.upper(),
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(alert.timestamp)),
                            "short": True
                        }
                    ],
                    "footer": "Tag Sentinel Monitoring",
                    "ts": int(alert.timestamp)
                }]
            }

            # Add labels as fields if present
            if alert.labels:
                for key, value in alert.labels.items():
                    payload["attachments"][0]["fields"].append({
                        "title": key,
                        "value": value,
                        "short": True
                    })

            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False


class AlertManager:
    """Main alert manager for handling alerts and notifications."""

    def __init__(self, config: Optional[AlertConfig] = None):
        """Initialize alert manager.

        Args:
            config: Alert configuration
        """
        self.config = config or AlertConfig()
        self.rules: Dict[str, AlertRule] = {}
        self.channels: Dict[str, AlertChannel] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []

        # Background tasks
        self._evaluation_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        # Default logging channel
        self.add_channel(LoggingChannel())

        logger.info("AlertManager initialized")

    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule.

        Args:
            rule: Alert rule to add
        """
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, name: str) -> None:
        """Remove alert rule.

        Args:
            name: Rule name to remove
        """
        if name in self.rules:
            del self.rules[name]
            logger.info(f"Removed alert rule: {name}")

    def add_channel(self, channel: AlertChannel) -> None:
        """Add notification channel.

        Args:
            channel: Notification channel to add
        """
        self.channels[channel.name] = channel
        logger.info(f"Added alert channel: {channel.name}")

    def remove_channel(self, name: str) -> None:
        """Remove notification channel.

        Args:
            name: Channel name to remove
        """
        if name in self.channels:
            del self.channels[name]
            logger.info(f"Removed alert channel: {name}")

    async def fire_alert(
        self,
        name: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None
    ) -> Alert:
        """Fire an alert manually.

        Args:
            name: Alert name
            message: Alert message
            severity: Alert severity
            labels: Optional labels
            annotations: Optional annotations

        Returns:
            Created alert
        """
        alert_id = f"{name}_{int(time.time())}"

        alert = Alert(
            id=alert_id,
            name=name,
            message=message,
            severity=severity,
            labels=labels or {},
            annotations=annotations or {}
        )

        await self._process_alert(alert)
        return alert

    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert.

        Args:
            alert_id: Alert ID to resolve

        Returns:
            True if alert was resolved
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = time.time()

            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]

            logger.info(f"Resolved alert: {alert.name}")
            return True

        return False

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert.

        Args:
            alert_id: Alert ID to acknowledge

        Returns:
            True if alert was acknowledged
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = time.time()

            logger.info(f"Acknowledged alert: {alert.name}")
            return True

        return False

    async def _process_alert(self, alert: Alert) -> None:
        """Process and send alert through channels."""
        if not self.config.enable_alerts:
            return

        # Check if we've exceeded max active alerts
        if len(self.active_alerts) >= self.config.max_active_alerts:
            logger.warning("Maximum active alerts reached, dropping oldest alerts")
            # Remove oldest alerts
            oldest_alerts = sorted(self.active_alerts.values(), key=lambda a: a.timestamp)
            for old_alert in oldest_alerts[:10]:  # Remove 10 oldest
                self.alert_history.append(old_alert)
                del self.active_alerts[old_alert.id]

        # Add to active alerts
        self.active_alerts[alert.id] = alert

        # Send through all enabled channels
        enabled_channels = [ch for ch in self.channels.values() if ch.enabled]

        if enabled_channels:
            tasks = []
            for channel in enabled_channels:
                task = asyncio.create_task(self._send_alert_with_retry(channel, alert))
                tasks.append(task)

            # Wait for all notifications to complete
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Processed alert: {alert.name} ({alert.severity})")

    async def _send_alert_with_retry(self, channel: AlertChannel, alert: Alert) -> None:
        """Send alert with retry logic."""
        for attempt in range(self.config.retry_attempts):
            try:
                success = await asyncio.wait_for(
                    channel.send_alert(alert),
                    timeout=self.config.notification_timeout
                )

                if success:
                    logger.debug(f"Alert sent successfully via {channel.name}")
                    return

            except asyncio.TimeoutError:
                logger.warning(f"Timeout sending alert via {channel.name} (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Error sending alert via {channel.name} (attempt {attempt + 1}): {e}")

            # Wait before retry
            if attempt < self.config.retry_attempts - 1:
                await asyncio.sleep(self.config.retry_delay)

        logger.error(f"Failed to send alert via {channel.name} after {self.config.retry_attempts} attempts")

    async def evaluate_rules(self) -> None:
        """Evaluate all alert rules."""
        if not self.config.enable_alerts:
            return

        current_time = time.time()

        for rule in self.rules.values():
            try:
                # Check cooldown
                if (rule.last_triggered and
                    current_time - rule.last_triggered < rule.cooldown_seconds):
                    continue

                # Evaluate condition
                should_trigger = rule.condition()

                if should_trigger:
                    # Generate alert
                    alert_id = f"{rule.name}_{int(current_time)}"
                    message = rule.message_template

                    alert = Alert(
                        id=alert_id,
                        name=rule.name,
                        message=message,
                        severity=rule.severity,
                        labels=rule.labels.copy(),
                        annotations=rule.annotations.copy()
                    )

                    await self._process_alert(alert)
                    rule.last_triggered = current_time

                elif rule.auto_resolve:
                    # Check if we should auto-resolve existing alerts for this rule
                    alerts_to_resolve = [
                        alert for alert in self.active_alerts.values()
                        if alert.name == rule.name and alert.status == AlertStatus.ACTIVE
                    ]

                    for alert in alerts_to_resolve:
                        await self.resolve_alert(alert.id)

            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")

    def start_evaluation(self, interval: int = 60) -> None:
        """Start background alert rule evaluation.

        Args:
            interval: Evaluation interval in seconds
        """
        if self._evaluation_task is None:
            self._evaluation_task = asyncio.create_task(self._evaluation_loop(interval))
            logger.info(f"Started alert rule evaluation with {interval}s interval")

    async def _evaluation_loop(self, interval: int) -> None:
        """Background loop for evaluating alert rules."""
        while True:
            try:
                await self.evaluate_rules()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(interval)

    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: Optional[int] = None) -> List[Alert]:
        """Get alert history.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of historical alerts
        """
        history = sorted(self.alert_history, key=lambda a: a.timestamp, reverse=True)
        if limit:
            return history[:limit]
        return history

    async def cleanup_old_alerts(self) -> None:
        """Clean up old alerts from history."""
        cutoff_time = time.time() - (self.config.alert_retention_hours * 3600)
        initial_count = len(self.alert_history)

        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]

        cleaned_count = initial_count - len(self.alert_history)
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old alerts")

    async def close(self) -> None:
        """Close alert manager and clean up resources."""
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("AlertManager closed")


# Global alert manager instance
_global_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
    return _global_alert_manager


def set_alert_manager(manager: AlertManager) -> None:
    """Set global alert manager instance."""
    global _global_alert_manager
    _global_alert_manager = manager