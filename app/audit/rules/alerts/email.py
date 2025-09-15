"""Email alert dispatcher implementation.

This module provides email-based alert dispatching with SMTP support,
HTML/text templating, retry logic, and comprehensive authentication options.
"""

import asyncio
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Union
from email.utils import formataddr

from pydantic import BaseModel, Field, EmailStr

from .base import (
    BaseAlertDispatcher,
    AlertContext,
    AlertDispatchResult,
    AlertStatus,
    register_dispatcher
)


class SMTPConfig(BaseModel):
    """SMTP server configuration."""
    
    # Server settings
    host: str = Field(description="SMTP server hostname")
    port: int = Field(
        default=587,
        description="SMTP server port (587 for TLS, 465 for SSL, 25 for plain)"
    )
    
    # Security settings
    use_tls: bool = Field(
        default=True,
        description="Use TLS encryption (STARTTLS)"
    )
    use_ssl: bool = Field(
        default=False,
        description="Use SSL encryption (implicit TLS)"
    )
    
    # Authentication
    username: Optional[str] = Field(
        default=None,
        description="SMTP authentication username"
    )
    password: Optional[str] = Field(
        default=None,
        description="SMTP authentication password"
    )
    
    # Connection settings
    timeout_seconds: float = Field(
        default=30.0,
        description="Connection timeout in seconds"
    )
    
    # Retry configuration
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )
    retry_delay_seconds: float = Field(
        default=30.0,
        description="Initial retry delay in seconds"
    )


class EmailConfig(BaseModel):
    """Email message configuration."""
    
    # Sender information
    from_email: EmailStr = Field(description="Sender email address")
    from_name: Optional[str] = Field(
        default=None,
        description="Sender display name"
    )
    
    # Recipients
    to_emails: List[EmailStr] = Field(description="Primary recipients")
    cc_emails: List[EmailStr] = Field(
        default_factory=list,
        description="CC recipients"
    )
    bcc_emails: List[EmailStr] = Field(
        default_factory=list,
        description="BCC recipients"
    )
    
    # Reply-to
    reply_to: Optional[EmailStr] = Field(
        default=None,
        description="Reply-to email address"
    )
    
    # Message settings
    subject_prefix: str = Field(
        default="[Tag Sentinel Alert]",
        description="Subject line prefix"
    )
    
    # Template settings
    html_template: Optional[str] = Field(
        default=None,
        description="HTML email template"
    )
    text_template: Optional[str] = Field(
        default=None,
        description="Plain text email template"
    )
    include_html: bool = Field(
        default=True,
        description="Include HTML version of email"
    )
    include_text: bool = Field(
        default=True,
        description="Include text version of email"
    )


@register_dispatcher("email")
class EmailAlertDispatcher(BaseAlertDispatcher):
    """Email-based alert dispatcher with SMTP support and retry logic."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate and parse email-specific configuration
        self.smtp_config = SMTPConfig(**config.get('smtp', {}))
        self.email_config = EmailConfig(**config.get('email', {}))
    
    async def dispatch(self, context: AlertContext) -> AlertDispatchResult:
        """Dispatch alert via email with retry logic.
        
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
        
        # Execute dispatch with retry logic (now async)
        return await self._dispatch_with_retry(alert_payload, context)
    
    
    async def _dispatch_with_retry(self, alert_payload, context: AlertContext) -> AlertDispatchResult:
        """Execute email dispatch with retry logic."""
        last_error = None
        retry_delay = self.smtp_config.retry_delay_seconds
        
        for attempt in range(self.smtp_config.max_retries + 1):
            try:
                start_time = time.time()
                
                # Execute email send
                result = await self._send_email(alert_payload, context, attempt + 1)
                
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
                last_error = f"Email dispatch exception: {str(e)}"
            
            # If not the last attempt, wait before retry
            if attempt < self.smtp_config.max_retries:
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5  # Moderate backoff
        
        # All attempts failed
        return self._create_dispatch_result(
            alert_id=alert_payload.alert_id,
            success=False,
            error_message=f"Failed after {self.smtp_config.max_retries + 1} attempts. Last error: {last_error}",
            attempt_number=self.smtp_config.max_retries + 1
        )
    
    async def _send_email(self, alert_payload, context: AlertContext, attempt_number: int) -> Dict[str, Any]:
        """Send email using SMTP.
        
        Args:
            alert_payload: Formatted alert payload
            context: Alert context
            attempt_number: Current attempt number
            
        Returns:
            Result dictionary with success status and details
        """
        try:
            # Create email message
            msg = self._create_email_message(alert_payload, context, attempt_number)
            
            # Connect to SMTP server and send
            if self.smtp_config.use_ssl:
                # Use implicit SSL/TLS
                smtp_class = smtplib.SMTP_SSL
            else:
                # Use plain SMTP, potentially with STARTTLS
                smtp_class = smtplib.SMTP
            
            with smtp_class(
                self.smtp_config.host,
                self.smtp_config.port,
                timeout=self.smtp_config.timeout_seconds
            ) as server:
                # Enable TLS if requested (and not already using SSL)
                if self.smtp_config.use_tls and not self.smtp_config.use_ssl:
                    server.starttls()
                
                # Authenticate if credentials provided
                if self.smtp_config.username and self.smtp_config.password:
                    server.login(self.smtp_config.username, self.smtp_config.password)
                
                # Send email
                recipients = self._get_all_recipients()
                server.send_message(msg, to_addrs=recipients)
                
                return {
                    "success": True,
                    "response_data": {
                        "recipients_count": len(recipients),
                        "message_id": msg.get("Message-ID"),
                        "attempt": attempt_number
                    }
                }
                
        except smtplib.SMTPAuthenticationError as e:
            return {
                "success": False,
                "error_message": f"SMTP authentication failed: {str(e)}"
            }
        except smtplib.SMTPRecipientsRefused as e:
            return {
                "success": False,
                "error_message": f"Recipients rejected: {str(e)}"
            }
        except smtplib.SMTPServerDisconnected as e:
            return {
                "success": False,
                "error_message": f"SMTP server disconnected: {str(e)}"
            }
        except smtplib.SMTPException as e:
            return {
                "success": False,
                "error_message": f"SMTP error: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Unexpected error: {str(e)}"
            }
    
    def _create_email_message(self, alert_payload, context: AlertContext, attempt_number: int) -> MIMEMultipart:
        """Create email message with HTML and/or text content.
        
        Args:
            alert_payload: Alert payload data
            context: Alert context
            attempt_number: Current attempt number
            
        Returns:
            Composed email message
        """
        # Create message container
        msg = MIMEMultipart('alternative')
        
        # Set headers
        sender = formataddr((
            self.email_config.from_name or "Tag Sentinel",
            self.email_config.from_email
        ))
        
        msg['From'] = sender
        msg['To'] = ', '.join(self.email_config.to_emails)
        
        if self.email_config.cc_emails:
            msg['Cc'] = ', '.join(self.email_config.cc_emails)
        
        if self.email_config.reply_to:
            msg['Reply-To'] = self.email_config.reply_to
        
        # Create subject with prefix
        subject = f"{self.email_config.subject_prefix} {alert_payload.title}"
        if attempt_number > 1:
            subject += f" (Retry {attempt_number})"
        msg['Subject'] = subject
        
        # Add message ID and other headers
        msg['X-Mailer'] = 'tag-sentinel-email-dispatcher/1.0'
        msg['X-Alert-ID'] = alert_payload.alert_id
        msg['X-Alert-Severity'] = alert_payload.severity.value
        if context.environment:
            msg['X-Environment'] = context.environment
        
        # Create text content
        if self.email_config.include_text:
            text_content = self._create_text_content(alert_payload, context)
            text_part = MIMEText(text_content, 'plain', 'utf-8')
            msg.attach(text_part)
        
        # Create HTML content
        if self.email_config.include_html:
            html_content = self._create_html_content(alert_payload, context)
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
        
        return msg
    
    def _create_text_content(self, alert_payload, context: AlertContext) -> str:
        """Create plain text email content.
        
        Args:
            alert_payload: Alert payload data
            context: Alert context
            
        Returns:
            Plain text email content
        """
        if self.email_config.text_template:
            # Use custom template
            return self._format_template(self.email_config.text_template, alert_payload, context)
        
        # Default text template
        lines = [
            f"Alert: {alert_payload.title}",
            "=" * 50,
            "",
            f"Severity: {alert_payload.severity.value.upper()}",
            f"Environment: {alert_payload.environment or 'Unknown'}",
            f"Timestamp: {alert_payload.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "Summary:",
            f"- Total Rules: {alert_payload.total_rules}",
            f"- Failed Rules: {alert_payload.failed_rules}",
            f"- Critical Failures: {alert_payload.critical_failures}",
            f"- Warning Failures: {alert_payload.warning_failures}",
            "",
            "Message:",
            alert_payload.message,
            "",
        ]
        
        # Add target URLs if available
        if alert_payload.target_urls:
            lines.append("Target URLs:")
            for url in alert_payload.target_urls:
                lines.append(f"- {url}")
            lines.append("")
        
        # Add failure details if available
        if alert_payload.failures:
            lines.append("Failure Details:")
            lines.append("-" * 20)
            for i, failure in enumerate(alert_payload.failures[:10], 1):  # Limit to 10
                lines.append(f"{i}. {failure.get('check_id', 'Unknown')}")
                lines.append(f"   Severity: {failure.get('severity', 'unknown')}")
                lines.append(f"   Message: {failure.get('message', 'No message')}")
                if failure.get('evidence_count', 0) > 0:
                    lines.append(f"   Evidence: {failure['evidence_count']} items")
                lines.append("")
            
            if len(alert_payload.failures) > 10:
                lines.append(f"... and {len(alert_payload.failures) - 10} more failures")
                lines.append("")
        
        lines.extend([
            "This is an automated alert from Tag Sentinel.",
            "Please check the system for detailed information."
        ])
        
        return '\n'.join(lines)
    
    def _create_html_content(self, alert_payload, context: AlertContext) -> str:
        """Create HTML email content.
        
        Args:
            alert_payload: Alert payload data
            context: Alert context
            
        Returns:
            HTML email content
        """
        if self.email_config.html_template:
            # Use custom template
            return self._format_template(self.email_config.html_template, alert_payload, context)
        
        # Default HTML template
        severity_colors = {
            'critical': '#dc3545',  # Red
            'high': '#fd7e14',      # Orange
            'medium': '#ffc107',    # Yellow
            'low': '#6c757d'        # Gray
        }
        
        severity_color = severity_colors.get(alert_payload.severity.value, '#6c757d')
        
        html_parts = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '    <meta charset="utf-8">',
            '    <title>Tag Sentinel Alert</title>',
            '    <style>',
            '        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }',
            '        .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            '        .header { background: #343a40; color: white; padding: 20px; text-align: center; }',
            '        .content { padding: 20px; }',
            '        .severity-badge { display: inline-block; padding: 4px 12px; border-radius: 16px; color: white; font-size: 12px; font-weight: bold; text-transform: uppercase; }',
            '        .summary-table { width: 100%; border-collapse: collapse; margin: 20px 0; }',
            '        .summary-table th, .summary-table td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #dee2e6; }',
            '        .summary-table th { background-color: #f8f9fa; font-weight: bold; }',
            '        .failure-item { background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; padding: 10px; margin: 10px 0; }',
            '        .failure-title { font-weight: bold; color: #856404; }',
            '        .failure-message { margin: 5px 0; color: #6c757d; }',
            '        .footer { background-color: #f8f9fa; padding: 15px 20px; font-size: 12px; color: #6c757d; text-align: center; }',
            '        .url-list { list-style: none; padding: 0; }',
            '        .url-list li { background-color: #e9ecef; padding: 8px; margin: 4px 0; border-radius: 4px; word-break: break-all; }',
            '    </style>',
            '</head>',
            '<body>',
            '    <div class="container">',
            '        <div class="header">',
            '            <h1>🚨 Tag Sentinel Alert</h1>',
            f'            <p>{alert_payload.title}</p>',
            '        </div>',
            '        <div class="content">',
            f'            <p><span class="severity-badge" style="background-color: {severity_color}">{alert_payload.severity.value}</span></p>',
            '',
            '            <table class="summary-table">',
            '                <tr><th>Property</th><th>Value</th></tr>',
            f'                <tr><td>Environment</td><td>{alert_payload.environment or "Unknown"}</td></tr>',
            f'                <tr><td>Timestamp</td><td>{alert_payload.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}</td></tr>',
            f'                <tr><td>Total Rules</td><td>{alert_payload.total_rules}</td></tr>',
            f'                <tr><td>Failed Rules</td><td>{alert_payload.failed_rules}</td></tr>',
            f'                <tr><td>Critical Failures</td><td>{alert_payload.critical_failures}</td></tr>',
            f'                <tr><td>Warning Failures</td><td>{alert_payload.warning_failures}</td></tr>',
            '            </table>',
            '',
            '            <h3>Message</h3>',
            f'            <p>{alert_payload.message}</p>',
        ]
        
        # Add target URLs
        if alert_payload.target_urls:
            html_parts.extend([
                '            <h3>Target URLs</h3>',
                '            <ul class="url-list">'
            ])
            for url in alert_payload.target_urls:
                html_parts.append(f'                <li><a href="{url}" target="_blank">{url}</a></li>')
            html_parts.append('            </ul>')
        
        # Add failure details
        if alert_payload.failures:
            html_parts.extend([
                '            <h3>Failure Details</h3>',
                '            <div class="failures">'
            ])
            
            for i, failure in enumerate(alert_payload.failures[:10], 1):  # Limit to 10
                html_parts.extend([
                    '                <div class="failure-item">',
                    f'                    <div class="failure-title">{i}. {failure.get("check_id", "Unknown")}</div>',
                    f'                    <div class="failure-message">{failure.get("message", "No message")}</div>',
                    f'                    <small>Severity: {failure.get("severity", "unknown")}</small>',
                ])
                
                if failure.get('evidence_count', 0) > 0:
                    html_parts.append(f'                    <small> | Evidence: {failure["evidence_count"]} items</small>')
                
                html_parts.append('                </div>')
            
            if len(alert_payload.failures) > 10:
                html_parts.append(f'                <p><em>... and {len(alert_payload.failures) - 10} more failures</em></p>')
            
            html_parts.append('            </div>')
        
        html_parts.extend([
            '        </div>',
            '        <div class="footer">',
            '            <p>This is an automated alert from Tag Sentinel.</p>',
            '            <p>Please check the system for detailed information.</p>',
            '        </div>',
            '    </div>',
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)
    
    def _format_template(self, template: str, alert_payload, context: AlertContext) -> str:
        """Format template with alert data.
        
        Args:
            template: Template string
            alert_payload: Alert payload data
            context: Alert context
            
        Returns:
            Formatted template
        """
        # Create template variables
        template_vars = {
            'title': alert_payload.title,
            'message': alert_payload.message,
            'severity': alert_payload.severity.value,
            'environment': alert_payload.environment or 'Unknown',
            'timestamp': alert_payload.timestamp.isoformat(),
            'total_rules': alert_payload.total_rules,
            'failed_rules': alert_payload.failed_rules,
            'critical_failures': alert_payload.critical_failures,
            'warning_failures': alert_payload.warning_failures,
            'target_urls': '\n'.join(alert_payload.target_urls) if alert_payload.target_urls else '',
            'alert_id': alert_payload.alert_id
        }
        
        # Simple variable substitution
        formatted = template
        for var_name, var_value in template_vars.items():
            placeholder = f'{{{var_name}}}'
            if placeholder in formatted:
                formatted = formatted.replace(placeholder, str(var_value))
        
        return formatted
    
    def _get_all_recipients(self) -> List[str]:
        """Get all email recipients (to, cc, bcc).
        
        Returns:
            List of all recipient email addresses
        """
        recipients = []
        recipients.extend(self.email_config.to_emails)
        recipients.extend(self.email_config.cc_emails)
        recipients.extend(self.email_config.bcc_emails)
        return recipients
    
    def validate_config(self) -> List[str]:
        """Validate email dispatcher configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate SMTP configuration
        if not self.smtp_config.host:
            errors.append("SMTP host is required")
        
        if not (1 <= self.smtp_config.port <= 65535):
            errors.append("SMTP port must be between 1 and 65535")
        
        if self.smtp_config.use_ssl and self.smtp_config.use_tls:
            errors.append("Cannot use both SSL and TLS - choose one")
        
        if self.smtp_config.max_retries < 0:
            errors.append("max_retries must be >= 0")
        
        if self.smtp_config.retry_delay_seconds <= 0:
            errors.append("retry_delay_seconds must be > 0")
        
        if self.smtp_config.timeout_seconds <= 0:
            errors.append("timeout_seconds must be > 0")
        
        # Validate email configuration
        if not self.email_config.to_emails:
            errors.append("At least one recipient email is required")
        
        if not self.email_config.include_html and not self.email_config.include_text:
            errors.append("Must include at least HTML or text content")
        
        return errors
    
    async def test_connectivity(self) -> Dict[str, Any]:
        """Test SMTP connectivity and configuration.
        
        Returns:
            Test result with connectivity status and details
        """
        try:
            # Test SMTP connection
            if self.smtp_config.use_ssl:
                smtp_class = smtplib.SMTP_SSL
            else:
                smtp_class = smtplib.SMTP
            
            with smtp_class(
                self.smtp_config.host,
                self.smtp_config.port,
                timeout=self.smtp_config.timeout_seconds
            ) as server:
                # Enable TLS if requested
                if self.smtp_config.use_tls and not self.smtp_config.use_ssl:
                    server.starttls()
                
                # Test authentication if configured
                if self.smtp_config.username and self.smtp_config.password:
                    server.login(self.smtp_config.username, self.smtp_config.password)
                
                return {
                    "success": True,
                    "message": "SMTP connection successful",
                    "smtp_host": self.smtp_config.host,
                    "smtp_port": self.smtp_config.port,
                    "auth_required": bool(self.smtp_config.username),
                    "tls_enabled": self.smtp_config.use_tls,
                    "ssl_enabled": self.smtp_config.use_ssl,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        except Exception as e:
            return {
                "success": False,
                "error_message": f"SMTP connectivity test failed: {str(e)}",
                "smtp_host": self.smtp_config.host,
                "smtp_port": self.smtp_config.port,
                "timestamp": datetime.utcnow().isoformat()
            }