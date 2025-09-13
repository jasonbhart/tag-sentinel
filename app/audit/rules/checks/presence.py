"""Presence and count validation checks.

This module implements fundamental rule checks for validating the presence,
absence, and count of various elements in audit data including network requests,
cookies, console messages, and tag events.
"""

import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from ...models.capture import RequestLog, CookieRecord, ConsoleLog
from ...detectors.base import TagEvent
from ..models import Severity
from .base import BaseCheck, CheckContext, CheckResult, register_check


@register_check("request_present")
class RequestPresentCheck(BaseCheck):
    """Check for presence of network requests matching specified criteria."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
    
    def get_supported_config_keys(self) -> List[str]:
        return [
            'url_pattern',
            'url_regex', 
            'domain',
            'method',
            'status_code',
            'min_count',
            'max_count',
            'resource_type',
            'request_headers',
            'response_headers'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute request presence check."""
        config = context.check_config
        
        # Get all requests to filter
        query = context.query.query()
        
        # Apply domain filter
        if 'domain' in config:
            query = query.where_regex('url', f'https?://{re.escape(config["domain"])}')
        
        # Apply URL pattern filter
        if 'url_pattern' in config:
            # Convert simple pattern to regex (support * wildcards)
            pattern = config['url_pattern'].replace('*', '.*')
            query = query.where_regex('url', pattern)
        
        # Apply URL regex filter
        if 'url_regex' in config:
            query = query.where_regex('url', config['url_regex'])
        
        # Apply method filter
        if 'method' in config:
            query = query.where('method', config['method'].upper())
        
        # Apply status code filter
        if 'status_code' in config:
            query = query.where('status_code', config['status_code'])
        
        # Apply resource type filter
        if 'resource_type' in config:
            query = query.where('resource_type', config['resource_type'])
        
        # Get filtered requests
        requests = query.requests()
        
        # Apply header filters (more complex, done post-query)
        if 'request_headers' in config:
            requests = self._filter_by_headers(requests, config['request_headers'], 'request')
        
        if 'response_headers' in config:
            requests = self._filter_by_headers(requests, config['response_headers'], 'response')
        
        found_count = len(requests)
        min_count = config.get('min_count', 1)
        max_count = config.get('max_count')
        
        # Determine if check passed
        passed = found_count >= min_count
        if max_count is not None:
            passed = passed and found_count <= max_count
        
        # Create result message
        if max_count is not None:
            expected_range = f"{min_count}-{max_count}"
            message = f"Found {found_count} requests (expected {expected_range})"
        else:
            message = f"Found {found_count} requests (expected ≥{min_count})"
        
        return self._create_result(
            context=context,
            passed=passed,
            message=message,
            found_count=found_count,
            expected_count=min_count,
            evidence=self._extract_evidence(requests)
        )
    
    def _filter_by_headers(
        self, 
        requests: List[RequestLog], 
        header_filters: Dict[str, str], 
        header_type: str
    ) -> List[RequestLog]:
        """Filter requests by header values."""
        filtered = []
        
        for request in requests:
            headers = request.request_headers if header_type == 'request' else request.response_headers
            match = True
            
            for header_name, expected_value in header_filters.items():
                header_value = headers.get(header_name, '')
                
                # Support regex matching
                if expected_value.startswith('regex:'):
                    pattern = expected_value[6:]  # Remove 'regex:' prefix
                    if not re.search(pattern, header_value):
                        match = False
                        break
                else:
                    # Exact match
                    if header_value != expected_value:
                        match = False
                        break
            
            if match:
                filtered.append(request)
        
        return filtered


@register_check("cookie_present")
class CookiePresentCheck(BaseCheck):
    """Check for presence of cookies matching specified criteria."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
    
    def get_supported_config_keys(self) -> List[str]:
        return [
            'name',
            'name_pattern',
            'name_regex',
            'domain',
            'domain_pattern',
            'path',
            'secure',
            'http_only',
            'same_site',
            'is_first_party',
            'is_session',
            'min_count',
            'max_count'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute cookie presence check."""
        config = context.check_config
        
        # Get all cookies to filter
        query = context.query.query()
        
        # Apply name filters
        if 'name' in config:
            query = query.where('name', config['name'])
        elif 'name_pattern' in config:
            pattern = config['name_pattern'].replace('*', '.*')
            query = query.where_regex('name', pattern)
        elif 'name_regex' in config:
            query = query.where_regex('name', config['name_regex'])
        
        # Apply domain filters
        if 'domain' in config:
            query = query.where('domain', config['domain'])
        elif 'domain_pattern' in config:
            pattern = config['domain_pattern'].replace('*', '.*')
            query = query.where_regex('domain', pattern)
        
        # Apply path filter
        if 'path' in config:
            query = query.where('path', config['path'])
        
        # Apply security attribute filters
        if 'secure' in config:
            query = query.where('secure', config['secure'])
        
        if 'http_only' in config:
            query = query.where('http_only', config['http_only'])
        
        if 'same_site' in config:
            query = query.where('same_site', config['same_site'])
        
        if 'is_first_party' in config:
            query = query.where('is_first_party', config['is_first_party'])
        
        if 'is_session' in config:
            query = query.where('is_session', config['is_session'])
        
        # Get filtered cookies
        cookies = query.cookies()
        
        found_count = len(cookies)
        min_count = config.get('min_count', 1)
        max_count = config.get('max_count')
        
        # Determine if check passed
        passed = found_count >= min_count
        if max_count is not None:
            passed = passed and found_count <= max_count
        
        # Create result message
        if max_count is not None:
            expected_range = f"{min_count}-{max_count}"
            message = f"Found {found_count} cookies (expected {expected_range})"
        else:
            message = f"Found {found_count} cookies (expected ≥{min_count})"
        
        return self._create_result(
            context=context,
            passed=passed,
            message=message,
            found_count=found_count,
            expected_count=min_count,
            evidence=self._extract_evidence(cookies)
        )


@register_check("tag_event_present")
class TagEventPresentCheck(BaseCheck):
    """Check for presence of tag events matching specified criteria."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
    
    def get_supported_config_keys(self) -> List[str]:
        return [
            'vendor',
            'event_type',
            'tag_id',
            'event_name',
            'min_count',
            'max_count',
            'parameters',
            'page_url_pattern'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute tag event presence check."""
        config = context.check_config
        
        # Get all events to filter
        query = context.query.query()
        
        # Apply vendor filter
        if 'vendor' in config:
            query = query.where('vendor', config['vendor'])
        
        # Apply event type filter
        if 'event_type' in config:
            query = query.where('event_type', config['event_type'])
        
        # Apply tag ID filter
        if 'tag_id' in config:
            query = query.where('tag_id', config['tag_id'])
        
        # Apply event name filter
        if 'event_name' in config:
            query = query.where('event_name', config['event_name'])
        
        # Get filtered events
        events = query.events()
        
        # Apply parameter filters (complex, done post-query)
        if 'parameters' in config:
            events = self._filter_by_parameters(events, config['parameters'])
        
        # Apply page URL filter
        if 'page_url_pattern' in config:
            pattern = config['page_url_pattern'].replace('*', '.*')
            events = [e for e in events if re.search(pattern, e.page_url or '')]
        
        found_count = len(events)
        min_count = config.get('min_count', 1)
        max_count = config.get('max_count')
        
        # Determine if check passed
        passed = found_count >= min_count
        if max_count is not None:
            passed = passed and found_count <= max_count
        
        # Create result message
        if max_count is not None:
            expected_range = f"{min_count}-{max_count}"
            message = f"Found {found_count} tag events (expected {expected_range})"
        else:
            message = f"Found {found_count} tag events (expected ≥{min_count})"
        
        return self._create_result(
            context=context,
            passed=passed,
            message=message,
            found_count=found_count,
            expected_count=min_count,
            evidence=self._extract_evidence(events)
        )
    
    def _filter_by_parameters(
        self, 
        events: List[TagEvent], 
        parameter_filters: Dict[str, Any]
    ) -> List[TagEvent]:
        """Filter events by parameter values."""
        filtered = []
        
        for event in events:
            match = True
            
            for param_name, expected_value in parameter_filters.items():
                event_value = event.parameters.get(param_name)
                
                # Handle different comparison types
                if isinstance(expected_value, dict) and 'regex' in expected_value:
                    # Regex match
                    if not event_value or not re.search(expected_value['regex'], str(event_value)):
                        match = False
                        break
                elif isinstance(expected_value, dict) and 'exists' in expected_value:
                    # Existence check
                    exists = event_value is not None
                    if exists != expected_value['exists']:
                        match = False
                        break
                else:
                    # Direct value comparison
                    if event_value != expected_value:
                        match = False
                        break
            
            if match:
                filtered.append(event)
        
        return filtered


@register_check("console_message_present")
class ConsoleMessagePresentCheck(BaseCheck):
    """Check for presence of console messages matching specified criteria."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
    
    def get_supported_config_keys(self) -> List[str]:
        return [
            'level',
            'text_pattern',
            'text_regex',
            'url_pattern',
            'min_count',
            'max_count'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute console message presence check."""
        config = context.check_config
        
        # Get all console logs from pages
        console_logs = []
        for page in context.indexes.pages.pages:
            console_logs.extend(page.console_logs)
        
        # Apply level filter
        if 'level' in config:
            console_logs = [log for log in console_logs if log.level == config['level']]
        
        # Apply text filters
        if 'text_pattern' in config:
            pattern = config['text_pattern'].replace('*', '.*')
            console_logs = [log for log in console_logs if re.search(pattern, log.text)]
        elif 'text_regex' in config:
            console_logs = [log for log in console_logs if re.search(config['text_regex'], log.text)]
        
        # Apply URL filter
        if 'url_pattern' in config:
            pattern = config['url_pattern'].replace('*', '.*')
            console_logs = [log for log in console_logs if log.url and re.search(pattern, log.url)]
        
        found_count = len(console_logs)
        min_count = config.get('min_count', 1)
        max_count = config.get('max_count')
        
        # Determine if check passed
        passed = found_count >= min_count
        if max_count is not None:
            passed = passed and found_count <= max_count
        
        # Create result message
        if max_count is not None:
            expected_range = f"{min_count}-{max_count}"
            message = f"Found {found_count} console messages (expected {expected_range})"
        else:
            message = f"Found {found_count} console messages (expected ≥{min_count})"
        
        return self._create_result(
            context=context,
            passed=passed,
            message=message,
            found_count=found_count,
            expected_count=min_count,
            evidence=self._extract_evidence(console_logs)
        )


@register_check("request_absent")
class RequestAbsentCheck(BaseCheck):
    """Check for absence of network requests matching specified criteria."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
    
    def get_supported_config_keys(self) -> List[str]:
        return [
            'url_pattern',
            'url_regex',
            'domain',
            'method',
            'status_code',
            'resource_type'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute request absence check."""
        # Use RequestPresentCheck to find matches, then invert the result
        present_check = RequestPresentCheck(
            check_id=f"{self.check_id}_internal",
            name="Internal presence check",
            description="Internal check for absence validation"
        )
        
        # Modify config to check for min_count = 0, max_count = 0
        modified_config = context.check_config.copy()
        modified_config['min_count'] = 0
        modified_config['max_count'] = 0
        
        modified_context = CheckContext(
            indexes=context.indexes,
            query=context.query,
            rule_id=context.rule_id,
            rule_config=context.rule_config,
            check_config=modified_config,
            environment=context.environment,
            target_urls=context.target_urls,
            debug=context.debug,
            timeout_ms=context.timeout_ms
        )
        
        result = present_check.execute(modified_context)
        
        # Invert the result for absence check
        return CheckResult(
            check_id=self.check_id,
            check_name=self.name,
            passed=result.found_count == 0,
            severity=self._determine_severity(context),
            message=f"Found {result.found_count} requests (expected 0 for absence check)",
            found_count=result.found_count,
            expected_count=0,
            evidence=result.evidence
        )