"""Duplicate detection checks for requests, events, and other audit data.

This module implements sophisticated duplicate detection capabilities to identify
duplicate tag firings, redundant requests, and other implementation issues that
can indicate problems with analytics instrumentation.
"""

import hashlib
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse, parse_qs

from ...models.capture import RequestLog, CookieRecord
from ...detectors.base import TagEvent
from ..models import Severity
from .base import BaseCheck, CheckContext, CheckResult, register_check


class DuplicateGroup:
    """Represents a group of duplicate items."""
    
    def __init__(self, group_key: str, items: List[Any]):
        self.group_key = group_key
        self.items = items
        self.count = len(items)
        self.first_occurrence = min(item.timestamp for item in items if hasattr(item, 'timestamp'))
        self.last_occurrence = max(item.timestamp for item in items if hasattr(item, 'timestamp'))
        self.time_span = (self.last_occurrence - self.first_occurrence).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for evidence reporting."""
        return {
            'group_key': self.group_key,
            'duplicate_count': self.count,
            'first_occurrence': self.first_occurrence.isoformat(),
            'last_occurrence': self.last_occurrence.isoformat(),
            'time_span_seconds': self.time_span,
            'items': [self._item_to_dict(item) for item in self.items[:5]]  # Limit to first 5
        }
    
    def _item_to_dict(self, item: Any) -> Dict[str, Any]:
        """Convert item to dictionary representation."""
        if hasattr(item, 'dict'):
            return item.dict()
        elif hasattr(item, '__dict__'):
            return {k: str(v) for k, v in item.__dict__.items()}
        else:
            return {'item': str(item)}


@register_check("request_duplicates")
class RequestDuplicateCheck(BaseCheck):
    """Detect duplicate network requests based on configurable criteria."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
    
    def get_supported_config_keys(self) -> List[str]:
        return [
            'grouping_fields',
            'ignore_fields',
            'time_window_seconds',
            'min_duplicates',
            'max_allowed_duplicates',
            'url_normalize',
            'include_query_params',
            'include_headers',
            'domain_filter',
            'exclude_resource_types'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute request duplicate detection."""
        config = context.check_config
        
        # Get all requests
        requests = context.query.query().requests()
        
        # Apply domain filter if specified
        if 'domain_filter' in config:
            domain_pattern = config['domain_filter']
            requests = [r for r in requests if domain_pattern in urlparse(r.url).netloc]
        
        # Exclude resource types if specified
        if 'exclude_resource_types' in config:
            excluded_types = set(config['exclude_resource_types'])
            requests = [r for r in requests if r.resource_type not in excluded_types]
        
        # Group requests by similarity
        duplicate_groups = self._group_requests(requests, config)
        
        # Filter groups based on duplicate criteria
        min_duplicates = config.get('min_duplicates', 2)
        max_allowed = config.get('max_allowed_duplicates', 0)
        
        significant_groups = [
            group for group in duplicate_groups 
            if group.count >= min_duplicates
        ]
        
        # Check if duplicates exceed allowed threshold
        total_duplicates = sum(group.count - 1 for group in significant_groups)  # -1 because first occurrence isn't a duplicate
        passed = total_duplicates <= max_allowed
        
        message = f"Found {len(significant_groups)} duplicate request groups with {total_duplicates} total duplicates (max allowed: {max_allowed})"
        
        return self._create_result(
            context=context,
            passed=passed,
            message=message,
            found_count=total_duplicates,
            expected_count=max_allowed,
            evidence=[group.to_dict() for group in significant_groups[:10]]
        )
    
    def _group_requests(self, requests: List[RequestLog], config: Dict[str, Any]) -> List[DuplicateGroup]:
        """Group requests by similarity criteria."""
        grouping_fields = config.get('grouping_fields', ['url', 'method'])
        ignore_fields = set(config.get('ignore_fields', []))
        time_window = config.get('time_window_seconds', 0)
        url_normalize = config.get('url_normalize', True)
        include_query_params = config.get('include_query_params', True)
        include_headers = config.get('include_headers', [])
        
        # Create groups based on similarity
        groups = defaultdict(list)
        
        for request in requests:
            group_key = self._create_group_key(
                request, grouping_fields, ignore_fields, url_normalize,
                include_query_params, include_headers
            )
            groups[group_key].append(request)
        
        # Apply time window filtering if specified
        if time_window > 0:
            groups = self._apply_time_window_filter(groups, time_window)
        
        # Convert to DuplicateGroup objects
        duplicate_groups = [
            DuplicateGroup(key, items) for key, items in groups.items()
            if len(items) > 1
        ]
        
        return duplicate_groups
    
    def _create_group_key(
        self,
        request: RequestLog,
        grouping_fields: List[str],
        ignore_fields: Set[str],
        url_normalize: bool,
        include_query_params: bool,
        include_headers: List[str]
    ) -> str:
        """Create a unique key for grouping requests."""
        key_parts = []
        
        for field in grouping_fields:
            if field in ignore_fields:
                continue
                
            if field == 'url':
                url = request.url
                if url_normalize:
                    # Parse and normalize URL
                    parsed = urlparse(url)
                    if not include_query_params:
                        url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    else:
                        # Sort query parameters for consistent grouping
                        query_params = parse_qs(parsed.query)
                        sorted_query = '&'.join(
                            f"{k}={','.join(sorted(v))}" for k, v in sorted(query_params.items())
                        )
                        url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{sorted_query}" if sorted_query else f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                key_parts.append(url)
            elif field == 'method':
                key_parts.append(request.method)
            elif field == 'status_code':
                key_parts.append(str(request.status_code or ''))
            elif field == 'resource_type':
                key_parts.append(request.resource_type)
            elif field == 'request_body':
                # Hash request body to avoid huge keys
                body = request.request_body or ''
                key_parts.append(hashlib.md5(body.encode()).hexdigest())
        
        # Include specific headers if requested
        for header_name in include_headers:
            header_value = request.request_headers.get(header_name, '')
            key_parts.append(f"{header_name}:{header_value}")
        
        return '|'.join(key_parts)
    
    def _apply_time_window_filter(
        self,
        groups: Dict[str, List[RequestLog]],
        time_window_seconds: int
    ) -> Dict[str, List[RequestLog]]:
        """Apply time window filtering to group only requests within the specified window."""
        filtered_groups = {}
        
        for group_key, requests in groups.items():
            # Sort requests by timestamp
            sorted_requests = sorted(requests, key=lambda r: r.start_time)
            
            # Group requests within time windows
            window_groups = []
            current_window = []
            window_start = None
            
            for request in sorted_requests:
                if window_start is None:
                    window_start = request.start_time
                    current_window = [request]
                elif (request.start_time - window_start).total_seconds() <= time_window_seconds:
                    current_window.append(request)
                else:
                    # Start new window
                    if len(current_window) > 1:
                        window_groups.append(current_window)
                    window_start = request.start_time
                    current_window = [request]
            
            # Don't forget the last window
            if len(current_window) > 1:
                window_groups.append(current_window)
            
            # Add window groups as separate groups
            for i, window_requests in enumerate(window_groups):
                filtered_groups[f"{group_key}_window_{i}"] = window_requests
        
        return filtered_groups


@register_check("event_duplicates")
class EventDuplicateCheck(BaseCheck):
    """Detect duplicate tag events based on configurable criteria."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
    
    def get_supported_config_keys(self) -> List[str]:
        return [
            'grouping_fields',
            'ignore_parameters',
            'time_window_seconds',
            'min_duplicates',
            'max_allowed_duplicates',
            'vendor_filter',
            'event_type_filter',
            'parameter_keys'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute event duplicate detection."""
        config = context.check_config
        
        # Get all events
        events = context.query.query().events()
        
        # Apply vendor filter if specified
        if 'vendor_filter' in config:
            vendor = config['vendor_filter']
            events = [e for e in events if e.vendor == vendor]
        
        # Apply event type filter if specified
        if 'event_type_filter' in config:
            event_type = config['event_type_filter']
            events = [e for e in events if e.event_type == event_type]
        
        # Group events by similarity
        duplicate_groups = self._group_events(events, config)
        
        # Filter groups based on duplicate criteria
        min_duplicates = config.get('min_duplicates', 2)
        max_allowed = config.get('max_allowed_duplicates', 0)
        
        significant_groups = [
            group for group in duplicate_groups 
            if group.count >= min_duplicates
        ]
        
        total_duplicates = sum(group.count - 1 for group in significant_groups)
        passed = total_duplicates <= max_allowed
        
        message = f"Found {len(significant_groups)} duplicate event groups with {total_duplicates} total duplicates (max allowed: {max_allowed})"
        
        return self._create_result(
            context=context,
            passed=passed,
            message=message,
            found_count=total_duplicates,
            expected_count=max_allowed,
            evidence=[group.to_dict() for group in significant_groups[:10]]
        )
    
    def _group_events(self, events: List[TagEvent], config: Dict[str, Any]) -> List[DuplicateGroup]:
        """Group events by similarity criteria."""
        grouping_fields = config.get('grouping_fields', ['vendor', 'event_type', 'tag_id'])
        ignore_parameters = set(config.get('ignore_parameters', ['timestamp', '_t']))
        parameter_keys = config.get('parameter_keys', [])
        time_window = config.get('time_window_seconds', 0)
        
        # Create groups based on similarity
        groups = defaultdict(list)
        
        for event in events:
            group_key = self._create_event_group_key(
                event, grouping_fields, ignore_parameters, parameter_keys
            )
            groups[group_key].append(event)
        
        # Apply time window filtering if specified
        if time_window > 0:
            groups = self._apply_event_time_window_filter(groups, time_window)
        
        # Convert to DuplicateGroup objects
        duplicate_groups = [
            DuplicateGroup(key, items) for key, items in groups.items()
            if len(items) > 1
        ]
        
        return duplicate_groups
    
    def _create_event_group_key(
        self,
        event: TagEvent,
        grouping_fields: List[str],
        ignore_parameters: Set[str],
        parameter_keys: List[str]
    ) -> str:
        """Create a unique key for grouping events."""
        key_parts = []
        
        for field in grouping_fields:
            if field == 'vendor':
                key_parts.append(event.vendor)
            elif field == 'event_type':
                key_parts.append(event.event_type)
            elif field == 'tag_id':
                key_parts.append(event.tag_id or '')
            elif field == 'event_name':
                key_parts.append(event.event_name or '')
            elif field == 'page_url':
                key_parts.append(event.page_url or '')
        
        # Include specific parameter keys if requested
        if parameter_keys:
            for param_key in parameter_keys:
                if param_key not in ignore_parameters:
                    param_value = event.parameters.get(param_key, '')
                    key_parts.append(f"{param_key}:{param_value}")
        else:
            # Include all parameters except ignored ones
            sorted_params = sorted(
                (k, v) for k, v in event.parameters.items()
                if k not in ignore_parameters
            )
            param_string = '&'.join(f"{k}={v}" for k, v in sorted_params)
            if param_string:
                key_parts.append(param_string)
        
        return '|'.join(key_parts)
    
    def _apply_event_time_window_filter(
        self,
        groups: Dict[str, List[TagEvent]],
        time_window_seconds: int
    ) -> Dict[str, List[TagEvent]]:
        """Apply time window filtering to group only events within the specified window."""
        filtered_groups = {}
        
        for group_key, events in groups.items():
            # Sort events by timestamp
            sorted_events = sorted(events, key=lambda e: e.timestamp)
            
            # Group events within time windows
            window_groups = []
            current_window = []
            window_start = None
            
            for event in sorted_events:
                if window_start is None:
                    window_start = event.timestamp
                    current_window = [event]
                elif (event.timestamp - window_start).total_seconds() <= time_window_seconds:
                    current_window.append(event)
                else:
                    # Start new window
                    if len(current_window) > 1:
                        window_groups.append(current_window)
                    window_start = event.timestamp
                    current_window = [event]
            
            # Don't forget the last window
            if len(current_window) > 1:
                window_groups.append(current_window)
            
            # Add window groups as separate groups
            for i, window_events in enumerate(window_groups):
                filtered_groups[f"{group_key}_window_{i}"] = window_events
        
        return filtered_groups


@register_check("cookie_duplicates")
class CookieDuplicateCheck(BaseCheck):
    """Detect duplicate cookies that might indicate implementation issues."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
    
    def get_supported_config_keys(self) -> List[str]:
        return [
            'grouping_fields',
            'ignore_domains',
            'min_duplicates',
            'max_allowed_duplicates',
            'same_name_different_domain',
            'same_name_same_domain'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute cookie duplicate detection."""
        config = context.check_config
        
        # Get all cookies
        cookies = context.query.query().cookies()
        
        # Apply domain filters if specified
        if 'ignore_domains' in config:
            ignored_domains = set(config['ignore_domains'])
            cookies = [c for c in cookies if c.domain not in ignored_domains]
        
        # Group cookies by similarity
        duplicate_groups = self._group_cookies(cookies, config)
        
        # Filter groups based on duplicate criteria
        min_duplicates = config.get('min_duplicates', 2)
        max_allowed = config.get('max_allowed_duplicates', 0)
        
        significant_groups = [
            group for group in duplicate_groups 
            if group.count >= min_duplicates
        ]
        
        total_duplicates = sum(group.count - 1 for group in significant_groups)
        passed = total_duplicates <= max_allowed
        
        message = f"Found {len(significant_groups)} duplicate cookie groups with {total_duplicates} total duplicates (max allowed: {max_allowed})"
        
        return self._create_result(
            context=context,
            passed=passed,
            message=message,
            found_count=total_duplicates,
            expected_count=max_allowed,
            evidence=[group.to_dict() for group in significant_groups[:10]]
        )
    
    def _group_cookies(self, cookies: List[CookieRecord], config: Dict[str, Any]) -> List[DuplicateGroup]:
        """Group cookies by similarity criteria."""
        grouping_fields = config.get('grouping_fields', ['name', 'domain'])
        same_name_different_domain = config.get('same_name_different_domain', True)
        same_name_same_domain = config.get('same_name_same_domain', True)
        
        groups = defaultdict(list)
        
        for cookie in cookies:
            group_key = self._create_cookie_group_key(
                cookie, grouping_fields, same_name_different_domain, same_name_same_domain
            )
            if group_key:  # Only group if key is valid
                groups[group_key].append(cookie)
        
        # Convert to DuplicateGroup objects
        duplicate_groups = [
            DuplicateGroup(key, items) for key, items in groups.items()
            if len(items) > 1
        ]
        
        return duplicate_groups
    
    def _create_cookie_group_key(
        self,
        cookie: CookieRecord,
        grouping_fields: List[str],
        same_name_different_domain: bool,
        same_name_same_domain: bool
    ) -> Optional[str]:
        """Create a unique key for grouping cookies."""
        key_parts = []
        
        # Determine grouping strategy
        if same_name_same_domain and 'name' in grouping_fields and 'domain' in grouping_fields:
            # Group cookies with same name and domain (exact duplicates)
            key_parts.extend([cookie.name, cookie.domain])
        elif same_name_different_domain and 'name' in grouping_fields:
            # Group cookies with same name across different domains
            key_parts.append(cookie.name)
        else:
            # Custom grouping based on specified fields
            for field in grouping_fields:
                if field == 'name':
                    key_parts.append(cookie.name)
                elif field == 'domain':
                    key_parts.append(cookie.domain)
                elif field == 'path':
                    key_parts.append(cookie.path)
                elif field == 'secure':
                    key_parts.append(str(cookie.secure))
                elif field == 'http_only':
                    key_parts.append(str(cookie.http_only))
                elif field == 'same_site':
                    key_parts.append(cookie.same_site or '')
        
        return '|'.join(key_parts) if key_parts else None