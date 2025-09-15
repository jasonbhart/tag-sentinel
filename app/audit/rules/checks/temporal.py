"""Temporal and sequencing checks for validating analytics implementation timing.

This module implements timing and sequence validation checks to ensure that
analytics tags fire in the correct order, within appropriate time windows,
and with proper dependencies between different components.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

from ...models.capture import RequestLog, PageResult
from ...detectors.base import TagEvent
from ..indexing import TimelineEntry, PageIndex
from ..models import Severity
from .base import BaseCheck, CheckContext, CheckResult, register_check


class SequenceOrderType(str, Enum):
    """Types of sequence ordering validation."""
    STRICT = "strict"          # Must occur in exact order
    LOOSE = "loose"            # Must occur in relative order, but gaps allowed
    ANY_ORDER = "any_order"    # All items must be present, order doesn't matter


class TimingComparison(str, Enum):
    """Types of timing comparisons."""
    BEFORE = "before"
    AFTER = "after"
    WITHIN = "within"
    OUTSIDE = "outside"


class SequenceItem:
    """Represents an item in a timing sequence."""
    
    def __init__(self, identifier: str, timestamp: datetime, item_data: Any = None):
        self.identifier = identifier
        self.timestamp = timestamp
        self.item_data = item_data
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp
    
    def __repr__(self):
        return f"SequenceItem({self.identifier}, {self.timestamp})"


@register_check("load_timing")
class LoadTimingCheck(BaseCheck):
    """Check that page elements load within specified time thresholds."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
    
    def get_supported_config_keys(self) -> List[str]:
        return [
            'max_load_time_ms',
            'min_load_time_ms', 
            'resource_type_thresholds',
            'domain_thresholds',
            'critical_resources',
            'page_url_pattern'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute load timing validation."""
        config = context.check_config
        
        # Get timing thresholds
        max_load_time = config.get('max_load_time_ms', 5000)
        min_load_time = config.get('min_load_time_ms', 0)
        resource_thresholds = config.get('resource_type_thresholds', {})
        domain_thresholds = config.get('domain_thresholds', {})
        critical_resources = config.get('critical_resources', [])
        page_url_pattern = config.get('page_url_pattern')
        
        # Get pages to analyze
        pages = context.indexes.pages.pages
        if page_url_pattern:
            import re
            pattern = re.compile(page_url_pattern)
            pages = [p for p in pages if pattern.search(p.url)]
        
        violations = []
        total_checks = 0
        
        for page in pages:
            page_violations = self._check_page_timing(
                page, max_load_time, min_load_time, 
                resource_thresholds, domain_thresholds, critical_resources
            )
            violations.extend(page_violations)
            total_checks += 1
        
        passed = len(violations) == 0
        message = f"Found {len(violations)} timing violations across {total_checks} pages"
        
        return self._create_result(
            context=context,
            passed=passed,
            message=message,
            found_count=len(violations),
            expected_count=0,
            evidence=violations[:20]  # Limit evidence to first 20 violations
        )
    
    def _check_page_timing(
        self,
        page: PageResult,  # PageIndex → PageResult
        max_load_time: int,
        min_load_time: int,
        resource_thresholds: Dict[str, int],
        domain_thresholds: Dict[str, int],
        critical_resources: List[str]
    ) -> List[Dict[str, Any]]:
        """Check timing violations for a single page."""
        violations = []
        
        # Check overall page load time
        if page.load_time_ms is not None:
            if page.load_time_ms > max_load_time:
                violations.append({
                    'type': 'page_load_slow',
                    'page_url': page.url,
                    'actual_time_ms': page.load_time_ms,
                    'threshold_ms': max_load_time,
                    'violation_ms': page.load_time_ms - max_load_time
                })
            elif page.load_time_ms < min_load_time:
                violations.append({
                    'type': 'page_load_fast',
                    'page_url': page.url,
                    'actual_time_ms': page.load_time_ms,
                    'threshold_ms': min_load_time,
                    'violation_ms': min_load_time - page.load_time_ms
                })
        
        # Check individual request timing
        for req in page.network_requests:
            duration = req.duration_ms
            if duration is None:
                continue
            
            # Check resource type thresholds
            threshold = resource_thresholds.get(req.resource_type)
            if threshold and duration > threshold:
                violations.append({
                    'type': 'resource_slow',
                    'page_url': page.url,
                    'resource_url': req.url,
                    'resource_type': req.resource_type,
                    'actual_time_ms': duration,
                    'threshold_ms': threshold,
                    'violation_ms': duration - threshold
                })
            
            # Check domain thresholds
            from urllib.parse import urlparse
            domain = urlparse(req.url).netloc
            threshold = domain_thresholds.get(domain)
            if threshold and duration > threshold:
                violations.append({
                    'type': 'domain_slow',
                    'page_url': page.url,
                    'resource_url': req.url,
                    'domain': domain,
                    'actual_time_ms': duration,
                    'threshold_ms': threshold,
                    'violation_ms': duration - threshold
                })
            
            # Check critical resources
            for critical_pattern in critical_resources:
                import re
                if re.search(critical_pattern, req.url) and duration > max_load_time / 2:
                    violations.append({
                        'type': 'critical_resource_slow',
                        'page_url': page.url,
                        'resource_url': req.url,
                        'pattern': critical_pattern,
                        'actual_time_ms': duration,
                        'threshold_ms': max_load_time / 2,
                        'violation_ms': duration - (max_load_time / 2)
                    })
        
        return violations


@register_check("sequence_order")
class SequenceOrderCheck(BaseCheck):
    """Validate that events/requests occur in the expected sequence."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
    
    def get_supported_config_keys(self) -> List[str]:
        return [
            'sequence_items',
            'order_type',
            'tolerance_ms',
            'page_url_pattern',
            'required_items',
            'optional_items'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute sequence order validation."""
        config = context.check_config
        
        sequence_items = config.get('sequence_items', [])
        order_type = SequenceOrderType(config.get('order_type', 'strict'))
        tolerance_ms = config.get('tolerance_ms', 100)
        page_url_pattern = config.get('page_url_pattern')
        required_items = config.get('required_items', [])
        optional_items = config.get('optional_items', [])
        
        # Get pages to analyze
        pages = context.indexes.pages.pages
        if page_url_pattern:
            import re
            pattern = re.compile(page_url_pattern)
            pages = [p for p in pages if pattern.search(p.url)]
        
        violations = []
        total_checks = 0
        
        for page in pages:
            page_violations = self._check_page_sequence(
                page, sequence_items, order_type, tolerance_ms,
                required_items, optional_items, context
            )
            violations.extend(page_violations)
            total_checks += 1
        
        passed = len(violations) == 0
        message = f"Found {len(violations)} sequence violations across {total_checks} pages"
        
        return self._create_result(
            context=context,
            passed=passed,
            message=message,
            found_count=len(violations),
            expected_count=0,
            evidence=violations[:20]
        )
    
    def _check_page_sequence(
        self,
        page: PageIndex,
        sequence_items: List[Dict[str, Any]],
        order_type: SequenceOrderType,
        tolerance_ms: int,
        required_items: List[str],
        optional_items: List[str],
        context: CheckContext
    ) -> List[Dict[str, Any]]:
        """Check sequence violations for a single page."""
        violations = []
        
        # Build sequence from timeline
        sequence = self._build_sequence_from_timeline(page, sequence_items, context)
        
        # Check for missing required items
        found_identifiers = {item.identifier for item in sequence}
        for required_id in required_items:
            if required_id not in found_identifiers:
                violations.append({
                    'type': 'missing_required_item',
                    'page_url': page.url,
                    'missing_item': required_id,
                    'found_items': list(found_identifiers)
                })
        
        # Sort sequence by timestamp
        sorted_sequence = sorted(sequence)
        
        if order_type == SequenceOrderType.STRICT:
            violations.extend(self._check_strict_order(page, sorted_sequence, sequence_items, tolerance_ms))
        elif order_type == SequenceOrderType.LOOSE:
            violations.extend(self._check_loose_order(page, sorted_sequence, sequence_items))
        elif order_type == SequenceOrderType.ANY_ORDER:
            # Just check that all required items are present (already done above)
            pass
        
        return violations
    
    def _build_sequence_from_timeline(
        self,
        page: PageResult,  # PageIndex → PageResult
        sequence_items: List[Dict[str, Any]],
        context: CheckContext
    ) -> List[SequenceItem]:
        """Build sequence items from page timeline."""
        sequence = []
        
        for seq_config in sequence_items:
            identifier = seq_config.get('identifier')
            item_type = seq_config.get('type', 'request')  # request, event, cookie, etc.
            
            if item_type == 'request':
                # Find matching requests
                url_pattern = seq_config.get('url_pattern')
                method = seq_config.get('method')
                
                # Get requests from PageResult.network_requests
                network_requests = context.page_result.network_requests if context.page_result else []
                for req in network_requests:
                    if url_pattern:
                        import re
                        if not re.search(url_pattern, req.url):
                            continue
                    if method and req.method != method.upper():
                        continue
                    
                    sequence.append(SequenceItem(
                        identifier=identifier or f"request_{req.url}",
                        timestamp=req.start_time,
                        item_data=req
                    ))
            
            elif item_type == 'event':
                # Find matching events
                vendor = seq_config.get('vendor')
                name = seq_config.get('name')
                event_name = seq_config.get('event_name')
                
                # Get timeline from indexes instead of page.timeline 
                timeline = context.indexes.pages.timelines.get(page.url, [])
                for entry in timeline:
                    if entry.event_type == 'tag_event' and 'tag_event' in entry.metadata:
                        event = entry.metadata['tag_event']
                        
                        if vendor and event.vendor != vendor:
                            continue
                        if name and event.name != name:
                            continue
                        if event_name and event.event_name != event_name:
                            continue
                        
                        sequence.append(SequenceItem(
                            identifier=identifier or f"event_{event.vendor}_{event.name}",
                            timestamp=entry.timestamp,
                            item_data=event
                        ))
        
        return sequence
    
    def _check_strict_order(
        self,
        page: PageIndex,
        sorted_sequence: List[SequenceItem],
        expected_sequence: List[Dict[str, Any]],
        tolerance_ms: int
    ) -> List[Dict[str, Any]]:
        """Check strict ordering with tolerance."""
        violations = []
        
        # Create expected order mapping
        expected_order = {item['identifier']: i for i, item in enumerate(expected_sequence)}
        
        # Check if actual sequence matches expected order
        last_expected_index = -1
        
        for actual_item in sorted_sequence:
            expected_index = expected_order.get(actual_item.identifier)
            if expected_index is None:
                continue
            
            if expected_index < last_expected_index:
                violations.append({
                    'type': 'wrong_order',
                    'page_url': page.url,
                    'item': actual_item.identifier,
                    'expected_after': [seq['identifier'] for seq in expected_sequence[:expected_index]],
                    'actual_timestamp': actual_item.timestamp.isoformat()
                })
            
            last_expected_index = max(last_expected_index, expected_index)
        
        return violations
    
    def _check_loose_order(
        self,
        page: PageIndex,
        sorted_sequence: List[SequenceItem],
        expected_sequence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check loose ordering (relative order maintained)."""
        violations = []
        
        # Create mapping of expected relative orders
        expected_order = {item['identifier']: i for i, item in enumerate(expected_sequence)}
        
        # Track the highest expected index seen so far for each item
        seen_items = {}
        
        for actual_item in sorted_sequence:
            expected_index = expected_order.get(actual_item.identifier)
            if expected_index is None:
                continue
            
            # Check if this item comes before any item that should come after it
            for seen_id, seen_index in seen_items.items():
                seen_expected_index = expected_order.get(seen_id)
                if (seen_expected_index is not None and 
                    expected_index < seen_expected_index):
                    violations.append({
                        'type': 'loose_order_violation',
                        'page_url': page.url,
                        'item': actual_item.identifier,
                        'should_come_before': seen_id,
                        'actual_timestamp': actual_item.timestamp.isoformat()
                    })
            
            seen_items[actual_item.identifier] = expected_index
        
        return violations


@register_check("relative_timing")
class RelativeTimingCheck(BaseCheck):
    """Check timing relationships between different events or requests."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
    
    def get_supported_config_keys(self) -> List[str]:
        return [
            'timing_rules',
            'page_url_pattern',
            'tolerance_ms'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute relative timing validation."""
        config = context.check_config
        
        timing_rules = config.get('timing_rules', [])
        page_url_pattern = config.get('page_url_pattern')
        tolerance_ms = config.get('tolerance_ms', 100)
        
        # Get pages to analyze
        pages = context.indexes.pages.pages
        if page_url_pattern:
            import re
            pattern = re.compile(page_url_pattern)
            pages = [p for p in pages if pattern.search(p.url)]
        
        violations = []
        total_checks = 0
        
        for page in pages:
            page_violations = self._check_page_relative_timing(
                page, timing_rules, tolerance_ms, context
            )
            violations.extend(page_violations)
            total_checks += 1
        
        passed = len(violations) == 0
        message = f"Found {len(violations)} relative timing violations across {total_checks} pages"
        
        return self._create_result(
            context=context,
            passed=passed,
            message=message,
            found_count=len(violations),
            expected_count=0,
            evidence=violations[:20]
        )
    
    def _check_page_relative_timing(
        self,
        page: PageIndex,
        timing_rules: List[Dict[str, Any]],
        tolerance_ms: int,
        context: CheckContext
    ) -> List[Dict[str, Any]]:
        """Check relative timing violations for a single page."""
        violations = []
        
        for rule in timing_rules:
            rule_violations = self._check_timing_rule(page, rule, tolerance_ms, context)
            violations.extend(rule_violations)
        
        return violations
    
    def _check_timing_rule(
        self,
        page: PageIndex,
        rule: Dict[str, Any],
        tolerance_ms: int,
        context: CheckContext
    ) -> List[Dict[str, Any]]:
        """Check a specific timing rule."""
        violations = []
        
        first_item_config = rule.get('first_item', {})
        second_item_config = rule.get('second_item', {})
        comparison = TimingComparison(rule.get('comparison', 'before'))
        threshold_ms = rule.get('threshold_ms', 0)
        
        # Find first items
        first_items = self._find_timeline_items(page, first_item_config, context)
        # Find second items
        second_items = self._find_timeline_items(page, second_item_config, context)
        
        # Compare timing between item pairs
        for first_item in first_items:
            for second_item in second_items:
                violation = self._check_timing_pair(
                    page, first_item, second_item, comparison, 
                    threshold_ms, tolerance_ms
                )
                if violation:
                    violations.append(violation)
        
        return violations
    
    def _find_timeline_items(
        self,
        page: PageIndex,
        item_config: Dict[str, Any],
        context: CheckContext
    ) -> List[SequenceItem]:
        """Find timeline items matching the configuration."""
        items = []
        item_type = item_config.get('type', 'request')
        
        if item_type == 'request':
            url_pattern = item_config.get('url_pattern')
            method = item_config.get('method')
            
            # Get requests from PageResult.network_requests
            network_requests = context.page_result.network_requests if context.page_result else []
            for req in network_requests:
                if url_pattern:
                    import re
                    if not re.search(url_pattern, req.url):
                        continue
                if method and req.method != method.upper():
                    continue
                
                items.append(SequenceItem(
                    identifier=f"request_{req.url}",
                    timestamp=req.start_time,
                    item_data=req
                ))
        
        elif item_type == 'event':
            vendor = item_config.get('vendor')
            name = item_config.get('name')
            
            # Get timeline from indexes instead of page.timeline
            timeline = context.indexes.pages.timelines.get(page.url, [])
            for entry in timeline:
                if entry.event_type == 'tag_event' and 'tag_event' in entry.metadata:
                    event = entry.metadata['tag_event']
                    
                    if vendor and event.vendor != vendor:
                        continue
                    if name and event.name != name:
                        continue
                    
                    items.append(SequenceItem(
                        identifier=f"event_{event.vendor}_{event.name}",
                        timestamp=entry.timestamp,
                        item_data=event
                    ))
        
        return items
    
    def _check_timing_pair(
        self,
        page: PageIndex,
        first_item: SequenceItem,
        second_item: SequenceItem,
        comparison: TimingComparison,
        threshold_ms: int,
        tolerance_ms: int
    ) -> Optional[Dict[str, Any]]:
        """Check timing relationship between two items."""
        time_diff = (second_item.timestamp - first_item.timestamp).total_seconds() * 1000
        
        if comparison == TimingComparison.BEFORE:
            # First item should be before second item
            if time_diff <= tolerance_ms:
                return {
                    'type': 'timing_order_violation',
                    'page_url': page.url,
                    'first_item': first_item.identifier,
                    'second_item': second_item.identifier,
                    'expected': 'first_before_second',
                    'actual_diff_ms': time_diff,
                    'tolerance_ms': tolerance_ms
                }
        
        elif comparison == TimingComparison.AFTER:
            # First item should be after second item
            if time_diff >= -tolerance_ms:
                return {
                    'type': 'timing_order_violation',
                    'page_url': page.url,
                    'first_item': first_item.identifier,
                    'second_item': second_item.identifier,
                    'expected': 'first_after_second',
                    'actual_diff_ms': time_diff,
                    'tolerance_ms': tolerance_ms
                }
        
        elif comparison == TimingComparison.WITHIN:
            # Items should be within threshold of each other
            if abs(time_diff) > threshold_ms + tolerance_ms:
                return {
                    'type': 'timing_distance_violation',
                    'page_url': page.url,
                    'first_item': first_item.identifier,
                    'second_item': second_item.identifier,
                    'expected': f'within_{threshold_ms}ms',
                    'actual_diff_ms': abs(time_diff),
                    'threshold_ms': threshold_ms,
                    'tolerance_ms': tolerance_ms
                }
        
        elif comparison == TimingComparison.OUTSIDE:
            # Items should be outside threshold of each other
            if abs(time_diff) < threshold_ms - tolerance_ms:
                return {
                    'type': 'timing_distance_violation',
                    'page_url': page.url,
                    'first_item': first_item.identifier,
                    'second_item': second_item.identifier,
                    'expected': f'outside_{threshold_ms}ms',
                    'actual_diff_ms': abs(time_diff),
                    'threshold_ms': threshold_ms,
                    'tolerance_ms': tolerance_ms
                }
        
        return None