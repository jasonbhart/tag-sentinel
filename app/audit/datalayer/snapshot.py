"""DataLayer snapshot capture and normalization.

This module provides safe JavaScript execution for capturing dataLayer objects
from browser pages, handling various patterns (arrays, objects, missing),
and normalizing them into consistent structures.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from urllib.parse import urlparse

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from .models import DataLayerSnapshot, DLContext
from .config import CaptureConfig
from .runtime_validation import validate_types, validate_dl_context, validate_datalayer_snapshot

logger = logging.getLogger(__name__)


class SnapshotError(Exception):
    """Errors during dataLayer snapshot capture."""
    pass


class Snapshotter:
    """Captures dataLayer snapshots from browser pages with safety measures."""
    
    def __init__(self, config: CaptureConfig | None = None):
        """Initialize snapshotter with configuration.
        
        Args:
            config: Capture configuration settings
        """
        self.config = config or CaptureConfig()
        self._capture_script = self._build_capture_script()
    
    @validate_types()
    async def take_snapshot(
        self,
        page: Page,
        context: DLContext,
        page_url: str | None = None
    ) -> DataLayerSnapshot:
        """Take a complete dataLayer snapshot from the current page.
        
        Args:
            page: Playwright page instance
            context: DataLayer capture context
            page_url: Override page URL (uses page.url if not provided)
            
        Returns:
            Complete dataLayer snapshot
            
        Raises:
            SnapshotError: If snapshot capture fails
        """
        if not page_url:
            page_url = page.url
        
        logger.debug(f"Taking dataLayer snapshot for {page_url}")
        
        try:
            # Execute safe capture script
            capture_result = await self._execute_capture_script(
                page, context.data_layer_object, context
            )
            
            # Parse and normalize the captured data
            normalized_data = await self._normalize_capture_result(
                capture_result, context
            )
            
            # Create snapshot object
            snapshot = DataLayerSnapshot(
                page_url=page_url,
                capture_time=datetime.utcnow(),
                exists=normalized_data.get('exists', False),
                latest=normalized_data.get('latest'),
                events=normalized_data.get('events', []),
                object_name=context.data_layer_object,
                size_bytes=normalized_data.get('size_bytes'),
                truncated=normalized_data.get('truncated', False),
                depth_reached=normalized_data.get('depth_reached'),
                entries_captured=normalized_data.get('entries_captured', 0)
            )
            
            logger.debug(f"Snapshot captured: exists={snapshot.exists}, "
                        f"variables={snapshot.variable_count}, events={snapshot.event_count}")
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to capture dataLayer snapshot for {page_url}: {e}")
            raise SnapshotError(f"Snapshot capture failed: {e}") from e
    
    @validate_types()
    async def capture_from_page(
        self,
        page: Page,
        context: DLContext
    ) -> DataLayerSnapshot:
        """Capture dataLayer from a page with browser integration support.
        
        This method is specifically designed for browser integration scenarios
        where the page object and context are already established.
        
        Args:
            page: Playwright Page object
            context: DataLayer capture context
            
        Returns:
            DataLayer snapshot
            
        Raises:
            SnapshotError: If capture fails
        """
        try:
            # Get page URL from the page object
            page_url = page.url
            
            # Use the main take_snapshot method
            return await self.take_snapshot(page, context, page_url)
            
        except Exception as e:
            logger.error(f"Browser integration capture failed for {page.url}: {e}")
            raise SnapshotError(f"Browser integration capture failed: {e}") from e
    
    async def _execute_capture_script(
        self,
        page: Page,
        object_name: str,
        context: DLContext
    ) -> Dict[str, Any]:
        """Execute the JavaScript capture script safely.
        
        Args:
            page: Playwright page instance
            object_name: Name of the dataLayer object to capture
            context: Capture context with limits
            
        Returns:
            Raw capture result from JavaScript
            
        Raises:
            SnapshotError: If input validation fails
        """
        # Sanitize and validate JavaScript parameters
        sanitized_object_name = self._sanitize_js_identifier(object_name)
        sanitized_fallbacks = [
            self._sanitize_js_identifier(obj) for obj in self.config.fallback_objects
        ]
        sanitized_patterns = [
            self._sanitize_string(pattern) for pattern in self.config.event_detection_patterns
        ]
        
        # Validate numeric limits
        max_depth = max(1, min(context.max_depth, 20))  # Clamp between 1-20
        max_entries = max(1, min(context.max_entries, 10000))  # Clamp between 1-10000
        max_size = max(1024, min(context.max_size_bytes or 1048576, 10485760))  # Max 10MB
        
        script_args = {
            'objectName': sanitized_object_name,
            'fallbackObjects': sanitized_fallbacks,
            'maxDepth': max_depth,
            'maxEntries': max_entries,
            'maxSize': max_size,
            'safeMode': self.config.safe_mode,
            'normalizePushes': self.config.normalize_pushes,
            'extractEvents': self.config.extract_events,
            'eventPatterns': sanitized_patterns
        }
        
        try:
            # Execute with timeout protection
            result = await page.evaluate(
                self._capture_script,
                script_args,
                timeout=self.config.execution_timeout_ms
            )
            
            # Validate the result before returning
            validated_result = self._validate_script_result(result)
            return validated_result
            
        except PlaywrightTimeoutError:
            logger.warning(f"Capture script timeout after {self.config.execution_timeout_ms}ms")
            return {
                'exists': False,
                'error': 'Script execution timeout',
                'timeout': True
            }
        except Exception as e:
            logger.error(f"JavaScript execution error: {e}")
            return {
                'exists': False,
                'error': str(e),
                'jsError': True
            }
    
    async def _normalize_capture_result(
        self,
        capture_result: Dict[str, Any],
        context: DLContext
    ) -> Dict[str, Any]:
        """Normalize and validate capture result.
        
        Args:
            capture_result: Raw result from JavaScript capture
            context: Capture context
            
        Returns:
            Normalized capture data
        """
        if not capture_result.get('exists', False):
            return {
                'exists': False,
                'latest': None,
                'events': [],
                'size_bytes': 0,
                'truncated': False,
                'depth_reached': 0,
                'entries_captured': 0
            }
        
        # Extract normalized data
        latest = capture_result.get('latest', {})
        events = capture_result.get('events', [])
        
        # Apply additional normalization if needed
        if self.config.normalize_pushes and 'raw' in capture_result:
            latest, events = await self._normalize_push_array(
                capture_result['raw'], context
            )
        
        # Calculate metrics
        size_bytes = self._estimate_size(latest, events)
        depth_reached = self._calculate_max_depth(latest)
        entries_captured = self._count_entries(latest, events)
        
        # Check truncation
        truncated = (
            capture_result.get('truncated', False) or
            size_bytes > context.max_size_bytes or
            entries_captured > context.max_entries or
            depth_reached > context.max_depth
        )
        
        return {
            'exists': True,
            'latest': latest,
            'events': events,
            'size_bytes': size_bytes,
            'truncated': truncated,
            'depth_reached': depth_reached,
            'entries_captured': entries_captured
        }
    
    @validate_types()
    async def _normalize_push_array(
        self,
        raw_data: List[Dict[str, Any]],
        context: DLContext
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Normalize push-based dataLayer array into latest state + events.
        
        Implements sophisticated normalization for GTM-style dataLayer patterns:
        - Event separation with multiple detection strategies
        - Nested object merging with array handling
        - GTM-specific patterns like ecommerce data
        - Historical state preservation
        - Conditional and functional push patterns
        
        Args:
            raw_data: Raw dataLayer push array
            context: Capture context
            
        Returns:
            Tuple of (latest_state, events_list)
        """
        if not isinstance(raw_data, list):
            return raw_data if isinstance(raw_data, dict) else {}, []
        
        latest_state = {}
        events = []
        
        # Process pushes in chronological order
        for i, push in enumerate(raw_data):
            if not isinstance(push, dict):
                logger.debug(f"Skipping non-dict push at index {i}: {type(push)}")
                continue
            
            try:
                # Enhanced event detection with multiple strategies
                if self._is_event_push(push):
                    # Process and store event with metadata
                    processed_event = self._process_event_push(push, i, latest_state.copy())
                    events.append(processed_event)
                else:
                    # Process variable update with sophisticated merging
                    self._merge_variable_push(latest_state, push, i)
            
            except Exception as e:
                logger.warning(f"Error processing push at index {i}: {e}")
                # Continue processing other pushes
                continue
        
        # Post-processing: clean up state and validate events
        latest_state = self._clean_normalized_state(latest_state)
        events = self._validate_and_enhance_events(events, latest_state)
        
        logger.debug(f"Normalized {len(raw_data)} pushes into {len(events)} events and "
                    f"{len(latest_state)} state variables")
        
        return latest_state, events
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source dictionary into target.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def _is_event_push(self, push: Dict[str, Any]) -> bool:
        """Enhanced event detection with multiple strategies.
        
        Detects events using:
        1. Standard GTM patterns (event, eventName, etc.)
        2. Ecommerce patterns (purchase, add_to_cart, etc.)
        3. Custom event indicators
        4. Functional patterns (functions, callbacks)
        
        Args:
            push: Push object to analyze
            
        Returns:
            True if this push represents an event
        """
        # Strategy 1: Standard event patterns
        for pattern in self.config.event_detection_patterns:
            if pattern in push:
                return True
        
        # Strategy 2: GTM ecommerce event patterns
        ecommerce_indicators = [
            'ecommerce', 'purchase', 'add_to_cart', 'remove_from_cart',
            'view_item', 'view_item_list', 'select_item', 'begin_checkout',
            'add_payment_info', 'add_shipping_info'
        ]
        if any(indicator in push for indicator in ecommerce_indicators):
            return True
        
        # Strategy 3: Custom event patterns
        custom_event_patterns = [
            'gtm.', '_event', 'eventAction', 'eventCategory', 'eventLabel',
            'customEvent', 'trackingEvent'
        ]
        for pattern in custom_event_patterns:
            if any(key.startswith(pattern) for key in push.keys()):
                return True
        
        # Strategy 4: Functional patterns (GTM functions)
        if 'gtm.element' in push or 'gtm.elementId' in push or 'gtm.elementClasses' in push:
            return True
        
        # Strategy 5: Event value patterns (one-time data)
        event_value_patterns = ['value', 'revenue', 'currency', 'items', 'products']
        if len(push) == 1 and any(key in event_value_patterns for key in push.keys()):
            return True
        
        return False
    
    def _process_event_push(self, push: Dict[str, Any], index: int, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enhance an event push with metadata.
        
        Args:
            push: Raw event push
            index: Position in push array
            current_state: Current dataLayer state at time of event
            
        Returns:
            Enhanced event object with metadata
        """
        processed_event = push.copy()
        
        # Add processing metadata
        processed_event['_meta'] = {
            'push_index': index,
            'processed_at': datetime.utcnow().isoformat(),
            'event_type': self._classify_event_type(push),
            'context_variables': self._extract_relevant_context(push, current_state)
        }
        
        # Enhance ecommerce events
        if 'ecommerce' in push:
            processed_event = self._enhance_ecommerce_event(processed_event, current_state)
        
        # Enhance GTM events
        if any(key.startswith('gtm.') for key in push.keys()):
            processed_event = self._enhance_gtm_event(processed_event, current_state)
        
        return processed_event
    
    def _merge_variable_push(self, latest_state: Dict[str, Any], push: Dict[str, Any], index: int) -> None:
        """Merge a variable push into the latest state with sophisticated handling.
        
        Handles:
        - Nested object merging
        - Array handling strategies (replace vs append)
        - GTM-specific variable patterns
        - Conditional variable updates
        
        Args:
            latest_state: Current state to merge into
            push: Push containing variable updates
            index: Position in push array
        """
        try:
            # Handle special GTM clear operations
            if push.get('event') == 'gtm.clear' or 'gtm.clear' in push:
                self._handle_clear_operation(latest_state, push)
                return
            
            # Process each key-value pair in push
            for key, value in push.items():
                if key.startswith('_') or key.startswith('gtm.'):
                    # Skip internal GTM variables unless explicitly configured
                    if not self.config.global_settings.get('capture_gtm_internals', False):
                        continue
                
                self._merge_variable_value(latest_state, key, value, index)
                
        except Exception as e:
            logger.warning(f"Error merging variable push at index {index}: {e}")
            # Fallback to simple merge
            self._deep_merge(latest_state, push)
    
    def _merge_variable_value(self, state: Dict[str, Any], key: str, value: Any, index: int) -> None:
        """Merge a single variable value with sophisticated handling.
        
        Args:
            state: State to merge into
            key: Variable key
            value: Variable value
            index: Push index for debugging
        """
        if key not in state:
            # New variable - simple assignment
            state[key] = value
            return
        
        current_value = state[key]
        
        # Handle array merging strategies
        if isinstance(current_value, list) and isinstance(value, list):
            # Array merging strategy based on content
            if self._should_append_arrays(key, current_value, value):
                state[key] = current_value + value
            else:
                state[key] = value  # Replace
        elif isinstance(current_value, dict) and isinstance(value, dict):
            # Deep merge objects
            self._deep_merge(current_value, value)
        else:
            # Simple replacement for primitive values
            state[key] = value
    
    def _should_append_arrays(self, key: str, current: List[Any], new: List[Any]) -> bool:
        """Determine if arrays should be appended or replaced.
        
        Args:
            key: Variable key
            current: Current array value
            new: New array value
            
        Returns:
            True if arrays should be appended
        """
        # Append for known accumulative patterns
        accumulative_patterns = ['events', 'history', 'items', 'products', 'errors']
        if any(pattern in key.lower() for pattern in accumulative_patterns):
            return True
        
        # Replace for configuration-like arrays
        config_patterns = ['config', 'settings', 'options']
        if any(pattern in key.lower() for pattern in config_patterns):
            return False
        
        # Default: replace if arrays are different lengths, append if similar
        return len(current) == len(new)
    
    def _classify_event_type(self, event: Dict[str, Any]) -> str:
        """Classify the type of event for metadata.
        
        Args:
            event: Event push to classify
            
        Returns:
            Event type string
        """
        if 'ecommerce' in event:
            return 'ecommerce'
        elif any(key.startswith('gtm.') for key in event.keys()):
            return 'gtm'
        elif 'event' in event:
            return 'custom'
        elif any(pattern in event for pattern in self.config.event_detection_patterns):
            return 'standard'
        else:
            return 'unknown'
    
    def _extract_relevant_context(self, event: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant context variables for an event.
        
        Args:
            event: Event to extract context for
            state: Current dataLayer state
            
        Returns:
            Relevant context variables
        """
        context = {}
        
        # Always include user and session context if available
        context_keys = ['user', 'userId', 'session', 'sessionId', 'page', 'pageType']
        for key in context_keys:
            if key in state:
                context[key] = state[key]
        
        # Include ecommerce context for ecommerce events
        if 'ecommerce' in event or any(key in event for key in ['purchase', 'add_to_cart']):
            ecommerce_context = ['currency', 'affiliation', 'coupon']
            for key in ecommerce_context:
                if key in state:
                    context[key] = state[key]
        
        return context
    
    def _enhance_ecommerce_event(self, event: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance ecommerce events with additional processing.
        
        Args:
            event: Ecommerce event to enhance
            state: Current dataLayer state
            
        Returns:
            Enhanced event
        """
        if 'ecommerce' not in event:
            return event
        
        ecommerce_data = event['ecommerce']
        
        # Calculate derived metrics
        if 'items' in ecommerce_data or 'products' in ecommerce_data:
            items = ecommerce_data.get('items', ecommerce_data.get('products', []))
            if isinstance(items, list):
                event['_derived'] = {
                    'item_count': len(items),
                    'total_value': sum(float(item.get('value', 0)) for item in items if isinstance(item, dict)),
                    'unique_categories': len(set(item.get('item_category', '') for item in items if isinstance(item, dict)))
                }
        
        return event
    
    def _enhance_gtm_event(self, event: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance GTM-specific events.
        
        Args:
            event: GTM event to enhance
            state: Current dataLayer state
            
        Returns:
            Enhanced event
        """
        # Add GTM trigger information if available
        gtm_keys = [key for key in event.keys() if key.startswith('gtm.')]
        if gtm_keys:
            event['_gtm_context'] = {key: event[key] for key in gtm_keys}
        
        return event
    
    def _handle_clear_operation(self, state: Dict[str, Any], push: Dict[str, Any]) -> None:
        """Handle GTM clear operations.
        
        Args:
            state: State to clear from
            push: Push containing clear instructions
        """
        # Implement GTM-style clearing
        if 'ecommerce' in push and push.get('ecommerce') is None:
            # Clear ecommerce data
            state.pop('ecommerce', None)
    
    def _clean_normalized_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and optimize the normalized state.
        
        Args:
            state: Raw normalized state
            
        Returns:
            Cleaned state
        """
        cleaned = {}
        
        for key, value in state.items():
            # Skip empty values unless they're meaningful
            if value is None or value == '' or value == []:
                if key not in ['userId', 'event', 'ecommerce']:  # Keep meaningful empty values
                    continue
            
            # Clean nested objects
            if isinstance(value, dict):
                cleaned_value = self._clean_normalized_state(value)
                if cleaned_value:  # Only include non-empty objects
                    cleaned[key] = cleaned_value
            elif isinstance(value, list):
                # Clean and filter list items
                cleaned_list = [item for item in value if item is not None]
                if cleaned_list:
                    cleaned[key] = cleaned_list
            else:
                cleaned[key] = value
        
        return cleaned
    
    def _validate_and_enhance_events(self, events: List[Dict[str, Any]], final_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate and enhance the extracted events.
        
        Args:
            events: Raw extracted events
            final_state: Final normalized state
            
        Returns:
            Validated and enhanced events
        """
        validated_events = []
        
        for i, event in enumerate(events):
            try:
                # Basic validation
                if not isinstance(event, dict) or not event:
                    logger.debug(f"Skipping invalid event at index {i}")
                    continue
                
                # Add sequence information
                if '_meta' in event:
                    event['_meta']['sequence'] = i
                    event['_meta']['final_state_snapshot'] = {
                        key: final_state.get(key) for key in ['userId', 'sessionId', 'page']
                        if key in final_state
                    }
                
                validated_events.append(event)
                
            except Exception as e:
                logger.warning(f"Error validating event at index {i}: {e}")
                continue
        
        logger.debug(f"Validated {len(validated_events)} out of {len(events)} events")
        return validated_events
    
    def _estimate_size(self, latest: Dict[str, Any], events: List[Dict[str, Any]]) -> int:
        """Estimate size of captured data in bytes.
        
        Args:
            latest: Latest state data
            events: Events list
            
        Returns:
            Estimated size in bytes
        """
        try:
            # Use JSON serialization to estimate size
            latest_json = json.dumps(latest, default=str)
            events_json = json.dumps(events, default=str)
            return len(latest_json.encode('utf-8')) + len(events_json.encode('utf-8'))
        except Exception:
            # Fallback estimation
            return len(str(latest)) + len(str(events))
    
    def _calculate_max_depth(self, data: Dict[str, Any]) -> int:
        """Calculate maximum nesting depth in data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Maximum depth found
        """
        def _depth(obj: Any, current_depth: int = 0) -> int:
            if not isinstance(obj, dict):
                return current_depth
            
            if not obj:  # Empty dict
                return current_depth
            
            max_child_depth = current_depth
            for value in obj.values():
                if isinstance(value, dict):
                    child_depth = _depth(value, current_depth + 1)
                    max_child_depth = max(max_child_depth, child_depth)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            child_depth = _depth(item, current_depth + 1)
                            max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth
        
        return _depth(data)
    
    def _count_entries(self, latest: Dict[str, Any], events: List[Dict[str, Any]]) -> int:
        """Count total entries in captured data.
        
        Args:
            latest: Latest state data
            events: Events list
            
        Returns:
            Total entry count
        """
        def _count_recursive(obj: Any) -> int:
            if isinstance(obj, dict):
                count = len(obj)
                for value in obj.values():
                    count += _count_recursive(value)
                return count
            elif isinstance(obj, list):
                count = len(obj)
                for item in obj:
                    count += _count_recursive(item)
                return count
            else:
                return 0
        
        latest_count = _count_recursive(latest)
        events_count = _count_recursive(events)
        return latest_count + events_count
    
    def _sanitize_js_identifier(self, identifier: str) -> str:
        """Sanitize JavaScript identifier to prevent injection.
        
        Args:
            identifier: JavaScript identifier to sanitize
            
        Returns:
            Sanitized identifier
            
        Raises:
            SnapshotError: If identifier is invalid
        """
        if not identifier or not isinstance(identifier, str):
            raise SnapshotError("JavaScript identifier must be a non-empty string")
        
        # Allow only alphanumeric, underscore, and dollar sign
        # This matches JavaScript identifier rules
        import re
        if not re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', identifier):
            raise SnapshotError(f"Invalid JavaScript identifier: {identifier}")
        
        # Block dangerous keywords
        dangerous_keywords = {
            'eval', 'Function', 'setTimeout', 'setInterval', 'document', 'window',
            'alert', 'confirm', 'prompt', 'console', 'XMLHttpRequest', 'fetch'
        }
        
        if identifier in dangerous_keywords:
            raise SnapshotError(f"Dangerous JavaScript identifier blocked: {identifier}")
        
        return identifier
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string value to prevent injection.
        
        Args:
            value: String value to sanitize
            
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            return str(value)
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\x00', '\n', '\r', '\t']
        sanitized = value
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length to prevent DoS
        return sanitized[:100]
    
    def _validate_script_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JavaScript execution result.
        
        Args:
            result: Raw result from JavaScript
            
        Returns:
            Validated result
            
        Raises:
            SnapshotError: If result is invalid
        """
        if not isinstance(result, dict):
            raise SnapshotError(f"Invalid script result type: {type(result).__name__}")
        
        # Validate required fields
        if 'exists' not in result:
            raise SnapshotError("Script result missing 'exists' field")
        
        if not isinstance(result['exists'], bool):
            raise SnapshotError("Script result 'exists' field must be boolean")
        
        # Validate optional fields with proper types
        if 'latest' in result and result['latest'] is not None:
            if not isinstance(result['latest'], dict):
                raise SnapshotError("Script result 'latest' field must be dict or null")
        
        if 'events' in result and result['events'] is not None:
            if not isinstance(result['events'], list):
                raise SnapshotError("Script result 'events' field must be list or null")
        
        return result
    
    def _build_capture_script(self) -> str:
        """Build the JavaScript capture script.
        
        Returns:
            Complete JavaScript code for safe dataLayer capture
        """
        return """
        (function(args) {
            const {
                objectName,
                fallbackObjects,
                maxDepth,
                maxEntries,
                maxSize,
                safeMode,
                normalizePushes,
                extractEvents,
                eventPatterns
            } = args;
            
            // Utility functions
            function isObject(obj) {
                return obj !== null && typeof obj === 'object' && !Array.isArray(obj);
            }
            
            function isCircular(obj, seen = new WeakSet()) {
                if (obj === null || typeof obj !== 'object') return false;
                if (seen.has(obj)) return true;
                seen.add(obj);
                
                for (let key in obj) {
                    if (obj.hasOwnProperty(key) && isCircular(obj[key], seen)) {
                        return true;
                    }
                }
                
                seen.delete(obj);
                return false;
            }
            
            function safeClone(obj, depth = 0, maxDepth = 10, entriesCount = { count: 0 }) {
                // Prevent infinite recursion
                if (depth > maxDepth || entriesCount.count > maxEntries) {
                    return '[TRUNCATED_DEPTH]';
                }
                
                // Handle null/undefined
                if (obj === null || obj === undefined) {
                    return obj;
                }
                
                // Handle primitives
                if (typeof obj !== 'object') {
                    return obj;
                }
                
                // Handle arrays
                if (Array.isArray(obj)) {
                    entriesCount.count++;
                    const result = [];
                    for (let i = 0; i < obj.length && entriesCount.count < maxEntries; i++) {
                        result[i] = safeClone(obj[i], depth + 1, maxDepth, entriesCount);
                    }
                    if (obj.length > result.length) {
                        result.push('[TRUNCATED_ARRAY]');
                    }
                    return result;
                }
                
                // Handle objects
                entriesCount.count++;
                const result = {};
                
                // Protect against getter side effects in safe mode
                if (safeMode) {
                    try {
                        // Check if object has custom getters
                        const descriptor = Object.getOwnPropertyDescriptor(obj, 'toString');
                        if (descriptor && descriptor.get) {
                            return '[OBJECT_WITH_GETTERS]';
                        }
                    } catch (e) {
                        return '[GETTER_ERROR]';
                    }
                }
                
                try {
                    for (let key in obj) {
                        if (entriesCount.count >= maxEntries) {
                            result['[TRUNCATED_ENTRIES]'] = true;
                            break;
                        }
                        
                        if (obj.hasOwnProperty(key)) {
                            try {
                                const value = obj[key];
                                
                                // Skip functions in safe mode
                                if (safeMode && typeof value === 'function') {
                                    result[key] = '[FUNCTION]';
                                    continue;
                                }
                                
                                // Skip DOM nodes and complex objects
                                if (safeMode && value && typeof value === 'object') {
                                    if (value.nodeType || value.window === value) {
                                        result[key] = '[DOM_OBJECT]';
                                        continue;
                                    }
                                }
                                
                                result[key] = safeClone(value, depth + 1, maxDepth, entriesCount);
                            } catch (e) {
                                result[key] = `[ERROR: ${e.message}]`;
                            }
                        }
                    }
                } catch (e) {
                    return `[OBJECT_ERROR: ${e.message}]`;
                }
                
                return result;
            }
            
            function findDataLayerObject() {
                // Try primary object name
                if (typeof window[objectName] !== 'undefined') {
                    return { name: objectName, object: window[objectName] };
                }
                
                // Try fallback objects
                for (let fallback of fallbackObjects || []) {
                    if (typeof window[fallback] !== 'undefined') {
                        return { name: fallback, object: window[fallback] };
                    }
                }
                
                return null;
            }
            
            function normalizeDataLayer(dataLayerObj) {
                if (!dataLayerObj) {
                    return { latest: {}, events: [], raw: null };
                }
                
                // If it's an array (typical GTM pattern)
                if (Array.isArray(dataLayerObj)) {
                    if (!normalizePushes) {
                        return { latest: {}, events: [], raw: dataLayerObj };
                    }
                    
                    const latest = {};
                    const events = [];
                    
                    for (let push of dataLayerObj) {
                        if (!isObject(push)) continue;
                        
                        // Check if this is an event push
                        const isEvent = eventPatterns.some(pattern => 
                            push.hasOwnProperty(pattern)
                        );
                        
                        if (isEvent && extractEvents) {
                            events.push(push);
                        } else {
                            // Merge into latest state
                            Object.assign(latest, push);
                        }
                    }
                    
                    return { latest, events, raw: dataLayerObj };
                } else if (isObject(dataLayerObj)) {
                    // If it's a plain object
                    return { latest: dataLayerObj, events: [], raw: dataLayerObj };
                } else {
                    // Unknown format
                    return { latest: {}, events: [], raw: dataLayerObj };
                }
            }
            
            function estimateSize(obj) {
                try {
                    return JSON.stringify(obj).length * 2; // Rough Unicode estimate
                } catch (e) {
                    return 0;
                }
            }
            
            // Main capture logic
            try {
                const startTime = Date.now();
                
                // Find the dataLayer object
                const foundObject = findDataLayerObject();
                
                if (!foundObject) {
                    return {
                        exists: false,
                        objectName: objectName,
                        error: 'Object not found',
                        timestamp: new Date().toISOString(),
                        processingTime: Date.now() - startTime
                    };
                }
                
                // Check for circular references if in safe mode
                if (safeMode && isCircular(foundObject.object)) {
                    return {
                        exists: true,
                        objectName: foundObject.name,
                        error: 'Circular reference detected',
                        latest: {},
                        events: [],
                        truncated: true,
                        timestamp: new Date().toISOString(),
                        processingTime: Date.now() - startTime
                    };
                }
                
                // Normalize the dataLayer structure
                const normalized = normalizeDataLayer(foundObject.object);
                
                // Safe clone with limits
                const entriesCount = { count: 0 };
                const clonedLatest = safeClone(normalized.latest, 0, maxDepth, entriesCount);
                const clonedEvents = safeClone(normalized.events, 0, maxDepth, entriesCount);
                
                // Check size limits
                const totalSize = estimateSize(clonedLatest) + estimateSize(clonedEvents);
                const truncated = totalSize > maxSize || entriesCount.count > maxEntries;
                
                return {
                    exists: true,
                    objectName: foundObject.name,
                    latest: clonedLatest,
                    events: clonedEvents,
                    raw: normalizePushes ? null : safeClone(normalized.raw, 0, maxDepth, { count: 0 }),
                    truncated: truncated,
                    sizeBytes: totalSize,
                    entriesCount: entriesCount.count,
                    timestamp: new Date().toISOString(),
                    processingTime: Date.now() - startTime
                };
                
            } catch (error) {
                return {
                    exists: false,
                    objectName: objectName,
                    error: error.message,
                    stack: safeMode ? null : error.stack,
                    timestamp: new Date().toISOString(),
                    processingTime: Date.now() - Date.now()
                };
            }
        })
        """


class BatchSnapshotter:
    """Handles batch processing of multiple page snapshots."""
    
    def __init__(self, snapshotter: Snapshotter, max_concurrency: int = 5):
        """Initialize batch snapshotter.
        
        Args:
            snapshotter: Individual page snapshotter
            max_concurrency: Maximum concurrent snapshots
        """
        self.snapshotter = snapshotter
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def take_batch_snapshots(
        self,
        page_contexts: List[Tuple[Page, DLContext, str]],
        progress_callback: Callable | None = None
    ) -> List[DataLayerSnapshot]:
        """Take snapshots from multiple pages concurrently.
        
        Args:
            page_contexts: List of (page, context, url) tuples
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of snapshots in same order as input
        """
        async def _take_single_snapshot(page_context: Tuple[Page, DLContext, str], index: int):
            async with self.semaphore:
                page, context, url = page_context
                try:
                    snapshot = await self.snapshotter.take_snapshot(page, context, url)
                    if progress_callback:
                        await progress_callback(index, len(page_contexts), snapshot)
                    return snapshot
                except Exception as e:
                    logger.error(f"Batch snapshot failed for {url}: {e}")
                    # Return empty snapshot on error
                    return DataLayerSnapshot(
                        page_url=url,
                        exists=False,
                        object_name=context.data_layer_object
                    )
        
        # Execute all snapshots concurrently
        tasks = [
            _take_single_snapshot(page_context, i)
            for i, page_context in enumerate(page_contexts)
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=False)