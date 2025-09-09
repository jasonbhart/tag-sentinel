"""Duplicate event detection and canonical hashing for analytics tag analysis."""

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import (
    BaseDetector,
    DetectContext,
    DetectResult,
    TagEvent,
    TagStatus,
    Confidence,
    Vendor,
    NoteCategory,
    NoteSeverity
)
from ..models.capture import PageResult


class EventCanonicalizer:
    """Handles canonical representation and hashing of events for duplicate detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize canonicalizer with configuration.
        
        Args:
            config: Configuration dict with canonicalization settings
        """
        self.config = config or {}
        
        # Parameters to exclude from hash by default
        self.default_exclude_params = {
            # Timing-related
            "timing_ms", "detected_at", "start_time", "end_time", "timestamp",
            
            # Request-specific 
            "request_url", "status_code", "response_headers", "request_headers",
            
            # Session-specific that may vary
            "cache_buster", "cb", "z", "_", "t",
            
            # Client-specific that may vary within same session
            "viewport_size", "screen_resolution"
        }
        
        # Get configured exclusions
        self.exclude_params = set(self.config.get("exclude_params", self.default_exclude_params))
        
        # Parameters to always include (even if they vary)
        self.always_include = set(self.config.get("always_include", {
            "measurement_id", "container_id", "event_name", "client_id", "session_id"
        }))
    
    def canonicalize_event(self, event: TagEvent) -> Dict[str, Any]:
        """Create canonical representation of an event for comparison.
        
        Args:
            event: TagEvent to canonicalize
            
        Returns:
            Canonical dict representation
        """
        canonical = {
            "vendor": event.vendor.value,
            "name": event.name,
            "category": event.category,
            "id": event.id,
        }
        
        # Process parameters with filtering and normalization
        canonical_params = {}
        for key, value in event.params.items():
            # Skip excluded parameters unless they're in always_include
            if key in self.exclude_params and key not in self.always_include:
                continue
            
            # Normalize the value
            normalized_value = self._normalize_value(value)
            canonical_params[key] = normalized_value
        
        canonical["params"] = canonical_params
        
        return canonical
    
    def _normalize_value(self, value: Any) -> Any:
        """Normalize a parameter value for consistent comparison.
        
        Args:
            value: Parameter value to normalize
            
        Returns:
            Normalized value
        """
        if value is None:
            return None
        
        # Convert to string and strip whitespace
        if isinstance(value, str):
            normalized = value.strip()
            
            # Normalize boolean-like strings
            lower_val = normalized.lower()
            if lower_val in ("true", "1", "yes", "on"):
                return True
            elif lower_val in ("false", "0", "no", "off"):
                return False
            
            return normalized
        
        # Keep numbers as-is
        elif isinstance(value, (int, float)):
            return value
        
        # Keep booleans as-is
        elif isinstance(value, bool):
            return value
        
        # For complex objects, convert to JSON string for comparison
        else:
            try:
                return json.dumps(value, sort_keys=True)
            except (TypeError, ValueError):
                return str(value)
    
    def generate_hash(self, canonical_event: Dict[str, Any]) -> str:
        """Generate consistent hash for canonical event.
        
        Args:
            canonical_event: Canonical event representation
            
        Returns:
            SHA-256 hash string
        """
        # Convert to JSON with sorted keys for consistency
        json_str = json.dumps(canonical_event, sort_keys=True, ensure_ascii=True)
        
        # Generate SHA-256 hash
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    def get_event_hash(self, event: TagEvent) -> str:
        """Get hash for an event (convenience method).
        
        Args:
            event: TagEvent to hash
            
        Returns:
            Event hash string
        """
        canonical = self.canonicalize_event(event)
        return self.generate_hash(canonical)


class DuplicateGroup:
    """Represents a group of duplicate events."""
    
    def __init__(self, canonical_hash: str, first_event: TagEvent):
        self.canonical_hash = canonical_hash
        self.events: List[TagEvent] = [first_event]
        self.count = 1
        self.first_timestamp = first_event.detected_at
        self.last_timestamp = first_event.detected_at
        self.time_span_ms = 0
        
        # Analysis data
        self.unique_request_urls = {first_event.request_url}
        self.unique_timing_values = {first_event.timing_ms} if first_event.timing_ms else set()
        self.status_counts = defaultdict(int)
        self.status_counts[first_event.status] += 1
    
    def add_event(self, event: TagEvent) -> None:
        """Add an event to this duplicate group.
        
        Args:
            event: Event to add to the group
        """
        self.events.append(event)
        self.count += 1
        
        # Update timestamps and time span
        if event.detected_at < self.first_timestamp:
            self.first_timestamp = event.detected_at
        if event.detected_at > self.last_timestamp:
            self.last_timestamp = event.detected_at
        
        self.time_span_ms = int((self.last_timestamp - self.first_timestamp).total_seconds() * 1000)
        
        # Update analysis data
        if event.request_url:
            self.unique_request_urls.add(event.request_url)
        if event.timing_ms:
            self.unique_timing_values.add(event.timing_ms)
        self.status_counts[event.status] += 1
    
    @property
    def is_duplicate(self) -> bool:
        """Check if this group contains duplicates."""
        return self.count > 1
    
    @property 
    def representative_event(self) -> TagEvent:
        """Get the first (representative) event from the group."""
        return self.events[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert group to dictionary for reporting.
        
        Returns:
            Dictionary representation of the duplicate group
        """
        return {
            "canonical_hash": self.canonical_hash,
            "count": self.count,
            "time_span_ms": self.time_span_ms,
            "first_timestamp": self.first_timestamp.isoformat(),
            "last_timestamp": self.last_timestamp.isoformat(),
            "unique_request_urls": len(self.unique_request_urls),
            "unique_timing_values": len(self.unique_timing_values),
            "status_distribution": dict(self.status_counts),
            "representative_event": {
                "name": self.representative_event.name,
                "vendor": self.representative_event.vendor.value,
                "id": self.representative_event.id,
                "confidence": self.representative_event.confidence.value
            }
        }


class DuplicateAnalyzer(BaseDetector):
    """Analyzer for detecting duplicate analytics events."""
    
    def __init__(self):
        super().__init__("DuplicateAnalyzer", "1.0.0")
        self.canonicalizer = None  # Will be initialized with config
    
    @property
    def supported_vendors(self) -> Set[Vendor]:
        """Duplicate analyzer works with all vendors."""
        return {Vendor.GA4, Vendor.GTM, Vendor.ADOBE, Vendor.FACEBOOK, Vendor.UNKNOWN}
    
    def detect(self, page: PageResult, ctx: DetectContext) -> DetectResult:
        """Analyze events for duplicates.
        
        Note: This detector expects to be run after other detectors have
        already populated events. It analyzes existing events for duplicates
        rather than generating new events from the page.
        
        Args:
            page: Page capture result (not used directly)
            ctx: Detection context with existing events and configuration
            
        Returns:
            Detection results with duplicate analysis notes
        """
        result = self._create_result()
        start_time = datetime.utcnow()
        
        try:
            # Initialize canonicalizer with context configuration
            duplicate_config = ctx.config.get("duplicates", {})
            self.canonicalizer = EventCanonicalizer(duplicate_config)
            
            # Get events from context (set by previous detectors)
            events = self._get_events_from_context(ctx)
            
            if not events:
                result.add_info_note(
                    "No events available for duplicate analysis",
                    category=NoteCategory.DATA_QUALITY
                )
                return result
            
            # Analyze duplicates with time window
            time_window_ms = duplicate_config.get("window_ms", 4000)
            duplicate_groups = self._analyze_duplicates(events, time_window_ms)
            
            # Generate analysis notes
            self._generate_duplicate_notes(result, duplicate_groups, ctx)
            
            result.processed_requests = len(events)
            
        except Exception as e:
            result.success = False
            result.error_message = f"Duplicate analysis failed: {str(e)}"
            result.add_error_note(
                f"Duplicate analyzer encountered an error: {str(e)}",
                category=NoteCategory.VALIDATION
            )
        
        # Calculate processing time
        end_time = datetime.utcnow()
        result.processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        return result
    
    def _get_events_from_context(self, ctx: DetectContext) -> List[TagEvent]:
        """Extract events from detection context.
        
        In a real implementation, this would get events from a shared context
        or event store populated by previous detectors. For now, return empty list.
        
        Args:
            ctx: Detection context
            
        Returns:
            List of events to analyze
        """
        # In real implementation, this would be:
        # return ctx.get("detected_events", [])
        return []
    
    def _analyze_duplicates(self, events: List[TagEvent], 
                          time_window_ms: int) -> List[DuplicateGroup]:
        """Analyze events for duplicates within time windows.
        
        Args:
            events: List of events to analyze
            time_window_ms: Time window for considering events as potential duplicates
            
        Returns:
            List of duplicate groups
        """
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.detected_at)
        
        # Group events by hash within time windows
        groups_by_hash: Dict[str, DuplicateGroup] = {}
        
        for event in sorted_events:
            event_hash = self.canonicalizer.get_event_hash(event)
            
            # Check if we have an existing group for this hash within time window
            existing_group = self._find_group_within_window(
                groups_by_hash.get(event_hash), event, time_window_ms
            )
            
            if existing_group:
                existing_group.add_event(event)
            else:
                # Create new group
                new_group = DuplicateGroup(event_hash, event)
                groups_by_hash[event_hash] = new_group
        
        return list(groups_by_hash.values())
    
    def _find_group_within_window(self, existing_group: Optional[DuplicateGroup],
                                event: TagEvent, time_window_ms: int) -> Optional[DuplicateGroup]:
        """Check if event falls within time window of existing group.
        
        Args:
            existing_group: Existing group to check, if any
            event: Event to check
            time_window_ms: Time window in milliseconds
            
        Returns:
            Group if within window, None otherwise
        """
        if not existing_group:
            return None
        
        # Check if event is within time window of the group
        time_diff_ms = abs((event.detected_at - existing_group.last_timestamp).total_seconds() * 1000)
        
        return existing_group if time_diff_ms <= time_window_ms else None
    
    def _generate_duplicate_notes(self, result: DetectResult, 
                                duplicate_groups: List[DuplicateGroup],
                                ctx: DetectContext) -> None:
        """Generate analysis notes based on duplicate detection results.
        
        Args:
            result: Detection result to add notes to
            duplicate_groups: List of all duplicate groups
            ctx: Detection context
        """
        # Filter to only groups with actual duplicates
        actual_duplicates = [group for group in duplicate_groups if group.is_duplicate]
        
        if not actual_duplicates:
            result.add_info_note(
                "No duplicate events detected",
                category=NoteCategory.DUPLICATE
            )
            return
        
        # Overall duplicate summary
        total_duplicates = sum(group.count - 1 for group in actual_duplicates)
        total_events = sum(group.count for group in duplicate_groups)
        
        result.add_warning_note(
            f"Detected {len(actual_duplicates)} duplicate event groups with {total_duplicates} duplicate events out of {total_events} total events",
            category=NoteCategory.DUPLICATE,
            duplicate_groups=len(actual_duplicates),
            duplicate_events=total_duplicates,
            total_events=total_events,
            duplicate_percentage=round((total_duplicates / total_events) * 100, 1) if total_events > 0 else 0
        )
        
        # Detailed notes for significant duplicate groups
        significant_groups = [group for group in actual_duplicates if group.count >= 3]
        for group in significant_groups:
            severity = NoteSeverity.ERROR if group.count >= 5 else NoteSeverity.WARNING
            
            result.add_note({
                "severity": severity,
                "category": NoteCategory.DUPLICATE,
                "message": f"High duplicate count for {group.representative_event.name}: {group.count} identical events in {group.time_span_ms}ms",
                "page_url": "",  # Will be set by detector
                "related_events": [group.representative_event.name],
                "details": group.to_dict(),
                "detector_name": self.name,
                "created_at": datetime.utcnow()
            })
        
        # Analysis by vendor
        vendor_duplicates = defaultdict(list)
        for group in actual_duplicates:
            vendor = group.representative_event.vendor
            vendor_duplicates[vendor].append(group)
        
        for vendor, groups in vendor_duplicates.items():
            duplicate_count = sum(group.count - 1 for group in groups)
            
            if duplicate_count > 0:
                result.add_warning_note(
                    f"{vendor.value.upper()} has {duplicate_count} duplicate events across {len(groups)} event types",
                    category=NoteCategory.DUPLICATE,
                    vendor=vendor.value,
                    duplicate_count=duplicate_count,
                    event_types=len(groups)
                )
        
        # Performance impact note
        high_frequency_groups = [group for group in actual_duplicates 
                               if group.count >= 3 and group.time_span_ms < 1000]
        
        if high_frequency_groups:
            result.add_warning_note(
                f"{len(high_frequency_groups)} event types are firing very frequently (3+ times per second), which may impact performance",
                category=NoteCategory.PERFORMANCE,
                high_frequency_groups=len(high_frequency_groups),
                affected_events=[group.representative_event.name for group in high_frequency_groups]
            )


# Standalone function for analyzing events from other detectors
def analyze_events_for_duplicates(events: List[TagEvent], 
                                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Analyze a list of events for duplicates (utility function).
    
    Args:
        events: List of TagEvent objects to analyze
        config: Optional configuration for duplicate detection
        
    Returns:
        Dictionary with duplicate analysis results
    """
    if not events:
        return {"duplicate_groups": [], "summary": {"duplicates_found": 0}}
    
    # Initialize analyzer components
    duplicate_config = config or {}
    canonicalizer = EventCanonicalizer(duplicate_config)
    time_window_ms = duplicate_config.get("window_ms", 4000)
    
    # Create temporary analyzer instance for processing
    analyzer = DuplicateAnalyzer()
    analyzer.canonicalizer = canonicalizer
    
    # Analyze duplicates
    duplicate_groups = analyzer._analyze_duplicates(events, time_window_ms)
    actual_duplicates = [group for group in duplicate_groups if group.is_duplicate]
    
    # Build results summary
    summary = {
        "duplicates_found": len(actual_duplicates),
        "total_duplicate_events": sum(group.count - 1 for group in actual_duplicates),
        "total_events_analyzed": len(events),
        "duplicate_groups": [group.to_dict() for group in actual_duplicates]
    }
    
    return {
        "duplicate_groups": actual_duplicates,
        "summary": summary
    }