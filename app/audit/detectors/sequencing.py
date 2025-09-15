"""Tag sequencing analysis for detecting timing and dependency issues."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import (
    BaseDetector,
    DetectContext,
    DetectResult,
    TagEvent,
    Vendor,
    NoteCategory,
    NoteSeverity
)
from ..models.capture import PageResult


class SequencingRule:
    """Represents a sequencing rule or expectation."""
    
    def __init__(self, name: str, description: str, 
                 prerequisite_pattern: str, dependent_pattern: str,
                 max_delay_ms: Optional[int] = None,
                 severity: NoteSeverity = NoteSeverity.WARNING):
        """Initialize a sequencing rule.
        
        Args:
            name: Rule name
            description: Human-readable description
            prerequisite_pattern: Pattern to match prerequisite events
            dependent_pattern: Pattern to match dependent events
            max_delay_ms: Maximum acceptable delay between events
            severity: Severity level for violations
        """
        self.name = name
        self.description = description
        self.prerequisite_pattern = prerequisite_pattern
        self.dependent_pattern = dependent_pattern
        self.max_delay_ms = max_delay_ms
        self.severity = severity
    
    def matches_prerequisite(self, event: TagEvent) -> bool:
        """Check if event matches prerequisite pattern."""
        return self._matches_pattern(event, self.prerequisite_pattern)
    
    def matches_dependent(self, event: TagEvent) -> bool:
        """Check if event matches dependent pattern."""
        return self._matches_pattern(event, self.dependent_pattern)
    
    def _matches_pattern(self, event: TagEvent, pattern: str) -> bool:
        """Check if event matches a pattern string."""
        # Simple pattern matching - in real implementation might use regex
        pattern_lower = pattern.lower()
        
        # Check event name
        if event.name and pattern_lower in event.name.lower():
            return True
        
        # Check vendor
        if event.vendor.value.lower() == pattern_lower:
            return True
        
        # Check category
        if event.category and pattern_lower in event.category.lower():
            return True
        
        return False


class SequencingViolation:
    """Represents a detected sequencing violation."""
    
    def __init__(self, rule: SequencingRule, violation_type: str,
                 prerequisite_event: Optional[TagEvent] = None,
                 dependent_event: Optional[TagEvent] = None,
                 timing_delta_ms: Optional[int] = None):
        """Initialize a sequencing violation.
        
        Args:
            rule: The rule that was violated
            violation_type: Type of violation (missing_prerequisite, timing_violation, etc.)
            prerequisite_event: The prerequisite event (if found)
            dependent_event: The dependent event
            timing_delta_ms: Time delta between events
        """
        self.rule = rule
        self.violation_type = violation_type
        self.prerequisite_event = prerequisite_event
        self.dependent_event = dependent_event
        self.timing_delta_ms = timing_delta_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary for reporting."""
        return {
            "rule_name": self.rule.name,
            "rule_description": self.rule.description,
            "violation_type": self.violation_type,
            "timing_delta_ms": self.timing_delta_ms,
            "prerequisite_event": {
                "name": self.prerequisite_event.name,
                "vendor": self.prerequisite_event.vendor.value,
                "timing_ms": self.prerequisite_event.timing_ms
            } if self.prerequisite_event else None,
            "dependent_event": {
                "name": self.dependent_event.name,
                "vendor": self.dependent_event.vendor.value,
                "timing_ms": self.dependent_event.timing_ms
            } if self.dependent_event else None
        }


class SequencingAnalyzer(BaseDetector):
    """Analyzer for tag loading sequence and timing dependencies."""
    
    def __init__(self, name: str = "SequencingAnalyzer"):
        super().__init__(name, "1.0.0")
        self.default_rules = self._create_default_rules()
    
    @property
    def supported_vendors(self) -> Set[Vendor]:
        """Sequencing analyzer works with all vendors."""
        return {Vendor.GA4, Vendor.GTM, Vendor.ADOBE, Vendor.FACEBOOK, Vendor.UNKNOWN}
    
    def detect(self, page: PageResult, ctx: DetectContext) -> DetectResult:
        """Analyze events for sequencing issues.
        
        Args:
            page: Page capture result
            ctx: Detection context with existing events and configuration
            
        Returns:
            Detection results with sequencing analysis notes
        """
        result = self._create_result()
        start_time = datetime.utcnow()
        
        try:
            # Get events from context (set by previous detectors)
            events = self._get_events_from_context(ctx)
            
            if not events:
                result.add_info_note(
                    "No events available for sequencing analysis",
                    category=NoteCategory.SEQUENCING
                )
                return result
            
            # Get sequencing rules from config
            rules = self._get_sequencing_rules(ctx)
            
            # Analyze sequencing violations
            violations = self._analyze_sequencing(events, rules)
            
            # Generate analysis notes
            self._generate_sequencing_notes(result, events, violations, ctx)
            
            result.processed_requests = len(events)
            
        except Exception as e:
            result.success = False
            result.error_message = f"Sequencing analysis failed: {str(e)}"
            result.add_error_note(
                f"Sequencing analyzer encountered an error: {str(e)}",
                category=NoteCategory.VALIDATION
            )
        
        # Calculate processing time
        end_time = datetime.utcnow()
        result.processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        return result
    
    def _get_events_from_context(self, ctx: DetectContext) -> List[TagEvent]:
        """Extract events from detection context.

        Gets events from the shared context populated by previous detectors.

        Args:
            ctx: Detection context

        Returns:
            List of events to analyze
        """
        return ctx.detected_events
    
    def _get_sequencing_rules(self, ctx: DetectContext) -> List[SequencingRule]:
        """Get sequencing rules from configuration.
        
        Args:
            ctx: Detection context with configuration
            
        Returns:
            List of sequencing rules to apply
        """
        sequencing_config = ctx.config.get("sequencing", {})
        rules = []
        
        # Add default rules if enabled
        if sequencing_config.get("use_default_rules", True):
            rules.extend(self.default_rules)
        
        # Add custom rules from config
        custom_rules = sequencing_config.get("custom_rules", [])
        for rule_config in custom_rules:
            rule = SequencingRule(
                name=rule_config["name"],
                description=rule_config["description"],
                prerequisite_pattern=rule_config["prerequisite_pattern"],
                dependent_pattern=rule_config["dependent_pattern"],
                max_delay_ms=rule_config.get("max_delay_ms"),
                severity=NoteSeverity(rule_config.get("severity", "warning"))
            )
            rules.append(rule)
        
        return rules
    
    def _create_default_rules(self) -> List[SequencingRule]:
        """Create default sequencing rules.
        
        Returns:
            List of default sequencing rules
        """
        return [
            # GTM should load before GA4 events
            SequencingRule(
                name="gtm_before_ga4",
                description="GTM container should load before GA4 events fire",
                prerequisite_pattern="gtm",
                dependent_pattern="ga4",
                max_delay_ms=5000,
                severity=NoteSeverity.WARNING
            ),
            
            # Container load should happen before events
            SequencingRule(
                name="container_before_events",
                description="Tag container should load before analytics events",
                prerequisite_pattern="container_load",
                dependent_pattern="analytics_event",
                max_delay_ms=3000,
                severity=NoteSeverity.INFO
            ),
            
            # Page view should typically be first GA4 event
            SequencingRule(
                name="page_view_first",
                description="Page view event should typically fire first",
                prerequisite_pattern="page_view",
                dependent_pattern="ga4",
                severity=NoteSeverity.INFO
            )
        ]
    
    def _analyze_sequencing(self, events: List[TagEvent], 
                          rules: List[SequencingRule]) -> List[SequencingViolation]:
        """Analyze events against sequencing rules.
        
        Args:
            events: List of events to analyze
            rules: List of sequencing rules to check
            
        Returns:
            List of detected violations
        """
        # Sort events by timing for analysis
        sorted_events = self._sort_events_by_timing(events)
        violations = []
        
        for rule in rules:
            rule_violations = self._check_rule(rule, sorted_events)
            violations.extend(rule_violations)
        
        return violations
    
    def _sort_events_by_timing(self, events: List[TagEvent]) -> List[TagEvent]:
        """Sort events by timing information.
        
        Args:
            events: Events to sort
            
        Returns:
            Events sorted by timing
        """
        # Sort by timing_ms if available, then by detected_at timestamp
        def sort_key(event):
            if event.timing_ms is not None:
                return event.timing_ms
            else:
                # Fall back to detected_at timestamp (convert to ms)
                epoch = datetime(1970, 1, 1)
                return int((event.detected_at - epoch).total_seconds() * 1000)
        
        return sorted(events, key=sort_key)
    
    def _check_rule(self, rule: SequencingRule, 
                   sorted_events: List[TagEvent]) -> List[SequencingViolation]:
        """Check a single sequencing rule against events.
        
        Args:
            rule: Rule to check
            sorted_events: Events sorted by timing
            
        Returns:
            List of violations for this rule
        """
        violations = []
        
        # Find prerequisite and dependent events
        prerequisite_events = [e for e in sorted_events if rule.matches_prerequisite(e)]
        dependent_events = [e for e in sorted_events if rule.matches_dependent(e)]
        
        if not dependent_events:
            # No dependent events to check
            return violations
        
        if not prerequisite_events:
            # Dependent events exist but no prerequisite found
            for dependent in dependent_events:
                violations.append(SequencingViolation(
                    rule=rule,
                    violation_type="missing_prerequisite",
                    dependent_event=dependent
                ))
            return violations
        
        # Check timing relationships
        violations.extend(self._check_timing_relationships(
            rule, prerequisite_events, dependent_events
        ))
        
        return violations
    
    def _check_timing_relationships(self, rule: SequencingRule,
                                  prerequisite_events: List[TagEvent],
                                  dependent_events: List[TagEvent]) -> List[SequencingViolation]:
        """Check timing relationships between prerequisite and dependent events.
        
        Args:
            rule: Sequencing rule
            prerequisite_events: Prerequisite events
            dependent_events: Dependent events
            
        Returns:
            List of timing violations
        """
        violations = []
        
        for dependent in dependent_events:
            # Find the closest preceding prerequisite event
            best_prerequisite = None
            best_timing_delta = None
            
            dependent_timing = self._get_event_timing(dependent)
            
            for prerequisite in prerequisite_events:
                prereq_timing = self._get_event_timing(prerequisite)
                
                # Only consider prerequisites that occur before dependent
                if prereq_timing < dependent_timing:
                    timing_delta = dependent_timing - prereq_timing
                    
                    if best_prerequisite is None or timing_delta < best_timing_delta:
                        best_prerequisite = prerequisite
                        best_timing_delta = timing_delta
            
            if best_prerequisite is None:
                # No prerequisite found before this dependent event
                violations.append(SequencingViolation(
                    rule=rule,
                    violation_type="prerequisite_after_dependent",
                    dependent_event=dependent
                ))
            elif rule.max_delay_ms and best_timing_delta > rule.max_delay_ms:
                # Prerequisite found but timing violation
                violations.append(SequencingViolation(
                    rule=rule,
                    violation_type="timing_violation",
                    prerequisite_event=best_prerequisite,
                    dependent_event=dependent,
                    timing_delta_ms=best_timing_delta
                ))
        
        return violations
    
    def _get_event_timing(self, event: TagEvent) -> int:
        """Get event timing in milliseconds.
        
        Args:
            event: Event to get timing for
            
        Returns:
            Timing in milliseconds
        """
        if event.timing_ms is not None:
            return event.timing_ms
        else:
            # Fall back to detected_at timestamp
            epoch = datetime(1970, 1, 1)
            return int((event.detected_at - epoch).total_seconds() * 1000)
    
    def _generate_sequencing_notes(self, result: DetectResult, 
                                 events: List[TagEvent],
                                 violations: List[SequencingViolation],
                                 ctx: DetectContext) -> None:
        """Generate analysis notes based on sequencing results.
        
        Args:
            result: Detection result to add notes to
            events: All analyzed events
            violations: Detected violations
            ctx: Detection context
        """
        if not violations:
            result.add_info_note(
                "No sequencing violations detected",
                category=NoteCategory.SEQUENCING
            )
            return
        
        # Overall summary
        result.add_warning_note(
            f"Detected {len(violations)} sequencing violations across {len(events)} events",
            category=NoteCategory.SEQUENCING,
            violation_count=len(violations),
            total_events=len(events)
        )
        
        # Group violations by type
        violations_by_type = {}
        for violation in violations:
            violation_type = violation.violation_type
            if violation_type not in violations_by_type:
                violations_by_type[violation_type] = []
            violations_by_type[violation_type].append(violation)
        
        # Generate specific notes for each violation type
        for violation_type, type_violations in violations_by_type.items():
            self._generate_violation_type_note(result, violation_type, type_violations)
        
        # Performance impact assessment
        severe_violations = [v for v in violations 
                           if v.rule.severity == NoteSeverity.ERROR]
        if severe_violations:
            result.add_error_note(
                f"{len(severe_violations)} severe sequencing violations detected that may impact functionality",
                category=NoteCategory.SEQUENCING,
                severe_violations=len(severe_violations),
                violation_details=[v.to_dict() for v in severe_violations[:3]]  # Limit detail
            )
        
        # Generate timing analysis summary
        self._generate_timing_summary(result, events)
    
    def _generate_violation_type_note(self, result: DetectResult, 
                                    violation_type: str, 
                                    violations: List[SequencingViolation]) -> None:
        """Generate note for a specific violation type.
        
        Args:
            result: Detection result
            violation_type: Type of violation
            violations: Violations of this type
        """
        violation_count = len(violations)
        
        if violation_type == "missing_prerequisite":
            result.add_warning_note(
                f"{violation_count} events fired without required prerequisite events",
                category=NoteCategory.SEQUENCING,
                violation_type=violation_type,
                affected_events=[v.dependent_event.name for v in violations if v.dependent_event]
            )
        
        elif violation_type == "timing_violation":
            avg_delay = sum(v.timing_delta_ms for v in violations if v.timing_delta_ms) // violation_count
            result.add_warning_note(
                f"{violation_count} events exceeded maximum timing constraints (avg delay: {avg_delay}ms)",
                category=NoteCategory.SEQUENCING,
                violation_type=violation_type,
                average_delay_ms=avg_delay
            )
        
        elif violation_type == "prerequisite_after_dependent":
            result.add_error_note(
                f"{violation_count} events fired before their prerequisite events",
                category=NoteCategory.SEQUENCING,
                violation_type=violation_type,
                affected_events=[v.dependent_event.name for v in violations if v.dependent_event]
            )
    
    def _generate_timing_summary(self, result: DetectResult, events: List[TagEvent]) -> None:
        """Generate timing analysis summary.
        
        Args:
            result: Detection result
            events: All events
        """
        events_with_timing = [e for e in events if e.timing_ms is not None]
        
        if not events_with_timing:
            return
        
        # Calculate timing statistics
        timings = [e.timing_ms for e in events_with_timing]
        min_timing = min(timings)
        max_timing = max(timings)
        avg_timing = sum(timings) // len(timings)
        
        # Identify early and late events
        early_events = [e for e in events_with_timing if e.timing_ms < 1000]
        late_events = [e for e in events_with_timing if e.timing_ms > 10000]
        
        result.add_info_note(
            f"Timing analysis: {len(events_with_timing)} events ranged from {min_timing}ms to {max_timing}ms (avg: {avg_timing}ms)",
            category=NoteCategory.SEQUENCING,
            timing_stats={
                "min_timing_ms": min_timing,
                "max_timing_ms": max_timing,
                "avg_timing_ms": avg_timing,
                "early_events": len(early_events),
                "late_events": len(late_events)
            }
        )