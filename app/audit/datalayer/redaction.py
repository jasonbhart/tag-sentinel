"""Sensitive data redaction for dataLayer snapshots.

This module provides comprehensive redaction capabilities for sensitive data
in dataLayer snapshots, supporting multiple redaction methods and path-based
targeting with JSON Pointer paths and glob patterns.
"""

import hashlib
import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
import fnmatch

from .models import RedactionMethod
from .config import RedactionConfig, RedactionRuleConfig
from .runtime_validation import validate_types

logger = logging.getLogger(__name__)


class RedactionError(Exception):
    """Errors during data redaction."""
    pass


class RedactionAuditEntry:
    """Audit trail entry for redacted data."""
    
    def __init__(
        self,
        path: str,
        original_type: str,
        method: RedactionMethod,
        reason: str | None = None,
        pattern_matched: str | None = None
    ):
        self.path = path
        self.original_type = original_type
        self.method = method
        self.reason = reason
        self.pattern_matched = pattern_matched
        self.timestamp = datetime.utcnow()
        
        # Generate content hash for integrity verification
        content_to_hash = f"{path}:{original_type}:{method}:{reason}"
        self.integrity_hash = hashlib.sha256(content_to_hash.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'path': self.path,
            'original_type': self.original_type,
            'method': self.method,
            'reason': self.reason,
            'pattern_matched': self.pattern_matched,
            'timestamp': self.timestamp.isoformat(),
            'integrity_hash': self.integrity_hash
        }


class PathMatcher:
    """Handles path matching for redaction rules."""
    
    def __init__(self):
        self._compiled_patterns: Dict[str, re.Pattern] = {}
    
    def matches_json_pointer(self, path: str, target_path: str) -> bool:
        """Check if target path matches JSON Pointer pattern.
        
        Args:
            path: JSON Pointer path from data (e.g., '/user/email')
            target_path: Target pattern (e.g., '/user/email' or '/user/*')
            
        Returns:
            True if path matches pattern
        """
        # Exact match
        if path == target_path:
            return True
        
        # Convert glob pattern to regex
        if '*' in target_path or '**' in target_path:
            return self._matches_glob_pattern(path, target_path)
        
        return False
    
    def _matches_glob_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches glob-style pattern.
        
        Args:
            path: Actual path to check
            pattern: Glob pattern with * and **
            
        Returns:
            True if path matches pattern
        """
        # Cache compiled patterns for performance
        if pattern not in self._compiled_patterns:
            # Convert glob pattern to regex
            regex_pattern = self._glob_to_regex(pattern)
            self._compiled_patterns[pattern] = re.compile(regex_pattern)
        
        return bool(self._compiled_patterns[pattern].match(path))
    
    def _glob_to_regex(self, pattern: str) -> str:
        """Convert glob pattern to regex.
        
        Args:
            pattern: Glob pattern string
            
        Returns:
            Regex pattern string
        """
        # Escape special regex characters except * and **
        escaped = re.escape(pattern)
        
        # Convert escaped glob patterns back to regex
        escaped = escaped.replace(r'\*\*', '.*')  # ** matches any characters including /
        escaped = escaped.replace(r'\*', '[^/]*')  # * matches any characters except /
        
        # Ensure full string match
        return f'^{escaped}$'


class SensitiveDataPattern:
    """Represents a pattern for detecting sensitive data."""
    
    def __init__(
        self,
        name: str,
        pattern: str,
        method: RedactionMethod = RedactionMethod.HASH,
        confidence: float = 1.0,
        description: str = "",
        category: str = "general",
        severity: str = "medium",
        context_required: bool = False,
        validation_func: Optional[callable] = None
    ):
        """Initialize sensitive data pattern.
        
        Args:
            name: Pattern name
            pattern: Regex pattern string
            method: Preferred redaction method
            confidence: Confidence level (0.0-1.0)
            description: Human-readable description
            category: Pattern category (pii, financial, etc.)
            severity: Severity level (low, medium, high, critical)
            context_required: Whether context validation is needed
            validation_func: Optional validation function for matches
        """
        self.name = name
        self.pattern = pattern
        self.method = method
        self.confidence = confidence
        self.description = description
        self.category = category
        self.severity = severity
        self.context_required = context_required
        self.validation_func = validation_func
        
        try:
            self.compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            logger.error(f"Invalid pattern '{name}': {e}")
            raise RedactionError(f"Invalid pattern '{name}': {e}")
    
    def matches(self, text: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find matches in text with optional context validation.
        
        Args:
            text: Text to search
            context: Context information for validation
            
        Returns:
            List of match details
        """
        matches = []
        
        for match in self.compiled_pattern.finditer(text):
            match_info = {
                'start': match.start(),
                'end': match.end(),
                'value': match.group(0),
                'confidence': self.confidence,
                'category': self.category,
                'severity': self.severity
            }
            
            # Apply validation function if provided
            if self.validation_func:
                try:
                    if not self.validation_func(match.group(0), context or {}):
                        continue
                except Exception as e:
                    logger.debug(f"Pattern validation failed for {self.name}: {e}")
                    continue
            
            # Apply context validation if required
            if self.context_required and context:
                if not self._validate_context(match.group(0), context):
                    continue
            
            matches.append(match_info)
        
        return matches
    
    def _validate_context(self, match_value: str, context: Dict[str, Any]) -> bool:
        """Validate match based on context.
        
        Args:
            match_value: The matched value
            context: Context information
            
        Returns:
            True if match is valid in context
        """
        # Context-specific validation logic can be added here
        # For now, return True to maintain compatibility
        return True


class PatternLibrary:
    """Library of sensitive data patterns with categorization and management."""
    
    def __init__(self):
        """Initialize pattern library with built-in patterns."""
        self.patterns: Dict[str, SensitiveDataPattern] = {}
        self.categories: Dict[str, List[str]] = {}
        self._load_builtin_patterns()
    
    def _load_builtin_patterns(self) -> None:
        """Load built-in sensitive data patterns."""
        
        # PII Patterns
        pii_patterns = [
            SensitiveDataPattern(
                name="email_address",
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                method=RedactionMethod.HASH,
                confidence=0.95,
                description="Email addresses",
                category="pii",
                severity="medium"
            ),
            SensitiveDataPattern(
                name="us_ssn",
                pattern=r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
                method=RedactionMethod.REMOVE,
                confidence=0.90,
                description="US Social Security Numbers",
                category="pii",
                severity="critical",
                validation_func=self._validate_ssn
            ),
            SensitiveDataPattern(
                name="phone_number",
                pattern=r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                method=RedactionMethod.MASK,
                confidence=0.85,
                description="Phone numbers",
                category="pii",
                severity="medium"
            ),
            SensitiveDataPattern(
                name="full_name",
                pattern=r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                method=RedactionMethod.HASH,
                confidence=0.60,
                description="Potential full names",
                category="pii",
                severity="low",
                context_required=True
            )
        ]
        
        # Financial Patterns
        financial_patterns = [
            SensitiveDataPattern(
                name="credit_card",
                pattern=r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
                method=RedactionMethod.REMOVE,
                confidence=0.95,
                description="Credit card numbers",
                category="financial",
                severity="critical",
                validation_func=self._validate_credit_card
            ),
            SensitiveDataPattern(
                name="bank_account",
                pattern=r'\b\d{8,12}\b',
                method=RedactionMethod.REMOVE,
                confidence=0.70,
                description="Bank account numbers",
                category="financial",
                severity="high",
                context_required=True
            ),
            SensitiveDataPattern(
                name="routing_number",
                pattern=r'\b[0-9]{9}\b',
                method=RedactionMethod.REMOVE,
                confidence=0.75,
                description="US bank routing numbers",
                category="financial",
                severity="high",
                context_required=True
            )
        ]
        
        # Technical Patterns
        technical_patterns = [
            SensitiveDataPattern(
                name="ip_address",
                pattern=r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                method=RedactionMethod.HASH,
                confidence=0.90,
                description="IP addresses",
                category="technical",
                severity="low"
            ),
            SensitiveDataPattern(
                name="api_key",
                pattern=r'\b[A-Za-z0-9]{20,}\b',
                method=RedactionMethod.REMOVE,
                confidence=0.60,
                description="Potential API keys",
                category="technical",
                severity="high",
                context_required=True
            ),
            SensitiveDataPattern(
                name="jwt_token",
                pattern=r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b',
                method=RedactionMethod.REMOVE,
                confidence=0.95,
                description="JWT tokens",
                category="technical",
                severity="high"
            )
        ]
        
        # Healthcare Patterns
        healthcare_patterns = [
            SensitiveDataPattern(
                name="medical_record_number",
                pattern=r'\bMRN[\s-]?\d{6,}\b',
                method=RedactionMethod.REMOVE,
                confidence=0.90,
                description="Medical record numbers",
                category="healthcare",
                severity="critical"
            ),
            SensitiveDataPattern(
                name="dea_number",
                pattern=r'\b[A-Z]{2}\d{7}\b',
                method=RedactionMethod.REMOVE,
                confidence=0.85,
                description="DEA numbers",
                category="healthcare",
                severity="critical"
            )
        ]
        
        # Geographic Patterns
        geographic_patterns = [
            SensitiveDataPattern(
                name="us_zip_code",
                pattern=r'\b\d{5}(-\d{4})?\b',
                method=RedactionMethod.TRUNCATE,
                confidence=0.80,
                description="US ZIP codes",
                category="geographic",
                severity="low"
            ),
            SensitiveDataPattern(
                name="coordinates",
                pattern=r'-?\d{1,3}\.\d+,\s*-?\d{1,3}\.\d+',
                method=RedactionMethod.HASH,
                confidence=0.85,
                description="Geographic coordinates",
                category="geographic",
                severity="medium"
            )
        ]
        
        # Load all patterns
        all_patterns = pii_patterns + financial_patterns + technical_patterns + healthcare_patterns + geographic_patterns
        
        for pattern in all_patterns:
            self.add_pattern(pattern)
    
    def add_pattern(self, pattern: SensitiveDataPattern) -> None:
        """Add a pattern to the library.
        
        Args:
            pattern: Pattern to add
        """
        self.patterns[pattern.name] = pattern
        
        # Update categories
        if pattern.category not in self.categories:
            self.categories[pattern.category] = []
        
        if pattern.name not in self.categories[pattern.category]:
            self.categories[pattern.category].append(pattern.name)
    
    def get_patterns_by_category(self, category: str) -> List[SensitiveDataPattern]:
        """Get all patterns in a category.
        
        Args:
            category: Category name
            
        Returns:
            List of patterns in category
        """
        if category not in self.categories:
            return []
        
        return [self.patterns[name] for name in self.categories[category]]
    
    def get_patterns_by_severity(self, severity: str) -> List[SensitiveDataPattern]:
        """Get patterns by severity level.
        
        Args:
            severity: Severity level
            
        Returns:
            List of patterns with specified severity
        """
        return [pattern for pattern in self.patterns.values() if pattern.severity == severity]
    
    def scan_text(self, text: str, categories: List[str] = None, min_confidence: float = 0.0) -> Dict[str, List[Dict[str, Any]]]:
        """Scan text for sensitive data patterns.
        
        Args:
            text: Text to scan
            categories: Categories to scan (None for all)
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary mapping pattern names to matches
        """
        results = {}
        patterns_to_check = self.patterns.values()
        
        # Filter by categories if specified
        if categories:
            patterns_to_check = []
            for category in categories:
                patterns_to_check.extend(self.get_patterns_by_category(category))
        
        # Check each pattern
        for pattern in patterns_to_check:
            if pattern.confidence >= min_confidence:
                matches = pattern.matches(text)
                if matches:
                    results[pattern.name] = matches
        
        return results
    
    @staticmethod
    def _validate_ssn(ssn: str, context: Dict[str, Any]) -> bool:
        """Validate SSN using basic checksum validation.
        
        Args:
            ssn: SSN to validate
            context: Context information
            
        Returns:
            True if valid SSN format
        """
        # Remove formatting
        digits = re.sub(r'[^0-9]', '', ssn)
        if len(digits) != 9:
            return False
        
        # Basic validation rules
        if digits == '000000000':
            return False
        
        if digits[:3] == '000' or digits[3:5] == '00' or digits[5:] == '0000':
            return False
        
        return True
    
    @staticmethod
    def _validate_credit_card(card_number: str, context: Dict[str, Any]) -> bool:
        """Validate credit card using Luhn algorithm.
        
        Args:
            card_number: Credit card number to validate
            context: Context information
            
        Returns:
            True if valid credit card number
        """
        # Remove spaces and dashes
        digits = re.sub(r'[^0-9]', '', card_number)
        
        if len(digits) < 13 or len(digits) > 19:
            return False
        
        # Luhn algorithm
        checksum = 0
        is_even = False
        
        for i in range(len(digits) - 1, -1, -1):
            digit = int(digits[i])
            
            if is_even:
                digit *= 2
                if digit > 9:
                    digit -= 9
            
            checksum += digit
            is_even = not is_even
        
        return checksum % 10 == 0


class AdvancedPatternDetector:
    """Advanced pattern detection with context awareness and machine learning-like features."""
    
    def __init__(self, pattern_library: PatternLibrary = None):
        """Initialize advanced pattern detector.
        
        Args:
            pattern_library: Pattern library to use
        """
        self.pattern_library = pattern_library or PatternLibrary()
        self.detection_stats: Dict[str, int] = {}
        self.false_positive_patterns: Set[str] = set()
        
    def detect_sensitive_data(
        self,
        data: Any,
        path: str = "",
        context: Dict[str, Any] = None,
        categories: List[str] = None,
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Detect sensitive data with advanced analysis.
        
        Args:
            data: Data to analyze
            path: Current path in data structure
            context: Context information
            categories: Categories to check
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detection results
        """
        detections = []
        context = context or {}
        
        if isinstance(data, str):
            detections.extend(self._analyze_string(data, path, context, categories, min_confidence))
        elif isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}/{key}"
                new_context = context.copy()
                new_context['parent_key'] = key
                new_context['sibling_keys'] = list(data.keys())
                
                detections.extend(
                    self.detect_sensitive_data(value, new_path, new_context, categories, min_confidence)
                )
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]"
                detections.extend(
                    self.detect_sensitive_data(item, new_path, context, categories, min_confidence)
                )
        
        return detections
    
    def _analyze_string(
        self,
        text: str,
        path: str,
        context: Dict[str, Any],
        categories: List[str],
        min_confidence: float
    ) -> List[Dict[str, Any]]:
        """Analyze string for sensitive patterns.
        
        Args:
            text: String to analyze
            path: Path to string
            context: Context information
            categories: Categories to check
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detections
        """
        detections = []
        
        # Get pattern matches
        matches = self.pattern_library.scan_text(text, categories, min_confidence)
        
        for pattern_name, pattern_matches in matches.items():
            pattern = self.pattern_library.patterns[pattern_name]
            
            # Apply context-based confidence adjustment
            adjusted_confidence = self._adjust_confidence_by_context(
                pattern, text, context
            )
            
            if adjusted_confidence >= min_confidence:
                for match in pattern_matches:
                    detection = {
                        'path': path,
                        'pattern_name': pattern_name,
                        'pattern_category': pattern.category,
                        'original_confidence': match['confidence'],
                        'adjusted_confidence': adjusted_confidence,
                        'match_value': match['value'],
                        'match_start': match['start'],
                        'match_end': match['end'],
                        'severity': pattern.severity,
                        'recommended_method': pattern.method,
                        'description': pattern.description,
                        'context': context.copy()
                    }
                    detections.append(detection)
                    
                    # Update statistics
                    self.detection_stats[pattern_name] = self.detection_stats.get(pattern_name, 0) + 1
        
        return detections
    
    def _adjust_confidence_by_context(
        self,
        pattern: SensitiveDataPattern,
        text: str,
        context: Dict[str, Any]
    ) -> float:
        """Adjust pattern confidence based on context.
        
        Args:
            pattern: The pattern that matched
            text: The matched text
            context: Context information
            
        Returns:
            Adjusted confidence score
        """
        confidence = pattern.confidence
        
        # Context-based adjustments
        parent_key = context.get('parent_key', '').lower()
        sibling_keys = [key.lower() for key in context.get('sibling_keys', [])]
        
        # Increase confidence for contextually relevant keys
        relevant_keywords = {
            'email': ['email', 'mail', 'address'],
            'phone': ['phone', 'tel', 'mobile', 'number'],
            'ssn': ['ssn', 'social', 'security'],
            'credit_card': ['card', 'credit', 'payment', 'cc'],
            'ip_address': ['ip', 'address', 'host'],
            'name': ['name', 'first', 'last', 'full']
        }
        
        for category, keywords in relevant_keywords.items():
            if pattern.category == category or pattern.name.startswith(category):
                if any(keyword in parent_key for keyword in keywords):
                    confidence = min(1.0, confidence + 0.15)
                elif any(any(keyword in sibling for keyword in keywords) for sibling in sibling_keys):
                    confidence = min(1.0, confidence + 0.10)
        
        # Decrease confidence for likely false positives
        if pattern.name in self.false_positive_patterns:
            confidence *= 0.8
        
        # Adjust based on text length and context
        if pattern.name == 'full_name' and len(text.split()) > 3:
            confidence *= 0.7  # Longer "names" less likely to be actual names
        
        if pattern.name == 'api_key' and not any(keyword in parent_key for keyword in ['key', 'token', 'secret']):
            confidence *= 0.6  # API keys should be in relevant fields
        
        return confidence
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection statistics and insights.
        
        Returns:
            Dictionary with detection statistics
        """
        total_detections = sum(self.detection_stats.values())
        
        return {
            'total_detections': total_detections,
            'patterns_detected': len(self.detection_stats),
            'top_patterns': sorted(
                self.detection_stats.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'categories_active': len(set(
                self.pattern_library.patterns[name].category
                for name in self.detection_stats.keys()
            )),
            'false_positive_patterns': list(self.false_positive_patterns)
        }
    
    def mark_false_positive(self, pattern_name: str) -> None:
        """Mark a pattern as prone to false positives.
        
        Args:
            pattern_name: Name of pattern to mark
        """
        self.false_positive_patterns.add(pattern_name)


class Redactor:
    """Redacts sensitive data from dataLayer snapshots."""
    
    def __init__(self, config: RedactionConfig | None = None):
        """Initialize redactor with configuration.
        
        Args:
            config: Redaction configuration
        """
        self.config = config or RedactionConfig()
        self.path_matcher = PathMatcher()
        self.audit_trail: List[RedactionAuditEntry] = []
        
        # Initialize advanced pattern detection
        self.pattern_library = PatternLibrary()
        self.advanced_detector = AdvancedPatternDetector(self.pattern_library)
        
        # Compile regex patterns for performance
        self._compiled_patterns = {}
        if self.config.pattern_detection:
            for name, pattern in self.config.patterns.items():
                try:
                    self._compiled_patterns[name] = re.compile(pattern)
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{name}': {e}")
    
    @validate_types()
    def redact_data(
        self,
        data: Dict[str, Any],
        additional_rules: List[RedactionRuleConfig | None] = None
    ) -> Tuple[Dict[str, Any], List[RedactionAuditEntry]]:
        """Redact sensitive data from a dictionary.
        
        Args:
            data: Data to redact
            additional_rules: Additional redaction rules to apply
            
        Returns:
            Tuple of (redacted_data, audit_trail)
        """
        if not self.config.enabled:
            return data, []
        
        logger.debug("Starting data redaction")
        
        # Clear previous audit trail
        self.audit_trail.clear()
        
        # Combine all redaction rules
        all_rules = list(self.config.rules)
        if additional_rules:
            all_rules.extend(additional_rules)
        
        # Create a deep copy for redaction
        redacted_data = self._deep_copy(data)
        
        # Apply path-based redaction rules
        for rule in all_rules:
            self._apply_redaction_rule(redacted_data, rule)
        
        # Apply pattern-based redaction if enabled
        if self.config.pattern_detection:
            self._apply_pattern_redaction(redacted_data)
        
        logger.debug(f"Redaction complete: {len(self.audit_trail)} items redacted")
        
        # Verify redaction was successful
        verification_issues = self._verify_redaction(data, redacted_data, self.audit_trail)
        if verification_issues:
            logger.error(f"Redaction verification failed: {len(verification_issues)} issues found")
            for issue in verification_issues:
                logger.error(f"Redaction issue: {issue}")
            raise RedactionError(f"Redaction verification failed: {verification_issues}")
        
        return redacted_data, list(self.audit_trail)
    
    def _apply_redaction_rule(
        self,
        data: Dict[str, Any],
        rule: RedactionRuleConfig,
        current_path: str = ""
    ) -> None:
        """Apply a single redaction rule to data.
        
        Args:
            data: Data to modify
            rule: Redaction rule to apply
            current_path: Current JSON Pointer path
        """
        if isinstance(data, dict):
            # Check keys to avoid modification during iteration
            keys_to_process = list(data.keys())
            
            for key in keys_to_process:
                key_path = f"{current_path}/{key}"
                
                # Check if this path matches the rule
                if self.path_matcher.matches_json_pointer(key_path, rule.path):
                    original_value = data[key]
                    original_type = type(original_value).__name__
                    
                    # Apply redaction
                    redacted_value = self._apply_redaction_method(
                        original_value, rule.method
                    )
                    data[key] = redacted_value
                    
                    # Add to audit trail
                    self.audit_trail.append(RedactionAuditEntry(
                        path=key_path,
                        original_type=original_type,
                        method=rule.method,
                        reason=rule.reason
                    ))
                    
                    logger.debug(f"Redacted {key_path} using {rule.method}")
                
                # Recursively process nested objects
                elif isinstance(data[key], (dict, list)):
                    self._apply_redaction_rule(data[key], rule, key_path)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                item_path = f"{current_path}/{i}"
                if isinstance(item, (dict, list)):
                    self._apply_redaction_rule(item, rule, item_path)
    
    def _apply_pattern_redaction(
        self,
        data: Dict[str, Any],
        current_path: str = ""
    ) -> None:
        """Apply pattern-based redaction to detect sensitive data.
        
        Args:
            data: Data to scan and redact
            current_path: Current JSON Pointer path
        """
        if isinstance(data, dict):
            keys_to_process = list(data.keys())
            
            for key in keys_to_process:
                key_path = f"{current_path}/{key}"
                value = data[key]
                
                # Check string values against patterns
                if isinstance(value, str):
                    # Use advanced pattern detector with context
                    context = {
                        'field_name': key,
                        'path': key_path,
                        'parent_data': data,
                        'full_data': self._get_root_data(data, current_path)
                    }
                    
                    # First try advanced pattern detection
                    detections = self.advanced_detector.detect_patterns(value, context)
                    
                    if detections:
                        # Use the highest confidence detection
                        best_detection = max(detections, key=lambda d: d.confidence)
                        
                        if best_detection.confidence >= self.config.confidence_threshold:
                            original_type = type(value).__name__
                            
                            # Apply pattern-specific redaction method
                            redacted_value = self._apply_redaction_method(
                                value, best_detection.pattern.method
                            )
                            data[key] = redacted_value
                            
                            # Add to audit trail with enhanced information
                            self.audit_trail.append(RedactionAuditEntry(
                                path=key_path,
                                original_type=original_type,
                                method=best_detection.pattern.method,
                                reason=f"Advanced pattern detection: {best_detection.pattern.name} "
                                       f"(confidence: {best_detection.confidence:.2f}, "
                                       f"category: {best_detection.pattern.category})",
                                pattern_matched=best_detection.pattern.name
                            ))
                            
                            logger.debug(
                                f"Advanced pattern redacted {key_path}: "
                                f"{best_detection.pattern.name} "
                                f"(confidence: {best_detection.confidence:.2f})"
                            )
                            continue
                    
                    # Fall back to legacy pattern matching if no advanced detection
                    for pattern_name, compiled_pattern in self._compiled_patterns.items():
                        if compiled_pattern.search(value):
                            original_type = type(value).__name__
                            
                            # Apply default redaction method
                            redacted_value = self._apply_redaction_method(
                                value, self.config.default_method
                            )
                            data[key] = redacted_value
                            
                            # Add to audit trail
                            self.audit_trail.append(RedactionAuditEntry(
                                path=key_path,
                                original_type=original_type,
                                method=self.config.default_method,
                                reason=f"Legacy pattern detection: {pattern_name}",
                                pattern_matched=pattern_name
                            ))
                            
                            logger.debug(f"Legacy pattern redacted {key_path}: {pattern_name}")
                            break  # Only apply first matching pattern
                
                # Recursively process nested structures
                elif isinstance(value, (dict, list)):
                    self._apply_pattern_redaction(value, key_path)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                item_path = f"{current_path}/{i}"
                if isinstance(item, (dict, list, str)):
                    self._apply_pattern_redaction(item, item_path)
    
    def _get_root_data(self, current_data: Any, current_path: str) -> Dict[str, Any]:
        """Get the root data structure for context.
        
        Args:
            current_data: Current data being processed
            current_path: Current JSON Pointer path
            
        Returns:
            Root data structure or empty dict if not available
        """
        # For now, return the current data as we don't have a reference to root
        # In a full implementation, this could traverse back to root
        if isinstance(current_data, dict):
            return current_data
        return {}
    
    def _apply_redaction_method(
        self,
        value: Any,
        method: RedactionMethod
    ) -> Any:
        """Apply specific redaction method to a value.
        
        Args:
            value: Value to redact
            method: Redaction method to apply
            
        Returns:
            Redacted value
        """
        if method == RedactionMethod.REMOVE:
            return "[REDACTED]"
        
        elif method == RedactionMethod.HASH:
            # Convert to string and hash
            str_value = str(value)
            hash_obj = hashlib.sha256(str_value.encode())
            return f"[HASH:{hash_obj.hexdigest()[:16]}]"
        
        elif method == RedactionMethod.MASK:
            # Mask with asterisks, preserving length for short values
            str_value = str(value)
            if len(str_value) <= 3:
                return "*" * len(str_value)
            else:
                # Show first and last character for longer values
                return f"{str_value[0]}{'*' * (len(str_value) - 2)}{str_value[-1]}"
        
        elif method == RedactionMethod.TRUNCATE:
            # Keep first 3 characters
            str_value = str(value)
            if len(str_value) <= 3:
                return str_value
            else:
                return f"{str_value[:3]}..."
        
        else:
            logger.warning(f"Unknown redaction method: {method}")
            return "[REDACTED]"
    
    def _deep_copy(self, obj: Any) -> Any:
        """Create a deep copy of an object for redaction.
        
        Args:
            obj: Object to copy
            
        Returns:
            Deep copy of object
        """
        if isinstance(obj, dict):
            return {key: self._deep_copy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj
    
    def validate_redaction_rules(
        self,
        rules: List[RedactionRuleConfig]
    ) -> List[str]:
        """Validate redaction rules for correctness.
        
        Args:
            rules: Rules to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        for i, rule in enumerate(rules):
            # Validate path format
            if not rule.path:
                errors.append(f"Rule {i}: Empty path")
                continue
            
            # Validate JSON Pointer format if not using glob
            if not ('*' in rule.path or '**' in rule.path):
                if not rule.path.startswith('/'):
                    errors.append(f"Rule {i}: JSON Pointer path must start with '/'")
            
            # Validate glob patterns
            if '*' in rule.path or '**' in rule.path:
                try:
                    # Test pattern compilation
                    self.path_matcher._glob_to_regex(rule.path)
                except Exception as e:
                    errors.append(f"Rule {i}: Invalid glob pattern - {e}")
            
            # Validate method
            if rule.method not in RedactionMethod:
                errors.append(f"Rule {i}: Invalid redaction method '{rule.method}'")
        
        return errors
    
    def get_redaction_statistics(self) -> Dict[str, Any]:
        """Get statistics about the last redaction operation.
        
        Returns:
            Dictionary with redaction statistics
        """
        if not self.audit_trail:
            return {
                'total_redacted': 0,
                'methods_used': {},
                'patterns_matched': {},
                'paths_affected': []
            }
        
        methods_count = {}
        patterns_count = {}
        paths = []
        
        for entry in self.audit_trail:
            # Count methods
            method_str = str(entry.method)
            methods_count[method_str] = methods_count.get(method_str, 0) + 1
            
            # Count patterns
            if entry.pattern_matched:
                patterns_count[entry.pattern_matched] = patterns_count.get(entry.pattern_matched, 0) + 1
            
            # Collect paths
            paths.append(entry.path)
        
        return {
            'total_redacted': len(self.audit_trail),
            'methods_used': methods_count,
            'patterns_matched': patterns_count,
            'paths_affected': paths,
            'unique_paths': len(set(paths))
        }
    
    def _verify_redaction(
        self,
        original_data: Dict[str, Any],
        redacted_data: Dict[str, Any],
        audit_trail: List[RedactionAuditEntry]
    ) -> List[str]:
        """Verify that redaction was successful and no sensitive data leaked.
        
        Args:
            original_data: Original data before redaction
            redacted_data: Data after redaction
            audit_trail: Audit trail of redaction operations
            
        Returns:
            List of verification issues found (empty if verification passes)
        """
        issues = []
        
        # Check that redacted fields don't contain original sensitive values
        for entry in audit_trail:
            path_parts = entry.path.strip('/').split('/')
            
            try:
                # Get original value
                original_value = self._get_nested_value(original_data, path_parts)
                redacted_value = self._get_nested_value(redacted_data, path_parts)
                
                if original_value is None or redacted_value is None:
                    continue
                
                # Verify redaction based on method
                if entry.method == RedactionMethod.REMOVE:
                    # Value should be completely removed or replaced with placeholder
                    if redacted_value == original_value:
                        issues.append(f"REMOVE redaction failed at {entry.path}: value unchanged")
                
                elif entry.method == RedactionMethod.HASH:
                    # Value should be hashed (completely different)
                    if redacted_value == original_value:
                        issues.append(f"HASH redaction failed at {entry.path}: value unchanged")
                    elif not isinstance(redacted_value, str) or len(redacted_value) != 64:
                        issues.append(f"HASH redaction failed at {entry.path}: invalid hash format")
                
                elif entry.method == RedactionMethod.MASK:
                    # Value should be masked with asterisks
                    if redacted_value == original_value:
                        issues.append(f"MASK redaction failed at {entry.path}: value unchanged")
                    elif not str(redacted_value).startswith('*'):
                        issues.append(f"MASK redaction failed at {entry.path}: not properly masked")
                
                elif entry.method == RedactionMethod.TRUNCATE:
                    # Value should be truncated
                    if redacted_value == original_value:
                        issues.append(f"TRUNCATE redaction failed at {entry.path}: value unchanged")
                    elif len(str(redacted_value)) >= len(str(original_value)):
                        issues.append(f"TRUNCATE redaction failed at {entry.path}: not truncated")
                        
            except Exception as e:
                issues.append(f"Verification error at {entry.path}: {e}")
        
        # Check for sensitive patterns in redacted data
        if self.config.pattern_detection:
            pattern_issues = self._verify_patterns_removed(redacted_data)
            issues.extend(pattern_issues)
        
        return issues
    
    def _verify_patterns_removed(self, data: Dict[str, Any]) -> List[str]:
        """Verify that sensitive patterns are removed from redacted data.
        
        Args:
            data: Redacted data to check
            
        Returns:
            List of pattern verification issues
        """
        issues = []
        
        for pattern_name, pattern in self._compiled_patterns.items():
            found_matches = self._find_pattern_matches(data, pattern, pattern_name)
            if found_matches:
                for path, value in found_matches:
                    issues.append(f"Sensitive pattern '{pattern_name}' still present at {path}: {value}")
        
        return issues
    
    def _find_pattern_matches(
        self,
        data: Any,
        pattern: re.Pattern,
        pattern_name: str,
        current_path: str = ""
    ) -> List[Tuple[str, str]]:
        """Find matches of sensitive patterns in data.
        
        Args:
            data: Data to search
            pattern: Compiled regex pattern
            pattern_name: Name of the pattern
            current_path: Current path in the data structure
            
        Returns:
            List of (path, value) tuples where pattern was found
        """
        matches = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                key_path = f"{current_path}/{key}"
                
                # Check the value
                if isinstance(value, str) and pattern.search(value):
                    matches.append((key_path, value))
                elif isinstance(value, (dict, list)):
                    matches.extend(self._find_pattern_matches(value, pattern, pattern_name, key_path))
                    
        elif isinstance(data, list):
            for i, item in enumerate(data):
                item_path = f"{current_path}/{i}"
                
                if isinstance(item, str) and pattern.search(item):
                    matches.append((item_path, item))
                elif isinstance(item, (dict, list)):
                    matches.extend(self._find_pattern_matches(item, pattern, pattern_name, item_path))
        
        elif isinstance(data, str) and pattern.search(data):
            matches.append((current_path, data))
        
        return matches
    
    def _get_nested_value(self, data: Dict[str, Any], path_parts: List[str]) -> Any:
        """Get value from nested data structure using path parts.
        
        Args:
            data: Data structure to navigate
            path_parts: List of keys/indices to navigate
            
        Returns:
            Value at the specified path, or None if not found
        """
        current = data
        
        try:
            for part in path_parts:
                if isinstance(current, dict):
                    current = current[part]
                elif isinstance(current, list):
                    current = current[int(part)]
                else:
                    return None
            return current
        except (KeyError, IndexError, ValueError, TypeError):
            return None
    
    def get_pattern_detection_stats(self) -> Dict[str, Any]:
        """Get pattern detection statistics.
        
        Returns:
            Dictionary with detection statistics
        """
        return self.advanced_detector.get_statistics()
    
    def add_custom_pattern(self, pattern: SensitiveDataPattern) -> None:
        """Add a custom pattern to the pattern library.
        
        Args:
            pattern: Pattern to add
        """
        self.pattern_library.add_pattern(pattern)
    
    def remove_pattern(self, pattern_name: str) -> bool:
        """Remove a pattern from the pattern library.
        
        Args:
            pattern_name: Name of pattern to remove
            
        Returns:
            True if pattern was removed, False if not found
        """
        return self.pattern_library.remove_pattern(pattern_name)
    
    def get_available_patterns(self) -> List[str]:
        """Get list of available pattern names.
        
        Returns:
            List of pattern names
        """
        return list(self.pattern_library.patterns.keys())


class RedactionManager:
    """Manages redaction operations with caching and optimization."""
    
    def __init__(self, config: RedactionConfig | None = None):
        """Initialize redaction manager.
        
        Args:
            config: Redaction configuration
        """
        self.config = config or RedactionConfig()
        self._redactor = Redactor(config)
        self._rule_cache: Dict[str, List[RedactionRuleConfig]] = {}
        self._audit_history: List[RedactionAuditEntry] = []
    
    def redact_snapshot_data(
        self,
        latest_data: Dict[str, Any],
        events_data: List[Dict[str, Any]],
        site_domain: str | None = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[RedactionAuditEntry]]:
        """Redact data from a complete dataLayer snapshot.
        
        Args:
            latest_data: Latest state data to redact
            events_data: Events data to redact
            site_domain: Site domain for site-specific rules
            
        Returns:
            Tuple of (redacted_latest, redacted_events, audit_trail)
        """
        all_audit_entries = []
        
        # Get site-specific rules
        additional_rules = self._get_site_rules(site_domain)
        
        # Redact latest state data
        redacted_latest, latest_audit = self._redactor.redact_data(
            latest_data, additional_rules
        )
        all_audit_entries.extend(latest_audit)
        
        # Redact events data
        redacted_events = []
        for i, event in enumerate(events_data):
            redacted_event, event_audit = self._redactor.redact_data(
                event, additional_rules
            )
            redacted_events.append(redacted_event)
            
            # Adjust paths for event context
            for audit_entry in event_audit:
                audit_entry.path = f"/events/{i}{audit_entry.path}"
            all_audit_entries.extend(event_audit)
        
        # Store in audit history if configured
        if self.config.keep_audit_trail:
            self._audit_history.extend(all_audit_entries)
        
        return redacted_latest, redacted_events, all_audit_entries
    
    def _get_site_rules(self, site_domain: str | None) -> List[RedactionRuleConfig]:
        """Get site-specific redaction rules.
        
        Args:
            site_domain: Site domain to get rules for
            
        Returns:
            List of additional rules for the site
        """
        if not site_domain:
            return []
        
        # Check cache first
        if site_domain in self._rule_cache:
            return self._rule_cache[site_domain]
        
        # Load site-specific rules (placeholder for future implementation)
        # This would load from configuration or database
        site_rules = []
        
        # Cache the rules
        self._rule_cache[site_domain] = site_rules
        
        return site_rules
    
    def cleanup_audit_history(self, retention_days: int | None = None) -> int:
        """Clean up old audit history entries.
        
        Args:
            retention_days: Days to retain (uses config default if None)
            
        Returns:
            Number of entries removed
        """
        if not retention_days:
            retention_days = self.config.audit_trail_retention_days
        
        if not self._audit_history:
            return 0
        
        cutoff_date = datetime.utcnow().replace(
            day=datetime.utcnow().day - retention_days
        )
        
        original_count = len(self._audit_history)
        self._audit_history = [
            entry for entry in self._audit_history
            if entry.timestamp > cutoff_date
        ]
        
        removed_count = original_count - len(self._audit_history)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old audit entries")
        
        return removed_count
    
    def export_audit_trail(
        self,
        output_path: Path,
        format: str = 'json'
    ) -> None:
        """Export audit trail to file.
        
        Args:
            output_path: Path to write audit trail
            format: Export format ('json' or 'csv')
        """
        import json
        
        audit_data = [entry.to_dict() for entry in self._audit_history]
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(audit_data, f, indent=2, default=str)
        elif format.lower() == 'csv':
            import csv
            
            if audit_data:
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=audit_data[0].keys())
                    writer.writeheader()
                    writer.writerows(audit_data)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(audit_data)} audit entries to {output_path}")
    
    def get_redaction_summary(self) -> Dict[str, Any]:
        """Get comprehensive redaction summary.
        
        Returns:
            Summary of all redaction operations
        """
        return {
            'config': {
                'enabled': self.config.enabled,
                'default_method': str(self.config.default_method),
                'pattern_detection': self.config.pattern_detection,
                'rules_count': len(self.config.rules),
                'patterns_count': len(self.config.patterns)
            },
            'audit_history': {
                'total_entries': len(self._audit_history),
                'retention_days': self.config.audit_trail_retention_days
            },
            'last_operation': self._redactor.get_redaction_statistics()
        }