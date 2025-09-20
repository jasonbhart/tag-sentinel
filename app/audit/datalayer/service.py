"""Main DataLayer service orchestrating all dataLayer operations.

This module provides the primary service interface that coordinates snapshot capture,
redaction, validation, and aggregation operations for dataLayer integrity auditing.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path

from playwright.async_api import Page

from .models import (
    DataLayerSnapshot, DLContext, DLResult, ValidationIssue, 
    DLAggregate, ValidationSeverity
)
from .runtime_validation import validate_types, validate_dl_context, validate_datalayer_snapshot
from .config_validation import ConfigValidator, ConfigValidationError
from .error_handling import (
    DataLayerErrorHandler, resilient_operation, graceful_degradation,
    ComponentType, ErrorSeverity, ErrorContext, DataLayerError,
    global_error_handler
)

# Error tracking models
class ServiceError:
    """Represents a service-level error with context."""
    
    def __init__(
        self,
        error_type: str,
        message: str,
        page_url: str | None = None,
        component: str | None = None,
        exception: Exception | None = None,
        timestamp: datetime | None = None
    ):
        self.error_type = error_type
        self.message = message
        self.page_url = page_url
        self.component = component
        self.exception = exception
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_type': self.error_type,
            'message': self.message,
            'page_url': self.page_url,
            'component': self.component,
            'exception_type': type(self.exception).__name__ if self.exception else None,
            'timestamp': self.timestamp.isoformat()
        }
from .config import (
    DataLayerConfig, get_datalayer_config, get_site_datalayer_config
)
from .snapshot import Snapshotter, SnapshotError
from .redaction import RedactionManager, RedactionError
from .validation import Validator, SchemaManager, ValidationError

logger = logging.getLogger(__name__)


class DataLayerServiceError(Exception):
    """Errors in DataLayer service operations."""
    pass


class DataLayerService:
    """Main service orchestrating all dataLayer integrity operations."""
    
    def __init__(self, config: DataLayerConfig | None = None, validate_config: bool = True):
        """Initialize DataLayer service.
        
        Args:
            config: DataLayer configuration (uses global config if None)
            validate_config: Whether to validate configuration on initialization
        """
        self.config = config or get_datalayer_config()
        
        # Validate configuration if requested
        if validate_config:
            self._validate_configuration()
        
        # Initialize error handling
        self.error_handler = DataLayerErrorHandler()
        self._register_error_callbacks()
        
        # Initialize components
        self.snapshotter = Snapshotter(self.config.capture)
        self.redaction_manager = RedactionManager(self.config.redaction)
        self.validator = Validator(self.config.validation)
        self.schema_manager = SchemaManager(self.config.validation)
        
        # Initialize aggregation
        self.aggregator: DLAggregate | None = None
        self._processing_stats = {
            'total_processed': 0,
            'successful_captures': 0,
            'failed_captures': 0,
            'total_issues': 0,
            'start_time': None
        }
        
        # Error tracking
        self._errors: List[ServiceError] = []
        self._error_counts: Dict[str, int] = {}
        self._max_errors = 1000  # Limit error history to prevent memory issues
    
    def _register_error_callbacks(self) -> None:
        """Register error callbacks for different components."""
        # Register callback for capture errors
        self.error_handler.register_error_callback(
            ComponentType.CAPTURE, 
            self._handle_capture_error
        )
        
        # Register callback for validation errors
        self.error_handler.register_error_callback(
            ComponentType.VALIDATION,
            self._handle_validation_error
        )
        
        # Register callback for redaction errors
        self.error_handler.register_error_callback(
            ComponentType.REDACTION,
            self._handle_redaction_error
        )
    
    def _handle_capture_error(self, error: DataLayerError) -> None:
        """Handle capture-specific errors."""
        logger.error(f"Capture error for {error.context.page_url}: {error.message}")
        
        # Update internal error tracking
        error_key = f"capture:{error.error_type}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        
        # Add to legacy error tracking
        service_error = ServiceError(
            error_type=error.error_type,
            message=error.message,
            page_url=error.context.page_url,
            component="capture",
            exception=error.exception,
            timestamp=error.context.timestamp
        )
        
        self._add_error(service_error)
    
    def _handle_validation_error(self, error: DataLayerError) -> None:
        """Handle validation-specific errors."""
        logger.warning(f"Validation error for {error.context.page_url}: {error.message}")
        
        # Update internal error tracking
        error_key = f"validation:{error.error_type}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
    
    def _handle_redaction_error(self, error: DataLayerError) -> None:
        """Handle redaction-specific errors."""
        logger.warning(f"Redaction error for {error.context.page_url}: {error.message}")
        
        # Update internal error tracking  
        error_key = f"redaction:{error.error_type}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
    
    @resilient_operation(
        component=ComponentType.CAPTURE,
        operation="capture_and_validate",
        max_retries=2,
        retry_delay=1.0,
        fallback_result=None
    )
    async def capture_and_validate(
        self,
        page: Page,
        context_or_page_url: DLContext | str | None = None,
        schema_or_site_domain: dict | str | None = None,
        site_domain_or_run_id: str | None = None,
        run_id: str | None = None,
        # Backward compatibility parameters
        page_url: str | None = None,
        site_domain: str | None = None
    ) -> DLResult:
        """Capture and validate dataLayer from a single page.

        Args:
            page: Playwright page instance
            context_or_page_url: Either DLContext for new style calls or page URL string for legacy calls
            schema_or_site_domain: Schema dict (if DLContext provided) or site domain (if page_url provided)
            site_domain_or_run_id: Site domain (if schema provided) or run_id (if no schema)
            run_id: Run identifier for aggregation (if site_domain provided)

        Returns:
            Complete dataLayer result with snapshot, validation, and metadata
        """
        start_time = datetime.utcnow()
        processing_start = datetime.now()

        # Handle both calling conventions with backward compatibility
        if isinstance(context_or_page_url, DLContext):
            # New style: capture_and_validate(page, context, schema)
            provided_context = context_or_page_url
            schema = schema_or_site_domain if isinstance(schema_or_site_domain, dict) else None
            resolved_site_domain = site_domain_or_run_id if isinstance(site_domain_or_run_id, str) else None
            resolved_page_url = page.url  # DLContext doesn't have url field
        else:
            # Legacy style or explicit keyword arguments
            provided_context = None

            # Check for explicit keyword arguments first (backward compatibility)
            if page_url is not None:
                resolved_page_url = page_url
            else:
                resolved_page_url = context_or_page_url or page.url

            if site_domain is not None:
                resolved_site_domain = site_domain
                # When using explicit site_domain keyword, schema comes from second positional or is None
                schema = schema_or_site_domain if isinstance(schema_or_site_domain, dict) else None
            else:
                # Original positional logic
                if isinstance(schema_or_site_domain, dict):
                    # Third parameter is a schema dict
                    schema = schema_or_site_domain
                    resolved_site_domain = site_domain_or_run_id if isinstance(site_domain_or_run_id, str) else None
                else:
                    # Third parameter is site_domain string (legacy style)
                    schema = None
                    resolved_site_domain = schema_or_site_domain if isinstance(schema_or_site_domain, str) else None
                    if not run_id:
                        run_id = site_domain_or_run_id

        logger.debug(f"Starting dataLayer capture and validation for {resolved_page_url}")

        # Get site-specific configuration if needed
        if resolved_site_domain:
            site_config = get_site_datalayer_config(resolved_site_domain)
        else:
            site_config = self.config

        # Create or merge context for this capture
        if provided_context:
            # Start from provided context and only update missing values with defaults
            context = provided_context.model_copy()
            # Fill in schema_path from site config if not provided
            if not context.schema_path:
                context.schema_path = site_config.validation.schema_path
        else:
            # Legacy path: create context from site config
            context = DLContext(
                env=site_config.environment,
                data_layer_object=site_config.capture.object_name,
                max_depth=site_config.capture.max_depth,
                max_entries=site_config.capture.max_entries,
                max_size_bytes=site_config.capture.max_size_bytes,
                schema_path=site_config.validation.schema_path
            )
        
        result = DLResult(
            snapshot=DataLayerSnapshot(
                page_url=resolved_page_url,
                exists=False,
                object_name=context.data_layer_object
            ),
            issues=[],
            processing_time_ms=0.0
        )
        
        try:
            # Step 1: Capture dataLayer snapshot with graceful degradation
            with graceful_degradation(
                fallback_value=DataLayerSnapshot(page_url=resolved_page_url, exists=False, object_name=context.data_layer_object),
                operation_name="capture_snapshot",
                component=ComponentType.CAPTURE,
                error_handler=self.error_handler
            ) as snapshot:
                snapshot = await self._capture_snapshot(page, context, resolved_page_url)
                result.snapshot = snapshot
            
            if not result.snapshot.exists:
                result.add_note("DataLayer object not found on page")
                logger.debug(f"No dataLayer found on {resolved_page_url}")
            else:
                logger.debug(f"DataLayer captured: {result.snapshot.variable_count} variables, {result.snapshot.event_count} events")

                # Step 2: Perform schema validation if enabled with graceful degradation (BEFORE redaction)
                if site_config.validation.enabled and (schema or context.schema_path):
                    with graceful_degradation(
                        fallback_value=None,
                        operation_name="perform_validation",
                        component=ComponentType.VALIDATION,
                        error_handler=self.error_handler
                    ):
                        if schema:
                            # Use inline schema dict
                            await self._perform_validation_with_schema(result, schema, resolved_page_url)
                        else:
                            # Use schema from path
                            await self._perform_validation(result, context.schema_path, resolved_page_url)

                # Step 3: Apply redaction if enabled with graceful degradation (AFTER validation)
                if site_config.redaction.enabled and (result.snapshot.latest or result.snapshot.events):
                    with graceful_degradation(
                        fallback_value=None,
                        operation_name="apply_redaction",
                        component=ComponentType.REDACTION,
                        error_handler=self.error_handler
                    ):
                        await self._apply_redaction(result, resolved_site_domain)

            # Calculate processing time first
            processing_end = datetime.now()
            processing_time = (processing_end - processing_start).total_seconds() * 1000
            result.processing_time_ms = processing_time

            # Step 4: Update aggregation if running (moved outside if/else to count all page attempts)
            if self.aggregator:
                self._update_aggregation(result, run_id)
            
            # Update statistics
            self._update_processing_stats(result)
            
            logger.debug(f"DataLayer processing complete for {resolved_page_url} in {processing_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"DataLayer processing failed for {resolved_page_url}: {e}")
            
            # Track the error
            error = ServiceError(
                error_type="processing_failure",
                message=str(e),
                page_url=resolved_page_url,
                component="capture_and_validate",
                exception=e
            )
            self._track_error(error)
            
            result.capture_error = str(e)
            
            # Still update stats for failed captures
            self._update_processing_stats(result)
            
            if self.config.global_settings.get('fail_fast', False):
                raise DataLayerServiceError(f"DataLayer processing failed: {e}") from e
            
            return result
    
    @validate_types()
    @resilient_operation(
        component=ComponentType.AGGREGATION,
        operation="process_multiple_pages",
        max_retries=1,
        retry_delay=0.5,
        fallback_result=[]
    )
    async def process_multiple_pages(
        self,
        page_contexts: List[Tuple[Page, str, str | None]],  # (page, url, site_domain)
        run_id: str | None = None,
        progress_callback: Callable | None = None
    ) -> Tuple[List[DLResult], DLAggregate]:
        """Process multiple pages and generate run-level aggregation.
        
        Args:
            page_contexts: List of (page, url, site_domain) tuples
            run_id: Run identifier for aggregation
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (results_list, aggregation_summary)
        """
        logger.info(f"Starting batch processing of {len(page_contexts)} pages")
        
        # Initialize aggregation for this run
        self.start_aggregation(run_id)
        
        results = []
        
        # Process pages with concurrency control
        semaphore = asyncio.Semaphore(self.config.performance.max_concurrent_captures)
        
        async def _process_single_page(page_context, index):
            async with semaphore:
                page, url, site_domain = page_context
                try:
                    result = await self.capture_and_validate(page, url, site_domain, run_id)
                    
                    if progress_callback:
                        await progress_callback(index + 1, len(page_contexts), result)
                    
                    return result
                except Exception as e:
                    logger.error(f"Failed to process page {url}: {e}")
                    # Return minimal result on error
                    error_result = DLResult(
                        snapshot=DataLayerSnapshot(page_url=url, exists=False),
                        capture_error=str(e)
                    )

                    # Update aggregation to count failed pages
                    if self.aggregator:
                        self._update_aggregation(error_result, run_id)

                    return error_result
        
        # Execute all captures concurrently
        tasks = [
            _process_single_page(page_context, i)
            for i, page_context in enumerate(page_contexts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Finalize aggregation
        aggregation = self.finalize_aggregation()
        
        logger.info(f"Batch processing complete: {len(results)} pages processed")
        return results, aggregation
    
    async def _capture_snapshot(
        self,
        page: Page,
        context: DLContext,
        page_url: str
    ) -> DataLayerSnapshot:
        """Capture dataLayer snapshot from page.
        
        Args:
            page: Playwright page instance
            context: Capture context
            page_url: Page URL
            
        Returns:
            DataLayer snapshot
        """
        try:
            snapshot = await self.snapshotter.take_snapshot(page, context, page_url)
            return snapshot
        except SnapshotError as e:
            logger.warning(f"Snapshot capture failed for {page_url}: {e}")
            # Return empty snapshot
            return DataLayerSnapshot(
                page_url=page_url,
                exists=False,
                object_name=context.data_layer_object
            )
    
    async def _apply_redaction(
        self,
        result: DLResult,
        site_domain: str | None
    ) -> None:
        """Apply redaction to captured data.
        
        Args:
            result: Result object to modify
            site_domain: Site domain for site-specific rules
        """
        try:
            snapshot = result.snapshot
            
            if snapshot.latest or snapshot.events:
                redacted_latest, redacted_events, audit_trail = (
                    self.redaction_manager.redact_snapshot_data(
                        snapshot.latest or {},
                        snapshot.events,
                        site_domain
                    )
                )
                
                # Update snapshot with redacted data
                snapshot.latest = redacted_latest
                snapshot.events = redacted_events
                
                # Track redaction in snapshot metadata
                if audit_trail:
                    snapshot.redacted_paths = [entry.path for entry in audit_trail]
                    snapshot.redaction_method_used = self.config.redaction.default_method
                    
                    result.add_note(f"Applied redaction to {len(audit_trail)} fields")
                    logger.debug(f"Redacted {len(audit_trail)} fields from snapshot")
        
        except Exception as e:
            logger.warning(f"Redaction failed: {e}")
            result.add_note(f"Redaction error: {e}")
            self._track_error(ServiceError(
                error_type="redaction_failure",
                message=str(e),
                page_url=result.snapshot.page_url,
                component="redaction",
                exception=e
            ))
    
    async def _perform_validation(
        self,
        result: DLResult,
        schema_path: str,
        page_url: str
    ) -> None:
        """Perform schema validation on captured data.

        Args:
            result: Result object to update with validation issues
            schema_path: Path to schema file
            page_url: Page URL for validation context
        """
        try:
            snapshot = result.snapshot

            if snapshot.latest or snapshot.events:
                issues = self.validator.validate_snapshot(
                    snapshot.latest or {},
                    snapshot.events,
                    schema_path,
                    page_url
                )

                result.issues.extend(issues)

                # Add summary note about validation results (maintain consistency with inline schema validation)
                if issues:
                    critical_count = len([i for i in issues if i.severity == ValidationSeverity.CRITICAL])
                    warning_count = len([i for i in issues if i.severity == ValidationSeverity.WARNING])

                    result.add_note(
                        f"Schema validation found {len(issues)} issues "
                        f"({critical_count} critical, {warning_count} warnings)"
                    )
                else:
                    result.add_note("Schema validation passed")

                logger.debug(f"Schema validation completed for {page_url}: {len(issues)} issues found")
        except Exception as e:
            logger.error(f"Schema validation failed for {page_url}: {e}")
            result.validation_error = str(e)
            result.add_note(f"Validation error: {e}")

    async def _perform_validation_with_schema(
        self,
        result: DLResult,
        schema_dict: dict,
        page_url: str
    ) -> None:
        """Perform schema validation on captured data using inline schema.

        Args:
            result: Result object to update with validation issues
            schema_dict: Schema dictionary for validation
            page_url: Page URL for validation context
        """
        try:
            snapshot = result.snapshot

            if snapshot.latest or snapshot.events:
                all_issues = []

                # Validate latest data using validate_data with proper page_url
                if snapshot.latest:
                    latest_issues = self.validator.validate_data(snapshot.latest, schema_dict, page_url)
                    all_issues.extend(latest_issues)
                    result.issues.extend(latest_issues)

                # Validate events if present, mirroring validate_snapshot behavior
                if snapshot.events:
                    for i, event in enumerate(snapshot.events):
                        event_issues = self.validator.validate_data(event, schema_dict, page_url)

                        # Adjust paths to indicate event context (mirror validate_snapshot logic)
                        for issue in event_issues:
                            issue.path = f"/events/{i}{issue.path}"
                            issue.event_type = event.get('event') or event.get('eventName')

                        all_issues.extend(event_issues)
                        result.issues.extend(event_issues)

                if all_issues:
                    critical_count = len([i for i in all_issues if i.severity == ValidationSeverity.CRITICAL])
                    warning_count = len([i for i in all_issues if i.severity == ValidationSeverity.WARNING])

                    result.add_note(
                        f"Schema validation found {len(all_issues)} issues "
                        f"({critical_count} critical, {warning_count} warnings)"
                    )

                    logger.debug(f"Schema validation found {len(all_issues)} issues for {page_url}")
                else:
                    result.add_note("Schema validation passed")

        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            result.validation_error = str(e)
            result.add_note(f"Validation error: {e}")
            self._track_error(ServiceError(
                error_type="validation_failure",
                message=str(e),
                page_url=page_url,
                component="validation",
                exception=e
            ))
    
    def _update_aggregation(
        self,
        result: DLResult,
        run_id: str | None
    ) -> None:
        """Update run-level aggregation with page result.
        
        Args:
            result: Page result to aggregate
            run_id: Run identifier
        """
        if not self.aggregator:
            return
        
        try:
            self.aggregator.add_page_result(result)
            
            # Update aggregate delta for this result
            result.aggregate_delta = {
                'variables_found': list(result.snapshot.get_variable_names()),
                'events_found': result.snapshot.get_event_types(),
                'validation_issues': len(result.issues),
                'processing_time': result.processing_time_ms
            }
            
        except Exception as e:
            logger.warning(f"Aggregation update failed: {e}")
    
    def _update_processing_stats(self, result: DLResult) -> None:
        """Update internal processing statistics.
        
        Args:
            result: Processing result
        """
        self._processing_stats['total_processed'] += 1
        
        if result.is_successful:
            self._processing_stats['successful_captures'] += 1
        else:
            self._processing_stats['failed_captures'] += 1
        
        self._processing_stats['total_issues'] += len(result.issues)
        
        if not self._processing_stats['start_time']:
            self._processing_stats['start_time'] = datetime.utcnow()
    
    def start_aggregation(self, run_id: str | None = None) -> None:
        """Start a new aggregation run.
        
        Args:
            run_id: Optional run identifier
        """
        self.aggregator = DLAggregate(
            run_id=run_id,
            start_time=datetime.utcnow()
        )
        
        # Reset processing stats
        self._processing_stats = {
            'total_processed': 0,
            'successful_captures': 0,
            'failed_captures': 0,
            'total_issues': 0,
            'start_time': datetime.utcnow()
        }
        
        logger.debug(f"Started new aggregation run: {run_id}")
    
    def finalize_aggregation(self) -> DLAggregate | None:
        """Finalize and return current aggregation.
        
        Returns:
            Finalized aggregation or None if no aggregation active
        """
        if not self.aggregator:
            return None
        
        self.aggregator.finalize()
        
        final_aggregation = self.aggregator
        self.aggregator = None  # Reset for next run
        
        logger.debug(f"Finalized aggregation: {final_aggregation.total_pages} pages")
        return final_aggregation
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary statistics.
        
        Returns:
            Processing summary
        """
        stats = self._processing_stats.copy()
        
        if stats['start_time']:
            duration = (datetime.utcnow() - stats['start_time']).total_seconds()
            stats['duration_seconds'] = duration
            
            if stats['total_processed'] > 0:
                stats['pages_per_second'] = stats['total_processed'] / duration
                stats['success_rate'] = (stats['successful_captures'] / stats['total_processed']) * 100
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform service health check.
        
        Returns:
            Health status information
        """
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Check component health
        try:
            # Snapshotter health
            health['components']['snapshotter'] = {
                'status': 'healthy',
                'config': {
                    'enabled': self.config.capture.enabled,
                    'safe_mode': self.config.capture.safe_mode
                }
            }
            
            # Redaction health
            health['components']['redaction'] = {
                'status': 'healthy',
                'config': {
                    'enabled': self.config.redaction.enabled,
                    'rules_count': len(self.config.redaction.rules)
                }
            }
            
            # Validation health
            health['components']['validation'] = {
                'status': 'healthy',
                'config': {
                    'enabled': self.config.validation.enabled,
                    'cache_enabled': self.config.validation.cache_schemas
                }
            }
            
            # Overall processing stats
            health['processing'] = self.get_processing_summary()
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health['status'] = 'degraded'
            health['error'] = str(e)
        
        return health
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        try:
            # Cleanup redaction audit history
            if self.config.redaction.keep_audit_trail:
                removed = self.redaction_manager.cleanup_audit_history()
                if removed > 0:
                    logger.info(f"Cleaned up {removed} old redaction audit entries")
            
            # Clear schema caches
            self.schema_manager.clear_schemas()
            
            logger.debug("DataLayer service cleanup complete")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
            self._track_error(ServiceError(
                error_type="cleanup_failure",
                message=str(e),
                component="cleanup",
                exception=e
            ))
    
    def _validate_configuration(self) -> None:
        """Validate the service configuration and log any issues.
        
        Raises:
            DataLayerServiceError: If configuration is invalid
        """
        logger.info("Validating DataLayer service configuration")
        
        try:
            validator = ConfigValidator(strict_mode=self.config.is_production)
            is_valid, error_messages = validator.validate_config(self.config)
            
            if not is_valid:
                error_summary = f"Configuration validation failed with {len(error_messages)} errors: " + \
                              "; ".join(error_messages[:3])  # Show first 3 errors
                if len(error_messages) > 3:
                    error_summary += f" (and {len(error_messages) - 3} more errors)"
                
                logger.error(error_summary)
                
                # In production, fail fast on config errors
                if self.config.is_production:
                    raise DataLayerServiceError(f"Configuration validation failed: {error_summary}")
                else:
                    logger.warning("Continuing with invalid configuration in non-production environment")
            else:
                logger.info("Configuration validation passed")
                
        except Exception as e:
            error_msg = f"Configuration validation error: {e}"
            logger.error(error_msg)
            
            if self.config.is_production:
                raise DataLayerServiceError(error_msg) from e
    
    def validate_config_interactively(self) -> Dict[str, Any]:
        """Validate configuration and return detailed results for troubleshooting.
        
        Returns:
            Dictionary with validation results, errors, and warnings
        """
        validator = ConfigValidator(strict_mode=False)  # Always non-strict for interactive use
        is_valid, error_messages = validator.validate_config(self.config)
        
        return {
            'is_valid': is_valid,
            'total_errors': len(validator.errors),
            'total_warnings': len(validator.warnings),
            'errors': [
                {
                    'component': error.component,
                    'field': error.field,
                    'message': error.message,
                    'value': str(error.value) if error.value is not None else None
                }
                for error in validator.errors
            ],
            'warnings': [
                {
                    'component': warning.component,
                    'field': warning.field,
                    'message': warning.message,
                    'value': str(warning.value) if warning.value is not None else None
                }
                for warning in validator.warnings
            ],
            'summary': f"Configuration validation {'passed' if is_valid else 'failed'} "
                      f"with {len(validator.errors)} errors and {len(validator.warnings)} warnings"
        }
    
    def _track_error(self, error: ServiceError) -> None:
        """Track a service error with automatic cleanup.
        
        Args:
            error: ServiceError to track
        """
        # Add to error list with size limit
        self._errors.append(error)
        if len(self._errors) > self._max_errors:
            # Remove oldest errors
            self._errors = self._errors[-self._max_errors:]
        
        # Update error counts
        error_key = f"{error.component}:{error.error_type}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        
        logger.debug(f"Tracked error: {error_key} (total: {self._error_counts[error_key]})")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary.
        
        Returns:
            Error statistics and recent errors
        """
        recent_errors = []
        for error in self._errors[-10:]:  # Last 10 errors
            recent_errors.append(error.to_dict())
        
        return {
            'total_errors': len(self._errors),
            'error_counts': dict(self._error_counts),
            'recent_errors': recent_errors,
            'most_common_errors': sorted(
                self._error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def clear_error_history(self) -> int:
        """Clear error history.
        
        Returns:
            Number of errors cleared
        """
        cleared_count = len(self._errors)
        self._errors.clear()
        self._error_counts.clear()
        logger.info(f"Cleared error history: {cleared_count} errors")
        return cleared_count
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the dataLayer service.
        
        Returns:
            Health status information including error rates, component status, and recommendations
        """
        # Get error handler statistics
        error_stats = self.error_handler.get_error_statistics()
        
        # Calculate component health
        component_health = {}
        for component in ComponentType:
            component_name = component.value
            error_count = sum(
                count for key, count in self._error_counts.items()
                if key.startswith(f"{component_name}:")
            )
            
            if error_count == 0:
                component_health[component_name] = "healthy"
            elif error_count < 5:
                component_health[component_name] = "warning"
            elif error_count < 20:
                component_health[component_name] = "degraded"
            else:
                component_health[component_name] = "critical"
        
        # Overall health calculation
        critical_components = sum(1 for status in component_health.values() if status == "critical")
        degraded_components = sum(1 for status in component_health.values() if status == "degraded")
        
        if critical_components > 0:
            overall_health = "critical"
        elif degraded_components > 1:
            overall_health = "degraded"
        elif degraded_components == 1 or sum(1 for status in component_health.values() if status == "warning") > 2:
            overall_health = "warning"
        else:
            overall_health = "healthy"
        
        return {
            "overall_health": overall_health,
            "component_health": component_health,
            "error_statistics": error_stats,
            "processing_statistics": self._processing_stats,
            "resilience_status": {
                "error_handler_active": self.error_handler is not None,
                "total_errors_handled": len(self.error_handler.error_history),
                "recovery_success_rate": self._calculate_recovery_success_rate()
            },
            "recommendations": self._generate_health_recommendations(
                overall_health, component_health, error_stats
            )
        }
    
    def get_resilience_metrics(self) -> Dict[str, Any]:
        """Get detailed resilience and error handling metrics.
        
        Returns:
            Comprehensive resilience metrics
        """
        return {
            "error_handling": {
                "total_errors": len(self._errors),
                "error_counts_by_type": self._error_counts.copy(),
                "recent_errors": [
                    error.to_dict() for error in self._errors[-10:]
                ]
            },
            "graceful_degradation": {
                "degraded_operations": self._count_degraded_operations(),
                "fallback_activations": self._count_fallback_activations()
            },
            "recovery_statistics": self.error_handler.get_error_statistics()
        }
    
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate recovery success rate from error handler."""
        if not self.error_handler.error_history:
            return 100.0
        
        # Simplified calculation - in real implementation would track recovery attempts
        total_errors = len(self.error_handler.error_history)
        critical_errors = sum(
            1 for error in self.error_handler.error_history 
            if error.severity == ErrorSeverity.CRITICAL
        )
        
        # Assume most non-critical errors were handled gracefully
        successful_recoveries = total_errors - critical_errors
        return (successful_recoveries / total_errors * 100) if total_errors > 0 else 100.0
    
    def _generate_health_recommendations(
        self,
        overall_health: str,
        component_health: Dict[str, str],
        error_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate health-based recommendations."""
        recommendations = []
        
        if overall_health == "critical":
            recommendations.append("URGENT: System in critical state - investigate immediately")
        elif overall_health == "degraded":
            recommendations.append("System degraded - review error patterns and consider scaling back operations")
        
        # Component-specific recommendations
        for component, health in component_health.items():
            if health == "critical":
                recommendations.append(f"CRITICAL: {component} component failing - disable temporarily")
            elif health == "degraded":
                recommendations.append(f"Review {component} component configuration and error patterns")
        
        # Error pattern recommendations
        if error_stats.get("recent_errors_24h", 0) > 50:
            recommendations.append("High error rate detected - consider reducing load or investigating root causes")
        
        return recommendations
    
    def _count_degraded_operations(self) -> int:
        """Count operations that required graceful degradation."""
        # Simplified implementation - would track actual degradation events in production
        return sum(1 for error in self._errors if "degraded" in error.message.lower())
    
    def _count_fallback_activations(self) -> int:
        """Count fallback mechanism activations."""
        # Simplified implementation - would track actual fallback events in production
        return sum(1 for error in self._errors if "fallback" in error.message.lower())


# Convenience functions for common operations

async def capture_page_datalayer(
    page: Page,
    page_url: str | None = None,
    schema_path: str | None = None,
    site_domain: str | None = None
) -> DLResult:
    """Convenience function to capture and validate a single page's dataLayer.
    
    Args:
        page: Playwright page instance
        page_url: Override page URL
        schema_path: Path to validation schema
        site_domain: Site domain for configuration
        
    Returns:
        DataLayer processing result
    """
    service = DataLayerService()
    
    # Override schema path if provided
    if schema_path:
        service.config.validation.schema_path = schema_path
    
    return await service.capture_and_validate(page, page_url, site_domain)


def create_dataLayer_service(
    config_path: Path | None = None,
    site_domain: str | None = None
) -> DataLayerService:
    """Create a configured DataLayer service.
    
    Args:
        config_path: Path to configuration file
        site_domain: Site domain for site-specific configuration
        
    Returns:
        Configured DataLayer service
    """
    if config_path:
        from .config import load_datalayer_config
        config = load_datalayer_config(config_path)
    elif site_domain:
        config = get_site_datalayer_config(site_domain)
    else:
        config = get_datalayer_config()
    
    return DataLayerService(config)