"""Repository pattern implementations for Tag Sentinel API.

This module provides data access layer abstractions using the repository pattern,
supporting both in-memory and persistent storage backends.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import json
import asyncio

from app.api.schemas.requests import ListAuditsRequest
from app.api.schemas.responses import AuditStatus
from app.api.schemas.exports import RequestLogExport, CookieExport, TagExport, DataLayerExport
from .models import PersistentAuditRecord, ExportMetadata

logger = logging.getLogger(__name__)


class AuditRepository(ABC):
    """Abstract base class for audit data repositories."""

    @abstractmethod
    async def create_audit(self, audit: PersistentAuditRecord) -> None:
        """Create a new audit record."""
        pass

    @abstractmethod
    async def get_audit(self, audit_id: str) -> Optional[PersistentAuditRecord]:
        """Get audit by ID."""
        pass

    @abstractmethod
    async def update_audit(self, audit: PersistentAuditRecord) -> None:
        """Update an existing audit record."""
        pass

    @abstractmethod
    async def delete_audit(self, audit_id: str) -> bool:
        """Delete an audit record."""
        pass

    @abstractmethod
    async def list_audits(
        self,
        request: ListAuditsRequest,
        limit: int = 20,
        offset: int = 0
    ) -> tuple[List[PersistentAuditRecord], int]:
        """List audits with filtering and pagination."""
        pass

    @abstractmethod
    async def find_by_idempotency_key(
        self,
        idempotency_key: str,
        max_age_hours: int = 24
    ) -> Optional[PersistentAuditRecord]:
        """Find audit by idempotency key within time window."""
        pass

    @abstractmethod
    async def get_audit_count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Get total count of audits matching filters."""
        pass


class InMemoryAuditRepository(AuditRepository):
    """In-memory implementation of audit repository for testing and development."""

    def __init__(self):
        """Initialize in-memory storage."""
        self._audits: Dict[str, PersistentAuditRecord] = {}
        self._lock = asyncio.Lock()
        logger.info("InMemoryAuditRepository initialized")

    async def create_audit(self, audit: PersistentAuditRecord) -> None:
        """Create a new audit record."""
        async with self._lock:
            if audit.id in self._audits:
                raise ValueError(f"Audit with ID {audit.id} already exists")

            self._audits[audit.id] = audit
            logger.debug(f"Created audit {audit.id} in memory repository")

    async def get_audit(self, audit_id: str) -> Optional[PersistentAuditRecord]:
        """Get audit by ID."""
        async with self._lock:
            audit = self._audits.get(audit_id)
            if audit:
                logger.debug(f"Retrieved audit {audit_id} from memory repository")
            return audit

    async def update_audit(self, audit: PersistentAuditRecord) -> None:
        """Update an existing audit record."""
        async with self._lock:
            if audit.id not in self._audits:
                raise ValueError(f"Audit with ID {audit.id} does not exist")

            audit.updated_at = datetime.now(timezone.utc)
            self._audits[audit.id] = audit
            logger.debug(f"Updated audit {audit.id} in memory repository")

    async def delete_audit(self, audit_id: str) -> bool:
        """Delete an audit record."""
        async with self._lock:
            if audit_id in self._audits:
                del self._audits[audit_id]
                logger.debug(f"Deleted audit {audit_id} from memory repository")
                return True
            return False

    async def list_audits(
        self,
        request: ListAuditsRequest,
        limit: int = 20,
        offset: int = 0
    ) -> tuple[List[PersistentAuditRecord], int]:
        """List audits with filtering and pagination."""
        async with self._lock:
            # Apply filters
            filtered_audits = []
            for audit in self._audits.values():
                if self._matches_filters(audit, request):
                    filtered_audits.append(audit)

            # Sort audits
            sorted_audits = self._sort_audits(filtered_audits, request.sort_by, request.sort_order)

            # Apply pagination
            total_count = len(sorted_audits)
            paginated_audits = sorted_audits[offset:offset + limit]

            logger.debug(
                f"Listed {len(paginated_audits)} audits from memory repository "
                f"(total: {total_count}, offset: {offset}, limit: {limit})"
            )

            return paginated_audits, total_count

    async def find_by_idempotency_key(
        self,
        idempotency_key: str,
        max_age_hours: int = 24
    ) -> Optional[PersistentAuditRecord]:
        """Find audit by idempotency key within time window."""
        async with self._lock:
            cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)

            for audit in self._audits.values():
                if (audit.idempotency_key == idempotency_key and
                    audit.created_at.timestamp() > cutoff_time):
                    logger.debug(f"Found audit {audit.id} by idempotency key {idempotency_key}")
                    return audit

            return None

    async def get_audit_count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Get total count of audits matching filters."""
        async with self._lock:
            if not filters:
                return len(self._audits)

            # Create a minimal request object for filtering
            request = ListAuditsRequest(**filters)
            count = 0
            for audit in self._audits.values():
                if self._matches_filters(audit, request):
                    count += 1

            return count

    def _matches_filters(self, audit: PersistentAuditRecord, request: ListAuditsRequest) -> bool:
        """Check if audit matches the provided filters."""
        # Site ID filter
        if request.site_id and audit.site_id != request.site_id:
            return False

        # Environment filter
        if request.env and audit.env != request.env:
            return False

        # Status filter
        if request.status and audit.status not in request.status:
            return False

        # Date range filters (both dates should be timezone-aware at this point)
        if request.date_from and audit.created_at < request.date_from:
            return False
        if request.date_to and audit.created_at > request.date_to:
            return False

        # Tags filter
        if request.tags:
            audit_tags = audit.metadata.get('tags') if audit.metadata else None
            if not audit_tags:
                return False

            # Use the same iterable handling logic as the service layer
            from collections.abc import Iterable
            if not isinstance(audit_tags, Iterable) or isinstance(audit_tags, (str, bytes)):
                return False

            audit_tags_set = set(audit_tags)
            request_tags_set = set(request.tags)
            if not request_tags_set.intersection(audit_tags_set):
                return False

        # Search filter (simple text search across metadata)
        if request.search:
            search_text = request.search.lower()
            searchable_text = f"{audit.site_id} {audit.env} {audit.id}"
            if audit.metadata:
                searchable_text += f" {json.dumps(audit.metadata).lower()}"
            if search_text not in searchable_text.lower():
                return False

        return True

    def _sort_audits(
        self,
        audits: List[PersistentAuditRecord],
        sort_by: str,
        sort_order: str
    ) -> List[PersistentAuditRecord]:
        """Sort audits by the specified field and order."""
        reverse = sort_order == "desc"

        if sort_by == "created_at":
            return sorted(audits, key=lambda a: a.created_at, reverse=reverse)
        elif sort_by == "updated_at":
            return sorted(audits, key=lambda a: a.updated_at, reverse=reverse)
        elif sort_by == "site_id":
            return sorted(audits, key=lambda a: a.site_id, reverse=reverse)
        elif sort_by == "status":
            return sorted(audits, key=lambda a: a.status.value, reverse=reverse)
        elif sort_by == "duration":
            def get_duration(audit):
                if audit.started_at and audit.finished_at:
                    return (audit.finished_at - audit.started_at).total_seconds()
                return 0
            return sorted(audits, key=get_duration, reverse=reverse)
        else:
            # Default to created_at
            return sorted(audits, key=lambda a: a.created_at, reverse=reverse)


class ExportDataRepository(ABC):
    """Abstract base class for export data repositories."""

    @abstractmethod
    async def get_request_logs(
        self,
        audit_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RequestLogExport]:
        """Get request log data for an audit."""
        pass

    @abstractmethod
    async def get_cookies(
        self,
        audit_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[CookieExport]:
        """Get cookie data for an audit."""
        pass

    @abstractmethod
    async def get_tags(
        self,
        audit_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[TagExport]:
        """Get tag detection data for an audit."""
        pass

    @abstractmethod
    async def get_data_layer_snapshots(
        self,
        audit_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DataLayerExport]:
        """Get data layer snapshot data for an audit."""
        pass

    @abstractmethod
    async def store_export_metadata(self, metadata: ExportMetadata) -> None:
        """Store export generation metadata."""
        pass

    @abstractmethod
    async def get_export_metadata(
        self,
        audit_id: str,
        export_type: str,
        format: str
    ) -> Optional[ExportMetadata]:
        """Get export metadata."""
        pass


class InMemoryExportDataRepository(ExportDataRepository):
    """In-memory implementation of export data repository."""

    def __init__(self):
        """Initialize in-memory storage."""
        self._request_logs: Dict[str, List[RequestLogExport]] = {}
        self._cookies: Dict[str, List[CookieExport]] = {}
        self._tags: Dict[str, List[TagExport]] = {}
        self._data_layer_snapshots: Dict[str, List[DataLayerExport]] = {}
        self._export_metadata: Dict[str, ExportMetadata] = {}
        self._lock = asyncio.Lock()
        logger.info("InMemoryExportDataRepository initialized")

    async def get_request_logs(
        self,
        audit_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RequestLogExport]:
        """Get request log data for an audit."""
        async with self._lock:
            logs = self._request_logs.get(audit_id, [])
            if filters:
                logs = self._filter_request_logs(logs, filters)
            return logs

    async def get_cookies(
        self,
        audit_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[CookieExport]:
        """Get cookie data for an audit."""
        async with self._lock:
            cookies = self._cookies.get(audit_id, [])
            if filters:
                cookies = self._filter_cookies(cookies, filters)
            return cookies

    async def get_tags(
        self,
        audit_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[TagExport]:
        """Get tag detection data for an audit."""
        async with self._lock:
            tags = self._tags.get(audit_id, [])
            if filters:
                tags = self._filter_tags(tags, filters)
            return tags

    async def get_data_layer_snapshots(
        self,
        audit_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DataLayerExport]:
        """Get data layer snapshot data for an audit."""
        async with self._lock:
            snapshots = self._data_layer_snapshots.get(audit_id, [])
            if filters:
                snapshots = self._filter_data_layer_snapshots(snapshots, filters)
            return snapshots

    async def store_export_metadata(self, metadata: ExportMetadata) -> None:
        """Store export generation metadata."""
        async with self._lock:
            key = f"{metadata.audit_id}:{metadata.export_type}:{metadata.format}"
            self._export_metadata[key] = metadata

    async def get_export_metadata(
        self,
        audit_id: str,
        export_type: str,
        format: str
    ) -> Optional[ExportMetadata]:
        """Get export metadata."""
        async with self._lock:
            key = f"{audit_id}:{export_type}:{format}"
            return self._export_metadata.get(key)

    def _filter_request_logs(
        self,
        logs: List[RequestLogExport],
        filters: Dict[str, Any]
    ) -> List[RequestLogExport]:
        """Apply filters to request logs."""
        filtered = logs
        if filters.get("status"):
            filtered = [log for log in filtered if log.status == filters["status"]]
        if filters.get("resource_type"):
            filtered = [log for log in filtered if log.resource_type == filters["resource_type"]]
        if filters.get("analytics_only"):
            filtered = [log for log in filtered if log.is_analytics]
        return filtered

    def _filter_cookies(
        self,
        cookies: List[CookieExport],
        filters: Dict[str, Any]
    ) -> List[CookieExport]:
        """Apply filters to cookies."""
        filtered = cookies
        if filters.get("category"):
            filtered = [c for c in filtered if c.category == filters["category"]]
        if filters.get("vendor"):
            filtered = [c for c in filtered if c.vendor == filters["vendor"]]
        if filters.get("essential_only"):
            filtered = [c for c in filtered if c.is_essential]
        if filters.get("privacy_compliant"):
            # Filter for cookies that have GDPR compliance info and don't require consent
            filtered = [c for c in filtered if c.gdpr_compliance and not c.gdpr_compliance.get("requires_consent", True)]
        return filtered

    def _filter_tags(
        self,
        tags: List[TagExport],
        filters: Dict[str, Any]
    ) -> List[TagExport]:
        """Apply filters to tags."""
        filtered = tags
        if filters.get("vendor"):
            filtered = [t for t in filtered if t.vendor == filters["vendor"]]
        if filters.get("tag_type"):
            filtered = [t for t in filtered if t.tag_type == filters["tag_type"]]
        if filters.get("implementation_method"):
            filtered = [t for t in filtered if t.implementation_method == filters["implementation_method"]]
        if filters.get("has_errors"):
            filtered = [t for t in filtered if t.has_errors]
        return filtered

    def _filter_data_layer_snapshots(
        self,
        snapshots: List[DataLayerExport],
        filters: Dict[str, Any]
    ) -> List[DataLayerExport]:
        """Apply filters to data layer snapshots."""
        filtered = snapshots
        if filters.get("trigger_event"):
            filtered = [s for s in filtered if s.trigger_event == filters["trigger_event"]]
        if filters.get("contains_pii"):
            filtered = [s for s in filtered if s.contains_pii]
        if filters.get("schema_valid") is not None:
            filtered = [s for s in filtered if s.schema_valid == filters["schema_valid"]]
        if filters.get("min_properties") is not None:
            filtered = [s for s in filtered if s.total_properties >= filters["min_properties"]]
        return filtered