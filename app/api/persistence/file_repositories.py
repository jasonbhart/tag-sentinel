"""File-based repository implementations for Tag Sentinel API.

This module provides persistent storage implementations using JSON files
for development and small-scale deployments.
"""

import logging
import json
import os
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid

from app.api.schemas.requests import ListAuditsRequest
from app.api.schemas.exports import RequestLogExport, CookieExport, TagExport, DataLayerExport
from .repositories import AuditRepository, ExportDataRepository
from .models import PersistentAuditRecord, ExportMetadata

logger = logging.getLogger(__name__)


class FileBasedAuditRepository(AuditRepository):
    """File-based implementation of audit repository using JSON files."""

    def __init__(self, storage_path: str = "./data/audits"):
        """Initialize file-based storage.

        Args:
            storage_path: Directory path for storing audit files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info(f"FileBasedAuditRepository initialized with storage_path={storage_path}")

    def _get_audit_file_path(self, audit_id: str) -> Path:
        """Get file path for an audit."""
        return self.storage_path / f"{audit_id}.json"

    def _get_index_file_path(self) -> Path:
        """Get path for the audit index file."""
        return self.storage_path / "index.json"

    async def _load_audit_index(self) -> Dict[str, Dict[str, Any]]:
        """Load audit index from file."""
        index_path = self._get_index_file_path()
        if not index_path.exists():
            return {}

        try:
            async with aiofiles.open(index_path, 'r') as f:
                content = await f.read()
                return json.loads(content) if content.strip() else {}
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load audit index: {e}")
            return {}

    async def _save_audit_index(self, index: Dict[str, Dict[str, Any]]) -> None:
        """Save audit index to file."""
        index_path = self._get_index_file_path()
        try:
            async with aiofiles.open(index_path, 'w') as f:
                await f.write(json.dumps(index, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save audit index: {e}")
            raise

    async def _load_audit_from_file(self, audit_id: str) -> Optional[PersistentAuditRecord]:
        """Load audit data from file."""
        file_path = self._get_audit_file_path(audit_id)
        if not file_path.exists():
            return None

        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                data = json.loads(content)
                return PersistentAuditRecord.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load audit {audit_id}: {e}")
            return None

    async def _save_audit_to_file(self, audit: PersistentAuditRecord) -> None:
        """Save audit data to file."""
        file_path = self._get_audit_file_path(audit.id)
        try:
            async with aiofiles.open(file_path, 'w') as f:
                data = audit.to_dict()
                await f.write(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save audit {audit.id}: {e}")
            raise

    async def _update_index_entry(self, audit: PersistentAuditRecord) -> None:
        """Update audit entry in index."""
        index = await self._load_audit_index()
        index[audit.id] = {
            "site_id": audit.site_id,
            "env": audit.env,
            "status": audit.status.value,
            "created_at": audit.created_at.isoformat(),
            "updated_at": audit.updated_at.isoformat(),
            "idempotency_key": audit.idempotency_key,
            "metadata": audit.metadata
        }
        await self._save_audit_index(index)

    async def create_audit(self, audit: PersistentAuditRecord) -> None:
        """Create a new audit record."""
        async with self._lock:
            file_path = self._get_audit_file_path(audit.id)
            if file_path.exists():
                raise ValueError(f"Audit with ID {audit.id} already exists")

            await self._save_audit_to_file(audit)
            await self._update_index_entry(audit)
            logger.info(f"Created audit {audit.id} in file repository")

    async def get_audit(self, audit_id: str) -> Optional[PersistentAuditRecord]:
        """Get audit by ID."""
        async with self._lock:
            audit = await self._load_audit_from_file(audit_id)
            if audit:
                logger.debug(f"Retrieved audit {audit_id} from file repository")
            return audit

    async def update_audit(self, audit: PersistentAuditRecord) -> None:
        """Update an existing audit record."""
        async with self._lock:
            file_path = self._get_audit_file_path(audit.id)
            if not file_path.exists():
                raise ValueError(f"Audit with ID {audit.id} does not exist")

            audit.updated_at = datetime.utcnow()
            await self._save_audit_to_file(audit)
            await self._update_index_entry(audit)
            logger.info(f"Updated audit {audit.id} in file repository")

    async def delete_audit(self, audit_id: str) -> bool:
        """Delete an audit record."""
        async with self._lock:
            file_path = self._get_audit_file_path(audit_id)
            if not file_path.exists():
                return False

            try:
                file_path.unlink()

                # Remove from index
                index = await self._load_audit_index()
                if audit_id in index:
                    del index[audit_id]
                    await self._save_audit_index(index)

                logger.info(f"Deleted audit {audit_id} from file repository")
                return True
            except Exception as e:
                logger.error(f"Failed to delete audit {audit_id}: {e}")
                return False

    async def list_audits(
        self,
        request: ListAuditsRequest,
        limit: int = 20,
        offset: int = 0
    ) -> tuple[List[PersistentAuditRecord], int]:
        """List audits with filtering and pagination."""
        async with self._lock:
            index = await self._load_audit_index()

            # Filter audits based on index for performance
            matching_ids = []
            for audit_id, index_data in index.items():
                if self._matches_index_filters(index_data, request):
                    matching_ids.append(audit_id)

            # Apply date range filters (requires full audit data)
            if request.date_from or request.date_to:
                filtered_ids = []
                for audit_id in matching_ids:
                    audit = await self._load_audit_from_file(audit_id)
                    if audit and self._matches_date_filters(audit, request):
                        filtered_ids.append(audit_id)
                matching_ids = filtered_ids

            # Sort by loading minimal data
            sorted_ids = await self._sort_audit_ids(matching_ids, request.sort_by, request.sort_order)

            # Apply pagination
            total_count = len(sorted_ids)
            paginated_ids = sorted_ids[offset:offset + limit]

            # Load full audit data for results
            audits = []
            for audit_id in paginated_ids:
                audit = await self._load_audit_from_file(audit_id)
                if audit:
                    audits.append(audit)

            logger.debug(
                f"Listed {len(audits)} audits from file repository "
                f"(total: {total_count}, offset: {offset}, limit: {limit})"
            )

            return audits, total_count

    async def find_by_idempotency_key(
        self,
        idempotency_key: str,
        max_age_hours: int = 24
    ) -> Optional[PersistentAuditRecord]:
        """Find audit by idempotency key within time window."""
        async with self._lock:
            index = await self._load_audit_index()
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

            for audit_id, index_data in index.items():
                if (index_data.get("idempotency_key") == idempotency_key and
                    datetime.fromisoformat(index_data["created_at"]) > cutoff_time):

                    audit = await self._load_audit_from_file(audit_id)
                    if audit:
                        logger.debug(f"Found audit {audit.id} by idempotency key {idempotency_key}")
                        return audit

            return None

    async def get_audit_count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Get total count of audits matching filters."""
        async with self._lock:
            if not filters:
                index = await self._load_audit_index()
                return len(index)

            # Use list_audits with high limit to get count
            request = ListAuditsRequest(**filters)
            _, total_count = await self.list_audits(request, limit=10000, offset=0)
            return total_count

    def _matches_index_filters(self, index_data: Dict[str, Any], request: ListAuditsRequest) -> bool:
        """Check if audit matches filters using index data only."""
        # Site ID filter
        if request.site_id and index_data.get("site_id") != request.site_id:
            return False

        # Environment filter
        if request.env and index_data.get("env") != request.env:
            return False

        # Status filter
        if request.status and index_data.get("status") not in request.status:
            return False

        # Tags filter
        if request.tags:
            metadata = index_data.get("metadata") or {}
            audit_tags = metadata.get("tags") or []

            from collections.abc import Iterable
            if not isinstance(audit_tags, Iterable) or isinstance(audit_tags, (str, bytes)):
                return False

            if not any(tag in audit_tags for tag in request.tags):
                return False

        # Search filter
        if request.search:
            search_text = request.search.lower()
            searchable = f"{index_data.get('site_id', '')} {index_data.get('env', '')} {index_data.get('audit_id', '')}"
            if index_data.get("metadata"):
                searchable += f" {json.dumps(index_data['metadata']).lower()}"
            if search_text not in searchable.lower():
                return False

        return True

    def _matches_date_filters(self, audit: PersistentAuditRecord, request: ListAuditsRequest) -> bool:
        """Check if audit matches date filters."""
        # Date range filters
        if request.date_from and audit.created_at < request.date_from:
            return False
        if request.date_to and audit.created_at > request.date_to:
            return False
        return True

    async def _sort_audit_ids(
        self,
        audit_ids: List[str],
        sort_by: str,
        sort_order: str
    ) -> List[str]:
        """Sort audit IDs based on criteria."""
        if not audit_ids:
            return audit_ids

        reverse = sort_order == "desc"

        if sort_by in ["created_at", "updated_at", "site_id", "status"]:
            # Can sort using index data
            index = await self._load_audit_index()

            def sort_key(audit_id: str):
                data = index.get(audit_id, {})
                if sort_by in ["created_at", "updated_at"]:
                    return datetime.fromisoformat(data.get(sort_by, "1970-01-01T00:00:00"))
                else:
                    return data.get(sort_by, "")

            return sorted(audit_ids, key=sort_key, reverse=reverse)

        elif sort_by == "duration":
            # Needs full audit data
            audit_durations = []
            for audit_id in audit_ids:
                audit = await self._load_audit_from_file(audit_id)
                if audit and audit.started_at and audit.finished_at:
                    duration = (audit.finished_at - audit.started_at).total_seconds()
                else:
                    duration = 0
                audit_durations.append((audit_id, duration))

            audit_durations.sort(key=lambda x: x[1], reverse=reverse)
            return [audit_id for audit_id, _ in audit_durations]

        else:
            # Default to created_at
            return await self._sort_audit_ids(audit_ids, "created_at", sort_order)


class FileBasedExportDataRepository(ExportDataRepository):
    """File-based implementation of export data repository."""

    def __init__(self, storage_path: str = "./data/exports"):
        """Initialize file-based export storage.

        Args:
            storage_path: Directory path for storing export data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info(f"FileBasedExportDataRepository initialized with storage_path={storage_path}")

    def _get_export_file_path(self, audit_id: str, export_type: str) -> Path:
        """Get file path for export data."""
        return self.storage_path / audit_id / f"{export_type}.json"

    def _get_metadata_file_path(self, audit_id: str) -> Path:
        """Get file path for export metadata."""
        return self.storage_path / audit_id / "metadata.json"

    async def _load_export_data(self, audit_id: str, export_type: str, model_class) -> List:
        """Load export data from file."""
        file_path = self._get_export_file_path(audit_id, export_type)
        if not file_path.exists():
            return []

        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                data = json.loads(content) if content.strip() else []
                return [model_class.model_validate(item) for item in data]
        except Exception as e:
            logger.error(f"Failed to load {export_type} data for audit {audit_id}: {e}")
            return []

    async def _save_export_data(self, audit_id: str, export_type: str, data: List) -> None:
        """Save export data to file."""
        file_path = self._get_export_file_path(audit_id, export_type)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            serialized_data = [item.model_dump(mode='json') for item in data]
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(serialized_data, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save {export_type} data for audit {audit_id}: {e}")
            raise

    async def get_request_logs(
        self,
        audit_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RequestLogExport]:
        """Get request log data for an audit."""
        async with self._lock:
            logs = await self._load_export_data(audit_id, "request_logs", RequestLogExport)
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
            cookies = await self._load_export_data(audit_id, "cookies", CookieExport)
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
            tags = await self._load_export_data(audit_id, "tags", TagExport)
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
            snapshots = await self._load_export_data(audit_id, "data_layer_snapshots", DataLayerExport)
            if filters:
                snapshots = self._filter_data_layer_snapshots(snapshots, filters)
            return snapshots

    async def store_export_metadata(self, metadata: ExportMetadata) -> None:
        """Store export generation metadata."""
        async with self._lock:
            metadata_file = self._get_metadata_file_path(metadata.audit_id)
            metadata_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing metadata
            existing_metadata = {}
            if metadata_file.exists():
                try:
                    async with aiofiles.open(metadata_file, 'r') as f:
                        content = await f.read()
                        existing_metadata = json.loads(content) if content.strip() else {}
                except Exception as e:
                    logger.warning(f"Failed to load existing metadata: {e}")

            # Update with new metadata
            key = f"{metadata.export_type}:{metadata.format}"
            existing_metadata[key] = metadata.model_dump(mode='json')

            # Save updated metadata
            try:
                async with aiofiles.open(metadata_file, 'w') as f:
                    await f.write(json.dumps(existing_metadata, indent=2, default=str))
            except Exception as e:
                logger.error(f"Failed to save export metadata: {e}")
                raise

    async def get_export_metadata(
        self,
        audit_id: str,
        export_type: str,
        format: str
    ) -> Optional[ExportMetadata]:
        """Get export metadata."""
        async with self._lock:
            metadata_file = self._get_metadata_file_path(audit_id)
            if not metadata_file.exists():
                return None

            try:
                async with aiofiles.open(metadata_file, 'r') as f:
                    content = await f.read()
                    all_metadata = json.loads(content) if content.strip() else {}

                key = f"{export_type}:{format}"
                metadata_data = all_metadata.get(key)
                if metadata_data:
                    return ExportMetadata.model_validate(metadata_data)
                return None
            except Exception as e:
                logger.error(f"Failed to load export metadata: {e}")
                return None

    # Helper methods for data storage (for seeding test data)
    async def store_request_logs(self, audit_id: str, logs: List[RequestLogExport]) -> None:
        """Store request log data (for testing/seeding)."""
        await self._save_export_data(audit_id, "request_logs", logs)

    async def store_cookies(self, audit_id: str, cookies: List[CookieExport]) -> None:
        """Store cookie data (for testing/seeding)."""
        await self._save_export_data(audit_id, "cookies", cookies)

    async def store_tags(self, audit_id: str, tags: List[TagExport]) -> None:
        """Store tag data (for testing/seeding)."""
        await self._save_export_data(audit_id, "tags", tags)

    async def store_data_layer_snapshots(self, audit_id: str, snapshots: List[DataLayerExport]) -> None:
        """Store data layer snapshot data (for testing/seeding)."""
        await self._save_export_data(audit_id, "data_layer_snapshots", snapshots)

    # Import filter methods from in-memory implementation
    def _filter_request_logs(self, logs: List[RequestLogExport], filters: Dict[str, Any]) -> List[RequestLogExport]:
        """Apply filters to request logs."""
        filtered = logs
        if filters.get("status"):
            filtered = [log for log in filtered if log.status == filters["status"]]
        if filters.get("resource_type"):
            filtered = [log for log in filtered if log.resource_type == filters["resource_type"]]
        if filters.get("analytics_only"):
            filtered = [log for log in filtered if log.is_analytics]
        return filtered

    def _filter_cookies(self, cookies: List[CookieExport], filters: Dict[str, Any]) -> List[CookieExport]:
        """Apply filters to cookies."""
        filtered = cookies
        if filters.get("category"):
            filtered = [c for c in filtered if c.category == filters["category"]]
        if filters.get("vendor"):
            filtered = [c for c in filtered if c.vendor == filters["vendor"]]
        if filters.get("essential_only"):
            filtered = [c for c in filtered if c.is_essential]
        if filters.get("privacy_compliant"):
            filtered = [c for c in filtered if c.privacy_compliant]
        return filtered

    def _filter_tags(self, tags: List[TagExport], filters: Dict[str, Any]) -> List[TagExport]:
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

    def _filter_data_layer_snapshots(self, snapshots: List[DataLayerExport], filters: Dict[str, Any]) -> List[DataLayerExport]:
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