"""Database repository implementations for Tag Sentinel API.

This module provides persistent storage implementations using SQLAlchemy
for production deployments with proper database support.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import asyncio

from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, Index, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy import select, update, delete, and_, or_

from app.api.schemas.requests import ListAuditsRequest
from app.api.schemas.responses import AuditStatus
from app.api.schemas.exports import RequestLogExport, CookieExport, TagExport, DataLayerExport
from .repositories import AuditRepository, ExportDataRepository
from .models import PersistentAuditRecord, ExportMetadata

logger = logging.getLogger(__name__)

Base = declarative_base()


class AuditTable(Base):
    """SQLAlchemy model for audit records."""
    __tablename__ = 'audits'

    id = Column(String(255), primary_key=True)
    site_id = Column(String(100), nullable=False, index=True)
    env = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False, index=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    idempotency_key = Column(String(100), nullable=True, index=True)
    priority = Column(Integer, default=0, index=True)
    params = Column(Text, nullable=True)  # JSON string
    rules_path = Column(String(500), nullable=True)
    audit_metadata = Column("metadata", Text, nullable=True)  # JSON string
    error_message = Column(Text, nullable=True)
    error_details = Column(Text, nullable=True)  # JSON string
    progress_percent = Column(Integer, nullable=True)
    summary = Column(Text, nullable=True)  # JSON string

    # Composite indexes for common query patterns
    __table_args__ = (
        Index('idx_audit_site_env', 'site_id', 'env'),
        Index('idx_audit_status_created', 'status', 'created_at'),
        Index('idx_audit_idempotency_created', 'idempotency_key', 'created_at'),
    )


class ExportDataTable(Base):
    """SQLAlchemy model for export data records."""
    __tablename__ = 'export_data'

    id = Column(String(255), primary_key=True)
    audit_id = Column(String(255), nullable=False, index=True)
    export_type = Column(String(50), nullable=False, index=True)  # request_logs, cookies, tags, data_layer
    data = Column(Text, nullable=False)  # JSON string
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_export_audit_type', 'audit_id', 'export_type'),
    )


class ExportMetadataTable(Base):
    """SQLAlchemy model for export metadata."""
    __tablename__ = 'export_metadata'

    id = Column(String(255), primary_key=True)
    audit_id = Column(String(255), nullable=False, index=True)
    export_type = Column(String(50), nullable=False)
    format = Column(String(20), nullable=False)
    generated_at = Column(DateTime, nullable=False)
    record_count = Column(Integer, nullable=False)
    size_bytes = Column(Integer, nullable=False)
    filters_applied = Column(Text, nullable=True)  # JSON string
    fields_included = Column(Text, nullable=True)  # JSON string

    __table_args__ = (
        Index('idx_export_meta_audit_type_format', 'audit_id', 'export_type', 'format'),
    )


class DatabaseAuditRepository(AuditRepository):
    """Database implementation of audit repository using SQLAlchemy."""

    def __init__(self, database_url: str = "sqlite:///./data/tag_sentinel.db"):
        """Initialize database repository.

        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
        )

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.SessionLocal = scoped_session(sessionmaker(bind=self.engine))
        self._lock = asyncio.Lock()

        logger.info(f"DatabaseAuditRepository initialized with database_url={database_url}")

    def _to_persistent_record(self, audit_row: AuditTable) -> PersistentAuditRecord:
        """Convert database row to PersistentAuditRecord."""
        return PersistentAuditRecord(
            id=audit_row.id,
            site_id=audit_row.site_id,
            env=audit_row.env,
            status=AuditStatus(audit_row.status),
            created_at=audit_row.created_at,
            updated_at=audit_row.updated_at,
            started_at=audit_row.started_at,
            finished_at=audit_row.finished_at,
            idempotency_key=audit_row.idempotency_key,
            priority=audit_row.priority or 0,
            params=json.loads(audit_row.params) if audit_row.params else None,
            rules_path=audit_row.rules_path,
            metadata=json.loads(audit_row.audit_metadata) if audit_row.audit_metadata else None,
            error_message=audit_row.error_message,
            error_details=json.loads(audit_row.error_details) if audit_row.error_details else None,
            progress_percent=audit_row.progress_percent,
            summary=json.loads(audit_row.summary) if audit_row.summary else None
        )

    def _to_database_row(self, audit: PersistentAuditRecord) -> AuditTable:
        """Convert PersistentAuditRecord to database row."""
        return AuditTable(
            id=audit.id,
            site_id=audit.site_id,
            env=audit.env,
            status=audit.status.value,
            created_at=audit.created_at,
            updated_at=audit.updated_at,
            started_at=audit.started_at,
            finished_at=audit.finished_at,
            idempotency_key=audit.idempotency_key,
            priority=audit.priority,
            params=json.dumps(audit.params) if audit.params else None,
            rules_path=audit.rules_path,
            audit_metadata=json.dumps(audit.metadata) if audit.metadata else None,
            error_message=audit.error_message,
            error_details=json.dumps(audit.error_details) if audit.error_details else None,
            progress_percent=audit.progress_percent,
            summary=json.dumps(audit.summary) if audit.summary else None
        )

    async def create_audit(self, audit: PersistentAuditRecord) -> None:
        """Create a new audit record."""
        async with self._lock:
            session = self.SessionLocal()
            try:
                # Check if audit already exists
                existing = session.query(AuditTable).filter_by(id=audit.id).first()
                if existing:
                    raise ValueError(f"Audit with ID {audit.id} already exists")

                # Create new audit
                audit_row = self._to_database_row(audit)
                session.add(audit_row)
                session.commit()

                logger.info(f"Created audit {audit.id} in database repository")

            except Exception as e:
                session.rollback()
                logger.error(f"Failed to create audit {audit.id}: {e}")
                raise
            finally:
                session.close()

    async def get_audit(self, audit_id: str) -> Optional[PersistentAuditRecord]:
        """Get audit by ID."""
        async with self._lock:
            session = self.SessionLocal()
            try:
                audit_row = session.query(AuditTable).filter_by(id=audit_id).first()
                if audit_row:
                    logger.debug(f"Retrieved audit {audit_id} from database repository")
                    return self._to_persistent_record(audit_row)
                return None
            finally:
                session.close()

    async def update_audit(self, audit: PersistentAuditRecord) -> None:
        """Update an existing audit record."""
        async with self._lock:
            session = self.SessionLocal()
            try:
                # Check if audit exists
                existing = session.query(AuditTable).filter_by(id=audit.id).first()
                if not existing:
                    raise ValueError(f"Audit with ID {audit.id} does not exist")

                # Update audit
                audit.updated_at = datetime.utcnow()
                updated_row = self._to_database_row(audit)

                session.merge(updated_row)
                session.commit()

                logger.info(f"Updated audit {audit.id} in database repository")

            except Exception as e:
                session.rollback()
                logger.error(f"Failed to update audit {audit.id}: {e}")
                raise
            finally:
                session.close()

    async def delete_audit(self, audit_id: str) -> bool:
        """Delete an audit record."""
        async with self._lock:
            session = self.SessionLocal()
            try:
                rows_deleted = session.query(AuditTable).filter_by(id=audit_id).delete()
                session.commit()

                if rows_deleted > 0:
                    logger.info(f"Deleted audit {audit_id} from database repository")
                    return True
                return False

            except Exception as e:
                session.rollback()
                logger.error(f"Failed to delete audit {audit_id}: {e}")
                return False
            finally:
                session.close()

    async def list_audits(
        self,
        request: ListAuditsRequest,
        limit: int = 20,
        offset: int = 0
    ) -> Tuple[List[PersistentAuditRecord], int]:
        """List audits with filtering and pagination."""
        async with self._lock:
            session = self.SessionLocal()
            try:
                # Build base query
                query = session.query(AuditTable)

                # Apply filters
                if request.site_id:
                    query = query.filter(AuditTable.site_id == request.site_id)

                if request.env:
                    query = query.filter(AuditTable.env == request.env)

                if request.status:
                    query = query.filter(AuditTable.status.in_(request.status))

                if request.date_from:
                    query = query.filter(AuditTable.created_at >= request.date_from)

                if request.date_to:
                    query = query.filter(AuditTable.created_at <= request.date_to)

                # Search filter (basic text search)
                if request.search:
                    search_pattern = f"%{request.search.lower()}%"
                    query = query.filter(or_(
                        AuditTable.site_id.ilike(search_pattern),
                        AuditTable.env.ilike(search_pattern),
                        AuditTable.id.ilike(search_pattern),
                        AuditTable.audit_metadata.ilike(search_pattern)
                    ))

                # Get total count before applying pagination
                total_count = query.count()

                # Apply sorting
                sort_column = getattr(AuditTable, request.sort_by or 'created_at', AuditTable.created_at)
                if request.sort_order == 'desc':
                    query = query.order_by(sort_column.desc())
                else:
                    query = query.order_by(sort_column.asc())

                # Apply pagination
                query = query.offset(offset).limit(limit)

                # Execute query
                audit_rows = query.all()
                audits = [self._to_persistent_record(row) for row in audit_rows]

                logger.debug(
                    f"Listed {len(audits)} audits from database repository "
                    f"(total: {total_count}, offset: {offset}, limit: {limit})"
                )

                return audits, total_count

            finally:
                session.close()

    async def find_by_idempotency_key(
        self,
        idempotency_key: str,
        max_age_hours: int = 24
    ) -> Optional[PersistentAuditRecord]:
        """Find audit by idempotency key within time window."""
        async with self._lock:
            session = self.SessionLocal()
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

                audit_row = session.query(AuditTable).filter(
                    and_(
                        AuditTable.idempotency_key == idempotency_key,
                        AuditTable.created_at > cutoff_time
                    )
                ).first()

                if audit_row:
                    logger.debug(f"Found audit {audit_row.id} by idempotency key {idempotency_key}")
                    return self._to_persistent_record(audit_row)

                return None

            finally:
                session.close()

    async def get_audit_count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Get total count of audits matching filters."""
        async with self._lock:
            session = self.SessionLocal()
            try:
                query = session.query(func.count(AuditTable.id))

                if filters:
                    # Apply basic filters
                    if filters.get('site_id'):
                        query = query.filter(AuditTable.site_id == filters['site_id'])
                    if filters.get('env'):
                        query = query.filter(AuditTable.env == filters['env'])
                    if filters.get('status'):
                        statuses = filters['status'] if isinstance(filters['status'], list) else [filters['status']]
                        query = query.filter(AuditTable.status.in_(statuses))

                return query.scalar()

            finally:
                session.close()


class DatabaseExportDataRepository(ExportDataRepository):
    """Database implementation of export data repository."""

    def __init__(self, database_url: str = "sqlite:///./data/tag_sentinel.db"):
        """Initialize database export repository.

        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
        )

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.SessionLocal = scoped_session(sessionmaker(bind=self.engine))
        self._lock = asyncio.Lock()

        logger.info(f"DatabaseExportDataRepository initialized with database_url={database_url}")

    async def _get_export_data(self, audit_id: str, export_type: str, model_class) -> List:
        """Generic method to get export data."""
        session = self.SessionLocal()
        try:
            export_row = session.query(ExportDataTable).filter(
                and_(
                    ExportDataTable.audit_id == audit_id,
                    ExportDataTable.export_type == export_type
                )
            ).first()

            if not export_row:
                return []

            # Parse JSON data
            data_list = json.loads(export_row.data)
            return [model_class.model_validate(item) for item in data_list]

        except Exception as e:
            logger.error(f"Failed to get {export_type} data for audit {audit_id}: {e}")
            return []
        finally:
            session.close()

    async def _store_export_data(self, audit_id: str, export_type: str, data: List) -> None:
        """Generic method to store export data."""
        session = self.SessionLocal()
        try:
            # Serialize data
            serialized_data = [item.model_dump(mode='json') for item in data]
            json_data = json.dumps(serialized_data, default=str)

            # Check if record exists
            existing = session.query(ExportDataTable).filter(
                and_(
                    ExportDataTable.audit_id == audit_id,
                    ExportDataTable.export_type == export_type
                )
            ).first()

            if existing:
                # Update existing record
                existing.data = json_data
                existing.created_at = datetime.utcnow()
            else:
                # Create new record
                export_row = ExportDataTable(
                    id=f"{audit_id}_{export_type}_{datetime.utcnow().timestamp()}",
                    audit_id=audit_id,
                    export_type=export_type,
                    data=json_data
                )
                session.add(export_row)

            session.commit()
            logger.debug(f"Stored {export_type} data for audit {audit_id}")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store {export_type} data for audit {audit_id}: {e}")
            raise
        finally:
            session.close()

    async def get_request_logs(
        self,
        audit_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RequestLogExport]:
        """Get request log data for an audit."""
        async with self._lock:
            logs = await self._get_export_data(audit_id, "request_logs", RequestLogExport)
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
            cookies = await self._get_export_data(audit_id, "cookies", CookieExport)
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
            tags = await self._get_export_data(audit_id, "tags", TagExport)
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
            snapshots = await self._get_export_data(audit_id, "data_layer_snapshots", DataLayerExport)
            if filters:
                snapshots = self._filter_data_layer_snapshots(snapshots, filters)
            return snapshots

    async def store_export_metadata(self, metadata: ExportMetadata) -> None:
        """Store export generation metadata."""
        async with self._lock:
            session = self.SessionLocal()
            try:
                # Check if metadata already exists
                existing = session.query(ExportMetadataTable).filter(
                    and_(
                        ExportMetadataTable.audit_id == metadata.audit_id,
                        ExportMetadataTable.export_type == metadata.export_type,
                        ExportMetadataTable.format == metadata.format
                    )
                ).first()

                if existing:
                    # Update existing metadata
                    existing.generated_at = metadata.generated_at
                    existing.record_count = metadata.record_count
                    existing.size_bytes = metadata.size_bytes
                    existing.filters_applied = json.dumps(metadata.filters_applied) if metadata.filters_applied else None
                    existing.fields_included = json.dumps(metadata.fields_included) if metadata.fields_included else None
                else:
                    # Create new metadata record
                    metadata_row = ExportMetadataTable(
                        id=f"{metadata.audit_id}_{metadata.export_type}_{metadata.format}_{datetime.utcnow().timestamp()}",
                        audit_id=metadata.audit_id,
                        export_type=metadata.export_type,
                        format=metadata.format,
                        generated_at=metadata.generated_at,
                        record_count=metadata.record_count,
                        size_bytes=metadata.size_bytes,
                        filters_applied=json.dumps(metadata.filters_applied) if metadata.filters_applied else None,
                        fields_included=json.dumps(metadata.fields_included) if metadata.fields_included else None
                    )
                    session.add(metadata_row)

                session.commit()
                logger.debug(f"Stored export metadata for {metadata.audit_id}/{metadata.export_type}")

            except Exception as e:
                session.rollback()
                logger.error(f"Failed to store export metadata: {e}")
                raise
            finally:
                session.close()

    async def get_export_metadata(
        self,
        audit_id: str,
        export_type: str,
        format: str
    ) -> Optional[ExportMetadata]:
        """Get export metadata."""
        async with self._lock:
            session = self.SessionLocal()
            try:
                metadata_row = session.query(ExportMetadataTable).filter(
                    and_(
                        ExportMetadataTable.audit_id == audit_id,
                        ExportMetadataTable.export_type == export_type,
                        ExportMetadataTable.format == format
                    )
                ).first()

                if metadata_row:
                    return ExportMetadata(
                        audit_id=metadata_row.audit_id,
                        export_type=metadata_row.export_type,
                        format=metadata_row.format,
                        generated_at=metadata_row.generated_at,
                        record_count=metadata_row.record_count,
                        size_bytes=metadata_row.size_bytes,
                        filters_applied=json.loads(metadata_row.filters_applied) if metadata_row.filters_applied else None,
                        fields_included=json.loads(metadata_row.fields_included) if metadata_row.fields_included else None
                    )

                return None

            finally:
                session.close()

    # Storage methods for seeding data
    async def store_request_logs(self, audit_id: str, logs: List[RequestLogExport]) -> None:
        """Store request log data."""
        await self._store_export_data(audit_id, "request_logs", logs)

    async def store_cookies(self, audit_id: str, cookies: List[CookieExport]) -> None:
        """Store cookie data."""
        await self._store_export_data(audit_id, "cookies", cookies)

    async def store_tags(self, audit_id: str, tags: List[TagExport]) -> None:
        """Store tag data."""
        await self._store_export_data(audit_id, "tags", tags)

    async def store_data_layer_snapshots(self, audit_id: str, snapshots: List[DataLayerExport]) -> None:
        """Store data layer snapshot data."""
        await self._store_export_data(audit_id, "data_layer_snapshots", snapshots)

    # Import filter methods from in-memory implementation
    def _filter_request_logs(self, logs: List[RequestLogExport], filters: Dict[str, Any]) -> List[RequestLogExport]:
        """Apply filters to request logs."""
        filtered = logs
        if filters.get("status"):
            filtered = [log for log in filtered if log.status == filters["status"]]
        if filters.get("resource_type"):
            filtered = [log for log in filtered if log.resource_type == filters["resource_type"]]
        if filters.get("analytics_only"):
            filtered = [log for log in filtered if log.is_analytics_request]
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