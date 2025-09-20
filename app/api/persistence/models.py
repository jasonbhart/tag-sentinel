"""Persistence models for Tag Sentinel API.

This module defines data models for persistent storage,
bridging API schemas with underlying storage systems.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

from app.api.schemas.responses import AuditDetail, AuditStatus, AuditLinks, AuditSummary


@dataclass
class PersistentAuditRecord:
    """Persistent representation of an audit record.

    This model bridges the API AuditDetail schema with persistent storage,
    providing serialization and data transformation capabilities.
    """

    # Core audit identification
    id: str
    site_id: str
    env: str
    status: AuditStatus

    # Timing information
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    # Audit configuration
    params: Optional[Dict[str, Any]] = None
    rules_path: Optional[str] = None

    # Results and metrics
    pages_count: int = 0
    failures_count: int = 0
    progress_percent: float = 0.0

    # Summary and metadata
    summary: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    # Idempotency and tracking
    idempotency_key: Optional[str] = None
    priority: int = 0

    # Error information
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Storage-specific fields
    storage_version: int = 1
    storage_metadata: Optional[Dict[str, Any]] = None

    def to_api_model(self, base_url: str = "") -> AuditDetail:
        """Convert to API response model.

        Args:
            base_url: Base URL for generating links

        Returns:
            AuditDetail model for API response
        """
        # Generate links for exports and artifacts
        links_dict = self._generate_links(base_url)
        links = AuditLinks(**links_dict)

        # Convert summary dict to AuditSummary if present
        summary_obj = None
        if self.summary:
            summary_obj = AuditSummary(**self.summary)

        return AuditDetail(
            id=self.id,
            site_id=self.site_id,
            env=self.env,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            params=self.params or {},
            rules_path=self.rules_path,
            pages_count=self.pages_count,
            failures_count=self.failures_count,
            progress_percent=self.progress_percent,
            summary=summary_obj,
            metadata=self.metadata or {},
            idempotency_key=self.idempotency_key,
            priority=self.priority,
            error_message=self.error_message,
            error_details=self.error_details,
            links=links
        )

    @classmethod
    def from_api_model(cls, api_model: AuditDetail) -> "PersistentAuditRecord":
        """Create from API model.

        Args:
            api_model: AuditDetail from API

        Returns:
            PersistentAuditRecord for storage
        """
        return cls(
            id=api_model.id,
            site_id=api_model.site_id,
            env=api_model.env,
            status=api_model.status,
            created_at=api_model.created_at,
            updated_at=api_model.updated_at,
            started_at=api_model.started_at,
            finished_at=api_model.finished_at,
            params=api_model.params,
            rules_path=api_model.rules_path,
            pages_count=api_model.pages_count,
            failures_count=api_model.failures_count,
            progress_percent=api_model.progress_percent,
            summary=api_model.summary,
            metadata=api_model.metadata,
            idempotency_key=api_model.idempotency_key,
            priority=api_model.priority,
            error_message=api_model.error_message,
            error_details=api_model.error_details
        )

    def _generate_links(self, base_url: str) -> Dict[str, str]:
        """Generate API links for this audit.

        Args:
            base_url: Base URL for API

        Returns:
            Dictionary of link names to URLs matching AuditLinks schema
        """
        if not base_url:
            base_url = "http://localhost:8000"

        # Required links
        links = {
            "self": f"{base_url}/api/audits/{self.id}",
            "requests_export": f"{base_url}/api/exports/{self.id}/requests.json",
            "cookies_export": f"{base_url}/api/exports/{self.id}/cookies.csv",
            "tags_export": f"{base_url}/api/exports/{self.id}/tags.json"
        }

        # Optional artifacts link (only for completed audits)
        if self.status == AuditStatus.COMPLETED:
            links["artifacts"] = f"{base_url}/artifacts/{self.id}/"

        return links

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "site_id": self.site_id,
            "env": self.env,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "params": self.params,
            "rules_path": self.rules_path,
            "pages_count": self.pages_count,
            "failures_count": self.failures_count,
            "progress_percent": self.progress_percent,
            "summary": self.summary,
            "metadata": self.metadata,
            "idempotency_key": self.idempotency_key,
            "priority": self.priority,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "storage_version": self.storage_version,
            "storage_metadata": self.storage_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersistentAuditRecord":
        """Create from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            PersistentAuditRecord instance
        """
        return cls(
            id=data["id"],
            site_id=data["site_id"],
            env=data["env"],
            status=AuditStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            finished_at=datetime.fromisoformat(data["finished_at"]) if data.get("finished_at") else None,
            params=data.get("params"),
            rules_path=data.get("rules_path"),
            pages_count=data.get("pages_count", 0),
            failures_count=data.get("failures_count", 0),
            progress_percent=data.get("progress_percent", 0.0),
            summary=data.get("summary"),
            metadata=data.get("metadata"),
            idempotency_key=data.get("idempotency_key"),
            priority=data.get("priority", 0),
            error_message=data.get("error_message"),
            error_details=data.get("error_details"),
            storage_version=data.get("storage_version", 1),
            storage_metadata=data.get("storage_metadata")
        )


@dataclass
class ExportMetadata:
    """Metadata for export data storage."""

    audit_id: str
    export_type: str  # requests, cookies, tags, data_layer
    format: str      # json, csv
    created_at: datetime
    record_count: int
    file_size_bytes: Optional[int] = None
    compression: Optional[str] = None
    checksum: Optional[str] = None
    storage_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "audit_id": self.audit_id,
            "export_type": self.export_type,
            "format": self.format,
            "created_at": self.created_at.isoformat(),
            "record_count": self.record_count,
            "file_size_bytes": self.file_size_bytes,
            "compression": self.compression,
            "checksum": self.checksum,
            "storage_path": self.storage_path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExportMetadata":
        """Create from dictionary."""
        return cls(
            audit_id=data["audit_id"],
            export_type=data["export_type"],
            format=data["format"],
            created_at=datetime.fromisoformat(data["created_at"]),
            record_count=data["record_count"],
            file_size_bytes=data.get("file_size_bytes"),
            compression=data.get("compression"),
            checksum=data.get("checksum"),
            storage_path=data.get("storage_path")
        )