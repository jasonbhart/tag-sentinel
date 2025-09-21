"""SQLAlchemy ORM models for Tag Sentinel persistence layer."""

import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from sqlalchemy import (
    String,
    Integer,
    DateTime,
    Boolean,
    Float,
    Text,
    JSON,
    LargeBinary,
    Index,
    ForeignKey,
    CheckConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .database import Base


class RunStatus(str, Enum):
    """Status of an audit run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PageStatus(str, Enum):
    """Status of a page capture."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ArtifactKind(str, Enum):
    """Types of artifacts that can be stored."""
    HAR = "har"
    SCREENSHOT = "screenshot"
    TRACE = "trace"
    PAGE_SOURCE = "page_source"
    VIDEO = "video"


class SeverityLevel(str, Enum):
    """Severity levels for rule failures."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Run(Base):
    """Audit run metadata and summary."""

    __tablename__ = "runs"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Run identification
    site_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    environment: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Timestamps
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        index=True
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Status and metrics
    status: Mapped[RunStatus] = mapped_column(
        String(20),
        nullable=False,
        default=RunStatus.PENDING,
        index=True
    )

    # Configuration and summary
    config_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Run configuration used"
    )
    summary_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Run summary statistics and metrics"
    )

    # Error information
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    page_results: Mapped[List["PageResult"]] = relationship(
        "PageResult",
        back_populates="run",
        cascade="all, delete-orphan"
    )
    rule_failures: Mapped[List["RuleFailure"]] = relationship(
        "RuleFailure",
        back_populates="run",
        cascade="all, delete-orphan"
    )
    artifacts: Mapped[List["Artifact"]] = relationship(
        "Artifact",
        back_populates="run",
        cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("ix_runs_site_env", "site_id", "environment"),
        Index("ix_runs_status_started", "status", "started_at"),
    )

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate run duration in seconds."""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    @property
    def is_complete(self) -> bool:
        """Check if run is in a terminal state."""
        return self.status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]


class PageResult(Base):
    """Results from capturing a single page."""

    __tablename__ = "page_results"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Foreign key to run
    run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Page identification
    url: Mapped[str] = mapped_column(String(2048), nullable=False, index=True)
    final_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Capture metadata
    status: Mapped[PageStatus] = mapped_column(
        String(20),
        nullable=False,
        index=True
    )
    capture_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now()
    )
    load_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Performance and timing data
    timings_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Detailed timing information"
    )

    # Error information
    errors_json: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="JavaScript and capture errors"
    )
    capture_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metrics
    metrics_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Performance metrics and additional data"
    )

    # Relationships
    run: Mapped["Run"] = relationship("Run", back_populates="page_results")
    request_logs: Mapped[List["RequestLog"]] = relationship(
        "RequestLog",
        back_populates="page_result",
        cascade="all, delete-orphan"
    )
    cookies: Mapped[List["Cookie"]] = relationship(
        "Cookie",
        back_populates="page_result",
        cascade="all, delete-orphan"
    )
    datalayer_snapshots: Mapped[List["DataLayerSnapshot"]] = relationship(
        "DataLayerSnapshot",
        back_populates="page_result",
        cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("ix_page_results_run_status", "run_id", "status"),
        Index("ix_page_results_url", "url"),
    )

    @property
    def is_successful(self) -> bool:
        """Check if page capture was successful."""
        return self.status == PageStatus.SUCCESS


class RequestLog(Base):
    """Network request lifecycle data."""

    __tablename__ = "request_logs"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Foreign key to page result
    page_result_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("page_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Request identification
    url: Mapped[str] = mapped_column(String(2048), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(10), nullable=False, default="GET")
    resource_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Response data
    status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status_text: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Headers (stored as JSON)
    request_headers_json: Mapped[Optional[Dict[str, str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Request headers"
    )
    response_headers_json: Mapped[Optional[Dict[str, str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Response headers"
    )

    # Timing data
    timings_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Request timing breakdown"
    )
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now()
    )
    end_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Size information
    sizes_json: Mapped[Optional[Dict[str, int]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Request/response size information"
    )

    # Vendor tag detection
    vendor_tags_json: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Detected vendor tags in this request"
    )

    # Request lifecycle
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Protocol information
    protocol: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    remote_address: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Relationships
    page_result: Mapped["PageResult"] = relationship("PageResult", back_populates="request_logs")

    # Indexes
    __table_args__ = (
        Index("ix_request_logs_page_url", "page_result_id", "url"),
        Index("ix_request_logs_status_code", "status_code"),
        Index("ix_request_logs_start_time", "start_time"),
    )

    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate request duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    @property
    def host(self) -> str:
        """Extract host from URL."""
        from urllib.parse import urlparse
        return urlparse(self.url).netloc


class Cookie(Base):
    """Cookie information with privacy compliance fields."""

    __tablename__ = "cookies"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Foreign key to page result
    page_result_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("page_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Cookie identification
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    path: Mapped[str] = mapped_column(String(1024), nullable=False, default="/")

    # Cookie attributes
    expires: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    max_age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Security attributes
    secure: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    http_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    same_site: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Privacy classification
    first_party: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    essential: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    is_session: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Privacy metadata
    value_redacted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional cookie metadata and classification"
    )

    # Timing
    set_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    modified_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Relationships
    page_result: Mapped["PageResult"] = relationship("PageResult", back_populates="cookies")

    # Indexes
    __table_args__ = (
        Index("ix_cookies_page_name_domain", "page_result_id", "name", "domain"),
        Index("ix_cookies_domain_first_party", "domain", "first_party"),
        Index("ix_cookies_essential", "essential"),
    )

    @property
    def cookie_key(self) -> str:
        """Unique identifier for the cookie."""
        return f"{self.name}@{self.domain}{self.path}"


class DataLayerSnapshot(Base):
    """Data layer snapshot metadata."""

    __tablename__ = "datalayer_snapshots"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Foreign key to page result
    page_result_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("page_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Data layer state
    exists: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    truncated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Sample data (may be redacted/truncated)
    sample_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Sample or truncated data layer content"
    )

    # Validation results
    schema_valid: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    validation_errors_json: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Schema validation error messages"
    )

    # Metadata
    capture_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now()
    )
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional data layer analysis metadata"
    )

    # Relationships
    page_result: Mapped["PageResult"] = relationship("PageResult", back_populates="datalayer_snapshots")

    # Indexes
    __table_args__ = (
        Index("ix_datalayer_page_exists", "page_result_id", "exists"),
        Index("ix_datalayer_size", "size_bytes"),
    )


class RuleFailure(Base):
    """Rule evaluation failure record."""

    __tablename__ = "rule_failures"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Foreign key to run
    run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Rule identification
    rule_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    rule_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Failure details
    severity: Mapped[SeverityLevel] = mapped_column(
        String(20),
        nullable=False,
        index=True
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)

    # Context
    page_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True, index=True)
    details_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Detailed failure context and data"
    )

    # Timestamps
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now()
    )

    # Relationships
    run: Mapped["Run"] = relationship("Run", back_populates="rule_failures")

    # Indexes
    __table_args__ = (
        Index("ix_rule_failures_run_rule", "run_id", "rule_id"),
        Index("ix_rule_failures_severity_detected", "severity", "detected_at"),
        Index("ix_rule_failures_page_url", "page_url"),
    )


class Artifact(Base):
    """Stored artifact metadata and references."""

    __tablename__ = "artifacts"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Foreign key to run
    run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Artifact identification
    kind: Mapped[ArtifactKind] = mapped_column(
        String(50),
        nullable=False,
        index=True
    )
    path: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)

    # File metadata
    checksum: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    content_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Context
    page_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        index=True
    )

    # Storage metadata
    storage_backend: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="local"
    )
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Storage-specific metadata"
    )

    # Relationships
    run: Mapped["Run"] = relationship("Run", back_populates="artifacts")

    # Constraints
    __table_args__ = (
        Index("ix_artifacts_run_kind", "run_id", "kind"),
        Index("ix_artifacts_path", "path"),
        Index("ix_artifacts_checksum", "checksum"),
        CheckConstraint("size_bytes >= 0", name="ck_artifacts_size_positive"),
    )

    @property
    def file_extension(self) -> str:
        """Get file extension from path."""
        return self.path.split('.')[-1] if '.' in self.path else ""

    @property
    def size_mb(self) -> float:
        """Get size in megabytes."""
        return self.size_bytes / (1024 * 1024)