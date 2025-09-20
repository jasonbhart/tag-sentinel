"""Progress tracking and real-time output for CLI operations.

This module provides progress tracking capabilities for long-running audit operations,
including progress bars, real-time status updates, and structured output formatting.
"""

import asyncio
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, TextIO, Union


class ProgressState(Enum):
    """Progress tracking states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressStep:
    """Individual progress step."""
    name: str
    description: str
    state: ProgressState = ProgressState.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    @property
    def duration(self) -> Optional[timedelta]:
        """Get step duration if started."""
        if not self.start_time:
            return None
        end_time = self.end_time or datetime.now()
        return end_time - self.start_time

    @property
    def duration_ms(self) -> Optional[int]:
        """Get step duration in milliseconds."""
        duration = self.duration
        return int(duration.total_seconds() * 1000) if duration else None


class ProgressTracker:
    """Thread-safe progress tracker for CLI operations."""

    def __init__(self, quiet: bool = False, verbose: bool = False):
        self.quiet = quiet
        self.verbose = verbose
        self.steps: List[ProgressStep] = []
        self.current_step: Optional[ProgressStep] = None
        self.start_time = datetime.now()
        self._lock = Lock()
        self._output_stream: TextIO = sys.stderr

    def add_step(self, name: str, description: str) -> ProgressStep:
        """Add a new progress step."""
        with self._lock:
            step = ProgressStep(name=name, description=description)
            self.steps.append(step)
            return step

    def start_step(self, name: str) -> Optional[ProgressStep]:
        """Start a progress step by name."""
        with self._lock:
            step = next((s for s in self.steps if s.name == name), None)
            if step:
                step.state = ProgressState.RUNNING
                step.start_time = datetime.now()
                self.current_step = step

                if not self.quiet:
                    self._print_step_start(step)

            return step

    def update_step(self, name: str, progress: float, details: Optional[Dict[str, Any]] = None):
        """Update progress for a step."""
        with self._lock:
            step = next((s for s in self.steps if s.name == name), None)
            if step:
                step.progress = max(0.0, min(1.0, progress))
                if details:
                    step.details.update(details)

                if not self.quiet and self.verbose:
                    self._print_step_update(step)

    def complete_step(self, name: str, details: Optional[Dict[str, Any]] = None):
        """Mark a step as completed."""
        with self._lock:
            step = next((s for s in self.steps if s.name == name), None)
            if step:
                step.state = ProgressState.COMPLETED
                step.end_time = datetime.now()
                step.progress = 1.0
                if details:
                    step.details.update(details)

                if not self.quiet:
                    self._print_step_complete(step)

    def fail_step(self, name: str, error: str, details: Optional[Dict[str, Any]] = None):
        """Mark a step as failed."""
        with self._lock:
            step = next((s for s in self.steps if s.name == name), None)
            if step:
                step.state = ProgressState.FAILED
                step.end_time = datetime.now()
                step.error = error
                if details:
                    step.details.update(details)

                if not self.quiet:
                    self._print_step_failed(step)

    def _print_step_start(self, step: ProgressStep):
        """Print step start message."""
        timestamp = step.start_time.strftime("%H:%M:%S") if step.start_time else ""
        print(f"🔄 [{timestamp}] {step.description}...", file=self._output_stream)

    def _print_step_update(self, step: ProgressStep):
        """Print step progress update."""
        progress_bar = self._format_progress_bar(step.progress)
        details_str = ""

        if step.details:
            # Format key details for display
            important_keys = ["url", "page", "requests", "cookies", "rules"]
            details_items = []
            for key in important_keys:
                if key in step.details:
                    details_items.append(f"{key}={step.details[key]}")

            if details_items:
                details_str = f" ({', '.join(details_items)})"

        print(f"   {progress_bar} {step.progress:.0%}{details_str}", file=self._output_stream)

    def _print_step_complete(self, step: ProgressStep):
        """Print step completion message."""
        duration_str = ""
        if step.duration_ms:
            if step.duration_ms < 1000:
                duration_str = f" ({step.duration_ms}ms)"
            else:
                duration_str = f" ({step.duration_ms/1000:.1f}s)"

        details_str = ""
        if step.details:
            # Show summary details for completed steps
            summary_items = []
            if "total" in step.details:
                summary_items.append(f"total={step.details['total']}")
            if "successful" in step.details:
                summary_items.append(f"success={step.details['successful']}")
            if "failed" in step.details:
                summary_items.append(f"failed={step.details['failed']}")

            if summary_items:
                details_str = f" ({', '.join(summary_items)})"

        print(f"✅ {step.description} completed{duration_str}{details_str}", file=self._output_stream)

    def _print_step_failed(self, step: ProgressStep):
        """Print step failure message."""
        duration_str = ""
        if step.duration_ms:
            duration_str = f" after {step.duration_ms}ms"

        print(f"❌ {step.description} failed{duration_str}: {step.error}", file=self._output_stream)

    def _format_progress_bar(self, progress: float, width: int = 20) -> str:
        """Format a simple progress bar."""
        filled = int(progress * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    @property
    def total_duration(self) -> timedelta:
        """Get total duration since tracker creation."""
        return datetime.now() - self.start_time

    @property
    def summary(self) -> Dict[str, Any]:
        """Get summary of all progress steps."""
        with self._lock:
            completed = [s for s in self.steps if s.state == ProgressState.COMPLETED]
            failed = [s for s in self.steps if s.state == ProgressState.FAILED]
            running = [s for s in self.steps if s.state == ProgressState.RUNNING]
            pending = [s for s in self.steps if s.state == ProgressState.PENDING]

            return {
                "total_steps": len(self.steps),
                "completed": len(completed),
                "failed": len(failed),
                "running": len(running),
                "pending": len(pending),
                "total_duration_ms": int(self.total_duration.total_seconds() * 1000),
                "steps": [
                    {
                        "name": step.name,
                        "description": step.description,
                        "state": step.state.value,
                        "duration_ms": step.duration_ms,
                        "progress": step.progress,
                        "error": step.error,
                        "details": step.details
                    }
                    for step in self.steps
                ]
            }

    def print_summary(self):
        """Print final summary."""
        if self.quiet:
            return

        summary = self.summary
        duration_s = summary["total_duration_ms"] / 1000

        print(f"\n📊 OPERATION SUMMARY", file=self._output_stream)
        print(f"   Total Steps: {summary['total_steps']}", file=self._output_stream)
        print(f"   ✅ Completed: {summary['completed']}", file=self._output_stream)

        if summary["failed"] > 0:
            print(f"   ❌ Failed: {summary['failed']}", file=self._output_stream)

        if summary["running"] > 0:
            print(f"   🔄 Running: {summary['running']}", file=self._output_stream)

        print(f"   ⏱️  Total Time: {duration_s:.1f}s", file=self._output_stream)


@contextmanager
def ProgressContext(
    name: str,
    description: str,
    tracker: ProgressTracker,
    auto_complete: bool = True
):
    """Context manager for tracking a progress step."""
    step = tracker.add_step(name, description)
    tracker.start_step(name)

    try:
        yield step
        if auto_complete and step.state == ProgressState.RUNNING:
            tracker.complete_step(name)
    except Exception as e:
        tracker.fail_step(name, str(e))
        raise


class RealTimeOutput:
    """Real-time output formatter for streaming results."""

    def __init__(self, format_type: str = "text", quiet: bool = False):
        self.format_type = format_type.lower()
        self.quiet = quiet
        self._buffer: List[Dict[str, Any]] = []

    def emit_event(self, event_type: str, data: Dict[str, Any], timestamp: Optional[datetime] = None):
        """Emit a real-time event."""
        if self.quiet:
            return

        event = {
            "type": event_type,
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "data": data
        }

        self._buffer.append(event)

        if self.format_type == "json":
            self._emit_json_event(event)
        else:
            self._emit_text_event(event)

    def _emit_json_event(self, event: Dict[str, Any]):
        """Emit event as JSON line."""
        import json
        print(json.dumps(event), file=sys.stderr)

    def _emit_text_event(self, event: Dict[str, Any]):
        """Emit event as formatted text."""
        event_type = event["type"]
        data = event["data"]
        timestamp = datetime.fromisoformat(event["timestamp"]).strftime("%H:%M:%S")

        if event_type == "page_capture_start":
            print(f"🔍 [{timestamp}] Capturing: {data.get('url', 'unknown')}", file=sys.stderr)

        elif event_type == "page_capture_complete":
            status_emoji = "✅" if data.get("success", False) else "❌"
            url = data.get("url", "unknown")
            duration = data.get("duration_ms", 0)
            print(f"{status_emoji} [{timestamp}] {url} ({duration}ms)", file=sys.stderr)

            # Show additional details in verbose mode
            if data.get("requests"):
                print(f"   📡 {data['requests']} requests", file=sys.stderr)
            if data.get("cookies"):
                print(f"   🍪 {data['cookies']} cookies", file=sys.stderr)

        elif event_type == "rule_evaluation_start":
            print(f"⚖️  [{timestamp}] Evaluating {data.get('rule_count', 0)} rules...", file=sys.stderr)

        elif event_type == "rule_evaluation_complete":
            passed = data.get("passed", 0)
            failed = data.get("failed", 0)
            duration = data.get("duration_ms", 0)
            print(f"⚖️  [{timestamp}] Rules evaluated: {passed} passed, {failed} failed ({duration}ms)", file=sys.stderr)

        elif event_type == "alert_dispatched":
            dispatcher = data.get("dispatcher", "unknown")
            success = data.get("success", False)
            status_emoji = "📤" if success else "📥"
            print(f"{status_emoji} [{timestamp}] Alert via {dispatcher}: {'sent' if success else 'failed'}", file=sys.stderr)

    def get_events(self) -> List[Dict[str, Any]]:
        """Get all buffered events."""
        return self._buffer.copy()

    def clear_buffer(self):
        """Clear the event buffer."""
        self._buffer.clear()


# Factory functions for common use cases

def create_audit_progress_tracker(quiet: bool = False, verbose: bool = False) -> ProgressTracker:
    """Create a progress tracker configured for audit operations."""
    tracker = ProgressTracker(quiet=quiet, verbose=verbose)

    # Pre-define common audit steps
    tracker.add_step("initialization", "Initializing audit engine")
    tracker.add_step("page_capture", "Capturing pages")
    tracker.add_step("rule_evaluation", "Evaluating rules")
    tracker.add_step("alert_dispatch", "Dispatching alerts")
    tracker.add_step("output_generation", "Generating output")

    return tracker


def create_real_time_output(format_type: str = "text", quiet: bool = False) -> RealTimeOutput:
    """Create a real-time output formatter."""
    return RealTimeOutput(format_type=format_type, quiet=quiet)