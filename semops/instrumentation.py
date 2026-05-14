from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Protocol

from semops.monitor import MonitorReport


@dataclass(frozen=True)
class SemanticEvent:
    """Generic local event record for semantic measurements."""
    operation: str
    timestamp: str
    score: float | None = None
    threshold: float | None = None
    sample_size: int | None = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventSink(Protocol):
    def emit(self, event: SemanticEvent) -> None:
        ...


class ListSink:
    """Simple in-memory sink for notebooks and tests."""

    def __init__(self) -> None:
        self.events: List[SemanticEvent] = []

    def emit(self, event: SemanticEvent) -> None:
        self.events.append(event)

    def rows(self) -> List[Dict[str, Any]]:
        return [asdict(event) for event in self.events]


def make_event(
    operation: str,
    *,
    score: float | None = None,
    threshold: float | None = None,
    sample_size: int | None = None,
    tags: Dict[str, str] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> SemanticEvent:
    """Build a timestamped semantic event with no transport behavior."""
    return SemanticEvent(
        operation=operation,
        timestamp=datetime.now(timezone.utc).isoformat(),
        score=score,
        threshold=threshold,
        sample_size=sample_size,
        tags=tags or {},
        metadata=metadata or {},
    )


def emit_event(sink: EventSink, event: SemanticEvent) -> SemanticEvent:
    """Send an event to any local sink that implements emit()."""
    sink.emit(event)
    return event


def emit_metric(
    sink: EventSink,
    operation: str,
    *,
    score: float | None = None,
    threshold: float | None = None,
    sample_size: int | None = None,
    tags: Dict[str, str] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> SemanticEvent:
    """Convenience helper for recording a semantic metric locally."""
    event = make_event(
        operation,
        score=score,
        threshold=threshold,
        sample_size=sample_size,
        tags=tags,
        metadata=metadata,
    )
    sink.emit(event)
    return event


def emit_report(
    sink: EventSink,
    report: MonitorReport,
    *,
    tags: Dict[str, str] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> SemanticEvent:
    """Convert a local monitor report into a lightweight event."""
    event_metadata = dict(report.metadata)
    if metadata:
        event_metadata.update(metadata)

    return emit_metric(
        sink,
        report.metric,
        score=report.mean_score,
        threshold=report.threshold,
        sample_size=len(report.items),
        tags=tags,
        metadata=event_metadata,
    )
