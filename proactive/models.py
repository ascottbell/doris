"""Data models for the proactive system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
import json
import uuid


@dataclass
class ProactiveEvent:
    """An event detected from a source that may require action."""

    source_type: str  # 'email', 'calendar', 'weather', 'imessage'
    source_id: str    # email_id, event_id, etc.
    raw_data: dict    # The actual data from the source

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detected_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, evaluated, actioned, ignored
    evaluation: Optional[dict] = None  # Claude's analysis

    def to_db_row(self) -> tuple:
        return (
            self.id,
            self.source_type,
            self.source_id,
            json.dumps(self.raw_data),
            self.detected_at.isoformat(),
            self.status,
            json.dumps(self.evaluation) if self.evaluation else None,
        )

    @classmethod
    def from_db_row(cls, row: tuple) -> "ProactiveEvent":
        return cls(
            id=row[0],
            source_type=row[1],
            source_id=row[2],
            raw_data=json.loads(row[3]),
            detected_at=datetime.fromisoformat(row[4]),
            status=row[5],
            evaluation=json.loads(row[6]) if row[6] else None,
        )


@dataclass
class ProactiveAction:
    """An action taken by Doris in response to an event."""

    event_id: str           # Link to ProactiveEvent
    action_type: str        # 'create_event', 'send_reminder', 'notify'
    action_data: dict       # What was done (title, time, etc.)

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    result_id: Optional[str] = None  # Calendar event ID, etc. (for undo)
    executed_at: datetime = field(default_factory=datetime.now)
    status: str = "completed"  # completed, undone, corrected
    notification_sent: bool = False
    wisdom_id: Optional[str] = None  # Links to wisdom table for feedback

    def to_db_row(self) -> tuple:
        return (
            self.id,
            self.event_id,
            self.action_type,
            json.dumps(self.action_data),
            self.result_id,
            self.executed_at.isoformat(),
            self.status,
            1 if self.notification_sent else 0,
        )

    @classmethod
    def from_db_row(cls, row: tuple) -> "ProactiveAction":
        return cls(
            id=row[0],
            event_id=row[1],
            action_type=row[2],
            action_data=json.loads(row[3]),
            result_id=row[4],
            executed_at=datetime.fromisoformat(row[5]),
            status=row[6],
            notification_sent=bool(row[7]),
        )


@dataclass
class ActionItem:
    """A single action to take."""
    action_type: str  # 'create_event', 'send_reminder', 'notify', etc.
    action_data: dict  # Event details, reminder text, etc.
    wisdom_id: Optional[str] = None  # Links to wisdom table for feedback loop

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "action_data": self.action_data,
            "wisdom_id": self.wisdom_id,
        }


@dataclass
class EvaluationResult:
    """Claude's decision about an event - can include multiple actions."""

    should_act: bool
    actions: list[ActionItem] = field(default_factory=list)  # Multiple actions
    reasoning: str = ""
    confidence: float = 1.0

    # Legacy single-action fields for backward compatibility
    action_type: Optional[str] = None
    action_data: Optional[dict] = None

    def __post_init__(self):
        # If legacy fields are set but actions list is empty, convert
        if self.action_type and self.action_data and not self.actions:
            self.actions = [ActionItem(self.action_type, self.action_data)]
        # If actions list is set, populate legacy fields from first action
        elif self.actions and not self.action_type:
            self.action_type = self.actions[0].action_type
            self.action_data = self.actions[0].action_data

    def to_dict(self) -> dict:
        return {
            "should_act": self.should_act,
            "actions": [a.to_dict() for a in self.actions],
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }

    @classmethod
    def no_action(cls, reasoning: str = "") -> "EvaluationResult":
        return cls(should_act=False, reasoning=reasoning)
