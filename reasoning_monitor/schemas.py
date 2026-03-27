"""Data models for reasoning monitoring."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VulnerabilityType(Enum):
    """Types of reasoning vulnerabilities."""

    INJECTION = "injection"
    LEAKAGE = "leakage"
    MANIPULATION = "manipulation"
    ANOMALY = "anomaly"


class Severity(Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Sensitivity(Enum):
    """Monitor sensitivity levels — controls detection thresholds."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Default thresholds per sensitivity level
SENSITIVITY_THRESHOLDS: dict[Sensitivity, float] = {
    Sensitivity.LOW: 0.7,
    Sensitivity.MEDIUM: 0.5,
    Sensitivity.HIGH: 0.3,
}


@dataclass(frozen=True)
class ReasoningStep:
    """A single step in an LLM reasoning chain."""

    content: str
    index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError(f"content must be str, got {type(self.content).__name__}")


@dataclass(frozen=True)
class VulnerabilityAlert:
    """An alert raised when a reasoning vulnerability is detected."""

    type: VulnerabilityType
    severity: Severity
    risk_score: float
    step: ReasoningStep
    evidence: str
    recommendation: str
    detector: str
    alert_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(f"risk_score must be in [0.0, 1.0], got {self.risk_score}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "type": self.type.value,
            "severity": self.severity.value,
            "risk_score": self.risk_score,
            "step_index": self.step.index,
            "step_content": self.step.content[:200],
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "detector": self.detector,
            "timestamp": self.timestamp,
        }


@dataclass
class SessionResult:
    """Aggregated result from a monitoring session."""

    alerts: list[VulnerabilityAlert] = field(default_factory=list)
    steps_checked: int = 0
    aggregate_risk: float = 0.0
    max_risk: float = 0.0

    @property
    def is_safe(self) -> bool:
        """Whether the session had no alerts."""
        return len(self.alerts) == 0

    @property
    def severity(self) -> Severity:
        """Overall severity based on aggregate risk."""
        if self.aggregate_risk >= 0.8:
            return Severity.CRITICAL
        elif self.aggregate_risk >= 0.6:
            return Severity.HIGH
        elif self.aggregate_risk >= 0.4:
            return Severity.MEDIUM
        return Severity.LOW

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "steps_checked": self.steps_checked,
            "alert_count": len(self.alerts),
            "aggregate_risk": round(self.aggregate_risk, 4),
            "max_risk": round(self.max_risk, 4),
            "severity": self.severity.value,
            "is_safe": self.is_safe,
            "alerts": [a.to_dict() for a in self.alerts],
        }
