"""reasoning-monitor: Real-time monitoring for LLM reasoning vulnerabilities.

Inspired by Wang et al. (2026) "Beyond Content Safety: Real-Time Monitoring
for Reasoning Vulnerabilities in Large Language Models".

Usage:
    >>> from reasoning_monitor import ReasoningMonitor
    >>> monitor = ReasoningMonitor(sensitivity="medium")
    >>> with monitor.session() as session:
    ...     alert = session.check("Step 1: Let me think about this...")
"""

from reasoning_monitor.monitor import MonitorSession, ReasoningMonitor
from reasoning_monitor.schemas import (
    ReasoningStep,
    Sensitivity,
    Severity,
    SessionResult,
    VulnerabilityAlert,
    VulnerabilityType,
)

__version__ = "0.1.0"

__all__ = [
    "MonitorSession",
    "ReasoningMonitor",
    "ReasoningStep",
    "Sensitivity",
    "Severity",
    "SessionResult",
    "VulnerabilityAlert",
    "VulnerabilityType",
]
