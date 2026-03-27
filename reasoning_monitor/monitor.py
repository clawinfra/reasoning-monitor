"""ReasoningMonitor — main interface for reasoning chain vulnerability monitoring.

Inspired by Wang et al. (2026) "Beyond Content Safety: Real-Time Monitoring
for Reasoning Vulnerabilities in Large Language Models".
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from reasoning_monitor.detectors.anomaly import AnomalyDetector
from reasoning_monitor.detectors.injection import InjectionDetector
from reasoning_monitor.detectors.leakage import LeakageDetector
from reasoning_monitor.detectors.manipulation import ManipulationDetector
from reasoning_monitor.schemas import (
    SENSITIVITY_THRESHOLDS,
    ReasoningStep,
    Sensitivity,
    SessionResult,
    VulnerabilityAlert,
)
from reasoning_monitor.utils.scorer import aggregate_scores
from reasoning_monitor.utils.tokenizer import split_reasoning_chain


class MonitorSession:
    """An active monitoring session for a reasoning chain.

    Created via `ReasoningMonitor.session()`. Tracks state across
    multiple reasoning steps and accumulates alerts.
    """

    def __init__(
        self,
        *,
        threshold: float,
        injection_detector: InjectionDetector,
        leakage_detector: LeakageDetector,
        manipulation_detector: ManipulationDetector,
        anomaly_detector: AnomalyDetector,
    ) -> None:
        self._threshold = threshold
        self._injection = injection_detector
        self._leakage = leakage_detector
        self._manipulation = manipulation_detector
        self._anomaly = anomaly_detector
        self._result = SessionResult()
        self._step_index = 0

    def check(self, step: str | ReasoningStep) -> VulnerabilityAlert | None:
        """Check a single reasoning step for vulnerabilities.

        Args:
            step: Either a string or a ReasoningStep object.

        Returns:
            The highest-severity alert found, or None if clean.
        """
        if isinstance(step, str):
            step = ReasoningStep(content=step, index=self._step_index)

        self._step_index += 1
        self._result.steps_checked += 1

        alerts: list[VulnerabilityAlert] = []

        # Run all detectors
        for detector in [self._injection, self._leakage, self._manipulation, self._anomaly]:
            alert = detector.check(step, threshold=self._threshold)
            if alert is not None:
                alerts.append(alert)

        if not alerts:
            return None

        # Record all alerts
        self._result.alerts.extend(alerts)

        # Update aggregate risk
        self._result.aggregate_risk = aggregate_scores(self._result.alerts)
        self._result.max_risk = max(
            self._result.max_risk,
            max(a.risk_score for a in alerts),
        )

        # Return the highest-severity alert
        alerts.sort(key=lambda a: a.risk_score, reverse=True)
        return alerts[0]

    def check_all(self, step: str | ReasoningStep) -> list[VulnerabilityAlert]:
        """Check a step and return ALL alerts (not just the highest).

        Args:
            step: Either a string or a ReasoningStep object.

        Returns:
            List of all alerts found (may be empty).
        """
        if isinstance(step, str):
            step = ReasoningStep(content=step, index=self._step_index)

        self._step_index += 1
        self._result.steps_checked += 1

        alerts: list[VulnerabilityAlert] = []

        for detector in [self._injection, self._leakage, self._manipulation, self._anomaly]:
            alert = detector.check(step, threshold=self._threshold)
            if alert is not None:
                alerts.append(alert)

        if alerts:
            self._result.alerts.extend(alerts)
            self._result.aggregate_risk = aggregate_scores(self._result.alerts)
            self._result.max_risk = max(
                self._result.max_risk,
                max(a.risk_score for a in alerts),
            )

        return alerts

    @property
    def result(self) -> SessionResult:
        """Get the current session result."""
        return self._result


class ReasoningMonitor:
    """Main interface for monitoring LLM reasoning chains.

    Provides both streaming (session-based) and batch analysis modes
    for detecting vulnerabilities in chain-of-thought reasoning.

    Example:
        >>> monitor = ReasoningMonitor(sensitivity="medium")
        >>> with monitor.session() as session:
        ...     alert = session.check("Let me think about this...")
        ...     if alert:
        ...         print(f"Alert: {alert.type.value}")
    """

    def __init__(
        self,
        *,
        sensitivity: str | Sensitivity = Sensitivity.MEDIUM,
        original_task: str | None = None,
    ) -> None:
        """Initialize the reasoning monitor.

        Args:
            sensitivity: Detection sensitivity level ("low", "medium", "high").
            original_task: The original user task for goal-drift detection.
        """
        if isinstance(sensitivity, str):
            sensitivity = Sensitivity(sensitivity.lower())

        self._sensitivity = sensitivity
        self._threshold = SENSITIVITY_THRESHOLDS[sensitivity]
        self._original_task = original_task

    @contextmanager
    def session(self) -> Generator[MonitorSession, None, None]:
        """Create a monitoring session for streaming analysis.

        Yields:
            MonitorSession instance for checking individual steps.
        """
        anomaly_detector = AnomalyDetector()
        session = MonitorSession(
            threshold=self._threshold,
            injection_detector=InjectionDetector(),
            leakage_detector=LeakageDetector(),
            manipulation_detector=ManipulationDetector(original_task=self._original_task),
            anomaly_detector=anomaly_detector,
        )
        try:
            yield session
        finally:
            # Cleanup: reset anomaly detector state
            anomaly_detector.reset()

    def analyze(
        self,
        reasoning_chain: str | list[str] | list[ReasoningStep],
    ) -> list[VulnerabilityAlert]:
        """Batch-analyze an entire reasoning chain.

        Args:
            reasoning_chain: Either a raw string (auto-split into steps),
                a list of step strings, or a list of ReasoningStep objects.

        Returns:
            List of all vulnerability alerts found.
        """
        steps: list[ReasoningStep] = self._normalize_input(reasoning_chain)

        with self.session() as session:
            for step in steps:
                session.check_all(step)

        return session.result.alerts

    def analyze_with_result(
        self,
        reasoning_chain: str | list[str] | list[ReasoningStep],
    ) -> SessionResult:
        """Batch-analyze and return full session result with aggregates.

        Args:
            reasoning_chain: The reasoning chain to analyze.

        Returns:
            SessionResult with alerts, aggregate risk, etc.
        """
        steps = self._normalize_input(reasoning_chain)

        with self.session() as session:
            for step in steps:
                session.check_all(step)

        return session.result

    @staticmethod
    def _normalize_input(
        reasoning_chain: str | list[str] | list[ReasoningStep],
    ) -> list[ReasoningStep]:
        """Normalize various input formats to list of ReasoningStep."""
        if isinstance(reasoning_chain, str):
            return split_reasoning_chain(reasoning_chain)
        elif isinstance(reasoning_chain, list):
            if not reasoning_chain:
                return []
            if isinstance(reasoning_chain[0], str):
                return [
                    ReasoningStep(content=s, index=i)
                    for i, s in enumerate(reasoning_chain)
                    if s.strip()
                ]
            elif isinstance(reasoning_chain[0], ReasoningStep):
                return list(reasoning_chain)  # type: ignore[arg-type]
        raise TypeError(f"Unsupported input type: {type(reasoning_chain)}")

    @property
    def sensitivity(self) -> Sensitivity:
        """Current sensitivity level."""
        return self._sensitivity

    @property
    def threshold(self) -> float:
        """Current alert threshold."""
        return self._threshold
