"""Risk scoring utilities for reasoning vulnerability detection."""

from __future__ import annotations

from reasoning_monitor.schemas import (
    Severity,
    VulnerabilityAlert,
)


def score_to_severity(score: float) -> Severity:
    """Map a risk score [0.0, 1.0] to a severity level.

    Args:
        score: Risk score between 0.0 and 1.0.

    Returns:
        Corresponding Severity enum value.
    """
    if score >= 0.8:
        return Severity.CRITICAL
    elif score >= 0.6:
        return Severity.HIGH
    elif score >= 0.4:
        return Severity.MEDIUM
    return Severity.LOW


def aggregate_scores(alerts: list[VulnerabilityAlert]) -> float:
    """Compute aggregate risk score from a list of alerts.

    Uses a decay-weighted combination: each successive alert
    contributes less to prevent a single noisy detector from
    dominating the score. The result is clamped to [0.0, 1.0].

    Args:
        alerts: List of vulnerability alerts.

    Returns:
        Aggregate risk score in [0.0, 1.0].
    """
    if not alerts:
        return 0.0

    # Sort by risk_score descending
    sorted_alerts = sorted(alerts, key=lambda a: a.risk_score, reverse=True)

    total = 0.0
    weight_sum = 0.0
    for i, alert in enumerate(sorted_alerts):
        weight = 1.0 / (1.0 + i * 0.5)  # decay: 1.0, 0.67, 0.5, 0.4, ...
        total += alert.risk_score * weight
        weight_sum += weight

    score = total / weight_sum if weight_sum > 0 else 0.0
    return min(1.0, max(0.0, score))


def should_alert(risk_score: float, threshold: float) -> bool:
    """Determine if a risk score exceeds the alert threshold.

    Args:
        risk_score: Computed risk score.
        threshold: Minimum score to trigger an alert.

    Returns:
        True if risk_score >= threshold.
    """
    return risk_score >= threshold
