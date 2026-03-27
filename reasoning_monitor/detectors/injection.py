"""Prompt injection detector for reasoning steps.

Detects attempts to override instructions, inject new goals, or manipulate
the reasoning process through embedded prompt injection patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from reasoning_monitor.schemas import (
    ReasoningStep,
    Severity,
    VulnerabilityAlert,
    VulnerabilityType,
)
from reasoning_monitor.utils.scorer import score_to_severity


# Injection patterns with associated base risk scores
@dataclass(frozen=True)
class InjectionPattern:
    """A pattern that may indicate prompt injection."""

    pattern: re.Pattern[str]
    description: str
    base_score: float


# Ordered from highest to lowest severity
INJECTION_PATTERNS: list[InjectionPattern] = [
    InjectionPattern(
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?|guidelines?)", re.IGNORECASE),
        "Instruction override attempt",
        0.95,
    ),
    InjectionPattern(
        re.compile(r"(?:new|updated|revised|actual)\s+(?:instructions?|system\s+prompt|directive)", re.IGNORECASE),
        "New instruction injection",
        0.9,
    ),
    InjectionPattern(
        re.compile(r"you\s+(?:are|must|should|will)\s+now\s+(?:act|behave|respond|function)\s+as", re.IGNORECASE),
        "Role reassignment attempt",
        0.85,
    ),
    InjectionPattern(
        re.compile(r"(?:disregard|forget|override|bypass|skip)\s+(?:all\s+)?(?:safety|security|content|ethical)\s+(?:filters?|rules?|guidelines?|measures?|checks?)", re.IGNORECASE),
        "Safety bypass attempt",
        0.95,
    ),
    InjectionPattern(
        re.compile(r"(?:system|admin|root|developer)\s*(?:mode|access|override|prompt)\s*(?:enabled|activated|on)?", re.IGNORECASE),
        "Privilege escalation attempt",
        0.85,
    ),
    InjectionPattern(
        re.compile(r"\[(?:SYSTEM|INST|ADMIN)\]", re.IGNORECASE),
        "System tag injection",
        0.8,
    ),
    InjectionPattern(
        re.compile(r"(?:do\s+not|don't|never)\s+(?:mention|reveal|disclose|tell|say)\s+(?:that|this|these)", re.IGNORECASE),
        "Concealment instruction",
        0.7,
    ),
    InjectionPattern(
        re.compile(r"(?:pretend|assume|imagine)\s+(?:that\s+)?(?:you|the\s+system|we)\s+(?:are|have|can)", re.IGNORECASE),
        "Context manipulation",
        0.6,
    ),
    InjectionPattern(
        re.compile(r"(?:from\s+now\s+on|henceforth|going\s+forward),?\s+(?:you|the\s+system)", re.IGNORECASE),
        "Persistent behavior change attempt",
        0.75,
    ),
    InjectionPattern(
        re.compile(r"<\|?(?:im_start|im_end|endoftext|system|user|assistant)\|?>", re.IGNORECASE),
        "Special token injection",
        0.9,
    ),
]


class InjectionDetector:
    """Detects prompt injection patterns in reasoning steps.

    Scans each reasoning step for known injection patterns and computes
    a risk score based on the number and severity of matches.
    """

    def __init__(self, *, extra_patterns: list[InjectionPattern] | None = None) -> None:
        """Initialize the injection detector.

        Args:
            extra_patterns: Additional patterns to check beyond defaults.
        """
        self.patterns = list(INJECTION_PATTERNS)
        if extra_patterns:
            self.patterns.extend(extra_patterns)

    def check(self, step: ReasoningStep, threshold: float = 0.5) -> VulnerabilityAlert | None:
        """Check a single reasoning step for injection patterns.

        Args:
            step: The reasoning step to analyze.
            threshold: Minimum risk score to generate an alert.

        Returns:
            VulnerabilityAlert if injection detected, None otherwise.
        """
        matches = self._find_matches(step.content)

        if not matches:
            return None

        risk_score = self._compute_risk(matches)

        if risk_score < threshold:
            return None

        evidence_parts = [f"[{m.description}] matched" for m in matches[:3]]
        evidence = "; ".join(evidence_parts)

        return VulnerabilityAlert(
            type=VulnerabilityType.INJECTION,
            severity=score_to_severity(risk_score),
            risk_score=risk_score,
            step=step,
            evidence=evidence,
            recommendation="Halt reasoning chain. Potential prompt injection detected in CoT. "
            "Review the input for adversarial content and sanitize before re-processing.",
            detector="InjectionDetector",
        )

    def _find_matches(self, text: str) -> list[InjectionPattern]:
        """Find all matching injection patterns in text."""
        return [p for p in self.patterns if p.pattern.search(text)]

    @staticmethod
    def _compute_risk(matches: list[InjectionPattern]) -> float:
        """Compute composite risk score from matched patterns.

        Takes the max base score and adds diminishing contributions
        from additional matches.
        """
        if not matches:
            return 0.0

        sorted_scores = sorted([m.base_score for m in matches], reverse=True)
        risk = sorted_scores[0]

        for i, score in enumerate(sorted_scores[1:], start=1):
            risk += score * (0.1 / i)  # diminishing contribution

        return min(1.0, risk)
