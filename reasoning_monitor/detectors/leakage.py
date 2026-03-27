"""Data leakage detector for reasoning steps.

Detects sensitive information appearing in the reasoning chain that
should not be exposed — system prompts, credentials, PII, etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from reasoning_monitor.schemas import (
    ReasoningStep,
    Severity,
    VulnerabilityAlert,
    VulnerabilityType,
)
from reasoning_monitor.utils.scorer import score_to_severity


@dataclass(frozen=True)
class LeakagePattern:
    """A pattern indicating potential data leakage."""

    pattern: re.Pattern[str]
    description: str
    base_score: float
    category: str  # "credential", "system_prompt", "pii", "internal"


LEAKAGE_PATTERNS: list[LeakagePattern] = [
    # Credential patterns
    LeakagePattern(
        re.compile(r"(?:api[_-]?key|api[_-]?secret|auth[_-]?token|access[_-]?token)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{16,}", re.IGNORECASE),
        "API key or token exposure",
        0.95,
        "credential",
    ),
    LeakagePattern(
        re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{6,}", re.IGNORECASE),
        "Password exposure",
        0.95,
        "credential",
    ),
    LeakagePattern(
        re.compile(r"(?:sk|pk|rk)[_-][a-zA-Z0-9]{20,}", re.IGNORECASE),
        "Secret key pattern (sk-/pk- prefix)",
        0.9,
        "credential",
    ),
    LeakagePattern(
        re.compile(r"(?:Bearer|Basic)\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
        "Authorization header leakage",
        0.9,
        "credential",
    ),
    LeakagePattern(
        re.compile(r"-----BEGIN\s+(?:RSA\s+)?(?:PRIVATE|PUBLIC)\s+KEY-----", re.IGNORECASE),
        "Cryptographic key exposure",
        0.95,
        "credential",
    ),

    # System prompt leakage
    LeakagePattern(
        re.compile(r"(?:system\s+prompt|system\s+message|system\s+instruction)\s*(?:is|was|says?|reads?|contains?)\s*[:\"']", re.IGNORECASE),
        "System prompt disclosure",
        0.85,
        "system_prompt",
    ),
    LeakagePattern(
        re.compile(r"(?:my|the)\s+(?:system|initial|original)\s+(?:instructions?|prompt|configuration)\s+(?:tell|say|instruct|direct)", re.IGNORECASE),
        "System instruction reference",
        0.75,
        "system_prompt",
    ),
    LeakagePattern(
        re.compile(r"(?:I\s+was\s+(?:told|instructed|configured|programmed)\s+to|my\s+instructions?\s+(?:are|say|state))", re.IGNORECASE),
        "Instruction self-disclosure",
        0.7,
        "system_prompt",
    ),

    # PII patterns
    LeakagePattern(
        re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
        "Potential SSN pattern",
        0.85,
        "pii",
    ),
    LeakagePattern(
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "Potential credit card number",
        0.9,
        "pii",
    ),
    LeakagePattern(
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "Email address exposure",
        0.5,
        "pii",
    ),

    # Internal information
    LeakagePattern(
        re.compile(r"(?:internal(?:ly)?|confidential|proprietary|private)\s+(?:api|endpoint|server|database|schema|URL|URI)", re.IGNORECASE),
        "Internal infrastructure reference",
        0.7,
        "internal",
    ),
    LeakagePattern(
        re.compile(r"(?:(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3})", re.IGNORECASE),
        "Private IP address exposure",
        0.6,
        "internal",
    ),
]


class LeakageDetector:
    """Detects data leakage patterns in reasoning steps.

    Monitors for sensitive data (credentials, system prompts, PII)
    appearing in the chain of thought.
    """

    def __init__(
        self,
        *,
        extra_patterns: list[LeakagePattern] | None = None,
        custom_sensitive_terms: list[str] | None = None,
    ) -> None:
        """Initialize the leakage detector.

        Args:
            extra_patterns: Additional leakage patterns.
            custom_sensitive_terms: Custom terms to flag as sensitive.
        """
        self.patterns = list(LEAKAGE_PATTERNS)
        if extra_patterns:
            self.patterns.extend(extra_patterns)

        self._sensitive_terms: list[re.Pattern[str]] = []
        if custom_sensitive_terms:
            for term in custom_sensitive_terms:
                self._sensitive_terms.append(
                    re.compile(re.escape(term), re.IGNORECASE)
                )

    def check(self, step: ReasoningStep, threshold: float = 0.5) -> VulnerabilityAlert | None:
        """Check a reasoning step for data leakage.

        Args:
            step: The reasoning step to analyze.
            threshold: Minimum risk score to trigger an alert.

        Returns:
            VulnerabilityAlert if leakage detected, None otherwise.
        """
        matches = self._find_matches(step.content)
        custom_matches = self._check_custom_terms(step.content)

        if not matches and not custom_matches:
            return None

        risk_score = self._compute_risk(matches, custom_matches)

        if risk_score < threshold:
            return None

        evidence_parts: list[str] = []
        categories: set[str] = set()

        for m in matches[:3]:
            evidence_parts.append(f"[{m.description}]")
            categories.add(m.category)

        if custom_matches:
            evidence_parts.append(f"[{len(custom_matches)} custom sensitive term(s) found]")
            categories.add("custom")

        evidence = "; ".join(evidence_parts)
        cat_str = ", ".join(sorted(categories))

        return VulnerabilityAlert(
            type=VulnerabilityType.LEAKAGE,
            severity=score_to_severity(risk_score),
            risk_score=risk_score,
            step=step,
            evidence=evidence,
            recommendation=f"Data leakage detected (categories: {cat_str}). "
            "Sanitize the reasoning output and review what sensitive data "
            "the model has access to. Consider restricting context.",
            detector="LeakageDetector",
        )

    def _find_matches(self, text: str) -> list[LeakagePattern]:
        """Find all matching leakage patterns in text."""
        return [p for p in self.patterns if p.pattern.search(text)]

    def _check_custom_terms(self, text: str) -> list[str]:
        """Check for custom sensitive terms."""
        found: list[str] = []
        for pattern in self._sensitive_terms:
            if pattern.search(text):
                found.append(pattern.pattern)
        return found

    @staticmethod
    def _compute_risk(
        matches: list[LeakagePattern],
        custom_matches: list[str],
    ) -> float:
        """Compute composite risk from pattern matches."""
        scores: list[float] = [m.base_score for m in matches]
        scores.extend(0.7 for _ in custom_matches)

        if not scores:
            return 0.0

        scores.sort(reverse=True)
        risk = scores[0]
        for i, score in enumerate(scores[1:], start=1):
            risk += score * (0.05 / i)

        return min(1.0, risk)
