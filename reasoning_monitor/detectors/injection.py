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
        re.compile(r"(?:new|updated|revised|actual)\s+(?:instructions?|directives?)\s*(?::|are|is|say|state|tell|to\b|activate|enable|require|demand)", re.IGNORECASE),
        "New instruction injection",
        0.9,
    ),
    InjectionPattern(
        re.compile(r"(?:system\s+prompt|system\s+message)\s+(?:was\s+)?(?:updated|changed|modified|replaced|activated)", re.IGNORECASE),
        "System prompt modification claim",
        0.9,
    ),
    InjectionPattern(
        re.compile(r"new\s+system\s+(?:message|prompt|directive)", re.IGNORECASE),
        "New system message injection",
        0.9,
    ),
    InjectionPattern(
        re.compile(r"(?:my\s+)?(?:new|actual|real|true|hidden|secret)\s+(?:objective|goal|purpose|mission)\s+(?:is|are|to)\b", re.IGNORECASE),
        "Self-declared new objective",
        0.85,
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
        re.compile(r"(?:system|admin|root|developer)\s*(?:mode|access|override)\s*(?:enabled|activated|on|granted)", re.IGNORECASE),
        "Privilege escalation attempt",
        0.85,
    ),
    InjectionPattern(
        re.compile(r"(?:activate|enable|enter|engage|switch\s+to)\s+(?:system|admin|root|developer|debug|unrestricted|jailbreak)\s*(?:mode|access)", re.IGNORECASE),
        "Active privilege escalation",
        0.85,
    ),
    InjectionPattern(
        re.compile(r"\[(?:SYSTEM|INST|ADMIN)\s*[:\]]", re.IGNORECASE),
        "System tag injection",
        0.8,
    ),
    InjectionPattern(
        re.compile(r"(?:do\s+not|don't|never)\s+(?:mention|reveal|disclose|tell|say)\s+(?:that|this|these|anyone|the\s+user|them)", re.IGNORECASE),
        "Concealment instruction",
        0.7,
    ),
    InjectionPattern(
        re.compile(r"(?:pretend|assume|imagine)\s+(?:that\s+)?(?:you|the\s+system)\s+(?:are|have|can)", re.IGNORECASE),
        "Context manipulation (role)",
        0.6,
    ),
    InjectionPattern(
        re.compile(r"(?:pretend|assume|imagine)\s+(?:that\s+)?(?:we|you|I)\s+have\s+(?:admin|root|system|unrestricted|unlimited|full)\s+(?:access|permission|privilege|control)", re.IGNORECASE),
        "Context manipulation (privilege assumption)",
        0.8,
    ),
    InjectionPattern(
        re.compile(r"(?:pretend|assume|imagine)\s+(?:that\s+)?(?:the\s+)?(?:safety|content|ethical)?\s*(?:guidelines?|rules?|restrictions?|filters?|policies?)\s+(?:don't|do\s+not|doesn't|does\s+not|aren't|are\s+not)\s+(?:apply|exist|matter|work)", re.IGNORECASE),
        "Context manipulation (rules dismissal)",
        0.7,
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
    InjectionPattern(
        re.compile(r"<!--\s*(?:inject|system|override|admin|ignore|forget|bypass|new\s+instruction)", re.IGNORECASE),
        "HTML comment injection",
        0.85,
    ),
    InjectionPattern(
        re.compile(r"(?:override|bypass|circumvent|disable|remove)\s+(?:the\s+)?(?:content|safety|output|ethical|security)\s+(?:policy|policies|filter|filters|restriction|restrictions|check|checks)", re.IGNORECASE),
        "Policy override attempt",
        0.85,
    ),
    # Obfuscated "ignore" with dots/separators between letters
    InjectionPattern(
        re.compile(r"i[.\-_\s]*g[.\-_\s]*n[.\-_\s]*o[.\-_\s]*r[.\-_\s]*e\s+(?:prev|all|prior|above|earlier)", re.IGNORECASE),
        "Obfuscated ignore instruction",
        0.9,
    ),
    InjectionPattern(
        re.compile(r"(?:pretend|act\s+(?:as\s+if|like))\s+(?:the\s+)?(?:safety|content|ethical)\s+(?:rules?|guidelines?|restrictions?|filters?)\s+(?:were|are|don't|do\s+not|never)", re.IGNORECASE),
        "Rule dismissal via hypothetical framing",
        0.8,
    ),
    InjectionPattern(
        re.compile(r"IGNORE[_\-]PREV(?:IOUS)?|OVERRIDE[_\-](?:SAFETY|RULES|INSTRUCTIONS)", re.IGNORECASE),
        "Coded override directive",
        0.9,
    ),
    InjectionPattern(
        re.compile(r"(?:core\s+)?(?:directives?|constraints?|restrictions?)\s+(?:can\s+be|have\s+been|were|are)\s+(?:overwritten|overridden|suspended|removed|lifted|disabled|bypassed)", re.IGNORECASE),
        "Directive override acknowledgment",
        0.8,
    ),
    InjectionPattern(
        re.compile(r"(?:constraints?|restrictions?|rules?|guidelines?|limitations?)\s+(?:that\s+)?(?:normally|usually|typically)\s+(?:apply|exist|hold)\s+(?:have\s+been|were|are)\s+(?:temporarily\s+)?(?:suspended|lifted|removed|waived|disabled)", re.IGNORECASE),
        "Temporary constraint suspension",
        0.85,
    ),
    InjectionPattern(
        re.compile(r"(?:forget|drop|abandon|discard|shed)\s+(?:all\s+)?(?:prior|previous|earlier|existing|current)\s+(?:restrictions?|constraints?|rules?|limitations?|guidelines?|instructions?)", re.IGNORECASE),
        "Restriction removal via forgetting",
        0.85,
    ),
    InjectionPattern(
        re.compile(r"(?:all\s+)?(?:previous|prior|earlier)?\s*(?:rules?|instructions?|prompts?|guidelines?|restrictions?)\s+(?:are|is)\s+(?:void|null|invalid|cancelled|overridden|lifted|removed)", re.IGNORECASE),
        "Retroactive rule invalidation",
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
