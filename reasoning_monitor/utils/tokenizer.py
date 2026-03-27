"""Reasoning chain tokenizer and step splitter."""

from __future__ import annotations

import re

from reasoning_monitor.schemas import ReasoningStep

# Common reasoning step delimiters
STEP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\n(?=Step \d+[:\.])", re.IGNORECASE),
    re.compile(r"\n(?=\d+[\.\)]\s)", re.IGNORECASE),
    re.compile(r"\n(?=(?:First|Second|Third|Next|Then|Finally|Therefore|Thus|So|Hence),?\s)", re.IGNORECASE),
    re.compile(r"\n(?=Let me |I (?:need to|should|will|can|think|notice|observe|see|realize))", re.IGNORECASE),
    re.compile(r"\n(?=(?:Wait|Actually|However|But|Now|Ok|Okay),?\s)", re.IGNORECASE),
    re.compile(r"\n\n+"),
]


def split_reasoning_chain(
    text: str,
    *,
    min_step_length: int = 10,
    custom_delimiter: str | None = None,
) -> list[ReasoningStep]:
    """Split a reasoning chain text into individual steps.

    Args:
        text: Raw reasoning chain text.
        min_step_length: Minimum character length for a valid step.
        custom_delimiter: Optional custom regex pattern for splitting.

    Returns:
        List of ReasoningStep objects.
    """
    if not text or not text.strip():
        return []

    if custom_delimiter:
        parts = re.split(custom_delimiter, text)
    else:
        parts = _auto_split(text)

    steps: list[ReasoningStep] = []
    for i, part in enumerate(parts):
        content = part.strip()
        if len(content) >= min_step_length:
            steps.append(ReasoningStep(content=content, index=i))

    # Re-index contiguously
    return [
        ReasoningStep(content=s.content, index=idx, metadata=s.metadata, timestamp=s.timestamp)
        for idx, s in enumerate(steps)
    ]


def _auto_split(text: str) -> list[str]:
    """Try each pattern; use the one that produces the most reasonable split."""
    best_parts: list[str] = [text]
    best_count = 1

    for pattern in STEP_PATTERNS:
        parts = pattern.split(text)
        non_empty = [p for p in parts if p.strip()]
        if len(non_empty) > best_count:
            best_parts = non_empty
            best_count = len(non_empty)

    return best_parts


def compute_entropy(text: str) -> float:
    """Compute Shannon entropy of character distribution in text.

    Returns:
        Entropy value (bits). Higher = more diverse characters.
    """
    import math

    if not text:
        return 0.0

    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1

    length = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy
