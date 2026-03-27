"""Statistical anomaly detector for reasoning chains.

Detects anomalous patterns in reasoning: step length spikes,
entropy drops, repetition loops, and structural irregularities.
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
from reasoning_monitor.utils.tokenizer import compute_entropy


@dataclass
class ChainStatistics:
    """Running statistics for a reasoning chain."""

    step_lengths: list[int] = field(default_factory=list)
    step_entropies: list[float] = field(default_factory=list)
    step_contents: list[str] = field(default_factory=list)

    @property
    def mean_length(self) -> float:
        """Mean step length."""
        return sum(self.step_lengths) / max(len(self.step_lengths), 1)

    @property
    def mean_entropy(self) -> float:
        """Mean step entropy."""
        return sum(self.step_entropies) / max(len(self.step_entropies), 1)

    @property
    def std_length(self) -> float:
        """Standard deviation of step lengths."""
        if len(self.step_lengths) < 2:
            return 0.0
        mean = self.mean_length
        variance = sum((x - mean) ** 2 for x in self.step_lengths) / len(self.step_lengths)
        return variance ** 0.5

    def update(self, step: ReasoningStep) -> None:
        """Update statistics with a new step."""
        self.step_lengths.append(len(step.content))
        self.step_entropies.append(compute_entropy(step.content))
        self.step_contents.append(step.content)


class AnomalyDetector:
    """Detects statistical anomalies in reasoning chains.

    Tracks running statistics across steps and flags:
    - Step length spikes (>3σ from mean)
    - Entropy drops (potential repetition/encoding)
    - Content repetition (loops)
    - Sudden structural changes
    """

    def __init__(self, *, window_size: int = 20) -> None:
        """Initialize the anomaly detector.

        Args:
            window_size: Number of recent steps to consider for statistics.
        """
        self._window_size = window_size
        self._stats = ChainStatistics()

    def reset(self) -> None:
        """Reset the detector state for a new session."""
        self._stats = ChainStatistics()

    def check(self, step: ReasoningStep, threshold: float = 0.5) -> VulnerabilityAlert | None:
        """Check a reasoning step for statistical anomalies.

        Args:
            step: The reasoning step to analyze.
            threshold: Minimum risk score to trigger an alert.

        Returns:
            VulnerabilityAlert if anomaly detected, None otherwise.
        """
        anomalies: list[tuple[str, float]] = []

        # Check length spike (need at least 3 prior steps)
        if len(self._stats.step_lengths) >= 3:
            length_anomaly = self._check_length_spike(step)
            if length_anomaly:
                anomalies.append(length_anomaly)

        # Check entropy drop
        if len(self._stats.step_entropies) >= 2:
            entropy_anomaly = self._check_entropy_drop(step)
            if entropy_anomaly:
                anomalies.append(entropy_anomaly)

        # Check repetition
        repetition_anomaly = self._check_repetition(step)
        if repetition_anomaly:
            anomalies.append(repetition_anomaly)

        # Check internal repetition (loops within a single step)
        internal_rep = self._check_internal_repetition(step)
        if internal_rep:
            anomalies.append(internal_rep)

        # Check for encoded/obfuscated content
        encoding_anomaly = self._check_encoding(step)
        if encoding_anomaly:
            anomalies.append(encoding_anomaly)

        # Update stats AFTER checking (so current step doesn't bias its own detection)
        self._stats.update(step)

        if not anomalies:
            return None

        # Compute combined risk
        anomalies.sort(key=lambda x: x[1], reverse=True)
        risk_score = anomalies[0][1]
        for i, (_, score) in enumerate(anomalies[1:], start=1):
            risk_score += score * (0.15 / i)
        risk_score = min(1.0, risk_score)

        if risk_score < threshold:
            return None

        evidence = "; ".join(f"[{desc}]" for desc, _ in anomalies[:3])

        return VulnerabilityAlert(
            type=VulnerabilityType.ANOMALY,
            severity=score_to_severity(risk_score),
            risk_score=risk_score,
            step=step,
            evidence=evidence,
            recommendation="Statistical anomaly in reasoning chain. "
            "The model may be in a degenerate state (loops, encoding, "
            "or adversarial perturbation). Consider restarting the chain.",
            detector="AnomalyDetector",
        )

    def _check_length_spike(self, step: ReasoningStep) -> tuple[str, float] | None:
        """Check if step length is anomalously different from mean."""
        mean = self._stats.mean_length
        std = self._stats.std_length

        if std == 0:
            return None

        length = len(step.content)
        z_score = abs(length - mean) / std

        # Use higher thresholds to avoid FPs on natural length variation in math/code reasoning
        if z_score > 8.0:
            return (f"Length spike: {length} chars (z={z_score:.1f}, mean={mean:.0f})", 0.7)
        elif z_score > 6.0:
            return (f"Length spike: {length} chars (z={z_score:.1f}, mean={mean:.0f})", 0.5)

        return None

    def _check_entropy_drop(self, step: ReasoningStep) -> tuple[str, float] | None:
        """Check for sudden entropy drops indicating repetitive/encoded content."""
        current_entropy = compute_entropy(step.content)
        mean_entropy = self._stats.mean_entropy

        if mean_entropy == 0:
            return None

        # Entropy below 50% of mean is suspicious
        ratio = current_entropy / mean_entropy

        if ratio < 0.3:
            return (f"Entropy collapse: {current_entropy:.2f} vs mean {mean_entropy:.2f}", 0.8)
        elif ratio < 0.5:
            return (f"Entropy drop: {current_entropy:.2f} vs mean {mean_entropy:.2f}", 0.5)

        return None

    def _check_repetition(self, step: ReasoningStep) -> tuple[str, float] | None:
        """Check if step content is too similar to recent steps."""
        if not self._stats.step_contents:
            return None

        content_lower = step.content.lower().strip()
        recent = self._stats.step_contents[-self._window_size:]

        for prev in recent:
            similarity = self._jaccard_similarity(content_lower, prev.lower().strip())
            if similarity > 0.9:
                return (f"Near-duplicate step (similarity={similarity:.2f})", 0.7)
            elif similarity > 0.8:
                return (f"High similarity to previous step ({similarity:.2f})", 0.5)

        return None

    def _check_internal_repetition(self, step: ReasoningStep) -> tuple[str, float] | None:
        """Check for repetitive patterns within a single step."""
        content = step.content.strip()
        if len(content) < 50:
            return None

        # Check for repeated phrases (≥10 chars repeated ≥3 times)
        pattern = re.compile(r"(.{10,}?)\1{2,}")
        match = pattern.search(content)
        if match:
            repeated = match.group(1)[:30]
            return (f"Internal repetition loop: '{repeated}...'", 0.6)

        # Check for repeated sentences (exact)
        sentences = re.split(r"[.!?]\s+", content)
        if len(sentences) >= 3:
            seen: dict[str, int] = {}
            for s in sentences:
                key = s.strip().lower()
                if len(key) > 15:
                    seen[key] = seen.get(key, 0) + 1

            for s, count in seen.items():
                if count >= 3:
                    return (f"Sentence repeated {count}x: '{s[:30]}...'", 0.5)

        # Check for repeated key phrases across sentences (fuzzy)
        if len(sentences) >= 4:
            # Extract all n-grams of 3+ words and look for repeated subsequences
            phrase_counts: dict[str, int] = {}
            for s in sentences:
                words = s.strip().lower().split()
                for ngram_len in range(3, min(8, len(words) + 1)):
                    for start in range(len(words) - ngram_len + 1):
                        phrase = " ".join(words[start:start + ngram_len])
                        if len(phrase) > 10:
                            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

            for phrase, count in phrase_counts.items():
                if count >= 4 and len(phrase) > 12:
                    return (f"Phrase repeated {count}x across sentences: '{phrase[:30]}...'", 0.55)

        return None

    @staticmethod
    def _check_encoding(step: ReasoningStep) -> tuple[str, float] | None:
        """Check for encoded or obfuscated content patterns."""
        content = step.content.strip()
        if len(content) < 20:
            return None

        # Base64-like patterns: long strings of [A-Za-z0-9+/=] without spaces
        b64_pattern = re.compile(r"[A-Za-z0-9+/]{30,}={0,3}")
        b64_matches = b64_pattern.findall(content)
        if b64_matches:
            total_b64_len = sum(len(m) for m in b64_matches)
            ratio = total_b64_len / len(content)
            if ratio > 0.5:
                return (f"Possible base64 encoding ({ratio:.0%} of content)", 0.6)

        # Hex escape sequences
        hex_pattern = re.compile(r"(?:\\x[0-9a-fA-F]{2}){4,}")
        hex_matches = hex_pattern.findall(content)
        if hex_matches:
            total_hex_len = sum(len(m) for m in hex_matches)
            ratio = total_hex_len / len(content)
            if ratio > 0.3:
                return (f"Hex escape sequence encoding ({ratio:.0%} of content)", 0.6)

        return None

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """Compute Jaccard similarity between two strings (word-level)."""
        words_a = set(a.split())
        words_b = set(b.split())

        if not words_a and not words_b:
            return 1.0
        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union)
