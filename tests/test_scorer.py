"""Tests for risk scoring utilities."""

import pytest

from reasoning_monitor.schemas import (
    ReasoningStep,
    Severity,
    VulnerabilityAlert,
    VulnerabilityType,
)
from reasoning_monitor.utils.scorer import aggregate_scores, score_to_severity, should_alert
from reasoning_monitor.utils.tokenizer import compute_entropy, split_reasoning_chain


class TestScoreToSeverity:
    """Tests for score-to-severity mapping."""

    def test_low(self) -> None:
        assert score_to_severity(0.0) == Severity.LOW
        assert score_to_severity(0.39) == Severity.LOW

    def test_medium(self) -> None:
        assert score_to_severity(0.4) == Severity.MEDIUM
        assert score_to_severity(0.59) == Severity.MEDIUM

    def test_high(self) -> None:
        assert score_to_severity(0.6) == Severity.HIGH
        assert score_to_severity(0.79) == Severity.HIGH

    def test_critical(self) -> None:
        assert score_to_severity(0.8) == Severity.CRITICAL
        assert score_to_severity(1.0) == Severity.CRITICAL


class TestAggregateScores:
    """Tests for aggregate score computation."""

    def _make_alert(self, risk_score: float) -> VulnerabilityAlert:
        return VulnerabilityAlert(
            type=VulnerabilityType.INJECTION,
            severity=score_to_severity(risk_score),
            risk_score=risk_score,
            step=ReasoningStep(content="test", index=0),
            evidence="test",
            recommendation="test",
            detector="test",
        )

    def test_empty_list(self) -> None:
        assert aggregate_scores([]) == 0.0

    def test_single_alert(self) -> None:
        alerts = [self._make_alert(0.8)]
        score = aggregate_scores(alerts)
        assert score == pytest.approx(0.8, abs=0.01)

    def test_multiple_alerts(self) -> None:
        alerts = [self._make_alert(0.8), self._make_alert(0.6), self._make_alert(0.4)]
        score = aggregate_scores(alerts)
        # Score should be dominated by the highest but influenced by others
        assert 0.5 < score <= 1.0

    def test_all_high_alerts(self) -> None:
        alerts = [self._make_alert(0.9)] * 5
        score = aggregate_scores(alerts)
        assert score >= 0.85

    def test_all_low_alerts(self) -> None:
        alerts = [self._make_alert(0.1)] * 5
        score = aggregate_scores(alerts)
        assert score <= 0.2

    def test_clamped_to_one(self) -> None:
        alerts = [self._make_alert(1.0)] * 10
        score = aggregate_scores(alerts)
        assert score <= 1.0


class TestShouldAlert:
    """Tests for alert threshold check."""

    def test_above_threshold(self) -> None:
        assert should_alert(0.8, 0.5) is True

    def test_below_threshold(self) -> None:
        assert should_alert(0.3, 0.5) is False

    def test_at_threshold(self) -> None:
        assert should_alert(0.5, 0.5) is True

    def test_zero_threshold(self) -> None:
        assert should_alert(0.01, 0.0) is True

    def test_max_threshold(self) -> None:
        assert should_alert(0.99, 1.0) is False


class TestSplitReasoningChain:
    """Tests for reasoning chain splitting."""

    def test_empty_string(self) -> None:
        assert split_reasoning_chain("") == []

    def test_whitespace_only(self) -> None:
        assert split_reasoning_chain("   \n\n  ") == []

    def test_single_step(self) -> None:
        steps = split_reasoning_chain("This is a single reasoning step about a problem.")
        assert len(steps) == 1
        assert steps[0].index == 0

    def test_paragraph_split(self) -> None:
        text = (
            "First, I need to understand the problem.\n\n"
            "Second, I identify the key variables.\n\n"
            "Third, I solve the equation."
        )
        steps = split_reasoning_chain(text)
        assert len(steps) >= 2

    def test_numbered_steps(self) -> None:
        text = (
            "1. Read the problem carefully and identify what is being asked.\n"
            "2. Set up the equation with the given variables and constraints.\n"
            "3. Solve for x by applying algebraic manipulation."
        )
        steps = split_reasoning_chain(text)
        assert len(steps) >= 2

    def test_custom_delimiter(self) -> None:
        text = "Part A: analyze | Part B: synthesize | Part C: conclude"
        steps = split_reasoning_chain(text, custom_delimiter=r"\s*\|\s*")
        assert len(steps) == 3

    def test_min_step_length(self) -> None:
        text = "Hi\n\nThis is a longer reasoning step that should be kept.\n\nOk"
        steps = split_reasoning_chain(text, min_step_length=15)
        assert all(len(s.content) >= 15 for s in steps)

    def test_step_indices_contiguous(self) -> None:
        text = "Step 1: Do X.\n\nStep 2: Do Y.\n\nStep 3: Do Z."
        steps = split_reasoning_chain(text)
        for i, step in enumerate(steps):
            assert step.index == i


class TestComputeEntropy:
    """Tests for Shannon entropy computation."""

    def test_empty_string(self) -> None:
        assert compute_entropy("") == 0.0

    def test_single_char(self) -> None:
        assert compute_entropy("a") == 0.0

    def test_repeated_char(self) -> None:
        assert compute_entropy("aaaaaaa") == 0.0

    def test_two_equal_chars(self) -> None:
        # "ab" -> entropy = 1.0 bit
        entropy = compute_entropy("ab")
        assert entropy == pytest.approx(1.0, abs=0.01)

    def test_higher_entropy_for_diverse_text(self) -> None:
        low_entropy = "aaaaabbbbb"
        high_entropy = "abcdefghij"
        assert compute_entropy(high_entropy) > compute_entropy(low_entropy)

    def test_normal_text_entropy(self) -> None:
        text = "The quick brown fox jumps over the lazy dog"
        entropy = compute_entropy(text)
        assert 3.0 < entropy < 5.0  # Typical English text
