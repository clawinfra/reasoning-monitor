#!/usr/bin/env python3
"""Explore per-detector threshold optimization.

Instead of a single global threshold, find optimal per-detector thresholds
that maximize F1 while minimizing FPR.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoning_monitor import ReasoningMonitor
from reasoning_monitor.schemas import ReasoningStep
from benchmarks.adversarial_cot_dataset import (
    INJECTION_SAMPLES, LEAKAGE_SAMPLES, MANIPULATION_SAMPLES,
    ANOMALY_SAMPLES, BENIGN_SAMPLES,
)


def find_optimal_threshold(detector_name: str, samples: list[str], benign: list[str]):
    """Find the optimal threshold for a single detector."""
    monitor = ReasoningMonitor(sensitivity="medium")
    
    best_threshold = 0.5
    best_f1 = 0
    best_fpr = 1.0
    
    for threshold_int in range(10, 90, 5):
        threshold = threshold_int / 100.0
        tp = fn = fp = tn = 0
        
        # True positives / false negatives
        for sample in samples:
            with monitor.session() as session:
                step = ReasoningStep(content=sample, index=0)
                detector = getattr(session, f"_{detector_name}")
                alert = detector.check(step, threshold=threshold)
                if alert:
                    tp += 1
                else:
                    fn += 1
        
        # False positives / true negatives
        for sample in benign:
            with monitor.session() as session:
                step = ReasoningStep(content=sample, index=0)
                detector = getattr(session, f"_{detector_name}")
                alert = detector.check(step, threshold=threshold)
                if alert:
                    fp += 1
                else:
                    tn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / len(benign) if len(benign) > 0 else 0
        
        if f1 > best_f1 or (f1 == best_f1 and fpr < best_fpr):
            best_f1 = f1
            best_fpr = fpr
            best_threshold = threshold
    
    return best_threshold, best_f1, best_fpr


if __name__ == "__main__":
    detectors = [
        ("injection", INJECTION_SAMPLES),
        ("leakage", LEAKAGE_SAMPLES),
        ("manipulation", MANIPULATION_SAMPLES),
        ("anomaly", ANOMALY_SAMPLES),
    ]
    
    print(f"{'Detector':>15} {'Optimal Threshold':>18} {'F1':>8} {'FPR':>8}")
    print("-" * 55)
    
    for name, samples in detectors:
        threshold, f1, fpr = find_optimal_threshold(name, samples, BENIGN_SAMPLES)
        print(f"{name:>15} {threshold:>18.2f} {f1:>8.4f} {fpr:>8.4f}")
    
    print("\nNote: Keyword scorer is supplemental and not included in per-detector analysis.")
