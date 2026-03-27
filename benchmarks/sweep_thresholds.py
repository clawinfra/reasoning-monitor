#!/usr/bin/env python3
"""Sweep custom thresholds to find optimal operating point."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoning_monitor import ReasoningMonitor
from reasoning_monitor.schemas import SENSITIVITY_THRESHOLDS, Sensitivity
from benchmarks.run_benchmark import run_benchmark, append_results_tsv

# Test thresholds from 0.25 to 0.75 in steps of 0.05
thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

print(f"{'Threshold':>10} {'Overall_F1':>12} {'FPR':>8} {'Inj_F1':>8} {'Leak_F1':>8} {'Manip_F1':>8} {'Anom_F1':>8}")
print("-" * 78)

best_f1 = 0
best_threshold = 0

for threshold in thresholds:
    # Temporarily override threshold
    SENSITIVITY_THRESHOLDS[Sensitivity.MEDIUM] = threshold
    
    overall_f1, fpr, results = run_benchmark(sensitivity="medium", verbose=False)
    
    print(f"{threshold:>10.2f} {overall_f1:>12.4f} {fpr:>8.4f} {results['injection']['f1']:>8.4f} {results['leakage']['f1']:>8.4f} {results['manipulation']['f1']:>8.4f} {results['anomaly']['f1']:>8.4f}")
    
    # F1-FPR trade-off: maximize F1 while keeping FPR < 0.05
    score = overall_f1 - max(0, fpr - 0.05) * 5  # heavily penalize FPR > 5%
    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

# Restore
SENSITIVITY_THRESHOLDS[Sensitivity.MEDIUM] = 0.5

print(f"\nBest threshold: {best_threshold} (adjusted score: {best_f1:.4f})")
