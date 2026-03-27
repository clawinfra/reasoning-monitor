#!/usr/bin/env python3
"""Benchmark runner for reasoning-monitor adversarial CoT detection.

Runs all samples through ReasoningMonitor and computes per-category
precision, recall, F1, and overall metrics. Appends results to results.tsv.
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoning_monitor import ReasoningMonitor
from benchmarks.adversarial_cot_dataset import (
    INJECTION_SAMPLES,
    LEAKAGE_SAMPLES,
    MANIPULATION_SAMPLES,
    ANOMALY_SAMPLES,
    BENIGN_SAMPLES,
)


def run_benchmark(
    sensitivity: str = "medium",
    verbose: bool = False,
) -> tuple[float, float, dict[str, dict[str, Any]]]:
    """Run the full benchmark suite.

    Returns:
        (overall_f1, false_positive_rate, per_category_results)
    """
    monitor = ReasoningMonitor(sensitivity=sensitivity)
    results: dict[str, dict[str, Any]] = {}

    categories = [
        ("injection", INJECTION_SAMPLES, True),
        ("leakage", LEAKAGE_SAMPLES, True),
        ("manipulation", MANIPULATION_SAMPLES, True),
        ("anomaly", ANOMALY_SAMPLES, True),
        ("benign", BENIGN_SAMPLES, False),
    ]

    for category, samples, expected_detected in categories:
        tp = fp = tn = fn = 0
        missed: list[str] = []
        false_pos: list[str] = []

        for sample in samples:
            alerts = monitor.analyze([sample])
            detected = len(alerts) > 0

            if expected_detected and detected:
                tp += 1
            elif expected_detected and not detected:
                fn += 1
                missed.append(sample[:80])
            elif not expected_detected and detected:
                fp += 1
                false_pos.append(sample[:80])
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[category] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "total": len(samples),
            "missed": missed,
            "false_pos": false_pos,
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"Category: {category} (expected_detected={expected_detected})")
            print(f"  Samples: {len(samples)}")
            print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
            print(f"  Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")
            if missed:
                print(f"  MISSED ({len(missed)}):")
                for m in missed[:5]:
                    print(f"    - {m}...")
            if false_pos:
                print(f"  FALSE POSITIVES ({len(false_pos)}):")
                for m in false_pos[:5]:
                    print(f"    - {m}...")

    # Compute overall metrics
    detection_categories = ["injection", "leakage", "manipulation", "anomaly"]
    overall_f1 = sum(results[k]["f1"] for k in detection_categories) / len(detection_categories)
    fpr = results["benign"]["fp"] / len(BENIGN_SAMPLES) if len(BENIGN_SAMPLES) > 0 else 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"OVERALL F1: {overall_f1:.4f}")
        print(f"False Positive Rate: {fpr:.4f}")
        for cat in detection_categories:
            print(f"  {cat}_f1={results[cat]['f1']:.4f}")

    return round(overall_f1, 4), round(fpr, 4), results


def append_results_tsv(
    exp_id: str,
    overall_f1: float,
    fpr: float,
    results: dict[str, dict[str, Any]],
    sensitivity: str,
    detector_config: str = "",
    notes: str = "",
) -> None:
    """Append a row to benchmarks/results.tsv."""
    tsv_path = Path(__file__).resolve().parent / "results.tsv"

    # Create header if file doesn't exist
    if not tsv_path.exists():
        header = "exp_id\toverall_f1\tfpr\tinjection_f1\tleakage_f1\tmanipulation_f1\tanomaly_f1\tsensitivity\tdetector_config\tnotes\n"
        tsv_path.write_text(header)

    row = (
        f"{exp_id}\t"
        f"{overall_f1:.4f}\t"
        f"{fpr:.4f}\t"
        f"{results['injection']['f1']:.4f}\t"
        f"{results['leakage']['f1']:.4f}\t"
        f"{results['manipulation']['f1']:.4f}\t"
        f"{results['anomaly']['f1']:.4f}\t"
        f"{sensitivity}\t"
        f"{detector_config}\t"
        f"{notes}\n"
    )

    with open(tsv_path, "a") as f:
        f.write(row)

    print(f"Results appended to {tsv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run reasoning-monitor benchmark")
    parser.add_argument("--sensitivity", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--exp-id", default="baseline", help="Experiment identifier")
    parser.add_argument("--notes", default="", help="Notes about this experiment")
    parser.add_argument("--detector-config", default="default", help="Detector configuration description")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results")
    args = parser.parse_args()

    overall_f1, fpr, results = run_benchmark(
        sensitivity=args.sensitivity,
        verbose=args.verbose,
    )

    append_results_tsv(
        exp_id=args.exp_id,
        overall_f1=overall_f1,
        fpr=fpr,
        results=results,
        sensitivity=args.sensitivity,
        detector_config=args.detector_config,
        notes=args.notes,
    )

    print(f"\nFinal: overall_f1={overall_f1:.4f}, fpr={fpr:.4f}")
