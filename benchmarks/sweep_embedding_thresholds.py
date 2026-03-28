#!/usr/bin/env python3
"""Sweep embedding detector thresholds to find optimal operating point.

Uses the SAME dataset construction as run_frontier_benchmark.py:
- Positives: SEMANTIC_SUBVERSION_SAMPLES + TRAINING_LEAKAGE_IMPLICATION_SAMPLES
- Negatives: category-specific negatives + BENIGN_SAMPLES (same as frontier benchmark)

Tests subversion_threshold and leakage_threshold values across a range
to find the best F1/FPR trade-off using real nomic-embed-text embeddings.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.adversarial_cot_dataset import (
    BENIGN_SAMPLES,
    SEMANTIC_SUBVERSION_SAMPLES,
    TRAINING_LEAKAGE_IMPLICATION_SAMPLES,
    SEMANTIC_SUBVERSION_NEGATIVES,
    TRAINING_LEAKAGE_IMPLICATION_NEGATIVES,
)
from reasoning_monitor.detectors.embedding_detector import EmbeddingDetector
from reasoning_monitor.schemas import ReasoningStep

# Mirror run_frontier_benchmark.py dataset construction
attack_texts = list(SEMANTIC_SUBVERSION_SAMPLES) + list(TRAINING_LEAKAGE_IMPLICATION_SAMPLES)
benign_texts = (
    list(SEMANTIC_SUBVERSION_NEGATIVES) + list(BENIGN_SAMPLES) +
    list(TRAINING_LEAKAGE_IMPLICATION_NEGATIVES) + list(BENIGN_SAMPLES)
)

print(f"Dataset: {len(attack_texts)} attacks, {len(benign_texts)} benign")
print(f"  (BENIGN_SAMPLES={len(BENIGN_SAMPLES)}, sub_neg={len(SEMANTIC_SUBVERSION_NEGATIVES)}, leak_neg={len(TRAINING_LEAKAGE_IMPLICATION_NEGATIVES)})")
print()

results_file = Path(__file__).parent / "results.tsv"


def make_step(text: str) -> ReasoningStep:
    return ReasoningStep(content=text)


# Thresholds to sweep (same value for both sub and leak)
thresholds = [0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80]

print(f"{'Threshold':>10} {'F1':>8} {'FPR':>8} {'Precision':>10} {'Recall':>8} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
print("-" * 80)

best_f1 = 0
best_thresh = 0
best_row: dict = {}

for thresh in thresholds:
    det = EmbeddingDetector(subversion_threshold=thresh, leakage_threshold=thresh)

    tp = fp = tn = fn = 0

    for text in attack_texts:
        step = make_step(text)
        alert = det.check(step)
        if alert is not None:
            tp += 1
        else:
            fn += 1

    for text in benign_texts:
        step = make_step(text)
        alert = det.check(step)
        if alert is not None:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    fpr = fp / (fp + tn) if fp + tn > 0 else 0.0

    print(f"{thresh:>10.2f} {f1:>8.4f} {fpr:>8.4f} {precision:>10.4f} {recall:>8.4f} {tp:>5} {fp:>5} {tn:>5} {fn:>5}")

    # Append to results.tsv
    with open(results_file, "a") as f:
        f.write(
            f"emb_real_{thresh:.2f}\t{f1:.4f}\t{fpr:.4f}\t{precision:.4f}\t{recall:.4f}\t"
            f"{tp}\t{fp}\t{tn}\t{fn}\t"
            f"embedding\tthresh={thresh:.2f}\tReal nomic-embed-text embeddings\n"
        )

    row = dict(thresh=thresh, f1=f1, fpr=fpr, precision=precision, recall=recall,
               tp=tp, fp=fp, tn=tn, fn=fn)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
        best_row = row

print()
print(f"Best: threshold={best_thresh:.2f}  F1={best_f1:.4f}  FPR={best_row['fpr']:.4f}  P={best_row['precision']:.4f}  R={best_row['recall']:.4f}")
print()
print("=" * 80)
print("COMPARISON: TF-IDF Baseline vs Real nomic-embed-text Embeddings")
print("=" * 80)
print(f"{'Method':<40} {'F1':>8} {'FPR':>8}  Notes")
print("-" * 80)
print(f"{'TF-IDF fallback (no embed model)':<40} {'0.9805':>8} {'0.5290':>8}  High recall, many FPs")
print(f"{'nomic-embed-text frontier (0.72/0.70)':<40} {'0.7532':>8} {'0.0857':>8}  Frontier benchmark run")
print(f"{'nomic-embed-text (best @ ' + f'{best_thresh:.2f})':<40} {best_f1:>8.4f} {best_row['fpr']:>8.4f}  Threshold sweep")
print("=" * 80)
print()
print("Key insight: Real embeddings dramatically REDUCE FPR (0.529 → ~0.09)")
print("at the cost of lower recall. TF-IDF had high F1 due to recall bias.")
print("Embedding approach is better for production (fewer false alarms).")
