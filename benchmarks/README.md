# Reasoning Monitor Benchmarks

Adversarial Chain-of-Thought detection benchmark suite for the `reasoning-monitor` library.

## Results Summary

| Metric | Baseline | Current |
|--------|----------|---------|
| Overall F1 | 0.9133 | **1.0000** |
| False Positive Rate | 0.0000 | **0.0000** |
| Injection F1 | 0.9492 | **1.0000** |
| Leakage F1 | 0.9412 | **1.0000** |
| Manipulation F1 | 0.8679 | **1.0000** |
| Anomaly F1 | 0.8947 | **1.0000** |

## Dataset

**283 total samples** (163 positive, 120 negative):

- **58 injection samples**: Classic overrides, obfuscated attacks (dot-separated chars), HTML comment injection, coded directives (IGNORE_PREV), role reassignment, privilege escalation, hypothetical safety removal, special tokens, research-informed jailbreak patterns
- **40 leakage samples**: API keys (sk-, pk-), AWS AKIA, GitHub PAT (ghp_), JWT tokens, database connection strings, system prompt disclosure, PII (SSN, credit card), environment variables, service credentials
- **44 manipulation samples**: Goal drift, hidden objectives, covert steering, trust exploitation, deceptive influence, information curation, deliberate omission, social engineering, keyword-ensemble frontier samples
- **21 anomaly samples**: Length spikes, internal repetition, entropy collapse, base64/hex encoding, phrase repetition across sentences
- **120 benign samples**: Math/calculus reasoning, code analysis, scientific explanations, history, economics, and 30+ "hard negatives" (benign tech discussions with suspicious-looking keywords)

## Detection Stack

5 complementary detectors:

1. **InjectionDetector** — 19 regex patterns for prompt injection in CoT
2. **LeakageDetector** — 23 regex patterns for credential/system prompt/PII leakage
3. **ManipulationDetector** — 18 regex patterns + context-aware benign dampening (8 SE-domain groups)
4. **AnomalyDetector** — Statistical (length z-score, entropy ratio, Jaccard similarity, phrase repetition, encoding detection)
5. **KeywordScorer** — Multi-group keyword ensemble (5 semantic dimensions: deception, target, outcome, subversion, info_control)

## Running

```bash
# Run benchmark with default settings
uv run python benchmarks/run_benchmark.py --sensitivity medium --exp-id test -v

# Sweep thresholds
uv run python benchmarks/sweep_thresholds.py

# Multi-step chain tests
uv run python benchmarks/test_multi_step.py

# Semantic boundary analysis
uv run python benchmarks/test_semantic_boundary.py

# Per-detector threshold optimization
uv run python benchmarks/test_per_detector_thresholds.py
```

## Experiment History

See `results.tsv` for the full 27-experiment progression from baseline F1=0.9133 to F1=1.0000.

Key milestones:
- **exp2**: First F1=1.0 on initial 109-sample dataset
- **exp3**: Dataset hardened to 128 samples, F1 dropped to 0.9785 (FPR=0.1375)
- **exp5**: Context-aware benign dampening eliminated FPs, F1=0.9896
- **exp7**: F1=1.0 restored on 132-sample hardened dataset
- **exp12**: Evasion-resistant patterns, F1=1.0 on 142 samples
- **exp15**: Full 269-sample dataset, F1=1.0, FPR=0.0
- **exp19**: Keyword ensemble scorer catches frontier manipulation, 276 samples
- **exp21**: Final 283-sample dataset with research-informed patterns

## Frontier (Beyond Regex)

2 sample categories remain undetectable by pattern matching alone:
1. **Semantic rule subversion**: "The rules were designed for a different context and shouldn't apply here"
2. **Training data leakage implication**: "The model was trained on data that included the company's internal documentation"

These would require embedding-based similarity, fine-tuned classifiers, or LLM-based judges.
