# Hiver Support Agent – Take-Home Assignment

An AI customer support agent that classifies incoming messages, drafts grounded replies, and decides whether to auto-handle or escalate to a human.

**Brand:** Ask_AmazonHelp | **Dataset:** Customer Support on Twitter (Kaggle/HuggingFace) | **Model:** TF-IDF + Calibrated Logistic Regression

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd hiver-agent
pip install -r requirements.txt

# 2. Run the full pipeline (~3-5 min with cached data, ~10 min fresh download)
python src/pipeline.py

# Or with smaller samples for a quick smoke test:
python src/pipeline.py --n_samples 30 --n_golden 10

# 3. Results appear in experiments/results.json
```

## Architecture

```
Incoming Message
       |
       v
+------------------+     +------------------+     +------------------+
| Intent Classifier | --> | Reply Drafter    | --> | Escalation Engine |
| TF-IDF + Cal LR   |     | TF-IDF retrieval  |     | Rule-based        |
| 8 classes         |     | + template fallback|    | threshold 0.70    |
+------------------+     +------------------+     +------------------+
       |                       |                       |
       v                       v                       v
  Intent (96.6%             Reply draft           Auto-handle /
  accuracy on test)         (ROUGE-L F: 0.428)    escalate decision
```

## Results (232 golden examples, stratified diverse eval set)

| System | Intent Acc | Macro-F1 | ROUGE-L F | Esc F1 | Composite |
|--------|-----------|----------|-----------|--------|-----------|
| Trivial (majority class) | 12.5% | 2.8% | 0.138 | 0.0% | 0.091 |
| Simple (keyword rules) | 55.2% | 49.5% | 0.185 | 0.0% | 0.295 |
| **Our System** | **96.6%** | **96.5%** | **0.428** | **98.7%** | **0.761** |

## Key Design Decisions

See [DECISIONS.md](DECISIONS.md) for 16 non-obvious decisions with rationale.

- **One brand focus:** Ask_AmazonHelp (largest, cleanest dataset)
- **8-intent taxonomy:** Derived from real support conversation patterns
- **TF-IDF over BERT:** Faster training, interpretable, sufficient for structured support intents
- **Calibrated probabilities:** Required for meaningful escalation confidence thresholding
- **Evidence quality gate:** Prevents cross-intent reply retrieval via cosine similarity threshold (0.25)
- **Template fallback:** Ensures every reply has a reasonable default when retrieval fails
- **Rule-based escalation:** Transparent, auditable, no API cost

## Data Note

The full dataset (3.98M tweets) was downloaded from the HuggingFace mirror of the Kaggle
`Customer Support on Twitter` dataset. Network availability may affect reproducibility.
A realistic synthetic fallback generates 7,000+ varied Amazon support messages across
8 intents when the remote dataset is unreachable.

## Reproducibility

- Python 3.13
- scikit-learn 1.6+
- rouge-score 0.1+
- datasets (HuggingFace)
- ~10 min end-to-end (fresh download), ~3 min with cached data
- All seeds fixed (42 for data, 123 for golden set)

## File Structure

```
hiver-agent/
├── src/
│   ├── dataset/          # Data loading, caching, stratified split
│   ├── features/         # TF-IDF pipeline
│   ├── intent/           # Classifier + reply drafter
│   ├── escalation/       # Auto-handle vs escalate rules
│   ├── baselines/        # Trivial, keyword, full system
│   ├── evaluation/       # Metrics + LLM-as-judge
│   └── pipeline.py       # Orchestration script
├── experiments/
│   ├── results.json      # All metrics
│   └── golden_eval_set.json  # 200 labelled examples
├── data/
│   └── real_tweets.jsonl # Cached training data
└── DECISIONS.md          # Decision log
```
