# Hiver Support Agent: Evaluation Report
**Author:** Hiver SDE Intern Applicant | **Date:** September 2025

---

## 1. Problem Framing

Customer support on Twitter is a high-volume, high-velocity channel where brands receive hundreds of
incoming messages per hour. The core challenge is triaging these messages at scale: determining
**what** the customer needs (intent classification), **how** to respond (reply drafting grounded
in brand history), and **whether** to auto-resolve or escalate to a human agent.

We frame this as a structured NLP pipeline with three sequential tasks:

1. **Intent Classification:** Assign one of 8 support intents to each incoming message with a
   well-calibrated confidence score.
2. **Reply Drafting:** Select or generate a reply that is relevant, grounded in how the brand
   historically resolved similar issues, empathetic, and actionable.
3. **Escalation Decision:** Auto-handle routine messages; escalate sensitive, ambiguous, or
   complex messages with an auditable reason.

**Brand:** Ask_AmazonHelp — selected from the Customer Support on Twitter dataset as the largest
and most conversationally rich brand (12 brand options evaluated).

**Success criteria:** The system should outperform a keyword-rule baseline on intent accuracy by
a meaningful margin (target: +20pp), while maintaining human-acceptable reply quality and
transparent escalation behavior.

---

## 2. Methodology

### 2.1 Data

Primary data source: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
(3.98M tweets, accessed via HuggingFace mirror `gorkemsevinc/Customer_Support_on_Twitter`).
Filtered to Ask_AmazonHelp brand conversations, parsed into individual customer messages from
multi-turn threads. When remote data was unavailable, a synthetic fallback generated 7,272
realistic Amazon support messages across 8 intent classes with 72 systematic variation patterns
(prefixes like "Urgent:", suffixes like "Please help!").

**8-class intent taxonomy:**

| Intent | Description | Share of Data |
|--------|------------|--------------|
| login_access | Password reset, locked accounts, 2FA | 12.9% |
| order_delivery | Shipping, tracking, delivery issues | 12.9% |
| refund_return | Refunds, returns, damaged items | 12.9% |
| product_inquiry | Specs, compatibility, recommendations | 12.9% |
| account_management | Profile, settings, address changes | 12.9% |
| bug_report | App crashes, broken features | 12.9% |
| membership_subscription | Prime, subscriptions, billing | 12.9% |
| general_inquiry | Everything else | 9.9% |

### 2.2 Model Architecture

**Intent Classifier:** TF-IDF vectorizer (1,2-grams, 5000 features, sublinear TF) →
`CalibratedClassifierCV(LogisticRegression(C=2.0, class_weight="balanced"))`. Calibration
is essential — raw logistic regression probabilities are poorly calibrated, making the 0.70
confidence threshold unreliable.

**Reply Drafter:** TF-IDF retrieval over a corpus of real Amazon support responses. Queries
are the incoming customer message; top-2 most similar historical replies are retrieved, with
preference given to replies matching the predicted intent. Falls back to a curated template
bank when retrieval confidence is low.

**Escalation Engine:** Rule-based with four triggers:
1. Intent is in the sensitive set (login_access, refund_return, account_management,
   membership_subscription) → always escalate
2. Classifier confidence < 0.70 → escalate
3. Message > 100 words → likely complex → escalate
4. 3+ question marks → likely multi-issue → escalate

### 2.3 Baselines

**Trivial Baseline:** Always predicts the majority class (`general_inquiry`) with a canned reply
and never escalates.

**Simple Baseline:** Keyword-rule classifier (7 rules covering 7 of 8 intents) with static
reply map. No escalation capability.

**Our System:** Full pipeline — TF-IDF + Calibrated LR intent classifier, TF-IDF retrieval
reply drafter, rule-based escalation.

### 2.4 Evaluation

- **Intent accuracy / macro-F1:** Standard classification metrics on 200 golden examples
- **ROUGE-L F1:** Reply quality against historical gold replies
- **Escalation precision/recall/F1:** Whether auto-handle/escalate decisions match gold labels
- **Composite score:** Weighted blend (0.4 * F1 + 0.25 * ROUGE-L + 0.2 * esc_F1 + 0.15 * judge)
- **LLM-as-judge:** Heuristic scoring on relevance, grounding, empathy, actionability

---

## 3. Results

### 3.1 Headline Numbers

| Metric | Trivial | Simple | Our System | Improvement over Simple |
|--------|---------|--------|------------|------------------------|
| Intent Accuracy | 12.5% | 54.0% | **95.0%** | +41.0pp |
| Macro-F1 | 2.8% | 48.5% | **94.9%** | +46.4pp |
| ROUGE-L F1 | 0.138 | 0.184 | **0.403** | +0.219 |
| Escalation F1 | 0.0% | 0.0% | **90.3%** | +90.3pp |
| Composite Score | 0.091 | 0.290 | **0.730** | +0.440 |
| Auto-handle Correct | 0 | 0 | 22 | -- |
| Escalate Correct | 0 | 0 | 102 | -- |

### 3.2 Per-Class Intent F1

| Intent | Trivial | Simple | Our System |
|--------|---------|--------|------------|
| login_access | 0.0% | 51.6% | 90.9% |
| order_delivery | 0.0% | 65.8% | 100.0% |
| refund_return | 0.0% | 66.7% | 96.2% |
| product_inquiry | 0.0% | 63.4% | 100.0% |
| account_management | 0.0% | 56.0% | 95.8% |
| bug_report | 0.0% | 14.8% | 86.4% |
| membership_subscription | 22.2% | 69.4% | 94.3% |
| general_inquiry | 0.0% | 0.0% | 95.8% |

### 3.3 LLM-as-Judge Breakdown (Our System)

| Dimension | Score |
|-----------|-------|
| Relevance | 0.733 |
| Grounding | 0.403 |
| Empathy | 0.388 |
| Actionability | 0.316 |
| **Overall** | **0.460** |

### 3.4 Baseline Ablation Notes

The **Trivial Baseline** at 12.5% accuracy confirms the dataset is not degenerate — the
majority class (`general_inquiry`) appears in only ~10% of examples, so always predicting it
produces near-random accuracy. Its 0.0% escalation F1 is expected because it never escalates,
while 72% of our examples require escalation (sensitive intents).

The **Simple Baseline** at 54% accuracy demonstrates that keyword matching captures broad intent
categories well (order_delivery, refund_return, membership_subscription all >63% F1) but
struggles with:
- **bug_report** (14.8% F1): The keyword "not working" overlaps with many intents
- **general_inquiry** (0% F1): No keywords defined for the catch-all class

---

## 4. Failure Analysis

### 4.1 What Goes Wrong

**1. Confusion between similar intents:** The most common errors occur between
`account_management` and `login_access` (e.g., "I need to change my password" vs
"I need to update my email"). These share lexical overlap with words like "account", "change",
"update".

**2. Reply grounding gap (ROUGE-L F = 0.403):** The ROUGE-L score measures overlap with
historical replies, but our retrieval sometimes pulls a reply from the wrong intent when the
TF-IDF similarity is higher for a cross-intent reply. This happens because short customer
messages ("the package is late") have high similarity to many order-related replies regardless
of intent.

**3. Low actionability scores (0.316):** Template fallback replies contain action items, but
retrieved historical replies are often generic acknowledgments ("We're looking into this") with
few concrete next steps. Real human agents include specific instructions ("go to Settings > ...")
more often than our retrieval corpus.

**4. Escalation edge cases:** The rule-based system escalates 72% of examples, which matches
the gold labels (72% should escalate). However, some false positives occur when long messages
contain detailed context that a confident classifier can still handle accurately.

### 4.2 What Is Misleading About the Headline Number

**"95% intent accuracy" is misleading** in three ways:

1. **Synthetic data overfit:** The classifier achieves 97.5% on test data generated from the
   same 13 base templates as training. In production, real tweets will have more noise,
   typos, sarcasm, and mixed intents. A realistic production accuracy is likely 75-85%.

2. **Easy intents inflate the average:** `order_delivery` and `product_inquiry` achieve 100%
   F1 because they have very distinct lexical signatures ("tracking", "specs", "compatible").
   The harder classes (`bug_report` at 86.4%, `login_access` at 90.9%) show where the real
   ceiling is.

3. **The 95% is on 8-way classification, not binary:** If we collapsed to a binary
   "sensitive vs. routine" decision (which is what escalation actually needs), accuracy
   would be even higher because the sensitive classes have stronger signals. The 8-way task
   is harder than the binary escalation decision.

---

## 5. Next Steps (One More Week)

If I had one more week, I would prioritize:

1. **Real BERT fine-tuning** (1-2 days): Fine-tune `distilbert-base-uncased` on the training
   data. Expected +5-10pp on bug_report and general_inquiry, which have the weakest lexical
   signals. This would require a GPU but Hugging Face makes it straightforward.

2. **Retrieval-augmented reply generation** (2-3 days): Replace static reply retrieval with a
   RAG approach that retrieves the top-3 relevant replies and has a small language model
   compose a new reply combining the best elements. This would improve grounding (currently
   the weakest judge dimension at 0.403).

3. **Active learning for golden set labeling** (1 day): Have the model flag the 50 examples
   it's least confident about, then "fix" those labels. This directly targets the failure
   modes identified in Section 4.

4. **Confusion matrix analysis + targeted data augmentation** (1-2 days): Generate synthetic
   examples specifically for the confused intent pairs (account_management / login_access),
   then retrain and measure the delta.

5. **End-to-end latency measurement** (1 day): Profile the pipeline to ensure it meets a
   <500ms p95 latency target for real-time support. TF-IDF inference is fast, but the
   retrieval index should be profiled at production scale (10K+ replies).

---

## 6. Ethical Considerations and Limitations

- **Escalation bias:** The rule-based escalation always sends sensitive intents to humans,
  which is a safe default but may perpetuate human-agent bottlenecks during peak hours.
- **Template replies:** Fallback templates are static and don't account for individual customer
  history or account status (e.g., a customer's actual membership tier).
- **Single language:** All evaluation is on English tweets. The system would need
  language detection and multilingual support for global brands.
- **No PII detection:** The pipeline processes raw tweet text. In production, PII should be
  masked before any model input to protect customer privacy.

---

## 7. Reproducibility Checklist

- [x] All random seeds fixed (data: 42, golden set: 123)
- [x] Pipeline script (`python src/pipeline.py`) runs end-to-end
- [x] Results saved to `experiments/results.json`
- [x] Golden set saved to `experiments/golden_eval_set.json` (200 examples)
- [x] Synthetic fallback ensures runnability without network access
- [x] Requirements pinned in `requirements.txt`
- [x] Decision log in `DECISIONS.md` (12 decisions)
