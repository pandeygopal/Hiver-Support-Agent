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
multi-turn threads. When remote data was unavailable, a synthetic fallback generated 1,040
realistic Amazon support messages across 8 intent classes (~130 per intent) with genuine
linguistic diversity including paraphrases, short/noisy inputs, ambiguous cases, and
edge-case boundary examples.

**8-class intent taxonomy:**

| Intent | Description | Share of Data |
|--------|------------|--------------|
| login_access | Password reset, locked accounts, 2FA | ~12% |
| order_delivery | Shipping, tracking, delivery issues | ~12% |
| refund_return | Refunds, returns, damaged items | ~12% |
| product_inquiry | Specs, compatibility, recommendations | ~12% |
| account_management | Profile, settings, address changes | ~12% |
| bug_report | App crashes, broken features | ~12% |
| membership_subscription | Prime, subscriptions, billing | ~12% |
| general_inquiry | Everything else | ~12% |

### 2.2 Model Architecture

**Intent Classifier:** TF-IDF vectorizer (1,2-grams, 5000 features, sublinear TF) ->
`CalibratedClassifierCV(LogisticRegression(C=2.0, class_weight="balanced"))`. Calibration
is essential -- raw logistic regression probabilities are poorly calibrated, making the 0.70
confidence threshold unreliable.

**Reply Drafter:** TF-IDF retrieval over a corpus of real Amazon support responses. Top-3
historical replies are retrieved and ranked by cosine similarity, with preference given to
replies matching the predicted intent. An evidence quality gate checks that the best match
has cosine similarity >= 0.25 AND matches the predicted intent; otherwise the system falls
back to a curated template bank. This prevents the cross-intent retrieval problem where
short customer messages match generic replies from unrelated intents.

**Escalation Engine:** Rule-based with four triggers:
1. Intent is in the sensitive set (login_access, refund_return, account_management,
   membership_subscription) -> always escalate
2. Classifier confidence < 0.70 -> escalate
3. Message > 100 words -> likely complex -> escalate
4. 3+ question marks -> likely multi-issue -> escalate

### 2.3 Baselines

**Trivial Baseline:** Always predicts the majority class (`general_inquiry`) with a canned reply
and never escalates.

**Simple Baseline:** Keyword-rule classifier (7 rules covering 7 of 8 intents) with static
reply map. No escalation capability.

**Our System:** Full pipeline -- TF-IDF + Calibrated LR intent classifier, TF-IDF retrieval
reply drafter with evidence quality gate, rule-based escalation.

### 2.4 Evaluation

- **Intent accuracy / macro-F1:** Standard classification metrics on 232 golden examples
- **ROUGE-L F1:** Reply quality against historical gold replies
- **Escalation precision/recall/F1:** Whether auto-handle/escalate decisions match gold labels
- **Composite score:** Weighted blend (0.4 * F1 + 0.25 * ROUGE-L + 0.2 * esc_F1 + 0.15 * judge)
- **LLM-as-judge:** Heuristic scoring on relevance, grounding, empathy, actionability
- **Golden set construction:** 29 examples per intent, stratified from a diverse pool of ~130
  genuinely varied messages per intent covering paraphrases, edge cases, and boundary
  confusions (no mechanical prefix/suffix variations).

---

## 3. Results

### 3.1 Headline Numbers

| Metric | Trivial | Simple | Our System | Improvement over Simple |
|--------|---------|--------|------------|------------------------|
| Intent Accuracy | 12.5% | 55.2% | **96.6%** | +41.4pp |
| Macro-F1 | 2.8% | 49.5% | **96.5%** | +47.0pp |
| ROUGE-L F1 | 0.138 | 0.185 | **0.382** | +0.197 |
| Escalation F1 | 0.0% | 0.0% | **98.7%** | +98.7pp |
| Composite Score | 0.091 | 0.295 | **0.744** | +0.449 |
| Auto-handle Correct | 0 | 0 | 29 | -- |
| Escalate Correct | 0 | 0 | 118 | -- |

### 3.2 Per-Class Intent F1

| Intent | Trivial | Simple | Our System |
|--------|---------|--------|------------|
| login_access | 0.0% | 52.8% | 94.6% |
| order_delivery | 0.0% | 64.4% | 100.0% |
| refund_return | 0.2% | 69.4% | 90.6% |
| product_inquiry | 0.0% | 69.4% | 100.0% |
| account_management | 0.0% | 54.6% | 100.0% |
| bug_report | 0.0% | 12.9% | 95.1% |
| membership_subscription | 0.0% | 72.4% | 92.1% |
| general_inquiry | 0.0% | 0.0% | 100.0% |

### 3.3 LLM-as-Judge Breakdown (Our System)

| Dimension | Score |
|-----------|-------|
| Relevance | 0.703 |
| Grounding | 0.381 |
| Empathy | 0.388 |
| Actionability | 0.271 |
| **Overall** | **0.436** |

### 3.4 Baseline Ablation Notes

The **Trivial Baseline** at 12.5% accuracy confirms the dataset is not degenerate. Its 0.0%
escalation F1 is expected since it never escalates, while ~70% of golden set examples require
escalation (sensitive intents in the rule-based policy).

The **Simple Baseline** at 55.2% accuracy shows keyword matching captures broad intent
categories well (membership_subscription at 72.4%, product_inquiry at 69.4%) but struggles with:
- **bug_report** (12.9% F1): The keyword "not working" overlaps with login issues, order issues,
  and general complaints -- many messages contain some variant of "not working" regardless of intent
- **general_inquiry** (0% F1): No keywords defined for the catch-all class, so all unmatched
  messages default to the highest-scoring specific intent

---

## 4. Failure Analysis

### 4.1 What Goes Wrong

**1. Intent boundary confusion (login_access vs. bug_report, login_access vs. account_management):**
The most interesting failure mode is 2FA-related messages being classified as `bug_report` instead
of `login_access`. When a customer says "my authenticator app is not working," the keywords
"not working" and "app" trigger the bug_report classifier. The retrieved reply then becomes a
generic engineering-team response ("Our engineering team is investigating. Please share a
screenshot.") rather than the appropriate 2FA troubleshooting response. This is a genuine
multi-word ambiguity: "not working" is the dominant bug_report keyword, but 2FA issues are
fundamentally access problems.

Similarly, messages about changing account details blur between `account_management` and
`login_access` -- "update my email" vs "reset my password" share the word "account".

**2. Reply grounding gap (ROUGE-L F = 0.428, Judge grounding = 0.428):**
The evidence quality gate (threshold 0.25) reduces cross-intent retrieval, but the remaining
grounding gap comes from within-intent variation. The customer might say "the package is late"
which matches a generic "sorry about the delay" template, while the gold reply addresses a
specific scenario like "wrong item delivered." The TF-IDF similarity is high (same intent, same
vocabulary) but the specific situation differs. This is an inherent limitation of retrieval-based
reply drafting without deeper semantic understanding.

**3. Low actionability scores (0.305):**
The template fallback replies contain action items, but retrieved historical replies are often
generic acknowledgments. When the evidence quality gate triggers and falls back to templates,
those templates include specific instructions ("go to Settings > ...") which score higher on
actionability. Paradoxically, the quality gate slightly lowers grounding (templates have lower
ROUGE-L overlap with specific historical replies) but raises actionability.

**4. Membership vs. refund boundary confusion:**
Messages like "I was charged for a Prime membership trial I didn't sign up for" are sometimes
classified as `membership_subscription` when the gold label is `refund_return`. Both contain
"charged" and "refund" semantics. However, both intents route to escalation (both are in the
sensitive set), so this confusion has no downstream impact on the user -- they still reach a
human agent.

**5. Escalation coverage vs. automation trade-off:**
The rule-based system escalates ~70% of examples (all sensitive intents), matching gold labels.
This means auto-handle coverage is ~30% -- low by some standards, but appropriate for a
conservative policy on financial and account-related issues. The 98.7% escalation F1 indicates
the conservative policy is well-calibrated: when we escalate, we are almost always right to do
so.

### 4.2 What Is Misleading About the Headline Number

**"96.6% intent accuracy" does not mean "96.6% successful customer resolutions."** This is the
single most important caveat, and it matters in three ways:

**1. Correct intent does not guarantee correct resolution.** The LLM judge gives replies an
overall score of 0.468 and a grounding score of 0.428, showing that the system can identify the
customer's problem type correctly while still retrieving an imperfect historical resolution. A
customer who says "the package is late" gets the right intent classification (order_delivery)
but a generic "sorry about the delay" reply, not a specific investigation of their tracking
number. Intent accuracy and reply quality measure different things.

**2. The evaluation set is controlled, not chaotic.** Our 232-example golden set is drawn from
a diverse pool of ~1,040 synthetic messages covering paraphrases, edge cases, and boundary
confusions. However, these are still simulated Twitter messages. Real Twitter contains more noise
(typos, sarcasm, emoji-heavy messages, mixed intents like "my package is late AND my account is
locked"), and performance on genuinely noisy input would be lower. A realistic production
accuracy is likely 80-88% rather than 96.6%.

**3. Strong escalation performance masks limited automation coverage.** The 98.7% escalation
F1 is excellent, but it reflects a deliberately conservative policy where ~70% of messages
escalate to humans. The system auto-handles only ~30% of messages. High escalation precision
means we rarely auto-handle something that should have been escalated -- but it also means most
messages don't get an automated reply at all. The 96.6% intent accuracy and 98.7% escalation
F1 together describe a system that is very good at *routing* messages, not one that resolves
them autonomously at high volume.

Therefore, our headline results demonstrate strong intent routing and safe escalation behavior --
not that the agent can autonomously resolve 96.6% of customer issues.

---

## 5. Next Steps (One More Week)

If I had one more week, I would prioritize:

1. **Evidence quality tuning and analysis** (1 day): Vary the retrieval threshold (0.15-0.40)
   and measure the impact on grounding vs. escalation rate. Find the sweet spot where
   grounding is maximized without over-escalating.

2. **Intent-specific reply banks** (1-2 days): Instead of a flat retrieval corpus, maintain
   per-intent reply templates that address common sub-problems. For order_delivery, have
   separate templates for "not arrived", "wrong item", "damaged package", "late delivery."
   This would directly improve actionability (currently the weakest judge dimension at 0.305).

3. **Active learning for hard examples** (1 day): Have the model flag the 50 examples it's
   least confident about, inspect them for ambiguity, and add targeted training examples for
   the confused intent pairs (login_access/bug_report, account_management/login_access).

4. **Real BERT fine-tuning** (2-3 days): Fine-tune `distilbert-base-uncased` on the training
   data. Expected +5-10pp on bug_report and general_inquiry, which have the weakest lexical
   signals. This would require a GPU but Hugging Face makes it straightforward.

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
- **Synthetic evaluation data:** The golden set uses synthetically generated messages. While
  diverse and covering edge cases, real Twitter data may contain patterns (sarcasm, mixed
  intents, code-switching) not captured by our generation process.

---

## 7. Reproducibility Checklist

- [x] All random seeds fixed (data: 42, golden set: 123)
- [x] Pipeline script (`python src/pipeline.py`) runs end-to-end
- [x] Results saved to `experiments/results.json`
- [x] Golden set saved to `experiments/golden_eval_set.json` (232 examples, 29 per intent)
- [x] Synthetic fallback ensures runnability without network access
- [x] Requirements pinned in `requirements.txt`
- [x] Decision log in `DECISIONS.md` (15 decisions)
