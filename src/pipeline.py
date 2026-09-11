"""
Main pipeline – orchestrates the full Hiver support agent.
Run this to reproduce headline results in under 15 minutes.

Usage:
    python src/pipeline.py
    python src/pipeline.py --n_samples 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import load_or_generate, get_train_test
from intent.intent_classifier import IntentClassifier
from intent.reply_drafter import ReplyDrafter
from escalation.escalation import EscalationEngine
from baselines.baselines import TrivialBaseline, SimpleBaseline, OurSystem
from evaluation.evaluator import Evaluator
from evaluation.golden_set import build_golden_set


def run_pipeline(n_per_intent: int = 60, n_golden: int = 29):
    t0 = time.time()

    print("=" * 60)
    print("HIVER SUPPORT AGENT – EVALUATION PIPELINE")
    print("=" * 60)

    # 1. Load / generate dataset
    print("\n[1/7] Loading dataset...")
    tweets = load_or_generate(n_per_intent=n_per_intent)
    print(f"       Loaded {len(tweets)} tweets across {len(set(t.intent for t in tweets))} intents")

    # 2. Train/test split
    print("[2/7] Splitting train/test...")
    train, test = get_train_test(tweets, test_frac=0.2)
    print(f"       Train: {len(train)}, Test: {len(test)}")

    # 3. Train intent classifier
    print("[3/7] Training intent classifier...")
    classifier = IntentClassifier()
    classifier.train(train)
    eval_self = classifier.evaluate(test)
    print(f"       Accuracy: {eval_self['accuracy']:.1%}, Macro-F1: {eval_self['macro_f1']:.1%}")

    # 4. Train reply drafter
    print("[4/7] Training reply drafter (TF-IDF retrieval index)...")
    drafter = ReplyDrafter()
    drafter.fit(tweets)
    print("       Done")

    # 5. Train escalation engine
    print("[5/7] Initializing escalation engine...")
    esc = EscalationEngine()
    esc.fit(tweets)
    print("       Done")

    # 6. Build golden eval set
    print(f"[6/7] Building golden eval set ({n_golden} per intent)...")
    golden = build_golden_set(n_per_intent=n_golden)
    print(f"       {len(golden)} gold examples")

    # 7. Evaluate all systems
    print("[7/7] Running evaluation...")
    evaluator = Evaluator()

    systems = {
        "Trivial Baseline": TrivialBaseline(),
        "Simple (Keyword) Baseline": SimpleBaseline(),
        "Our System": OurSystem(),
    }
    # Train baselines / our system
    systems["Trivial Baseline"].fit(tweets)
    systems["Simple (Keyword) Baseline"].fit(tweets)
    systems["Our System"].fit(tweets)

    results = {}
    for name, system in systems.items():
        predictions = []
        for g in golden:
            p = system.predict(g["text"])
            predictions.append(p)

        result = evaluator.evaluate(predictions, golden)
        results[name] = {
            "metrics": {
                "intent_accuracy": result.intent_accuracy,
                "intent_macro_f1": result.intent_macro_f1,
                "reply_rouge_l_f": result.reply_rouge_l_f,
                "escalation_f1": result.escalation_f1,
                "composite_score": result.composite_score,
                "num_samples": result.num_samples,
                "num_auto_correct": result.num_auto_correct,
                "num_escalate_correct": result.num_escalate_correct,
                "per_class_f1": result.intent_per_class_f1,
            },
            "predictions": predictions,
        }

        print(f"\n  --- {name} ---")
        m = results[name]["metrics"]
        print(f"  Intent Accuracy:  {m['intent_accuracy']:.1%}")
        print(f"  Intent Macro-F1:  {m['intent_macro_f1']:.1%}")
        print(f"  Reply ROUGE-L F:  {m['reply_rouge_l_f']:.3f}")
        print(f"  Escalation F1:    {m['escalation_f1']:.1%}")
        print(f"  Composite Score:  {m['composite_score']:.3f}")

    # Judge analysis for our system
    print("\n[Extra] LLM-as-judge analysis for Our System...")
    our_preds = results["Our System"]["predictions"]
    judge_results = evaluator.judge_analysis(our_preds, golden)
    avg_scores = {"relevance": 0, "grounding": 0, "empathy": 0, "actionability": 0, "overall": 0}
    for s in judge_results["samples"]:
        for dim in avg_scores:
            avg_scores[dim] += s["judge_scores"][dim]
    n = len(judge_results["samples"])
    for dim in avg_scores:
        avg_scores[dim] = round(avg_scores[dim] / n, 3)
    print(f"  Judge dimensions (Our System):")
    for dim, val in avg_scores.items():
        print(f"    {dim}: {val:.3f}")

    # Save results
    out_dir = "experiments"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        # Convert for JSON serialization
        out = {}
        for name, r in results.items():
            out[name] = r["metrics"]
        out["judge_analysis"] = {"samples": judge_results["samples"], "avg_scores": avg_scores}
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_dir}/results.json")

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed:.1f}s")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=60, help="Tweets per intent")
    parser.add_argument("--n_golden", type=int, default=29, help="Gold samples per intent")
    args = parser.parse_args()
    run_pipeline(n_per_intent=args.n_samples, n_golden=args.n_golden)
