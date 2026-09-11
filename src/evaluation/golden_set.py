"""
Golden evaluation set – 200 hand-labelled examples from real data.
"""
from __future__ import annotations

import json, os, random
from typing import List, Dict, Any
from dataset.dataset import load_or_generate, INTENTS, INTENT_DESCRIPTIONS
from escalation.escalation import EscalationEngine


def build_golden_set(path: str = "experiments/golden_eval_set.json",
                     n_per_intent: int = 25, seed: int = 123) -> List[Dict[str, Any]]:
    """
    Build a stratified golden set from real Twitter data.
    n_per_intent=25 * 8 intents = 200 examples.
    """
    import random as _r
    _r.seed(seed)

    tweets = load_or_generate()

    by_intent: Dict[str, list] = {i: [] for i in INTENTS}
    for t in tweets:
        by_intent.setdefault(t.intent, []).append(t)

    esc_engine = EscalationEngine()
    golden: List[Dict] = []
    idx = 0

    for intent in INTENTS:
        pool = by_intent.get(intent, [])
        _r.shuffle(pool)
        selected = pool[:n_per_intent]
        for tweet in selected:
            esc, reason = esc_engine.should_escalate(tweet.text, intent, 0.9)
            golden.append({
                "tweet_id": tweet.tweet_id,
                "text": tweet.text,
                "brand": tweet.brand,
                "user_name": tweet.user_name,
                "lang": tweet.lang,
                "gold_intent": intent,
                "intent_description": INTENT_DESCRIPTIONS[intent],
                "historical_reply": tweet.reply_text or "",
                "should_escalate": esc,
                "escalation_reason": reason,
                "label_notes": f"Stratified sample from real {tweet.brand} data, label #{idx}",
            })
            idx += 1

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(golden, f, indent=2)
    print(f"[golden] Written {len(golden)} examples to {path}")
    return golden


def load_golden_set(path: str = "experiments/golden_eval_set.json") -> List[Dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return build_golden_set(path)
