"""
Baselines for the Hiver support agent.

Three baselines:
1. Trivial: always predict most-common intent, canned reply, no escalation
2. Simple: keyword-rule-based classifier, static reply map, no escalation
3. Ours: TF-IDF + Logistic Regression intent, TF-IDF retrieval reply, rule-based escalation
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Dict

from dataset.dataset import Tweet, INTENTS, INTENT_DESCRIPTIONS

# ---------------------------------------------------------------------------
# Keyword rules (matches our 8-intent taxonomy)
# ---------------------------------------------------------------------------
_KEYWORD_RULES: List[tuple] = [
    ("login_access", ["login", "password", "sign in", "locked", "2fa", "access", "locked out", "forgot"]),
    ("order_delivery", ["order", "delivery", "shipping", "track", "delivered", "shipment", "arrive", "late"]),
    ("refund_return", ["refund", "return", "money back", "charge back", "reimburse", "cancel order"]),
    ("product_inquiry", ["product", "item", "compatible", "spec", "recommend", "look for", "available"]),
    ("account_management", ["account", "profile", "settings", "email change", "phone", "address"]),
    ("bug_report", ["bug", "crash", "broken", "error", "not working", "freeze", "glitch"]),
    ("membership_subscription", ["prime", "membership", "subscription"]),
]

_SIMPLE_REPLIES: Dict[str, str] = {
    "login_access": "Hi! For login issues, try resetting your password or using incognito mode. DM us if it persists!",
    "order_delivery": "Hi! Let me check your order status. Please share your order number so I can look into it.",
    "refund_return": "Hi! I'm sorry about the issue. Please DM us your order number and we'll process your refund.",
    "product_inquiry": "Hi! Thanks for your interest. Could you tell me more about what you're looking for?",
    "account_management": "Hi! Go to Account Settings to update your profile. Let me know if you need specific help!",
    "bug_report": "Hi! Thanks for reporting this. Our engineering team is investigating. Please share a screenshot.",
    "membership_subscription": "Hi! For membership questions, please check our Prime benefits page or DM us your account details.",
    "general_inquiry": "Hi! Thanks for reaching out. Let me connect you with the right team to help.",
}


class TrivialBaseline:
    """Always predicts the majority class with a canned reply."""

    def __init__(self):
        self.majority_intent = "general_inquiry"
        self.canned_reply = "Hi! Thanks for reaching out. Our team will get back to you soon."

    def fit(self, tweets: List) -> None:
        counts = Counter(t.intent for t in tweets)
        self.majority_intent = counts.most_common(1)[0][0]

    def predict(self, text: str) -> Dict:
        return {
            "intent": self.majority_intent,
            "intent_confidence": 1.0,
            "reply_text": self.canned_reply,
            "escalate": False,
            "escalation_reason": "Baseline: always auto-handle",
        }


class SimpleBaseline:
    """Keyword-rule-based intent classifier with static reply map."""

    def fit(self, tweets: List) -> None:
        pass

    def predict(self, text: str) -> Dict:
        text_lower = text.lower()
        scores: Dict[str, int] = {i: 0 for i in INTENTS}
        for intent, keywords in _KEYWORD_RULES:
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[intent] = score

        best_intent = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_intent] / 3.0) if scores[best_intent] > 0 else 0.5

        return {
            "intent": best_intent,
            "intent_confidence": round(confidence, 3),
            "reply_text": _SIMPLE_REPLIES.get(best_intent, _SIMPLE_REPLIES["general_inquiry"]),
            "escalate": False,
            "escalation_reason": "Baseline: always auto-handle",
        }


class OurSystem:
    """Full pipeline: trained intent classifier + TF-IDF reply retrieval + escalation rules."""

    def __init__(self):
        from intent.intent_classifier import IntentClassifier
        from intent.reply_drafter import ReplyDrafter
        from escalation.escalation import EscalationEngine

        self.classifier = IntentClassifier()
        self.drafter = ReplyDrafter()
        self.escalation = EscalationEngine()

    def fit(self, tweets: List) -> None:
        from dataset.dataset import get_train_test
        train, _ = get_train_test(tweets, test_frac=0.2)
        self.classifier.train(train)
        self.drafter.fit(tweets)

    def predict(self, text: str) -> Dict:
        intent, conf = self.classifier.predict(text)
        reply, retr_conf, _ = self.drafter.draft(text, intent)
        esc, reason = self.escalation.should_escalate(text, intent, conf)
        return {
            "intent": intent,
            "intent_confidence": round(conf, 3),
            "reply_text": reply,
            "retrieval_confidence": round(retr_conf, 3),
            "escalate": esc,
            "escalation_reason": reason,
        }
