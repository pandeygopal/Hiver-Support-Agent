"""
Escalation decision layer – decides whether to auto-handle or escalate.

Decision factors:
1. Intent sensitivity (login, refund, account management always escalate)
2. Classifier confidence (below threshold -> escalate)
3. Message complexity (very long, multiple questions -> escalate)
4. Language (non-English -> escalate)

Confidence threshold: 0.70
"""

SENSITIVE_INTENTS = {"login_access", "refund_return", "account_management", "membership_subscription"}


class EscalationEngine:
    """Rule-based escalation engine."""

    CONFIDENCE_THRESHOLD = 0.70

    def should_escalate(self, text: str, intent: str, confidence: float,
                        lang: str = "en") -> tuple[bool, str]:
        """Return (should_escalate, reason)."""
        reasons = []

        if intent in SENSITIVE_INTENTS:
            reasons.append(f"Intent '{intent}' requires human review (sensitive)")

        if confidence < self.CONFIDENCE_THRESHOLD:
            reasons.append(f"Low classifier confidence ({confidence:.2f} < {self.CONFIDENCE_THRESHOLD})")

        words = text.split()
        if len(words) > 100:
            reasons.append(f"Message is very long ({len(words)} words) – likely complex issue")

        question_marks = text.count("?")
        if question_marks >= 3:
            reasons.append(f"Multiple questions ({question_marks}) – likely multi-issue")

        if reasons:
            return True, "; ".join(reasons)
        return False, "Routine message within auto-handle scope"

    def fit(self, tweets):
        pass
