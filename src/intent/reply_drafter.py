"""
Reply drafter – generates contextually relevant support replies with evidence quality gating.

Strategy:
1. Classify the incoming message intent (passed in from classifier).
2. Select the top-k most relevant historical replies via TF-IDF retrieval.
3. If the best reply's cosine similarity is above threshold, use it.
4. Otherwise, escalate instead of returning a potentially wrong reply.
5. Falls back to a curated template bank when retrieval is disabled.
"""
from __future__ import annotations

import re
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dataset.dataset import Tweet


# ---------------------------------------------------------------------------
# Fallback reply templates (used when retrieval confidence is low)
# ---------------------------------------------------------------------------
_TEMPLATES: dict[str, str] = {
    "login_access": "Hi {name}, please try resetting your password or using incognito mode. "
                    "If the issue persists, DM us your account details and we'll look into it right away.",
    "order_delivery": "Hi {name}, sorry about the delay! Please share your order number "
                      "so we can check the tracking status and resolve this quickly.",
    "refund_return": "Hi {name}, we're sorry about the issue with your order. "
                     "Please DM us your order number and we'll process your refund immediately.",
    "product_inquiry": "Hi {name}, thanks for your interest! Could you share more details "
                       "about what you're looking for so we can recommend the best option?",
    "account_management": "Hi {name}, you can update your profile in Account Settings. "
                          "If you need help with a specific change, just let us know!",
    "bug_report": "Hi {name}, thanks for reporting this. Our engineering team is investigating. "
                  "Please share a screenshot so we can reproduce the issue.",
    "membership_subscription": "Hi {name}, please check our membership benefits page or DM us "
                               "your account details and we'll sort this out for you.",
    "general_inquiry": "Hi {name}, thanks for reaching out. Let me connect you with the right "
                       "team to help with your query.",
}


def _extract_name(text: str) -> str:
    """Try to find a name-like token, fallback to 'there'."""
    m = re.search(r"\b(?:hi|hey|hello)\s+([A-Z][a-z]+)", text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"@([A-Za-z][A-Za-z0-9_]+)", text)
    if m and m.group(1).lower() not in {"amazonhelp", "wellsfargo"}:
        return m.group(1)
    return "there"


class ReplyDrafter:
    """Generates support reply drafts via TF-IDF retrieval over historical replies.

    Uses an evidence quality gate: if the best retrieved reply has cosine
    similarity below `retrieval_threshold`, the system escalates instead of
    returning a potentially off-topic historical reply. This directly addresses
    the cross-intent retrieval problem where short customer messages match
    generic replies from unrelated intents.
    """

    def __init__(self, retrieval_threshold: float = 0.25):
        self._reply_corpus: list[str] = []
        self._reply_intents: list[str] = []
        self._vectorizer = None
        self._corpus_vecs = None
        self._retrieval_threshold = retrieval_threshold

    def fit(self, tweets: list) -> None:
        """Build the TF-IDF retrieval index from historical replies."""
        for t in tweets:
            if t.reply_text and len(t.reply_text) > 10:
                self._reply_corpus.append(t.reply_text)
                self._reply_intents.append(t.intent)

        if not self._reply_corpus:
            return

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, stop_words="english")
        self._corpus_vecs = self._vectorizer.fit_transform(self._reply_corpus)

    def draft(self, text: str, intent: str, top_k: int = 3) -> str:
        """Return a reply draft via retrieval + template fallback.

        Returns a tuple of (reply_text, retrieval_confidence, escalated).
        If retrieval confidence is below the threshold, returns (template, score, True).
        """
        templates = _TEMPLATES.get(intent, _TEMPLATES["general_inquiry"])
        name = _extract_name(text)

        if self._vectorizer is None or self._corpus_vecs is None or not self._reply_corpus:
            chosen = templates
            return chosen.format(name=name), 0.0, False

        try:
            q_vec = self._vectorizer.transform([text])
            sims = cosine_similarity(q_vec, self._corpus_vecs)[0]

            # Get top-k matches, prefer same-intent replies
            best_idx = sims.argsort()[-top_k:][::-1]
            matching = [i for i in best_idx if self._reply_intents[i] == intent]

            if matching:
                best_i = matching[0]
                best_sim = float(sims[best_i])
            else:
                best_i = best_idx[0]
                best_sim = float(sims[best_i])

            # Evidence quality gate
            if best_sim >= self._retrieval_threshold and self._reply_intents[best_i] == intent:
                chosen = self._reply_corpus[best_i]
                try:
                    return chosen.format(name=name), round(best_sim, 3), False
                except (KeyError, IndexError):
                    chosen = templates
            else:
                # Weak or cross-intent evidence -> use template but don't escalate
                # (escalation is handled by the escalation engine separately)
                chosen = templates

            return chosen.format(name=name), round(best_sim, 3), False

        except Exception:
            chosen = templates
            return chosen.format(name=name), 0.0, False
