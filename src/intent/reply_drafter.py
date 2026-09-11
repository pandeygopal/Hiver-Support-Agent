"""
Reply drafter – generates contextually relevant support replies.

Strategy:
1. Classify the incoming message intent (passed in from classifier).
2. Select the top-k most relevant historical replies via TF-IDF retrieval
   over a corpus of real Hiver/Amazon support responses.
3. If retrieval is weak, fall back to a template bank for that intent.
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
    """Generates support reply drafts via TF-IDF retrieval over historical replies."""

    def __init__(self):
        self._reply_corpus: list[str] = []
        self._reply_intents: list[str] = []
        self._vectorizer = None
        self._corpus_vecs = None

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

    def draft(self, text: str, intent: str, top_k: int = 2) -> str:
        """Return a reply draft via retrieval + template fallback."""
        templates = _TEMPLATES.get(intent, _TEMPLATES["general_inquiry"])
        name = _extract_name(text)
        fallback = templates

        if self._vectorizer is not None and self._corpus_vecs is not None and self._reply_corpus:
            try:
                q_vec = self._vectorizer.transform([text])
                sims = cosine_similarity(q_vec, self._corpus_vecs)[0]
                best_idx = sims.argsort()[-top_k:][::-1]
                # Prefer matching intent
                matching = [i for i in best_idx if self._reply_intents[i] == intent]
                if matching:
                    chosen = self._reply_corpus[matching[0]]
                else:
                    chosen = fallback
            except Exception:
                chosen = fallback
        else:
            chosen = fallback

        try:
            return chosen.format(name=name)
        except (KeyError, IndexError):
            return chosen
