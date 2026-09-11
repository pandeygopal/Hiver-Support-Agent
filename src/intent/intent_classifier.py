"""
Intent classifier – trains on labelled tweets, predicts intent with confidence.

Model: TF-IDF (1,2-grams) → Calibrated Logistic Regression
  - CalibratedClassifierCV gives well-behaved probabilities for the confidence
    threshold used in the auto-handle / escalate decision.
"""
from __future__ import annotations

import pickle
from typing import List, Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score

from dataset.dataset import Tweet, INTENTS, get_train_test
from features.features import build_feature_pipeline


class IntentClassifier:
    def __init__(self):
        self.pipeline: Pipeline = build_feature_pipeline()
        self.is_trained = False

    def train(self, train_tweets: List[Tweet]) -> None:
        texts = [t.text for t in train_tweets]
        labels = [t.intent for t in train_tweets]
        self.pipeline.fit(texts, labels)
        self.is_trained = True

    def predict(self, text: str) -> Tuple[str, float]:
        """Return (intent, confidence)."""
        if not self.is_trained:
            raise RuntimeError("Classifier not trained")
        probs = self.pipeline.predict_proba([text])[0]
        best_idx = probs.argmax()
        return self.pipeline.classes_[best_idx], float(probs[best_idx])

    def evaluate(self, test_tweets: List[Tweet]) -> dict:
        texts = [t.text for t in test_tweets]
        labels = [t.intent for t in test_tweets]
        preds = self.pipeline.predict(texts)
        return {
            "accuracy": round(accuracy_score(labels, preds), 4),
            "macro_f1": round(f1_score(labels, preds, average="macro"), 4),
            "weighted_f1": round(f1_score(labels, preds, average="weighted"), 4),
        }

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    @classmethod
    def load(cls, path: str) -> "IntentClassifier":
        obj = cls()
        with open(path, "rb") as f:
            obj.pipeline = pickle.load(f)
        obj.is_trained = True
        return obj
