"""
Features module – text vectorisation for intent classification.
Uses TF-IDF (fast, interpretable) as primary feature extractor.
"""
from __future__ import annotations

import pickle
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


def build_feature_pipeline() -> Pipeline:
    """Return an sklearn Pipeline: TF-IDF → Calibrated Logistic Regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2,
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced"),
            cv=3,
        )),
    ])
