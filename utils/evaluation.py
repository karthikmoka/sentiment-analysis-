"""
SentimentIQ – Evaluation utilities v2.0
Loads evaluation results JSON and formats metrics for templates.
"""

from __future__ import annotations

import os
import json
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from config.config import EVAL_PATH


def load_evaluation() -> dict:
    """Load evaluation results from JSON file. Returns empty dict if missing."""
    if os.path.exists(EVAL_PATH):
        try:
            with open(EVAL_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def format_metrics(raw: dict) -> dict:
    """
    Normalise the raw evaluation JSON into a consistent dict
    for template rendering. Handles both old and new formats.
    """
    if not raw:
        return {}

    # New format (v2.0) already has everything we need
    per_class = raw.get("per_class", {})

    # Provide defaults for all three classes
    for cls in ["Negative", "Neutral", "Positive"]:
        if cls not in per_class:
            per_class[cls] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "accuracy":          raw.get("accuracy", 0.0),
        "macro_f1":          raw.get("macro_f1", 0.0),
        "weighted_f1":       raw.get("weighted_f1", raw.get("macro_f1", 0.0)),
        "cv_mean_f1":        raw.get("cv_mean_f1"),
        "cv_std_f1":         raw.get("cv_std_f1"),
        "per_class":         per_class,
        "confusion_matrix":  raw.get("confusion_matrix"),
        "n_train":           raw.get("n_train"),
        "n_test":            raw.get("n_test"),
        "train_time_sec":    raw.get("train_time_sec") or raw.get("training_time_sec"),
        "inference_ms_per_sample": raw.get("inference_ms_per_sample"),
        "model_name":        raw.get("model_name", "SentimentIQ"),
        "timestamp":         raw.get("timestamp", ""),
    }
