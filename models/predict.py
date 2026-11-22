"""
Sentiment Analysis – Smart Prediction Module v2.1
Auto-detects model type: DistilBERT (transformers) or sklearn Pipeline.
Falls back to sklearn if no transformer model is found.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from config.config import MODEL_PATH, LABEL_NAMES, LABEL_COLORS

log = logging.getLogger(__name__)

_pipeline     = None   # sklearn pipeline (lazy)
_bert_model   = None   # DistilBERT model (lazy)
_bert_tokenizer = None
_model_type   = None   # "sklearn" | "distilbert"

DISTILBERT_PATH = os.path.join(_ROOT, "models", "distilbert_sst3")


def _load_sklearn():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    import joblib
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run: python models/train_model.py"
        )
    _pipeline = joblib.load(MODEL_PATH)
    log.info("Sklearn model loaded from %s", MODEL_PATH)
    return _pipeline


def _load_distilbert():
    global _bert_model, _bert_tokenizer
    if _bert_model is not None:
        return _bert_model, _bert_tokenizer
    try:
        import torch
        import torch.nn as nn
        from transformers import DistilBertTokenizerFast, DistilBertModel

        # Custom model class (must match train_distilbert.py exactly)
        class DistilBert3Class(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.distilbert     = base_model
                self.pre_classifier = nn.Linear(768, 768)
                self.classifier     = nn.Linear(768, 3)
                self.dropout        = nn.Dropout(0.3)
                self.relu           = nn.ReLU()

            def forward(self, input_ids, attention_mask):
                out    = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
                hidden = out.last_hidden_state[:, 0]
                hidden = self.pre_classifier(hidden)
                hidden = self.relu(hidden)
                hidden = self.dropout(hidden)
                logits = self.classifier(hidden)
                return {"logits": logits}

        checkpoint_path = os.path.join(DISTILBERT_PATH, "model.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("config", {})
        base_name = cfg.get("base_model", "distilbert-base-uncased-finetuned-sst-2-english")

        base_model = DistilBertModel.from_pretrained(base_name)
        model = DistilBert3Class(base_model)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        _bert_tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_PATH)
        _bert_model = model
        log.info("DistilBERT (custom head) loaded from %s", DISTILBERT_PATH)
        return _bert_model, _bert_tokenizer
    except Exception as e:
        log.warning("Could not load DistilBERT: %s", e)
        return None, None


def _detect_model_type() -> str:
    global _model_type
    if _model_type:
        return _model_type
    if os.path.exists(DISTILBERT_PATH) and os.path.exists(
        os.path.join(DISTILBERT_PATH, "config.json")
    ):
        m, t = _load_distilbert()
        if m is not None:
            _model_type = "distilbert"
            return "distilbert"
    _model_type = "sklearn"
    return "sklearn"


def _predict_distilbert(text: str) -> dict:
    import torch
    model, tokenizer = _load_distilbert()
    enc = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        # Support both dict output (custom model) and HuggingFace output
        logits = out["logits"] if isinstance(out, dict) else out.logits
        proba = torch.softmax(logits, dim=1).squeeze().numpy()

    # DistilBERT trained order: 0=Negative, 1=Neutral, 2=Positive
    label_int  = int(np.argmax(proba))
    label_name = LABEL_NAMES[label_int]
    confidence = float(proba[label_int])
    classes    = [0, 1, 2]
    probs      = {LABEL_NAMES[c]: round(float(proba[c]), 4) for c in classes}

    return {
        "sentiment":        label_name,
        "label":            label_int,
        "confidence":       round(confidence, 4),
        "confidence_level": _confidence_level(confidence),
        "probabilities":    probs,
        "color":            LABEL_COLORS[label_int],
        "original_text":    text,
        "cleaned_text":     text,
        "top_features":     [],
        "explanation":      (
            f"DistilBERT predicts {label_name} with "
            f"{_confidence_level(confidence).lower()} confidence "
            f"({confidence*100:.1f}%)."
        ),
        "model_type":       "DistilBERT (fine-tuned)",
    }


def _predict_sklearn(text: str) -> dict:
    from preprocessing.text_preprocessing import clean_text

    pipe    = _load_sklearn()
    cleaned = clean_text(text)
    if not cleaned.strip():
        cleaned = text.lower()

    proba     = pipe.predict_proba([cleaned])[0]
    classes   = [int(c) for c in pipe.classes_]
    label_int = int(np.argmax(proba))
    label_name = LABEL_NAMES[label_int]
    confidence = float(proba[label_int])
    probs      = {LABEL_NAMES[c]: round(float(p), 4) for c, p in zip(classes, proba)}

    top_features = _get_top_words_sklearn(pipe, cleaned, label_int)
    conf_level   = _confidence_level(confidence)

    if top_features:
        words_str = ", ".join(f'"{f["word"]}"' for f in top_features[:3])
        explanation = (
            f"The model predicts {label_name} with {conf_level.lower()} confidence "
            f"({confidence*100:.1f}%). Key words: {words_str}."
        )
    else:
        explanation = (
            f"The model predicts {label_name} with {conf_level.lower()} confidence "
            f"({confidence*100:.1f}%)."
        )

    return {
        "sentiment":        label_name,
        "label":            label_int,
        "confidence":       round(confidence, 4),
        "confidence_level": conf_level,
        "probabilities":    probs,
        "color":            LABEL_COLORS[label_int],
        "original_text":    text,
        "cleaned_text":     cleaned,
        "top_features":     top_features,
        "explanation":      explanation,
        "model_type":       "TF-IDF + Logistic Regression",
    }


def _get_top_words_sklearn(pipeline, cleaned_text: str, label_int: int, top_n: int = 8) -> list:
    try:
        clf = None
        vec = None

        if hasattr(pipeline, "named_steps"):
            clf_step = pipeline.named_steps.get("clf")
            if hasattr(clf_step, "coef_"):
                clf = clf_step
            tfidf_step = pipeline.named_steps.get("tfidf")
            features_step = pipeline.named_steps.get("features")
            if tfidf_step is not None:
                vec = tfidf_step
            elif features_step is not None:
                for name, t in features_step.transformer_list:
                    if name == "word":
                        vec = t; break
        elif hasattr(pipeline, "clf") and hasattr(pipeline, "features"):
            # FullPipeline wrapper from max_tuning
            clf_obj = pipeline.clf
            if hasattr(clf_obj, "coef_"):
                clf = clf_obj
            feat = pipeline.features
            for name, t in feat.transformer_list:
                if name == "word":
                    vec = t; break

        if clf is None or vec is None:
            return []

        X = vec.transform([cleaned_text])
        feature_names = vec.get_feature_names_out()
        coef = clf.coef_[label_int] if clf.coef_.shape[0] > 1 else clf.coef_[0]

        cx = X.tocsr()
        nz = cx.indices
        if len(nz) == 0:
            return []

        scores = [(feature_names[i], float(X[0, i] * coef[i])) for i in nz]
        scores.sort(key=lambda x: abs(x[1]), reverse=True)

        return [
            {"word": w, "contribution": round(s, 4), "sign": "positive" if s > 0 else "negative"}
            for w, s in scores[:top_n]
        ]
    except Exception as e:
        log.debug("Feature extraction failed: %s", e)
        return []


def _confidence_level(conf: float) -> str:
    if conf >= 0.75: return "High"
    elif conf >= 0.55: return "Moderate"
    return "Low"


# ── Public API ────────────────────────────────────────────────

def predict(text: str) -> dict:
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string.")
    mtype = _detect_model_type()
    if mtype == "distilbert":
        return _predict_distilbert(text)
    return _predict_sklearn(text)


def predict_batch(texts: list[str]) -> list[dict]:
    if not texts:
        return []
    mtype = _detect_model_type()
    if mtype == "distilbert":
        import torch
        model, tokenizer = _load_distilbert()
        results = []
        for t in texts:
            if not t or not t.strip():
                results.append(_predict_sklearn(t) if t else {})
                continue
            enc = tokenizer(t, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                logits = out["logits"] if isinstance(out, dict) else out.logits
                proba = torch.softmax(logits, dim=1).squeeze().numpy()
            label_int  = int(np.argmax(proba))
            label_name = LABEL_NAMES[label_int]
            confidence = float(proba[label_int])
            probs      = {LABEL_NAMES[c]: round(float(proba[c]), 4) for c in [0, 1, 2]}
            results.append({
                "sentiment": label_name, "label": label_int,
                "confidence": round(confidence, 4),
                "confidence_level": _confidence_level(confidence),
                "probabilities": probs, "color": LABEL_COLORS[label_int],
                "original_text": t, "cleaned_text": t, "top_features": [],
                "explanation": f"{label_name} ({confidence*100:.1f}%)",
                "model_type": "DistilBERT (fine-tuned)",
            })
        return results
    # Sklearn batch (faster)
    from preprocessing.text_preprocessing import clean_text
    pipe    = _load_sklearn()
    cleaned = [clean_text(t) if t else "" for t in texts]
    cleaned = [c if c.strip() else t.lower() for c, t in zip(cleaned, texts)]
    probas  = pipe.predict_proba(cleaned)
    classes = [int(c) for c in pipe.classes_]
    results = []
    for raw, clean, proba in zip(texts, cleaned, probas):
        label_int  = int(np.argmax(proba))
        label_name = LABEL_NAMES[label_int]
        probs      = {LABEL_NAMES[c]: round(float(p), 4) for c, p in zip(classes, proba)}
        confidence = float(proba[label_int])
        results.append({
            "sentiment":        label_name,
            "label":            label_int,
            "confidence":       round(confidence, 4),
            "confidence_level": _confidence_level(confidence),
            "probabilities":    probs,
            "color":            LABEL_COLORS[label_int],
            "original_text":    raw,
            "cleaned_text":     clean,
            "top_features":     [],
            "explanation":      f"{label_name} ({confidence*100:.1f}%)",
            "model_type":       "TF-IDF + LR",
        })
    return results


def reload_model():
    global _pipeline, _bert_model, _bert_tokenizer, _model_type
    _pipeline = _bert_model = _bert_tokenizer = _model_type = None
    _detect_model_type()
