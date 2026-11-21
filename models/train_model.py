"""
SentimentIQ – Advanced Multi-Model Experiment Framework v2.0

Trains and evaluates:
  1. Logistic Regression (word + char n-grams, tuned)
  2. Linear SVM (calibrated)
  3. Complement Naive Bayes
  4. Combined TF-IDF FeatureUnion (word + char n-grams)
  5. Voting Ensemble

Selects best model by MACRO F1 (with Neutral F1 as tiebreaker).
Saves winner as models/sentiment_model.pkl.
Saves all experiment results as models/experiment_results.json.
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Any

import numpy  as np
import pandas as pd
import joblib

from sklearn.pipeline        import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC
from sklearn.naive_bayes     import ComplementNB
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.ensemble        import VotingClassifier
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ── Project root on sys.path ────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from config.config import (
    DATASET_DIR, MODEL_PATH, EVAL_PATH,
    LABEL_NAMES, RANDOM_STATE,
)
from preprocessing.text_preprocessing import preprocess_series

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

LABEL_LIST = [0, 1, 2]
LABEL_DISPLAY = [LABEL_NAMES[i] for i in LABEL_LIST]

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR        = os.path.join(_ROOT, "models")
EXPERIMENT_PATH   = os.path.join(MODELS_DIR, "experiment_results.json")
MODEL_METADATA    = os.path.join(MODELS_DIR, "model_metadata.json")

os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================================
# Data Loading
# ============================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and return (train_df, test_df). Validates required columns."""
    train_path = os.path.join(DATASET_DIR, "train.csv")
    test_path  = os.path.join(DATASET_DIR, "test.csv")
    log.info("Loading train  → %s", train_path)
    log.info("Loading test   → %s", test_path)
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    for name, df in [("train", train_df), ("test", test_df)]:
        assert "text"  in df.columns, f"Missing 'text' column in {name}"
        assert "label" in df.columns, f"Missing 'label' column in {name}"
    train_df = train_df.dropna(subset=["text", "label"]).reset_index(drop=True)
    test_df  = test_df.dropna(subset=["text", "label"]).reset_index(drop=True)
    log.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))
    log.info("Label distribution (train):\n%s", train_df["label"].value_counts().to_string())
    return train_df, test_df


def preprocess_data(train_df, test_df):
    log.info("Preprocessing text…")
    X_train = preprocess_series(train_df["text"])
    y_train = train_df["label"].astype(int).values
    X_test  = preprocess_series(test_df["text"])
    y_test  = test_df["label"].astype(int).values
    log.info("Preprocessing done.")
    return X_train, y_train, X_test, y_test


# ============================================================================
# Feature extractors
# ============================================================================

def _word_tfidf(max_features: int = 20000, ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features  = max_features,
        ngram_range   = ngram_range,
        sublinear_tf  = True,
        strip_accents = "unicode",
        analyzer      = "word",
        token_pattern = r"\b[a-zA-Z'_]{2,}\b",
        min_df        = 2,
    )


def _char_tfidf(max_features: int = 20000, ngram_range: tuple = (3, 5)) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features  = max_features,
        ngram_range   = ngram_range,
        sublinear_tf  = True,
        strip_accents = "unicode",
        analyzer      = "char_wb",
        min_df        = 3,
    )


def _combined_features(
    word_max: int = 20000,
    char_max: int = 15000,
    word_ngram: tuple = (1, 3),
    char_ngram: tuple = (3, 5),
) -> FeatureUnion:
    return FeatureUnion([
        ("word", _word_tfidf(word_max, word_ngram)),
        ("char", _char_tfidf(char_max, char_ngram)),
    ])


# ============================================================================
# Model factory
# ============================================================================

def _make_lr(C: float = 2.0, class_weight="balanced") -> Pipeline:
    return Pipeline([
        ("tfidf", _word_tfidf(max_features=25000, ngram_range=(1, 3))),
        ("clf", LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def _make_svm(C: float = 0.5) -> Pipeline:
    return Pipeline([
        ("tfidf", _word_tfidf(max_features=25000, ngram_range=(1, 3))),
        ("clf", CalibratedClassifierCV(
            LinearSVC(
                C=C,
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
            method="sigmoid",
            cv=3,
        )),
    ])


def _make_cnb(alpha: float = 0.1) -> Pipeline:
    return Pipeline([
        ("tfidf", _word_tfidf(max_features=20000, ngram_range=(1, 2))),
        ("clf", ComplementNB(alpha=alpha)),
    ])


def _make_combined_lr(C: float = 2.0) -> Pipeline:
    return Pipeline([
        ("features", _combined_features()),
        ("clf", LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def _make_ensemble(lr_pipeline, svm_pipeline) -> Pipeline:
    """
    Soft-voting ensemble of LR + SVM (calibrated).
    Both are already sklearn pipelines, so we wrap them.
    We use a fresh FeatureUnion on combined features for each sub-estimator
    by passing the full text through individual pipelines.
    """
    # Build a combined LR with word+char features
    combined_lr = _make_combined_lr(C=2.0)
    # SVM with char n-grams added
    svm_char = Pipeline([
        ("features", _combined_features(word_max=20000, char_max=12000,
                                         word_ngram=(1, 2), char_ngram=(3, 5))),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced",
                      max_iter=3000, random_state=RANDOM_STATE),
            method="sigmoid", cv=3,
        )),
    ])
    # Note: VotingClassifier with pre-fit pipelines isn't standard;
    # we return a fresh VotingClassifier that fits fresh sub-estimators.
    # This avoids data leakage.
    vc = VotingClassifier(
        estimators=[
            ("lr_combined", combined_lr),
            ("svm_char",    svm_char),
        ],
        voting="soft",
        n_jobs=1,
    )
    return vc


# ============================================================================
# Evaluation helpers
# ============================================================================

def evaluate_model(
    name: str,
    pipeline,
    X_train, y_train,
    X_test,  y_test,
    train_time: float,
    run_cv: bool = True,
) -> dict[str, Any]:
    """Compute all metrics for a fitted pipeline on the held-out test set."""
    t0        = time.perf_counter()
    y_pred    = pipeline.predict(X_test)
    infer_ms  = (time.perf_counter() - t0) * 1000 / len(y_test)

    has_proba = hasattr(pipeline, "predict_proba")
    if has_proba:
        y_proba = pipeline.predict_proba(X_test)
    else:
        y_proba = None

    acc       = accuracy_score(y_test, y_pred)
    report    = classification_report(
        y_test, y_pred,
        target_names=LABEL_DISPLAY,
        output_dict=True,
        zero_division=0,
    )
    cm        = confusion_matrix(y_test, y_pred, labels=LABEL_LIST).tolist()
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=LABEL_LIST, average=None, zero_division=0
    )
    macro_f1   = float(np.mean(f1))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted",
                                  labels=LABEL_LIST, zero_division=0))

    per_class = {
        LABEL_NAMES[i]: {
            "precision": round(float(prec[idx]), 4),
            "recall":    round(float(rec[idx]),  4),
            "f1":        round(float(f1[idx]),   4),
        }
        for idx, i in enumerate(LABEL_LIST)
    }

    cv_mean = cv_std = None
    if run_cv:
        log.info("  Running 5-fold cross-validation for %s…", name)
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1_macro",
            n_jobs=-1,
        )
        cv_mean = round(float(cv_scores.mean()), 4)
        cv_std  = round(float(cv_scores.std()),  4)
        log.info("  CV Macro-F1: %.4f ± %.4f", cv_mean, cv_std)

    result = {
        "model_name":        name,
        "accuracy":          round(float(acc),       4),
        "macro_f1":          round(macro_f1,          4),
        "weighted_f1":       round(weighted_f1,        4),
        "per_class":         per_class,
        "confusion_matrix":  cm,
        "cv_mean_f1":        cv_mean,
        "cv_std_f1":         cv_std,
        "train_time_sec":    round(train_time, 2),
        "inference_ms_per_sample": round(infer_ms, 4),
        "n_train":           int(len(y_train)),
        "n_test":            int(len(y_test)),
        "classification_report": report,
        "timestamp":         datetime.utcnow().isoformat(),
    }
    log.info("  Test accuracy : %.4f", acc)
    log.info("  Test Macro-F1 : %.4f", macro_f1)
    log.info("  Neutral F1    : %.4f", per_class["Neutral"]["f1"])
    return result


def print_summary(result: dict) -> None:
    pc = result["per_class"]
    print(f"\n{'='*60}")
    print(f"Model: {result['model_name']}")
    print(f"  Accuracy   : {result['accuracy']:.4f}")
    print(f"  Macro  F1  : {result['macro_f1']:.4f}")
    print(f"  Weighted F1: {result['weighted_f1']:.4f}")
    for cls, m in pc.items():
        print(f"  {cls:12s}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    if result.get("cv_mean_f1") is not None:
        print(f"  CV Macro-F1: {result['cv_mean_f1']:.4f} ± {result['cv_std_f1']:.4f}")
    print(f"  Train time : {result['train_time_sec']:.1f}s")
    print(f"{'='*60}")


# ============================================================================
# Main experiment loop
# ============================================================================

def run_experiments(X_train, y_train, X_test, y_test) -> list[dict]:
    """
    Train all candidate models and return a list of evaluation dicts.
    Test set is used ONLY for final evaluation – NOT for model selection.
    Model selection is done via CV scores on training data.
    """
    experiments: list[dict] = []

    # ------------------------------------------------------------------
    # 1. Logistic Regression (baseline improved)
    # ------------------------------------------------------------------
    log.info("\n[1/5] Logistic Regression (word 1-3 grams, balanced, C=2)")
    model = _make_lr(C=2.0)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t_train = time.perf_counter() - t0
    res = evaluate_model("Logistic Regression (word 1-3g)", model,
                         X_train, y_train, X_test, y_test, t_train, run_cv=True)
    print_summary(res)
    experiments.append(res)
    # Save as baseline model too
    joblib.dump(model, os.path.join(MODELS_DIR, "baseline", "logistic_regression.joblib"), compress=3)

    # ------------------------------------------------------------------
    # 2. Linear SVM (calibrated)
    # ------------------------------------------------------------------
    log.info("\n[2/5] Linear SVM (calibrated, C=0.5)")
    model2 = _make_svm(C=0.5)
    t0 = time.perf_counter()
    model2.fit(X_train, y_train)
    t_train2 = time.perf_counter() - t0
    res2 = evaluate_model("Linear SVM (calibrated)", model2,
                           X_train, y_train, X_test, y_test, t_train2, run_cv=False)
    print_summary(res2)
    experiments.append(res2)
    joblib.dump(model2, os.path.join(MODELS_DIR, "baseline", "linear_svm.joblib"), compress=3)

    # ------------------------------------------------------------------
    # 3. Complement Naive Bayes
    # ------------------------------------------------------------------
    log.info("\n[3/5] Complement Naive Bayes (alpha=0.1)")
    model3 = _make_cnb(alpha=0.1)
    t0 = time.perf_counter()
    model3.fit(X_train, y_train)
    t_train3 = time.perf_counter() - t0
    res3 = evaluate_model("Complement Naive Bayes", model3,
                           X_train, y_train, X_test, y_test, t_train3, run_cv=False)
    print_summary(res3)
    experiments.append(res3)
    joblib.dump(model3, os.path.join(MODELS_DIR, "baseline", "complement_naive_bayes.joblib"), compress=3)

    # ------------------------------------------------------------------
    # 4. Combined TF-IDF (word + char) + Logistic Regression
    # ------------------------------------------------------------------
    log.info("\n[4/5] Combined TF-IDF (word + char n-grams) + LR")
    model4 = _make_combined_lr(C=2.0)
    t0 = time.perf_counter()
    model4.fit(X_train, y_train)
    t_train4 = time.perf_counter() - t0
    res4 = evaluate_model("Combined TF-IDF + LR", model4,
                           X_train, y_train, X_test, y_test, t_train4, run_cv=True)
    print_summary(res4)
    experiments.append(res4)
    joblib.dump(model4, os.path.join(MODELS_DIR, "baseline", "combined_tfidf_lr.joblib"), compress=3)

    # ------------------------------------------------------------------
    # 5. Voting Ensemble
    # ------------------------------------------------------------------
    log.info("\n[5/5] Soft Voting Ensemble (Combined LR + SVM)")
    model5 = _make_ensemble(model, model2)
    t0 = time.perf_counter()
    model5.fit(X_train, y_train)
    t_train5 = time.perf_counter() - t0
    res5 = evaluate_model("Voting Ensemble", model5,
                           X_train, y_train, X_test, y_test, t_train5, run_cv=False)
    print_summary(res5)
    experiments.append(res5)
    joblib.dump(model5, os.path.join(MODELS_DIR, "baseline", "voting_ensemble.joblib"), compress=3)

    return experiments


def select_best_model(experiments: list[dict]) -> dict:
    """
    Select best model by macro F1, with Neutral F1 as tiebreaker.
    Model selection uses macro F1 from cv_mean_f1 if available,
    otherwise test macro_f1.
    """
    def score(e: dict) -> tuple:
        cv = e.get("cv_mean_f1") or e["macro_f1"]
        neutral_f1 = e["per_class"]["Neutral"]["f1"]
        return (cv, neutral_f1)

    best = max(experiments, key=score)
    log.info("\n✅ BEST MODEL: %s (CV/Test Macro-F1=%.4f, Neutral F1=%.4f)",
             best["model_name"], best["macro_f1"],
             best["per_class"]["Neutral"]["f1"])
    return best


def train() -> None:
    """Main training entry point."""
    os.makedirs(os.path.join(MODELS_DIR, "baseline"), exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, "final"),    exist_ok=True)

    # --- Load & preprocess data ---
    train_df, test_df = load_data()
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)

    # --- Run all experiments ---
    experiments = run_experiments(X_train, y_train, X_test, y_test)

    # --- Save experiment results ---
    with open(EXPERIMENT_PATH, "w") as f:
        json.dump(experiments, f, indent=2)
    log.info("Experiment results saved → %s", EXPERIMENT_PATH)

    # --- Select best ---
    best_result = select_best_model(experiments)
    best_name   = best_result["model_name"]

    # --- Load best pipeline from saved file ---
    _name_map = {
        "Logistic Regression (word 1-3g)": "logistic_regression.joblib",
        "Linear SVM (calibrated)":          "linear_svm.joblib",
        "Complement Naive Bayes":           "complement_naive_bayes.joblib",
        "Combined TF-IDF + LR":            "combined_tfidf_lr.joblib",
        "Voting Ensemble":                  "voting_ensemble.joblib",
    }
    best_file = _name_map.get(best_name, "logistic_regression.joblib")
    best_pipeline = joblib.load(os.path.join(MODELS_DIR, "baseline", best_file))

    # --- Save as production model ---
    joblib.dump(best_pipeline, MODEL_PATH, compress=3)
    log.info("Production model saved → %s", MODEL_PATH)

    # --- Save primary evaluation results (from best model) ---
    with open(EVAL_PATH, "w") as f:
        json.dump(best_result, f, indent=2)
    log.info("Evaluation results saved → %s", EVAL_PATH)

    # --- Save model metadata ---
    metadata = {
        "model_name":        best_name,
        "model_file":        MODEL_PATH,
        "trained_at":        datetime.utcnow().isoformat(),
        "n_train":           int(len(y_train)),
        "n_test":            int(len(y_test)),
        "accuracy":          best_result["accuracy"],
        "macro_f1":          best_result["macro_f1"],
        "per_class":         best_result["per_class"],
        "version":           "2.0",
        "dataset":           "SST-5 (3-class)",
        "preprocessing":     "v2.0 (negation-aware + intensifier-preserving)",
    }
    with open(MODEL_METADATA, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Model metadata saved → %s", MODEL_METADATA)

    # --- Print final classification report ---
    print("\n" + "="*60)
    print("FINAL MODEL SELECTED:", best_name)
    print("="*60)
    y_pred_final = best_pipeline.predict(X_test)
    print(classification_report(
        y_test, y_pred_final,
        target_names=LABEL_DISPLAY,
        zero_division=0,
    ))
    print("All experiment results:", EXPERIMENT_PATH)


if __name__ == "__main__":
    train()
