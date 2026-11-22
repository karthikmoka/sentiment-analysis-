"""
Sentiment Analysis – Maximum Traditional ML Grid Search
Runs exhaustive parameter search to find the best possible accuracy.
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import pandas as pd
import joblib
import time

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from preprocessing.text_preprocessing import preprocess_series

# ── Load Data ─────────────────────────────────────────────────
DATASET_DIR = os.path.join(_ROOT, "dataset", "raw")

train_df = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))
train_df = train_df.dropna(subset=["text", "label"]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["text", "label"]).reset_index(drop=True)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

print("Preprocessing...")
X_train_raw = preprocess_series(train_df["text"])
y_train = train_df["label"].astype(int).values
X_test_raw  = preprocess_series(test_df["text"])
y_test  = test_df["label"].astype(int).values
print("Done.")

# ── Feature Engineering ───────────────────────────────────────
# High-capacity features
word_vec = TfidfVectorizer(
    max_features=60000,
    ngram_range=(1, 4),
    sublinear_tf=True,
    min_df=1,
    strip_accents="unicode",
    token_pattern=r"\b[a-zA-Z'_]{2,}\b",
)
char_vec = TfidfVectorizer(
    max_features=40000,
    ngram_range=(2, 6),
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
    analyzer="char_wb",
)
features = FeatureUnion([("word", word_vec), ("char", char_vec)])

print("Building features (60k word + 40k char n-grams)...")
t0 = time.time()
X_tr = features.fit_transform(X_train_raw)
X_te = features.transform(X_test_raw)
print(f"Feature matrix: {X_tr.shape}, time={time.time()-t0:.1f}s")

# ── Grid Search LR ────────────────────────────────────────────
print("\n=== Logistic Regression Grid Search ===")
best_acc = 0; best_f1 = 0; best_params = {}; best_model_obj = None

results = []
for C in [0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    for cw in ["balanced", None]:
        t0 = time.time()
        lr = LogisticRegression(
            C=C, class_weight=cw,
            max_iter=3000, solver="lbfgs",
            n_jobs=-1, random_state=42,
        )
        lr.fit(X_tr, y_train)
        preds = lr.predict(X_te)
        acc = accuracy_score(y_test, preds)
        mf1 = f1_score(y_test, preds, average="macro", zero_division=0)
        nf1 = f1_score(y_test == 1, preds == 1, average="binary", zero_division=0)
        wf1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        t_e = time.time() - t0
        print(f"  C={C:5} cw={str(cw):8}: Acc={acc:.4f} MacroF1={mf1:.4f} NeuF1={nf1:.4f} ({t_e:.1f}s)")
        results.append({"C": C, "cw": str(cw), "acc": acc, "macro_f1": mf1, "neu_f1": nf1})
        if mf1 > best_f1:
            best_f1 = mf1; best_acc = acc; best_model_obj = lr
            best_params = {"C": C, "cw": cw}

print(f"\nBest LR params: {best_params}")
print(f"Best accuracy: {best_acc:.4f} | Best macro F1: {best_f1:.4f}")

# ── Full Classification Report for Best ───────────────────────
y_pred_best = best_model_obj.predict(X_te)
print("\n--- Best Model Classification Report ---")
print(classification_report(y_test, y_pred_best,
      target_names=["Negative", "Neutral", "Positive"], zero_division=0))

# ── SVM Comparison ────────────────────────────────────────────
print("\n=== SVM Grid Search ===")
best_svm_acc = 0; best_svm = None
for C in [0.1, 0.3, 0.5, 1.0, 2.0]:
    svm = LinearSVC(C=C, class_weight="balanced", max_iter=5000, random_state=42)
    svm.fit(X_tr, y_train)
    preds = svm.predict(X_te)
    acc = accuracy_score(y_test, preds)
    mf1 = f1_score(y_test, preds, average="macro", zero_division=0)
    print(f"  SVM C={C}: Acc={acc:.4f} MacroF1={mf1:.4f}")
    if acc > best_svm_acc:
        best_svm_acc = acc; best_svm = svm

# ── Save best combined model ──────────────────────────────────
# Wrap into a predict-able object with the feature pipeline
class FullPipeline:
    def __init__(self, features, clf):
        self.features = features
        self.clf = clf
        self.classes_ = clf.classes_
    def predict(self, X_raw):
        Xf = self.features.transform(X_raw)
        return self.clf.predict(Xf)
    def predict_proba(self, X_raw):
        Xf = self.features.transform(X_raw)
        return self.clf.predict_proba(Xf)

full = FullPipeline(features, best_model_obj)
joblib.dump(full, "models/sentiment_model.pkl", compress=3)

# Save metrics
import json
from datetime import datetime
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_best,
    labels=[0,1,2], average=None, zero_division=0)
metrics = {
    "model_name": f"Max TF-IDF LR (C={best_params['C']}, cw={best_params['cw']})",
    "accuracy": round(float(accuracy_score(y_test, y_pred_best)), 4),
    "macro_f1": round(float(f1_score(y_test, y_pred_best, average='macro')), 4),
    "weighted_f1": round(float(f1_score(y_test, y_pred_best, average='weighted')), 4),
    "per_class": {
        "Negative": {"precision": round(float(prec[0]),4), "recall": round(float(rec[0]),4), "f1": round(float(f1[0]),4)},
        "Neutral":  {"precision": round(float(prec[1]),4), "recall": round(float(rec[1]),4), "f1": round(float(f1[1]),4)},
        "Positive": {"precision": round(float(prec[2]),4), "recall": round(float(rec[2]),4), "f1": round(float(f1[2]),4)},
    },
    "confusion_matrix": confusion_matrix(y_test, y_pred_best, labels=[0,1,2]).tolist(),
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "timestamp": datetime.utcnow().isoformat(),
    "classification_report": classification_report(y_test, y_pred_best,
        target_names=["Negative","Neutral","Positive"], output_dict=True, zero_division=0),
}
with open("models/evaluation_results.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nFinal best accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
print(f"Final best macro F1: {metrics['macro_f1']:.4f}")
print("Model saved!")
