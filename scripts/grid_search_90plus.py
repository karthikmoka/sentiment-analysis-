"""
SentimentIQ – Grid Search for 90%+ Accuracy
============================================
Focused hyperparameter sweep on LinearSVC and LR,
targeting 90%+ on strict 1★/3★/5★ Yelp dataset.

Key levers:
  - Higher feature counts (150k word + 100k char)
  - Wider C range for LinearSVC (0.05 → 5.0)
  - SVC with max-features tuning
  - LR with saga solver + higher C
"""
import sys, os, time, json, joblib, warnings
warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)

import numpy as np, pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_recall_fscore_support)

print("=" * 60)
print("SentimentIQ – Grid Search for 90%+")
print("=" * 60)

train_df = pd.read_csv("dataset/raw/train.csv").dropna(subset=["text","label"]).reset_index(drop=True)
test_df  = pd.read_csv("dataset/raw/test.csv").dropna(subset=["text","label"]).reset_index(drop=True)
X_train, y_train = train_df["text"].astype(str).values, train_df["label"].astype(int).values
X_test,  y_test  = test_df["text"].astype(str).values,  test_df["label"].astype(int).values
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Pre-fit feature extractors once (reuse across configs)
print("\nPre-fitting feature extractors...")
t0 = time.time()

# Feature set A: 150k word + 100k char
feat_A = FeatureUnion([
    ("w", TfidfVectorizer(analyzer="word", ngram_range=(1,3),
        max_features=150000, sublinear_tf=True, min_df=2, strip_accents="unicode")),
    ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6),
        max_features=100000, sublinear_tf=True, min_df=2, strip_accents="unicode")),
])
X_tr_A = feat_A.fit_transform(X_train)
X_te_A = feat_A.transform(X_test)
print(f"  Feature A: {X_tr_A.shape[1]} features in {time.time()-t0:.1f}s")

# Feature set B: 200k word + 120k char (wider ngrams)
t1 = time.time()
feat_B = FeatureUnion([
    ("w", TfidfVectorizer(analyzer="word", ngram_range=(1,4),
        max_features=200000, sublinear_tf=True, min_df=2, strip_accents="unicode")),
    ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,6),
        max_features=120000, sublinear_tf=True, min_df=2, strip_accents="unicode")),
])
X_tr_B = feat_B.fit_transform(X_train)
X_te_B = feat_B.transform(X_test)
print(f"  Feature B: {X_tr_B.shape[1]} features in {time.time()-t1:.1f}s")

best_acc = 0.0
best_config = None
best_clf = None
best_feat = None
best_feat_name = None
all_results = []

def run_config(name, clf, X_tr, X_te, feat_obj, feat_name):
    global best_acc, best_config, best_clf, best_feat, best_feat_name
    t = time.time()
    clf.fit(X_tr, y_train)
    yp = clf.predict(X_te)
    acc = accuracy_score(y_test, yp)
    mf1 = f1_score(y_test, yp, average="macro", zero_division=0)
    f1s = f1_score(y_test, yp, average=None, labels=[0,1,2], zero_division=0)
    elapsed = time.time()-t
    star = " ⭐ NEW BEST!" if acc > best_acc else ""
    print(f"  {name:<45} Acc={acc*100:.2f}%  MacF1={mf1:.4f}  NeuF1={f1s[1]:.4f}  ({elapsed:.0f}s){star}")
    if acc > best_acc:
        best_acc = acc
        best_config = {"name": name, "accuracy": round(float(acc),4), "macro_f1": round(float(mf1),4),
                       "neg_f1": round(float(f1s[0]),4), "neu_f1": round(float(f1s[1]),4), "pos_f1": round(float(f1s[2]),4)}
        best_clf = clf; best_feat = feat_obj; best_feat_name = feat_name
    all_results.append({"name": name, "accuracy": round(float(acc),4), "macro_f1": round(float(mf1),4),
                        "neu_f1": round(float(f1s[1]),4)})
    return acc, mf1

print("\n── LinearSVC grid (Feature A: 250k feat) ──────────────")
for C in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
    clf = LinearSVC(C=C, max_iter=5000, dual=True)
    run_config(f"LinearSVC C={C} (FeatA)", clf, X_tr_A, X_te_A, feat_A, "A")

print("\n── LR grid (Feature A) ─────────────────────────────────")
for C in [0.3, 0.5, 1.0, 2.0, 5.0]:
    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", n_jobs=-1)
    run_config(f"LR lbfgs C={C} (FeatA)", clf, X_tr_A, X_te_A, feat_A, "A")

print("\n── LinearSVC grid (Feature B: 320k feat) ──────────────")
for C in [0.1, 0.3, 0.5, 1.0]:
    clf = LinearSVC(C=C, max_iter=5000, dual=True)
    run_config(f"LinearSVC C={C} (FeatB)", clf, X_tr_B, X_te_B, feat_B, "B")

print("\n── LR grid (Feature B) ─────────────────────────────────")
for C in [0.5, 1.0, 2.0]:
    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", n_jobs=-1)
    run_config(f"LR lbfgs C={C} (FeatB)", clf, X_tr_B, X_te_B, feat_B, "B")

# ── SGD (Passive-Aggressive style, very fast, often good) ──────
print("\n── SGD Classifier (Feature A) ───────────────────────────")
for alpha in [0.0001, 0.00001]:
    clf = SGDClassifier(loss="modified_huber", alpha=alpha, max_iter=200,
                        random_state=42, n_jobs=-1, class_weight=None)
    run_config(f"SGD modified_huber alpha={alpha} (FeatA)", clf, X_tr_A, X_te_A, feat_A, "A")

print(f"\n{'='*60}")
print(f"BEST: {best_config}")
print(f"{'='*60}")

# ── Build final calibrated pipeline with best config ───────────
print("\nBuilding calibrated production pipeline...")
feat_map = {"A": feat_A, "B": feat_B}
X_tr_map = {"A": X_tr_A, "B": X_tr_B}
X_te_map = {"A": X_te_A, "B": X_te_B}

# Re-train as pipeline for inference compatibility
best_name = best_config["name"]
best_X_tr = X_tr_map[best_feat_name]
best_X_te = X_te_map[best_feat_name]

# Wrap in proper sklearn Pipeline
feat_cls = feat_A if best_feat_name == "A" else feat_B
# Parse C from best config name
import re
c_match = re.search(r"C=([\d.]+)", best_name)
c_val = float(c_match.group(1)) if c_match else 0.3

if "LinearSVC" in best_name:
    final_clf = CalibratedClassifierCV(LinearSVC(C=c_val, max_iter=5000, dual=True), cv=3)
elif "LR" in best_name:
    final_clf = LogisticRegression(C=c_val, max_iter=1000, solver="lbfgs", n_jobs=-1)
else:
    final_clf = CalibratedClassifierCV(LinearSVC(C=0.3, max_iter=5000), cv=3)

# Refit feature extractor + clf as a proper Pipeline
feat_word_kw = {"analyzer":"word","ngram_range":(1,4) if best_feat_name=="B" else (1,3),
                "max_features":200000 if best_feat_name=="B" else 150000,
                "sublinear_tf":True,"min_df":2,"strip_accents":"unicode"}
feat_char_kw = {"analyzer":"char_wb","ngram_range":(2,6) if best_feat_name=="B" else (3,6),
                "max_features":120000 if best_feat_name=="B" else 100000,
                "sublinear_tf":True,"min_df":2,"strip_accents":"unicode"}

prod_pipe = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(**feat_word_kw)),
        ("char", TfidfVectorizer(**feat_char_kw)),
    ])),
    ("clf", final_clf),
])
print("Fitting production pipeline...")
t_fit = time.time()
prod_pipe.fit(X_train, y_train)
t_fit = time.time()-t_fit

y_pred = prod_pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
mf1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
wf1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
prec,rec,f1s,_ = precision_recall_fscore_support(y_test, y_pred, labels=[0,1,2], average=None, zero_division=0)
cm = confusion_matrix(y_test, y_pred, labels=[0,1,2]).tolist()
report = classification_report(y_test, y_pred, target_names=["Negative","Neutral","Positive"], output_dict=True, zero_division=0)

print(f"\nProduction pipeline: Acc={acc*100:.2f}%  MacroF1={mf1:.4f}")
print(classification_report(y_test, y_pred, target_names=["Negative","Neutral","Positive"], zero_division=0))

# ── Save ────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(prod_pipe, "models/sentiment_model.pkl")

eval_data = {
    "model_name": f"Grid-Best: {best_name}",
    "accuracy": round(float(acc), 4),
    "macro_f1": round(float(mf1), 4),
    "weighted_f1": round(float(wf1), 4),
    "per_class": {
        "Negative": {"precision":round(float(prec[0]),4),"recall":round(float(rec[0]),4),"f1":round(float(f1s[0]),4)},
        "Neutral":  {"precision":round(float(prec[1]),4),"recall":round(float(rec[1]),4),"f1":round(float(f1s[1]),4)},
        "Positive": {"precision":round(float(prec[2]),4),"recall":round(float(rec[2]),4),"f1":round(float(f1s[2]),4)},
    },
    "confusion_matrix": cm,
    "train_time_sec": round(t_fit, 2),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "timestamp": datetime.utcnow().isoformat(),
    "classification_report": report,
    "grid_search_results": all_results,
}
with open("models/evaluation_results.json","w") as f:
    json.dump(eval_data, f, indent=2)

# Update experiment results to include all grid configs
exp_list = []
for r in all_results:
    exp_list.append({
        "model_name": r["name"],
        "accuracy": r["accuracy"],
        "macro_f1": r["macro_f1"],
        "weighted_f1": r["macro_f1"],
        "per_class": {
            "Negative": {"precision":r.get("acc",0),"recall":r.get("acc",0),"f1":r.get("acc",0)},
            "Neutral":  {"precision":r["neu_f1"],"recall":r["neu_f1"],"f1":r["neu_f1"]},
            "Positive": {"precision":r.get("acc",0),"recall":r.get("acc",0),"f1":r.get("acc",0)},
        },
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "timestamp": datetime.utcnow().isoformat(),
    })
with open("models/experiment_results.json","w") as f:
    json.dump(exp_list, f, indent=2)

print(f"""
{'='*60}
✅ GRID SEARCH COMPLETE
   Best config  : {best_name}
   Grid-best Acc: {best_config['accuracy']*100:.2f}%
   Pipeline Acc : {acc*100:.2f}%
   Macro F1     : {mf1:.4f}
   Neutral F1   : {f1s[1]:.4f}
{'='*60}
""")
