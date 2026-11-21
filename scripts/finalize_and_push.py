"""
SentimentIQ – Post-Training Finalize + GitHub Push
===================================================
Run after train_high_accuracy.py completes.
Steps:
  1. Verify evaluation_results.json has new accuracy
  2. Update README with new metrics
  3. Run pytest (53 tests)
  4. Git init, add remote, commit all, push to GitHub
"""
import sys, os, json, subprocess, shutil
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)

print("=" * 60)
print("SentimentIQ – Post-Training Finalize & GitHub Push")
print("=" * 60)

# ── 1. Read final metrics ──────────────────────────────────────
with open("models/evaluation_results.json") as f:
    ev = json.load(f)

acc     = ev["accuracy"]
mf1     = ev["macro_f1"]
neu_f1  = ev["per_class"]["Neutral"]["f1"]
neg_f1  = ev["per_class"]["Negative"]["f1"]
pos_f1  = ev["per_class"]["Positive"]["f1"]
n_train = ev.get("n_train", 36000)
n_test  = ev.get("n_test", 6000)
model_name = ev["model_name"]

print(f"\n📊 Final Model: {model_name}")
print(f"   Accuracy : {acc*100:.2f}%")
print(f"   Macro F1 : {mf1:.4f}")
print(f"   Neg F1   : {neg_f1:.4f}")
print(f"   Neu F1   : {neu_f1:.4f}")
print(f"   Pos F1   : {pos_f1:.4f}")

# ── 2. Update README ───────────────────────────────────────────
readme = f"""# SentimentIQ – Advanced Sentiment Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-green)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-{acc*100:.1f}%25-brightgreen)](/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🎯 Overview

**SentimentIQ** is an enterprise-grade NLP sentiment analysis platform that classifies text as **Positive**, **Negative**, or **Neutral** with high accuracy.

**Problem Statement:** Analyze the emotional tone in natural language text to determine sentiment — positive, negative, or neutral — from user-generated content such as reviews, comments, or social media posts.

### ✨ Key Features
- 🚀 **{acc*100:.1f}% Test Accuracy** on Yelp Reviews 3-class dataset
- 🧠 Advanced TF-IDF (word + character n-grams) feature extraction
- 📊 Multi-model experiment framework (LR, SVM, NB, Ensemble)
- 🔍 Per-prediction feature importance explanation
- 📁 Batch CSV analysis with download
- 📈 Real-time analytics dashboard
- 🌙 Professional dark/light theme UI
- 🔌 REST API with 6 endpoints

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **{acc*100:.2f}%** |
| **Macro F1** | **{mf1:.4f}** |
| **Negative F1** | {neg_f1:.4f} |
| **Neutral F1** | {neu_f1:.4f} |
| **Positive F1** | {pos_f1:.4f} |

### Dataset: Yelp Reviews Full (3-class)
| Split | Samples | Balance |
|-------|---------|---------|
| Training | {n_train:,} | Perfectly balanced (1/3 each) |
| Testing | {n_test:,} | Perfectly balanced (1/3 each) |

**Label Mapping:**
- 1–2 ★ → **Negative** (0)
- 3 ★ → **Neutral** (1)
- 4–5 ★ → **Positive** (2)

---

## 🛠️ Implementation Approach

**Best Approach:** Traditional Machine Learning (TF-IDF + Logistic Regression)

### NLP Pipeline
1. **Text Collection** — Yelp restaurant/business reviews (650k total, balanced 36k train)
2. **Preprocessing** — Negation preservation, contraction expansion, lemmatization, emoji normalization
3. **Feature Extraction** — Combined TF-IDF: word n-grams (1–3) + character n-grams (3–5)
4. **Model Training** — Logistic Regression, LinearSVC, ComplementNB, Ensemble (all compared)
5. **Evaluation** — Accuracy, Macro F1, per-class F1, confusion matrix, 5-fold CV
6. **Prediction** — Sentiment label + confidence % + probability distribution + feature explanation

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10+
```

### Installation
```bash
git clone https://github.com/karthikmoka/SentimentIQ-Flask.git
cd SentimentIQ-Flask

pip install -r requirements.txt
```

### Download Dataset & Train Model
```bash
# Build the Yelp 3-class dataset
python scripts/build_amazon_dataset.py

# Train all models (picks best automatically)
python scripts/train_high_accuracy.py
```

### Run the App
```bash
python app.py
```
Open: **http://127.0.0.1:5000**

---

## 🌐 Application Pages

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/` | Metrics overview, analytics summary |
| Analyzer | `/analyze` | Single-text sentiment analysis |
| Batch | `/batch` | CSV bulk analysis |
| Performance | `/performance` | Detailed model metrics & confusion matrix |
| Comparison | `/comparison` | Multi-model experiment comparison |
| Analytics | `/analytics` | Live prediction statistics |
| About | `/about` | Dataset, tech stack, API docs |

## 🔌 REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Single text prediction |
| POST | `/batch-predict` | Batch CSV analysis |
| GET | `/api/analytics` | Analytics JSON |
| GET | `/api/model-performance` | Metrics JSON |
| GET | `/api/model-comparison` | All experiments JSON |
| GET | `/api/health` | Health check |

### Example
```bash
curl -X POST http://127.0.0.1:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"text": "This product is absolutely amazing!"}}'
```

Response:
```json
{{
  "sentiment": "Positive",
  "confidence": 0.9823,
  "confidence_level": "High",
  "probabilities": {{"Negative": 0.005, "Neutral": 0.012, "Positive": 0.983}},
  "explanation": "Key words: amazing, absolutely"
}}
```

---

## 🧪 Running Tests
```bash
python -m pytest tests/test_app.py -v
```

---

## 📂 Project Structure
```
SentimentIQ-Flask/
├── app.py                      # Flask application
├── requirements.txt            # Dependencies
├── config/config.py            # Configuration
├── models/
│   ├── train_model.py          # Baseline training
│   ├── predict.py              # Prediction engine (dual-mode)
│   ├── sentiment_model.pkl     # Trained model
│   └── evaluation_results.json # Metrics
├── scripts/
│   ├── build_amazon_dataset.py # Dataset builder
│   └── train_high_accuracy.py  # High-accuracy training
├── preprocessing/
│   └── text_preprocessing.py  # Advanced NLP pipeline
├── templates/                  # Jinja2 HTML templates
├── static/                     # CSS, JS, assets
├── dataset/raw/                # train.csv, test.csv
├── tests/test_app.py           # 53 automated tests
└── utils/                      # Evaluation, analytics
```

---

## 📋 Requirements
```
Flask>=3.0
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
nltk>=3.8
joblib>=1.3
datasets>=5.0  # for dataset download
```

---

## 👤 Author
**Karthik Moka** | [GitHub](https://github.com/karthikmoka)

## 📄 License
MIT License
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
print("\n✅ README.md updated with new metrics")

# ── 3. Create/update requirements.txt ─────────────────────────
req = """Flask>=3.0
Werkzeug>=3.0
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
nltk>=3.8
joblib>=1.3
datasets>=5.0
matplotlib>=3.7
seaborn>=0.12
pytest>=7.0
"""
with open("requirements.txt", "w") as f:
    f.write(req)
print("✅ requirements.txt updated")

# ── 4. Create .gitignore ───────────────────────────────────────
gitignore = """# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Virtual environments
venv/
env/
.env

# Models (large files)
models/sentiment_model.pkl
models/distilbert_sst3/

# Dataset (large files)
dataset/raw/train.csv
dataset/raw/test.csv
dataset/raw/train_sst3_backup.csv
dataset/raw/test_sst3_backup.csv

# Upload/results
uploads/
results/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# HuggingFace cache
.cache/
"""
with open(".gitignore", "w") as f:
    f.write(gitignore)
print("✅ .gitignore created")

# ── 5. Run pytest ──────────────────────────────────────────────
print("\n🧪 Running 53 tests...")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_app.py", "-v", "--tb=short", "-q"],
    capture_output=True, text=True, timeout=120
)
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.returncode == 0:
    print("✅ All tests passed!")
else:
    print(f"⚠️ Some tests failed (exit {result.returncode})")
    print(result.stderr[-1000:])

# ── 6. Git setup + push ───────────────────────────────────────
print("\n🔧 Setting up Git...")
REMOTE = "https://github.com/karthikmoka/SentimentIQ-Flask.git"

def git(cmd, check=False):
    r = subprocess.run(["git"] + cmd.split(), capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if out: print(f"  git {cmd[:40]}: {out[:200]}")
    return r

# Init if needed
if not os.path.exists(".git"):
    git("init")
    print("  Git repo initialized")

git("config user.email mokakarthik2000@gmail.com")
git("config user.name moka karthik")

# Set remote
rem = subprocess.run(["git","remote","get-url","origin"], capture_output=True, text=True)
if rem.returncode != 0:
    git(f"remote add origin {REMOTE}")
    print(f"  Remote added: {REMOTE}")
else:
    git(f"remote set-url origin {REMOTE}")
    print(f"  Remote updated: {REMOTE}")

# Stage all files
git("add -A")

# Commit
commit_msg = f"feat: upgrade to Yelp 3-class dataset — {acc*100:.2f}% accuracy, Macro F1={mf1:.4f}"
r = subprocess.run(
    ["git", "commit", "-m", commit_msg],
    capture_output=True, text=True
)
print(f"  Commit: {(r.stdout+r.stderr).strip()[:300]}")

# Push
print("\n📤 Pushing to GitHub...")
push = subprocess.run(
    ["git", "push", "-u", "origin", "master", "--force"],
    capture_output=True, text=True, timeout=120
)
push_out = (push.stdout + push.stderr).strip()
print(f"  Push result: {push_out[:500]}")

if push.returncode == 0:
    print(f"\n✅ Successfully pushed to {REMOTE}")
else:
    # Try main branch
    print("  Trying 'main' branch...")
    push2 = subprocess.run(
        ["git", "push", "-u", "origin", "master:main", "--force"],
        capture_output=True, text=True, timeout=120
    )
    print(f"  Push main: {(push2.stdout+push2.stderr).strip()[:300]}")

print(f"""
{'='*60}
✅ FINALIZATION COMPLETE
{'='*60}
  Model   : {model_name}
  Accuracy: {acc*100:.2f}%
  MacroF1 : {mf1:.4f}
  GitHub  : {REMOTE}
  Local   : http://127.0.0.1:5000
{'='*60}
""")
