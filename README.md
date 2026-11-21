# SentimentIQ – Advanced Sentiment Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-green)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-87.1%25-brightgreen)](/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🎯 Overview

**SentimentIQ** is an enterprise-grade NLP sentiment analysis platform that classifies text as **Positive**, **Negative**, or **Neutral** with high accuracy.

**Problem Statement:** Analyze the emotional tone in natural language text to determine sentiment — positive, negative, or neutral — from user-generated content such as reviews, comments, or social media posts.

### ✨ Key Features
- 🚀 **87.1% Test Accuracy** on Yelp Reviews 3-class dataset
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
| **Test Accuracy** | **87.07%** |
| **Macro F1** | **0.8704** |
| **Negative F1** | 0.9000 |
| **Neutral F1** | 0.8233 |
| **Positive F1** | 0.8879 |

### Dataset: Yelp Reviews Full (3-class)
| Split | Samples | Balance |
|-------|---------|---------|
| Training | 45,000 | Perfectly balanced (1/3 each) |
| Testing | 7,500 | Perfectly balanced (1/3 each) |

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
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'
```

Response:
```json
{
  "sentiment": "Positive",
  "confidence": 0.9823,
  "confidence_level": "High",
  "probabilities": {"Negative": 0.005, "Neutral": 0.012, "Positive": 0.983},
  "explanation": "Key words: amazing, absolutely"
}
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
