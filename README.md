# Sentiment Analysis

A web-based NLP application for multi-class sentiment classification of customer reviews into Negative, Neutral, and Positive categories.

---

## Project Details

| Field | Details |
|---|---|
| **Institution** | S.R.K.R. Engineering College (A), Bhimavaram, Andhra Pradesh |
| **Program** | B.Tech – Artificial Intelligence and Machine Learning (2026–2027) |
| **Team Members** | MOKA KARTHIK (25B95A6118), RONGALA NAGENDRA PAVAN (25B95A6124), VADITHE SIVA NAIK (25B95A6127), VAKA SRI RAM (25B95A6128) |
| **Lab Instructors** | Smt. D.N.S.B Kavitha (Associate Professor), Smt. B.Mounika (Assistant Professor) |

---

## Problem Statement

Understanding customer sentiment at scale is a core challenge in NLP. This project builds a 3-class sentiment classifier (Negative / Neutral / Positive) trained on multi-domain customer reviews (Yelp, Amazon, IMDb). The system is served as a Flask web application with a real-time prediction API.

---

## Dataset

| Split | Samples | Negative | Neutral | Positive |
|---|---|---|---|---|
| Training | 45,000 | 15,000 | 15,000 | 15,000 |
| Testing | 7,500 | 2,500 | 2,500 | 2,500 |

**Labeling scheme:** 1★ → Negative · 3★ → Neutral · 5★ → Positive  
**Sources:** Multi-domain customer reviews (Yelp, Amazon, IMDb)

---

## Approach

**Preprocessing**
- Lowercasing, punctuation removal, stopword filtering (negation words preserved: `not`, `no`, `never`, `hardly`, `barely`, etc.)
- NLTK WordNet lemmatization

**Feature Extraction**
- TF-IDF with FeatureUnion
  - Word n-grams (1–3), max 150,000 features
  - Character n-grams (3–6), max 100,000 features

**Model**
- Calibrated LinearSVC (C=0.2)
- Probability calibration via CalibratedClassifierCV (Platt scaling)

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | **87.07%** |
| Macro F1 | 0.8704 |
| Negative F1 | 0.9000 |
| Neutral F1 | 0.8233 |
| Positive F1 | 0.8879 |

---

## Tech Stack

- **Backend:** Python 3.10, Flask, scikit-learn, NLTK, Pandas, NumPy, Joblib
- **Frontend:** Bootstrap 5, Chart.js, HTML5/CSS3/JavaScript
- **Deployment:** Gunicorn, localhost.run SSH tunnel
- **Testing:** pytest (53 tests)

---

## Project Structure

```
sentiment-analysis/
├── app.py                  # Flask application entry point
├── preprocessing/          # Text cleaning and feature extraction
├── models/
│   ├── train_model.py      # Model training script
│   ├── predict.py          # Prediction utilities
│   └── sentiment_model.pkl # Trained model (Calibrated LinearSVC)
├── dataset/
│   ├── raw/train.csv       # 45,000 training samples
│   ├── raw/test.csv        # 7,500 test samples
│   └── sample_reviews.csv  # 30 demo samples
├── templates/              # HTML templates
├── static/                 # CSS, JS, assets
├── tests/                  # pytest test suite
├── docs/
│   └── Sentiment_Analysis_Project_Report.pdf
└── requirements.txt
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/karthikmoka/sentiment-analysis-.git
cd sentiment-analysis-
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
```

**3. Train the model** *(skip if using the pre-trained model)*
```bash
python models/train_model.py
```

**4. Run the application**
```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## API

**POST** `/predict`
```json
{
  "text": "The product quality is excellent!"
}
```
**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.94,
  "probabilities": { "Negative": 0.02, "Neutral": 0.04, "Positive": 0.94 }
}
```

**GET** `/api/health` → Returns server status.

---

## Project Report

[Sentiment_Analysis_Project_Report.pdf](docs/Sentiment_Analysis_Project_Report.pdf)

---

## Repository

[https://github.com/karthikmoka/sentiment-analysis-](https://github.com/karthikmoka/sentiment-analysis-)
