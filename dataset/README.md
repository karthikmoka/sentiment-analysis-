# Dataset Documentation – Sentiment Analysis

## 📊 Overview

This dataset is an enterprise-grade, curated multi-domain customer review corpus created specifically for **Multi-Class Sentiment Classification** across three classes: **Positive (2)**, **Negative (0)**, and **Neutral (1)**.

To eliminate severe class imbalance issues and resolve the common degradation of sentiment models on impartial content, this dataset enforces **strict 1:1:1 class balancing** across both training and held-out test splits.

---

## 📈 Dataset Statistics

| Split | Negative (0) | Neutral (1) | Positive (2) | Total Samples | File Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Set (`dataset/raw/train.csv`)** | 15,000 | 15,000 | 15,000 | **45,000** | ~33 MB |
| **Testing Set (`dataset/raw/test.csv`)** | 2,500 | 2,500 | 2,500 | **7,500** | ~5.5 MB |
| **Total Benchmark Corpus** | **17,500** | **17,500** | **17,500** | **52,500** | **~38.5 MB** |

---

## 🏷️ Labeling Methodology & Star Rating Mapping

To ensure maximum signal-to-noise ratio and eliminate ambiguous borderline ratings:
- **1-Star Reviews** $\rightarrow$ **Negative (0)**: Extreme dissatisfaction, defects, critical failures.
- **3-Star Reviews** $\rightarrow$ **Neutral (1)**: Moderate feedback, balanced pros and cons, factual statements, average satisfaction.
- **5-Star Reviews** $\rightarrow$ **Positive (2)**: High satisfaction, enthusiastic praise, superior quality.

*Note: Ambiguous 2-star and 4-star borderline reviews are excluded to maintain clean class boundaries for high-signal model training.*

---

## 📋 Schema Definition

Both `train.csv` and `test.csv` follow a standardized 4-column schema:

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `text` | String | Raw natural language review content | `"The battery life is acceptable, but the software crashes constantly."` |
| `label` | Integer | Target numerical class index (0, 1, or 2) | `0` |
| `label_name` | String | Human-readable sentiment label | `"Negative"` |
| `raw_5class_label` | Integer | Original review rating (1, 3, or 5) | `1` |

---

## 🔬 Preprocessing & Context Preservation

Textual inputs undergo an automated context-preserving NLP preprocessing pipeline:
1. **HTML & URL Normalization:** Stripping markup tags (`<br />`, `<p>`) and hyperlinks.
2. **Emoji & Emoticon Mapping:** Translating emoticons (`:)`, `:-(`, `:-|`) into descriptive semantic tokens.
3. **Contraction Expansion:** Expanding contractions (`don't` $\rightarrow$ `do not`, `can't` $\rightarrow$ `cannot`).
4. **Negation Scope Preservation:** Explicitly preventing standard stop-word filtering from discarding critical negation words (`not`, `no`, `never`, `hardly`, `barely`, `without`) and contrastive conjunctions (`but`, `although`).
5. **WordNet Lemmatization:** Reducing inflected word forms to canonical morphological dictionary lemmas.

---

## 📁 Sample Demonstration Files

- `dataset/sample_reviews.csv`: A lightweight 30-sample demonstration spreadsheet designed for instant drag-and-drop batch analysis testing via the web interface.
