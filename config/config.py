"""
Sentiment Analysis – Configuration v2.0
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR    = os.path.join(BASE_DIR, "dataset", "raw")
PROCESSED_DIR  = os.path.join(BASE_DIR, "dataset", "processed")
MODEL_PATH     = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
EVAL_PATH      = os.path.join(BASE_DIR, "models", "evaluation_results.json")
EXPERIMENT_PATH= os.path.join(BASE_DIR, "models", "experiment_results.json")
ANALYTICS_PATH = os.path.join(BASE_DIR, "results", "analytics.json")
UPLOAD_FOLDER  = os.path.join(BASE_DIR, "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")

# ── Labels & colours ─────────────────────────────────────────────────────────
LABEL_NAMES  = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_COLORS = {0: "#EF4444", 1: "#F59E0B", 2: "#22C55E"}  # red / amber / green

# ── TF-IDF (kept for backward compat – train_model.py now overrides) ─────────
TFIDF_MAX_FEATURES = 25000
TFIDF_NGRAM_RANGE  = (1, 3)
TFIDF_SUBLINEAR_TF = True

# ── Logistic Regression defaults ─────────────────────────────────────────────
LR_C            = 2.0
LR_CLASS_WEIGHT = "balanced"
LR_MAX_ITER     = 2000
LR_SOLVER       = "lbfgs"
LR_MULTI_CLASS  = "multinomial"

# ── Split ────────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── Upload ───────────────────────────────────────────────────────────────────
MAX_UPLOAD_MB      = 10
ALLOWED_EXTENSIONS = {"csv"}

# ── Flask ────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("SENTIMENT_ANALYSIS_SECRET", "sentiment-analysis-secret-key")
DEBUG      = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
