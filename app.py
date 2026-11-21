"""
SentimentIQ Flask – Main Application v2.0
Run: python app.py  →  http://127.0.0.1:5000
"""

import os
import sys
import io
import json
import uuid
import logging
from datetime import datetime

import pandas as pd
from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, abort,
)

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from config.config import (
    SECRET_KEY, DEBUG, MAX_UPLOAD_MB,
    ALLOWED_EXTENSIONS, LABEL_NAMES,
    UPLOAD_FOLDER, RESULTS_FOLDER,
    EXPERIMENT_PATH, EVAL_PATH,
)
from models.predict   import predict, predict_batch, reload_model
from utils.evaluation import load_evaluation, format_metrics
from utils.analytics  import record_prediction, record_batch, get_analytics

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key                   = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _model_ready() -> bool:
    from config.config import MODEL_PATH
    return os.path.exists(MODEL_PATH)


def _load_experiments() -> list:
    if os.path.exists(EXPERIMENT_PATH):
        try:
            with open(EXPERIMENT_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return []


# ── Page Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    analytics   = get_analytics()
    model_ready = _model_ready()
    raw_eval    = load_evaluation()
    return render_template("index.html",
                           analytics=analytics,
                           model_ready=model_ready,
                           metrics=format_metrics(raw_eval))


@app.route("/analyze")
def analyze():
    return render_template("analyze.html", model_ready=_model_ready())


@app.route("/batch")
def batch():
    return render_template("batch_analysis.html", model_ready=_model_ready())


@app.route("/performance")
def performance():
    raw     = load_evaluation()
    metrics = format_metrics(raw)
    experiments = _load_experiments()
    return render_template("model_performance.html",
                           metrics=metrics, raw=raw,
                           experiments=experiments)


@app.route("/comparison")
def comparison():
    experiments = _load_experiments()
    return render_template("model_comparison.html", experiments=experiments)


@app.route("/analytics")
def analytics_page():
    return render_template("analytics.html",
                           analytics=get_analytics(),
                           model_ready=_model_ready())


@app.route("/about")
def about():
    return render_template("about.html")


# ── API: health ───────────────────────────────────────────────────────────────

@app.route("/api/health")
def api_health():
    return jsonify({
        "status":      "ok",
        "model_ready": _model_ready(),
        "timestamp":   datetime.utcnow().isoformat(),
        "version":     "2.0",
    })


# ── API: single prediction ────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    if not _model_ready():
        return jsonify({"error": "Model not trained yet. Run: python models/train_model.py"}), 503
    try:
        result = predict(text)
        record_prediction(result, source="single")
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        log.exception("Prediction error")
        return jsonify({"error": "Internal prediction error."}), 500


# ── API: batch prediction ─────────────────────────────────────────────────────

@app.route("/batch-predict", methods=["POST"])
def api_batch_predict():
    if not _model_ready():
        return jsonify({"error": "Model not trained yet."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if not file or not _allowed_file(file.filename):
        return jsonify({"error": "Please upload a valid .csv file."}), 400
    text_col = request.form.get("text_column", "text").strip()
    try:
        content = file.read().decode("utf-8", errors="replace")
        df      = pd.read_csv(io.StringIO(content))
    except Exception as exc:
        return jsonify({"error": f"Could not parse CSV: {exc}"}), 400

    if df.empty:
        return jsonify({"error": "CSV file is empty."}), 400
    if text_col not in df.columns:
        return jsonify({
            "error":   f"Column '{text_col}' not found.",
            "columns": df.columns.tolist(),
        }), 400

    texts = df[text_col].fillna("").astype(str).tolist()
    if not any(t.strip() for t in texts):
        return jsonify({"error": "All text values are empty."}), 400

    try:
        results = predict_batch(texts)
        record_batch(results)
    except Exception:
        log.exception("Batch prediction error")
        return jsonify({"error": "Batch prediction failed."}), 500

    df["predicted_sentiment"]   = [r["sentiment"]  for r in results]
    df["confidence"]            = [r["confidence"] for r in results]
    df["confidence_level"]      = [r["confidence_level"] for r in results]
    df["positive_probability"]  = [r["probabilities"].get("Positive", 0) for r in results]
    df["negative_probability"]  = [r["probabilities"].get("Negative", 0) for r in results]
    df["neutral_probability"]   = [r["probabilities"].get("Neutral",  0) for r in results]

    out_name = f"results_{uuid.uuid4().hex[:8]}.csv"
    df.to_csv(os.path.join(RESULTS_FOLDER, out_name), index=False)

    counts  = df["predicted_sentiment"].value_counts().to_dict()
    preview = [
        {
            "text":       str(row[text_col])[:150],
            "sentiment":  row["predicted_sentiment"],
            "confidence": f'{row["confidence"]*100:.1f}%',
            "pos_prob":   f'{row["positive_probability"]*100:.1f}%',
            "neg_prob":   f'{row["negative_probability"]*100:.1f}%',
            "neu_prob":   f'{row["neutral_probability"]*100:.1f}%',
        }
        for _, row in df.head(100).iterrows()
    ]
    return jsonify({
        "total":         len(results),
        "counts":        counts,
        "preview":       preview,
        "download_file": out_name,
        "columns":       df.columns.tolist(),
    })


# ── API: analytics ────────────────────────────────────────────────────────────

@app.route("/api/analytics")
def api_analytics():
    return jsonify(get_analytics())


# ── API: model performance ────────────────────────────────────────────────────

@app.route("/api/model-performance")
def api_model_performance():
    raw     = load_evaluation()
    metrics = format_metrics(raw)
    return jsonify({"metrics": metrics, "raw": raw})


# ── API: model comparison ─────────────────────────────────────────────────────

@app.route("/api/model-comparison")
def api_model_comparison():
    experiments = _load_experiments()
    return jsonify({"experiments": experiments})


# ── Download CSV ──────────────────────────────────────────────────────────────

@app.route("/download-results/<filename>")
def download_results(filename):
    safe = os.path.basename(filename)
    path = os.path.join(RESULTS_FOLDER, safe)
    if not os.path.exists(path):
        abort(404)
    return send_from_directory(RESULTS_FOLDER, safe, as_attachment=True)


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File too large. Max {MAX_UPLOAD_MB} MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _model_ready():
        log.warning("No model found. Run: python models/train_model.py")
    log.info("Starting SentimentIQ → http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
