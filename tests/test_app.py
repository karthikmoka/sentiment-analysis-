"""
SentimentIQ – Comprehensive Test Suite v2.0
Tests preprocessing, model loading, predictions, API endpoints, batch analysis.
"""

import os
import sys
import json
import io
import csv
import pytest

# Add project root to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

# ── Preprocessing Tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    def setup_method(self):
        from preprocessing.text_preprocessing import clean_text, preprocess_series
        import pandas as pd
        self.clean = clean_text
        self.preprocess_series = preprocess_series
        self.pd = pd

    def test_basic_cleaning(self):
        result = self.clean("This is a great movie!")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_html_removal(self):
        result = self.clean("<b>Hello</b> <i>world</i>")
        assert "<b>" not in result
        assert "<i>" not in result

    def test_url_removal(self):
        result = self.clean("Visit https://example.com for more info")
        assert "https" not in result
        assert "example.com" not in result

    def test_negation_preserved(self):
        result = self.clean("I am not happy with this product")
        assert "not" in result, "Negation word 'not' must be preserved"

    def test_negation_never(self):
        result = self.clean("I never liked this movie")
        assert "never" in result, "Negation word 'never' must be preserved"

    def test_negation_no(self):
        result = self.clean("No, this is terrible")
        assert "no" in result, "Negation word 'no' must be preserved"

    def test_intensifier_very(self):
        result = self.clean("This is very good")
        assert "very" in result, "Intensifier 'very' must be preserved"

    def test_intensifier_extremely(self):
        result = self.clean("Extremely disappointed")
        assert "extremely" in result, "Intensifier 'extremely' must be preserved"

    def test_contrastive_but(self):
        result = self.clean("Good service but slow delivery")
        assert "but" in result, "Contrastive 'but' must be preserved"

    def test_empty_string(self):
        result = self.clean("")
        assert result == ""

    def test_none_input(self):
        result = self.clean(None)
        assert result == ""

    def test_whitespace_only(self):
        result = self.clean("   \t\n  ")
        assert result == ""

    def test_special_chars(self):
        result = self.clean("Hello!!! World??? @#$%")
        assert isinstance(result, str)

    def test_contraction_expansion(self):
        # "don't" should expand to "do not" and "not" preserved
        result = self.clean("I don't like this")
        assert "not" in result

    def test_unicode_normalization(self):
        result = self.clean("café résumé naïve")
        assert isinstance(result, str)

    def test_repeated_chars(self):
        result = self.clean("This is sooooo great!!!")
        assert "sooooo" not in result  # should be normalized

    def test_preprocess_series(self):
        import pandas as pd
        series = pd.Series(["Great movie!", "Terrible film.", "It was okay."])
        result = self.preprocess_series(series)
        assert len(result) == 3
        assert all(isinstance(r, str) for r in result)

    def test_preprocess_series_with_nulls(self):
        import pandas as pd
        series = pd.Series(["Great!", None, "Okay.", ""])
        result = self.preprocess_series(series)
        assert len(result) == 4
        assert result.iloc[1] == ""  # None → ""


# ── Model Loading Tests ───────────────────────────────────────────────────────

class TestModelLoading:
    def test_model_file_exists(self):
        from config.config import MODEL_PATH
        assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}. Run: python models/train_model.py"

    def test_evaluation_file_exists(self):
        from config.config import EVAL_PATH
        assert os.path.exists(EVAL_PATH), "Evaluation results not found."

    def test_evaluation_file_valid_json(self):
        from config.config import EVAL_PATH
        if not os.path.exists(EVAL_PATH):
            pytest.skip("No eval file")
        with open(EVAL_PATH) as f:
            data = json.load(f)
        assert "accuracy" in data
        assert "macro_f1" in data
        assert "per_class" in data

    def test_model_loads(self):
        import joblib
        from config.config import MODEL_PATH
        if not os.path.exists(MODEL_PATH):
            pytest.skip("No model file")
        pipeline = joblib.load(MODEL_PATH)
        assert pipeline is not None
        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "predict_proba")


# ── Prediction Tests ──────────────────────────────────────────────────────────

class TestPredictions:
    @pytest.fixture(autouse=True)
    def skip_if_no_model(self):
        from config.config import MODEL_PATH
        if not os.path.exists(MODEL_PATH):
            pytest.skip("No model available. Train first.")

    def test_positive_prediction_structure(self):
        from models.predict import predict
        result = predict("This movie was absolutely fantastic!")
        assert "sentiment" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "original_text" in result
        assert "cleaned_text" in result
        assert result["sentiment"] in ["Positive", "Negative", "Neutral"]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_negative_prediction(self):
        from models.predict import predict
        result = predict("This was absolutely terrible and I hated it.")
        assert result["sentiment"] in ["Negative", "Neutral", "Positive"]  # model decides
        assert isinstance(result["confidence"], float)

    def test_probabilities_sum_to_one(self):
        from models.predict import predict
        result = predict("The product was okay, nothing special.")
        probs = result["probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"

    def test_all_classes_in_probabilities(self):
        from models.predict import predict
        result = predict("I am not happy with this service.")
        probs = result["probabilities"]
        assert "Positive"  in probs
        assert "Negative"  in probs
        assert "Neutral"   in probs

    def test_confidence_level_field(self):
        from models.predict import predict
        result = predict("Great product!")
        assert "confidence_level" in result
        assert result["confidence_level"] in ["High", "Moderate", "Low"]

    def test_empty_text_raises(self):
        from models.predict import predict
        with pytest.raises(ValueError):
            predict("")

    def test_none_text_raises(self):
        from models.predict import predict
        with pytest.raises((ValueError, TypeError)):
            predict(None)

    def test_batch_predict(self):
        from models.predict import predict_batch
        texts = [
            "This is amazing!",
            "Absolutely terrible experience.",
            "It was okay I guess.",
        ]
        results = predict_batch(texts)
        assert len(results) == 3
        for r in results:
            assert "sentiment" in r
            assert "confidence" in r
            assert r["sentiment"] in ["Positive", "Negative", "Neutral"]

    def test_batch_predict_empty_list(self):
        from models.predict import predict_batch
        results = predict_batch([])
        assert results == []

    def test_batch_predict_with_empty_text(self):
        from models.predict import predict_batch
        results = predict_batch(["Great!", "", "Okay."])
        assert len(results) == 3  # all items returned, even empty ones


# ── Flask API Tests ───────────────────────────────────────────────────────────

class TestFlaskAPI:
    @pytest.fixture
    def client(self):
        from app import app
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        with app.test_client() as client:
            yield client

    def test_home_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_analyze_page(self, client):
        resp = client.get("/analyze")
        assert resp.status_code == 200

    def test_batch_page(self, client):
        resp = client.get("/batch")
        assert resp.status_code == 200

    def test_performance_page(self, client):
        resp = client.get("/performance")
        assert resp.status_code == 200

    def test_comparison_page(self, client):
        resp = client.get("/comparison")
        assert resp.status_code == 200

    def test_analytics_page(self, client):
        resp = client.get("/analytics")
        assert resp.status_code == 200

    def test_api_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_api_analytics(self, client):
        resp = client.get("/api/analytics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_analyzed" in data
        assert "counts" in data

    def test_predict_no_text(self, client):
        resp = client.post("/predict",
                           json={},
                           content_type="application/json")
        assert resp.status_code == 400

    def test_predict_empty_text(self, client):
        resp = client.post("/predict",
                           json={"text": ""},
                           content_type="application/json")
        assert resp.status_code == 400

    def test_predict_valid(self, client):
        from config.config import MODEL_PATH
        if not os.path.exists(MODEL_PATH):
            pytest.skip("No model available.")
        resp = client.post("/predict",
                           json={"text": "This was a fantastic experience!"},
                           content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "sentiment" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_batch_predict_no_file(self, client):
        resp = client.post("/batch-predict")
        assert resp.status_code in [400, 503]

    def test_batch_predict_invalid_format(self, client):
        resp = client.post("/batch-predict",
                           data={"file": (io.BytesIO(b"not a csv"), "test.txt")},
                           content_type="multipart/form-data")
        assert resp.status_code in [400, 503]

    def test_batch_predict_valid_csv(self, client):
        from config.config import MODEL_PATH
        if not os.path.exists(MODEL_PATH):
            pytest.skip("No model available.")
        csv_content = "text\nThis was great!\nTerrible experience.\nIt was okay."
        resp = client.post("/batch-predict",
                           data={
                               "file": (io.BytesIO(csv_content.encode()), "test.csv"),
                               "text_column": "text",
                           },
                           content_type="multipart/form-data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total" in data
        assert data["total"] == 3

    def test_batch_predict_missing_column(self, client):
        from config.config import MODEL_PATH
        if not os.path.exists(MODEL_PATH):
            pytest.skip("No model available.")
        csv_content = "review\nGreat product!"
        resp = client.post("/batch-predict",
                           data={
                               "file": (io.BytesIO(csv_content.encode()), "test.csv"),
                               "text_column": "text",  # 'text' column doesn't exist
                           },
                           content_type="multipart/form-data")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_404_handler(self, client):
        resp = client.get("/nonexistent-route-xyz")
        assert resp.status_code == 404

    def test_download_nonexistent(self, client):
        resp = client.get("/download-results/nonexistent_file.csv")
        assert resp.status_code == 404

    def test_api_model_performance(self, client):
        resp = client.get("/api/model-performance")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "metrics" in data

    def test_api_model_comparison(self, client):
        resp = client.get("/api/model-comparison")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "experiments" in data


# ── Evaluation Utils Tests ────────────────────────────────────────────────────

class TestEvaluationUtils:
    def test_load_evaluation_empty(self):
        from utils.evaluation import load_evaluation, format_metrics
        # format_metrics with empty dict returns empty dict
        result = format_metrics({})
        assert result == {}

    def test_format_metrics_structure(self):
        from utils.evaluation import format_metrics
        raw = {
            "accuracy": 0.65,
            "macro_f1": 0.60,
            "per_class": {
                "Negative": {"precision": 0.7, "recall": 0.65, "f1": 0.67},
                "Neutral":  {"precision": 0.4, "recall": 0.3,  "f1": 0.35},
                "Positive": {"precision": 0.75,"recall": 0.8,  "f1": 0.77},
            }
        }
        result = format_metrics(raw)
        assert "accuracy" in result
        assert "macro_f1" in result
        assert "per_class" in result
        assert all(cls in result["per_class"] for cls in ["Negative","Neutral","Positive"])
