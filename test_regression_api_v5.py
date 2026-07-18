"""
Test suite for regression_api_v5.py

Run with:  pytest test_regression_api_v5.py -v

As with v4, this suite was written without being able to execute it (no
network access to install fastapi/scikit-learn/scipy in the environment
that produced it). Run it yourself before deploying - especially the
multi-feature and drift/A-B-test tests, which exercise the biggest changes
from v4.
"""
import time
import pytest
from fastapi.testclient import TestClient

from regression_api_v5 import app, rate_limiter, model_registry

client = TestClient(app)


def linear_payload(n=30, **overrides):
    data = client.get(f"/sample/linear?n_samples={n}&noise=0.1&seed=42").json()["data"]
    payload = {"data": data, "regression_type": "all", "seed": 42}
    payload.update(overrides)
    return payload


def logistic_payload(n=40, **overrides):
    data = client.get(f"/sample/logistic?n_samples={n}&noise=0.1&seed=42").json()["data"]
    payload = {"data": data, "regression_type": "logistic", "seed": 42}
    payload.update(overrides)
    return payload


def multifeature_payload(n=40, **overrides):
    data = client.get(f"/sample/multifeature?n_samples={n}&noise=0.1&seed=42").json()["data"]
    payload = {"data": data, "regression_type": "ridge", "seed": 42}
    payload.update(overrides)
    return payload


class TestJSONSafety:
    def test_json_safe_converts_inf_and_nan_to_none(self):
        from regression_api_v5 import _json_safe
        assert _json_safe(float("inf")) is None
        assert _json_safe(float("-inf")) is None
        assert _json_safe(float("nan")) is None
        assert _json_safe(1.5) == 1.5
        assert _json_safe({"a": float("inf"), "b": [1.0, float("nan"), 2.0]}) == {"a": None, "b": [1.0, None, 2.0]}

    def test_fit_survives_near_perfectly_separable_logistic_data(self):
        # Regression test: near-perfectly-separable classification can push
        # logistic coefficients large enough that exp(coef) or log_loss
        # produces a literal inf, which used to crash the whole /fit
        # response with a 500 (Starlette's JSON encoder rejects Infinity).
        data = [{"x": float(i), "y": float(i), "label": int(i > 15)} for i in range(30)]
        r = client.post("/fit", json={"data": data, "regression_type": "all"})
        assert r.status_code == 200


class TestBasics:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["version"] == "5.0"

    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert "cache_backend" in r.json()

    @pytest.mark.parametrize("kind", ["linear", "logistic", "nonlinear", "heteroscedastic", "outliers", "multifeature"])
    def test_sample_kinds(self, kind):
        r = client.get(f"/sample/{kind}?n_samples=20")
        assert r.status_code == 200
        assert len(r.json()["data"]) == 20


class TestMultiFeature:
    def test_multifeature_fit(self):
        r = client.post("/fit", json=multifeature_payload())
        assert r.status_code == 200
        ridge = r.json()["ridge"]
        assert set(ridge["feature_cols"]) == {"x1", "x2"}
        assert ridge["visualization"]["kind"] == "predicted_vs_actual"

    def test_single_feature_still_gets_line_viz(self):
        r = client.post("/fit", json=linear_payload(regression_type="linear"))
        assert r.status_code == 200
        assert r.json()["linear"]["visualization"]["kind"] == "line"

    def test_no_feature_columns_rejected(self):
        data = [{"y": float(i)} for i in range(6)]
        r = client.post("/fit", json={"data": data, "regression_type": "linear"})
        assert r.status_code == 400

    def test_inconsistent_feature_columns_rejected(self):
        data = [{"x1": 1.0, "x2": 2.0, "y": 1.0}] * 3 + [{"x1": 1.0, "y": 1.0}] * 3
        r = client.post("/fit", json={"data": data, "regression_type": "linear"})
        assert r.status_code == 400

    def test_non_numeric_feature_rejected(self):
        data = [{"x1": "not_a_number", "y": 1.0}] * 6
        r = client.post("/fit", json={"data": data, "regression_type": "linear"})
        assert r.status_code in (400, 422)

    def test_multifeature_predict_requires_all_features(self):
        r = client.post("/fit", json=multifeature_payload())
        model_id = r.json()["ridge"]["model_id"]

        missing = client.post(f"/models/{model_id}/predict", json={"features": {"x1": 1.0}})
        assert missing.status_code == 400

        ok = client.post(f"/models/{model_id}/predict", json={"features": {"x1": 1.0, "x2": 2.0}})
        assert ok.status_code == 200
        assert "prediction" in ok.json()

    def test_multifeature_batch_predict(self):
        r = client.post("/fit", json=multifeature_payload())
        model_id = r.json()["ridge"]["model_id"]
        batch = client.post(f"/models/{model_id}/batch_predict", json={
            "rows": [{"x1": 1.0, "x2": 2.0}, {"x1": 3.0, "x2": -1.0}]
        })
        assert batch.status_code == 200
        assert len(batch.json()["predictions"]) == 2


class TestFittingCore:
    def test_fit_all_models(self):
        r = client.post("/fit", json=linear_payload())
        assert r.status_code == 200
        for key in ("linear", "ridge", "lasso", "elasticnet", "logistic"):
            assert key in r.json()

    def test_fit_is_cached(self):
        payload = linear_payload()
        r1 = client.post("/fit", json=payload)
        r2 = client.post("/fit", json=payload)
        assert r1.json() == r2.json()

    def test_advanced_fit_bootstrap_and_cv(self):
        payload = linear_payload(regression_type="ridge", cross_validation=True, cv_folds=3, bootstrap_iterations=25)
        r = client.post("/advanced_fit", json=payload)
        assert r.status_code == 200
        assert r.json()["ridge"]["bootstrap_ci"]["iterations"] > 0

    def test_logistic_needs_both_classes(self):
        data = [{"x": float(i), "y": 0.0, "label": 0} for i in range(6)]
        r = client.post("/fit", json={"data": data, "regression_type": "logistic"})
        assert r.status_code == 400


class TestLegacyPredict:
    def test_predict_rejects_all(self):
        r = client.post("/predict", json={"model_type": "all", "coefficients": [1.0], "intercept": 0.0, "x_value": 1.0})
        assert r.status_code == 422

    def test_predict_linear(self):
        r = client.post("/predict", json={"model_type": "linear", "coefficients": [2.0], "intercept": 1.0, "x_value": 3.0})
        assert r.status_code == 200
        assert r.json()["prediction"] == pytest.approx(7.0)


class TestModelRegistry:
    def test_predict_matches_line(self):
        r = client.post("/fit", json=linear_payload(regression_type="linear"))
        result = r.json()["linear"]
        model_id = result["model_id"]
        feature = result["feature_names"][0]
        x_line = result["visualization"]["x_line"]
        y_line = result["visualization"]["y_line"]
        mid = len(x_line) // 2

        pred = client.post(f"/models/{model_id}/predict", json={"features": {feature: x_line[mid]}})
        assert pred.status_code == 200
        assert pred.json()["prediction"] == pytest.approx(y_line[mid], abs=1e-6)

    def test_unknown_model_404(self):
        r = client.post("/models/nope/predict", json={"features": {"x": 1.0}})
        assert r.status_code == 404

    def test_list_and_delete(self):
        r = client.post("/fit", json=linear_payload(regression_type="linear"))
        model_id = r.json()["linear"]["model_id"]
        assert any(m["model_id"] == model_id for m in client.get("/models").json()["models"])
        assert client.delete(f"/models/{model_id}").status_code == 200
        assert client.post(f"/models/{model_id}/predict", json={"features": {"x": 1.0}}).status_code == 404


class TestDrift:
    def test_no_drift_same_distribution(self):
        ref = client.get("/sample/linear?n_samples=200&seed=1").json()["data"]
        cur = client.get("/sample/linear?n_samples=200&seed=2").json()["data"]
        r = client.post("/drift", json={"reference_data": ref, "current_data": cur})
        assert r.status_code == 200
        assert "x" in r.json()["feature_drift"]

    def test_drift_detected_on_shifted_distribution(self):
        ref = client.get("/sample/linear?n_samples=200&seed=1").json()["data"]
        cur_raw = client.get("/sample/linear?n_samples=200&seed=1").json()["data"]
        shifted = [{"x": row["x"] + 50.0, "y": row["y"]} for row in cur_raw]
        r = client.post("/drift", json={"reference_data": ref, "current_data": shifted})
        assert r.status_code == 200
        body = r.json()
        assert body["drift_detected"] is True
        assert "x" in body["drifted_features"]

    def test_drift_mismatched_columns_rejected(self):
        ref = client.get("/sample/linear?n_samples=20").json()["data"]
        cur = client.get("/sample/multifeature?n_samples=20").json()["data"]
        r = client.post("/drift", json={"reference_data": ref, "current_data": cur})
        assert r.status_code == 400


class TestABTest:
    def test_ab_test_same_model_no_significant_difference(self):
        payload = linear_payload(regression_type="ridge")
        model_id = client.post("/fit", json=payload).json()["ridge"]["model_id"]
        r = client.post("/ab_test", json={
            "model_id_a": model_id, "model_id_b": model_id,
            "test_data": payload["data"], "bootstrap_iterations": 100,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["score_difference_a_minus_b"] == pytest.approx(0.0, abs=1e-9)
        assert body["significant"] is False

    def test_ab_test_mismatched_model_types_rejected(self):
        payload = linear_payload()
        results = client.post("/fit", json=payload).json()
        r = client.post("/ab_test", json={
            "model_id_a": results["linear"]["model_id"],
            "model_id_b": results["logistic"]["model_id"],
            "test_data": payload["data"], "bootstrap_iterations": 100,
        })
        assert r.status_code == 400

    def test_ab_test_unknown_model_404(self):
        r = client.post("/ab_test", json={
            "model_id_a": "nope", "model_id_b": "alsonope",
            "test_data": linear_payload()["data"],
        })
        assert r.status_code == 404


class TestAsyncJobs:
    def test_fit_async_roundtrip(self):
        submitted = client.post("/fit_async", json=linear_payload(regression_type="ridge"))
        assert submitted.status_code == 200
        job_id = submitted.json()["job_id"]

        deadline = time.time() + 10
        status = None
        while time.time() < deadline:
            poll = client.get(f"/jobs/{job_id}")
            assert poll.status_code == 200
            status = poll.json()
            if status["status"] in ("done", "failed"):
                break
            time.sleep(0.1)

        assert status["status"] == "done", status
        assert "ridge" in status["result"]

    def test_unknown_job_404(self):
        r = client.get("/jobs/does-not-exist")
        assert r.status_code == 404


class TestChartsAndExport:
    def test_chart_demo_png(self):
        r = client.get("/chart/demo?n_samples=30")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"

    def test_chart_multifeature(self):
        r = client.post("/chart", json={**multifeature_payload(regression_type="all"), "chart_format": "png"})
        assert r.status_code == 200

    def test_export_csv_includes_predictions(self):
        r = client.post("/export", json={**linear_payload(), "format": "csv"})
        assert r.status_code == 200
        assert "linear_predicted" in r.text


class TestRateLimiting:
    def test_rate_limit_429(self):
        orig_max, orig_win = rate_limiter.max_requests, rate_limiter.window_seconds
        rate_limiter.max_requests = 3
        rate_limiter.window_seconds = 60
        rate_limiter._hits.clear()
        try:
            statuses = [client.get("/theory").status_code for _ in range(6)]
            assert 429 in statuses
        finally:
            rate_limiter.max_requests = orig_max
            rate_limiter.window_seconds = orig_win
            rate_limiter._hits.clear()
