"""
Test suite for regression_api_v4.py

Run with:  pytest test_regression_api_v4.py -v

Note: this suite could not be executed in the environment that produced it
(no network access to install fastapi/scikit-learn/etc). It's provided so
you can verify the upgrade yourself with one command; please run it before
deploying. If anything fails, that's useful signal - open an issue/fix
rather than assuming the code is correct just because it shipped.
"""
import pytest
from fastapi.testclient import TestClient

from regression_api_v4 import app, rate_limiter

client = TestClient(app)


def sample_linear_payload(n=30, **overrides):
    resp = client.get(f"/sample/linear?n_samples={n}&noise=0.1&seed=42")
    assert resp.status_code == 200
    data = resp.json()["data"]
    payload = {"data": data, "regression_type": "all", "seed": 42}
    payload.update(overrides)
    return payload


def sample_logistic_payload(n=40, **overrides):
    resp = client.get(f"/sample/logistic?n_samples={n}&noise=0.1&seed=42")
    assert resp.status_code == 200
    data = resp.json()["data"]
    payload = {"data": data, "regression_type": "logistic", "seed": 42}
    payload.update(overrides)
    return payload


class TestBasics:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["version"] == "4.0"

    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "registered_models" in body

    def test_theory(self):
        r = client.get("/theory")
        assert r.status_code == 200
        assert "linear_regression" in r.json()

    @pytest.mark.parametrize("kind", ["linear", "logistic", "nonlinear", "heteroscedastic", "outliers"])
    def test_sample_kinds(self, kind):
        r = client.get(f"/sample/{kind}?n_samples=20")
        assert r.status_code == 200
        assert len(r.json()["data"]) == 20

    def test_removed_multicollinear_kind_is_rejected(self):
        # v4 dropped this sample kind because it emitted an `x2` field the
        # API never actually used as a second feature (see CHANGELOG).
        r = client.get("/sample/multicollinear?n_samples=20")
        assert r.status_code == 422


class TestFitting:
    def test_fit_all_models(self):
        r = client.post("/fit", json=sample_linear_payload())
        assert r.status_code == 200
        body = r.json()
        for key in ("linear", "ridge", "lasso", "elasticnet", "logistic"):
            assert key in body
            assert "model_id" in body[key]

    def test_fit_is_cached(self):
        payload = sample_linear_payload()
        r1 = client.post("/fit", json=payload)
        r2 = client.post("/fit", json=payload)
        assert r1.json() == r2.json()

    def test_advanced_fit_with_bootstrap_and_cv(self):
        payload = sample_linear_payload(
            regression_type="ridge",
            cross_validation=True,
            cv_folds=3,
            bootstrap_iterations=25,
        )
        r = client.post("/advanced_fit", json=payload)
        assert r.status_code == 200
        ridge = r.json()["ridge"]
        assert "cv_r2_mean" in ridge["metrics"]
        assert "bootstrap_ci" in ridge
        assert ridge["bootstrap_ci"]["iterations"] > 0

    def test_hyperparameter_tuning(self):
        payload = sample_linear_payload(regression_type="lasso", hyperparameter_tuning=True, cv_folds=3)
        r = client.post("/advanced_fit", json=payload)
        assert r.status_code == 200
        assert "optimal_alpha" in r.json()["lasso"]["metrics"]

    def test_feature_selection_does_not_crash_with_degree_1(self):
        # k_best (default 5) is larger than the single available feature at
        # degree=1; v4 should clamp instead of raising a raw sklearn error.
        payload = sample_linear_payload(regression_type="ridge", feature_selection="k_best_f")
        r = client.post("/fit", json=payload)
        assert r.status_code == 200

    def test_polynomial_and_interactions(self):
        payload = sample_linear_payload(regression_type="ridge", polynomial_degree=3, include_interactions=True)
        r = client.post("/fit", json=payload)
        assert r.status_code == 200
        assert len(r.json()["ridge"]["feature_names"]) >= 3

    def test_logistic_needs_both_classes(self):
        data = [{"x": float(i), "y": 0.0, "label": 0} for i in range(6)]
        r = client.post("/fit", json={"data": data, "regression_type": "logistic"})
        assert r.status_code == 400

    def test_test_ratio_too_high(self):
        payload = sample_linear_payload(n=5, test_ratio=0.49)
        payload["test_ratio"] = 0.49
        r = client.post("/fit", json=payload)
        # Either validated away or explicitly rejected - never a 500.
        assert r.status_code in (200, 400)

    def test_cv_folds_more_than_training_rows_is_rejected_cleanly(self):
        payload = sample_linear_payload(n=8, cross_validation=True, cv_folds=10)
        r = client.post("/fit", json=payload)
        assert r.status_code == 400


class TestModelRegistry:
    def test_fit_then_predict_by_model_id(self):
        r = client.post("/fit", json=sample_linear_payload(regression_type="linear"))
        model_id = r.json()["linear"]["model_id"]

        pred = client.post(f"/models/{model_id}/predict", json={"x_value": 3.0})
        assert pred.status_code == 200
        assert "prediction" in pred.json()

    def test_predict_matches_visualization_line_closely(self):
        r = client.post("/fit", json=sample_linear_payload(regression_type="linear"))
        result = r.json()["linear"]
        model_id = result["model_id"]
        x_line = result["visualization"]["x_line"]
        y_line = result["visualization"]["y_line"]
        mid = len(x_line) // 2

        pred = client.post(f"/models/{model_id}/predict", json={"x_value": x_line[mid]})
        assert pred.status_code == 200
        assert pred.json()["prediction"] == pytest.approx(y_line[mid], abs=1e-6)

    def test_batch_predict_by_model_id(self):
        r = client.post("/fit", json=sample_linear_payload(regression_type="ridge"))
        model_id = r.json()["ridge"]["model_id"]
        pred = client.post(f"/models/{model_id}/batch_predict", json={"x_values": [1.0, 2.0, 3.0]})
        assert pred.status_code == 200
        assert len(pred.json()["predictions"]) == 3

    def test_logistic_model_predict_returns_probability(self):
        r = client.post("/fit", json=sample_logistic_payload())
        model_id = r.json()["logistic"]["model_id"]
        pred = client.post(f"/models/{model_id}/predict", json={"x_value": 5.0})
        assert pred.status_code == 200
        body = pred.json()
        assert 0.0 <= body["probability"] <= 1.0

    def test_unknown_model_id_404(self):
        r = client.post("/models/does-not-exist/predict", json={"x_value": 1.0})
        assert r.status_code == 404

    def test_models_list_and_delete(self):
        r = client.post("/fit", json=sample_linear_payload(regression_type="linear"))
        model_id = r.json()["linear"]["model_id"]
        listed = client.get("/models").json()["models"]
        assert any(m["model_id"] == model_id for m in listed)

        deleted = client.delete(f"/models/{model_id}")
        assert deleted.status_code == 200
        assert client.post(f"/models/{model_id}/predict", json={"x_value": 1.0}).status_code == 404


class TestLegacyPredictBugfix:
    def test_predict_rejects_all_as_model_type(self):
        # Regression test: v3 tried to reject "all" via Field(exclude=...),
        # which only affects serialization, not validation - "all" was
        # silently accepted and would crash downstream. v4 uses a real
        # validator.
        r = client.post("/predict", json={
            "model_type": "all", "coefficients": [1.0], "intercept": 0.0, "x_value": 2.0,
        })
        assert r.status_code == 422

    def test_predict_linear_works(self):
        r = client.post("/predict", json={
            "model_type": "linear", "coefficients": [2.0], "intercept": 1.0, "x_value": 3.0,
        })
        assert r.status_code == 200
        assert r.json()["prediction"] == pytest.approx(7.0)

    def test_batch_predict_rejects_all(self):
        r = client.post("/batch_predict", json={
            "model_type": "all", "coefficients": [1.0], "intercept": 0.0, "x_values": [1.0, 2.0],
        })
        assert r.status_code == 422


class TestChartsAndExport:
    def test_chart_demo_png(self):
        r = client.get("/chart/demo?n_samples=30")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"

    def test_chart_demo_svg(self):
        r = client.get("/chart/demo?n_samples=30&format=svg")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/svg+xml"

    def test_export_csv(self):
        r = client.post("/export", json={**sample_linear_payload(), "format": "csv"})
        assert r.status_code == 200
        assert "linear_predicted" in r.text

    def test_export_json(self):
        r = client.post("/export", json={**sample_linear_payload(), "format": "json"})
        assert r.status_code == 200
        rows = r.json()
        assert "linear_predicted" in rows[0]

    def test_export_excel(self):
        r = client.post("/export", json={**sample_linear_payload(), "format": "excel"})
        assert r.status_code == 200
        assert len(r.content) > 0


class TestCompareAndDiagnostics:
    def test_compare(self):
        r = client.post("/compare", json=sample_linear_payload())
        assert r.status_code == 200
        assert "recommendation" in r.json()

    def test_compare_rejects_non_all(self):
        r = client.post("/compare", json=sample_linear_payload(regression_type="ridge"))
        assert r.status_code == 400

    def test_diagnostics(self):
        r = client.post("/diagnostics", json=sample_linear_payload())
        assert r.status_code == 200
        assert "linear" in r.json()


class TestRateLimiting:
    def test_rate_limit_returns_429(self):
        original_max = rate_limiter.max_requests
        original_window = rate_limiter.window_seconds
        rate_limiter.max_requests = 3
        rate_limiter.window_seconds = 60
        rate_limiter._hits.clear()
        try:
            statuses = [client.get("/theory").status_code for _ in range(6)]
            assert 429 in statuses
        finally:
            rate_limiter.max_requests = original_max
            rate_limiter.window_seconds = original_window
            rate_limiter._hits.clear()


class TestWebsocket:
    def test_websocket_fit_roundtrip(self):
        with client.websocket_connect("/ws/stream") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "connected"

            sample = client.get("/sample/linear?n_samples=15").json()["data"]
            ws.send_json({"action": "fit", "data": sample, "regression_type": "linear"})

            progress = ws.receive_json()
            assert progress["type"] == "progress"
            progress2 = ws.receive_json()
            assert progress2["type"] == "progress"
            result = ws.receive_json()
            assert result["type"] == "result"
            assert "linear" in result["data"]
