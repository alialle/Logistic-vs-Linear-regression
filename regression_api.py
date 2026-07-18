"""
Linear vs Logistic Regression API
----------------------------------
Fits linear and/or logistic regression on either generated sample data
or custom user-supplied data, returns metrics + plotting data as JSON,
and can also render a PNG chart directly.

Run it with:
    uvicorn regression_api:app --reload

Then open:
    http://127.0.0.1:8000/docs        <- interactive API playground
    http://127.0.0.1:8000/theory      <- background on both models
    http://127.0.0.1:8000/chart/demo  <- quick PNG demo, no setup needed
"""

import io
from typing import List, Literal, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # render charts without needing a display
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    log_loss,
)

app = FastAPI(title="Linear vs Logistic Regression API", version="2.0")

# ==================== DATA MODELS ====================


class DataPoint(BaseModel):
    x: float
    y: float
    label: Optional[int] = None  # 0 or 1, used for logistic regression


class RegressionRequest(BaseModel):
    data: List[DataPoint]
    regression_type: Literal["linear", "logistic", "both"]
    polynomial_degree: int = Field(default=1, ge=1, le=6)
    test_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = 42


class PredictionRequest(BaseModel):
    model_type: Literal["linear", "logistic"]
    coefficients: List[float]
    intercept: float
    x_value: float
    polynomial_degree: int = Field(default=1, ge=1, le=6)


class BatchPredictionRequest(BaseModel):
    model_type: Literal["linear", "logistic"]
    coefficients: List[float]
    intercept: float
    x_values: List[float]
    polynomial_degree: int = Field(default=1, ge=1, le=6)


# ==================== UTILITY FUNCTIONS ====================


def generate_sample_data(kind: str, n_samples: int = 100, noise: float = 0.1, seed: int = 42):
    """Generate demo datasets. Seeded per-call (not globally) so one
    request's randomness can't leak into another's."""
    rng = np.random.default_rng(seed)

    if kind == "linear":
        x = np.linspace(0, 10, n_samples)
        y = 2.5 * x + 1.0 + rng.normal(0, noise * 10, n_samples)
        return [{"x": float(xi), "y": float(yi)} for xi, yi in zip(x, y)]

    elif kind == "logistic":
        # "Hours studied -> pass/fail" style demo: one continuous driver,
        # a numeric score derived from it, and a pass/fail label from a
        # threshold on that score - label and y stay consistent by
        # construction, unlike a scheme that labels from a noisy
        # probability draw.
        x = np.linspace(0, 10, n_samples)
        true_w, true_b = 1.4, -7.0
        score = true_w * x + true_b + rng.normal(0, noise * 10, n_samples)
        probabilities = 1 / (1 + np.exp(-score))
        labels = (probabilities > 0.5).astype(int)
        return [
            {"x": float(xi), "y": float(pi), "label": int(li)}
            for xi, pi, li in zip(x, probabilities, labels)
        ]

    elif kind == "nonlinear":
        x = np.linspace(0, 10, n_samples)
        y = 0.5 * x**2 - 3 * x + 5 + rng.normal(0, noise * 5, n_samples)
        return [{"x": float(xi), "y": float(yi)} for xi, yi in zip(x, y)]

    else:
        raise ValueError(f"Unknown sample data kind: {kind}")


def _derive_binary_labels(data: List[DataPoint], y: np.ndarray) -> np.ndarray:
    """Build a label array aligned 1:1 with `data`/`y` (same length,
    same order). If every point already has a label, use those. If
    none do, derive from a median split of y. Mixing (some labelled,
    some not) is rejected explicitly rather than silently
    reindexing - silently dropping unlabelled rows here would
    desynchronize the label array from X/y and corrupt the split.
    """
    n_labelled = sum(1 for d in data if d.label is not None)

    if n_labelled == len(data):
        return np.array([d.label for d in data], dtype=int)

    if n_labelled == 0:
        return (y > np.median(y)).astype(int)

    raise HTTPException(
        status_code=400,
        detail=(
            f"{n_labelled} of {len(data)} points have a label and the rest don't. "
            "Either label every point or label none (in which case a median "
            "split of y is used automatically)."
        ),
    )


def _fit_linear(X, y, degree, test_ratio, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed
    )

    if degree > 1:
        poly = PolynomialFeatures(degree=degree)
        X_train_t = poly.fit_transform(X_train)
        X_test_t = poly.transform(X_test)
        X_plot_raw = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_plot_t = poly.transform(X_plot_raw)
        feature_names = list(poly.get_feature_names_out(["x"]))
    else:
        X_train_t, X_test_t = X_train, X_test
        X_plot_raw = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_plot_t = X_plot_raw
        feature_names = ["x"]

    model = LinearRegression()
    model.fit(X_train_t, y_train)

    y_pred_train = model.predict(X_train_t)
    y_pred_test = model.predict(X_test_t)
    y_plot = model.predict(X_plot_t)

    equation = " + ".join(f"{c:.4f}*{n}" for c, n in zip(model.coef_, feature_names))
    equation = f"y = {equation} + {model.intercept_:.4f}"

    return {
        "type": "Linear Regression",
        "equation": equation,
        "coefficients": [float(c) for c in model.coef_],
        "intercept": float(model.intercept_),
        "feature_names": feature_names,
        "metrics": {
            "r2_score": float(r2_score(y_test, y_pred_test)),
            "mse": float(mean_squared_error(y_test, y_pred_test)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        },
        "visualization": {
            "x_original": [float(v) for v in X.flatten()],
            "y_original": [float(v) for v in y],
            "x_line": [float(v[0]) for v in X_plot_raw],
            "y_line": [float(v) for v in y_plot],
            "predictions": {
                "train": [float(v) for v in y_pred_train],
                "test": [float(v) for v in y_pred_test],
            },
        },
        "_model": model,
    }


def _fit_logistic(X, y_log, test_ratio, seed):
    class_counts = np.bincount(y_log)
    if len(class_counts) < 2 or class_counts.min() == 0:
        raise HTTPException(
            status_code=400,
            detail="Logistic regression needs both classes (0 and 1) present in the data.",
        )

    # Stratify so the test split can't end up single-class (which broke
    # confusion_matrix/accuracy in the unstratified version).
    can_stratify = class_counts.min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_log,
        test_size=test_ratio,
        random_state=seed,
        stratify=y_log if can_stratify else None,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_prob_plot = model.predict_proba(X_plot)[:, 1]

    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    w, b = model.coef_[0][0], model.intercept_[0]

    return {
        "type": "Logistic Regression",
        "equation": f"P(y=1) = 1 / (1 + exp(-({w:.4f}*x + {b:.4f})))",
        "coefficients": [float(c) for c in model.coef_[0]],
        "intercept": float(b),
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred_test)),
            "log_loss": float(log_loss(y_test, y_prob_test, labels=[0, 1])),
            "confusion_matrix": cm.tolist(),
            "true_positives": int(cm[1, 1]),
            "false_positives": int(cm[0, 1]),
            "true_negatives": int(cm[0, 0]),
            "false_negatives": int(cm[1, 0]),
        },
        "visualization": {
            "x_original": [float(v) for v in X.flatten()],
            "labels": [int(v) for v in y_log],
            "x_line": [float(v[0]) for v in X_plot],
            "probability_line": [float(v) for v in y_prob_plot],
            "decision_boundary": float(-b / w) if w != 0 else None,
            "predictions": {
                "train": [int(v) for v in y_pred_train],
                "test": [int(v) for v in y_pred_test],
                "probabilities_train": [float(v) for v in y_prob_train],
                "probabilities_test": [float(v) for v in y_prob_test],
            },
        },
        "_model": model,
    }


def run_fit(request: RegressionRequest):
    """Core fitting logic shared by /fit, /compare, and /chart."""
    if len(request.data) < 5:
        raise HTTPException(status_code=400, detail="Need at least 5 data points")

    n_test = max(1, int(len(request.data) * request.test_ratio))
    if n_test >= len(request.data):
        raise HTTPException(
            status_code=400,
            detail="test_ratio leaves no data for training - lower it or add more points.",
        )

    X = np.array([[d.x] for d in request.data])
    y = np.array([d.y for d in request.data])

    results = {}

    if request.regression_type in ("linear", "both"):
        results["linear"] = _fit_linear(
            X, y, request.polynomial_degree, request.test_ratio, request.seed
        )

    if request.regression_type in ("logistic", "both"):
        y_log = _derive_binary_labels(request.data, y)
        results["logistic"] = _fit_logistic(X, y_log, request.test_ratio, request.seed)

    return results


def _predict_value(model_type, coefficients, intercept, x_value, polynomial_degree):
    if model_type == "linear":
        features = [x_value**i for i in range(1, polynomial_degree + 1)] if polynomial_degree > 1 else [x_value]
        if len(features) != len(coefficients):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(features)} coefficients for polynomial_degree={polynomial_degree}, got {len(coefficients)}.",
            )
        prediction = sum(c * f for c, f in zip(coefficients, features)) + intercept
        return {"model": "linear", "input": x_value, "prediction": float(prediction), "type": "continuous"}

    z = coefficients[0] * x_value + intercept
    probability = 1 / (1 + np.exp(-z))
    return {
        "model": "logistic",
        "input": x_value,
        "probability": float(probability),
        "prediction": int(probability > 0.5),
        "type": "binary",
    }


# ==================== CORE ENDPOINTS ====================


@app.get("/")
async def root():
    return {
        "message": "Linear vs Logistic Regression API",
        "endpoints": {
            "sample_data": "GET /sample/{type}?n_samples=100&noise=0.1&seed=42",
            "fit": "POST /fit - fit model(s) on your own data, get metrics + plot data back",
            "compare": "POST /compare - fit both models and get a side-by-side writeup",
            "predict": "POST /predict - predict a single x value from known coefficients",
            "batch_predict": "POST /batch_predict - predict many x values at once",
            "chart_demo": "GET /chart/demo - PNG chart, no setup needed",
            "chart": "POST /chart - PNG chart of your own data + fitted model(s)",
            "theory": "GET /theory",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/sample/{kind}")
async def get_sample_data(
    kind: Literal["linear", "logistic", "nonlinear"],
    n_samples: int = 100,
    noise: float = 0.1,
    seed: int = 42,
):
    """Get generated sample data for testing the other endpoints."""
    return {"data": generate_sample_data(kind, n_samples, noise, seed)}


@app.post("/fit")
async def fit_regression(request: RegressionRequest):
    """Fit regression model(s) and return metrics + visualization data."""
    results = run_fit(request)
    return {k: {kk: vv for kk, vv in v.items() if kk != "_model"} for k, v in results.items()}


@app.post("/compare")
async def compare_models(request: RegressionRequest):
    """Detailed side-by-side comparison between Linear and Logistic Regression."""
    if request.regression_type != "both":
        raise HTTPException(status_code=400, detail="Use regression_type='both' for comparison")

    results = run_fit(request)
    linear, logistic = results["linear"], results["logistic"]

    comparison = {
        "comparison_summary": {
            "linear_regression": {
                "best_for": "Continuous outcomes (price, temperature, sales)",
                "output_range": "(-\u221e, +\u221e)",
                "assumption": "Linear relationship between features and target",
                "loss_function": "Mean Squared Error (MSE)",
                "key_metric": f"R\u00b2 = {linear['metrics']['r2_score']:.4f}",
            },
            "logistic_regression": {
                "best_for": "Binary classification (yes/no, spam/not spam, disease/no disease)",
                "output_range": "[0, 1] (probability)",
                "assumption": "Log-linear relationship between features and log-odds",
                "loss_function": "Log Loss (Cross-Entropy)",
                "key_metric": f"Accuracy = {logistic['metrics']['accuracy']:.4f}",
            },
        },
        "when_to_use": {
            "choose_linear": [
                "Predicting continuous numerical values",
                "Understanding feature impact on magnitude",
                "Trend analysis and forecasting",
                "Relationship strength measurement (R\u00b2)",
            ],
            "choose_logistic": [
                "Binary classification problems",
                "Probability estimation of events",
                "Medical diagnosis (disease presence)",
                "Marketing (conversion probability)",
                "Risk assessment (default probability)",
            ],
        },
        "mathematical_difference": {
            "linear": "y = \u03b2\u2080 + \u03b2\u2081x\u2081 + ... + \u03b2\u2099x\u2099 + \u03b5",
            "logistic": "P(y=1) = 1 / (1 + e^-(\u03b2\u2080 + \u03b2\u2081x\u2081 + ... + \u03b2\u2099x\u2099))",
            "key_difference": "Linear predicts values directly; logistic predicts probabilities via a sigmoid transform.",
        },
        "verdict": (
            "These aren't really in competition - they're scored on different "
            "scales because they solve different kinds of problems. R\u00b2 near 1.0 "
            "means linear regression's numeric predictions track the data well; "
            "accuracy near 1.0 means logistic regression's pass/fail calls are "
            "mostly right. Comparing the two numbers directly isn't meaningful."
        ),
    }

    clean_results = {k: {kk: vv for kk, vv in v.items() if kk != "_model"} for k, v in results.items()}
    return {**clean_results, **comparison}


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make a prediction from already-fitted model parameters."""
    return _predict_value(
        request.model_type,
        request.coefficients,
        request.intercept,
        request.x_value,
        request.polynomial_degree,
    )


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    """Predict many x values at once from already-fitted model parameters."""
    predictions = [
        _predict_value(
            request.model_type,
            request.coefficients,
            request.intercept,
            x,
            request.polynomial_degree,
        )
        for x in request.x_values
    ]
    return {"predictions": predictions}


@app.get("/theory")
async def get_theory():
    """Theoretical background on both regression types."""
    return {
        "linear_regression": {
            "definition": "Models the linear relationship between independent variables and a continuous dependent variable",
            "formula": "y = X\u03b2 + \u03b5",
            "assumptions": [
                "Linearity: relationship between X and y is linear",
                "Independence: observations are independent",
                "Homoscedasticity: constant variance of errors",
                "Normality: errors are normally distributed",
            ],
            "optimization": "Minimizes Sum of Squared Residuals (Least Squares)",
            "interpretation": "\u03b2 coefficient represents change in y for a 1-unit change in x",
        },
        "logistic_regression": {
            "definition": "Models the probability of a binary outcome using the logistic (sigmoid) function",
            "formula": "log(p/(1-p)) = X\u03b2  \u2192  p = 1/(1+e^(-X\u03b2))",
            "assumptions": [
                "Binary outcome variable",
                "Log-linear relationship (linear in log-odds)",
                "No multicollinearity among predictors",
                "Large sample size (preferably > 100 per category)",
            ],
            "optimization": "Maximizes Log-Likelihood (minimizes Log Loss)",
            "interpretation": "\u03b2 coefficient represents change in log-odds for a 1-unit change in x; exp(\u03b2) is the odds ratio",
        },
        "comparison_table": {
            "output_type": {"linear": "Continuous", "logistic": "Probability (0-1)"},
            "target_variable": {"linear": "Numerical", "logistic": "Categorical (binary)"},
            "function": {"linear": "Identity", "logistic": "Sigmoid"},
            "error_distribution": {"linear": "Gaussian", "logistic": "Binomial"},
            "optimization": {"linear": "Least Squares", "logistic": "Maximum Likelihood"},
        },
    }


# ==================== CHART ENDPOINTS ====================


def _render_chart(X, y_numeric, linear_result, y_class, logistic_result, titles):
    has_linear = linear_result is not None
    has_logistic = logistic_result is not None
    n_panels = int(has_linear) + int(has_logistic)
    if n_panels == 0:
        raise HTTPException(status_code=400, detail="Nothing to chart.")

    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    panel = 0

    if has_linear:
        ax = axes[panel]
        panel += 1
        ax.scatter(X, y_numeric, alpha=0.6, label="Actual data")
        ax.plot(
            linear_result["visualization"]["x_line"],
            linear_result["visualization"]["y_line"],
            color="orange",
            linewidth=2,
            label="Fitted line",
        )
        ax.set_title(f"Linear Regression (R\u00b2={linear_result['metrics']['r2_score']:.3f})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()

    if has_logistic:
        ax = axes[panel]
        ax.scatter(X, y_class, alpha=0.6, label="Actual label (0/1)")
        ax.plot(
            logistic_result["visualization"]["x_line"],
            logistic_result["visualization"]["probability_line"],
            color="green",
            linewidth=2,
            label="Predicted probability",
        )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Decision threshold")
        ax.set_title(f"Logistic Regression (Accuracy={logistic_result['metrics']['accuracy']:.3f})")
        ax.set_xlabel("x")
        ax.set_ylabel("probability")
        ax.legend()

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/chart/demo")
async def chart_demo(n_samples: int = 100, noise: float = 0.1, seed: int = 42):
    """Quick PNG demo - no request body needed. Generates sample data
    and shows both models' fits side by side."""
    request = RegressionRequest(
        data=[
            DataPoint(**d)
            for d in generate_sample_data("logistic", n_samples, noise, seed)
        ],
        regression_type="both",
        seed=seed,
    )
    results = run_fit(request)
    X = np.array([d.x for d in request.data])
    y = np.array([d.y for d in request.data])
    labels = np.array([d.label for d in request.data])
    return _render_chart(X, y, results.get("linear"), labels, results.get("logistic"), None)


@app.post("/chart")
async def chart(request: RegressionRequest):
    """PNG chart of your own data with the fitted model(s) overlaid.
    Same request body as /fit."""
    results = run_fit(request)
    X = np.array([d.x for d in request.data])
    y = np.array([d.y for d in request.data])
    labels = _derive_binary_labels(request.data, y) if "logistic" in results else None
    return _render_chart(X, y, results.get("linear"), labels, results.get("logistic"), None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
