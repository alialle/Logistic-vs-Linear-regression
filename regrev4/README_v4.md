# Advanced Regression Comparison API v4.0

A FastAPI application for regression analysis: 5 model types, cross-validation,
hyperparameter tuning, bootstrap confidence intervals, residual diagnostics,
and interactive visualizations — now with a fitted-model registry so
predictions always match training exactly.

## 🚀 Quick Start

```bash
pip install -r requirements_v4.txt
uvicorn regression_api_v4:app --reload
```

Open:
- `http://127.0.0.1:8000/docs` — Interactive API playground
- `http://127.0.0.1:8000/chart/demo` — Zero-config PNG demo

Run the test suite before deploying:
```bash
pytest test_regression_api_v4.py -v
```

## 🩹 Why this upgrade (bugs fixed vs v3)

- **`/predict` silently accepted `model_type: "all"`.** It used
  `Field(..., exclude={RegressionType.all})`, but `exclude` controls
  serialization, not validation — it never rejected anything. Now a real
  validator returns `422` for it.
- **Scalers and cross-validation leaked test data.** v3 fit the
  `StandardScaler`/etc. on the *entire* dataset before the train/test split,
  so test-set statistics quietly influenced training. v4 fits every transform
  (polynomial features, feature selection, scaling) inside an sklearn
  `Pipeline` that's only ever fit on the training split; k-fold CV clones and
  refits that whole pipeline per fold instead of reusing one global scaler.
- **`mutual_info_regression` had no `random_state`.** It's a randomized
  estimator, so `feature_selection: k_best_mutual` gave different feature
  rankings on identical requests. Fixed.
- **CORS allowed `*` origins together with credentials.** Browsers reject
  that combination anyway, and it's a bad default. v4 only turns on
  `allow_credentials` when you set an explicit `ALLOWED_ORIGINS` env var.
- **`/sample/multicollinear`** emitted an `x2` field that nothing in the API
  ever used as a second feature — the "multicollinearity" it demonstrated was
  fake, since every model only ever saw a single `x` column. Removed rather
  than left silently misleading.
- **Durbin-Watson could divide by zero** on a perfect fit (zero residual
  variance). Now guarded.
- **The docstring promised "comprehensive logging" and "rate limiting"** that
  didn't exist anywhere in the v3 code. Both are now real (see below).
- Migrated from Pydantic v1-style `validator`/`root_validator`/`Config` to
  native Pydantic v2 (`field_validator`, `ConfigDict`, `model_dump`) — the
  requirements file already pinned `pydantic>=2.0.0`, but the code was using
  the deprecated v1 compatibility shims.

## ✨ What's new / improved

- **A fitted-model registry.** Every `/fit`, `/advanced_fit`, and `/compare`
  call now returns a `model_id` per model. `POST /models/{model_id}/predict`
  and `.../batch_predict` replay your *exact* fitted pipeline — polynomial
  expansion, feature selection, and scaling included — instead of asking you
  to hand-reconstruct that math yourself (which is what the legacy
  `/predict`/`/batch_predict` endpoints still require, and only correctly
  handle the simple degree-N-no-interactions-no-selection case). Models
  expire after `MODEL_TTL_SECONDS` (default 3600s); `GET /models` lists what's
  still live, `DELETE /models/{id}` removes one early.
- **Feature selection actually works now.** `feature_selection` /
  `k_best` were accepted by the request schema in v3 but never wired into
  fitting. They're now a real `SelectKBest` pipeline step, with `k_best`
  clamped to whatever's actually available post-polynomial-expansion instead
  of raising a raw sklearn `ValueError`.
- **Logistic regression supports polynomial features and scaling**, matching
  the other four model types (v3 only supported them for the continuous
  models).
- **`/export` uses exact predictions from the model registry** instead of
  approximating each row's prediction by nearest-neighbor lookup against the
  chart's 200-point line.
- **Structured logging** — every request gets an ID, method, path, status,
  and duration logged; unhandled exceptions are logged server-side and return
  a generic 500 with the request ID (not a stack trace) to the client.
- **A real in-memory rate limiter** (sliding window, per client IP),
  configurable via `RATE_LIMIT_MAX_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS`.
  It's single-process only — swap in Redis or similar for multi-worker
  deployments.
- **`/health`** now reports uptime, cache size, registered-model count, and
  rate-limit config.

## ⚠️ Known limitations (unchanged from v3, now documented instead of implied)

- **Single feature only.** Every `DataPoint` has one `x`. `include_interactions`
  is accepted but has no effect, since interaction terms require ≥2 base
  features — it's reserved for if/when the API grows multi-feature support.
- **The rate limiter and model registry are in-process memory.** They reset
  on restart and don't coordinate across multiple worker processes; that's
  fine for a single `uvicorn` process, not for a horizontally-scaled
  deployment without further work.
- **Bootstrap CI + hyperparameter tuning together is slow.** Each bootstrap
  iteration refits `RidgeCV`/`LassoCV`/`GridSearchCV`, which does its own
  internal cross-validation — expect meaningfully more time per request the
  more iterations you request.

## 📡 Endpoints

### Data Generation
```
GET /sample/{linear|logistic|nonlinear|heteroscedastic|outliers}
    ?n_samples=100&noise=0.1&seed=42
```

### Model Fitting
```
POST /fit              — Basic fit with metrics + viz data (cached)
POST /advanced_fit     — Full diagnostics (residuals, bootstrap CI)
POST /compare          — All models side-by-side with auto-recommendation
POST /diagnostics      — Residual assumption testing
```

### Predictions
```
GET  /models                            — List live fitted models
POST /models/{model_id}/predict         — Exact prediction from a fitted model
POST /models/{model_id}/batch_predict
DELETE /models/{model_id}
POST /predict          — Legacy: manual coefficients (limited, see docstring)
POST /batch_predict    — Legacy batch version
```

### Visualization
```
GET  /chart/demo       — Zero-config PNG/SVG chart
POST /chart            — Custom chart from your data
```

### Utilities
```
GET  /theory           — Mathematical background on all models
POST /export           — Export to CSV/Excel/JSON with exact predictions
GET  /health           — Health check, cache/registry/rate-limit stats
WS   /ws/stream        — WebSocket for real-time processing
```

## 🔧 Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | *(unset → `*`, no credentials)* | Comma-separated CORS origins |
| `RATE_LIMIT_MAX_REQUESTS` | `120` | Requests allowed per window per client IP |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window size |
| `MODEL_TTL_SECONDS` | `3600` | How long a fitted model stays predictable via `/models/{id}` |
| `LOG_LEVEL` | `INFO` | Python logging level |

## 📋 Example: Advanced Fit → Predict via the Model Registry

```bash
curl -X POST http://127.0.0.1:8000/advanced_fit \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"x": 1.0, "y": 12.5}, {"x": 2.0, "y": 18.3}, {"x": 3.0, "y": 25.1},
      {"x": 4.0, "y": 31.2}, {"x": 5.0, "y": 38.7}, {"x": 6.0, "y": 45.9}
    ],
    "regression_type": "ridge",
    "polynomial_degree": 2,
    "scaler": "standard",
    "cross_validation": true,
    "cv_folds": 3
  }'
# → response includes ridge.model_id, e.g. "a1b2c3d4e5f6g7h8"

curl -X POST http://127.0.0.1:8000/models/a1b2c3d4e5f6g7h8/predict \
  -H "Content-Type: application/json" \
  -d '{"x_value": 7.5}'
```

## 🧪 Sample Data Types

| Type | Description | Use Case |
|------|-------------|----------|
| `linear` | Clean linear relationship | Baseline testing |
| `logistic` | Binary classification pattern | Classification testing |
| `nonlinear` | Quadratic relationship | Polynomial testing |
| `heteroscedastic` | Increasing variance | Diagnostic testing |
| `outliers` | With extreme values | Robustness testing |

## 🐳 Docker

```bash
docker build -t regression-api .
docker run -p 8000:8000 -e RATE_LIMIT_MAX_REQUESTS=300 regression-api
```
