# Linear vs Logistic Regression API

A FastAPI app that fits linear and/or logistic regression on either
generated sample data or your own data, and lets you compare metrics,
get raw plot data back as JSON, or render a PNG chart directly.

This merges two earlier prototypes: a data-driven API (custom points,
polynomial features, prediction endpoints) and a demo/chart app
(matplotlib PNGs, plain-English explanations). A few correctness bugs
from both were fixed along the way - see "What changed" below.

## Setup

```bash
pip install fastapi uvicorn scikit-learn matplotlib numpy
```

## Run it

```bash
uvicorn regression_api:app --reload
```

Then open:

| URL | What it shows |
|---|---|
| `http://127.0.0.1:8000/docs` | Interactive API playground |
| `http://127.0.0.1:8000/chart/demo` | PNG chart, works with zero setup |
| `http://127.0.0.1:8000/theory` | Background on both models |

## Endpoints

- `GET /sample/{linear\|logistic\|nonlinear}?n_samples=100&noise=0.1&seed=42`
  Generate demo data.
- `POST /fit` - Fit model(s) on data you supply. Returns coefficients,
  metrics, and everything needed to plot the fit yourself (x/y series,
  fitted line/curve).
- `POST /compare` - Same as `/fit` with `regression_type="both"`, plus
  a side-by-side write-up of what each model is for and why their
  scores aren't directly comparable.
- `POST /predict` / `POST /batch_predict` - Predict from
  already-known coefficients (one x value or many).
- `GET /chart/demo` - Zero-config PNG: generates sample data and plots
  both models' fits side by side.
- `POST /chart` - Same idea, but PNG of a fit on data you supply
  (same request body as `/fit`).
- `GET /theory` - Definitions, assumptions, and a comparison table for
  both models.

### Example: fit your own data

```bash
curl -X POST http://127.0.0.1:8000/fit \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"x": 1, "y": 12, "label": 0},
      {"x": 2, "y": 18, "label": 0},
      {"x": 3, "y": 25, "label": 0},
      {"x": 4, "y": 41, "label": 1},
      {"x": 5, "y": 48, "label": 1},
      {"x": 6, "y": 55, "label": 1}
    ],
    "regression_type": "both",
    "test_ratio": 0.34,
    "seed": 42
  }'
```

## Metrics explained

**Linear regression** (predicts a number):
- `r2_score` - closer to 1.0 is a better fit
- `mse` / `rmse` - lower is better

**Logistic regression** (predicts a category):
- `accuracy` - fraction of correct pass/fail calls
- `log_loss` - penalizes confident wrong predictions, lower is better
- `confusion_matrix` - true/false positives and negatives on the test set

## Why the two models aren't scored head-to-head

Linear and logistic regression solve different kinds of problems (a
number vs. a category), so R² and accuracy aren't on the same scale
and "which one is better" isn't a meaningful comparison in absolute
terms. `/compare`'s `verdict` field says this explicitly.

## What changed in the merge

- **Fixed a real train/test split bug**: the original data-driven API
  split indices without stratifying by class, so a small or imbalanced
  logistic dataset could hand the test set only one class and quietly
  produce a broken confusion matrix. The split now stratifies when
  possible.
- **Fixed a label/data alignment bug**: if only *some* input points
  had a `label`, the original code filtered to just those before
  building the target array, but still indexed it using train/test
  indices computed against the *full* dataset - silently
  misaligning X and y. Now it's explicit: either every point is
  labeled, or none are (median split is used instead), and a mismatch
  is rejected with a clear error instead of quietly corrupting the fit.
- **Removed the hardcoded global `np.random.seed(42)`**: sample data
  generation now takes an explicit `seed` parameter per call instead
  of mutating global numpy state (which made every request affect
  every other request's randomness).
- **`polynomial_degree` and `test_ratio` are validated** (degree 1-6,
  ratio strictly between 0 and 1) instead of allowing values that would
  crash the fit deep inside sklearn.
- **`batch_predict` takes a JSON body** instead of a list-of-floats
  query parameter, which is awkward to call correctly over HTTP.
- **Added the PNG chart endpoints** (`/chart/demo`, `/chart`) from the
  second prototype, generalized to work with custom data and either
  or both model types, not just the fixed demo dataset.
- **Added `log_loss` and `/health`.**
