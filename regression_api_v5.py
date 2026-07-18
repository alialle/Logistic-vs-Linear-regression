"""
Linear vs Logistic Regression API v5.0
---------------------------------------
Builds on v4 with:
- Multi-feature datasets (DataPoint accepts arbitrary x1, x2, x3... columns)
- Pluggable cache backend (in-memory LRU or Redis) behind one async interface
- Drift detection (KS test, Wasserstein distance, PSI) per feature
- A/B testing between two registered models (paired bootstrap CI)
- An async job queue for long-running fits, pollable via /jobs/{job_id}

Everything from v4 (fitted-model registry, leak-free pipelines, rate
limiting, structured logging) is retained. See README_v5.md for the full
changelog and migration notes (some v4 request/response shapes changed to
accommodate multiple features).

Run: uvicorn regression_api_v5:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import io
import os
import time
import uuid
import json
import math
import hashlib
import logging
import asyncio
from collections import deque, OrderedDict
from functools import partial
from typing import List, Literal, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum

from sklearn.base import clone
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression,
    RidgeCV, LassoCV, ElasticNetCV
)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, explained_variance_score,
    accuracy_score, confusion_matrix, log_loss, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, mean_absolute_percentage_error
)

# ==================== LOGGING ====================

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("regression_api")

START_TIME = time.time()

# ==================== ENUMS & CONFIG ====================

class RegressionType(str, Enum):
    linear = "linear"
    ridge = "ridge"
    lasso = "lasso"
    elasticnet = "elasticnet"
    logistic = "logistic"
    all = "all"

class ScalerType(str, Enum):
    none = "none"
    standard = "standard"
    minmax = "minmax"
    robust = "robust"

class FeatureSelection(str, Enum):
    none = "none"
    k_best_f = "k_best_f"
    k_best_mutual = "k_best_mutual"

class ChartFormat(str, Enum):
    png = "png"
    svg = "svg"

class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"

RESERVED_FIELDS = {"y", "label", "group"}

# ==================== DATA MODELS ====================

class DataPoint(BaseModel):
    """Any field besides y/label/group is treated as a feature column, e.g.
    {"x1": 2.0, "x2": 7.5, "y": 12.0}. All points in one request must use the
    same set of feature keys - extract_features() validates that."""
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"example": {"x1": 5.2, "x2": 1.0, "y": 15.3, "label": 1, "group": "A"}},
    )

    y: float = Field(..., description="Target value (continuous or probability)")
    label: Optional[int] = Field(None, ge=0, le=1, description="Binary label for logistic (0 or 1)")
    group: Optional[str] = Field(None, description="Optional group/category for stratification")


def extract_features(data: List[DataPoint]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    """Turn a list of DataPoints into (X, y, labels_or_None, feature_column_names).
    Feature columns are whatever extra keys were supplied, sorted for a
    deterministic column order across requests."""
    rows = [d.model_dump() for d in data]
    df = pd.DataFrame(rows)

    feature_cols = sorted(c for c in df.columns if c not in RESERVED_FIELDS)
    if not feature_cols:
        raise HTTPException(
            status_code=400,
            detail="No feature columns found. Include at least one feature besides y/label/group, e.g. {\"x1\": 2.0, \"y\": 5.0}.",
        )

    missing = df[feature_cols].isna().any()
    if missing.any():
        bad_cols = missing[missing].index.tolist()
        raise HTTPException(
            status_code=400,
            detail=f"Inconsistent feature columns across points - some rows are missing: {bad_cols}. "
                   "Every point in a request must have the same feature keys.",
        )

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise HTTPException(status_code=400, detail=f"Feature columns must be numeric: {non_numeric}")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    labels = None
    if "label" in df.columns and df["label"].notna().all():
        labels = df["label"].to_numpy(dtype=int)
    return X, y, labels, feature_cols


class RegressionRequest(BaseModel):
    data: List[DataPoint] = Field(..., min_length=5, description="Dataset (minimum 5 points)")
    regression_type: RegressionType = Field(default=RegressionType.all)
    polynomial_degree: int = Field(default=1, ge=1, le=8, description="Polynomial feature degree")
    include_interactions: bool = Field(default=False, description="Include interaction terms between features")
    scaler: ScalerType = Field(default=ScalerType.standard)
    feature_selection: FeatureSelection = Field(default=FeatureSelection.none)
    k_best: int = Field(default=5, ge=1, le=20, description="Number of features to select")
    test_ratio: float = Field(default=0.2, gt=0.0, lt=0.5)
    seed: int = Field(default=42)
    cross_validation: bool = Field(default=False, description="Enable k-fold cross-validation")
    cv_folds: int = Field(default=5, ge=2, le=10)
    hyperparameter_tuning: bool = Field(default=False, description="Grid search for optimal params")
    alpha_values: Optional[List[float]] = Field(default=None, description="Custom alpha values for tuning")

    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
        if len(v) < 5:
            raise ValueError("Need at least 5 data points")
        return v


class AdvancedFitRequest(RegressionRequest):
    include_residuals: bool = Field(default=True, description="Include residual analysis")
    confidence_interval: float = Field(default=0.95, ge=0.8, le=0.99)
    bootstrap_iterations: int = Field(default=0, ge=0, le=1000, description="Bootstrap CI iterations (0=off)")


class PredictionRequest(BaseModel):
    """Legacy manual-coefficient prediction, single feature only. Prefer
    POST /models/{model_id}/predict, which replays your exact fitted
    pipeline (any number of features, polynomial expansion, feature
    selection, and scaling included)."""
    model_type: RegressionType
    coefficients: List[float]
    intercept: float
    x_value: float
    polynomial_degree: int = Field(default=1, ge=1, le=8)
    scaler_mean: Optional[float] = None
    scaler_std: Optional[float] = None

    @field_validator("model_type")
    @classmethod
    def reject_all(cls, v):
        if v == RegressionType.all:
            raise ValueError("model_type must be a concrete model, not 'all'")
        return v


class BatchPredictionRequest(BaseModel):
    model_type: RegressionType
    coefficients: List[float]
    intercept: float
    x_values: List[float]
    polynomial_degree: int = Field(default=1, ge=1, le=8)
    scaler_mean: Optional[float] = None
    scaler_std: Optional[float] = None

    @field_validator("model_type")
    @classmethod
    def reject_all(cls, v):
        if v == RegressionType.all:
            raise ValueError("model_type must be a concrete model, not 'all'")
        return v


class ModelPredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature name -> value, matching the columns the model was trained on")


class ModelBatchPredictRequest(BaseModel):
    rows: List[Dict[str, float]] = Field(..., min_length=1, max_length=100_000)


class ChartRequest(BaseModel):
    data: List[DataPoint]
    regression_type: RegressionType = Field(default=RegressionType.all)
    polynomial_degree: int = Field(default=1, ge=1, le=8)
    chart_format: ChartFormat = Field(default=ChartFormat.png)
    theme: Literal["default", "dark", "seaborn", "ggplot"] = Field(default="default")
    width: int = Field(default=1200, ge=400, le=3000)
    height: int = Field(default=600, ge=300, le=2000)
    dpi: int = Field(default=150, ge=72, le=300)
    seed: int = 42


class ExportRequest(BaseModel):
    data: List[DataPoint]
    format: Literal["json", "csv", "excel"] = Field(default="csv")
    include_predictions: bool = Field(default=True)
    regression_type: RegressionType = Field(default=RegressionType.all)
    polynomial_degree: int = Field(default=1, ge=1, le=8)
    seed: int = 42


class DriftRequest(BaseModel):
    reference_data: List[DataPoint] = Field(..., min_length=2)
    current_data: List[DataPoint] = Field(..., min_length=2)
    psi_bins: int = Field(default=10, ge=2, le=50)


class ABTestRequest(BaseModel):
    model_id_a: str
    model_id_b: str
    test_data: List[DataPoint] = Field(..., min_length=2)
    bootstrap_iterations: int = Field(default=1000, ge=100, le=5000)
    confidence: float = Field(default=0.95, ge=0.8, le=0.99)
    seed: int = 42


# ==================== CACHE (pluggable backend) ====================

class CacheBackend:
    """Async cache interface so the storage layer can be swapped without
    touching endpoint code."""
    async def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    async def set(self, key: str, value: Any, ttl: int = 600) -> None:
        raise NotImplementedError

    async def clear(self) -> None:
        raise NotImplementedError

    async def size(self) -> int:
        raise NotImplementedError


class MemoryCache(CacheBackend):
    """In-memory cache with TTL + LRU eviction once max_size is hit. No
    external dependency; state is lost on restart and isn't shared across
    processes (fine for a single uvicorn worker, not for a cluster)."""

    def __init__(self, ttl_seconds: int = 600, max_size: int = 1000):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: "OrderedDict[str, Tuple[Any, float]]" = OrderedDict()

    async def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, timestamp = entry
        if time.time() - timestamp >= self._ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    async def set(self, key: str, value: Any, ttl: int = 600) -> None:
        self._store[key] = (value, time.time())
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)  # evict least-recently-used

    async def clear(self) -> None:
        self._store.clear()

    async def size(self) -> int:
        return len(self._store)

    def sweep(self) -> None:
        now = time.time()
        expired = [k for k, (_, ts) in self._store.items() if now - ts >= self._ttl]
        for k in expired:
            del self._store[k]


class RedisCache(CacheBackend):
    """Redis-backed cache for multi-process/multi-instance deployments.
    Requires `pip install redis` (not a default dependency of this project -
    only import it if REDIS_URL / CACHE_BACKEND=redis is actually configured,
    so a plain in-memory deployment doesn't need the extra package)."""

    def __init__(self, redis_url: str):
        try:
            import redis.asyncio as redis_asyncio
        except ImportError as e:
            raise RuntimeError(
                "CACHE_BACKEND=redis requires the `redis` package: pip install redis"
            ) from e
        self._client = redis_asyncio.from_url(redis_url, decode_responses=True)

    async def get(self, key: str) -> Optional[Any]:
        raw = await self._client.get(key)
        return json.loads(raw) if raw is not None else None

    async def set(self, key: str, value: Any, ttl: int = 600) -> None:
        await self._client.set(key, json.dumps(value, default=str), ex=ttl)

    async def clear(self) -> None:
        # Deliberately not FLUSHDB (that would nuke unrelated keys sharing the
        # same Redis instance) - callers that need a full wipe should manage
        # a dedicated DB/keyspace instead.
        raise NotImplementedError("RedisCache.clear() is intentionally unimplemented - use a dedicated Redis DB and FLUSHDB yourself if needed.")

    async def size(self) -> int:
        return await self._client.dbsize()


def _build_cache() -> CacheBackend:
    backend = os.environ.get("CACHE_BACKEND", "memory").lower()
    ttl = int(os.environ.get("CACHE_TTL_SECONDS", "600"))
    if backend == "redis":
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        logger.info("Using RedisCache at %s", redis_url)
        return RedisCache(redis_url)
    logger.info("Using in-memory MemoryCache (ttl=%ss)", ttl)
    return MemoryCache(ttl_seconds=ttl, max_size=int(os.environ.get("CACHE_MAX_SIZE", "1000")))


result_cache: CacheBackend = _build_cache()


def _cache_key(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


# ==================== FITTED MODEL REGISTRY ====================

class ModelRegistry:
    """Holds fitted sklearn Pipelines (polynomial features + feature
    selection + scaling + estimator, all fit together) so predictions can
    replay the *exact* training-time transforms. Entries expire after a TTL.

    Deliberately in-process only: sklearn Pipelines aren't safely
    JSON-serializable, and unpickling arbitrary bytes from a shared store
    like Redis is a real security risk, so this isn't pushed behind the
    pluggable CacheBackend above."""

    def __init__(self, ttl_seconds: int = 3600):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl_seconds

    def register(self, pipeline: Pipeline, model_type: str, feature_cols: List[str],
                 feature_names: List[str], degree: int, interactions: bool) -> str:
        model_id = uuid.uuid4().hex[:16]
        self._store[model_id] = {
            "pipeline": pipeline,
            "model_type": model_type,
            "feature_cols": feature_cols,       # raw input column names, in training order
            "feature_names": feature_names,     # post-transform names (for display)
            "polynomial_degree": degree,
            "include_interactions": interactions,
            "created_at": time.time(),
        }
        return model_id

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        entry = self._store.get(model_id)
        if entry is None:
            return None
        if time.time() - entry["created_at"] > self._ttl:
            del self._store[model_id]
            return None
        return entry

    def delete(self, model_id: str) -> bool:
        return self._store.pop(model_id, None) is not None

    def list(self) -> List[Dict[str, Any]]:
        self.sweep()
        return [
            {
                "model_id": mid,
                "model_type": e["model_type"],
                "feature_cols": e["feature_cols"],
                "polynomial_degree": e["polynomial_degree"],
                "include_interactions": e["include_interactions"],
                "age_seconds": round(time.time() - e["created_at"], 1),
                "expires_in_seconds": round(self._ttl - (time.time() - e["created_at"]), 1),
            }
            for mid, e in self._store.items()
        ]

    def sweep(self):
        now = time.time()
        expired = [mid for mid, e in self._store.items() if now - e["created_at"] > self._ttl]
        for mid in expired:
            del self._store[mid]

    def __len__(self):
        return len(self._store)


model_registry = ModelRegistry(ttl_seconds=int(os.environ.get("MODEL_TTL_SECONDS", "3600")))


# ==================== ASYNC JOB QUEUE ====================

class JobQueue:
    """Runs long-running fits (large bootstrap iteration counts, big
    datasets) off the event loop via a thread pool, with bounded
    concurrency, so one huge request can't starve every other request.
    Poll status/result via GET /jobs/{job_id}. In-process only - jobs are
    lost on restart, same caveat as the model registry and MemoryCache."""

    def __init__(self, max_concurrency: int = 4, ttl_seconds: int = 3600):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._sem = asyncio.Semaphore(max_concurrency)
        self._ttl = ttl_seconds

    async def submit(self, func, *args, **kwargs) -> str:
        job_id = uuid.uuid4().hex[:16]
        self._jobs[job_id] = {
            "status": JobStatus.pending, "result": None, "error": None,
            "created_at": time.time(), "finished_at": None,
        }
        asyncio.create_task(self._run(job_id, func, *args, **kwargs))
        return job_id

    async def _run(self, job_id: str, func, *args, **kwargs):
        async with self._sem:
            self._jobs[job_id]["status"] = JobStatus.running
            loop = asyncio.get_running_loop()
            try:
                result = await loop.run_in_executor(None, partial(func, *args, **kwargs))
                self._jobs[job_id]["status"] = JobStatus.done
                self._jobs[job_id]["result"] = result
            except HTTPException as e:
                self._jobs[job_id]["status"] = JobStatus.failed
                self._jobs[job_id]["error"] = e.detail
            except Exception as e:
                logger.exception("job_failed job_id=%s", job_id)
                self._jobs[job_id]["status"] = JobStatus.failed
                self._jobs[job_id]["error"] = str(e)
            finally:
                self._jobs[job_id]["finished_at"] = time.time()

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        self.sweep()
        return self._jobs.get(job_id)

    def sweep(self):
        now = time.time()
        expired = [jid for jid, j in self._jobs.items() if now - j["created_at"] > self._ttl]
        for jid in expired:
            del self._jobs[jid]

    def __len__(self):
        return len(self._jobs)


job_queue = JobQueue(
    max_concurrency=int(os.environ.get("JOB_MAX_CONCURRENCY", "4")),
    ttl_seconds=int(os.environ.get("JOB_TTL_SECONDS", "3600")),
)


# ==================== RATE LIMITER ====================

class RateLimiter:
    """Sliding-window limiter, per client IP, in-memory. Single-process only
    - swap in Redis for multi-worker deployments."""

    def __init__(self, max_requests: int = 120, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: Dict[str, deque] = {}

    def check(self, client_key: str) -> Tuple[bool, float]:
        now = time.time()
        window_start = now - self.window_seconds
        hits = self._hits.setdefault(client_key, deque())
        while hits and hits[0] < window_start:
            hits.popleft()
        if len(hits) >= self.max_requests:
            retry_after = hits[0] + self.window_seconds - now
            return False, max(retry_after, 0.0)
        hits.append(now)
        return True, 0.0


rate_limiter = RateLimiter(
    max_requests=int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "120")),
    window_seconds=float(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60")),
)
RATE_LIMIT_EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}

# ==================== JSON SAFETY ====================
# Strict JSON has no representation for Infinity/NaN, and Starlette's
# JSONResponse enforces that (allow_nan=False) - so any literal `inf`/`nan`
# anywhere in a response body crashes the whole request with a 500 instead
# of just that one field being unusable. This can happen in more than one
# place (log_loss on a near-perfectly-separable test split, exp() overflow
# on a large logistic coefficient, zero-variance edge cases in R²/explained
# variance, degenerate bootstrap resamples, ...) so instead of chasing each
# individual source, every response is sanitized in one place: any
# non-finite float becomes `null` rather than crashing the request.

def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return super().render(_json_safe(content))


# ==================== APP SETUP ====================

app = FastAPI(
    title="Advanced Regression Comparison API",
    version="5.0",
    description="Multi-feature regression analysis with cross-validation, hyperparameter tuning, "
                 "a fitted-model registry, drift detection, A/B testing, and an async job queue.",
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=SafeJSONResponse,
)

_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
if _origins_env:
    _allowed_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
    _allow_credentials = True
else:
    _allowed_origins = ["*"]
    _allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_and_rate_limit_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()

    if request.url.path not in RATE_LIMIT_EXEMPT_PATHS:
        client_key = request.client.host if request.client else "unknown"
        allowed, retry_after = rate_limiter.check(client_key)
        if not allowed:
            logger.warning("rate_limited request_id=%s client=%s path=%s", request_id, client_key, request.url.path)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down.", "retry_after_seconds": round(retry_after, 1)},
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception("unhandled_error request_id=%s method=%s path=%s duration_ms=%.1f",
                          request_id, request.method, request.url.path, duration_ms)
        return JSONResponse(status_code=500, content={"detail": "Internal server error.", "request_id": request_id})

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info("request_id=%s method=%s path=%s status=%d duration_ms=%.1f",
                request_id, request.method, request.url.path, response.status_code, duration_ms)
    return response


# ==================== UTILITY FUNCTIONS ====================

def generate_sample_data(kind: str, n_samples: int = 100, noise: float = 0.1, seed: int = 42) -> List[Dict]:
    """Generate demo datasets with realistic patterns."""
    rng = np.random.default_rng(seed)

    if kind == "linear":
        x = rng.uniform(0, 10, n_samples)
        y = 2.5 * x + 1.0 + rng.normal(0, noise * 10, n_samples)
        return [{"x": float(xi), "y": float(yi)} for xi, yi in zip(x, y)]

    elif kind == "logistic":
        x = rng.uniform(0, 10, n_samples)
        true_w, true_b = 1.4, -7.0
        score = true_w * x + true_b + rng.normal(0, noise * 10, n_samples)
        probabilities = 1 / (1 + np.exp(-score))
        labels = (probabilities > 0.5).astype(int)
        return [{"x": float(xi), "y": float(pi), "label": int(li)} for xi, pi, li in zip(x, probabilities, labels)]

    elif kind == "nonlinear":
        x = rng.uniform(0, 10, n_samples)
        y = 0.5 * x**2 - 3 * x + 5 + rng.normal(0, noise * 5, n_samples)
        return [{"x": float(xi), "y": float(yi)} for xi, yi in zip(x, y)]

    elif kind == "heteroscedastic":
        x = rng.uniform(0, 10, n_samples)
        y = 2.0 * x + 5.0 + rng.normal(0, noise * x * 3, n_samples)
        return [{"x": float(xi), "y": float(yi)} for xi, yi in zip(x, y)]

    elif kind == "outliers":
        x = rng.uniform(0, 10, n_samples)
        y = 2.0 * x + 5.0 + rng.normal(0, noise * 5, n_samples)
        outlier_idx = rng.choice(n_samples, size=max(1, n_samples // 10), replace=False)
        y[outlier_idx] += rng.choice([-1, 1], size=len(outlier_idx)) * rng.uniform(20, 40, size=len(outlier_idx))
        return [{"x": float(xi), "y": float(yi)} for xi, yi in zip(x, y)]

    elif kind == "multifeature":
        x1 = rng.uniform(0, 10, n_samples)
        x2 = rng.uniform(-5, 5, n_samples)
        y = 3.0 * x1 - 2.0 * x2 + 5.0 + rng.normal(0, noise * 10, n_samples)
        return [{"x1": float(a), "x2": float(b), "y": float(c)} for a, b, c in zip(x1, x2, y)]

    else:
        raise ValueError(f"Unknown sample data kind: {kind}")


def _make_scaler(scaler_type: ScalerType):
    if scaler_type == ScalerType.standard:
        return StandardScaler()
    if scaler_type == ScalerType.minmax:
        return MinMaxScaler()
    if scaler_type == ScalerType.robust:
        return RobustScaler()
    return None


def _make_selector(feature_selection: FeatureSelection, k_best: int, seed: int):
    if feature_selection == FeatureSelection.none:
        return None
    if feature_selection == FeatureSelection.k_best_f:
        return SelectKBest(score_func=f_regression, k=k_best)
    if feature_selection == FeatureSelection.k_best_mutual:
        score_func = partial(mutual_info_regression, random_state=seed)
        return SelectKBest(score_func=score_func, k=k_best)
    return None


def _build_pipeline_steps(
    model_type: str, n_raw_features: int, degree: int, interactions: bool,
    scaler_type: ScalerType, feature_selection: FeatureSelection, k_best: int, seed: int,
    hyperparam_tuning: bool = False, alpha_values: Optional[List[float]] = None, cv_folds: int = 5,
) -> List[Tuple[str, Any]]:
    """Build the list of (name, transformer/estimator) steps for a Pipeline.
    Works for any number of raw features - PolynomialFeatures/SelectKBest/
    scalers are all generic over column count already."""
    steps: List[Tuple[str, Any]] = []
    n_features_after_poly = n_raw_features

    if degree > 1 or interactions:
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=interactions)
        steps.append(("poly", poly))
        dummy = np.vstack([np.zeros(n_raw_features), np.ones(n_raw_features)])
        n_features_after_poly = poly.fit(dummy).n_output_features_

    if feature_selection != FeatureSelection.none:
        k = min(k_best, n_features_after_poly)
        steps.append(("select", _make_selector(feature_selection, k, seed)))

    scaler = _make_scaler(scaler_type)
    if scaler is not None:
        steps.append(("scaler", scaler))

    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "ridge":
        if hyperparam_tuning:
            alphas = alpha_values or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            model = RidgeCV(alphas=alphas, cv=cv_folds)
        else:
            model = Ridge(alpha=1.0)
    elif model_type == "lasso":
        if hyperparam_tuning:
            alphas = alpha_values or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            model = LassoCV(alphas=alphas, cv=cv_folds, max_iter=5000)
        else:
            model = Lasso(alpha=1.0)
    elif model_type == "elasticnet":
        if hyperparam_tuning:
            model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], cv=cv_folds, max_iter=5000)
        else:
            model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    elif model_type == "logistic":
        if hyperparam_tuning:
            param_grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], "penalty": ["l1", "l2"]}
            model = GridSearchCV(
                LogisticRegression(max_iter=2000, solver="saga"),
                param_grid, cv=cv_folds, scoring="roc_auc",
            )
        else:
            model = LogisticRegression(max_iter=2000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    steps.append(("model", model))
    return steps


def _feature_names_from_pipeline(pipeline: Pipeline, base_names: List[str]) -> List[str]:
    names = list(base_names)
    if "poly" in pipeline.named_steps:
        names = list(pipeline.named_steps["poly"].get_feature_names_out(base_names))
    if "select" in pipeline.named_steps:
        names = list(pipeline.named_steps["select"].get_feature_names_out(names))
    return names


def _final_estimator(pipeline: Pipeline):
    model = pipeline.named_steps["model"]
    if hasattr(model, "best_estimator_"):
        return model.best_estimator_
    return model


def _derive_binary_labels(data: List[DataPoint], y: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
    """Build aligned label array with explicit validation. `labels` (from
    extract_features) is passed through if every row was labelled."""
    n_labelled = sum(1 for d in data if d.label is not None)

    if labels is not None and n_labelled == len(data):
        if not set(labels.tolist()).issubset({0, 1}):
            raise HTTPException(status_code=400, detail="Labels must be 0 or 1")
        return labels

    if n_labelled == 0:
        return (y > np.median(y)).astype(int)

    raise HTTPException(
        status_code=400,
        detail=(
            f"{n_labelled} of {len(data)} points have a label and the rest don't. "
            "Either label every point or label none (median split used automatically)."
        ),
    )


def _compute_residual_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    residuals = y_true - y_pred
    std = np.std(residuals)
    std_residuals = residuals / std if std > 0 else residuals
    ss = np.sum(residuals ** 2)

    return {
        "residuals": [float(r) for r in residuals],
        "standardized_residuals": [float(r) for r in std_residuals],
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(std),
        "max_residual": float(np.max(np.abs(residuals))) if len(residuals) else None,
        "durbin_watson": float(np.sum(np.diff(residuals) ** 2) / ss) if len(residuals) > 1 and ss > 0 else None,
        "heteroscedasticity_hint": "Check residual plot for funnel shape",
    }


def _bootstrap_confidence_intervals(
    fitted_pipeline: Pipeline, X_train_raw: np.ndarray, y_train: np.ndarray,
    n_iterations: int, confidence: float, seed: int,
) -> Dict:
    """Bootstrap CIs for the final estimator's coefficients. Resamples the
    training split only and refits a clone of the already-configured
    pipeline, so the CI reflects the same model spec that was evaluated."""
    rng = np.random.default_rng(seed)
    n_samples = len(y_train)
    coef_samples = []

    for _ in range(n_iterations):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot, y_boot = X_train_raw[idx], y_train[idx]
        try:
            pipe = clone(fitted_pipeline)
            pipe.fit(X_boot, y_boot)
            coef_samples.append(_final_estimator(pipe).coef_)
        except Exception:
            continue

    if not coef_samples:
        return {"confidence_level": confidence, "iterations": 0, "coefficient_intervals": [],
                "note": "All bootstrap resamples failed - try fewer iterations or a lower polynomial degree."}

    coef_samples = np.array(coef_samples)
    alpha = 1 - confidence
    lower = np.percentile(coef_samples, alpha / 2 * 100, axis=0)
    upper = np.percentile(coef_samples, (1 - alpha / 2) * 100, axis=0)

    return {
        "confidence_level": confidence,
        "iterations": len(coef_samples),
        "coefficient_intervals": [
            {"lower": float(lo), "upper": float(up), "contains_zero": bool(lo <= 0 <= up)}
            for lo, up in zip(lower, upper)
        ],
    }


# ==================== MODEL FITTING ====================

def _build_equation(coefs, intercept, feature_names) -> str:
    terms = [f"{c:.4f}*{n}" for c, n in zip(coefs, feature_names)]
    return f"y = {' + '.join(terms)} + {intercept:.4f}"


def _visualization_payload(X: np.ndarray, y: np.ndarray, pipeline: Pipeline, feature_cols: List[str]) -> Dict:
    """Single-feature datasets get a smooth fit line (200 points spanning the
    x range). Multi-feature datasets can't be drawn as a 2D line, so we
    return predicted-vs-actual pairs instead - still plottable, still
    meaningful, doesn't pretend a single line summarizes a multi-dimensional
    fit."""
    if X.shape[1] == 1:
        x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_line = pipeline.predict(x_line)
        return {
            "kind": "line",
            "feature": feature_cols[0],
            "x_original": [float(v) for v in X.flatten()],
            "y_original": [float(v) for v in y],
            "x_line": [float(v[0]) for v in x_line],
            "y_line": [float(v) for v in y_line],
        }
    y_pred_all = pipeline.predict(X)
    return {
        "kind": "predicted_vs_actual",
        "features": feature_cols,
        "y_true": [float(v) for v in y],
        "y_predicted": [float(v) for v in y_pred_all],
    }


def _fit_model(
    model_type: str, X: np.ndarray, y: np.ndarray, feature_cols: List[str],
    test_ratio: float, seed: int, degree: int = 1, interactions: bool = False,
    scaler_type: ScalerType = ScalerType.standard, feature_selection: FeatureSelection = FeatureSelection.none,
    k_best: int = 5, cv: bool = False, cv_folds: int = 5, hyperparam_tuning: bool = False,
    alpha_values: Optional[List[float]] = None, bootstrap_iters: int = 0, confidence: float = 0.95,
    include_residuals: bool = True,
) -> Dict:
    """Fit a single regression model with full diagnostics, via one Pipeline
    that owns every transform (poly/select/scale all fit on the training
    split only - no leakage from held-out rows)."""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

    steps = _build_pipeline_steps(
        model_type, X.shape[1], degree, interactions, scaler_type, feature_selection, k_best,
        seed, hyperparam_tuning, alpha_values, cv_folds,
    )
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)

    y_pred_test = pipeline.predict(X_test)
    estimator = _final_estimator(pipeline)
    feature_names = _feature_names_from_pipeline(pipeline, feature_cols)

    cv_scores = None
    if cv and not hyperparam_tuning:
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        cv_scores = cross_val_score(clone(pipeline), X, y, cv=kfold, scoring="r2")

    metrics = {
        "r2_score": float(r2_score(y_test, y_pred_test)),
        "mse": float(mean_squared_error(y_test, y_pred_test)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "mae": float(mean_absolute_error(y_test, y_pred_test)),
        "mape": float(mean_absolute_percentage_error(y_test, y_pred_test)) if np.all(y_test != 0) else None,
        "explained_variance": float(explained_variance_score(y_test, y_pred_test)),
    }
    if cv_scores is not None:
        metrics["cv_r2_mean"] = float(np.mean(cv_scores))
        metrics["cv_r2_std"] = float(np.std(cv_scores))
    if hasattr(estimator, "alpha_"):
        metrics["optimal_alpha"] = float(estimator.alpha_)
    if hasattr(estimator, "l1_ratio_"):
        metrics["optimal_l1_ratio"] = float(estimator.l1_ratio_)

    coefs = np.atleast_1d(estimator.coef_)
    feature_importance = sorted(
        ({"feature": name, "coefficient": float(c), "abs_coefficient": float(abs(c))}
         for name, c in zip(feature_names, coefs)),
        key=lambda item: item["abs_coefficient"], reverse=True,
    )

    model_id = model_registry.register(pipeline, model_type, feature_cols, feature_names, degree, interactions)

    result = {
        "type": model_type.title() + " Regression",
        "model_id": model_id,
        "equation": _build_equation(coefs, float(estimator.intercept_), feature_names),
        "coefficients": [float(c) for c in coefs],
        "intercept": float(estimator.intercept_),
        "feature_names": feature_names,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "visualization": _visualization_payload(X, y, pipeline, feature_cols),
        "hyperparameters": {
            "polynomial_degree": degree,
            "interactions": interactions,
            "scaler": scaler_type.value,
            "feature_selection": feature_selection.value,
            "cross_validation": cv,
            "hyperparameter_tuning": hyperparam_tuning,
        },
    }

    if include_residuals:
        result["residuals"] = _compute_residual_stats(y_test, y_pred_test)

    if bootstrap_iters > 0:
        result["bootstrap_ci"] = _bootstrap_confidence_intervals(
            pipeline, X_train, y_train, bootstrap_iters, confidence, seed
        )

    return result


def _fit_logistic_model(
    X: np.ndarray, y_log: np.ndarray, feature_cols: List[str], test_ratio: float, seed: int,
    degree: int = 1, interactions: bool = False, scaler_type: ScalerType = ScalerType.standard,
    cv: bool = False, cv_folds: int = 5, hyperparam_tuning: bool = False,
) -> Dict:
    class_counts = np.bincount(y_log)
    if len(class_counts) < 2 or class_counts.min() == 0:
        raise HTTPException(status_code=400, detail="Logistic regression needs both classes (0 and 1) present.")

    can_stratify = class_counts.min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=test_ratio, random_state=seed, stratify=y_log if can_stratify else None,
    )

    steps = _build_pipeline_steps(
        "logistic", X.shape[1], degree, interactions, scaler_type, FeatureSelection.none, 0,
        seed, hyperparam_tuning, None, cv_folds,
    )
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)

    y_pred_test = pipeline.predict(X_test)
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    estimator = _final_estimator(pipeline)
    feature_names = _feature_names_from_pipeline(pipeline, feature_cols)

    cv_scores = None
    if cv and not hyperparam_tuning:
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        cv_scores = cross_val_score(clone(pipeline), X, y_log, cv=kfold, scoring="roc_auc")

    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    coefs = np.atleast_1d(estimator.coef_[0])
    intercept = float(estimator.intercept_[0])

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_test)),
        "precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "log_loss": float(log_loss(y_test, y_prob_test, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_test, y_prob_test)) if len(set(y_test.tolist())) > 1 else None,
        "confusion_matrix": cm.tolist(),
        "true_positives": int(cm[1, 1]), "false_positives": int(cm[0, 1]),
        "true_negatives": int(cm[0, 0]), "false_negatives": int(cm[1, 0]),
    }
    if cv_scores is not None:
        metrics["cv_roc_auc_mean"] = float(np.mean(cv_scores))
        metrics["cv_roc_auc_std"] = float(np.std(cv_scores))
    if hasattr(pipeline.named_steps["model"], "best_params_"):
        metrics["optimal_C"] = float(pipeline.named_steps["model"].best_params_["C"])
        metrics["optimal_penalty"] = pipeline.named_steps["model"].best_params_["penalty"]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)

    decision_boundary = None
    if X.shape[1] == 1 and len(coefs) == 1 and coefs[0] != 0:
        decision_boundary = float(-intercept / coefs[0])

    if X.shape[1] == 1:
        x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_prob_line = pipeline.predict_proba(x_line)[:, 1]
        viz = {
            "kind": "line",
            "feature": feature_cols[0],
            "x_original": [float(v) for v in X.flatten()],
            "labels": [int(v) for v in y_log],
            "x_line": [float(v[0]) for v in x_line],
            "probability_line": [float(v) for v in y_prob_line],
            "decision_boundary": decision_boundary,
        }
    else:
        probs_all = pipeline.predict_proba(X)[:, 1]
        viz = {
            "kind": "predicted_vs_actual",
            "features": feature_cols,
            "labels": [int(v) for v in y_log],
            "predicted_probability": [float(v) for v in probs_all],
        }

    model_id = model_registry.register(pipeline, "logistic", feature_cols, feature_names, degree, interactions)

    return {
        "type": "Logistic Regression",
        "model_id": model_id,
        "equation": _build_equation(coefs, intercept, feature_names) + "  (log-odds; apply sigmoid for P(y=1))",
        "coefficients": [float(c) for c in coefs],
        "intercept": intercept,
        "feature_names": feature_names,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "visualization": viz,
        "roc_curve": {"fpr": [float(v) for v in fpr], "tpr": [float(v) for v in tpr], "thresholds": [float(v) for v in thresholds]},
        "feature_importance": [
            {"feature": name, "coefficient": float(c), "abs_coefficient": float(abs(c)), "odds_ratio": float(np.exp(c))}
            for name, c in zip(feature_names, coefs)
        ],
        "hyperparameters": {
            "polynomial_degree": degree, "interactions": interactions, "scaler": scaler_type.value,
            "cross_validation": cv, "hyperparameter_tuning": hyperparam_tuning,
        },
    }


def run_advanced_fit(request: AdvancedFitRequest) -> Dict:
    """Core fitting logic for all regression types. Synchronous and
    side-effect-free besides model_registry writes, so it's safe to run
    inside JobQueue's thread pool for /fit_async too."""
    if len(request.data) < 5:
        raise HTTPException(status_code=400, detail="Need at least 5 data points")

    n_test = max(1, int(len(request.data) * request.test_ratio))
    if n_test >= len(request.data):
        raise HTTPException(status_code=400, detail="test_ratio too high - no training data left.")
    if (request.cross_validation or request.hyperparameter_tuning) and len(request.data) - n_test < request.cv_folds:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough training rows ({len(request.data) - n_test}) for cv_folds={request.cv_folds}.",
        )

    X, y, labels, feature_cols = extract_features(request.data)

    results = {"request_summary": {
        "n_samples": len(request.data),
        "n_features": X.shape[1],
        "feature_cols": feature_cols,
        "test_size": n_test,
        "models_requested": request.regression_type.value,
    }}

    types_to_fit = (
        ["linear", "ridge", "lasso", "elasticnet", "logistic"]
        if request.regression_type == RegressionType.all else [request.regression_type.value]
    )

    for model_type in types_to_fit:
        if model_type == "logistic":
            y_log = _derive_binary_labels(request.data, y, labels)
            results[model_type] = _fit_logistic_model(
                X, y_log, feature_cols, request.test_ratio, request.seed,
                request.polynomial_degree, request.include_interactions, request.scaler,
                request.cross_validation, request.cv_folds, request.hyperparameter_tuning,
            )
        else:
            results[model_type] = _fit_model(
                model_type, X, y, feature_cols, request.test_ratio, request.seed,
                request.polynomial_degree, request.include_interactions,
                request.scaler, request.feature_selection, request.k_best,
                request.cross_validation, request.cv_folds,
                request.hyperparameter_tuning, request.alpha_values,
                request.bootstrap_iterations, request.confidence_interval,
                request.include_residuals,
            )

    return results


# ==================== LEGACY MANUAL PREDICTION (single feature only) ====================

def _predict_value(model_type, coefficients, intercept, x_value, polynomial_degree,
                    scaler_mean=None, scaler_std=None):
    """Manual, coefficient-based, single-feature-only prediction. Prefer
    /models/{model_id}/predict for anything with more than one feature or
    any feature selection - this can't faithfully replay those."""
    if model_type in ("linear", "ridge", "lasso", "elasticnet"):
        features = [x_value**i for i in range(1, polynomial_degree + 1)] if polynomial_degree > 1 else [x_value]
        if len(features) != len(coefficients):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(features)} coefficients for degree={polynomial_degree}, got {len(coefficients)}.",
            )
        if scaler_mean is not None and scaler_std is not None:
            features = [(f - scaler_mean) / scaler_std for f in features]
        prediction = sum(c * f for c, f in zip(coefficients, features)) + intercept
        return {"model": model_type, "input": x_value, "prediction": float(prediction), "type": "continuous"}

    z = coefficients[0] * x_value + intercept
    probability = 1 / (1 + np.exp(-z))
    return {
        "model": "logistic", "input": x_value, "probability": float(probability),
        "prediction": int(probability > 0.5), "type": "binary",
        "odds": float(np.exp(z)), "log_odds": float(z),
    }


# ==================== DRIFT DETECTION ====================

def _population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI computed over quantile bins of the *reference* (expected)
    distribution. Returns 0.0 if the reference distribution doesn't have
    enough distinct values to form meaningful bins, rather than raising or
    fabricating a number."""
    breakpoints = np.linspace(0, 100, bins + 1)
    bp_values = np.percentile(expected, breakpoints)
    bp_values = np.unique(bp_values)
    if len(bp_values) < 3:
        return 0.0
    bp_values[0], bp_values[-1] = -np.inf, np.inf

    expected_counts, _ = np.histogram(expected, bins=bp_values)
    actual_counts, _ = np.histogram(actual, bins=bp_values)
    expected_pct = expected_counts / max(len(expected), 1)
    actual_pct = actual_counts / max(len(actual), 1)
    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


# ==================== CHART RENDERING ====================

def _setup_theme(theme: str):
    if theme == "dark":
        plt.style.use("dark_background")
        return {"bg": "#1a1a2e"}
    elif theme == "seaborn":
        sns.set_style("whitegrid")
        return {"bg": "white"}
    elif theme == "ggplot":
        plt.style.use("ggplot")
        return {"bg": "#f0f0f0"}
    else:
        plt.style.use("default")
        return {"bg": "white"}


def _render_advanced_chart(
    results: Dict, format: ChartFormat = ChartFormat.png, theme: str = "default",
    width: int = 1200, height: int = 600, dpi: int = 150,
):
    """Render a comprehensive multi-panel chart across all fitted models.
    Single-feature model results get their fit line; multi-feature results
    get a predicted-vs-actual scatter with a diagonal reference line."""
    colors = _setup_theme(theme)

    linear_models = [k for k in results.keys() if k in ("linear", "ridge", "lasso", "elasticnet")]
    has_logistic = "logistic" in results
    n_panels = len(linear_models) + (1 if has_logistic else 0)

    if n_panels == 0:
        raise HTTPException(status_code=400, detail="Nothing to chart.")

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=dpi)
    gs = GridSpec(2, max(2, n_panels), figure=fig, hspace=0.35, wspace=0.35)

    for i, model_key in enumerate(linear_models):
        result = results[model_key]
        viz = result["visualization"]
        ax = fig.add_subplot(gs[0, i])
        r2 = result["metrics"]["r2_score"]
        if viz["kind"] == "line":
            ax.scatter(viz["x_original"], viz["y_original"], alpha=0.5, s=30, color="steelblue", label="Data")
            ax.plot(viz["x_line"], viz["y_line"], color="crimson", linewidth=2.5, label="Fit")
            ax.set_xlabel(viz["feature"])
            ax.set_ylabel("y")
        else:
            ax.scatter(viz["y_true"], viz["y_predicted"], alpha=0.5, s=30, color="steelblue")
            lims = [min(viz["y_true"] + viz["y_predicted"]), max(viz["y_true"] + viz["y_predicted"])]
            ax.plot(lims, lims, color="crimson", linewidth=1.5, linestyle="--", label="y = \u0177")
            ax.set_xlabel("Actual y")
            ax.set_ylabel("Predicted y")
        ax.set_title(f"{result['type']}\nR\u00b2 = {r2:.3f}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    if has_logistic:
        result = results["logistic"]
        viz = result["visualization"]
        ax = fig.add_subplot(gs[1, 0])
        acc = result["metrics"]["accuracy"]
        if viz["kind"] == "line":
            ax.scatter(viz["x_original"], viz["labels"], alpha=0.5, s=30, color="steelblue", label="Labels (0/1)")
            ax.plot(viz["x_line"], viz["probability_line"], color="forestgreen", linewidth=2.5, label="P(y=1)")
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Threshold")
            ax.set_xlabel(viz["feature"])
            ax.set_ylabel("Probability")
        else:
            idx = np.arange(len(viz["labels"]))
            ax.scatter(idx, viz["predicted_probability"], c=viz["labels"], cmap="coolwarm", alpha=0.6, s=25)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Threshold")
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Predicted P(y=1)")
        ax.set_title(f"Logistic Regression\nAccuracy = {acc:.3f}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        if "roc_curve" in result:
            ax2 = fig.add_subplot(gs[1, 1])
            ax2.plot(result["roc_curve"]["fpr"], result["roc_curve"]["tpr"], color="darkorange", linewidth=2)
            ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
            auc = result["metrics"].get("roc_auc")
            ax2.set_title(f"ROC Curve\nAUC = {auc:.3f}" if isinstance(auc, float) else "ROC Curve", fontsize=10, fontweight="bold")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.grid(True, alpha=0.3)

    residuals_plotted = 0
    for model_key in linear_models:
        if "residuals" in results[model_key] and residuals_plotted < 2:
            col = min(len(linear_models) + residuals_plotted, max(2, n_panels) - 1)
            ax = fig.add_subplot(gs[1, 1 + residuals_plotted] if not has_logistic else gs[0, col])
            residuals = results[model_key]["residuals"]["residuals"]
            ax.scatter(range(len(residuals)), residuals, alpha=0.6, s=20)
            ax.axhline(0, color="red", linestyle="--", linewidth=1)
            ax.set_title(f"{model_key.title()} Residuals", fontsize=9)
            ax.set_xlabel("Index")
            ax.set_ylabel("Residual")
            ax.grid(True, alpha=0.3)
            residuals_plotted += 1

    plt.tight_layout()
    buf = io.BytesIO()
    fmt = "svg" if format == ChartFormat.svg else "png"
    plt.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor=colors["bg"], edgecolor="none")
    plt.close(fig)
    buf.seek(0)

    media_type = "image/svg+xml" if format == ChartFormat.svg else "image/png"
    return StreamingResponse(buf, media_type=media_type)


# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "message": "Advanced Regression Comparison API v5.0",
        "version": "5.0",
        "features": [
            "Multi-feature datasets (any number of x1, x2, ... columns per point)",
            "5 regression types (Linear, Ridge, Lasso, ElasticNet, Logistic)",
            "Polynomial features, interactions & feature selection (SelectKBest)",
            "Feature scaling (Standard, MinMax, Robust)",
            "Cross-validation & hyperparameter tuning (leak-free: refit per fold)",
            "Bootstrap confidence intervals",
            "A fitted-model registry - /models/{id}/predict always matches training exactly",
            "Pluggable cache backend (in-memory LRU or Redis)",
            "Drift detection (KS test, Wasserstein distance, PSI) via /drift",
            "A/B testing between two registered models via /ab_test",
            "Async job queue for long-running fits via /fit_async + /jobs/{id}",
            "Residual analysis & diagnostics",
            "ROC curves & confusion matrices",
            "Interactive charts (PNG/SVG, multiple themes)",
            "Structured logging + a real in-memory rate limiter",
            "WebSocket streaming for large datasets",
            "Export to JSON, CSV, Excel",
        ],
        "endpoints": {
            "docs": "/docs",
            "sample": "GET /sample/{kind}?n_samples=100&noise=0.1&seed=42 (kinds include 'multifeature')",
            "fit": "POST /fit - basic fit (cached)",
            "fit_async": "POST /fit_async - submit a long-running fit, poll GET /jobs/{job_id}",
            "advanced_fit": "POST /advanced_fit - full diagnostics",
            "compare": "POST /compare - side-by-side comparison",
            "predict_legacy": "POST /predict - manual coefficients, single feature only (see docstring)",
            "batch_predict_legacy": "POST /batch_predict",
            "models_list": "GET /models - list fitted models still in the registry",
            "model_predict": "POST /models/{model_id}/predict - exact prediction, body: {\"features\": {...}}",
            "model_batch_predict": "POST /models/{model_id}/batch_predict - body: {\"rows\": [{...}, ...]}",
            "model_delete": "DELETE /models/{model_id}",
            "drift": "POST /drift - KS test / Wasserstein distance / PSI per feature",
            "ab_test": "POST /ab_test - paired bootstrap comparison of two registered models",
            "chart_demo": "GET /chart/demo",
            "chart": "POST /chart",
            "diagnostics": "POST /diagnostics - residual diagnostics",
            "export": "POST /export - export data + exact predictions",
            "theory": "GET /theory",
            "health": "GET /health",
            "websocket": "WS /ws/stream - real-time data streaming",
        },
    }


@app.get("/health")
async def health():
    if isinstance(result_cache, MemoryCache):
        result_cache.sweep()
    model_registry.sweep()
    job_queue.sweep()
    return {
        "status": "ok",
        "version": "5.0",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "cache_backend": type(result_cache).__name__,
        "cache_size": await result_cache.size(),
        "registered_models": len(model_registry),
        "active_jobs": len(job_queue),
        "rate_limit": {"max_requests": rate_limiter.max_requests, "window_seconds": rate_limiter.window_seconds},
    }


@app.get("/sample/{kind}")
async def get_sample_data(
    kind: Literal["linear", "logistic", "nonlinear", "heteroscedastic", "outliers", "multifeature"],
    n_samples: int = Query(default=100, ge=5, le=10000),
    noise: float = Query(default=0.1, ge=0.0, le=10.0),
    seed: int = 42,
):
    return {"data": generate_sample_data(kind, n_samples, noise, seed)}


@app.post("/fit")
async def fit_regression(request: RegressionRequest):
    """Fit regression model(s) with basic metrics."""
    cache_key = _cache_key(request.model_dump(mode="json"))
    cached = await result_cache.get(cache_key)
    if cached:
        return cached

    advanced_req = AdvancedFitRequest(**request.model_dump(), include_residuals=False, bootstrap_iterations=0)
    results = run_advanced_fit(advanced_req)
    await result_cache.set(cache_key, results)
    return results


@app.post("/fit_async")
async def fit_regression_async(request: AdvancedFitRequest):
    """Submit a fit to run in the background (useful for large datasets,
    many bootstrap iterations, or hyperparameter tuning that would otherwise
    block the request for a long time). Poll GET /jobs/{job_id} for status."""
    job_id = await job_queue.submit(run_advanced_fit, request)
    return {"job_id": job_id, "status": "pending", "poll_url": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job_id.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job["result"] if job["status"] == JobStatus.done else None,
        "error": job["error"],
        "created_at": job["created_at"],
        "finished_at": job["finished_at"],
    }


@app.post("/advanced_fit")
async def advanced_fit(request: AdvancedFitRequest):
    """Fit with full diagnostics: residuals, bootstrap CIs."""
    return run_advanced_fit(request)


@app.post("/compare")
async def compare_models(request: RegressionRequest):
    """Detailed comparison with verdict and recommendations."""
    if request.regression_type != RegressionType.all:
        raise HTTPException(status_code=400, detail="Use regression_type='all' for comparison")

    results = run_advanced_fit(AdvancedFitRequest(**request.model_dump(), include_residuals=True))

    comparison = {
        "model_comparison": {},
        "recommendation": {},
        "verdict": (
            "These models solve different problems and use different metrics. "
            "R\u00b2 measures how well a line fits continuous data; accuracy measures "
            "how often a classifier is correct. Direct comparison isn't meaningful."
        ),
    }
    best_for = {
        "linear": "Baseline continuous prediction, interpretable coefficients",
        "ridge": "Continuous prediction with multicollinearity, L2 regularization",
        "lasso": "Feature selection + prediction, L1 regularization (sparse models)",
        "elasticnet": "Best of Ridge+Lasso, handles correlated features",
        "logistic": "Binary classification, probability estimation",
    }
    for model_key in ["linear", "ridge", "lasso", "elasticnet", "logistic"]:
        if model_key in results:
            info = results[model_key]
            comparison["model_comparison"][model_key] = {
                "type": info.get("type"),
                "best_for": best_for.get(model_key),
                "key_metric": info["metrics"].get("r2_score") or info["metrics"].get("accuracy"),
                "equation": info.get("equation"),
                "model_id": info.get("model_id"),
            }

    _, y, _, _ = extract_features(request.data)
    y_unique = len(np.unique(y))
    if y_unique <= 2:
        comparison["recommendation"] = {"primary": "logistic", "reason": "Target has only 2 unique values - use logistic regression"}
    elif y_unique <= 10:
        comparison["recommendation"] = {"primary": "logistic", "reason": "Target has few unique values - consider logistic or ordinal regression"}
    else:
        comparison["recommendation"] = {"primary": "ridge", "reason": "Continuous target - start with Ridge for stability, compare with Lasso for feature selection"}

    return {**results, **comparison}


@app.post("/predict")
async def predict(request: PredictionRequest):
    return _predict_value(
        request.model_type.value, request.coefficients, request.intercept,
        request.x_value, request.polynomial_degree, request.scaler_mean, request.scaler_std,
    )


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    predictions = [
        _predict_value(request.model_type.value, request.coefficients, request.intercept,
                       x, request.polynomial_degree, request.scaler_mean, request.scaler_std)
        for x in request.x_values
    ]
    return {"predictions": predictions, "count": len(predictions)}


@app.get("/models")
async def list_models():
    return {"models": model_registry.list()}


def _features_dict_to_row(features: Dict[str, float], feature_cols: List[str]) -> np.ndarray:
    missing = [c for c in feature_cols if c not in features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing feature(s): {missing}. Expected: {feature_cols}")
    extra = [c for c in features if c not in feature_cols]
    if extra:
        raise HTTPException(status_code=400, detail=f"Unexpected feature(s): {extra}. Expected: {feature_cols}")
    return np.array([[features[c] for c in feature_cols]])


@app.post("/models/{model_id}/predict")
async def predict_with_model(model_id: str, request: ModelPredictRequest):
    """Predict using the actual fitted pipeline. Body: {"features": {"x1": 2.0, "x2": 7.5}}
    - keys must exactly match the feature_cols the model was trained on
    (see GET /models)."""
    entry = model_registry.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown or expired model_id. Fit a model first via /fit or /advanced_fit.")

    X = _features_dict_to_row(request.features, entry["feature_cols"])
    pipeline = entry["pipeline"]
    if entry["model_type"] == "logistic":
        prob = float(pipeline.predict_proba(X)[0, 1])
        return {"model_id": model_id, "model_type": "logistic", "input": request.features,
                "probability": prob, "prediction": int(prob > 0.5)}
    pred = float(pipeline.predict(X)[0])
    return {"model_id": model_id, "model_type": entry["model_type"], "input": request.features, "prediction": pred}


@app.post("/models/{model_id}/batch_predict")
async def batch_predict_with_model(model_id: str, request: ModelBatchPredictRequest):
    entry = model_registry.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown or expired model_id. Fit a model first via /fit or /advanced_fit.")

    feature_cols = entry["feature_cols"]
    X = np.vstack([_features_dict_to_row(row, feature_cols) for row in request.rows])
    pipeline = entry["pipeline"]
    if entry["model_type"] == "logistic":
        probs = pipeline.predict_proba(X)[:, 1]
        predictions = [{"input": row, "probability": float(p), "prediction": int(p > 0.5)}
                       for row, p in zip(request.rows, probs)]
    else:
        preds = pipeline.predict(X)
        predictions = [{"input": row, "prediction": float(p)} for row, p in zip(request.rows, preds)]
    return {"model_id": model_id, "model_type": entry["model_type"], "predictions": predictions, "count": len(predictions)}


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    if not model_registry.delete(model_id):
        raise HTTPException(status_code=404, detail="Unknown model_id.")
    return {"deleted": model_id}


@app.post("/drift")
async def detect_drift(request: DriftRequest):
    """Per-feature drift between a reference dataset and a current one:
    KS test (distributional shift), Wasserstein distance (magnitude of
    shift), and PSI (Population Stability Index, the usual MLOps threshold
    metric: >0.2 is commonly treated as significant drift)."""
    ref_X, ref_y, _, ref_cols = extract_features(request.reference_data)
    cur_X, cur_y, _, cur_cols = extract_features(request.current_data)

    if ref_cols != cur_cols:
        raise HTTPException(
            status_code=400,
            detail=f"reference_data and current_data must use the same feature columns. Got {ref_cols} vs {cur_cols}.",
        )

    feature_drift = {}
    drifted_features = []
    for i, col in enumerate(ref_cols):
        ref_vals, cur_vals = ref_X[:, i], cur_X[:, i]
        ks_stat, ks_p = ks_2samp(ref_vals, cur_vals)
        wdist = wasserstein_distance(ref_vals, cur_vals)
        psi = _population_stability_index(ref_vals, cur_vals, bins=request.psi_bins)
        is_drifted = bool(ks_p < 0.05 or psi > 0.2)
        if is_drifted:
            drifted_features.append(col)
        feature_drift[col] = {
            "ks_statistic": float(ks_stat), "ks_p_value": float(ks_p),
            "wasserstein_distance": float(wdist), "psi": psi, "drift_detected": is_drifted,
        }

    y_ks_stat, y_ks_p = ks_2samp(ref_y, cur_y)

    recommendations = []
    if drifted_features:
        recommendations.append(
            f"Distributional drift detected in: {', '.join(drifted_features)}. "
            "Consider retraining or investigating upstream data changes."
        )
    else:
        recommendations.append("No significant feature drift detected (KS p >= 0.05 and PSI <= 0.2 for all features).")
    if y_ks_p < 0.05:
        recommendations.append("Target distribution also shifted significantly - check whether the underlying relationship itself has changed, not just the inputs.")

    return {
        "drift_detected": bool(drifted_features),
        "drifted_features": drifted_features,
        "feature_drift": feature_drift,
        "target_drift": {"ks_statistic": float(y_ks_stat), "ks_p_value": float(y_ks_p)},
        "recommendations": recommendations,
        "psi_interpretation": {"<0.1": "no significant shift", "0.1-0.2": "moderate shift, monitor", ">0.2": "significant shift"},
    }


@app.post("/ab_test")
async def ab_test(request: ABTestRequest):
    """Compare two registered models (same model_type) on the same held-out
    data using a paired bootstrap: each bootstrap resample of test_data is
    scored by both models, so the confidence interval is on the *difference*
    in score, not two independent CIs compared informally."""
    entry_a = model_registry.get(request.model_id_a)
    entry_b = model_registry.get(request.model_id_b)
    if entry_a is None or entry_b is None:
        raise HTTPException(status_code=404, detail="One or both model_ids were not found or have expired.")
    if entry_a["model_type"] != entry_b["model_type"]:
        raise HTTPException(
            status_code=400,
            detail=f"Models must be the same type to compare fairly (got {entry_a['model_type']} vs {entry_b['model_type']}).",
        )
    if entry_a["feature_cols"] != entry_b["feature_cols"]:
        raise HTTPException(
            status_code=400,
            detail=f"Models were trained on different feature columns ({entry_a['feature_cols']} vs {entry_b['feature_cols']}) and can't be scored on the same test_data.",
        )

    X, y, labels, cols = extract_features(request.test_data)
    if cols != entry_a["feature_cols"]:
        raise HTTPException(status_code=400, detail=f"test_data feature columns {cols} don't match the models' training columns {entry_a['feature_cols']}.")

    is_classification = entry_a["model_type"] == "logistic"
    if is_classification:
        y_eval = _derive_binary_labels(request.test_data, y, labels)

        def score_fn(pipeline, X_, y_):
            if len(set(y_.tolist())) > 1:
                return roc_auc_score(y_, pipeline.predict_proba(X_)[:, 1])
            return accuracy_score(y_, pipeline.predict(X_))
        metric_name = "roc_auc (or accuracy if only one class present in a resample)"
    else:
        y_eval = y
        def score_fn(pipeline, X_, y_):
            return r2_score(y_, pipeline.predict(X_))
        metric_name = "r2_score"

    score_a = float(score_fn(entry_a["pipeline"], X, y_eval))
    score_b = float(score_fn(entry_b["pipeline"], X, y_eval))

    rng = np.random.default_rng(request.seed)
    n = len(y_eval)
    diffs = []
    for _ in range(request.bootstrap_iterations):
        idx = rng.choice(n, size=n, replace=True)
        X_boot, y_boot = X[idx], y_eval[idx]
        try:
            diffs.append(score_fn(entry_a["pipeline"], X_boot, y_boot) - score_fn(entry_b["pipeline"], X_boot, y_boot))
        except Exception:
            continue

    if not diffs:
        raise HTTPException(status_code=400, detail="All bootstrap resamples failed - try more test_data rows.")

    diffs = np.array(diffs)
    alpha = 1 - request.confidence
    lower = float(np.percentile(diffs, alpha / 2 * 100))
    upper = float(np.percentile(diffs, (1 - alpha / 2) * 100))
    significant = not (lower <= 0 <= upper)
    winner = ("model_a" if score_a > score_b else "model_b") if significant else None

    return {
        "metric": metric_name,
        "model_a": {"model_id": request.model_id_a, "score": score_a},
        "model_b": {"model_id": request.model_id_b, "score": score_b},
        "score_difference_a_minus_b": score_a - score_b,
        "bootstrap_iterations_used": len(diffs),
        "confidence_interval": {"lower": lower, "upper": upper, "confidence_level": request.confidence},
        "significant": significant,
        "winner": winner,
        "note": "Paired bootstrap: both models are scored on the same resampled rows each iteration, "
                "so the CI reflects the score difference directly rather than comparing two separate CIs.",
    }


@app.get("/chart/demo")
async def chart_demo(
    n_samples: int = Query(default=100, ge=10, le=500), noise: float = Query(default=0.1, ge=0.0, le=2.0),
    seed: int = 42, format: ChartFormat = ChartFormat.png, theme: str = "default",
    width: int = 1200, height: int = 600, dpi: int = 150,
):
    """Zero-config demo chart with all models."""
    data = [DataPoint(**d) for d in generate_sample_data("logistic", n_samples, noise, seed)]
    request = AdvancedFitRequest(data=data, regression_type=RegressionType.all, seed=seed, include_residuals=True)
    results = run_advanced_fit(request)
    return _render_advanced_chart(results, format, theme, width, height, dpi)


@app.post("/chart")
async def chart(request: ChartRequest):
    """Custom chart from your data."""
    adv_request = AdvancedFitRequest(
        data=request.data, regression_type=request.regression_type,
        polynomial_degree=request.polynomial_degree, seed=request.seed, include_residuals=True,
    )
    results = run_advanced_fit(adv_request)
    return _render_advanced_chart(results, request.chart_format, request.theme, request.width, request.height, request.dpi)


@app.post("/diagnostics")
async def diagnostics(request: RegressionRequest):
    """Residual diagnostics and assumption testing."""
    advanced_req = AdvancedFitRequest(**request.model_dump(), include_residuals=True)
    results = run_advanced_fit(advanced_req)

    diagnostics_report = {}
    for model_key in ["linear", "ridge", "lasso", "elasticnet"]:
        if model_key in results and "residuals" in results[model_key]:
            res = results[model_key]["residuals"]
            dw = res.get("durbin_watson")
            warnings = []
            if dw is not None and (dw < 1.5 or dw > 2.5):
                warnings.append(f"Durbin-Watson = {dw:.2f} suggests autocorrelation in residuals")
            diagnostics_report[model_key] = {
                "durbin_watson": dw, "mean_residual": res.get("mean_residual"), "std_residual": res.get("std_residual"),
                "assumptions": {
                    "linearity": "Check residual plot for patterns (should be random scatter)",
                    "homoscedasticity": "Check for funnel shape in residuals vs fitted",
                    "independence": f"Durbin-Watson \u2248 2.0 is good (got {dw if dw is not None else 'N/A'})",
                    "normality": "Use Q-Q plot or Shapiro-Wilk test",
                },
                "warnings": warnings,
            }
    return diagnostics_report


@app.post("/export")
async def export_data(request: ExportRequest):
    """Export data with exact predictions (from the fitted-model registry)
    to various formats."""
    adv_request = AdvancedFitRequest(
        data=request.data, regression_type=request.regression_type,
        polynomial_degree=request.polynomial_degree, seed=request.seed,
    )
    results = run_advanced_fit(adv_request)

    X, y, labels, feature_cols = extract_features(request.data)
    df_data = [{"index": i, **{c: float(X[i, j]) for j, c in enumerate(feature_cols)}, "y": float(y[i])}
               for i in range(len(request.data))]
    if labels is not None:
        for row, lab in zip(df_data, labels):
            row["label"] = int(lab)

    if request.include_predictions:
        for model_key in ["linear", "ridge", "lasso", "elasticnet", "logistic"]:
            if model_key not in results:
                continue
            entry = model_registry.get(results[model_key]["model_id"])
            if entry is None:
                continue
            pipeline = entry["pipeline"]
            preds = pipeline.predict_proba(X)[:, 1] if model_key == "logistic" else pipeline.predict(X)
            for row, p in zip(df_data, preds):
                row[f"{model_key}_predicted"] = float(p)

    df = pd.DataFrame(df_data)

    if request.format == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return StreamingResponse(io.BytesIO(buf.getvalue().encode()), media_type="text/csv",
                                  headers={"Content-Disposition": "attachment; filename=regression_data.csv"})
    elif request.format == "excel":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Data", index=False)
            metrics_data = []
            for model_key in ["linear", "ridge", "lasso", "elasticnet", "logistic"]:
                if model_key in results:
                    for metric_name, value in results[model_key]["metrics"].items():
                        if value is not None:
                            metrics_data.append({"model": model_key, "metric": metric_name, "value": value})
            if metrics_data:
                pd.DataFrame(metrics_data).to_excel(writer, sheet_name="Metrics", index=False)
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                  headers={"Content-Disposition": "attachment; filename=regression_data.xlsx"})
    else:
        return SafeJSONResponse(content=df.to_dict(orient="records"))


@app.get("/theory")
async def get_theory():
    return {
        "linear_regression": {
            "definition": "Models linear relationship between independent variables and continuous dependent variable",
            "formula": "y = X\u03b2 + \u03b5",
            "assumptions": ["Linearity", "Independence", "Homoscedasticity", "Normality of errors"],
            "optimization": "Ordinary Least Squares (minimize \u03a3(y\u1d62 - \u0177\u1d62)\u00b2)",
            "pros": ["Simple", "Interpretable", "Fast"],
            "cons": ["Sensitive to outliers", "Assumes linearity", "No regularization"],
        },
        "ridge_regression": {
            "definition": "Linear regression with L2 regularization (penalizes large coefficients)",
            "formula": "minimize \u03a3(y\u1d62 - \u0177\u1d62)\u00b2 + \u03b1\u03a3\u03b2\u2c7c\u00b2",
            "best_for": "Multicollinearity, overfitting prevention",
            "pros": ["Handles multicollinearity", "Stable predictions", "All features retained"],
            "cons": ["Doesn't perform feature selection", "Requires tuning \u03b1"],
        },
        "lasso_regression": {
            "definition": "Linear regression with L1 regularization (can zero out coefficients)",
            "formula": "minimize \u03a3(y\u1d62 - \u0177\u1d62)\u00b2 + \u03b1\u03a3|\u03b2\u2c7c|",
            "best_for": "Feature selection, sparse models",
            "pros": ["Automatic feature selection", "Interpretable sparse models"],
            "cons": ["Unstable with correlated features", "Can select only one from correlated pair"],
        },
        "elasticnet": {
            "definition": "Combines L1 and L2 regularization",
            "formula": "minimize \u03a3(y\u1d62 - \u0177\u1d62)\u00b2 + \u03b1[l1_ratio\u00b7\u03a3|\u03b2\u2c7c| + (1-l1_ratio)\u00b7\u03a3\u03b2\u2c7c\u00b2]",
            "best_for": "Many correlated features + feature selection",
            "pros": ["Best of both worlds", "Handles correlated features better than Lasso"],
            "cons": ["Two hyperparameters to tune", "More complex"],
        },
        "logistic_regression": {
            "definition": "Models probability of binary outcome using sigmoid function",
            "formula": "log(p/(1-p)) = X\u03b2 \u2192 p = 1/(1+e^(-X\u03b2))",
            "assumptions": ["Binary outcome", "Linear in log-odds", "No multicollinearity", "Large sample"],
            "optimization": "Maximum Likelihood Estimation",
            "pros": ["Probabilistic output", "Interpretable coefficients (odds ratios)", "Fast"],
            "cons": ["Assumes linear decision boundary", "Struggles with complex patterns"],
        },
        "drift_detection": {
            "ks_test": "Kolmogorov-Smirnov test - are two samples drawn from the same distribution? Low p-value (<0.05) suggests drift.",
            "wasserstein_distance": "'Earth mover's distance' - magnitude of the shift, in the feature's own units.",
            "psi": "Population Stability Index - bins the reference distribution and compares bin proportions. >0.2 is a common significant-drift threshold.",
        },
        "comparison_table": {
            "regularization": {"linear": "None", "ridge": "L2 (squared)", "lasso": "L1 (absolute)", "elasticnet": "L1 + L2", "logistic": "Optional L1/L2"},
            "feature_selection": {"linear": "No", "ridge": "No", "lasso": "Yes (sparse)", "elasticnet": "Partial", "logistic": "Optional"},
            "handles_multicollinearity": {"linear": "Poor", "ridge": "Excellent", "lasso": "Poor", "elasticnet": "Good", "logistic": "Moderate"},
        },
    }


# ==================== WEBSOCKET ====================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_progress(self, websocket: WebSocket, message: str):
        await websocket.send_json({"type": "progress", "message": message})


manager = ConnectionManager()


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time streaming for large dataset processing."""
    await manager.connect(websocket)
    try:
        await websocket.send_json({"type": "connected", "message": "Ready to receive data"})
        while True:
            data = await websocket.receive_json()

            if data.get("action") == "fit":
                await manager.send_progress(websocket, "Received data, validating...")
                try:
                    points = [DataPoint(**d) for d in data.get("data", [])]
                    request = AdvancedFitRequest(
                        data=points, regression_type=RegressionType(data.get("regression_type", "all")),
                        polynomial_degree=data.get("polynomial_degree", 1), seed=data.get("seed", 42),
                    )
                    await manager.send_progress(websocket, f"Fitting on {len(points)} points...")
                    results = run_advanced_fit(request)
                    await websocket.send_json({"type": "result", "data": results})
                except Exception as e:
                    logger.warning("websocket_fit_error: %s", e)
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif data.get("action") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
