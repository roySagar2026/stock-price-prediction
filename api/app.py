"""
api/app.py
----------
FastAPI REST API layer for StockPulse.

Endpoints
---------
  GET  /                        health check
  POST /predict                 run full prediction pipeline
  GET  /search?q=               ticker/name autocomplete suggestions
  GET  /history/{ticker}        OHLCV history for charting
  GET  /metrics                 API usage metrics

Security:  simple API-key header (X-API-Key) can be enabled via env var.
Rate limit: slowapi (token bucket, 30 req/min per IP).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from functools import lru_cache
from typing import Optional

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── optional: rate limiting ───────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    limiter = Limiter(key_func=get_remote_address)
    HAS_LIMITER = True
except ImportError:
    HAS_LIMITER = False
    limiter = None

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from backend.predict import OnTheFlyPredictor, PredictionResult

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("stockpulse.api")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config():
    p = pathlib.Path(__file__).parent.parent / "config" / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)

cfg = load_config()
API_KEY = os.getenv("STOCKPULSE_API_KEY", "")      # empty = no auth required
predictor = OnTheFlyPredictor(cfg.get("model", {}))

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="StockPulse API",
    description="ML-based global large-cap stock movement prediction",
    version="1.0.0",
    docs_url="/docs",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
if HAS_LIMITER:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Simple metrics store
_metrics = {"requests": 0, "errors": 0, "total_latency_ms": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Auth dependency
# ─────────────────────────────────────────────────────────────────────────────

async def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if not API_KEY:
        return          # auth disabled
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    query: str                          # "Apple" | "AAPL" | "Reliance" | "RELIANCE.NS"
    data_period: Optional[str] = "2y"  # yfinance period string


class PredictResponse(BaseModel):
    ok: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "service": "StockPulse API", "version": "1.0.0"}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse, tags=["prediction"],
          dependencies=[Depends(verify_api_key)])
async def predict(req: PredictRequest, request: Request):
    """
    Run the full ML prediction pipeline for a company name or ticker.

    - Resolves company name to ticker at runtime (no static mapping)
    - Fetches fresh OHLCV data via yfinance
    - Engineers 40+ features, quick-trains LightGBM
    - Returns direction (UP/DOWN), probability, signal, chart data
    """
    _metrics["requests"] += 1
    t0 = time.perf_counter()

    if not req.query or len(req.query.strip()) < 1:
        raise HTTPException(status_code=422, detail="'query' must not be empty.")

    # Swap period on predictor if requested
    if req.data_period and req.data_period != predictor.cfg.get("data_period"):
        predictor.cfg["data_period"] = req.data_period

    try:
        result: PredictionResult = predictor.predict(req.query.strip())
    except ValueError as e:
        _metrics["errors"] += 1
        return PredictResponse(ok=False, error=str(e),
                               latency_ms=_elapsed_ms(t0))
    except Exception as e:
        _metrics["errors"] += 1
        logger.exception("Unexpected error for query '%s'", req.query)
        return PredictResponse(ok=False, error="Internal server error. Check logs.",
                               latency_ms=_elapsed_ms(t0))

    latency = _elapsed_ms(t0)
    _metrics["total_latency_ms"] += latency

    return PredictResponse(
        ok=True,
        data=dataclasses.asdict(result),
        latency_ms=latency,
    )


@app.get("/search", tags=["search"])
async def search(q: str, limit: int = 8):
    """
    Ticker / company name autocomplete using yfinance Search.
    Returns list of {ticker, name, exchange, type} objects.
    """
    if not q or len(q) < 1:
        return {"results": []}
    try:
        import yfinance as yf
        results = yf.Search(q, max_results=limit)
        quotes = results.quotes or []
        out = []
        for item in quotes:
            qt = item.get("quoteType", "")
            if qt not in ("EQUITY", "ETF", "INDEX"):
                continue
            out.append({
                "ticker":   item.get("symbol", ""),
                "name":     item.get("shortname") or item.get("longname") or "",
                "exchange": item.get("exchDisp") or item.get("exchange") or "",
                "type":     qt,
            })
        return {"results": out}
    except Exception as e:
        logger.warning("Search failed for '%s': %s", q, e)
        return {"results": []}


@app.get("/history/{ticker}", tags=["data"])
async def history(ticker: str, period: str = "6mo"):
    """
    Returns OHLCV history for charting. No ML, just raw data.
    """
    try:
        import yfinance as yf
        raw = yf.download(ticker, period=period, interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")
        if isinstance(raw.columns, type(raw.columns)) and hasattr(raw.columns, 'get_level_values'):
            try:
                raw.columns = raw.columns.get_level_values(0)
            except Exception:
                pass
        rows = []
        for date, row in raw.iterrows():
            rows.append({
                "date":   str(date)[:10],
                "open":   round(float(row["Open"]),   4),
                "high":   round(float(row["High"]),   4),
                "low":    round(float(row["Low"]),    4),
                "close":  round(float(row["Close"]),  4),
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
            })
        return {"ticker": ticker, "period": period, "data": rows}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """API usage and performance metrics."""
    n = _metrics["requests"]
    avg_lat = _metrics["total_latency_ms"] / n if n else 0
    return {
        "total_requests": n,
        "total_errors":   _metrics["errors"],
        "avg_latency_ms": round(avg_lat, 1),
        "error_rate_pct": round(_metrics["errors"] / n * 100, 2) if n else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


import pandas as pd   # needed inside /history endpoint

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    api_cfg = cfg.get("api", {})
    uvicorn.run(
        "api.app:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=True,
        log_level="info",
    )
