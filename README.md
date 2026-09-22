# StockPulse 📈
### Global Large-Cap Stock Movement Prediction System

> **Production-grade ML platform** — LightGBM · XGBoost · LSTM · FastAPI · React  
> 4 markets · 40+ features · Dynamic resolution · Real-time data · Dark UI Dashboard

---

## Problem Statement

Predicting short-term equity price direction is a fundamental challenge in quantitative finance. This system converts a research-grade Jupyter notebook (covering 120 large-cap stocks across USA, India, China/HK, and Europe) into a fully deployable, production-ready platform.

**Input**: Company name or ticker symbol (e.g. `"Apple"` or `"AAPL"`)  
**Output**: 5-day price direction prediction (UP/DOWN), probability score, BUY/HOLD/SELL signal, interactive charts

---

## Features

| Feature | Detail |
|---|---|
| 🔍 Dynamic Resolution | Company names resolved to tickers at runtime via yfinance — no static dictionaries |
| 📡 Real-Time Data | Fresh OHLCV fetched on every query — no CSV, no stale data |
| 🧠 ML Pipeline | LightGBM (primary) + XGBoost + LSTM ensemble |
| 📊 40+ Features | Momentum, volatility, MA signals, RSI, MACD, Bollinger, volume |
| 🌍 4 Markets | US (NYSE/NASDAQ) · India (NSE) · China/HK (HKEX) · Europe (LSE, Xetra, Euronext) |
| 🎨 Dark UI | Professional dark-themed React dashboard with animated charts |
| 🚀 FastAPI Backend | REST API with autocomplete search, rate limiting, auth, metrics |
| 🐳 Docker Ready | Single `docker-compose up` deployment |

---

## Architecture

```
User Input ("Apple")
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  React Frontend (Vite + TailwindCSS-style inline)        │
│  SearchBar → Autocomplete → Charts → Prediction UI       │
└────────────────────────┬────────────────────────────────┘
                         │ POST /predict
                         ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI  (api/app.py)                                   │
│  Auth · Rate Limit · CORS · Metrics                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  ML Pipeline                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ data_         │  │ preprocessing│  │ feature_      │ │
│  │ ingestion.py  │→ │     .py      │→ │ engineering   │ │
│  │               │  │              │  │     .py       │ │
│  │ TickerResolver│  │ OHLC clean   │  │ 40+ features  │ │
│  │ StockFetcher  │  │ outliers     │  │ RSI,MACD,BB   │ │
│  └──────────────┘  └──────────────┘  └───────┬───────┘ │
│                                               │         │
│  ┌──────────────────────────────────────────▼───────┐  │
│  │  predict.py  (OnTheFlyPredictor)                  │  │
│  │  Quick-trains LightGBM on 70% data               │  │
│  │  Predicts latest row · Returns full result       │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
              yfinance (real-time OHLCV)
```

---

## Folder Structure

```
stockpulse/
├── backend/
│   ├── data_ingestion.py        # TickerResolver + StockDataFetcher
│   ├── preprocessing.py         # OHLCV validation, outliers, ffill
│   ├── feature_engineering.py   # 40+ technical features + target
│   ├── model.py                 # LightGBM, XGBoost, LSTM definitions
│   ├── train.py                 # CLI training pipeline (offline)
│   └── predict.py               # OnTheFlyPredictor (used by API)
│
├── api/
│   └── app.py                   # FastAPI: /predict /search /history /metrics
│
├── frontend/
│   ├── App.jsx                  # Complete React dashboard (single file)
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
├── config/
│   └── config.yaml              # All tuneable parameters
│
├── scripts/
│   └── (utility scripts)
│
├── models/                      # Saved model artefacts (auto-created)
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start (Local)

### Prerequisites
- Python 3.11+
- Node.js 20+

### 1. Backend

```bash
# Clone and enter directory
cd stockpulse

# Install Python deps
pip install -r requirements.txt

# Start the API server
uvicorn api.app:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open: http://localhost:3000

### 3. Docker (Full Stack)

```bash
docker-compose up --build
```

Open: http://localhost (Nginx serves frontend + proxies API)

---

## API Reference

### `POST /predict`

Run the full ML pipeline for a query.

**Request:**
```json
{
  "query": "Apple",
  "data_period": "2y"
}
```

**Response:**
```json
{
  "ok": true,
  "latency_ms": 4200.1,
  "data": {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "market": "US",
    "sector": "Technology",
    "direction": "UP",
    "probability": 0.6734,
    "confidence_label": "High",
    "signal": "BUY",
    "horizon_days": 5,
    "last_close": 213.49,
    "price_change_1d": 0.82,
    "price_change_5d": 2.14,
    "model_auc": 0.5812,
    "model_f1": 0.5634,
    "model_name": "LightGBM",
    "chart_data": [...],
    "top_features": [...]
  }
}
```

### `GET /search?q=apple`

Autocomplete suggestions. Returns `{results: [{ticker, name, exchange, type}]}`.

### `GET /history/{ticker}?period=6mo`

Raw OHLCV history for charting.

### `GET /metrics`

API usage and latency metrics.

---

## ML Pipeline Details

### Features (40+)

| Group | Features |
|---|---|
| Price Action | body_size, upper_wick, lower_wick, gap, hl_range |
| Momentum | ret_1d, ret_2d, ret_3d, ret_5d, ret_10d, ret_21d, log_ret_1d |
| Volatility | vol_5d, vol_21d, vol_63d, atr_14, atr_pct, Garman-Klass vol |
| MA Signals | close vs SMA5/10/20/50/200, SMA crosses, EMA cross |
| RSI | RSI-7, RSI-14, RSI-21, RSI divergence |
| MACD | MACD line, signal, histogram, cross signal |
| Bollinger | %B, bandwidth, close vs BB mid |
| Volume | vol z-score, OBV slope, VWAP ratio, vol ratios |
| Regime | trend strength, skewness, kurtosis |

### Model Selection

1. **LightGBM** (primary) — fastest, handles missing values, balanced class weight
2. **XGBoost** — fallback if LightGBM unavailable
3. **Logistic Regression** — baseline / last resort
4. **LSTM** — optional deep learning (install PyTorch)

### Target

`target = 1` if `close[t+5] / close[t] - 1 > 0.005` (UP by 0.5%+ in 5 days)  
`target = 0` otherwise (DOWN or flat)

---

## Configuration

Edit `config/config.yaml` to tune any parameter:

```yaml
model:
  return_horizon: 5       # prediction window (days)
  target_threshold: 0.005 # 0.5% = Class 1
  confidence_thresh: 0.60 # min probability for High confidence
  data_period: "2y"       # yfinance data period

api:
  port: 8000
  rate_limit: 30           # requests/minute per IP
```

---

## Training (Offline / Production)

To pre-train and cache models for fast inference:

```bash
# Single stock
python -m backend.train --ticker AAPL

# Multiple stocks  
python -m backend.train --ticker AAPL MSFT TSLA NVDA

# With Optuna tuning (takes longer, better AUC)
python -m backend.train --ticker AAPL --tune

# Indian / European stocks
python -m backend.train --ticker "Reliance" "Infosys" "ASML"
```

Saved artefacts: `models/<TICKER>/` → `lgbm_model.pkl`, `imputer.pkl`, `feature_cols.json`

---

## Dynamic Stock Resolution — How It Works

```
User types: "Apple"
       │
       ▼
TickerResolver._looks_like_ticker("Apple")  →  False  (not all-caps)
       │
       ▼
TickerResolver._search_yahoo("Apple")
  → yf.Search("Apple", max_results=5)
  → quotes[0] = {symbol: "AAPL", quoteType: "EQUITY", ...}
       │
       ▼
TickerResolver._validate_ticker("AAPL")
  → yf.Ticker("AAPL").fast_info.last_price = 213.49  ✓
       │
       ▼
Returns TickerInfo(ticker="AAPL", name="Apple Inc.", market="US", ...)
```

Handles: ambiguous names (Meta → META), Indian stocks (HDFC Bank → HDFCBANK.NS), 
HK stocks (Tencent → 0700.HK), European stocks (ASML → ASML).

---

## Error Handling

| Scenario | Response |
|---|---|
| Invalid company name | `{"ok": false, "error": "Could not resolve 'XYZ'..."}` |
| API rate limit | HTTP 429 with retry-after header |
| Insufficient data | `{"ok": false, "error": "Only N days available..."}` |
| yfinance timeout | Caught, returns error message |
| No ML library | Falls back to Logistic Regression |

---

## Improvements Over Original Notebook

| Notebook Problem | Production Fix |
|---|---|
| Static CSV file (`fin_stock.csv`) | Real-time yfinance fetch per query |
| Hardcoded Windows paths | Config-driven, OS-agnostic |
| Tight coupling (all in one file) | 6 modular Python files |
| No deployment | FastAPI + Docker + Nginx |
| No UI | Full React dashboard |
| No error handling | Graceful fallbacks throughout |
| Manual training only | CLI + API-triggered training |
| No model versioning | Per-ticker model directories with meta.json |

---

## Roadmap / Future Improvements

- [ ] **Macro data** (VIX, yield curve, DXY via FRED API)
- [ ] **Earnings calendar** feature (days to/from earnings)
- [ ] **Sentiment** (FinBERT on news headlines)
- [ ] **WebSocket** real-time streaming via Kafka
- [ ] **Walk-forward validation** in API response
- [ ] **PostgreSQL** prediction logging + drift detection
- [ ] **Redis** caching for identical queries
- [ ] **JWT auth** for multi-user deployments
- [ ] **LSTM attention** (Temporal Fusion Transformer)
- [ ] **Kelly Criterion** position sizing module

---

## HFT-Level Advanced Architecture (Optional)

For microsecond-level latency requirements:

```
Market Data Feed (FIX/ITCH)
         │
         ▼
Kafka Streams (real-time ingestion)
         │
         ▼
Feature computation (C++ or Rust via PyO3)
         │
         ▼
ONNX Runtime (model export for <1ms inference)
         │
         ▼
WebSocket → React dashboard (sub-100ms E2E)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data | yfinance, pandas, numpy |
| Features | scikit-learn, ta-lib patterns (custom) |
| Models | LightGBM, XGBoost, PyTorch LSTM |
| Tuning | Optuna (Bayesian) |
| API | FastAPI, uvicorn, pydantic |
| Rate Limit | slowapi |
| Frontend | React 18, Vite, custom SVG charts |
| Deployment | Docker, Nginx |
| Config | PyYAML |

---

## License

MIT License. For educational and research purposes.  
Not financial advice. Always do your own due diligence.

---

*Built from a 120-stock research notebook → production-grade fintech system.*
