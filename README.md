<div align="center">

# 📈 StockPulse

### Global Large-Cap Stock Movement Prediction System

<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
<img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/LightGBM%20%7C%20XGBoost%20%7C%20LSTM-ML%20Ensemble-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/Status-Live-success?style=for-the-badge" />

<br/>

A **production-grade ML platform** that predicts short-term equity price direction —
turning a 120-stock research notebook into a fully deployable, real-time system.

**4 markets** · **40+ features** · **Dynamic ticker resolution** · **Real-time data** · **Dark UI Dashboard**

<br/>

### 🔗 [**Live Demo →** stock-price-prediction-lyart.vercel.app](https://stock-price-prediction-lyart.vercel.app/)

</div>

---

# 🎯 Problem Statement

Predicting short-term equity price direction is a fundamental challenge in quantitative finance. StockPulse converts a research-grade Jupyter notebook (covering **120 large-cap stocks** across USA, India, China/HK, and Europe) into a fully deployable, production-ready platform.

<div align="center">

| Input | Output |
|---|---|
| Company name or ticker (`"Apple"` / `"AAPL"`) | 5-day direction (UP/DOWN), probability score, BUY/HOLD/SELL signal, interactive charts |

</div>

---

# ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 ML Pipeline
- LightGBM (primary) + XGBoost + LSTM ensemble
- 40+ engineered technical features
- Dynamic ticker resolution via yfinance
- No static CSVs — fresh OHLCV on every query

</td>

<td width="50%">

### 🌍 Platform
- 4 markets: US, India, China/HK, Europe
- Dark-themed React dashboard
- FastAPI backend — auth, rate limiting, metrics
- Single-command Docker deployment

</td>
</tr>
</table>

---

# 🏗️ Architecture

<div align="center">

```text
User Input ("Apple")
       │
       ▼
React Frontend (Vite)
SearchBar → Autocomplete → Charts → Prediction UI
       │
       ▼  POST /predict
FastAPI  (api/app.py)
Auth · Rate Limit · CORS · Metrics
       │
       ▼
ML Pipeline
data_ingestion → preprocessing → feature_engineering → predict.py
TickerResolver     OHLC clean      40+ features      OnTheFlyPredictor
StockFetcher       outliers        RSI/MACD/BB       Quick-trains LightGBM
       │
       ▼
yfinance (real-time OHLCV)
```

</div>

---

# 📂 Folder Structure

```bash
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

# 🚀 Quick Start

## 1️⃣ Backend

```bash
cd stockpulse
pip install -r requirements.txt
uvicorn api.app:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

## 2️⃣ Frontend

```bash
cd frontend
npm install
npm run dev
```

Open: `http://localhost:3000`

## 3️⃣ Docker (Full Stack)

```bash
docker-compose up --build
```

Open: `http://localhost` (Nginx serves frontend + proxies API)

## 4️⃣ Live

No setup needed — try it here: **[stock-price-prediction-lyart.vercel.app](https://stock-price-prediction-lyart.vercel.app/)**

---

# 📡 API Reference

### `POST /predict`

Runs the full ML pipeline for a query.

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

<div align="center">

| Endpoint | Description |
|---|---|
| `GET /search?q=apple` | Autocomplete suggestions → `{results: [{ticker, name, exchange, type}]}` |
| `GET /history/{ticker}?period=6mo` | Raw OHLCV history for charting |
| `GET /metrics` | API usage and latency metrics |

</div>

---

# 🧠 ML Pipeline Details

## Features (40+)

<div align="center">

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

</div>

## Model Selection

1. **LightGBM** (primary) — fastest, handles missing values, balanced class weight
2. **XGBoost** — fallback if LightGBM unavailable
3. **Logistic Regression** — baseline / last resort
4. **LSTM** — optional deep learning (install PyTorch)

## Target

`target = 1` if `close[t+5] / close[t] - 1 > 0.005` (UP by 0.5%+ in 5 days)
`target = 0` otherwise (DOWN or flat)

---

# ⚙️ Configuration

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

# 🏋️ Training (Offline / Production)

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

# 🔍 Dynamic Stock Resolution — How It Works

<div align="center">

```text
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

</div>

Handles ambiguous names (Meta → META), Indian stocks (HDFC Bank → HDFCBANK.NS),
HK stocks (Tencent → 0700.HK), and European stocks (ASML → ASML).

---

# ⚠️ Error Handling

<div align="center">

| Scenario | Response |
|---|---|
| Invalid company name | `{"ok": false, "error": "Could not resolve 'XYZ'..."}` |
| API rate limit | HTTP 429 with retry-after header |
| Insufficient data | `{"ok": false, "error": "Only N days available..."}` |
| yfinance timeout | Caught, returns error message |
| No ML library | Falls back to Logistic Regression |

</div>

---

# 🛠️ Improvements Over Original Notebook

<div align="center">

| Notebook Problem | Production Fix |
|---|---|
| Static CSV file (`fin_stock.csv`) | Real-time yfinance fetch per query |
| Hardcoded Windows paths | Config-driven, OS-agnostic |
| Tight coupling (all in one file) | 6 modular Python files |
| No deployment | FastAPI + Docker + Nginx, **live on Vercel** |
| No UI | Full React dashboard |
| No error handling | Graceful fallbacks throughout |
| Manual training only | CLI + API-triggered training |
| No model versioning | Per-ticker model directories with meta.json |

</div>

---

# 🎯 Platform Capabilities

<div align="center">

| Capability | Status |
|------------|--------|
| Dynamic Ticker Resolution | ✅ |
| Real-Time OHLCV Ingestion | ✅ |
| LightGBM / XGBoost Ensemble | ✅ |
| React Dashboard | ✅ |
| Live Deployment | ✅ |
| Macro Data (VIX, DXY, yield curve) | 🚧 Planned |
| Sentiment (FinBERT on news) | 🚧 Planned |
| WebSocket Real-Time Streaming | 🚧 Planned |
| Multi-User Auth (JWT) | 🚧 Planned |

</div>

---

# 🔮 Roadmap / Future Improvements

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

# ⚡ HFT-Level Advanced Architecture (Optional)

For microsecond-level latency requirements:

<div align="center">

```text
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

</div>

---

# 🧰 Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| Data | yfinance, pandas, numpy |
| Features | scikit-learn, ta-lib patterns (custom) |
| Models | LightGBM, XGBoost, PyTorch LSTM |
| Tuning | Optuna (Bayesian) |
| API | FastAPI, uvicorn, pydantic |
| Rate Limit | slowapi |
| Frontend | React 18, Vite, custom SVG charts |
| Deployment | Docker, Nginx, Vercel |
| Config | PyYAML |

</div>

---

# ⚠️ Disclaimer

This project is intended for:
- Educational purposes
- Quantitative finance research
- ML system design practice
- Fintech engineering learning

It is **not financial advice**. Always do your own due diligence.

---

<div align="center">

### ⭐ If you like this project, consider giving it a star!

Built with ❤️ using Python, FastAPI & React for Quantitative Finance

**[🔗 Try the live demo](https://stock-price-prediction-lyart.vercel.app/)**

</div>
