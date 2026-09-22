"""
predict.py
----------
On-the-fly prediction pipeline.

Two modes:
  1. On-the-fly: fetch data → preprocess → feature-engineer → train quick
     LightGBM → predict latest row. No pre-saved model required.
  2. Saved model: load artefacts from models/<TICKER>/ and run inference.

The API layer always uses mode 1 (on-the-fly) so every query gets fresh data
and a freshly fitted model. This is intentional for demo: it proves the full
pipeline works end-to-end without offline training.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from backend.data_ingestion import TickerInfo, resolve_and_fetch
from backend.feature_engineering import FeatureEngineer
from backend.model import HAS_LGB, HAS_XGB, build_lgbm, build_xgboost
from backend.preprocessing import DataPreprocessor, data_quality_report

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    # Identity
    ticker: str
    company_name: str
    sector: str
    market: str
    currency: str

    # Core prediction
    direction: str           # "UP" | "DOWN"
    probability: float       # P(UP) ∈ [0,1]
    confidence_label: str    # "High" | "Medium" | "Low"
    signal: str              # "BUY" | "HOLD" | "SELL"
    horizon_days: int

    # Price context
    last_close: float
    price_change_1d: float
    price_change_5d: float

    # Metrics
    model_auc: float
    model_f1: float
    model_name: str

    # Chart data (OHLCV + indicators for frontend)
    chart_data: list = field(default_factory=list)  # [{date, close, sma20, ...}]

    # Feature importance top-10
    top_features: list = field(default_factory=list)  # [{name, importance}]

    # Data quality
    data_quality: dict = field(default_factory=dict)

    # Metadata
    data_period_days: int = 0
    fetch_timestamp: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Main predictor
# ─────────────────────────────────────────────────────────────────────────────

class OnTheFlyPredictor:
    """
    Full pipeline in one call: resolve → fetch → preprocess → engineer
    → quick-train LightGBM → predict latest close.
    """

    DEFAULT_CFG = {
        "return_horizon":   5,
        "target_threshold": 0.005,
        "ffill_limit":      3,
        "vol_window":       252,
        "data_period":      "2y",
        "confidence_thresh": 0.60,
    }

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = {**self.DEFAULT_CFG, **(cfg or {})}

    def predict(self, query: str) -> PredictionResult:
        import datetime

        # 1. Resolve + fetch
        ticker_info, raw_df = resolve_and_fetch(query, period=self.cfg["data_period"])

        # 2. Preprocess
        proc = DataPreprocessor(ffill_limit=self.cfg["ffill_limit"])
        clean_df = proc.run(raw_df, ticker=ticker_info.ticker)
        dq = data_quality_report(raw_df, clean_df)

        # 3. Feature engineering
        fe = FeatureEngineer(
            return_horizon    = self.cfg["return_horizon"],
            target_threshold  = self.cfg["target_threshold"],
            vol_window        = self.cfg["vol_window"],
        )
        feat_df, feature_cols = fe.build(clean_df)

        # 4. Train on all labelled rows (exclude last N rows with no future label)
        labelled = feat_df.dropna(subset=["target"] + feature_cols)
        n = len(labelled)
        if n < 60:
            raise ValueError(f"Insufficient labelled rows ({n}) after feature engineering.")

        # 70/30 train-test split (time-aware)
        split = int(n * 0.70)
        train = labelled.iloc[:split]
        test  = labelled.iloc[split:]

        X_train = train[feature_cols]; y_train = train["target"]
        X_test  = test[feature_cols];  y_test  = test["target"]

        imputer = SimpleImputer(strategy="median")
        X_train_i = imputer.fit_transform(X_train)
        X_test_i  = imputer.transform(X_test)

        # 5. Quick LightGBM (fast, no tuning for API latency)
        model_name = "LightGBM"
        if HAS_LGB:
            import lightgbm as lgb
            model = build_lgbm({
                "n_estimators": 500, "max_depth": 5,
                "learning_rate": 0.05, "min_child_samples": 20,
            })
            model.fit(
                X_train_i, y_train,
                eval_set=[(X_test_i, y_test)],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
        elif HAS_XGB:
            model_name = "XGBoost"
            model = build_xgboost({"n_estimators": 500, "max_depth": 5})
            model.fit(X_train_i, y_train, eval_set=[(X_test_i, y_test)],
                      early_stopping_rounds=30, verbose=False)
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            model_name = "Logistic Regression"
            sc = StandardScaler()
            X_train_i = sc.fit_transform(X_train_i)
            X_test_i  = sc.transform(X_test_i)
            model = LogisticRegression(max_iter=500, C=0.1, class_weight="balanced")
            model.fit(X_train_i, y_train)

        # 6. Test metrics
        test_proba = model.predict_proba(X_test_i)[:, 1]
        test_pred  = (test_proba > 0.5).astype(int)
        auc = roc_auc_score(y_test, test_proba) if y_test.nunique() > 1 else 0.5
        f1  = f1_score(y_test, test_pred, zero_division=0)

        # 7. Predict LATEST row (most recent trading day with full features)
        latest_row = feat_df[feature_cols].dropna().iloc[[-1]]
        latest_imp = imputer.transform(latest_row)
        latest_prob = float(model.predict_proba(latest_imp)[0, 1])

        direction = "UP" if latest_prob >= 0.5 else "DOWN"
        conf_thresh = self.cfg["confidence_thresh"]
        if latest_prob >= conf_thresh or latest_prob <= (1 - conf_thresh):
            confidence_label = "High"
            signal = "BUY" if direction == "UP" else "SELL"
        elif latest_prob >= 0.52 or latest_prob <= 0.48:
            confidence_label = "Medium"
            signal = "HOLD"
        else:
            confidence_label = "Low"
            signal = "HOLD"

        # 8. Price context
        close_series = clean_df["Close"]
        last_close   = float(close_series.iloc[-1])
        chg_1d = float((close_series.iloc[-1] / close_series.iloc[-2] - 1) * 100)
        chg_5d = float((close_series.iloc[-1] / close_series.iloc[-6] - 1) * 100) if len(close_series) >= 6 else 0.0

        # 9. Chart data (last 180 trading days)
        chart_df = feat_df[["Close", "sma5_vs_sma20", "rsi_14", "vol_21d",
                             "macd_hist", "bb_pct_b"]].copy()
        # Re-attach close price
        chart_df["close"] = feat_df["Close"] if "Close" in feat_df.columns else close_series.values
        chart_tail = chart_df.tail(180).copy()
        chart_tail.index = chart_tail.index.astype(str)
        chart_data = []
        for date_str, row in chart_tail.iterrows():
            entry = {"date": str(date_str)[:10]}
            for col in chart_tail.columns:
                v = row[col]
                entry[col] = None if pd.isna(v) else round(float(v), 6)
            chart_data.append(entry)

        # 10. Feature importance
        top_features = []
        if hasattr(model, "feature_importances_"):
            imp_series = pd.Series(model.feature_importances_, index=feature_cols)
            for fname, imp_val in imp_series.nlargest(10).items():
                top_features.append({"name": fname, "importance": round(float(imp_val), 4)})

        return PredictionResult(
            ticker           = ticker_info.ticker,
            company_name     = ticker_info.name,
            sector           = ticker_info.sector,
            market           = ticker_info.market,
            currency         = ticker_info.currency,
            direction        = direction,
            probability      = round(latest_prob, 4),
            confidence_label = confidence_label,
            signal           = signal,
            horizon_days     = self.cfg["return_horizon"],
            last_close       = round(last_close, 4),
            price_change_1d  = round(chg_1d, 2),
            price_change_5d  = round(chg_5d, 2),
            model_auc        = round(auc, 4),
            model_f1         = round(f1, 4),
            model_name       = model_name,
            chart_data       = chart_data,
            top_features     = top_features,
            data_quality     = dq,
            data_period_days = len(clean_df),
            fetch_timestamp  = datetime.datetime.utcnow().isoformat() + "Z",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Saved-model predictor (optional, for production with pre-trained models)
# ─────────────────────────────────────────────────────────────────────────────

class SavedModelPredictor:
    """
    Loads pre-trained artefacts from models/<TICKER>/ for fast inference.
    Falls back to OnTheFlyPredictor if no saved model exists.
    """

    def __init__(self, models_dir: str = "models/", cfg: Optional[dict] = None):
        self.models_dir = Path(models_dir)
        self.otf = OnTheFlyPredictor(cfg)

    def predict(self, query: str) -> PredictionResult:
        from backend.data_ingestion import TickerResolver
        resolver = TickerResolver()
        try:
            info = resolver.resolve(query)
            ticker_dir = self.models_dir / info.ticker.replace(".", "_")
            if not ticker_dir.exists():
                logger.info("No saved model for %s, using on-the-fly.", info.ticker)
                return self.otf.predict(query)
        except Exception:
            return self.otf.predict(query)

        # Load artefacts
        with open(ticker_dir / "imputer.pkl", "rb") as f:
            imputer = pickle.load(f)
        with open(ticker_dir / "feature_cols.json") as f:
            feature_cols = json.load(f)
        with open(ticker_dir / "meta.json") as f:
            meta = json.load(f)

        best = meta["best_model"]
        with open(ticker_dir / f"{best}_model.pkl", "rb") as f:
            model = pickle.load(f)

        # Still fetch fresh data for latest prediction
        _, raw_df = resolve_and_fetch(query)
        proc = DataPreprocessor()
        clean_df = proc.run(raw_df)
        fe = FeatureEngineer()
        feat_df, _ = fe.build(clean_df)

        latest_row = feat_df[feature_cols].dropna().iloc[[-1]]
        latest_imp = imputer.transform(latest_row)
        prob = float(model.predict_proba(latest_imp)[0, 1])

        # Build minimal result (reuse otf for full chart data)
        full = self.otf.predict(query)
        full.probability = round(prob, 4)
        full.direction = "UP" if prob >= 0.5 else "DOWN"
        full.model_name = f"{best} (saved)"
        return full
