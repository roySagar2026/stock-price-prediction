"""
train.py
--------
End-to-end training pipeline.
Fetches data for a given ticker (or list), engineers features,
trains LightGBM + XGBoost, optionally tunes with Optuna,
and saves artefacts to models/.

Usage
-----
    python train.py --ticker AAPL
    python train.py --ticker AAPL MSFT TSLA --tune
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score

from backend.data_ingestion import resolve_and_fetch
from backend.feature_engineering import FeatureEngineer
from backend.model import HAS_LGB, HAS_XGB, build_lgbm, build_xgboost
from backend.preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Split helper (time-series aware)
# ─────────────────────────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame, train_pct: float, val_pct: float):
    n = len(df)
    i_train = int(n * train_pct)
    i_val   = int(n * (train_pct + val_pct))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker training
# ─────────────────────────────────────────────────────────────────────────────

def train_single(ticker_query: str, cfg: dict, output_dir: Path, tune: bool = False):
    mcfg = cfg["model"]
    logger.info("=== Training for query: %s ===", ticker_query)

    # 1. Ingest
    ticker_info, raw_df = resolve_and_fetch(ticker_query, period=mcfg["data_period"])
    logger.info("Resolved '%s' → %s (%s)", ticker_query, ticker_info.ticker, ticker_info.name)

    # 2. Preprocess
    proc = DataPreprocessor(ffill_limit=mcfg["ffill_limit"])
    clean_df = proc.run(raw_df, ticker=ticker_info.ticker)

    # 3. Feature engineering
    fe = FeatureEngineer(
        return_horizon=mcfg["return_horizon"],
        target_threshold=mcfg["target_threshold"],
        vol_window=mcfg["vol_window"],
    )
    feat_df, feature_cols = fe.build(clean_df)
    feat_df = feat_df.dropna(subset=["target"])

    # 4. Time-series split
    train, val, test = time_split(feat_df, mcfg["train_pct"], mcfg["val_pct"])
    logger.info("Split — Train: %d | Val: %d | Test: %d", len(train), len(val), len(test))

    X_train = train[feature_cols]; y_train = train["target"]
    X_val   = val[feature_cols];   y_val   = val["target"]
    X_test  = test[feature_cols];  y_test  = test["target"]

    # 5. Impute
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_val_i   = imputer.transform(X_val)
    X_test_i  = imputer.transform(X_test)

    results = {}

    # 6a. LightGBM
    if HAS_LGB:
        import lightgbm as lgb
        if tune:
            lgb_model = _tune_lgbm(X_train_i, y_train, X_val_i, y_val, cfg)
        else:
            lgb_model = build_lgbm(cfg.get("lgbm", {}))
            lgb_model.fit(
                X_train_i, y_train,
                eval_set=[(X_val_i, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
        proba = lgb_model.predict_proba(X_test_i)[:, 1]
        results["lgbm"] = {
            "model": lgb_model,
            "auc": roc_auc_score(y_test, proba),
            "f1":  f1_score(y_test, (proba > 0.5).astype(int)),
        }
        logger.info("LightGBM  AUC=%.4f  F1=%.4f", results["lgbm"]["auc"], results["lgbm"]["f1"])

    # 6b. XGBoost
    if HAS_XGB:
        xgb_model = build_xgboost(cfg.get("xgboost", {}))
        xgb_model.fit(
            X_train_i, y_train,
            eval_set=[(X_val_i, y_val)],
            early_stopping_rounds=50,
            verbose=False,
        )
        proba = xgb_model.predict_proba(X_test_i)[:, 1]
        results["xgb"] = {
            "model": xgb_model,
            "auc": roc_auc_score(y_test, proba),
            "f1":  f1_score(y_test, (proba > 0.5).astype(int)),
        }
        logger.info("XGBoost   AUC=%.4f  F1=%.4f", results["xgb"]["auc"], results["xgb"]["f1"])

    if not results:
        raise RuntimeError("No models trained. Install lightgbm or xgboost.")

    best_name = max(results, key=lambda k: results[k]["auc"])
    best_auc  = results[best_name]["auc"]
    logger.info("Best model: %s  AUC=%.4f", best_name, best_auc)

    # 7. Save artefacts
    ticker_dir = output_dir / ticker_info.ticker.replace(".", "_")
    ticker_dir.mkdir(parents=True, exist_ok=True)

    with open(ticker_dir / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    with open(ticker_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    for name, r in results.items():
        with open(ticker_dir / f"{name}_model.pkl", "wb") as f:
            pickle.dump(r["model"], f)

    meta = {
        "ticker": ticker_info.ticker,
        "name": ticker_info.name,
        "sector": ticker_info.sector,
        "market": ticker_info.market,
        "best_model": best_name,
        "test_auc": best_auc,
        "test_f1":  results[best_name]["f1"],
        "n_features": len(feature_cols),
        "train_rows": len(train),
        "test_rows":  len(test),
        "model_config": cfg["model"],
    }
    with open(ticker_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info("Artefacts saved to %s", ticker_dir)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Optuna tuning (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _tune_lgbm(X_tr, y_tr, X_va, y_va, cfg: dict):
    try:
        import lightgbm as lgb
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("optuna not installed, skipping tuning. pip install optuna")
        return build_lgbm(cfg.get("lgbm", {}))

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 1500),
            "max_depth":         trial.suggest_int("max_depth", 4, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "class_weight": "balanced", "random_state": 42, "verbose": -1, "n_jobs": -1,
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        return roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=40, show_progress_bar=False)
    logger.info("Optuna best AUC: %.4f", study.best_value)

    best = study.best_params
    best.update({"class_weight":"balanced","random_state":42,"verbose":-1,"n_jobs":-1})
    model = lgb.LGBMClassifier(**best)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockPulse training pipeline")
    parser.add_argument("--ticker", nargs="+", required=True,
                        help="Ticker symbol(s) or company name(s)")
    parser.add_argument("--tune", action="store_true", help="Enable Optuna tuning")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="models/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for q in args.ticker:
        try:
            meta = train_single(q, cfg, out, tune=args.tune)
            print(f"✅ {meta['ticker']} ({meta['name']})  AUC={meta['test_auc']:.4f}")
        except Exception as e:
            logger.error("Failed for '%s': %s", q, e)
