"""
feature_engineering.py
-----------------------
Computes 40+ technical & statistical features from clean OHLCV data.
Mirrors the full feature set from the original research notebook.

Feature groups
--------------
  price_action   : OHLC ratios, gaps, body size
  momentum       : Returns at 1/3/5/10/21 day horizons
  volatility     : Rolling std, ATR, Garman-Klass
  ma_signals     : SMA/EMA crosses (5/10/20/50/200)
  rsi            : RSI-14, RSI-7
  macd           : MACD line, signal, histogram
  bollinger      : %B, bandwidth
  volume         : Volume z-score, OBV, VWAP ratio
  regime         : 252-day vol regime, trend strength
  target         : 5-day forward return binary label
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()

def _ema(s: pd.Series, w: int) -> pd.Series:
    return s.ewm(span=w, adjust=False).mean()

def _rsi(close: pd.Series, w: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=w - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=w - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _true_range(df: pd.DataFrame) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)

def _atr(df: pd.DataFrame, w: int = 14) -> pd.Series:
    return _true_range(df).rolling(w, min_periods=w).mean()


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

class FeatureEngineer:

    def __init__(
        self,
        return_horizon: int = 5,
        target_threshold: float = 0.005,
        vol_window: int = 252,
    ):
        self.return_horizon = return_horizon
        self.target_threshold = target_threshold
        self.vol_window = vol_window

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self, df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """
        Computes all feature columns and the classification target.

        Returns
        -------
        (feature_df, feature_col_names)
            feature_df : DataFrame with all feature columns + 'target'
            feature_col_names : list of column names for model input (no target)
        """
        df = df.copy()
        close = df["Close"]

        # ── Price action ──────────────────────────────────────────────
        df["body_size"]     = (df["Close"] - df["Open"]).abs() / df["Open"]
        df["upper_wick"]    = (df["High"] - df[["Open","Close"]].max(axis=1)) / df["Open"]
        df["lower_wick"]    = (df[["Open","Close"]].min(axis=1) - df["Low"]) / df["Open"]
        df["gap"]           = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
        df["hl_range"]      = (df["High"] - df["Low"]) / df["Open"]

        # ── Returns ───────────────────────────────────────────────────
        for lag in [1, 2, 3, 5, 10, 21]:
            df[f"ret_{lag}d"] = close.pct_change(lag)
        df["log_ret_1d"] = np.log(close / close.shift(1))

        # ── Volatility ────────────────────────────────────────────────
        df["vol_5d"]  = df["log_ret_1d"].rolling(5).std()
        df["vol_21d"] = df["log_ret_1d"].rolling(21).std()
        df["vol_63d"] = df["log_ret_1d"].rolling(63).std()
        df["atr_14"]  = _atr(df, 14)
        df["atr_pct"] = df["atr_14"] / close

        # Garman-Klass volatility estimator (more efficient than close-to-close)
        log_hl = np.log(df["High"] / df["Low"]) ** 2
        log_co = np.log(df["Close"] / df["Open"]) ** 2
        df["gk_vol"] = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(21).mean() ** 0.5

        # Annual vol regime
        df["vol_regime"] = df["log_ret_1d"].rolling(self.vol_window).std() * np.sqrt(252)

        # ── Moving average signals ────────────────────────────────────
        for w in [5, 10, 20, 50, 200]:
            df[f"sma_{w}"] = _sma(close, w)
            df[f"close_vs_sma{w}"] = (close - df[f"sma_{w}"]) / df[f"sma_{w}"]

        df["sma5_vs_sma20"]   = (df["sma_5"]  - df["sma_20"])  / df["sma_20"]
        df["sma20_vs_sma50"]  = (df["sma_20"] - df["sma_50"])  / df["sma_50"]
        df["sma50_vs_sma200"] = (df["sma_50"] - df["sma_200"]) / df["sma_200"]

        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        df["ema_cross_5_20"] = (_ema(close, 5) - _ema(close, 20)) / close

        # ── RSI ───────────────────────────────────────────────────────
        df["rsi_14"] = _rsi(close, 14)
        df["rsi_7"]  = _rsi(close, 7)
        df["rsi_21"] = _rsi(close, 21)
        df["rsi_14_lag1"] = df["rsi_14"].shift(1)
        df["rsi_divergence"] = df["rsi_14"] - df["rsi_14_lag1"]

        # ── MACD ──────────────────────────────────────────────────────
        macd_line    = ema12 - ema26
        macd_signal  = _ema(macd_line, 9)
        df["macd_line"]    = macd_line / close
        df["macd_signal"]  = macd_signal / close
        df["macd_hist"]    = (macd_line - macd_signal) / close
        df["macd_cross"]   = (df["macd_line"] - df["macd_signal"]).apply(np.sign)

        # ── Bollinger Bands ───────────────────────────────────────────
        bb_mid = _sma(close, 20)
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df["bb_pct_b"]   = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
        df["bb_width"]   = (bb_upper - bb_lower) / bb_mid
        df["close_vs_bbmid"] = (close - bb_mid) / bb_mid

        # ── Volume ────────────────────────────────────────────────────
        vol_mean = df["Volume"].rolling(20).mean()
        vol_std  = df["Volume"].rolling(20).std()
        df["vol_zscore"]     = (df["Volume"] - vol_mean) / (vol_std + 1e-9)
        df["vol_ratio_20d"]  = df["Volume"] / (vol_mean + 1e-9)
        df["vol_ratio_5d"]   = df["Volume"] / (df["Volume"].rolling(5).mean() + 1e-9)

        # On-balance volume normalised
        obv = (np.sign(df["log_ret_1d"]) * df["Volume"]).cumsum()
        obv_std = obv.rolling(20).std()
        df["obv_slope"] = obv.diff(5) / (obv_std + 1e-9)

        # VWAP approximation (daily)
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        df["vwap_ratio"] = close / (
            (typical_price * df["Volume"]).rolling(20).sum()
            / (df["Volume"].rolling(20).sum() + 1e-9)
            + 1e-9
        )

        # ── Trend strength ────────────────────────────────────────────
        df["trend_strength_20"] = (close - close.shift(20)).abs() / (
            close.diff().abs().rolling(20).sum() + 1e-9
        )

        # ── Skewness / kurtosis ───────────────────────────────────────
        df["skew_21d"]  = df["log_ret_1d"].rolling(21).skew()
        df["kurt_21d"]  = df["log_ret_1d"].rolling(21).kurt()

        # ── Target ───────────────────────────────────────────────────
        fwd_return = close.pct_change(self.return_horizon).shift(-self.return_horizon)
        df["fwd_return"] = fwd_return
        df["target"] = (fwd_return > self.target_threshold).astype(int)

        # ── Drop intermediate MA columns not needed as features ───────
        drop_cols = [f"sma_{w}" for w in [5, 10, 20, 50, 200]]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Define feature column list (exclude raw OHLCV + target + index metadata)
        exclude = {"Open","High","Low","Close","Volume","log_return",
                   "return_extreme","fwd_return","target"}
        feature_cols = [c for c in df.columns if c not in exclude]

        logger.info("Feature engineering: %d features, %d rows", len(feature_cols), len(df))
        return df, feature_cols


def get_feature_groups() -> dict[str, list[str]]:
    """Returns human-readable group → feature name mapping for UI display."""
    return {
        "price_action": ["body_size","upper_wick","lower_wick","gap","hl_range"],
        "momentum":     ["ret_1d","ret_2d","ret_3d","ret_5d","ret_10d","ret_21d","log_ret_1d"],
        "volatility":   ["vol_5d","vol_21d","vol_63d","atr_14","atr_pct","gk_vol","vol_regime"],
        "ma_signals":   [f"close_vs_sma{w}" for w in [5,10,20,50,200]]
                       + ["sma5_vs_sma20","sma20_vs_sma50","sma50_vs_sma200","ema_cross_5_20"],
        "rsi":          ["rsi_14","rsi_7","rsi_21","rsi_divergence"],
        "macd":         ["macd_line","macd_signal","macd_hist","macd_cross"],
        "bollinger":    ["bb_pct_b","bb_width","close_vs_bbmid"],
        "volume":       ["vol_zscore","vol_ratio_20d","vol_ratio_5d","obv_slope","vwap_ratio"],
        "regime":       ["trend_strength_20","skew_21d","kurt_21d"],
    }
