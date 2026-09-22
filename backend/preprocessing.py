"""
preprocessing.py
----------------
Validates and cleans raw OHLCV data before feature engineering.
Handles: missing values, outliers, price-alignment anomalies, zero-volume days.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Cleans raw OHLCV DataFrame returned by StockDataFetcher.

    Steps
    -----
    1. Schema validation  – ensure OHLCV columns exist and are numeric
    2. Zero / negative price removal
    3. Forward-fill short gaps (≤ ffill_limit trading days)
    4. OHLC consistency check (Low ≤ Close ≤ High, etc.)
    5. Outlier winsorisation on log-returns
    6. Zero-volume imputation
    7. Return log-return series for downstream use
    """

    REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, ffill_limit: int = 3, outlier_zscore: float = 5.0):
        self.ffill_limit = ffill_limit
        self.outlier_zscore = outlier_zscore

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
        """
        Full preprocessing pipeline. Returns clean DataFrame.
        Raises ValueError if data is too damaged to proceed.
        """
        df = df.copy()
        tag = f"[{ticker}]" if ticker else ""

        df = self._validate_schema(df, tag)
        df = self._remove_bad_prices(df, tag)
        df = self._ffill_gaps(df, tag)
        df = self._fix_ohlc_consistency(df, tag)
        df = self._winsorise_returns(df, tag)
        df = self._impute_zero_volume(df, tag)
        df = self._add_log_return(df)

        logger.info("%s Preprocessing complete: %d rows", tag, len(df))
        return df

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _validate_schema(self, df: pd.DataFrame, tag: str) -> pd.DataFrame:
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{tag} Missing columns: {missing}")
        df[self.REQUIRED_COLS] = df[self.REQUIRED_COLS].apply(pd.to_numeric, errors="coerce")
        return df

    def _remove_bad_prices(self, df: pd.DataFrame, tag: str) -> pd.DataFrame:
        before = len(df)
        df = df[df["Close"] > 0].copy()
        removed = before - len(df)
        if removed:
            logger.warning("%s Removed %d rows with Close ≤ 0", tag, removed)
        return df

    def _ffill_gaps(self, df: pd.DataFrame, tag: str) -> pd.DataFrame:
        n_na = df["Close"].isna().sum()
        if n_na:
            df = df.ffill(limit=self.ffill_limit)
            remaining = df["Close"].isna().sum()
            df = df.dropna(subset=["Close"])
            logger.info("%s Filled %d NaN closes; dropped %d unfillable", tag, n_na - remaining, remaining)
        return df

    def _fix_ohlc_consistency(self, df: pd.DataFrame, tag: str) -> pd.DataFrame:
        """Clip High/Low to be consistent with Open/Close."""
        df["High"] = df[["Open", "High", "Close"]].max(axis=1)
        df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
        return df

    def _winsorise_returns(self, df: pd.DataFrame, tag: str) -> pd.DataFrame:
        """Winsorise extreme log-return days (likely data errors or splits)."""
        lr = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        mean, std = lr.mean(), lr.std()
        upper = mean + self.outlier_zscore * std
        lower = mean - self.outlier_zscore * std
        extreme = ((lr > upper) | (lr < lower)).sum()
        if extreme:
            logger.warning("%s Winsorising %d extreme return observations", tag, extreme)
        # We don't alter the prices themselves but mark them for feature use
        df["return_extreme"] = 0
        if len(lr):
            df.loc[lr[(lr > upper) | (lr < lower)].index, "return_extreme"] = 1
        return df

    def _impute_zero_volume(self, df: pd.DataFrame, tag: str) -> pd.DataFrame:
        zero_vol = (df["Volume"] == 0).sum()
        if zero_vol:
            df["Volume"] = df["Volume"].replace(0, np.nan)
            df["Volume"] = df["Volume"].ffill().fillna(df["Volume"].median())
            logger.info("%s Imputed %d zero-volume rows", tag, zero_vol)
        return df

    @staticmethod
    def _add_log_return(df: pd.DataFrame) -> pd.DataFrame:
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        return df


# ------------------------------------------------------------------
# Quality report (used by API for transparency)
# ------------------------------------------------------------------

def data_quality_report(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    return {
        "raw_rows": len(raw),
        "clean_rows": len(clean),
        "dropped_rows": len(raw) - len(clean),
        "date_range": {
            "start": str(clean.index.min().date()),
            "end": str(clean.index.max().date()),
        },
        "missing_pct_raw": float(raw["Close"].isna().mean() * 100),
        "zero_volume_raw": int((raw["Volume"] == 0).sum()),
    }
