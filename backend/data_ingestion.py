"""
data_ingestion.py
-----------------
Dynamic company-name → ticker resolution and real-time OHLCV ingestion via yfinance.
No CSV files. No static mappings. Every query fetches fresh data.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TickerInfo:
    ticker: str
    name: str
    sector: str
    market: str
    currency: str
    exchange: str


# ---------------------------------------------------------------------------
# Ticker resolution
# ---------------------------------------------------------------------------

class TickerResolver:
    """
    Resolves a free-text query (company name OR ticker symbol) to a valid
    yfinance ticker at runtime. No hard-coded dictionaries.

    Strategy
    --------
    1. If the input looks like a raw ticker (all-caps, optional .NS/.HK/.L
       suffix, ≤ 7 chars) → validate it directly via yfinance.
    2. Otherwise use yfinance's search endpoint (Ticker.search is not public,
       so we call the unofficial Yahoo Finance search API) to resolve the name.
    3. Fall back to a fuzzy suffix scan for Indian / Hong-Kong / European
       tickers when a plain US lookup fails.
    """

    EXCHANGE_SUFFIXES = {
        "india": ".NS",
        "nse": ".NS",
        "bse": ".BO",
        "hong kong": ".HK",
        "hkex": ".HK",
        "london": ".L",
        "lse": ".L",
        "frankfurt": ".DE",
        "xetra": ".DE",
        "paris": ".PA",
        "euronext": ".AS",
        "switzerland": ".SW",
        "milan": ".MI",
    }

    def resolve(self, query: str) -> TickerInfo:
        """
        Main entry point. Raises ValueError if resolution fails.
        """
        query = query.strip()
        logger.info("Resolving query: '%s'", query)

        # 1. Looks like an explicit ticker?
        if self._looks_like_ticker(query):
            info = self._validate_ticker(query.upper())
            if info:
                return info

        # 2. Try Yahoo Finance search
        info = self._search_yahoo(query)
        if info:
            return info

        raise ValueError(
            f"Could not resolve '{query}' to a valid stock ticker. "
            "Try using the exact ticker symbol (e.g. AAPL, RELIANCE.NS)."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_ticker(s: str) -> bool:
        """Heuristic: all caps, 1-7 chars, optional market suffix."""
        return bool(re.match(r'^[A-Z0-9]{1,7}(\.[A-Z]{1,3})?$', s.upper()))

    def _validate_ticker(self, ticker: str) -> Optional[TickerInfo]:
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info
            # fast_info always returns a dict-like; check for a price
            price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
            if price is None:
                return None
            full = t.info
            return TickerInfo(
                ticker=ticker,
                name=full.get("longName") or full.get("shortName") or ticker,
                sector=full.get("sector") or full.get("industry") or "Unknown",
                market=self._infer_market(ticker, full),
                currency=full.get("currency", "USD"),
                exchange=full.get("exchange") or full.get("fullExchangeName") or "Unknown",
            )
        except Exception as e:
            logger.debug("Ticker validation failed for %s: %s", ticker, e)
            return None

    def _search_yahoo(self, query: str) -> Optional[TickerInfo]:
        """Use yfinance search to find the best matching ticker."""
        try:
            results = yf.Search(query, max_results=5)
            quotes = results.quotes
            if not quotes:
                return None
            # Pick the first equity result
            for q in quotes:
                q_type = q.get("quoteType", "")
                if q_type in ("EQUITY", "ETF"):
                    ticker = q.get("symbol", "")
                    if ticker:
                        info = self._validate_ticker(ticker)
                        if info:
                            return info
        except Exception as e:
            logger.debug("Yahoo search failed for '%s': %s", query, e)
        return None

    @staticmethod
    def _infer_market(ticker: str, info: dict) -> str:
        exchange = (info.get("exchange") or "").upper()
        if ".NS" in ticker or ".BO" in ticker or exchange in ("NSI", "BSE"):
            return "India"
        if ".HK" in ticker or exchange in ("HKG",):
            return "China/HK"
        if any(s in ticker for s in (".L", ".DE", ".PA", ".AS", ".SW", ".MI")):
            return "Europe"
        return "US"


# ---------------------------------------------------------------------------
# Real-time data fetcher
# ---------------------------------------------------------------------------

class StockDataFetcher:
    """
    Fetches fresh OHLCV data for a resolved ticker at runtime.
    Returns a validated, clean DataFrame.
    """

    MIN_ROWS = 120  # need at least ~6 months for meaningful features

    def __init__(self, period: str = "2y", interval: str = "1d"):
        self.period = period
        self.interval = interval

    def fetch(self, ticker: str) -> pd.DataFrame:
        """
        Download OHLCV data for *ticker*. Returns clean DataFrame with
        columns [Open, High, Low, Close, Volume] indexed by Date (UTC).
        Raises ValueError on insufficient data.
        """
        logger.info("Fetching %s | period=%s interval=%s", ticker, self.period, self.interval)
        try:
            raw = yf.download(
                ticker,
                period=self.period,
                interval=self.interval,
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            raise ValueError(f"yfinance download failed for {ticker}: {e}") from e

        if raw.empty:
            raise ValueError(f"No data returned for {ticker}.")

        # Flatten multi-index columns (yfinance sometimes returns them)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Standardise
        raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        raw.index = pd.to_datetime(raw.index, utc=True)
        raw.index.name = "Date"
        raw = raw.dropna(subset=["Close"])

        if len(raw) < self.MIN_ROWS:
            raise ValueError(
                f"Only {len(raw)} trading days available for {ticker}. "
                f"Need at least {self.MIN_ROWS}."
            )

        logger.info("Fetched %d rows for %s", len(raw), ticker)
        return raw


# ---------------------------------------------------------------------------
# Convenience wrapper used by the API layer
# ---------------------------------------------------------------------------

_resolver = TickerResolver()
_fetcher = StockDataFetcher()


def resolve_and_fetch(query: str, period: str = "2y") -> tuple[TickerInfo, pd.DataFrame]:
    """
    One-call entry point:
        query  → TickerResolver → validated ticker
               → StockDataFetcher → OHLCV DataFrame
    Returns (TickerInfo, DataFrame).
    """
    global _fetcher
    if _fetcher.period != period:
        _fetcher = StockDataFetcher(period=period)

    ticker_info = _resolver.resolve(query)
    df = _fetcher.fetch(ticker_info.ticker)
    return ticker_info, df
