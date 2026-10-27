"""
Thin client for Yahoo Finance's public chart JSON endpoint.

We talk to the endpoint directly (rather than through the `yfinance`
library) so we control retry/backoff and can cache raw responses to disk --
this keeps the app usable even when Yahoo rate-limits a burst of requests
(common on shared/cloud IPs) and avoids refetching the same history on every
request.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_USER_AGENT = "Mozilla/5.0"

_CACHE_TTL_SECONDS = 15 * 60  # refresh cached history at most every 15 min


class MarketDataError(RuntimeError):
    pass


def _cache_path(symbol: str, interval: str) -> Path:
    safe = symbol.replace("=", "_").replace("^", "IDX_")
    return CACHE_DIR / f"{safe}_{interval}.csv"


def _http_get_json(url: str, timeout: int = 15) -> dict:
    """
    Fetch JSON via curl rather than `requests`/urllib. Yahoo's edge appears to
    fingerprint the HTTP client (TLS/header ordering) and rate-limits the
    Python `requests` stack far more aggressively than plain curl, even from
    the same IP -- shelling out to curl (present on macOS/Linux by default)
    sidesteps that.
    """
    try:
        proc = subprocess.run(
            ["curl", "-s", "-H", f"User-Agent: {_USER_AGENT}", "--max-time", str(timeout), url],
            capture_output=True, timeout=timeout + 5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise MarketDataError(f"HTTP request failed: {exc}") from exc
    if proc.returncode != 0 or not proc.stdout:
        raise MarketDataError(f"curl failed (code {proc.returncode}): {proc.stderr.decode(errors='ignore')[:200]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MarketDataError(f"Non-JSON response: {proc.stdout[:200]!r}") from exc


def _fetch_raw(symbol: str, start: Optional[str] = None, end: Optional[str] = None,
                interval: str = "1d", retries: int = 2) -> dict:
    """
    Fetch one symbol's chart data between `start` and `end` ('YYYY-MM-DD').
    Uses explicit period1/period2 unix timestamps rather than Yahoo's `range`
    shorthand -- `range=max` silently coerces interval=1d down to ~monthly
    bars for long histories, while explicit timestamps always return true
    daily bars.
    """
    if start:
        period1 = int(pd.Timestamp(start).timestamp())
    else:
        period1 = int(pd.Timestamp("1970-01-01").timestamp())
    period2 = int((pd.Timestamp(end) if end else pd.Timestamp.utcnow()).timestamp()) + 86400

    params = {"period1": period1, "period2": period2, "interval": interval, "includePrePost": "false"}
    url = _BASE_URL.format(symbol=symbol) + "?" + urlencode(params)

    last_error = None
    for attempt in range(retries):
        try:
            payload = _http_get_json(url)
        except MarketDataError as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
            continue
        result = payload.get("chart", {}).get("result")
        if result:
            return result[0]
        err = payload.get("chart", {}).get("error")
        last_error = MarketDataError(f"Yahoo returned no data for {symbol}: {err}")
        time.sleep(1.5 * (attempt + 1))
    raise last_error or MarketDataError(f"Failed to fetch {symbol}")


def _parse_chart_result(result: dict) -> pd.DataFrame:
    timestamps = result.get("timestamp")
    if not timestamps:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "adjclose"])
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
    )
    adj = result["indicators"].get("adjclose")
    df["adjclose"] = adj[0]["adjclose"] if adj else df["close"]
    df.index.name = "date"
    return df.dropna(subset=["close"])


def get_history(symbol: str, start: Optional[str] = None, end: Optional[str] = None,
                 interval: str = "1d", force_refresh: bool = False) -> pd.DataFrame:
    """
    Full daily history for `symbol`, sliced to [start, end] (inclusive,
    'YYYY-MM-DD' strings). Uses an on-disk cache so repeated requests (e.g.
    across the three case studies) don't keep re-hitting Yahoo.
    """
    path = _cache_path(symbol, interval)
    df: Optional[pd.DataFrame] = None

    if path.exists() and not force_refresh:
        age = time.time() - path.stat().st_mtime
        try:
            cached = pd.read_csv(path, index_col="date", parse_dates=["date"])
        except Exception:
            cached = None
        if cached is not None and not cached.empty:
            need_start = pd.Timestamp(start) if start else None
            have_enough_history = need_start is None or cached.index.min() <= need_start
            fresh_enough = age < _CACHE_TTL_SECONDS
            covers_recent = (pd.Timestamp.utcnow().tz_localize(None) - cached.index.max()).days <= 5
            if have_enough_history and (fresh_enough or covers_recent):
                df = cached

    if df is None:
        # Always fetch the full available history (from the symbol's listing
        # date) so the cache file can serve any future date-range request
        # without refetching -- cheap since it's cached to disk afterwards.
        raw = _fetch_raw(symbol, start="1970-01-01", end=None, interval=interval)
        df = _parse_chart_result(raw)
        if df.empty:
            raise MarketDataError(f"No historical data returned for {symbol}")
        df.to_csv(path, index_label="date")

    sliced = df
    if start:
        sliced = sliced[sliced.index >= pd.Timestamp(start)]
    if end:
        sliced = sliced[sliced.index <= pd.Timestamp(end)]
    return sliced.copy()


def get_multi_history(symbols: list[str], start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Adjusted-close price panel (columns = symbols) aligned on shared trading dates."""
    frames = {}
    notes = {}
    for sym in symbols:
        try:
            h = get_history(sym, start=start, end=end)
        except MarketDataError as exc:
            notes[sym] = str(exc)
            continue
        if h.empty:
            notes[sym] = "no data in requested range"
            continue
        frames[sym] = h["adjclose"]
        if start and h.index.min() > pd.Timestamp(start):
            notes[sym] = f"data only available from {h.index.min().date()}"
    if not frames:
        raise MarketDataError(f"No data available for any of {symbols}")
    panel = pd.DataFrame(frames).sort_index().dropna(how="all")
    panel.attrs["notes"] = notes
    return panel


def get_live_quotes(symbols: list[str]) -> list[dict]:
    quotes = []
    recent_start = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    for sym in symbols:
        try:
            raw = _fetch_raw(sym, start=recent_start, end=None, interval="1d", retries=2)
            meta = raw.get("meta", {})
            price = meta.get("regularMarketPrice")
            prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
            change_pct = meta.get("regularMarketChangePercent")
            if change_pct is None and price and prev_close:
                change_pct = (price / prev_close - 1) * 100
            quotes.append({
                "symbol": sym,
                "name": meta.get("shortName", sym),
                "price": price,
                "change_pct": change_pct,
                "as_of": datetime.fromtimestamp(meta.get("regularMarketTime", time.time()), tz=timezone.utc).isoformat(),
                "currency": meta.get("currency"),
            })
        except MarketDataError as exc:
            quotes.append({"symbol": sym, "error": str(exc)})
    return quotes
