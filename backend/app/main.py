from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .backtest.engine import run_backtest
from .data import market_data as md
from .data import synthetic_fallback
from .ml import event_matcher
from .ml.regime import REGIME_COLORS, REGIME_LABELS, current_regime, detect_regimes
from .schemas import BacktestRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("apo")

app = FastAPI(title="Adaptive Portfolio Optimizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_backtest_cache: dict[str, dict] = {}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/case-studies")
def case_studies():
    return list(config.CASE_STUDIES.values())


@app.get("/api/assets/default")
def default_assets():
    return {"assets": config.DEFAULT_ASSETS}


@app.get("/api/market/live")
def live_market():
    quotes = md.get_live_quotes(config.LIVE_TICKERS)
    return {"quotes": quotes}


@app.get("/api/regimes/current")
def regimes_current(symbol: str = "SPY", lookback_days: int = 900):
    try:
        hist = md.get_history(symbol)
        prices = hist["adjclose"].tail(lookback_days)
        result = current_regime(prices)
        result["symbol"] = symbol
        result["label_colors"] = REGIME_COLORS
        return result
    except Exception as exc:
        logger.exception("regimes_current failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/regimes/history")
def regimes_history(symbol: str = "SPY", start: str | None = None, end: str | None = None):
    try:
        hist = md.get_history(symbol, start=start, end=end)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol} in range")
        result = detect_regimes(hist["adjclose"])
        return {
            "symbol": symbol,
            "dates": [d.strftime("%Y-%m-%d") for d in result.dates],
            "prices": [round(float(p), 4) for p in hist["adjclose"].reindex(result.dates)],
            "regime": result.state_key.tolist(),
            "probabilities": {
                col: [round(float(v), 4) for v in result.probabilities[col]] for col in result.probabilities.columns
            },
            "state_stats": result.state_stats,
            "labels": REGIME_LABELS,
            "colors": REGIME_COLORS,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("regimes_history failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/regimes/compare")
def regimes_compare(
    symbol: str = "SPY", start: str | None = None, end: str | None = None,
    event_id: str | None = None,
):
    if not start or not end:
        raise HTTPException(status_code=400, detail="start and end are required")
    try:
        return event_matcher.match_window(symbol, start, end, target_event=event_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except md.MarketDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("regimes_compare failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    case = None
    if req.case_study:
        case = config.CASE_STUDIES.get(req.case_study)
        if not case:
            raise HTTPException(status_code=400, detail=f"Unknown case_study '{req.case_study}'")

    assets = req.assets or (case["assets"] if case else config.DEFAULT_ASSETS)
    start = req.start or (case["start"] if case else None)
    end = req.end or (case["end"] if case else None)
    split_date = case["crisis_start"] if case else None
    primary_symbol = req.primary_symbol or assets[0]

    cache_payload = {
        "assets": assets, "start": start, "end": end, "primary_symbol": primary_symbol,
        "split_date": split_date, "rebalance_freq": req.rebalance_freq, "cost_bps": req.cost_bps,
        "es_generations": req.es_generations, "es_population": req.es_population,
    }
    key = _cache_key(cache_payload)
    if key in _backtest_cache:
        return _backtest_cache[key]

    try:
        try:
            panel = md.get_multi_history(assets, start=start, end=end)
        except md.MarketDataError:
            if not case:
                raise
            logger.warning("Live data unavailable, using synthetic fallback for case_study=%s", case["id"])
            dates = pd.bdate_range(case["start"], case["end"])
            panel = synthetic_fallback.generate_case_study_panel(case["id"], dates)
        panel = panel.dropna()
        result = run_backtest(
            panel, primary_symbol=primary_symbol, split_date=split_date,
            rebalance_freq=req.rebalance_freq, cost_bps=req.cost_bps,
            es_generations=req.es_generations, es_population=req.es_population,
        )
        result["case_study"] = case["id"] if case else None
        result["data_notes"] = panel.attrs.get("notes", {})
        result["synthetic_data"] = bool(panel.attrs.get("synthetic", False))
        if case:
            result["crisis_window"] = {"start": case["crisis_start"], "end": case["crisis_end"]}
            result["description"] = case["description"]
        _backtest_cache[key] = result
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except md.MarketDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("backtest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
