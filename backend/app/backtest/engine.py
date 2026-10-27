"""
Backtest engine: runs Equal-Weight, Mean-Variance, and the Adaptive (ES-RL)
allocator over the same historical window, on identical rebalance dates, and
reports comparable performance metrics (Sharpe, max drawdown, cumulative
return) -- directly mirroring the "Benchmark against static strategies"
objective in the project report.

Methodology to avoid lookahead bias / overfitting:
  - Mean-Variance uses only a trailing rolling window of returns at each
    rebalance (no future data).
  - The Adaptive policy is TRAINED only on the segment before `split_date`
    (pre-crisis / in-sample) and then evaluated walk-forward, with no
    retraining, on the segment from `split_date` onward (crisis + recovery /
    out-of-sample). This is the honest way to demonstrate "the model reacts
    to a regime shift it did not train on."
  - All three strategies rebalance on the exact same calendar dates and pay
    the same per-rebalance transaction cost on turnover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RISK_FREE_RATE_ANNUAL, TRADING_DAYS_PER_YEAR, TRANSACTION_COST_BPS
from ..ml import allocator as alloc
from ..ml.regime import detect_regimes


def _metrics(values: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    # values[t] is portfolio value at day t, starting from an implicit base of 1.0 "before" day 0.
    prev_values = np.concatenate([[1.0], values[:-1]])
    daily_returns = values / prev_values - 1.0

    n_days = len(values)
    years = max(n_days / TRADING_DAYS_PER_YEAR, 1e-6)
    total_return = values[-1] / values[0] - 1.0
    ann_return = (values[-1] / values[0]) ** (1 / years) - 1.0
    ann_vol = float(np.std(daily_returns) * np.sqrt(TRADING_DAYS_PER_YEAR))
    rf_daily = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
    excess = daily_returns - rf_daily
    sharpe = float(np.mean(excess) / np.std(daily_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)) if np.std(daily_returns) > 1e-9 else 0.0

    peak = np.maximum.accumulate(values)
    drawdown = values / peak - 1.0
    max_dd = float(drawdown.min())
    calmar = float(ann_return / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0

    return {
        "total_return": round(float(total_return), 4),
        "annualized_return": round(float(ann_return), 4),
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "calmar_ratio": round(calmar, 3),
    }


def _rebalance_indices(n_days: int, freq: int) -> list[int]:
    return list(range(0, n_days, freq))


def _simulate_fixed_or_scheduled(returns: pd.DataFrame, rebalance_idxs: list[int],
                                  weight_fn, cost_bps: float) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """weight_fn(t) -> weights array; called only at rebalance indices t."""
    n_assets = returns.shape[1]
    weights_now = alloc.equal_weights(n_assets)
    value = 1.0
    values = np.empty(len(returns))
    weights_log: dict[int, np.ndarray] = {}
    rb_ptr = 0

    for t in range(len(returns)):
        if rb_ptr < len(rebalance_idxs) and t == rebalance_idxs[rb_ptr]:
            new_weights = weight_fn(t)
            turnover = float(np.abs(new_weights - weights_now).sum())
            value *= (1 - cost_bps / 10000.0 * turnover)
            weights_now = new_weights
            weights_log[t] = new_weights
            rb_ptr += 1
        day_ret = float(weights_now @ returns.iloc[t].values)
        value *= (1 + day_ret)
        values[t] = value
    return values, weights_log


def run_backtest(prices: pd.DataFrame, primary_symbol: str, split_date: str | None = None,
                  rebalance_freq: int = 21, cost_bps: float = TRANSACTION_COST_BPS,
                  es_generations: int = 40, es_population: int = 32) -> dict:
    """
    `prices` = adjusted-close panel (date index, columns = tickers), already
    limited to dates where every asset has data. `primary_symbol` is the
    market proxy used for regime detection (e.g. 'SPY').
    """
    prices = prices.dropna()
    if len(prices) < 60:
        raise ValueError("Not enough overlapping trading history across the selected assets/date range")

    returns = prices.pct_change().dropna()
    prices_aligned = prices.loc[returns.index]

    regime_result = detect_regimes(prices_aligned[primary_symbol])
    regime_proba = regime_result.probabilities.reindex(returns.index).ffill().bfill()
    regime_series = pd.Series(regime_result.state_key, index=regime_result.dates).reindex(returns.index).ffill().bfill()

    n_days = len(returns)
    rebalance_idxs = _rebalance_indices(n_days, rebalance_freq)

    if split_date is not None:
        split_ts = pd.Timestamp(split_date)
        split_pos = int(returns.index.searchsorted(split_ts))
        split_pos = min(max(split_pos, 30), n_days - 30) if n_days > 90 else max(1, n_days // 2)
    else:
        split_pos = int(n_days * 0.6)

    train_returns = returns.iloc[:split_pos]
    train_proba = regime_proba.iloc[:split_pos]
    train_rebalance_idxs = [i for i in rebalance_idxs if i < split_pos] or [0]

    # ---- Equal Weight ----
    ew_values, ew_weights = _simulate_fixed_or_scheduled(
        returns, rebalance_idxs, lambda t: alloc.equal_weights(returns.shape[1]), cost_bps)

    # ---- Mean-Variance (rolling window, causal only) ----
    mv_window = max(rebalance_freq * 3, 60)

    def mv_weight_fn(t):
        start = max(0, t - mv_window)
        window = returns.iloc[start:t] if t > start else returns.iloc[0:1]
        if len(window) < 10:
            return alloc.equal_weights(returns.shape[1])
        return alloc.mean_variance_weights(window)

    mv_values, mv_weights = _simulate_fixed_or_scheduled(returns, rebalance_idxs, mv_weight_fn, cost_bps)

    # ---- Adaptive (ES-trained neural policy, trained ONLY on train segment) ----
    policy = alloc.train_adaptive_policy(
        train_returns, train_proba, train_rebalance_idxs, cost_bps=cost_bps,
        generations=es_generations, population=es_population,
    )

    def ad_weight_fn(t):
        state = alloc.build_state_features(returns, regime_proba, t)
        return policy.act(state)

    ad_values, ad_weights = _simulate_fixed_or_scheduled(returns, rebalance_idxs, ad_weight_fn, cost_bps)

    dates = returns.index

    def series_payload(values, weights_log):
        return {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "values": [round(float(v), 6) for v in values],
            "weights": {
                dates[idx].strftime("%Y-%m-%d"): {sym: round(float(w), 4) for sym, w in zip(prices.columns, wts)}
                for idx, wts in weights_log.items()
            },
        }

    metrics_full = {
        "equal_weight": _metrics(ew_values, dates),
        "mean_variance": _metrics(mv_values, dates),
        "adaptive": _metrics(ad_values, dates),
    }
    split_date_str = dates[split_pos].strftime("%Y-%m-%d") if split_pos < len(dates) else dates[-1].strftime("%Y-%m-%d")
    metrics_test_period = {}
    if split_pos < n_days - 5:
        metrics_test_period = {
            "equal_weight": _metrics(ew_values[split_pos:] / ew_values[split_pos], dates[split_pos:]),
            "mean_variance": _metrics(mv_values[split_pos:] / mv_values[split_pos], dates[split_pos:]),
            "adaptive": _metrics(ad_values[split_pos:] / ad_values[split_pos], dates[split_pos:]),
        }

    regime_payload = {
        "dates": [d.strftime("%Y-%m-%d") for d in regime_series.index],
        "regime": regime_series.tolist(),
    }

    return {
        "assets": list(prices.columns),
        "primary_symbol": primary_symbol,
        "date_range": {"start": dates[0].strftime("%Y-%m-%d"), "end": dates[-1].strftime("%Y-%m-%d")},
        "train_test_split_date": split_date_str,
        "strategies": {
            "equal_weight": series_payload(ew_values, ew_weights),
            "mean_variance": series_payload(mv_values, mv_weights),
            "adaptive": series_payload(ad_values, ad_weights),
        },
        "metrics_full_period": metrics_full,
        "metrics_test_period": metrics_test_period,
        "regimes": regime_payload,
    }
