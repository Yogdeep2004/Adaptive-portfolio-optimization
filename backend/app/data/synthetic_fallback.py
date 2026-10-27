"""
Offline fallback data generator for the three case studies.

This is used ONLY when live Yahoo Finance data cannot be fetched at all
(e.g. no internet, or the shared network is rate-limited) -- results from
this module are always tagged `synthetic: True` so the frontend can show a
clear disclaimer. It exists so a live demo never simply breaks if the
network misbehaves.

Each segment below is calibrated (roughly) to the *real* historical
magnitude of that period for that asset -- e.g. the dot-com QQQ segment is
tuned to lose ~78% peak-to-trough, the COVID crash segment to lose ~34% in
~23 trading days -- via geometric Brownian motion with a shared market
factor (so SPY/QQQ move together and gold/bonds diversify against them,
exactly as they should for the portfolio-optimization story to make sense).
This is illustrative calibration for demo purposes, not a substitute for
the real market data the app uses whenever it's reachable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Each entry: (n_trading_days, {symbol: (mu_annual, sigma_annual, beta_to_market)})
_SCENARIOS: dict[str, list[tuple[int, dict[str, tuple[float, float, float]]]]] = {
    "dotcom": [
        (190, {"SPY": (0.18, 0.16, 0.9), "QQQ": (0.85, 0.45, 0.85), "GC=F": (0.02, 0.13, -0.05)}),
        (650, {"SPY": (-0.22, 0.24, 0.9), "QQQ": (-0.50, 0.50, 0.85), "GC=F": (0.06, 0.14, -0.15)}),
        (310, {"SPY": (0.30, 0.17, 0.85), "QQQ": (0.55, 0.34, 0.8), "GC=F": (0.10, 0.13, -0.1)}),
    ],
    "gfc": [
        (440, {"SPY": (0.12, 0.13, 0.9), "QQQ": (0.15, 0.17, 0.85), "GLD": (0.18, 0.15, -0.05), "TLT": (0.03, 0.07, -0.2)}),
        (355, {"SPY": (-0.48, 0.33, 0.9), "QQQ": (-0.46, 0.37, 0.85), "GLD": (0.08, 0.19, -0.25), "TLT": (0.16, 0.09, -0.35)}),
        (455, {"SPY": (0.38, 0.17, 0.85), "QQQ": (0.48, 0.21, 0.8), "GLD": (0.22, 0.17, -0.05), "TLT": (-0.03, 0.08, -0.1)}),
    ],
    "covid": [
        (285, {"SPY": (0.28, 0.12, 0.9), "QQQ": (0.40, 0.15, 0.85), "GLD": (0.18, 0.12, -0.05), "TLT": (0.14, 0.09, -0.2)}),
        (33, {"SPY": (-3.2, 0.75, 0.9), "QQQ": (-2.7, 0.70, 0.85), "GLD": (-0.1, 0.32, -0.1), "TLT": (0.55, 0.22, -0.4)}),
        (440, {"SPY": (0.45, 0.15, 0.85), "QQQ": (0.55, 0.19, 0.8), "GLD": (0.08, 0.13, -0.05), "TLT": (-0.06, 0.08, -0.1)}),
    ],
}

_START_PRICE = 100.0


def generate_case_study_panel(case_id: str, dates: pd.DatetimeIndex, seed: int = 20260906) -> pd.DataFrame:
    """
    Deterministic (fixed-seed) synthetic adjusted-close panel spanning
    `dates`, stitched together from the scenario's segments (trimmed/padded
    to match len(dates)).
    """
    if case_id not in _SCENARIOS:
        raise ValueError(f"No synthetic fallback scenario for case_study '{case_id}'")
    segments = _SCENARIOS[case_id]
    symbols = sorted({sym for _, seg in segments for sym in seg})
    rng = np.random.default_rng(seed)

    n_total = len(dates)
    seg_lengths = [n for n, _ in segments]
    scale = n_total / sum(seg_lengths)
    seg_lengths = [max(1, round(n * scale)) for n in seg_lengths]
    seg_lengths[-1] += n_total - sum(seg_lengths)  # absorb rounding drift

    log_prices = {sym: [np.log(_START_PRICE)] for sym in symbols}
    for (n_days, params), seg_len in zip(segments, seg_lengths):
        market_shock = rng.normal(0, 1, size=seg_len)
        for sym in symbols:
            mu_annual, sigma_annual, beta = params.get(sym, (0.03, 0.12, 0.0))
            mu_daily = mu_annual / 252
            sigma_daily = sigma_annual / np.sqrt(252)
            idio = rng.normal(0, 1, size=seg_len)
            beta_c = np.clip(beta, -0.98, 0.98)
            shock = beta_c * market_shock + np.sqrt(max(1 - beta_c ** 2, 0)) * idio
            daily_log_ret = (mu_daily - 0.5 * sigma_daily ** 2) + sigma_daily * shock
            path = log_prices[sym][-1] + np.cumsum(daily_log_ret)
            log_prices[sym].extend(path.tolist())

    data = {sym: np.exp(np.array(vals[1:n_total + 1])) for sym, vals in log_prices.items()}
    panel = pd.DataFrame(data, index=dates[:n_total])
    panel.attrs["notes"] = {sym: "SYNTHETIC (calibrated fallback -- live feed unavailable)" for sym in symbols}
    panel.attrs["synthetic"] = True
    return panel
