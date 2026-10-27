"""
Unsupervised market-regime detection via a Gaussian Hidden Markov Model.

We fit a 3-state HMM on daily [return, rolling volatility, rolling drawdown]
features of a broad market proxy (default: the first asset in the universe,
e.g. SPY). HMM state indices are arbitrary on each fit, so after fitting we
re-label the 3 states by their empirical statistics:

  - the state with the highest mean volatility -> "crisis"   (high-vol regime)
  - of the remaining two, the one with higher mean return -> "bull"
  - the last one -> "bear"

This mirrors the "regime-aware" allocation idea from the literature review
(Transformer-PPO / LSTM regime-detection papers) with a much lighter,
fully-local, dependency-light model appropriate for an interactive demo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

REGIME_LABELS = {"bull": "Bull / Calm", "bear": "Bear / Declining", "crisis": "Crisis / High-Volatility"}
REGIME_COLORS = {"bull": "#16a34a", "bear": "#eab308", "crisis": "#dc2626"}

VOL_WINDOW = 20
DRAWDOWN_WINDOW = 60


@dataclass
class RegimeResult:
    dates: pd.DatetimeIndex
    state_key: np.ndarray       # array of "bull"/"bear"/"crisis" strings
    probabilities: pd.DataFrame  # columns = bull/bear/crisis probabilities
    state_stats: dict


def _build_features(prices: pd.Series) -> pd.DataFrame:
    returns = np.log(prices).diff()
    vol = returns.rolling(VOL_WINDOW).std() * np.sqrt(252)
    roll_max = prices.rolling(DRAWDOWN_WINDOW, min_periods=5).max()
    drawdown = prices / roll_max - 1.0
    feats = pd.DataFrame({"return": returns, "volatility": vol, "drawdown": drawdown}).dropna()
    return feats


def detect_regimes(prices: pd.Series, n_states: int = 3, random_state: int = 7) -> RegimeResult:
    """
    `prices` must be a date-indexed Series of a single market proxy
    (e.g. adjusted close of SPY). Returns per-day regime labels + posterior
    probabilities for the whole series.
    """
    feats = _build_features(prices)
    if len(feats) < n_states * 10:
        # The stated minimum counts RAW trading days, not feature rows: the
        # rolling vol warm-up eats the first VOL_WINDOW days before a single
        # usable row exists, so quoting the n_states*10 row count alone sent
        # people away thinking a 35-day window should have worked.
        raise ValueError(
            f"Not enough data points to fit a regime model — need about "
            f"{VOL_WINDOW + n_states * 10} trading days "
            f"({VOL_WINDOW} are consumed by the rolling-volatility warm-up, "
            f"leaving the {n_states * 10} the model needs; this window gave {len(feats)})"
        )

    X = feats.values
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=300,
                         random_state=random_state, min_covar=1e-3)
    model.fit(Xs)
    hidden_states = model.predict(Xs)
    proba = model.predict_proba(Xs)

    # Re-label arbitrary HMM state indices by empirical stats.
    stats = []
    for s in range(n_states):
        mask = hidden_states == s
        stats.append({
            "state": s,
            "mean_return": float(feats["return"][mask].mean()) if mask.any() else 0.0,
            "mean_vol": float(feats["volatility"][mask].mean()) if mask.any() else 0.0,
            "count": int(mask.sum()),
        })

    crisis_state = max(stats, key=lambda d: d["mean_vol"])["state"]
    remaining = [d for d in stats if d["state"] != crisis_state]
    bull_state = max(remaining, key=lambda d: d["mean_return"])["state"]
    bear_state = [d["state"] for d in remaining if d["state"] != bull_state][0]

    key_by_state = {crisis_state: "crisis", bull_state: "bull", bear_state: "bear"}
    state_key = np.array([key_by_state[s] for s in hidden_states])

    # key_by_state is a bijection for n_states=3 (one state per regime label),
    # so columns can be renamed directly without collision.
    proba_df = pd.DataFrame(proba, index=feats.index, columns=[key_by_state[s] for s in range(n_states)])
    for key in ("bull", "bear", "crisis"):
        if key not in proba_df.columns:
            proba_df[key] = 0.0

    stats_named = {key_by_state[d["state"]]: d for d in stats}

    return RegimeResult(dates=feats.index, state_key=state_key, probabilities=proba_df, state_stats=stats_named)


def current_regime(prices: pd.Series) -> dict:
    """Convenience wrapper: regime classification + confidence for the most recent day."""
    result = detect_regimes(prices)
    last_date = result.dates[-1]
    last_state = result.state_key[-1]
    last_proba = result.probabilities.iloc[-1].to_dict()
    return {
        "as_of": last_date.strftime("%Y-%m-%d"),
        "regime": last_state,
        "label": REGIME_LABELS[last_state],
        "confidence": round(float(last_proba[last_state]), 4),
        "probabilities": {k: round(float(v), 4) for k, v in last_proba.items()},
    }
