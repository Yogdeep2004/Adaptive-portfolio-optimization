"""
Portfolio allocation strategies compared in the backtest:

  1. Equal-Weight        - naive 1/N baseline (rebalanced periodically)
  2. Mean-Variance       - rolling-window Markowitz max-Sharpe tangency
                           portfolio (classic baseline from the literature)
  3. Adaptive Allocator  - a small neural policy (state -> softmax weights)
                           trained with Evolution Strategies (ES), a
                           gradient-free reinforcement learning algorithm
                           (Salimans et al., 2017, "Evolution Strategies as
                           a Scalable Alternative to Reinforcement Learning").

Why ES instead of PPO/DDPG (as in the papers reviewed)?  ES needs no
autodiff framework, trains in seconds on CPU with no extra heavy
dependency, and is a legitimate deep-RL training method -- a good fit for
an interactive local demo. The README explains this trade-off and how to
swap in a PPO agent (e.g. stable-baselines3) for the full research version.

The policy's *state* deliberately mirrors what the report's research gap
calls for: regime probabilities (from the HMM), per-asset momentum and
volatility, and it is optimized against a reward that penalizes both
drawdown and turnover (i.e. transaction costs) rather than raw return.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def equal_weights(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets)


def mean_variance_weights(returns_window: pd.DataFrame, risk_free_daily: float = 0.0) -> np.ndarray:
    """
    Max-Sharpe (tangency) portfolio over a trailing window of daily returns,
    long-only, fully invested. Falls back to equal-weight if the optimizer
    fails or the window is degenerate (e.g. all-zero variance).
    """
    n = returns_window.shape[1]
    mean_ret = returns_window.mean().values
    cov = returns_window.cov().values
    cov += np.eye(n) * 1e-6  # numerical stability

    def neg_sharpe(w):
        port_ret = w @ mean_ret
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-9:
            return 0.0
        return -(port_ret - risk_free_daily) / port_vol

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    x0 = equal_weights(n)
    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-9})
    if not result.success or np.any(np.isnan(result.x)):
        return x0
    w = np.clip(result.x, 0, None)
    total = w.sum()
    return w / total if total > 1e-9 else x0


# ---------------------------------------------------------------------------
# Adaptive neural allocation policy trained via Evolution Strategies
# ---------------------------------------------------------------------------

MOMENTUM_WINDOW = 21
VOL_WINDOW = 21


def build_state_features(returns: pd.DataFrame, regime_proba: pd.DataFrame, as_of_idx: int) -> np.ndarray:
    """
    State at rebalance step `as_of_idx` (index into `returns`), using only
    data available up to and including that index (no lookahead).
    """
    window_ret = returns.iloc[max(0, as_of_idx - MOMENTUM_WINDOW + 1): as_of_idx + 1]
    momentum = (1 + window_ret).prod() - 1.0            # per-asset trailing return
    vol = window_ret.std() * np.sqrt(TRADING_DAYS)        # per-asset annualized vol
    reg_row = regime_proba.iloc[min(as_of_idx, len(regime_proba) - 1)]
    state = np.concatenate([
        reg_row.values.astype(float),                      # [P(bull), P(bear), P(crisis)]
        momentum.fillna(0.0).values.astype(float),
        vol.fillna(0.0).values.astype(float),
    ])
    return state


@dataclass
class PolicyNet:
    """Minimal 2-layer MLP: state -> tanh(hidden) -> softmax(weights). Pure numpy."""
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def random_init(cls, state_dim: int, n_assets: int, hidden: int = 16, rng=None) -> "PolicyNet":
        rng = rng or np.random.default_rng()
        scale1 = np.sqrt(2.0 / state_dim)
        scale2 = np.sqrt(2.0 / hidden)
        return cls(
            w1=rng.normal(0, scale1, size=(state_dim, hidden)),
            b1=np.zeros(hidden),
            w2=rng.normal(0, scale2, size=(hidden, n_assets)),
            b2=np.zeros(n_assets),
        )

    def flatten(self) -> np.ndarray:
        return np.concatenate([self.w1.ravel(), self.b1.ravel(), self.w2.ravel(), self.b2.ravel()])

    def shapes(self):
        return self.w1.shape, self.b1.shape, self.w2.shape, self.b2.shape

    @classmethod
    def from_flat(cls, flat: np.ndarray, state_dim: int, n_assets: int, hidden: int = 16) -> "PolicyNet":
        i = 0
        w1 = flat[i:i + state_dim * hidden].reshape(state_dim, hidden); i += state_dim * hidden
        b1 = flat[i:i + hidden]; i += hidden
        w2 = flat[i:i + hidden * n_assets].reshape(hidden, n_assets); i += hidden * n_assets
        b2 = flat[i:i + n_assets]
        return cls(w1, b1, w2, b2)

    def act(self, state: np.ndarray) -> np.ndarray:
        h = np.tanh(state @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        logits = logits - logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()


def _simulate_policy(theta: np.ndarray, state_dim: int, n_assets: int, hidden: int,
                      returns: pd.DataFrame, regime_proba: pd.DataFrame,
                      rebalance_idxs: list[int], cost_bps: float) -> dict:
    """Roll a policy forward over the given rebalance points; return reward + series."""
    policy = PolicyNet.from_flat(theta, state_dim, n_assets, hidden)
    n_days = len(returns)
    weights_now = equal_weights(n_assets)
    value = 1.0
    values = np.empty(n_days)
    peak = 1.0
    max_dd = 0.0
    turnover_total = 0.0
    next_rebalance_ptr = 0

    for t in range(n_days):
        if next_rebalance_ptr < len(rebalance_idxs) and t == rebalance_idxs[next_rebalance_ptr]:
            state = build_state_features(returns, regime_proba, t)
            new_weights = policy.act(state)
            turnover_total += float(np.abs(new_weights - weights_now).sum())
            value *= (1 - cost_bps / 10000.0 * np.abs(new_weights - weights_now).sum())
            weights_now = new_weights
            next_rebalance_ptr += 1
        day_ret = float(weights_now @ returns.iloc[t].values)
        value *= (1 + day_ret)
        values[t] = value
        peak = max(peak, value)
        max_dd = max(max_dd, 1 - value / peak)

    daily_returns = np.diff(values, prepend=1.0) / np.concatenate([[1.0], values[:-1]])
    sharpe = 0.0
    if daily_returns.std() > 1e-9:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(TRADING_DAYS)
    reward = sharpe - 2.0 * max_dd - 0.5 * (turnover_total / max(1, len(rebalance_idxs)))
    return {"reward": reward, "sharpe": sharpe, "max_dd": max_dd, "values": values}


def train_adaptive_policy(returns: pd.DataFrame, regime_proba: pd.DataFrame, rebalance_idxs: list[int],
                           cost_bps: float = 5.0, hidden: int = 16, generations: int = 40,
                           population: int = 32, sigma: float = 0.1, lr: float = 0.05,
                           seed: int = 42) -> PolicyNet:
    """
    Train the adaptive allocation policy with a standard OpenAI-ES update:
    perturb the parameter vector with antithetic Gaussian noise, weight each
    perturbation by its (rank-normalized) reward, and step in that direction.
    Runs entirely on the TRAIN segment of the backtest window (see engine.py)
    so evaluation on the TEST segment is genuinely out-of-sample.
    """
    rng = np.random.default_rng(seed)
    n_assets = returns.shape[1]
    state_dim = 3 + n_assets * 2  # regime probs + momentum + vol per asset
    theta = PolicyNet.random_init(state_dim, n_assets, hidden, rng).flatten()

    if len(rebalance_idxs) < 2:
        return PolicyNet.from_flat(theta, state_dim, n_assets, hidden)

    for gen in range(generations):
        noise = rng.normal(0, 1, size=(population // 2, theta.size))
        noise = np.concatenate([noise, -noise], axis=0)  # antithetic sampling
        rewards = np.empty(noise.shape[0])
        for i in range(noise.shape[0]):
            candidate = theta + sigma * noise[i]
            rewards[i] = _simulate_policy(candidate, state_dim, n_assets, hidden,
                                           returns, regime_proba, rebalance_idxs, cost_bps)["reward"]
        ranks = rewards.argsort().argsort().astype(float)
        std_reward = (ranks - ranks.mean()) / (ranks.std() + 1e-9)
        theta = theta + (lr / (noise.shape[0] * sigma)) * (noise.T @ std_reward)

    return PolicyNet.from_flat(theta, state_dim, n_assets, hidden)


def adaptive_weights_series(policy: PolicyNet, returns: pd.DataFrame, regime_proba: pd.DataFrame,
                             rebalance_idxs: list[int]) -> dict[int, np.ndarray]:
    """Weights the trained policy would choose at each rebalance index."""
    out = {}
    for idx in rebalance_idxs:
        state = build_state_features(returns, regime_proba, idx)
        out[idx] = policy.act(state)
    return out
