"""Experiment 1: backtests (default seed), corrected dot-com design, turnover, bootstrap CIs, dataset statistics.
Run with the backend venv:  backend/.venv/bin/python paper/scripts/exp_backtest.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app import config  # noqa: E402
from app.backtest.engine import run_backtest  # noqa: E402
from app.data import market_data as md  # noqa: E402

RES = ROOT / "paper" / "results"
RES.mkdir(exist_ok=True)

def _case(label, base, **over):
    d = dict(config.CASE_STUDIES[base]); d.update(over); d["label"] = label
    return d


CASES = {
    "dotcom_deployed": _case("Dot-com (as deployed: SPY, QQQ, GC=F)", "dotcom"),
    "dotcom_corrected": _case("Dot-com (corrected: SPY, QQQ, VUSTX)", "dotcom", assets=["SPY", "QQQ", "VUSTX"]),
    "gfc": _case("2008 GFC", "gfc"),
    "covid": _case("COVID-19", "covid"),
}


def sharpe(r, rf_annual=config.RISK_FREE_RATE_ANNUAL):
    rf = rf_annual / 252
    sd = np.std(r)
    return float(np.mean(r - rf) / sd * np.sqrt(252)) if sd > 1e-12 else 0.0


def stationary_bootstrap_diff(ra, rb, n_boot=5000, mean_block=10, seed=0):
    """95% CI for Sharpe(a) - Sharpe(b) with the stationary bootstrap (paired resampling of days)."""
    rng = np.random.default_rng(seed)
    n = len(ra)
    p = 1.0 / mean_block
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        jumps = rng.random(n) < p
        starts = rng.integers(n, size=n)
        for t in range(1, n):
            idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % n
        diffs[b] = sharpe(ra[idx]) - sharpe(rb[idx])
    point = sharpe(ra) - sharpe(rb)
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float((diffs <= 0).mean())


def turnover(weights_log):
    ws = [np.array([w[k] for k in sorted(w)]) for _, w in sorted(weights_log.items())]
    if len(ws) < 2:
        return 0.0
    return float(np.mean([np.abs(ws[i] - ws[i - 1]).sum() for i in range(1, len(ws))]))


def daily_returns(values):
    prev = np.concatenate([[1.0], values[:-1]])
    return values / prev - 1.0


out = {"cases": {}, "dataset": {}}
series = {}
for key, case in CASES.items():
    panel = md.get_multi_history(case["assets"], start=case["start"], end=case["end"]).dropna()
    res = run_backtest(panel, primary_symbol="SPY", split_date=case["crisis_start"])
    dates = res["strategies"]["equal_weight"]["dates"]
    split_idx = dates.index(res["train_test_split_date"]) if res["train_test_split_date"] in dates else None
    vals = {s: np.array(res["strategies"][s]["values"]) for s in ("equal_weight", "mean_variance", "adaptive")}
    rets = {s: daily_returns(v) for s, v in vals.items()}
    n_train_days = split_idx
    boot = {}
    for base in ("mean_variance", "equal_weight"):
        point, lo, hi, p_le0 = stationary_bootstrap_diff(rets["adaptive"][split_idx:], rets[base][split_idx:])
        boot[f"adaptive_minus_{base}"] = dict(sharpe_diff=point, ci_low=lo, ci_high=hi, p_diff_le_0=p_le0)
    out["cases"][key] = dict(
        label=case["label"], assets=res["assets"], date_range=res["date_range"], split=res["train_test_split_date"],
        crisis_window=[case["crisis_start"], case["crisis_end"]], n_days=len(dates), n_train_days=n_train_days,
        n_test_days=len(dates) - n_train_days,
        metrics_test=res["metrics_test_period"], metrics_full=res["metrics_full_period"],
        turnover={s: turnover(res["strategies"][s]["weights"]) for s in vals}, bootstrap=boot,
    )
    series[key] = dict(dates=dates, split_idx=split_idx, values=vals, assets=res["assets"],
                       weights=res["strategies"]["adaptive"]["weights"], regimes=res["regimes"],
                       crisis=[case["crisis_start"], case["crisis_end"]])
    # dataset descriptive statistics over the case window
    out["dataset"][key] = {}
    for a in res["assets"]:
        px = panel[a]
        r = px.pct_change().dropna()
        out["dataset"][key][a] = dict(n=int(len(px)), start=str(px.index[0].date()), end=str(px.index[-1].date()),
                                      ann_return=float((px.iloc[-1] / px.iloc[0]) ** (252 / len(px)) - 1),
                                      ann_vol=float(r.std() * np.sqrt(252)),
                                      max_drawdown=float((px / px.cummax() - 1).min()))
    print(key, "done; train days", n_train_days, "test days", len(dates) - n_train_days, flush=True)

json.dump(out, open(RES / "backtest.json", "w"), indent=1)
pickle.dump(series, open(RES / "backtest_series.pkl", "wb"))
print("saved")
