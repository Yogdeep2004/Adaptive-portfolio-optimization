"""Functional and unit test run for the report (Chapter 5). Exercises the REST API through FastAPI's TestClient and
checks core numerical properties of the analytics modules. Results are saved to paper/results/tests.json.
Run:  backend/.venv/bin/python paper/scripts/run_tests.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402
from app.ml import allocator, embedding, event_matcher as em  # noqa: E402
from app.ml.regime import detect_regimes  # noqa: E402
from app.data import market_data as md  # noqa: E402

client = TestClient(app)
results = []


def run(tid, kind, title, expected, fn):
    t0 = time.time()
    try:
        ok, observed = fn()
        status = "Pass" if ok else "Fail"
    except Exception as exc:  # noqa: BLE001
        status, observed = "Error", f"{type(exc).__name__}: {exc}"[:160]
    dt = time.time() - t0
    results.append(dict(id=tid, kind=kind, title=title, expected=expected, observed=str(observed), status=status,
                        seconds=round(dt, 2)))
    print(tid, status, round(dt, 1), "s |", observed, flush=True)


# ---------------------------------------------------------------- API tests
def t_health():
    r = client.get("/api/health")
    return r.status_code == 200, f"HTTP {r.status_code}, body keys {sorted(r.json())}"


def t_cases():
    r = client.get("/api/case-studies")
    j = r.json()
    items = j if isinstance(j, list) else j.get("case_studies", j)
    ids = [c["id"] for c in items] if isinstance(items, list) else list(items)
    return r.status_code == 200 and set(ids) == {"dotcom", "gfc", "covid"}, f"HTTP {r.status_code}, ids {ids}"


def t_assets():
    r = client.get("/api/assets/default")
    return r.status_code == 200, f"HTTP {r.status_code}, {r.json()}"


def t_live():
    r = client.get("/api/market/live")
    j = r.json()
    quotes = j.get("quotes", j) if isinstance(j, dict) else j
    return r.status_code == 200 and len(quotes) >= 1, f"HTTP {r.status_code}, {len(quotes)} quotes"


def t_current():
    r = client.get("/api/regimes/current", params={"symbol": "SPY"})
    j = r.json()
    probs = j.get("probabilities") or j.get("proba") or {}
    s = sum(probs.values()) if probs else float("nan")
    return r.status_code == 200 and abs(s - 1) < 0.02, f"HTTP {r.status_code}, regime {j.get('regime')}, prob sum {s:.3f}"


def t_history():
    r = client.get("/api/regimes/history", params={"symbol": "SPY", "start": "2007-01-01", "end": "2010-12-31"})
    j = r.json()
    n = len(j["dates"])
    ok = r.status_code == 200 and n == len(j["prices"]) == len(j["regime"]) and set(j["regime"]) <= {"bull", "bear", "crisis"}
    return ok, f"HTTP {r.status_code}, {n} days, states {sorted(set(j['regime']))}"


def t_compare_missing():
    r = client.get("/api/regimes/compare", params={"symbol": "SPY"})
    return r.status_code == 400, f"HTTP {r.status_code}, {r.json().get('detail')}"


def t_compare_short():
    r = client.get("/api/regimes/compare", params={"symbol": "SPY", "start": "2020-03-02", "end": "2020-03-10"})
    return r.status_code == 400, f"HTTP {r.status_code}, {r.json().get('detail')}"


def t_compare_covid():
    r = client.get("/api/regimes/compare", params={"symbol": "SPY", "start": "2020-01-15", "end": "2020-05-15",
                                                   "event_id": "covid"})
    j = r.json()
    m = j["match"]
    return r.status_code == 200 and m["matched"] and m["event_id"] == "covid", \
        f"HTTP {r.status_code}, matched={m['matched']}, event={m['event_id']}, confidence={m['confidence']:.2f}"


def t_compare_calm():
    r = client.get("/api/regimes/compare", params={"symbol": "SPY", "start": "2013-01-01", "end": "2015-12-31"})
    m = r.json()["match"]
    return r.status_code == 200 and not m["matched"], f"HTTP {r.status_code}, matched={m['matched']}, event={m['event_id']}"


def t_compare_badsym():
    r = client.get("/api/regimes/compare", params={"symbol": "ZZZZNOTASYMBOL", "start": "2020-01-15", "end": "2020-05-15"})
    return r.status_code in (400, 502), f"HTTP {r.status_code}, {str(r.json().get('detail'))[:70]}"


def t_bt_unknown():
    r = client.post("/api/backtest", json={"case_study": "nonexistent"})
    return r.status_code == 400, f"HTTP {r.status_code}, {r.json().get('detail')}"


BT = {}


def t_bt_covid():
    r = client.post("/api/backtest", json={"case_study": "covid"})
    j = r.json()
    BT["covid"] = j
    keys = sorted(j.keys())
    return r.status_code == 200, f"HTTP {r.status_code}, keys {keys[:8]}"


def t_bt_weights():
    j = BT["covid"]
    worst, n = 0.0, 0
    for strat, body in j["strategies"].items():
        for date, w in body["weights"].items():
            worst = max(worst, abs(sum(w.values()) - 1.0)); n += 1
    return n > 0 and worst < 1e-3, f"{n} rebalance rows across 3 strategies, max |sum(w) - 1| = {worst:.1e}"


def t_bt_cost():
    a = client.post("/api/backtest", json={"case_study": "gfc", "cost_bps": 0}).json()
    b = client.post("/api/backtest", json={"case_study": "gfc", "cost_bps": 50}).json()
    ma = a["strategies"]["mean_variance"]["values"][-1]; mb = b["strategies"]["mean_variance"]["values"][-1]
    return ma > mb, f"Mean-Variance final value {ma:.3f} at 0 bps vs {mb:.3f} at 50 bps (start 1.0)"


# ---------------------------------------------------------------- unit tests
def u_shape():
    idx = pd.bdate_range("2020-01-01", periods=600)
    rng = np.random.default_rng(1)
    base = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600))), index=idx)
    shapes = []
    for n in (33, 120, 600):
        shapes.append(embedding.resample_to_fixed(embedding.build_channels(base.iloc[:n])).shape)
    return all(s == (40, 3) for s in shapes), f"output shapes for 33, 120, 600 day windows: {shapes}"


def u_weights():
    rng = np.random.default_rng(3)
    r = pd.DataFrame(rng.normal(0.0004, 0.01, (63, 4)), columns=list("ABCD"))
    ew = allocator.equal_weights(4)
    mv = allocator.mean_variance_weights(r)
    ok = abs(ew.sum() - 1) < 1e-9 and abs(mv.sum() - 1) < 1e-6 and (mv >= -1e-9).all()
    return ok, f"equal-weight sum {ew.sum():.4f}; mean-variance sum {mv.sum():.4f}, min weight {mv.min():.4f}"


def u_policy():
    rng = np.random.default_rng(5)
    p = allocator.PolicyNet.random_init(state_dim=11, n_assets=4, rng=rng)
    w = p.act(rng.normal(size=11))
    flat = p.flatten()
    q = allocator.PolicyNet.from_flat(flat, 11, 4)
    ok = abs(w.sum() - 1) < 1e-9 and (w >= 0).all() and np.allclose(q.act(np.ones(11)), p.act(np.ones(11)))
    return ok, f"softmax weights sum {w.sum():.4f}, min {w.min():.3f}; flatten/from_flat round trip exact"


def u_hmm():
    h = md.get_history("SPY", start="2007-01-01", end="2010-12-31")
    res = detect_regimes(h["adjclose"])
    st = res.state_stats
    vols = {k: v.get("mean_vol", v.get("vol", np.nan)) if isinstance(v, dict) else v for k, v in st.items()}
    crisis_is_max = max(vols, key=lambda k: vols[k]) == "crisis"
    return crisis_is_max, f"mean volatility by state {({k: round(float(v), 3) for k, v in vols.items()})}"


def u_matcher():
    m = em.get_matcher()
    labs = {}
    for w in m.corpus:
        labs[w.label] = labs.get(w.label, 0) + 1
    ok = m.embeddings.shape == (len(m.corpus), 16) and sum(labs.values()) == 965
    return ok, f"embeddings {m.embeddings.shape}, labels {labs}, fit method {m.fit_method}, {len(m.cluster_meta)} clusters"


def u_determinism():
    a = em.match_window("SPY", "2020-01-15", "2020-05-15", target_event="covid")["match"]["confidence"]
    b = em.match_window("SPY", "2020-01-15", "2020-05-15", target_event="covid")["match"]["confidence"]
    return abs(a - b) < 1e-9, f"two identical queries gave confidence {a:.4f} and {b:.4f}"


tests = [
    ("TC01", "API", "GET /api/health", "HTTP 200", t_health),
    ("TC02", "API", "GET /api/case-studies", "three case studies", t_cases),
    ("TC03", "API", "GET /api/assets/default", "default four-asset universe", t_assets),
    ("TC04", "API", "GET /api/market/live", "at least one live quote", t_live),
    ("TC05", "API", "GET /api/regimes/current (SPY)", "regime label, probabilities sum to 1", t_current),
    ("TC06", "API", "GET /api/regimes/history (SPY 2007-2010)", "aligned arrays, states in {bull, bear, crisis}", t_history),
    ("TC07", "API", "Compare without start and end dates", "HTTP 400", t_compare_missing),
    ("TC08", "API", "Compare window of 6 trading days", "HTTP 400 (window too short)", t_compare_short),
    ("TC09", "API", "Compare COVID-19 window, targeted at covid", "matched, event covid", t_compare_covid),
    ("TC10", "API", "Compare calm 2013-2015 window, auto mode", "no crisis match", t_compare_calm),
    ("TC11", "API", "Compare with an invalid ticker", "HTTP 400 or 502", t_compare_badsym),
    ("TC12", "API", "POST /api/backtest with unknown case study", "HTTP 400", t_bt_unknown),
    ("TC13", "API", "POST /api/backtest, COVID-19 case study", "HTTP 200 with three strategies", t_bt_covid),
    ("TC14", "Unit", "Backtest weights are fully invested", "each rebalance row sums to 1", t_bt_weights),
    ("TC15", "Unit", "Higher cost lowers net return", "return at 0 bps above return at 50 bps", t_bt_cost),
    ("TC16", "Unit", "Resampling is length invariant", "(40, 3) for any window length", u_shape),
    ("TC17", "Unit", "Equal-weight and mean-variance weights", "sum to 1, long-only", u_weights),
    ("TC18", "Unit", "Adaptive policy output and parameter round trip", "softmax weights, lossless flatten", u_policy),
    ("TC19", "Unit", "HMM crisis state has highest volatility", "crisis = max volatility state", u_hmm),
    ("TC20", "Unit", "Reference corpus and embedding size", "965 windows, 16-d embeddings", u_matcher),
    ("TC21", "Unit", "Matcher is deterministic", "identical confidence on repeat", u_determinism),
]
for t in tests:
    run(*t)

summary = dict(total=len(results), passed=sum(r["status"] == "Pass" for r in results),
               failed=sum(r["status"] == "Fail" for r in results), errors=sum(r["status"] == "Error" for r in results))
print(summary)
json.dump(dict(summary=summary, results=results), open(ROOT / "paper" / "results" / "tests.json", "w"), indent=1)
