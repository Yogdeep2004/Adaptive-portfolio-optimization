# Adaptive Portfolio Optimizer — Complete Project Summary

*Everything built and executed to date, based on the code as it stands on 5 Oct 2026 and results re-run
that day on live data. Numbers below come from those runs, not from earlier notes.*

---

## 0. Read this first: honest status

The system runs end to end (backend, dashboard, live data, three ML components). Three things in it are
weaker than earlier descriptions of it suggested:

1. **The dot-com backtest is not a clean out-of-sample test.** The dot-com data starts 31 Aug 2000 (gold
   futures, `GC=F`, have no earlier history), which is *after* the crisis start (10 Mar 2000). The
   train/test split therefore falls back to a 30-trading-day minimum, so the adaptive policy trains on only
   the first ~30 days (about 2 rebalance points) of data that is already inside the bust. Its strong result
   there (Sharpe 0.84, max drawdown −15.2% vs −43.0% equal-weight) should not be presented as evidence of
   learning. In the two cases where the split *is* valid (GFC, COVID) the adaptive policy did **not** beat
   mean-variance (§6.1).
2. **The "auto-detect" mode of the pattern matcher misses the two real crashes.** On the current code, a
   COVID-crash window and a GFC window both come back as "normal" (0% confidence), while dot-com and a calm
   period come back correct. The event-targeted mode (what the dashboard's event buttons use) does score
   them sensibly: COVID 51% against COVID, GFC 52% against GFC (§6.2).
3. **Documentation drift.** `README.md`, `EXPLAINED.md` and the published explainer page describe the
   earlier centroid-based matching. The code was changed after that (see §5.4), so those documents are now
   out of date on that point.

Everything else described below was verified as working on 5 Oct 2026.

---

## 1. Where this came from

The project is the B.Tech course project **"Adaptive Portfolio Optimization in Dynamic Financial Markets"**
(BCSE497J, VIT Vellore, Fall 2026-27; team: Anushka Mishra, Yogdeep Benchimath, Mehul Arora; guide: Dr. Arti
Tiwari), described in the `Review-1 2.pptx` deck:

- **Aim:** a framework that reallocates assets in response to real-time market regime shifts, improving
  risk-adjusted returns over static allocation.
- **Objectives:** adaptive multi-asset allocation (deep RL); real-time regime detection; transaction costs
  in the reward; benchmarking vs mean-variance and equal-weight on Sharpe, drawdown, return; robustness
  across bull/bear/volatile regimes. Targets SDG 8 and SDG 9; TRL 3 (proof of concept).
- **Request:** a working product with a GUI, AI integrated, analyzing past crises (dot-com, 2008, COVID),
  using real-time data via APIs, plus an explanation of what was done and the stack. Later extended to the
  headline feature: *compare any selected date range against the historical events and report whether it
  follows one of them, otherwise show a normal-regime baseline*, using Transformers and HDBSCAN.

---

## 2. What was built

A local full-stack web app (**FastAPI** backend + **React** dashboard), ~3,200 lines (1,960 Python,
1,250 JS/JSX/CSS), no test suite.

| Dashboard tab | What it does |
|---|---|
| **Live ribbon** (all tabs) | SPY, QQQ, GLD, TLT, BTC-USD, ^VIX live quotes, refreshed every 30 s |
| **Portfolio Backtest** | Pick dot-com / 2008 GFC / COVID. Runs Equal-Weight, rolling Mean-Variance and the Adaptive (ES-trained) policy on identical rebalance dates and costs. Shows value curves (crisis window shaded, train/test split marked), a metrics table (full period and out-of-sample), and the adaptive policy's weights over time |
| **Regime Detection** | Any symbol and date range. HMM colours each day Bull / Bear / Crisis over the price chart; current-day regime badge. Below it the **Historical Pattern Comparison** report: matched event (or "normal"), confidence, overlay chart of your period vs a real reference window, return/vol/drawdown stats, HMM regime mix, a next-21-day baseline, and templated insights |
| **About / Tech Stack** | In-app explanation of the method |

Other deliverables: `README.md`, `EXPLAINED.md` (plain-English → technical walkthrough), a designed web
version of that explainer, `run.sh` (one-command launcher), and an updated one-page resume (§9).

---

## 3. Architecture

```
┌──────────────────────────── Browser: React 19 + Vite 8 + Recharts 3 ────────────────────────────┐
│  TickerRibbon   BacktestPanel   RegimeExplorer ──► EventComparisonReport   AboutPanel           │
└───────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                            │ HTTP/JSON (axios), CORS open for local dev
┌───────────────────────────────────────────▼─────────────────────────────────────────────────────┐
│ FastAPI  app/main.py                                                                            │
│  GET  /api/health · /api/case-studies · /api/assets/default · /api/market/live                  │
│  GET  /api/regimes/current · /api/regimes/history · /api/regimes/compare[?event_id=]            │
│  POST /api/backtest                                                                             │
├────────────────┬───────────────────────┬──────────────────────────┬─────────────────────────────┤
│ data/          │ ml/regime.py          │ ml/embedding.py          │ backtest/engine.py          │
│  market_data   │  Gaussian HMM         │  Transformer encoder     │  runs 3 strategies,         │
│  synthetic_    │  (hmmlearn)           │  (PyTorch)               │  metrics                    │
│  fallback      │                       │ ml/event_matcher.py      │ ml/allocator.py             │
│                │                       │  corpus + HDBSCAN + kNN  │  EW / Markowitz / ES policy │
└───────┬────────┴───────────────────────┴──────────────────────────┴─────────────────────────────┘
        │ curl subprocess (retry/backoff) ──► Yahoo Finance chart API ──► 15-min CSV cache (backend/cache)
        └─ if unreachable (case studies only): calibrated synthetic fallback, flagged in the response
```

**Request flows**

- *Backtest:* prices → HMM on the primary asset gives daily regime probabilities → policy trained on the
  pre-split segment only → all three strategies walked forward on the same rebalance dates (every 21
  trading days, 5 bps cost on turnover) → metrics (full period and test period). Results cached in memory.
- *Regime chart:* prices → HMM → per-day regime labels plus probabilities.
- *Pattern comparison:* prices → resample the window to a 40-point, 3-channel shape → Transformer embedding
  → score against the labelled reference corpus → report. The matcher (corpus build + encoder training +
  HDBSCAN) is built once on the first request (~24 s) and cached for the process; later requests take ~0.1 s.
- *Frontend:* the chart and comparison load independently with request-ticket guards, so a slow or failed
  comparison never blanks the chart and an older response can't overwrite a newer one.

---

## 4. Tech stack

| Layer | Technology | Role / why |
|---|---|---|
| Backend | Python 3.12, **FastAPI**, uvicorn, Pydantic | REST API, validation, auto docs at `/docs` |
| Frontend | **React 19**, **Vite 8**, **Recharts 3**, axios, plain CSS | Dashboard and charts |
| Market data | **Yahoo Finance chart API** over `curl` | Free, no key; history and live quotes |
| Regime detection | **hmmlearn** Gaussian HMM | Unsupervised 3-state bull/bear/crisis labelling |
| Sequence model | **PyTorch 2.5** Transformer encoder | Fixed-length embedding of any price window |
| Clustering | **scikit-learn HDBSCAN** (1.5.2) | Density clustering of window embeddings |
| Allocation (RL) | **NumPy** MLP + **Evolution Strategies** | Adaptive allocator without an autodiff framework |
| Baselines | **SciPy** SLSQP, pandas, NumPy | Markowitz max-Sharpe, equal-weight, metrics |

---

## 5. How each component works

### 5.1 Data layer (`data/market_data.py`, `data/synthetic_fallback.py`)
- Fetches daily bars with explicit `period1/period2` timestamps (Yahoo's `range=max` silently downgrades
  daily bars to monthly) through `curl` (Yahoo rate-limited Python's `requests` stack much harder than
  `curl` from the same IP). 2 retries with backoff; per-symbol CSV cache, 15-minute TTL.
- Live quotes come from the chart response's metadata block.
- For the three case studies only, if live data fails entirely, a seeded geometric-Brownian-motion panel
  calibrated to each crash's historical magnitude is substituted and every response is tagged
  `synthetic_data: true` (the UI shows a banner).

### 5.2 Regime detection (`ml/regime.py`)
- Features per day: log return, 20-day annualized volatility, 60-day drawdown from rolling peak.
- 3-state diagonal-covariance Gaussian HMM (300 EM iterations, seed 7). States are relabelled by their own
  statistics: highest volatility → Crisis; of the rest, higher mean return → Bull; the other → Bear.
- Never given crash dates. Visually verified: it flags March 2020 as the red Crisis band.

### 5.3 Adaptive allocator and baselines (`ml/allocator.py`, `backtest/engine.py`)
- **Equal-weight** and **rolling Markowitz** (max-Sharpe, long-only, SLSQP, 63-day window) are the baselines.
- **Adaptive policy:** 2-layer MLP (state → tanh(16) → softmax weights). State = 3 regime probabilities plus
  21-day momentum and 21-day volatility per asset. Trained with **OpenAI-style Evolution Strategies**
  (32 antithetic perturbations × 40 generations, rank-normalized reward). Reward = Sharpe − 2.0 × max
  drawdown − 0.5 × average turnover (the report's "costs in the reward" gap).
- **Protocol:** train only on data before the crisis start, then walk forward without retraining; all
  strategies share rebalance dates (21 days) and cost (5 bps on turnover). *See §0 item 1 for where this
  protocol breaks (dot-com).*
- ES was chosen over PPO/DDPG (the papers in the literature review) because it trains in seconds on a CPU
  with no heavy dependency; swapping in PPO is the natural next step.

### 5.4 Historical pattern comparison (`ml/embedding.py`, `ml/event_matcher.py`)
1. **Length-invariant shape:** any window is resampled (linear interpolation) to 40 points × 3 channels:
   cumulative return, 5-day rolling volatility, running drawdown, all measured from the window's own start.
2. **Transformer encoder:** linear projection to d_model 32 + learned positional embedding → 2 layers,
   4 heads, GELU → mean-pool → 16-dim embedding. Trained as an autoencoder with a symmetric decoder (MSE,
   Adam, 120 full-batch epochs, seed 7). It is a small model trained on ~1,000 samples: a nonlinear,
   attention-based compression, not a large pretrained model.
3. **Reference corpus (965 windows):** overlapping windows at 21 lengths (10–600 trading days) cut from the
   three case studies plus four calm reference periods. Labels: ≥50% of the window inside the crisis, *or*
   the window contains ≥80% of the crisis and is ≤4× its length (so the short COVID crash gets longer
   exemplars); 5–50% overlap is dropped as ambiguous. Event classes are capped at 120 and normal at 2.5× the
   event total, stratified by window length. Current build: normal 679, dot-com 105, GFC 101, COVID 80.
   The dot-com corpus uses QQQ (its most distinctive move) via `shape_symbol` in `config.py`.
4. **HDBSCAN:** a 9-point grid of (min_cluster_size, min_samples) is searched with a score favouring
   events that get their own ≥60%-pure cluster, high purity and low noise (not "most clusters"); falls back
   to per-label centroids if clustering degenerates. Current build: 3 clusters.
5. **Scoring (current code):** the query is scored directly against labelled windows, not cluster centroids
   (centroids of density clusters can sit in empty space, and ~16% of the corpus is HDBSCAN noise). For each
   label: mean distance to its 8 nearest same-label windows, divided by that label's in-class spacing,
   through a Gaussian kernel, times an inverse-distance-weighted purity over the 15 nearest windows
   overall. A match needs distance ≤ 1.4× spacing, confidence ≥ 0.20 and (in auto mode) a non-normal label
   with purity > 0.6.
6. **Two modes:** *auto* (`event_id` omitted: which of the 3 events or "normal" is closest) and *targeted*
   (`event_id=dotcom|gfc|covid`: how close is this period to that specific event, always reported). The
   dashboard's event buttons now use targeted mode on the user's own symbol/dates.
7. **Report:** real stats, HMM regime mix, next-21-day return baseline from the matched (or normal)
   neighbours, a reference-window overlay, and a note when a similar return curve hides a very different
   drawdown/volatility. All text is templated from computed numbers; there is no LLM in the app.

---

## 6. Results (re-run 5 Oct 2026, live Yahoo data, default settings)

### 6.1 Portfolio backtests: out-of-sample (test) segment

| Case (data window, split) | Strategy | Total return | Sharpe | Max drawdown | Ann. vol |
|---|---|---|---|---|---|
| **Dot-com** (31 Aug 2000 – 2003; split 13 Oct 2000 ⚠ see §0) | Equal-weight | −10.4% | −0.16 | −43.0% | 20.9% |
| | Mean-variance | +52.6% | 0.66 | −17.7% | 20.3% |
| | **Adaptive** | +52.6% | **0.84** | **−15.2%** | 14.8% |
| **2008 GFC** (2006 – 2010; split 9 Oct 2007) | Equal-weight | +28.9% | 0.47 | −26.6% | 14.7% |
| | Mean-variance | +43.8% | **0.66** | **−19.1%** | 16.0% |
| | Adaptive | +20.2% | 0.35 | −23.8% | 13.1% |
| **COVID** (2019 – 2021; split 19 Feb 2020) | Equal-weight | +34.6% | 1.03 | −15.5% | 14.4% |
| | Mean-variance | +40.1% | **1.15** | −12.4% | 14.9% |
| | Adaptive | +26.6% | 0.85 | −12.8% | 13.4% |

**Reading it honestly:** the adaptive policy produced the lowest volatility in all three cases but lost to
mean-variance on Sharpe and return in GFC and COVID. Its dot-com win comes from the flawed split (§0). It
also edges equal-weight on drawdown in GFC and COVID, which is the risk-reduction behaviour the project
targets, but this is a modest result, not a clear win. It uses ~40 ES generations (kept low so the demo
responds quickly) and a small state; more generations, richer features, a longer training window or PPO
are the levers.

### 6.2 Historical pattern comparison

Auto-detect mode (current code):

| Query | Result | Confidence | Notes |
|---|---|---|---|
| QQQ 2000-03-01 → 2002-10-01 (649 d) | **Dot-com ✓** | 89% | max drawdown −82.4% |
| SPY 2013 → 2015 (756 d, calm) | Normal ✓ | 66% in "normal" | +14.0% p.a., drawdown −11.9% |
| SPY 2026-03 → 2026-09 (131 d) | Normal | 80% in "normal" | recent window |
| SPY 2020-02-01 → 04-30 (62 d, COVID) | **Normal ✗** | 0% | drawdown −33.7%; missed |
| SPY 2008-09-01 → 2009-03-09 (130 d, GFC) | **Normal ✗** | 0% | drawdown −46.0%; missed |

Targeted mode (how the dashboard's event buttons work): confidence of each window against each event.

| Period | vs dot-com | vs GFC | vs COVID |
|---|---|---|---|
| COVID window | 29% | 3% | **51%** |
| GFC window | 35% | **52%** | 0% |
| Dot-com window (QQQ) | **89%** | 0% | 0% |

In each row the correct event scores highest, but note the crash windows also give a non-trivial score
against the dot-com cluster (crash-then-recover shapes are genuinely similar), and in targeted mode the
`matched` flag only needs ≥20% confidence, so weak matches can still read as "matched". Treat the number,
not the badge, as the signal.

### 6.3 Other measured facts
- HMM correctly flags the March 2020 crash as Crisis with no labelled dates.
- Matcher cold build ≈ 24 s; cached queries ≈ 0.1 s. Backtests: 8–17 s each (including ES training).
- Corpus 965 windows; HDBSCAN found 3 clusters; in-class spacing (lower = tighter): normal 0.16, COVID 0.33,
  GFC 0.34, dot-com 0.37.

---

## 7. Engineering log (problems met, and how they were resolved)

| Problem | Resolution |
|---|---|
| Yahoo returned HTTP 429 to Python `requests` and `yfinance` | Fetch via `curl`; rotated a throttled User-Agent; 2 retries; disk cache |
| `range=max&interval=1d` returned ~monthly bars | Switched to explicit `period1/period2` |
| Live feed can fail during a demo | Labelled synthetic fallback for the 3 case studies |
| Frontend race: a slow initial request overwrote a newer click's result | Per-request ticket guards in `RegimeExplorer` |
| Stale-closure bug in the case-study buttons (`setTimeout` over old state) | Functions take explicit arguments |
| Dot-com corpus built from SPY but queried with QQQ | Single source of truth: `shape_symbol` in `config.py` |
| "Representative window" showed a 2002 window under a GFC label | Representative and forward-return stats restricted to majority-label members |
| Naive HDBSCAN scoring produced 69 fragments | Score rewards events covered, purity, low noise |
| Short COVID class starved by long classes | Per-class and per-length stratified caps; later an envelope labelling rule |
| Centroid-distance gating could reject genuine matches (a density cluster's mean can sit in empty space) | Scoring later reworked to nearest-neighbour + Gaussian kernel (§5.4); this rework was made after my original version, in the Sep 8–9 changes |
| Text extractors saw the resume twice | See §9 |

---

## 8. Known limitations

- Dot-com backtest split invalid (§0). Fix: pick a longer-history defensive asset, or start the dot-com
  window earlier.
- Auto-detect misses the COVID and GFC windows (§0, §6.2).
- Adaptive allocator does not beat mean-variance in the valid cases; ES, not PPO; single seed; no
  hyperparameter study; no statistical significance testing.
- The Transformer is small and trained on ~1,000 windows from three events; confidence is a similarity
  score, not a probability, and slow bear markets (dot-com, GFC) look similar to each other.
- No automated tests; no authentication; Yahoo's endpoint is unofficial and may rate-limit or change.
- Intended as a student demo (TRL 3); not investment advice and not a trading system. The app labels itself
  that way.
- `README.md` / `EXPLAINED.md` describe the older centroid matching (§0 item 3). A stray test file
  `backend/cache/NOPE_1d.csv` exists and can be deleted.

---

## 9. Other work done in this engagement

- **Docs:** `README.md` (setup, stack, pipeline, scope notes), `EXPLAINED.md` and a designed explainer page
  ("How the Optimizer Thinks"), written for a non-technical reader first.
- **Resume (5 Oct 2026):** added this project to the Projects section of the user's current one-page
  resume (`Yogdeep_Benchimath_Resume_Projects.pdf`), spliced into the original PDF so existing content is
  untouched, with the original fonts and bullet style, 3 bullets, and one page (~96% scale). Two mistakes
  along the way, both corrected: an early version was built from an outdated resume because of a
  path-resolution error, and a first splice left the file's text duplicated for ATS-style parsers.
  **Needs revising:** the second bullet quotes the dot-com result (§0 item 1) and should be replaced with a
  claim the data supports.

---

## 10. How to run

```bash
cd ~/Desktop/AdaptivePortfolioOptimizer && ./run.sh
# backend  http://127.0.0.1:8000  (docs at /docs)
# frontend http://127.0.0.1:5173
```
Needs Python 3.10+ and Node 18+; no API keys. First pattern-comparison request trains the model (~25 s).

## 11. Suggested next steps

1. Fix the dot-com experimental design and re-run (or drop it as an out-of-sample claim).
2. Fix or retire auto-detect mode so a COVID/GFC window is not labelled "normal".
3. Strengthen the allocator (more ES generations/seeds, richer features, or PPO) and report variance over seeds.
4. Add a test suite for the data layer, matcher gates and backtest engine.
5. Refresh `README.md` / `EXPLAINED.md` for the current scoring method.
6. Revise the resume's allocator bullet to match what the results support.
