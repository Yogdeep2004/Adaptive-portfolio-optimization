# Adaptive Portfolio Optimizer

A working reference implementation of the **"Adaptive Portfolio Optimization in Dynamic
Financial Markets"** project (BCSE497J, Fall 2026-27). Its centerpiece is the
**Historical Pattern Comparison** report: pick any symbol and date range, and it tells
you — with real computed numbers, not a canned answer — whether that period's trend
shape closely resembles the **dot-com bust**, the **2008 Global Financial Crisis**, or
the **COVID-19 crash**, or whether it's tracking neither and looks like a normal market
instead. Around that, the app also detects day-by-day market regimes (bull / bear /
crisis) with an unsupervised ML model, allocates a multi-asset portfolio with a
regime-aware policy trained via reinforcement learning, and benchmarks it against the
mean-variance and equal-weight baselines the project's literature review cites — all on
**live market data** plus the three historical stress periods.

## Quick start

```bash
./run.sh
```

This sets up a Python virtualenv (first run only), installs frontend dependencies (first
run only), and starts both servers:

- Backend API: http://127.0.0.1:8000 (interactive docs at `/docs`)
- Frontend dashboard: http://127.0.0.1:5173 ← open this in your browser

Requires Python 3.10+ and Node 18+ already installed. No API keys needed.

To run the two halves manually instead:

```bash
# Terminal 1
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # first time only
.venv/bin/uvicorn app.main:app --reload --port 8000

# Terminal 2
cd frontend
npm install      # first time only
npm run dev
```

## What's in the dashboard

1. **Portfolio Backtest** — pick a case study (Dot-com, 2008 GFC, COVID) and run three
   allocation strategies over identical rebalance dates: Equal-Weight, rolling
   Mean-Variance, and the ES-trained Adaptive policy. Shows the cumulative value chart
   (with the crisis window shaded and the train/test split marked), a metrics table
   (Sharpe ratio, max drawdown, annualized return, Calmar ratio — both full-period and
   out-of-sample-only), and how the adaptive policy's allocation shifts over time.
2. **Regime Detection** — pick any symbol and date range; a 3-state Hidden Markov Model
   classifies each trading day as Bull/Bear/Crisis, shown as colored bands under the
   price chart, plus the live current-day classification. Below the chart, the
   **Historical Pattern Comparison** report shows whether this period's trend shape
   matches a historical crisis (with a reference-window overlay chart and a written
   insights report) or reads as normal — see [How the Historical Pattern Comparison
   works](#how-the-historical-pattern-comparison-works).
3. **About / Tech Stack** — the same explanation as below, inside the app.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Backend | **FastAPI** (Python) | async I/O, automatic OpenAPI docs, Pydantic validation |
| Frontend | **React + Vite**, plain CSS, `recharts` | fast dev loop, no framework lock-in, good chart primitives |
| Market data | **Yahoo Finance chart API**, fetched directly | free, no API key, full historical + live quotes |
| Regime detection | **`hmmlearn`** (Gaussian HMM, 3 states) | standard unsupervised regime-detection model |
| Adaptive allocator | **NumPy** policy net + **Evolution Strategies** | gradient-free RL, no heavy ML framework, trains in seconds |
| Baselines | **SciPy** (`SLSQP`) | classic Markowitz mean-variance optimization |
| Pattern matching | **PyTorch** Transformer encoder | self-attention embedding of a window's resampled shape |
| Clustering | **scikit-learn** `HDBSCAN` | density-based clustering of shape embeddings; flags "no match" |

## How the pipeline works

1. **Data** — pull adjusted-close history for the asset universe (SPY, QQQ, GLD, TLT by
   default; the dot-com case study swaps in gold futures `GC=F` since GLD/TLT don't exist
   that far back) over the case study's date window. Fetched via Yahoo Finance's public
   chart JSON endpoint directly (not the `yfinance` package) with retry/backoff and a
   15-minute on-disk CSV cache per symbol — see [Data reliability](#data-reliability-notes).
2. **Regime detection** — fit a 3-state Gaussian HMM on `[daily return, 20-day rolling
   volatility, 60-day drawdown]` of a market proxy (e.g. SPY). HMM state indices are
   arbitrary per fit, so states are re-labeled by their empirical stats: highest-volatility
   state → **Crisis**, and of the remaining two, higher-mean-return → **Bull**, the other →
   **Bear**. This is fully unsupervised — the model is never told where the real crashes are.
3. **State features** — at every rebalance date (~every 21 trading days), build a state
   vector of `[P(bull), P(bear), P(crisis), 21-day momentum per asset, 21-day
   annualized volatility per asset]`, using only data available up to that date (no
   lookahead).
4. **Train (in-sample)** — the adaptive policy (a small 2-layer `state → tanh(16) →
   softmax(weights)` network) is trained with **Evolution Strategies** (antithetic
   Gaussian perturbations, rank-normalized reward, OpenAI-ES update rule) — but **only**
   on the segment of data *before* the case study's crisis onset date. It never sees the
   crash during training.
5. **Evaluate (out-of-sample)** — the trained policy is walked forward, with no further
   training, through the crisis and recovery, rebalancing on the same calendar dates as
   the two baselines, paying the same 5bps transaction cost on turnover.
6. **Compare** — Sharpe ratio, annualized return/volatility, max drawdown, and Calmar
   ratio, reported both for the full period and for the out-of-sample test segment alone
   (the fairer comparison, since only that segment is genuinely unseen by the adaptive
   policy).

## How the Historical Pattern Comparison works

This is a different job from the HMM above (which labels each individual day for the
whole series) — here the unit of comparison is one whole window, of any length, and the
question is "which historical episode's trend shape does this look like, if any."

1. **Length-invariant shape** — any window (15 trading days or 1200) is resampled via
   linear interpolation to a fixed **40-point, 3-channel** sequence: cumulative return,
   short (5-day) rolling volatility, and running drawdown, each measured from the
   window's own start. This makes COVID's ~33-day crash and the dot-com bust's ~2.5-year
   decline directly comparable by shape.
2. **Transformer encoder** — a small `nn.TransformerEncoder` (d_model=32, 4 heads, 2
   layers) embeds the (40,3) sequence into 16 dims, trained self-supervised as an
   autoencoder (reconstruction loss) on the reference corpus below.
3. **Reference corpus** — thousands of overlapping windows, at many lengths, sliced from
   the 3 case studies plus several long calm reference periods, each labeled by
   date-overlap with that case study's real crisis window (≥50% overlap → that event;
   ≤5% → "normal"; ambiguous → dropped). Class-balanced with a **stratified, per-length
   cap** — not flat random subsampling — so COVID's short crisis span (which naturally
   yields far fewer windows than dot-com/GFC's multi-year spans) isn't crowded out, and
   so the sparse but highly distinctive *long* windows spanning a slow bear market aren't
   drowned out by an abundance of short, weakly-shaped ones.
4. **Clustering** — `sklearn.cluster.HDBSCAN` clusters the corpus embeddings (a small
   parameter search picks the configuration that best separates the 3 named events into
   clean, high-purity clusters — not just whichever produces the most clusters). Each
   cluster is labeled by its members' majority event, only if that majority exceeds 60%
   purity; otherwise it's generically "normal."
5. **Match** — the query window is embedded the same way and matched to its nearest
   cluster centroid, via distance relative to that cluster's own typical member spread
   (sklearn's HDBSCAN has no `approximate_predict` for new points, unlike the standalone
   `hdbscan` package, so this is a direct, transparent nearest-centroid distance rather
   than a black-box call). Close enough, high-purity, and not-"normal" → a match;
   otherwise, no match, and the "normal" cluster's own stats become the baseline.
6. **Report** — either way: a real historical reference window's own shape plotted
   against yours (an honest visual "does this actually look the same"), the query's own
   annualized return/vol/drawdown and day-by-day HMM regime mix, and what
   similarly-shaped historical windows were followed by over the next 21 trading days —
   this is the "normal prediction" the report falls back to when there's no crisis match.

Case-study quick-select buttons compare the **crisis window itself** (±padding), not the
full multi-year range shown in the chart above it — comparing years of surrounding calm
data as one blob would dilute the crisis shape into looking "normal." The exact symbol
and date range actually compared is always shown in the report, since it can differ
(e.g. the dot-com comparison uses QQQ, the Nasdaq's ~78% collapse, since that's the
corpus's most distinctive/recognizable shape for that event — not SPY's milder ~49%).

## Honest scope notes

- **Why Evolution Strategies instead of PPO/DDPG/Transformer-PPO** (as in the papers the
  project's literature review surveys)? ES is a legitimate, published deep-RL training
  method (Salimans et al., 2017, *"Evolution Strategies as a Scalable Alternative to
  Reinforcement Learning"*) that needs no autodiff framework, trains in a few seconds on a
  laptop CPU, and keeps the whole app dependency-light and instantly runnable for a live
  demo. The state/reward design here (regime probabilities + momentum/vol features;
  Sharpe − drawdown penalty − turnover cost) is deliberately set up so that swapping in a
  proper PPO agent (e.g. via `stable-baselines3` + a `gymnasium` environment) for the full
  research version is a drop-in next step, not a redesign.
- **The adaptive policy does not always beat the baselines** in every run — and that's
  reported honestly, not massaged. With only ~30-40 ES generations (kept low for instant
  demo response) and a fairly small state, it sometimes underperforms mean-variance on
  raw Sharpe while achieving a lower max drawdown (i.e. it trades some upside for
  downside protection), and sometimes underperforms on both. Levers to improve it as a
  research direction: more ES generations/population, richer state features
  (cross-asset correlation, macro indicators), a longer training window, or replacing ES
  with PPO.
- **The Transformer encoder is trained on a few hundred small, short samples** — not the
  scale a real pretrained model needs. What it learns is closer to a nonlinear,
  attention-based compression of the return/vol/drawdown path than deep
  "understanding," but that's still a genuine, correctly-scoped use of self-attention
  (it can represent *where in the window* a spike sits, which flat summary statistics
  can't). Match confidence is reported honestly and is often well under 100% — shape
  similarity to a specific historical crisis is a real but imperfect signal, not a
  certainty, and the UI never hides that.
- **Data reliability notes**: <a name="data-reliability-notes"></a> the app talks to
  Yahoo Finance's public (undocumented) chart endpoint. It can occasionally rate-limit a
  client — if that happens, the three case studies (not custom symbol lookups) fall back
  automatically to a **clearly-labeled, calibrated synthetic dataset** (matching the real
  historical crash magnitude for that period) so a live demo never simply breaks. Every
  response is tagged `synthetic_data: true/false` and the dashboard shows a visible
  banner whenever the fallback is used. This has nothing to do with API keys — it's a
  public endpoint — it's just occasionally rate-limited on shared/heavily-used networks.

## Project layout

```
backend/
  app/
    main.py              FastAPI app & routes
    config.py             default assets, case study definitions
    schemas.py             request models
    data/
      market_data.py       Yahoo Finance client (retry/backoff + disk cache)
      synthetic_fallback.py  calibrated offline fallback for the 3 case studies
    ml/
      regime.py             Gaussian HMM regime detection
      allocator.py           baselines + ES-trained adaptive policy
      embedding.py            Transformer shape encoder (build/resample/train/embed)
      event_matcher.py        reference corpus + HDBSCAN clustering + match_window()
    backtest/
      engine.py              runs all 3 strategies, computes metrics
  requirements.txt
frontend/
  src/
    api.js                 API client
    App.jsx                 tab layout
    components/             RegimeExplorer, EventComparisonReport, BacktestPanel,
                             TickerRibbon, AboutPanel
    utils.js
run.sh                     one-command launcher for both servers
```

## Extending this

- **Swap in real deep RL**: replace `ml/allocator.py`'s ES training with a
  `stable-baselines3` PPO agent over a `gymnasium.Env` wrapping the same state/reward —
  the backtest engine and frontend don't need to change.
- **More assets / a live options paper-trading loop**: `config.py` is the single place
  asset universes and case studies are defined.
- **Different regime model**: swap `ml/regime.py`'s Gaussian HMM for a
  Transformer/LSTM-based regime classifier (as in the "Regime-Aware RL" paper in the
  literature review) without touching the allocator or backtest engine — they only
  consume `regime_proba` (a DataFrame of per-day regime probabilities).
