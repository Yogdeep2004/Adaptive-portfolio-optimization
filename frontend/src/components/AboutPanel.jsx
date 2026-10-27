export default function AboutPanel() {
  return (
    <div className="panel about-content">
      <h3 className="panel-title" style={{ fontSize: 18 }}>What this is</h3>
      <p>
        A working reference implementation of the <b>"Adaptive Portfolio Optimization in Dynamic Financial
        Markets"</b> project. Its centerpiece is the <b>Historical Pattern Comparison</b> report in the Regime
        Detection tab: pick any symbol and date range, and it tells you — with real computed numbers, not a
        canned answer — whether that period's trend shape closely resembles the dot-com bust, the 2008 GFC, or
        the COVID crash, or whether it's tracking neither and looks like a normal market instead. Around that,
        the app also detects day-by-day market regimes (bull / bear / crisis) with an unsupervised ML model,
        allocates a multi-asset portfolio with a regime-aware policy trained via reinforcement learning, and
        benchmarks it against the mean-variance and equal-weight baselines your literature review cites — all on
        live market data plus the three historical stress periods.
      </p>

      <h3>Tech stack</h3>
      <div className="stack-grid">
        <div className="stack-card">
          <b>Backend — FastAPI (Python)</b>
          <p>REST API serving market data, regime detection, and backtests. Chosen for async I/O, automatic
            OpenAPI docs, and Pydantic request validation.</p>
        </div>
        <div className="stack-card">
          <b>Frontend — React + Vite</b>
          <p>Single-page dashboard, plain CSS (no framework lock-in), <code>recharts</code> for all charts,
            polling for the live ticker ribbon.</p>
        </div>
        <div className="stack-card">
          <b>Market data — Yahoo Finance chart API</b>
          <p>Free, no API key. Fetched directly (not via the <code>yfinance</code> package) for reliable retries,
            with a 15-minute on-disk CSV cache per symbol.</p>
        </div>
        <div className="stack-card">
          <b>Regime detection — <code>hmmlearn</code></b>
          <p>3-state Gaussian Hidden Markov Model on [return, rolling volatility, drawdown]. Fully unsupervised —
            it isn't told where the crashes are.</p>
        </div>
        <div className="stack-card">
          <b>Adaptive allocator — NumPy + Evolution Strategies</b>
          <p>A small 2-layer policy network (state → softmax weights) trained with a gradient-free RL algorithm
            (OpenAI-ES), reward = Sharpe − drawdown penalty − turnover cost.</p>
        </div>
        <div className="stack-card">
          <b>Baselines — SciPy</b>
          <p>Rolling-window mean-variance (Markowitz tangency portfolio) via SLSQP; naive 1/N equal-weight.</p>
        </div>
        <div className="stack-card">
          <b>Pattern matching — PyTorch Transformer</b>
          <p>A small self-attention encoder turns any window's resampled return/vol/drawdown shape into a 16-dim
            embedding, trained self-supervised (autoencoder) on a historical corpus.</p>
        </div>
        <div className="stack-card">
          <b>Clustering — scikit-learn HDBSCAN</b>
          <p>Density-based clustering over those embeddings discovers which shapes group with each historical
            crisis — and, importantly, which ones belong to none of them.</p>
        </div>
      </div>

      <h3>How the pipeline fits together</h3>
      <ol>
        <li><b>Data:</b> pull adjusted-close history for the asset universe (SPY, QQQ, GLD, TLT by default) over
          the chosen window.</li>
        <li><b>Regime detection:</b> fit the HMM on a market proxy (e.g. SPY), label each day Bull / Bear / Crisis,
          and get a daily probability distribution over the three regimes.</li>
        <li><b>Feature state:</b> at each rebalance date (every ~21 trading days), build a state vector of
          [regime probabilities, 21-day momentum per asset, 21-day volatility per asset] — using only data
          available up to that date.</li>
        <li><b>Train (in-sample):</b> the adaptive policy is trained with Evolution Strategies <i>only</i> on data
          before the crisis onset — it never sees the crash during training.</li>
        <li><b>Evaluate (out-of-sample):</b> walk the trained policy forward through the crisis and recovery,
          alongside mean-variance and equal-weight on the same dates, with the same 5bps transaction cost on
          turnover.</li>
        <li><b>Compare:</b> Sharpe ratio, max drawdown, annualized return, and Calmar ratio for all three,
          both full-period and out-of-sample-only.</li>
      </ol>

      <h3>How the Historical Pattern Comparison works</h3>
      <ol>
        <li><b>Length-invariant shape:</b> any window — 15 trading days or 1200 — is resampled via linear
          interpolation to a fixed 40-point, 3-channel sequence: cumulative return, short rolling volatility, and
          running drawdown, each measured from the window's own start. This makes COVID's ~33-day crash and the
          dot-com bust's ~2.5-year decline directly comparable by shape.</li>
        <li><b>Reference corpus:</b> thousands of overlapping windows (varying lengths) are sliced from the 3
          historical case studies plus several long calm periods, and labeled by their date-overlap with each
          crisis's actual start/end — with class-balancing so COVID's short crisis span (which naturally yields
          far fewer windows than dot-com/GFC's multi-year spans) isn't crowded out.</li>
        <li><b>Embed + cluster:</b> the Transformer encoder embeds every corpus window; HDBSCAN clusters the
          embeddings, and each cluster is labeled by its members' majority event (only if that majority exceeds
          60% purity — otherwise it's treated as "normal").</li>
        <li><b>Match:</b> your selected window is embedded the same way and matched to its nearest cluster
          centroid, gated by that cluster's own typical member spread — close enough, high-purity, and
          not-"normal" → a match; otherwise, no match.</li>
        <li><b>Report:</b> either way, the report shows a real historical reference window's own shape plotted
          against yours, the day-by-day HMM regime mix, and what similarly-shaped historical windows were
          followed by over the next 21 trading days — the "normal prediction" when there's no crisis match.</li>
      </ol>

      <h3>Honest scope notes</h3>
      <p>
        The literature review's papers use PPO / DDPG / Transformer-based deep RL agents trained on GPUs over many
        episodes. This build uses <b>Evolution Strategies</b> instead — a legitimate, published RL training method
        (Salimans et al., 2017) — because it needs no autodiff framework, trains in seconds on a laptop CPU, and
        keeps the whole app dependency-light and instantly runnable for a live demo. Swapping in a PPO agent (e.g.
        via <code>stable-baselines3</code>) with a richer <code>gymnasium</code> environment is the natural next
        step for the full research version of this project, and the state/reward design here is already set up to
        transfer directly onto that.
      </p>
      <p>
        Similarly, if Yahoo Finance is unreachable (rate-limited network, no internet), the three case studies fall
        back to a clearly-labeled synthetic dataset calibrated to the real historical magnitude of that crash, so
        the demo never simply breaks — but real Yahoo data is used whenever the network allows it.
      </p>
      <p>
        The Transformer encoder is trained on a few hundred small, short samples — not the scale a real
        pretrained model needs. What it learns is closer to a nonlinear, attention-based compression of the
        return/vol/drawdown path than deep "understanding," but that's still a genuine, correctly-scoped use of
        self-attention (it can represent <i>where in the window</i> a spike sits, which flat summary statistics
        can't). Confidence scores are reported honestly — they're often well under 100%, because shape similarity
        to a specific historical crisis is a real but imperfect signal, not a certainty.
      </p>
    </div>
  );
}
