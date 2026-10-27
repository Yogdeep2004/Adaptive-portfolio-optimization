import { useMemo, useState } from "react";
import {
  ResponsiveContainer, ComposedChart, Line, XAxis, YAxis, Tooltip, ReferenceArea, ReferenceLine,
  CartesianGrid, Legend, AreaChart, Area,
} from "recharts";
import api from "../api";
import { fmtNum, fmtPct } from "../utils";

const STRATEGY_META = {
  equal_weight: { label: "Equal-Weight (baseline)", color: "#9aa5b8" },
  mean_variance: { label: "Mean-Variance (baseline)", color: "#f0b429" },
  adaptive: { label: "Adaptive Allocator (regime-aware RL policy)", color: "#7aa2ff" },
};

const ASSET_COLORS = ["#7aa2ff", "#f0b429", "#22c55e", "#e879f9", "#38bdf8", "#fb923c"];

function buildValueSeries(result) {
  const dates = result.strategies.equal_weight.dates;
  return dates.map((d, i) => ({
    date: d,
    equal_weight: result.strategies.equal_weight.values[i],
    mean_variance: result.strategies.mean_variance.values[i],
    adaptive: result.strategies.adaptive.values[i],
  }));
}

function buildWeightsSeries(result) {
  const dates = result.strategies.adaptive.dates;
  const weightsByDate = result.strategies.adaptive.weights;
  const assets = result.assets;
  let last = Object.fromEntries(assets.map((a) => [a, 1 / assets.length]));
  return dates.map((d) => {
    if (weightsByDate[d]) last = weightsByDate[d];
    return { date: d, ...last };
  });
}

export default function BacktestPanel({ caseStudies, defaultAssets }) {
  const [caseStudyId, setCaseStudyId] = useState("gfc");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [metricsView, setMetricsView] = useState("test");

  const run = () => {
    setLoading(true);
    setError(null);
    api
      .runBacktest({ case_study: caseStudyId, es_generations: 40, es_population: 32 })
      .then(setResult)
      .catch((e) => setError(e.response?.data?.detail || e.message))
      .finally(() => setLoading(false));
  };

  const valueSeries = useMemo(() => (result ? buildValueSeries(result) : []), [result]);
  const weightSeries = useMemo(() => (result ? buildWeightsSeries(result) : []), [result]);
  const activeCase = caseStudies?.find((c) => c.id === caseStudyId);

  return (
    <div>
      <div className="panel">
        <h3 className="panel-title">Adaptive Portfolio Backtest</h3>
        <p className="panel-desc">
          Runs three allocation strategies — Equal-Weight, rolling Mean-Variance (Markowitz), and the ES-trained
          Adaptive regime-aware policy — on identical rebalance dates, with identical transaction costs, and compares
          Sharpe ratio, max drawdown, and cumulative return. The adaptive policy is trained only on data{" "}
          <b>before</b> the crisis onset and evaluated walk-forward through the crash and recovery — it never sees
          the test period during training.
        </p>

        <div className="controls-row">
          {caseStudies?.map((cs) => (
            <button
              key={cs.id}
              className="btn-secondary"
              style={caseStudyId === cs.id ? { borderColor: "#4f7cff", color: "#cdd9ff" } : {}}
              onClick={() => setCaseStudyId(cs.id)}
            >
              {cs.label}
            </button>
          ))}
          <button className="btn" onClick={run} disabled={loading}>
            {loading ? "Training & backtesting…" : "Run Backtest"}
          </button>
        </div>

        {activeCase && <p className="panel-desc" style={{ marginTop: -6 }}>{activeCase.description}</p>}

        {error && <div className="notice error">Backtest failed: {error}</div>}
        {result?.synthetic_data && (
          <div className="notice">
            ⚠️ Live market data was unreachable, so this run uses a <b>calibrated synthetic fallback</b> (same
            historical magnitude of the crash, generated locally) instead of real Yahoo Finance prices — clearly
            marked so results aren't mistaken for the real historical record.
          </div>
        )}

        {loading && (
          <div className="spinner-row">
            <span className="spinner" /> Fetching prices, fitting the regime model, training the ES policy…
          </div>
        )}

        {result && (
          <>
            <ResponsiveContainer width="100%" height={360}>
              <ComposedChart data={valueSeries} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#1c2436" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#8b96ab" }} minTickGap={60} />
                <YAxis tick={{ fontSize: 11, fill: "#8b96ab" }} width={55} />
                <Tooltip
                  contentStyle={{ background: "#171f30", border: "1px solid #232c3f", fontSize: 12 }}
                  labelStyle={{ color: "#8b96ab" }}
                  formatter={(v, name) => [v.toFixed(3), STRATEGY_META[name]?.label || name]}
                />
                <Legend
                  formatter={(v) => <span style={{ color: "#c7cfdd", fontSize: 12 }}>{STRATEGY_META[v]?.label || v}</span>}
                />
                {result.crisis_window && (
                  <ReferenceArea
                    x1={result.crisis_window.start}
                    x2={result.crisis_window.end}
                    fill="#dc2626"
                    fillOpacity={0.08}
                    stroke="none"
                  />
                )}
                <ReferenceLine
                  x={result.train_test_split_date}
                  stroke="#8b96ab"
                  strokeDasharray="4 3"
                  label={{ value: "train → test", position: "insideTopLeft", fill: "#8b96ab", fontSize: 11 }}
                />
                <Line type="monotone" dataKey="equal_weight" stroke={STRATEGY_META.equal_weight.color} dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="mean_variance" stroke={STRATEGY_META.mean_variance.color} dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="adaptive" stroke={STRATEGY_META.adaptive.color} dot={false} strokeWidth={2} />
              </ComposedChart>
            </ResponsiveContainer>
            <p className="panel-desc" style={{ marginTop: 8 }}>
              Shaded red = known crisis window ({result.crisis_window?.start} → {result.crisis_window?.end}). Dashed
              line = where the adaptive policy's training data ends and out-of-sample evaluation begins.
            </p>
          </>
        )}
      </div>

      {result && (
        <div className="panel">
          <div className="controls-row" style={{ justifyContent: "space-between" }}>
            <h3 className="panel-title" style={{ margin: 0 }}>Performance Metrics</h3>
            <div className="controls-row" style={{ margin: 0 }}>
              <button
                className="btn-secondary"
                style={metricsView === "test" ? { borderColor: "#4f7cff", color: "#cdd9ff" } : {}}
                onClick={() => setMetricsView("test")}
              >
                Out-of-sample (test)
              </button>
              <button
                className="btn-secondary"
                style={metricsView === "full" ? { borderColor: "#4f7cff", color: "#cdd9ff" } : {}}
                onClick={() => setMetricsView("full")}
              >
                Full period
              </button>
            </div>
          </div>
          <MetricsTable
            metrics={metricsView === "test" ? result.metrics_test_period : result.metrics_full_period}
          />
        </div>
      )}

      {result && (
        <div className="panel">
          <h3 className="panel-title">Adaptive Policy — Allocation Over Time</h3>
          <p className="panel-desc">
            How the trained policy shifts weight across {result.assets.join(", ")} at each rebalance, in response to
            the regime probabilities and momentum/volatility features it observes.
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={weightSeries} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="#1c2436" vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#8b96ab" }} minTickGap={60} />
              <YAxis tick={{ fontSize: 11, fill: "#8b96ab" }} width={40} domain={[0, 1]} />
              <Tooltip
                contentStyle={{ background: "#171f30", border: "1px solid #232c3f", fontSize: 12 }}
                labelStyle={{ color: "#8b96ab" }}
                formatter={(v) => `${(v * 100).toFixed(1)}%`}
              />
              <Legend formatter={(v) => <span style={{ color: "#c7cfdd", fontSize: 12 }}>{v}</span>} />
              {result.assets.map((a, i) => (
                <Area
                  key={a}
                  type="stepAfter"
                  dataKey={a}
                  stackId="1"
                  stroke={ASSET_COLORS[i % ASSET_COLORS.length]}
                  fill={ASSET_COLORS[i % ASSET_COLORS.length]}
                  fillOpacity={0.55}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function MetricsTable({ metrics }) {
  const rows = [
    { key: "equal_weight", cls: "strategy-ew" },
    { key: "mean_variance", cls: "strategy-mv" },
    { key: "adaptive", cls: "strategy-adaptive" },
  ];
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="metrics-table">
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Total Return</th>
            <th>Annualized Return</th>
            <th>Annualized Vol</th>
            <th>Sharpe Ratio</th>
            <th>Max Drawdown</th>
            <th>Calmar Ratio</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ key, cls }) => {
            const m = metrics[key];
            return (
              <tr key={key}>
                <td className={cls}>{STRATEGY_META[key].label}</td>
                <td className={m.total_return >= 0 ? "pos" : "neg"}>{fmtPct(m.total_return)}</td>
                <td className={m.annualized_return >= 0 ? "pos" : "neg"}>{fmtPct(m.annualized_return)}</td>
                <td>{fmtPct(m.annualized_volatility)}</td>
                <td>{fmtNum(m.sharpe_ratio)}</td>
                <td className="neg">{fmtPct(m.max_drawdown)}</td>
                <td>{fmtNum(m.calmar_ratio)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
