import { useEffect, useMemo, useRef, useState } from "react";
import {
  ResponsiveContainer, ComposedChart, Line, XAxis, YAxis, Tooltip, ReferenceArea, CartesianGrid,
} from "recharts";
import api from "../api";
import { groupIntoSegments } from "../utils";
import EventComparisonReport from "./EventComparisonReport";

const REGIME_COLORS = { bull: "#16a34a", bear: "#eab308", crisis: "#dc2626" };
const REGIME_LABELS = { bull: "Bull / Calm", bear: "Bear / Declining", crisis: "Crisis / High-Volatility" };

const SYMBOL_OPTIONS = ["SPY", "QQQ", "GLD", "TLT", "^IXIC", "BTC-USD"];

export default function RegimeExplorer({ caseStudies }) {
  const [symbol, setSymbol] = useState("SPY");
  const [start, setStart] = useState("2019-01-01");
  const [end, setEnd] = useState("2021-12-31");
  const [data, setData] = useState(null);
  const [current, setCurrent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Kept separate from `load()`'s state: the comparison call is meaningfully
  // slower (first run trains the model) and can fail independently of the
  // regime chart -- a comparison failure/slowness must never blank out the
  // existing chart above it.
  const [comparison, setComparison] = useState(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonError, setComparisonError] = useState(null);
  const [comparisonRange, setComparisonRange] = useState(null);
  // Which event button (if any) is the current comparison targeted at --
  // null means "auto-detect nearest of the 3 events, or normal" mode.
  const [selectedEvent, setSelectedEvent] = useState(null);

  // Request-sequencing guards: if the user fires a second request (e.g.
  // clicking a case-study button) before an earlier, slower one resolves --
  // the comparison endpoint's first-ever call trains a model and can take
  // 20-40s, easily overlapping with a quick follow-up click -- the earlier
  // request's response must NOT be allowed to land after and overwrite the
  // newer one's result. Each call stamps a ticket; only the response whose
  // ticket is still the latest gets applied.
  const loadTicket = useRef(0);
  const comparisonTicket = useRef(0);

  // Accept explicit (sym, s, e) defaults rather than only closing over
  // `symbol`/`start`/`end` state, so callers can pass fresh values in the
  // same tick they call setSymbol/setStart/setEnd, before React re-renders.
  const load = (sym = symbol, s = start, e = end) => {
    const ticket = ++loadTicket.current;
    setLoading(true);
    setError(null);
    Promise.all([api.getRegimeHistory(sym, s, e), api.getCurrentRegime(sym)])
      .then(([hist, cur]) => {
        if (ticket !== loadTicket.current) return; // superseded by a newer request
        setData(hist);
        setCurrent(cur);
      })
      .catch((err) => {
        if (ticket !== loadTicket.current) return;
        setError(err.response?.data?.detail || err.message);
        // Drop the previous period's chart rather than leaving it up next to
        // the error. A short window (say the ~35-day COVID crash) is too
        // short for the HMM but perfectly valid for the comparison below, so
        // this path is easy to hit -- and a chart still showing the LAST
        // period, under an error, next to date inputs showing the new one,
        // reads as though the new period rendered.
        setData(null);
      })
      .finally(() => {
        if (ticket === loadTicket.current) setLoading(false);
      });
  };

  const loadComparison = (sym = symbol, s = start, e = end, eventId = null) => {
    const ticket = ++comparisonTicket.current;
    setComparisonLoading(true);
    setComparisonError(null);
    setComparisonRange({ symbol: sym, start: s, end: e });
    api
      .getRegimeComparison(sym, s, e, eventId)
      .then((result) => {
        if (ticket !== comparisonTicket.current) return;
        setComparison(result);
      })
      .catch((err) => {
        if (ticket !== comparisonTicket.current) return;
        setComparisonError(err.response?.data?.detail || err.message);
      })
      .finally(() => {
        if (ticket === comparisonTicket.current) setComparisonLoading(false);
      });
  };

  const analyze = () => {
    setSelectedEvent(null);
    load();
    loadComparison();
  };

  useEffect(() => { analyze(); /* eslint-disable-next-line */ }, []);

  const chartData = useMemo(() => {
    if (!data) return [];
    return data.dates.map((d, i) => ({ date: d, price: data.prices[i], regime: data.regime[i] }));
  }, [data]);

  const segments = useMemo(() => {
    if (!data) return [];
    return groupIntoSegments(data.dates, data.regime);
  }, [data]);

  // Deliberately does NOT touch symbol/start/end -- the user picks their own
  // symbol + period with the controls below, and this just asks "how close
  // is THAT period to this one named historical event", using whichever
  // period is currently selected. Clicking a different event re-scores the
  // exact same period against a different reference cluster.
  const compareToEvent = (cs) => {
    setSelectedEvent(cs.id);
    load(symbol, start, end);
    loadComparison(symbol, start, end, cs.id);
  };

  return (
    <div>
      <div className="panel">
        <h3 className="panel-title">Market Regime Detection</h3>
        <p className="panel-desc">
          A 3-state Gaussian Hidden Markov Model, fit on daily return / rolling volatility / drawdown features,
          classifies each trading day into <b>Bull</b>, <b>Bear</b>, or <b>Crisis</b> regimes — fully unsupervised,
          no labeled crash dates given to the model.
        </p>

        <div className="controls-row">
          <select className="select" value={symbol} onChange={(e) => setSymbol(e.target.value)}>
            {SYMBOL_OPTIONS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <input className="input" type="date" value={start} onChange={(e) => setStart(e.target.value)} />
          <input className="input" type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
          <button className="btn" onClick={analyze} disabled={loading}>{loading ? "Analyzing…" : "Analyze"}</button>
        </div>

        <p className="panel-desc" style={{ marginTop: 4 }}>
          Pick a symbol + period above, then click one of the historical events below to score how closely
          <i> this exact period</i> resembles that event specifically (your symbol/dates stay as chosen).
        </p>
        <div className="controls-row">
          {caseStudies?.map((cs) => (
            <button
              key={cs.id}
              className="btn-secondary"
              style={selectedEvent === cs.id ? { borderColor: "#4f7cff", color: "#4f7cff" } : undefined}
              onClick={() => compareToEvent(cs)}
              disabled={comparisonLoading}
            >
              {cs.label}
            </button>
          ))}
        </div>

        {error && (
          <div className="notice error">
            Couldn't draw the day-by-day regime chart for this period: {error}
            {" "}The Historical Pattern Comparison below still works on short windows and is unaffected.
          </div>
        )}

        {current && !current.error && (
          <div className="controls-row" style={{ marginTop: 4 }}>
            <span
              className="badge"
              style={{ background: `${REGIME_COLORS[current.regime]}22`, color: REGIME_COLORS[current.regime] }}
            >
              <span className="dot" style={{ background: REGIME_COLORS[current.regime] }} />
              Current regime ({current.as_of}): {current.label} — {(current.confidence * 100).toFixed(0)}% confidence
            </span>
          </div>
        )}

        {loading && !data && (
          <div className="spinner-row"><span className="spinner" /> Fetching history & fitting HMM…</div>
        )}

        {data && (
          <>
            <ResponsiveContainer width="100%" height={340}>
              <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#1c2436" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#8b96ab" }} minTickGap={60} />
                <YAxis tick={{ fontSize: 11, fill: "#8b96ab" }} domain={["auto", "auto"]} width={55} />
                <Tooltip
                  contentStyle={{ background: "#171f30", border: "1px solid #232c3f", fontSize: 12 }}
                  labelStyle={{ color: "#8b96ab" }}
                />
                {segments.map((seg, i) => (
                  <ReferenceArea
                    key={i}
                    x1={seg.start}
                    x2={seg.end}
                    fill={REGIME_COLORS[seg.label]}
                    fillOpacity={0.14}
                    stroke="none"
                  />
                ))}
                <Line type="monotone" dataKey="price" stroke="#e7ecf5" strokeWidth={1.5} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
            <div className="legend-row">
              {Object.entries(REGIME_LABELS).map(([k, label]) => (
                <span key={k} className="legend-item">
                  <span className="legend-swatch" style={{ background: REGIME_COLORS[k] }} /> {label}
                </span>
              ))}
            </div>
          </>
        )}
      </div>

      <EventComparisonReport
        data={comparison} loading={comparisonLoading} error={comparisonError} range={comparisonRange}
      />
    </div>
  );
}
