import { useMemo } from "react";
import {
  ResponsiveContainer, ComposedChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend,
} from "recharts";
import { fmtNum, fmtPct } from "../utils";

const EVENT_COLORS = { dotcom: "#f59e0b", gfc: "#dc2626", covid: "#8b5cf6", normal: "#16a34a" };

function buildOverlayData(queryPath, referencePath) {
  const n = Math.max(queryPath?.length || 0, referencePath?.length || 0);
  const out = [];
  for (let i = 0; i < n; i++) {
    out.push({
      pct: Math.round((i / (n - 1)) * 100),
      selected: queryPath?.[i] ?? null,
      reference: referencePath?.[i] ?? null,
    });
  }
  return out;
}

export default function EventComparisonReport({ data, loading, error, range }) {
  const overlayData = useMemo(
    () => (data ? buildOverlayData(data.query_stats.cumulative_return_path, data.reference.cumulative_return_path) : []),
    [data]
  );

  const eventKey = data?.match?.event_id || "normal";
  const badgeColor = EVENT_COLORS[eventKey] || EVENT_COLORS.normal;

  return (
    <div className="panel">
      <h3 className="panel-title">Historical Pattern Comparison</h3>
      <p className="panel-desc">
        Resamples the selected period's return / volatility / drawdown path to a fixed-length shape and embeds it with
        a small Transformer sequence encoder. The reference corpus — windows drawn from the dot-com bust, the 2008 GFC,
        the COVID crash, and calm reference periods, grouped by density-based (HDBSCAN) clustering — is then searched
        for this shape's nearest labelled neighbours, scoring how close the period sits to genuine windows of each
        event and how unambiguously that neighbourhood belongs to one.
      </p>
      {range && (
        <p className="panel-desc" style={{ marginTop: -8, fontStyle: "italic" }}>
          Comparing {range.symbol ? `${range.symbol}, ` : ""}{range.start} to {range.end}
          {data?.window_trading_days ? ` (${data.window_trading_days} trading days)` : ""}
          {data?.match?.event_id ? ` against the ${data.match.event_label} reference cluster.` : "."}
        </p>
      )}

      {loading && (
        <div className="spinner-row">
          <span className="spinner" /> Comparing shape against historical case studies… (first run trains the model,
          ~20–40s; instant afterward)
        </div>
      )}

      {error && <div className="notice">Couldn't run the comparison: {error}</div>}

      {data && !loading && (
        <>
          <div className="controls-row" style={{ marginTop: 4 }}>
            <span className="badge" style={{ background: `${badgeColor}22`, color: badgeColor }}>
              <span className="dot" style={{ background: badgeColor }} />
              {data.match.event_id
                ? `${data.match.matched ? "Closely resembles" : "Weak resemblance to"}: ${data.match.event_label} — ${(data.match.confidence * 100).toFixed(0)}% confidence`
                : `No strong resemblance to a known historical crisis — normal / typical period (${(data.match.confidence * 100).toFixed(0)}% confidence in that reading)`}
            </span>
          </div>

          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={overlayData} margin={{ top: 14, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="#1c2436" vertical={false} />
              <XAxis
                dataKey="pct" tick={{ fontSize: 11, fill: "#8b96ab" }}
                label={{ value: "% through the period", position: "insideBottom", offset: -2, fill: "#8b96ab", fontSize: 11 }}
              />
              <YAxis
                tick={{ fontSize: 11, fill: "#8b96ab" }} width={55}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{ background: "#171f30", border: "1px solid #232c3f", fontSize: 12 }}
                labelStyle={{ color: "#8b96ab" }}
                formatter={(v, name) => [v == null ? "—" : `${(v * 100).toFixed(1)}%`, name]}
                labelFormatter={(v) => `${v}% through period`}
              />
              <Legend formatter={(v) => <span style={{ color: "#c7cfdd", fontSize: 12 }}>{v}</span>} />
              <Line type="monotone" dataKey="selected" name="Selected period" stroke="#4f7cff" strokeWidth={2} dot={false} />
              <Line
                type="monotone" dataKey="reference" name={data.match.event_id ? data.reference.event_label : "Typical normal period"}
                stroke={badgeColor} strokeWidth={2} strokeDasharray="5 3" dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>

          <div className="stat-chip-row">
            <div className="stat-chip">
              <div className="stat-label">Annualized Return</div>
              <div className={`stat-value ${data.query_stats.annualized_return >= 0 ? "pos" : "neg"}`}>
                {fmtPct(data.query_stats.annualized_return)}
              </div>
            </div>
            <div className="stat-chip">
              <div className="stat-label">Annualized Vol</div>
              <div className="stat-value">{fmtPct(data.query_stats.annualized_volatility)}</div>
            </div>
            <div className="stat-chip">
              <div className="stat-label">Max Drawdown</div>
              <div className="stat-value neg">{fmtPct(data.query_stats.max_drawdown)}</div>
            </div>
            <div className="stat-chip">
              {/* Without an event match this number is confidence in the "normal /
                  typical" reading, not in a crisis resemblance -- label it as such
                  rather than showing a bare "match confidence" the badge contradicts. */}
              <div className="stat-label">{data.match.event_id ? "Match Confidence" : "Confidence (Normal)"}</div>
              <div className="stat-value">{fmtNum(data.match.confidence * 100, 0)}%</div>
            </div>
          </div>

          <ul className="insight-list">
            {data.insights.map((text, i) => (
              <li key={i} className="insight-item">{text}</li>
            ))}
          </ul>

          <div className="disclaimer-text">{data.disclaimer}</div>
        </>
      )}
    </div>
  );
}
