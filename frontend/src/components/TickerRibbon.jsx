import { useEffect, useState } from "react";
import api from "../api";

export default function TickerRibbon() {
  const [quotes, setQuotes] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    const load = () => {
      api
        .getLiveMarket()
        .then((data) => alive && setQuotes(data.quotes))
        .catch((e) => alive && setError(e.message));
    };
    load();
    const id = setInterval(load, 30000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  if (error && !quotes) {
    return <div className="ticker-ribbon"><span className="ticker-error">Live feed unreachable: {error}</span></div>;
  }
  if (!quotes) {
    return <div className="ticker-ribbon"><span className="ticker-error">Loading live quotes…</span></div>;
  }

  return (
    <div className="ticker-ribbon">
      {quotes.map((q) => (
        <div className="ticker-item" key={q.symbol}>
          <span className="ticker-symbol">{q.symbol}</span>
          {q.error ? (
            <span className="ticker-error">unavailable</span>
          ) : (
            <>
              <span className="ticker-price">{q.price?.toFixed(2)}</span>
              <span className={`ticker-change ${q.change_pct >= 0 ? "up" : "down"}`}>
                {q.change_pct >= 0 ? "▲" : "▼"} {Math.abs(q.change_pct ?? 0).toFixed(2)}%
              </span>
            </>
          )}
        </div>
      ))}
    </div>
  );
}
