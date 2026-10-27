import { useEffect, useState } from "react";
import TickerRibbon from "./components/TickerRibbon";
import RegimeExplorer from "./components/RegimeExplorer";
import BacktestPanel from "./components/BacktestPanel";
import AboutPanel from "./components/AboutPanel";
import api from "./api";

const TABS = [
  { id: "backtest", label: "Portfolio Backtest" },
  { id: "regimes", label: "Regime Detection" },
  { id: "about", label: "About / Tech Stack" },
];

export default function App() {
  const [tab, setTab] = useState("backtest");
  const [caseStudies, setCaseStudies] = useState(null);
  const [defaultAssets, setDefaultAssets] = useState(null);

  useEffect(() => {
    api.getCaseStudies().then(setCaseStudies).catch(() => setCaseStudies([]));
    api.getDefaultAssets().then((d) => setDefaultAssets(d.assets)).catch(() => {});
  }, []);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-top">
          <div>
            <h1 className="app-title">Adaptive Portfolio Optimizer</h1>
            <div className="app-subtitle">
              Regime-aware, RL-trained multi-asset allocation — benchmarked across the dot-com bust, the 2008 GFC,
              and the COVID crash
            </div>
          </div>
          <div className="tabs">
            {TABS.map((t) => (
              <button
                key={t.id}
                className={`tab-btn ${tab === t.id ? "active" : ""}`}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
        <TickerRibbon />
      </header>

      <main className="app-main">
        {tab === "backtest" && <BacktestPanel caseStudies={caseStudies} defaultAssets={defaultAssets} />}
        {tab === "regimes" && <RegimeExplorer caseStudies={caseStudies} />}
        {tab === "about" && <AboutPanel />}
      </main>
    </div>
  );
}
