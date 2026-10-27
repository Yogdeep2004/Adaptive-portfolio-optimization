import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const client = axios.create({ baseURL: BASE_URL, timeout: 60000 });

export const api = {
  getCaseStudies: () => client.get("/api/case-studies").then((r) => r.data),
  getDefaultAssets: () => client.get("/api/assets/default").then((r) => r.data),
  getLiveMarket: () => client.get("/api/market/live").then((r) => r.data),
  getCurrentRegime: (symbol = "SPY") =>
    client.get("/api/regimes/current", { params: { symbol } }).then((r) => r.data),
  getRegimeHistory: (symbol, start, end) =>
    client.get("/api/regimes/history", { params: { symbol, start, end } }).then((r) => r.data),
  getRegimeComparison: (symbol, start, end, eventId) =>
    client
      .get("/api/regimes/compare", { params: { symbol, start, end, event_id: eventId || undefined } })
      .then((r) => r.data),
  runBacktest: (payload) => client.post("/api/backtest", payload).then((r) => r.data),
};

export default api;
