"""Diagrams for the project report: Gantt chart, data flow diagrams, use case, class and sequence diagrams.
Run:  PYTHONPATH=/tmp/apo_mpl python3 paper/scripts/fig_report_diagrams.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

FIG = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({"font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 9})
NAVY, TEAL, AMBER, GREY, RED, GREEN, PURPLE = "#1d3557", "#2a7f8e", "#d98e04", "#6b7280", "#b23a48", "#2f7d4f", "#6c4ba6"
LIGHT = {NAVY: "#e3e9f2", TEAL: "#dff0f2", AMBER: "#fbf0d8", GREY: "#eceef1", RED: "#f5e1e4", GREEN: "#e0f0e6", PURPLE: "#ebe4f5"}


def arrow(ax, p0, p1, color="#333", style="-|>", lw=1.1, rad=0.0, ls="-", ms=10, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------------ Gantt chart (from the Review-1 work plan)
def gantt():
    tasks = ["Literature survey and problem formulation", "Data collection and preprocessing",
             "Model design and algorithm development", "Implementation and backtesting",
             "Performance evaluation and tuning", "Documentation and report writing"]
    start = [0, 1, 3, 7, 11, 13]
    dur = [2, 3, 5, 5, 3, 3]
    cols = [TEAL, GREEN, PURPLE, AMBER, RED, NAVY]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    for i, (t, s, d, c) in enumerate(zip(tasks, start, dur, cols)):
        y = len(tasks) - 1 - i
        ax.barh(y, d, left=s, height=0.55, color=c, edgecolor="#222", lw=0.6)
        ax.text(s + d / 2, y, f"{d} wk", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks[::-1], fontsize=8.5)
    ax.set_xlim(0, 16)
    ax.set_xticks(range(0, 17, 2))
    ax.set_xlabel("Weeks from project start (four-month plan)", fontsize=9)
    for w, m in ((0, "Month 1"), (4, "Month 2"), (8, "Month 3"), (12, "Month 4")):
        ax.axvline(w, color="#999", lw=0.8, ls="--")
        ax.text(w + 2, len(tasks) - 0.35, m, ha="center", fontsize=8.5, color=GREY)
    ax.axvline(16, color="#999", lw=0.8, ls="--")
    ax.set_ylim(-0.6, len(tasks) - 0.1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    save(fig, "fig_gantt.png")


# ------------------------------------------------------------------ DFD helpers
def entity(ax, x, y, w, h, text, color=GREY):
    ax.add_patch(Rectangle((x, y), w, h, fc=LIGHT[color], ec=color, lw=1.4, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=4)


def process(ax, cx, cy, rx, ry, num, text, color=NAVY, fs=7.8):
    ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry, fc=LIGHT[color], ec=color, lw=1.4, zorder=2))
    ax.text(cx, cy + ry * 0.5, num, ha="center", va="center", fontsize=8, fontweight="bold", color=color, zorder=4)
    ax.text(cx, cy - ry * 0.1, text, ha="center", va="center", fontsize=fs, linespacing=1.2, zorder=4)


def store(ax, x, y, w, h, tag, text, color=AMBER):
    ax.add_patch(Rectangle((x, y), w, h, fc=LIGHT[color], ec="none", zorder=2))
    ax.plot([x, x + w], [y + h, y + h], color=color, lw=1.6, zorder=3)
    ax.plot([x, x + w], [y, y], color=color, lw=1.6, zorder=3)
    ax.plot([x + 0.55, x + 0.55], [y, y + h], color=color, lw=1.2, zorder=3)
    ax.text(x + 0.27, y + h / 2, tag, ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
    ax.text(x + 0.55 + (w - 0.55) / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6, zorder=4, linespacing=1.15)


def lab(ax, x, y, text, fs=7.2, ha="center", **kw):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, style="italic", color="#222",
            bbox=dict(fc="white", ec="none", pad=0.6), zorder=5, **kw)


def dfd0():
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    entity(ax, 0.15, 1.95, 1.9, 1.1, "User\n(analyst / investor)", TEAL)
    entity(ax, 7.95, 1.95, 1.9, 1.1, "Yahoo Finance\nchart API", GREEN)
    process(ax, 5.0, 2.5, 1.55, 1.15, "0", "Adaptive Portfolio\nOptimizer\n(web application)", NAVY, fs=8.5)
    arrow(ax, (2.05, 2.85), (3.5, 2.85)); lab(ax, 2.78, 3.28, "symbol, date range,\nevent, backtest settings")
    arrow(ax, (3.5, 2.15), (2.05, 2.15)); lab(ax, 2.78, 1.62, "regime charts, comparison\nreport, backtest results")
    arrow(ax, (6.5, 2.85), (7.95, 2.85)); lab(ax, 7.2, 3.3, "price request\n(symbol, period)")
    arrow(ax, (7.95, 2.15), (6.5, 2.15)); lab(ax, 7.2, 1.6, "daily OHLC and\nadjusted close")
    save(fig, "fig_dfd0.png")


def dfd1():
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 9.6); ax.axis("off")
    entity(ax, 0.1, 7.9, 1.6, 1.0, "Yahoo\nFinance", GREEN)
    process(ax, 3.8, 8.4, 1.3, 0.9, "1.0", "Acquire and\nclean prices", GREEN)
    store(ax, 6.3, 8.05, 3.5, 0.7, "D1", "Price cache (CSV)")
    process(ax, 1.9, 5.5, 1.35, 0.9, "2.0", "Detect market\nregimes (HMM)", AMBER)
    process(ax, 5.0, 5.5, 1.4, 0.9, "3.0", "Compare with\nhistorical crises", PURPLE)
    process(ax, 8.1, 5.5, 1.35, 0.9, "4.0", "Run strategy\nbacktest", RED)
    store(ax, 2.7, 3.35, 2.3, 0.75, "D2", "Reference corpus,\nfitted matcher")
    store(ax, 7.0, 3.35, 2.8, 0.75, "D3", "Backtest result cache")
    process(ax, 5.2, 1.2, 1.5, 0.85, "5.0", "Present results\n(React dashboard)", NAVY)
    entity(ax, 0.1, 0.7, 1.6, 1.0, "User", TEAL)
    arrow(ax, (1.7, 8.4), (2.5, 8.4)); lab(ax, 2.1, 9.0, "prices")
    arrow(ax, (5.1, 8.4), (6.3, 8.4)); lab(ax, 5.7, 8.85, "write")
    arrow(ax, (6.7, 8.05), (2.5, 6.3)); arrow(ax, (7.3, 8.05), (5.4, 6.4)); arrow(ax, (8.6, 8.05), (8.3, 6.4))
    lab(ax, 7.3, 7.35, "stored prices read by 2.0, 3.0, 4.0", fs=6.8)
    arrow(ax, (4.7, 4.6), (4.7, 4.1)); arrow(ax, (4.1, 4.1), (4.1, 4.6))
    lab(ax, 3.2, 4.45, "embed,\nscore", fs=6.6)
    arrow(ax, (8.0, 4.6), (8.0, 4.1)); arrow(ax, (8.6, 4.1), (8.6, 4.6))
    lab(ax, 9.45, 4.45, "result", fs=6.6)
    arrow(ax, (1.7, 4.65), (3.9, 1.6), rad=0.0); lab(ax, 2.2, 3.1, "regime labels,\nprobabilities", fs=6.8)
    arrow(ax, (5.6, 4.6), (5.5, 2.05)); lab(ax, 6.2, 3.0, "comparison\nreport", fs=6.8)
    arrow(ax, (8.3, 3.35), (6.5, 1.65)); lab(ax, 8.0, 2.3, "metrics, curves,\nweights", fs=6.8)
    arrow(ax, (1.7, 1.45), (3.7, 1.35)); lab(ax, 2.7, 1.95, "symbol, dates,\nevent, settings", fs=6.6)
    arrow(ax, (3.7, 0.9), (1.7, 0.95)); lab(ax, 2.7, 0.35, "charts, report, tables", fs=6.8)
    save(fig, "fig_dfd1.png")


def usecase():
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.add_patch(Rectangle((2.2, 0.2), 5.6, 7.6, fc="none", ec=NAVY, lw=1.3))
    ax.text(5.0, 7.55, "Adaptive Portfolio Optimizer", ha="center", fontsize=9.5, fontweight="bold", color=NAVY)
    cases = ["View live market ribbon", "View regime history chart", "View current regime of a symbol",
             "Compare a date range with historical crises", "Choose a target crisis event",
             "Read comparison report and overlay", "Run backtest for a case study", "Run backtest on custom assets",
             "Inspect allocations and metrics", "Read method and tech stack"]
    ys = [6.85 - i * 0.66 for i in range(len(cases))]
    for c, y in zip(cases, ys):
        ax.add_patch(Ellipse((5.0, y), 4.6, 0.52, fc=LIGHT[TEAL], ec=TEAL, lw=1.1))
        ax.text(5.0, y, c, ha="center", va="center", fontsize=7.8)

    def actor(x, y, name, color):
        ax.add_patch(plt.Circle((x, y + 0.55), 0.17, fc="white", ec=color, lw=1.4))
        ax.plot([x, x], [y + 0.38, y - 0.15], color=color, lw=1.4)
        ax.plot([x - 0.3, x + 0.3], [y + 0.22, y + 0.22], color=color, lw=1.4)
        ax.plot([x, x - 0.25], [y - 0.15, y - 0.6], color=color, lw=1.4)
        ax.plot([x, x + 0.25], [y - 0.15, y - 0.6], color=color, lw=1.4)
        ax.text(x, y - 1.25, name, ha="center", va="center", fontsize=8.5, fontweight="bold")
    actor(1.0, 4.0, "User\n(analyst / investor)", TEAL)
    actor(9.0, 4.0, "Yahoo Finance\n(data provider)", GREEN)
    for y in ys:
        arrow(ax, (1.3, 4.0), (2.75, y), color="#555", style="-", lw=0.8, z=1)
    for y in (ys[0], ys[1], ys[3], ys[6], ys[7]):
        arrow(ax, (8.7, 4.0), (7.3, y), color=GREEN, style="-", lw=0.8, ls=(0, (3, 2)), z=1)
    ax.text(9.0, 1.9, "dashed links: use cases\nthat need price data", ha="center", fontsize=7.4, style="italic", color=GREY)
    save(fig, "fig_usecase.png")


def uml_class(ax, x, y, w, title, attrs, methods, color=NAVY, fs=7.2):
    lh = 0.22
    h = 0.42 + lh * (len(attrs) + len(methods)) + 0.18
    ax.add_patch(Rectangle((x, y - h), w, h, fc="white", ec=color, lw=1.3, zorder=2))
    ax.add_patch(Rectangle((x, y - 0.42), w, 0.42, fc=LIGHT[color], ec=color, lw=1.3, zorder=2))
    ax.text(x + w / 2, y - 0.21, title, ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
    ya = y - 0.42 - 0.04
    for a in attrs:
        ya -= lh
        ax.text(x + 0.08, ya + lh / 2, a, fontsize=fs, va="center", zorder=4)
    sep = y - 0.42 - lh * len(attrs) - 0.09
    ax.plot([x, x + w], [sep, sep], color=color, lw=1.0, zorder=3)
    ym = sep - 0.04
    for m in methods:
        ym -= lh
        ax.text(x + 0.08, ym + lh / 2, m, fontsize=fs, va="center", zorder=4)
    return dict(l=(x, y - h / 2), r=(x + w, y - h / 2), t=(x + w / 2, y), b=(x + w / 2, y - h))


def box_h(attrs, methods):
    return 0.42 + 0.22 * (len(attrs) + len(methods)) + 0.18


def classdiag():
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 12.6); ax.axis("off")
    A, B, C = 0.1, 4.45, 8.8
    W = 3.5
    spec = {}

    def put(key, x, y, title, attrs, methods, color):
        spec[key] = uml_class(ax, x, y, W, title, attrs, methods, color)

    r1, r2, r3, r4 = 12.4, 8.4, 4.6, 1.9
    put("req", A, r1, "BacktestRequest", ["case_study: str | None", "assets: list[str] | None", "start, end: str | None",
        "primary_symbol: str | None", "rebalance_freq: int = 21", "cost_bps: float = 5.0", "es_generations: int = 40", "es_population: int = 32"], [], TEAL)
    put("api", B, r1, "FastAPI app (main.py)", ["_backtest_cache: dict"], ["health(), case_studies()", "live_market()", "regimes_current()",
        "regimes_history()", "regimes_compare(...)", "backtest(req)"], TEAL)
    put("md", C, r1, "MarketData (module)", ["CACHE_TTL = 15 min"], ["get_history(symbol, start, end)", "get_multi_history(symbols, ...)",
        "get_live_quotes(symbols)"], GREEN)
    put("reg", A, r2, "RegimeResult", ["dates: DatetimeIndex", "state_key: ndarray", "probabilities: DataFrame", "state_stats: dict"],
        ["detect_regimes(prices)", "current_regime(prices)"], AMBER)
    put("eng", B, r2, "BacktestEngine (module)", ["REBALANCE_FREQ = 21", "COST_BPS = 5"], ["run_backtest(prices, ...)", "_metrics(values, dates)",
        "equal_weights(n)", "mean_variance_weights(r)", "train_adaptive_policy(...)"], RED)
    put("mat", C, r2, "Matcher", ["encoder: ShapeEncoder", "scaler_mean, scaler_std", "corpus: list[CorpusWindow]", "embeddings: (n, 16)",
        "cluster_meta: dict", "fit_method: str", "label_scales: dict"], ["get_matcher()", "match_window(...)"], NAVY)
    put("pol", A, r3, "PolicyNet", ["w1, b1: ndarray", "w2, b2: ndarray"], ["random_init(state_dim, n_assets)", "flatten(), from_flat(flat)",
        "act(state) : weights"], RED)
    put("cw", B, r3, "CorpusWindow", ["symbol, label: str", "start, end: str", "length: int", "resampled: ndarray (40, 3)",
        "forward_return_21d: float | None"], [], NAVY)
    put("enc", C, r3, "ShapeEncoder (nn.Module)", ["input_proj: Linear(3, 32)", "pos_embed: Parameter", "encoder: TransformerEncoder x2",
        "out_proj: Linear(32, 16)"], ["forward(x) : (batch, 16)"], PURPLE)
    put("ae", C, r4, "ShapeAutoencoder (nn.Module)", ["encoder: ShapeEncoder", "decoder: TransformerEncoder", "reconstruct: Linear(32, 3)"],
        ["forward(x) : (batch, 40, 3)"], PURPLE)

    def edge(k1, s1, k2, s2, color="#333", style="-|>", ls="-"):
        arrow(ax, spec[k1][s1], spec[k2][s2], color=color, style=style, ls=ls, lw=1.1)

    edge("api", "l", "req", "r", TEAL)
    edge("api", "r", "md", "l", GREEN)
    edge("api", "b", "eng", "t", RED)
    arrow(ax, (B + 0.4, spec["api"]["b"][1]), spec["reg"]["t"], color=AMBER, lw=1.1)
    arrow(ax, (B + W - 0.4, spec["api"]["b"][1]), spec["mat"]["t"], color=NAVY, lw=1.1)
    edge("eng", "l", "reg", "r", AMBER)
    arrow(ax, (B + 0.5, spec["eng"]["b"][1]), spec["pol"]["t"], color=RED, lw=1.1)
    edge("mat", "b", "enc", "t", PURPLE)
    arrow(ax, (C, spec["mat"]["b"][1] + 0.6), spec["cw"]["r"], color=NAVY, lw=1.1)
    edge("ae", "t", "enc", "b", PURPLE)
    ax.text(C + W / 2 + 0.15, (spec["ae"]["t"][1] + spec["enc"]["b"][1]) / 2, "contains", fontsize=6.8, style="italic", color=PURPLE,
            ha="left", va="center")
    ax.text(0.3, 2.3, "Arrows point from a component to the one\nit uses. All classes live in the backend\nPython package app/.",
            fontsize=7.6, style="italic", color=GREY, ha="left", va="center")
    save(fig, "fig_class.png")


def lifelines(ax, names, xs, top, bottom, colors):
    for n, x, c in zip(names, xs, colors):
        ax.add_patch(FancyBboxPatch((x - 0.6, top), 1.2, 0.6, boxstyle="round,pad=0.02,rounding_size=0.06", fc=LIGHT[c], ec=c, lw=1.3, zorder=3))
        ax.text(x, top + 0.3, n, ha="center", va="center", fontsize=7.0, fontweight="bold", zorder=4, linespacing=1.1)
        ax.plot([x, x], [top, bottom], color="#888", lw=0.9, ls=(0, (4, 3)), zorder=1)


def msg(ax, x0, x1, y, text, dashed=False, above=True, color="#222", fs=7.2):
    arrow(ax, (x0, y), (x1, y), color=color, ls=(0, (4, 2)) if dashed else "-", lw=1.0, ms=9)
    ax.text((x0 + x1) / 2, y + 0.1, text, ha="center", va="bottom", fontsize=fs, zorder=5, bbox=dict(fc="white", ec="none", pad=0.8))


def seq_compare():
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.set_xlim(0, 11.1); ax.set_ylim(-0.6, 8.3); ax.axis("off")
    names = ["User", "React\ndashboard", "FastAPI\n(compare route)", "event_matcher", "market_data", "Yahoo / cache", "embedding\n(Transformer)"]
    xs = [0.7, 2.3, 4.1, 6.0, 7.7, 9.2, 10.4]
    lifelines(ax, names, xs, 7.55, 0.2, [TEAL, TEAL, NAVY, PURPLE, GREEN, GREEN, PURPLE])
    y = 7.15
    steps = [
        (0, 1, "choose symbol, dates, event; press Compare", False),
        (1, 2, "GET compare?symbol&start&end&event_id", False),
        (2, 3, "match_window(symbol, start, end, target)", False),
        (3, 3, None, False),
        (3, 4, "get_history(symbol, start, end)", False),
        (4, 5, "request prices (or read 15-min cache)", False),
        (5, 4, "daily adjusted close", True),
        (4, 3, "price series", True),
        (3, 6, "build_channels, resample 40 x 3, embed", False),
        (6, 3, "16-d embedding", True),
        (3, 3, "score labels: 8-NN spacing, kernel, purity", False),
        (3, 2, "match, confidence, overlay, stats, insights", True),
        (2, 1, "JSON report", True),
        (1, 0, "chart overlay and insight list", True),
    ]
    for a, b, t, d in steps:
        if t is None:
            y += 0.0
            continue
        if a == b:
            ax.add_patch(Rectangle((xs[a] - 0.08, y - 0.3), 0.16, 0.4, fc="white", ec="#555", lw=0.9, zorder=3))
            ax.text(xs[a] + 0.2, y - 0.1, t, fontsize=7.1, va="center", ha="left", bbox=dict(fc="white", ec="none", pad=0.8), zorder=5)
            y -= 0.7
            continue
        msg(ax, xs[a], xs[b], y, t, dashed=d)
        y -= 0.5
    ax.text(5.5, -0.4, "First call builds the matcher (about 25 s); later calls take about 0.1 s.", ha="center", fontsize=7.4, style="italic", color=GREY)
    save(fig, "fig_seq_compare.png")


def seq_backtest():
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.set_xlim(0, 11.1); ax.set_ylim(-0.6, 8.3); ax.axis("off")
    names = ["User", "React\ndashboard", "FastAPI\n/api/backtest", "market_data", "regime\n(HMM)", "allocator\n(ES training)", "backtest\nengine"]
    xs = [0.7, 2.3, 4.0, 5.7, 7.3, 8.9, 10.4]
    lifelines(ax, names, xs, 7.55, 0.2, [TEAL, TEAL, NAVY, GREEN, AMBER, RED, RED])
    y = 7.15
    steps = [
        (0, 1, "pick case study, press Run backtest", False),
        (1, 2, "POST {case_study, cost_bps, ...}", False),
        (2, 2, "check cache key (SHA-256 of settings)", False),
        (2, 3, "get_multi_history(assets, start, end)", False),
        (3, 2, "price panel (live, cached or labelled fallback)", True),
        (2, 6, "run_backtest(panel, split_date, ...)", False),
        (6, 4, "detect_regimes(primary symbol)", False),
        (4, 6, "regime probabilities", True),
        (6, 5, "train_adaptive_policy (train segment only)", False),
        (5, 6, "policy parameters", True),
        (6, 6, "simulate three strategies, apply 5 bps cost", False),
        (6, 2, "values, weights, metrics, regimes", True),
        (2, 1, "JSON (also cached)", True),
        (1, 0, "curves, metrics table, allocation chart", True),
    ]
    for a, b, t, d in steps:
        if a == b:
            ax.add_patch(Rectangle((xs[a] - 0.08, y - 0.3), 0.16, 0.4, fc="white", ec="#555", lw=0.9, zorder=3))
            ax.text(xs[a] - 0.2, y - 0.1, t, fontsize=7.1, va="center", ha="right", bbox=dict(fc="white", ec="none", pad=0.8), zorder=5)
            y -= 0.7
            continue
        msg(ax, xs[a], xs[b], y, t, dashed=d)
        y -= 0.5
    save(fig, "fig_seq_backtest.png")


if __name__ == "__main__":
    gantt(); dfd0(); dfd1(); usecase(); classdiag(); seq_compare(); seq_backtest()
