"""Architecture / pipeline diagrams (matplotlib, vector-style boxes) for the manuscript."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

FIG = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({"font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 9})

NAVY, TEAL, AMBER, GREY, RED, GREEN, PURPLE = "#1d3557", "#2a7f8e", "#d98e04", "#6b7280", "#b23a48", "#2f7d4f", "#6c4ba6"
LIGHT = {NAVY: "#e3e9f2", TEAL: "#dff0f2", AMBER: "#fbf0d8", GREY: "#eceef1", RED: "#f5e1e4", GREEN: "#e0f0e6",
         PURPLE: "#ebe4f5"}


def box(ax, x, y, w, h, text, color=NAVY, fs=8.5, bold=False, align="center", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=LIGHT[color],
                                ec=color, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#111",
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p0, p1, color="#333", style="-|>", lw=1.2, rad=0.0, ls="-", ms=11):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


def layer(ax, x, y, w, h, title, color):
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=color, lw=1.0, ls=(0, (4, 3))))
    ax.text(x + 0.12, y + h - 0.12, title, ha="left", va="top", fontsize=8.5, color=color, fontweight="bold")


# ============ Figure 1: system architecture ============
fig, ax = plt.subplots(figsize=(7.4, 6.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 10.8); ax.axis("off")

layer(ax, 0.2, 8.45, 9.6, 2.15, "Presentation layer: React 19 · Vite 8 · Recharts (browser)", TEAL)
for i, t in enumerate(["Live ticker\nribbon", "Portfolio\nBacktest tab", "Regime Detection\ntab", "Pattern Comparison\nreport", "About /\nTech stack"]):
    box(ax, 0.4 + i * 1.9, 8.65, 1.75, 1.1, t, TEAL, fs=7.8)

layer(ax, 0.2, 5.55, 9.6, 2.5, "Service layer: FastAPI (Python 3.12), REST / JSON", NAVY)
apis = ["GET\n/api/market/live", "GET /api/regimes/\nhistory · current", "GET /api/regimes/\ncompare[?event_id]", "POST\n/api/backtest", "GET /api/case-\nstudies · assets"]
for i, t in enumerate(apis):
    box(ax, 0.4 + i * 1.9, 6.15, 1.75, 1.1, t, NAVY, fs=6.9)
ax.text(5, 5.75, "In-memory caches: backtest results · fitted matcher singleton (built once, ≈25 s)", ha="center", fontsize=7.6,
        color=GREY, style="italic")

layer(ax, 0.2, 2.3, 9.6, 2.85, "Analytics layer", AMBER)
box(ax, 0.4, 2.55, 2.1, 1.6, "Regime detection\nGaussian HMM\n(3 states, hmmlearn)", AMBER, fs=7.6)
box(ax, 2.7, 2.55, 2.7, 1.6, "Pattern matcher\nresample → Transformer\nencoder (PyTorch) →\nHDBSCAN + k-NN scoring", AMBER, fs=7.4)
box(ax, 5.6, 2.55, 2.0, 1.6, "Adaptive allocator\nMLP policy trained\nby Evolution\nStrategies (NumPy)", AMBER, fs=7.4)
box(ax, 7.8, 2.55, 1.85, 1.6, "Backtest engine\nEqual-Weight ·\nMean-Variance\n(SciPy) · metrics", AMBER, fs=7.4)

layer(ax, 0.2, 0.2, 9.6, 1.75, "Data layer", GREEN)
box(ax, 0.4, 0.4, 2.9, 1.0, "Yahoo Finance chart API\n(curl · retry / backoff)", GREEN, fs=7.6)
box(ax, 3.5, 0.4, 2.6, 1.0, "15-min CSV cache\nbackend/cache/*.csv", GREEN, fs=7.6)
box(ax, 6.3, 0.4, 3.35, 1.0, "Calibrated synthetic fallback\n(case studies; flagged in output)", GREEN, fs=7.4)

for gap_top, gap_bot, col in [(8.45, 8.05, TEAL), (5.55, 5.15, NAVY), (2.3, 1.95, AMBER)]:
    for x in (1.5, 3.4, 5.3, 7.2, 9.0):
        arrow(ax, (x, gap_top), (x, gap_bot), col, style="<|-|>", lw=1.0, ms=6)
fig.savefig(FIG / "fig_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============ Figure 2: pattern comparison pipeline ============
fig, ax = plt.subplots(figsize=(7.4, 4.9))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis("off")
box(ax, 0.2, 4.9, 1.9, 1.3, "Selected symbol\n+ date range\n(≥ 20 trading days)", TEAL, fs=8)
box(ax, 2.5, 4.9, 2.1, 1.3, "Channels per day\ncum. return · 5-day vol ·\nrunning drawdown", NAVY, fs=7.8)
box(ax, 4.95, 4.9, 2.0, 1.3, "Resample to\n40 points × 3 channels\n(length-invariant)", NAVY, fs=7.8)
box(ax, 7.3, 4.9, 2.5, 1.3, "Transformer encoder\n→ 16-d embedding", PURPLE, fs=8.2, bold=True)
for a, b in [(2.1, 2.5), (4.6, 4.95), (6.95, 7.3)]:
    arrow(ax, (a, 5.55), (b, 5.55))

layer(ax, 0.2, 1.5, 6.1, 2.95, "Reference corpus (built once, 965 windows)", AMBER)
box(ax, 0.35, 2.7, 1.65, 1.2, "3 case studies\n+ 4 calm periods\n21 window lengths\n(10–600 days)", AMBER, fs=7.4)
box(ax, 2.3, 2.7, 1.65, 1.2, "Label by crisis\noverlap / envelope\nrule; drop ambiguous;\nstratified caps", AMBER, fs=7.4)
box(ax, 4.25, 2.7, 1.9, 1.2, "Transformer\nautoencoder trained;\nHDBSCAN grid\nsearch (3 clusters)", AMBER, fs=7.4)
arrow(ax, (2.0, 3.3), (2.3, 3.3), AMBER); arrow(ax, (3.95, 3.3), (4.25, 3.3), AMBER)
ax.text(3.25, 1.78, "labelled window embeddings (dot-com · GFC · COVID · normal)", ha="center", fontsize=7.4, style="italic",
        color=GREY)

box(ax, 6.7, 2.55, 3.1, 1.5, "Scoring (per label)\n8-NN mean distance ÷ in-class\nspacing → Gaussian kernel\n× local purity (15-NN)", RED, fs=7.6)
arrow(ax, (8.55, 4.9), (8.55, 4.07), PURPLE)
arrow(ax, (6.3, 3.3), (6.7, 3.3), AMBER)
box(ax, 6.7, 0.25, 3.1, 1.6, "Report: matched event or\n“normal”, confidence,\noverlay vs reference window,\nHMM mix, next-21-day baseline", GREEN, fs=7.6)
arrow(ax, (8.25, 2.55), (8.25, 1.87), RED)
ax.text(0.25, 0.75, "Two modes: auto-detect (best of 3 events / normal) and targeted (score the period\nagainst one chosen event).",
        fontsize=7.6, color=GREY, style="italic", va="center")
fig.savefig(FIG / "fig_pipeline_matcher.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============ Figure 3: Transformer autoencoder ============
fig, ax = plt.subplots(figsize=(8.0, 3.0))
ax.set_xlim(0, 13.4); ax.set_ylim(0.15, 4.3); ax.axis("off")
box(ax, 0.1, 1.9, 1.35, 1.4, "Input\n40 × 3\n(standardised)", TEAL, fs=7.4)
box(ax, 1.75, 1.9, 1.55, 1.4, "Linear 3→32\n+ learned\npositional emb.", NAVY, fs=7.2)
ax.add_patch(Rectangle((3.6, 1.15), 2.9, 2.9, fc=LIGHT[PURPLE], ec=PURPLE, lw=1.3))
ax.text(5.05, 3.92, "Encoder (×2 layers)", ha="center", va="top", fontsize=8, color=PURPLE, fontweight="bold")
box(ax, 3.75, 2.45, 2.6, 0.9, "Multi-head self-attention\n(4 heads, d_model = 32)", PURPLE, fs=7)
box(ax, 3.75, 1.3, 2.6, 0.9, "Feed-forward 32→64→32\n(GELU) + residual / LayerNorm", PURPLE, fs=6.8)
box(ax, 6.8, 1.9, 1.5, 1.4, "Mean-pool over\n40 steps, then\nLinear 32→16", NAVY, fs=7)
box(ax, 8.6, 1.9, 1.3, 1.4, "Embedding\n$z\\in\\mathbb{R}^{16}$", RED, fs=8, bold=True)
ax.add_patch(Rectangle((10.2, 1.15), 3.1, 2.9, fc="none", ec=GREY, lw=1.0, ls=(0, (4, 3))))
ax.text(11.75, 3.92, "Decoder (training only)", ha="center", va="top", fontsize=7.8, color=GREY, fontweight="bold")
box(ax, 10.35, 2.35, 2.8, 1.0, "Linear 16→32, repeat ×40\n+ pos. emb.; encoder ×2", GREY, fs=6.8)
box(ax, 10.35, 1.3, 2.8, 0.85, "Linear 32→3 → 40 × 3\nreconstruction (MSE loss)", GREY, fs=6.8)
for a_, b_ in [(1.45, 1.75), (3.3, 3.6), (6.5, 6.8), (8.3, 8.6), (9.9, 10.2)]:
    arrow(ax, (a_, 2.6), (b_, 2.6))
ax.text(6.7, 0.5, "Trained self-supervised: Adam (lr 2·10⁻³), 120 full-batch epochs, seed 7. Only the encoder is kept at inference.",
        ha="center", fontsize=7.4, color=GREY, style="italic")
fig.savefig(FIG / "fig_transformer.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============ Figure 4: backtest protocol + allocation policy ============
fig, ax = plt.subplots(figsize=(7.4, 4.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
# timeline
ax.add_patch(Rectangle((0.4, 4.9), 4.2, 0.55, fc=LIGHT[TEAL], ec=TEAL, lw=1.2))
ax.add_patch(Rectangle((4.6, 4.9), 5.0, 0.55, fc=LIGHT[RED], ec=RED, lw=1.2))
ax.text(2.5, 5.175, "TRAIN: pre-crisis data only", ha="center", va="center", fontsize=8.2, color=TEAL, fontweight="bold")
ax.text(7.1, 5.175, "TEST: crisis + recovery (walk-forward, no retraining)", ha="center", va="center", fontsize=8.2, color=RED,
        fontweight="bold")
ax.plot([4.6, 4.6], [4.7, 5.75], color="#222", lw=1.3)
ax.text(4.6, 5.9, "crisis_start", ha="center", fontsize=8)
ax.text(0.4, 4.6, "window start", fontsize=7.4, color=GREY); ax.text(9.6, 4.6, "window end", fontsize=7.4, color=GREY, ha="right")
for i in range(11):
    x = 0.55 + i * 0.9
    ax.plot([x, x], [4.78, 4.88], color="#444", lw=0.8)
ax.text(5.0, 4.28, "rebalance every 21 trading days · 5 bps cost on turnover · identical dates for all strategies",
        ha="center", fontsize=7.6, style="italic", color=GREY)

box(ax, 0.3, 2.35, 2.2, 1.45, "State at rebalance t\n[P(bull),P(bear),P(crisis),\n21-d momentum ×N,\n21-d volatility ×N]", NAVY, fs=7.4)
box(ax, 2.85, 2.35, 1.7, 1.45, "MLP policy\nlinear → tanh(16)\n→ linear → softmax", PURPLE, fs=7.5)
box(ax, 4.9, 2.35, 1.5, 1.45, "Weights w\n(long-only,\nΣw = 1)", GREEN, fs=7.7)
arrow(ax, (2.5, 3.07), (2.85, 3.07)); arrow(ax, (4.55, 3.07), (4.9, 3.07))
box(ax, 6.75, 2.35, 3.1, 1.45, "Evolution Strategies (OpenAI-ES)\n32 antithetic perturbations × 40 gens\nreward = Sharpe − 2·MaxDD\n− 0.5·mean turnover", AMBER, fs=7)
arrow(ax, (6.8, 3.07), (6.4, 3.07), AMBER, style="-|>", ls="--")
ax.text(8.3, 3.95, "updates policy parameters (train segment only)", ha="center", fontsize=7, color=AMBER, style="italic")
box(ax, 0.3, 0.45, 2.8, 1.2, "Baseline 1:\nEqual-Weight (1/N)", GREY, fs=7.8)
box(ax, 3.4, 0.45, 3.4, 1.2, "Baseline 2: rolling Markowitz\nmax-Sharpe (63-day window,\nSLSQP, long-only)", GREY, fs=7.4)
box(ax, 7.0, 0.45, 2.85, 1.2, "Metrics: return · volatility ·\nSharpe · MaxDD · Calmar ·\nturnover", GREEN, fs=7.4)
fig.savefig(FIG / "fig_backtest_protocol.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("diagrams saved")
