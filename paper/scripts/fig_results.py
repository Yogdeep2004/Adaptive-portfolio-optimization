"""Result figures from the saved experiment outputs (run after exp_*.py).
Run:  backend/.venv/bin/python paper/scripts/fig_results.py
"""
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
FIG, RES = ROOT / "paper" / "figures", ROOT / "paper" / "results"
plt.rcParams.update({"font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})

C_EW, C_MV, C_AD = "#6b7280", "#d98e04", "#1d3557"
EVC = {"dotcom": "#d98e04", "gfc": "#b23a48", "covid": "#6c4ba6", "normal": "#2f7d4f"}
REG = {"bull": "#2f7d4f", "bear": "#d9a404", "crisis": "#b23a48"}
CASE_TITLE = {"dotcom_deployed": "Dot-com (as deployed)", "dotcom_corrected": "Dot-com (corrected)", "gfc": "2008 GFC",
              "covid": "COVID-19"}
STRAT = [("equal_weight", "Equal-Weight", C_EW), ("mean_variance", "Mean-Variance", C_MV), ("adaptive", "Adaptive (ES)", C_AD)]

bt = json.load(open(RES / "backtest.json"))
series = pickle.load(open(RES / "backtest_series.pkl", "rb"))
abl = json.load(open(RES / "ablation.json"))

# ---------------- Fig: HMM regimes on three events ----------------
from app.data import market_data as md  # noqa: E402
from app.ml.regime import detect_regimes  # noqa: E402

events = [("(a) Dot-com bust (SPY)", "1999-06-01", "2003-12-31", "2000-03-10", "2002-10-09"),
          ("(b) 2008 GFC (SPY)", "2006-01-01", "2010-12-31", "2007-10-09", "2009-03-09"),
          ("(c) COVID-19 (SPY)", "2019-01-01", "2021-12-31", "2020-02-19", "2020-04-07")]
fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.0))
for ax, (title, s, e, c0, c1) in zip(axes, events):
    px = md.get_history("SPY", start=s, end=e)["adjclose"]
    r = detect_regimes(px)
    dates, st = r.dates, r.state_key
    seg_start = 0
    for i in range(1, len(st) + 1):
        if i == len(st) or st[i] != st[seg_start]:
            ax.axvspan(dates[seg_start], dates[i - 1], color=REG[st[seg_start]], alpha=0.22, lw=0)
            seg_start = i
    ax.plot(px.index, px.values, color="#111", lw=0.9)
    ax.axvline(pd.Timestamp(c0), color="#222", ls="--", lw=0.9)
    ax.axvline(pd.Timestamp(c1), color="#222", ls="--", lw=0.9)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Adj. close (USD)")
    mix = pd.Series(st).value_counts(normalize=True)
    ax.text(0.99, 0.95, f"Crisis days: {mix.get('crisis', 0) * 100:.0f}%", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=REG["crisis"], fontweight="bold")
handles = [plt.Rectangle((0, 0), 1, 1, color=REG[k], alpha=0.35) for k in ("bull", "bear", "crisis")]
fig.legend(handles + [plt.Line2D([0], [0], color="#222", ls="--")], ["Bull / calm", "Bear / declining", "Crisis / high-vol",
           "Known crisis start / end"], loc="lower center", ncol=4, frameon=False, fontsize=8.5)
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(FIG / "fig_regimes.png", dpi=300, facecolor="white")
plt.close(fig)

# ---------------- Fig: backtest curves ----------------
fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
for ax, key in zip(axes.ravel(), ["dotcom_deployed", "dotcom_corrected", "gfc", "covid"]):
    s = series[key]
    dates = pd.to_datetime(s["dates"])
    split = s["split_idx"]
    for name, lab, col in STRAT:
        v = np.array(s["values"][name])
        ax.plot(dates, v / v[split], color=col, lw=1.1, label=lab) if False else ax.plot(dates, v, color=col, lw=1.1, label=lab)
    ax.axvspan(pd.Timestamp(s["crisis"][0]), pd.Timestamp(s["crisis"][1]), color="#b23a48", alpha=0.08, lw=0)
    ax.axvline(dates[split], color="#222", ls="--", lw=0.9)
    ax.set_title(CASE_TITLE[key], fontsize=9.5, fontweight="bold", loc="left")
    ax.set_ylabel("Portfolio value (start = 1)")
    ax.tick_params(axis="x", labelsize=7.5, rotation=20)
axes[0, 0].legend(fontsize=7.5, frameon=False, loc="lower left")
fig.text(0.5, 0.005, "Dashed line = train/test split · shaded = known crisis window", ha="center", fontsize=8, style="italic")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(FIG / "fig_backtest_curves.png", dpi=300, facecolor="white")
plt.close(fig)

# ---------------- Fig: metric bars (test segment) ----------------
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
keys = ["dotcom_deployed", "dotcom_corrected", "gfc", "covid"]
for ax, (metric, ttl, scale) in zip(axes, [("sharpe_ratio", "Sharpe ratio", 1), ("max_drawdown", "Max drawdown (%)", 100),
                                           ("annualized_volatility", "Annualised volatility (%)", 100)]):
    w = 0.26
    for j, (name, lab, col) in enumerate(STRAT):
        vals = [bt["cases"][k]["metrics_test"][name][metric] * scale for k in keys]
        ax.bar(np.arange(len(keys)) + (j - 1) * w, vals, w, color=col, label=lab)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(["Dot-com\n(deployed)", "Dot-com\n(corrected)", "GFC", "COVID"], fontsize=7)
    ax.set_title(ttl, fontsize=9, fontweight="bold")
    ax.axhline(0, color="#222", lw=0.6)
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig_metrics_bars.png", dpi=300, facecolor="white")
plt.close(fig)

# ---------------- Fig: adaptive allocations ----------------
fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.2))
ASSET_COL = {"SPY": "#1d3557", "QQQ": "#d98e04", "GLD": "#d9a404", "TLT": "#6c4ba6", "GC=F": "#d9a404", "VUSTX": "#6c4ba6"}
for ax, key in zip(axes, ["dotcom_corrected", "gfc", "covid"]):
    s = series[key]
    wlog = s["weights"]
    ds = sorted(wlog)
    assets = s["assets"]
    M = np.array([[wlog[d][a] for a in assets] for d in ds]).T
    ax.stackplot(pd.to_datetime(ds), M, labels=assets, colors=[ASSET_COL.get(a, "#888") for a in assets], alpha=0.85, step="post")
    ax.axvline(pd.to_datetime(s["dates"][s["split_idx"]]), color="#fff", ls="--", lw=1.2)
    ax.set_ylim(0, 1)
    ax.set_title(CASE_TITLE[key] + ": adaptive policy weights (dashed = train/test split)", loc="left", fontsize=9, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5, frameon=False)
    ax.set_ylabel("Weight")
fig.tight_layout()
fig.savefig(FIG / "fig_allocations.png", dpi=300, facecolor="white")
plt.close(fig)

# ---------------- Fig: seeds + ablation ----------------
df = pd.DataFrame([dict(case=r["case"], cond=r["cond"], seed=r["seed"], sharpe=r["test"]["adaptive"]["sharpe_ratio"],
                        vol=r["test"]["adaptive"]["annualized_volatility"], mdd=r["test"]["adaptive"]["max_drawdown"],
                        ew=r["test"]["equal_weight"]["sharpe_ratio"], mv=r["test"]["mean_variance"]["sharpe_ratio"])
                   for r in abl])
df.to_csv(RES / "ablation_flat.csv", index=False)
fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.0), sharey=False)
conds = [("full", "Full"), ("no_regime", "No regime\nfeatures"), ("untrained", "Untrained\n(random)")]
for ax, key in zip(axes, ["dotcom_deployed", "dotcom_corrected", "gfc", "covid"]):
    sub = df[df.case == key]
    for i, (c, lab) in enumerate(conds):
        v = sub[sub.cond == c].sharpe.values
        ax.boxplot(v, positions=[i], widths=0.5, patch_artist=True, boxprops=dict(facecolor="#c9d3e6", color=C_AD),
                   medianprops=dict(color=C_AD), whiskerprops=dict(color=C_AD), capprops=dict(color=C_AD), showfliers=False)
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=9, color=C_AD, zorder=3)
    ax.axhline(sub.ew.iloc[0], color=C_EW, ls="--", lw=1.0)
    ax.axhline(sub.mv.iloc[0], color=C_MV, ls="--", lw=1.0)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in conds], fontsize=6.8)
    ax.set_title(CASE_TITLE[key], fontsize=8.5, fontweight="bold")
axes[0].set_ylabel("Out-of-sample Sharpe")
fig.legend([plt.Line2D([0], [0], color=C_EW, ls="--"), plt.Line2D([0], [0], color=C_MV, ls="--")],
           ["Equal-Weight", "Mean-Variance"], loc="lower center", ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(FIG / "fig_seed_ablation.png", dpi=300, facecolor="white")
plt.close(fig)
print("allocator/regime figures done")

# ---------------- Matcher figures (if the experiment has finished) ----------------
mpath = RES / "matcher.json"
if mpath.exists():
    m = json.load(open(mpath))
    z = np.load(RES / "embeddings.npz", allow_pickle=True)
    lab = z["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, key, ttl in [(axes[0], "pca", f"(a) PCA (var. explained {z['explained'][0] * 100:.0f}% + {z['explained'][1] * 100:.0f}%)"),
                         (axes[1], "tsne", "(b) t-SNE")]:
        for l in ["normal", "dotcom", "gfc", "covid"]:
            idx = lab == l
            ax.scatter(z[key][idx, 0], z[key][idx, 1], s=7 if l == "normal" else 11, color=EVC[l], alpha=0.35 if l == "normal" else 0.8,
                       label={"normal": "Normal", "dotcom": "Dot-com", "gfc": "2008 GFC", "covid": "COVID-19"}[l], lw=0)
        ax.set_title(ttl, fontsize=9, fontweight="bold", loc="left"); ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
    axes[0].legend(fontsize=7.5, frameon=False, markerscale=1.6)
    fig.tight_layout()
    fig.savefig(FIG / "fig_embedding.png", dpi=300, facecolor="white")
    plt.close(fig)

    # overlays
    qs = list(m["queries"].items())
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6))
    for ax, (name, q) in zip(axes.ravel(), qs):
        x = np.linspace(0, 100, len(q["query_path"]))
        col = EVC.get(q["target"] or "normal", EVC["normal"])
        ax.plot(x, np.array(q["query_path"]) * 100, color=C_AD, lw=1.4, label="Selected period")
        ax.plot(x, np.array(q["ref_path"]) * 100, color=col, lw=1.4, ls="--", label="Nearest reference window")
        t = q["targeted"]
        ax.set_title(f"{name}\nconfidence {t['confidence'] * 100:.0f}%" + (f" vs {q['target']}" if q["target"] else " (auto: normal)")
                     if q["target"] else f"{name}\nauto: {q['auto']['event'] or 'normal'}, conf. {q['auto']['confidence'] * 100:.0f}%",
                     fontsize=8, fontweight="bold", loc="left")
        ax.set_xlabel("% through period", fontsize=7.5); ax.set_ylabel("Cum. return (%)", fontsize=7.5)
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(fontsize=6.8, frameon=False, loc="upper right")
    for ax in axes.ravel()[len(qs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "fig_overlays.png", dpi=300, facecolor="white")
    plt.close(fig)

    # CV ablation bars
    cv = m["cv"]
    names = list(cv)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    x = np.arange(len(names)); w = 0.38
    axes[0].bar(x - w / 2, [cv[n]["balanced_acc"] for n in names], w, yerr=[cv[n]["balanced_acc_sd"] for n in names],
                color=C_AD, label="Balanced accuracy", capsize=2)
    axes[0].bar(x + w / 2, [cv[n]["macro_f1"] for n in names], w, yerr=[cv[n]["macro_f1_sd"] for n in names], color="#2a7f8e",
                label="Macro-F1", capsize=2)
    axes[0].axhline(0.25, color="#999", ls=":", lw=1); axes[0].text(len(names) - 0.5, 0.26, "chance (4 classes)", ha="right", fontsize=6.5, color="#777")
    axes[0].set_title("(a) 4-class event identification", fontsize=9, fontweight="bold", loc="left")
    axes[1].bar(x - w / 2, [cv[n]["crisis_recall"] for n in names], w, yerr=[cv[n]["crisis_recall_sd"] for n in names], color=EVC["gfc"],
                label="Crisis recall", capsize=2)
    axes[1].bar(x + w / 2, [cv[n]["crisis_fpr"] for n in names], w, yerr=[cv[n]["crisis_fpr_sd"] for n in names], color=EVC["normal"],
                label="False-alarm rate (normal → crisis)", capsize=2)
    axes[1].set_title("(b) Crisis vs normal", fontsize=9, fontweight="bold", loc="left")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=6.3)
        ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cv_ablation.png", dpi=300, facecolor="white")
    plt.close(fig)

    # confusion matrix
    cm = np.array(m["confusion"]["cv_transformer"])
    labs = m["confusion"]["labels"]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    rown = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n({rown[i, j] * 100:.0f}%)", ha="center", va="center", fontsize=7,
                    color="white" if rown[i, j] > 0.55 else "#111")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(["Dot-com", "GFC", "COVID", "Normal"], fontsize=7); ax.set_yticklabels(["Dot-com", "GFC", "COVID", "Normal"], fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8); ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_confusion.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("matcher figures done")
else:
    print("matcher.json not found yet, matcher figures skipped")
