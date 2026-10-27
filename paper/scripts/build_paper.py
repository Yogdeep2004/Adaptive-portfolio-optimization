"""Build the manuscript (DOCX) from experiment outputs. Run:
   backend/.venv/bin/python paper/scripts/build_paper.py     (needs python-docx + matplotlib on the path)
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from docx_helpers import Paper  # noqa: E402

FIG, RES = ROOT / "paper" / "figures", ROOT / "paper" / "results"
OUT = ROOT / "paper" / "Adaptive_Portfolio_Optimizer_Manuscript.docx"

BT = json.load(open(RES / "backtest.json"))
M = json.load(open(RES / "matcher.json"))
ABL = json.load(open(RES / "ablation.json"))
SER = pickle.load(open(RES / "backtest_series.pkl", "rb"))

# ------------------------------------------------------------------ references
REFS = {
    "markowitz52": "Markowitz H. Portfolio selection. *J Finance*. 1952;7(1):77–91.",
    "demiguel09": "DeMiguel V, Garlappi L, Uppal R. Optimal versus naive diversification: how inefficient is the 1/N portfolio strategy? *Rev Financ Stud*. 2009;22(5):1915–1953.",
    "sharpe94": "Sharpe WF. The Sharpe ratio. *J Portf Manag*. 1994;21(1):49–58.",
    "hamilton89": "Hamilton JD. A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*. 1989;57(2):357–384.",
    "rabiner89": "Rabiner LR. A tutorial on hidden Markov models and selected applications in speech recognition. *Proc IEEE*. 1989;77(2):257–286.",
    "angbekaert02": "Ang A, Bekaert G. International asset allocation with regime shifts. *Rev Financ Stud*. 2002;15(4):1137–1187.",
    "cont01": "Cont R. Empirical properties of asset returns: stylized facts and statistical issues. *Quant Finance*. 2001;1(2):223–236.",
    "moody01": "Moody J, Saffell M. Learning to trade via direct reinforcement. *IEEE Trans Neural Netw*. 2001;12(4):875–889.",
    "jiang17": "Jiang Z, Xu D, Liang J. A deep reinforcement learning framework for the financial portfolio management problem. *arXiv*:1706.10059. 2017.",
    "lillicrap16": "Lillicrap TP, Hunt JJ, Pritzel A, et al. Continuous control with deep reinforcement learning. In: *Proc ICLR*. 2016 (*arXiv*:1509.02971).",
    "schulman17": "Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O. Proximal policy optimization algorithms. *arXiv*:1707.06347. 2017.",
    "zhang25": "Zhang B. Graph attention-based heterogeneous multi-agent deep reinforcement learning for adaptive portfolio optimization. *Sci Rep*. 2025;15.",
    "drlapo25": "Deep reinforcement learning for adaptive portfolio optimization in dynamic financial environments. *Appl Artif Intell Comput Syst*. 2025.",
    "rlapm25": "Reinforcement learning for single-stock portfolio optimization in equity markets. *Int J Comput Intell Syst* (Springer). 2025.",
    "regimerl25": "Adaptive and regime-aware RL for portfolio optimization. *arXiv*:2509.14385. 2025.",
    "vaswani17": "Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. In: *Adv Neural Inf Process Syst*. 2017;30:5998–6008.",
    "zerveas21": "Zerveas G, Jayaraman S, Patel D, Bhamidipaty A, Eickhoff C. A transformer-based framework for multivariate time series representation learning. In: *Proc KDD*. 2021:2114–2124.",
    "campello13": "Campello RJGB, Moulavi D, Sander J. Density-based clustering based on hierarchical density estimates. In: *Adv Knowl Discov Data Min (PAKDD)*, LNCS 7819. 2013:160–172.",
    "mcinnes17": "McInnes L, Healy J, Astels S. hdbscan: hierarchical density based clustering. *J Open Source Softw*. 2017;2(11):205.",
    "salimans17": "Salimans T, Ho J, Chen X, Sidor S, Sutskever I. Evolution strategies as a scalable alternative to reinforcement learning. *arXiv*:1703.03864. 2017.",
    "kingma15": "Kingma DP, Ba J. Adam: a method for stochastic optimization. In: *Proc ICLR*. 2015 (*arXiv*:1412.6980).",
    "paszke19": "Paszke A, Gross S, Massa F, et al. PyTorch: an imperative style, high-performance deep learning library. In: *Adv Neural Inf Process Syst*. 2019;32:8024–8035.",
    "pedregosa11": "Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. *J Mach Learn Res*. 2011;12:2825–2830.",
    "politis94": "Politis DN, Romano JP. The stationary bootstrap. *J Am Stat Assoc*. 1994;89(428):1303–1313.",
    "hendrycks16": "Hendrycks D, Gimpel K. Gaussian error linear units (GELUs). *arXiv*:1606.08415. 2016.",
    "ba16": "Ba JL, Kiros JR, Hinton GE. Layer normalization. *arXiv*:1607.06450. 2016.",
    "yahoo": "Yahoo Finance. Historical market data (chart API). https://finance.yahoo.com (accessed October 2026).",
}
P = Paper(REFS)

# ------------------------------------------------------------------ helpers
pct = lambda x, d=1: f"{x * 100:.{d}f}%"
f2 = lambda x: f"{x:.2f}"
CASES = ["dotcom_deployed", "dotcom_corrected", "gfc", "covid"]
CNAME = {"dotcom_deployed": "Dot-com (as deployed)", "dotcom_corrected": "Dot-com (corrected)", "gfc": "2008 GFC",
         "covid": "COVID-19"}
SNAME = {"equal_weight": "Equal-Weight", "mean_variance": "Mean-Variance", "adaptive": "Adaptive (ES)"}
cv, lo, q = M["cv"], M["loeo"], M["queries"]
TR = cv["Transformer (trained)"]
UN = cv["Transformer (untrained)"]


def small_img(path, max_w=1900):
    """Downsample screenshots (JPEG) to keep the document light."""
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    out = path.with_suffix(".jpg")
    im.save(out, quality=88)
    return out


abl = pd.DataFrame([dict(case=r["case"], cond=r["cond"], seed=r["seed"],
                         **{f"{s}_{m}": r["test"][s][k] for s in ("equal_weight", "mean_variance", "adaptive")
                            for m, k in (("sharpe", "sharpe_ratio"), ("vol", "annualized_volatility"), ("mdd", "max_drawdown"),
                                         ("ret", "total_return"))}) for r in ABL])

# =================================================================== TITLE PAGE
P.title("Comparing Current Market Conditions with Historical Crises Using Transformer Embeddings and "
        "Density-Based Clustering: A Regime-Aware Portfolio Analytics Platform and a Critical Evaluation")
P.centered("**Faculty guide:** Dr. Arti Tiwari", size=12, space_after=6)
P.centered("**Contributors**", size=12, space_after=2)
P.centered("Yogdeep Benchimath 23BAI0002", size=12, space_after=1)
P.centered("Anushka Mishra 23BAI0014", size=12, space_after=1)
P.centered("Mehul Arora 23BAI0044", size=12, space_after=8)
P.centered("Vellore Institute of Technology, Vellore, Tamil Nadu, India", size=10.5, italic=True, space_after=2)
P.centered("B.Tech in Computer Science and Engineering (Artificial Intelligence and Machine Learning), course project "
           "BCSE497J, Fall Semester 2026–27", size=10, space_after=12)

# =================================================================== ABSTRACT
P.h("Abstract", 1)
P.p(
    "Investors and risk managers routinely ask whether current market conditions resemble a past crisis, and whether a "
    "portfolio should adapt when they do. This paper presents an open-source, full-stack platform that addresses both "
    "questions on live market data, and reports an evaluation that is deliberately critical of its own components. The "
    "platform combines (i) an unsupervised three-state Gaussian hidden Markov model (HMM) for day-level regime labelling; "
    "(ii) a historical-pattern comparison engine in which any price window is resampled to a length-invariant 40-point, "
    "three-channel shape, embedded by a small Transformer encoder (16 dimensions) and scored against a labelled corpus of "
    f"{sum(M['corpus'].values())} windows drawn from the dot-com bust, the 2008 global financial crisis (GFC), the COVID-19 crash "
    "and calm periods, with density-based (HDBSCAN) structuring of the corpus; and (iii) an adaptive allocation policy trained "
    "by evolution strategies and benchmarked, with walk-forward backtests, against equal-weight and rolling mean-variance "
    "portfolios. Under leakage-controlled blocked cross-validation the matcher separates crisis-like from normal windows "
    f"(crisis recall {pct(TR['crisis_recall'])}, false-alarm rate {pct(TR['crisis_fpr'])}) and, in leave-one-crisis-out tests, "
    f"flags a never-seen crisis as crisis-like in {pct(lo['gfc']['Transformer (trained)']['crisis_like_rate'], 0)}–"
    f"{pct(lo['covid']['Transformer (trained)']['crisis_like_rate'], 0)} of its windows; however, it identifies *which* crisis "
    f"it is only for the dot-com bust (balanced accuracy {TR['balanced_acc']:.2f} against 0.25 chance), and a trained Transformer "
    "was no better than an untrained encoder or seven hand-crafted statistics. The adaptive allocator did not deliver a robust "
    "advantage: its mean out-of-sample Sharpe ratio over six seeds was below that of mean-variance in three of four settings, and "
    "the fourth, an apparent success, was traced to a constant 100% gold position caused by an invalid train/test split. We release "
    "the platform, the evaluation protocols and the negative results to support more rigorous regime-aware allocation research.",
    space_after=4)
P.p("**Keywords:** market regimes; hidden Markov model; Transformer encoder; HDBSCAN; historical analogue retrieval; portfolio "
    "optimization; evolution strategies; walk-forward backtesting; financial crises", justify=False)

# =================================================================== 1 INTRODUCTION
P.h("1. Introduction", 1)
P.h("1.1 Background and motivation", 2)
P.p(
    "Financial crises reorganise markets abruptly. Volatility clusters, correlations between assets rise exactly when "
    "diversification is most needed, and return distributions become heavy-tailed {cite:cont01}. The dot-com bust "
    "(2000–2002), the 2008 global financial crisis and the COVID-19 crash of February–March 2020 differ in cause, speed and "
    "duration (the Nasdaq fell roughly four-fifths over two and a half years, whereas the S&P 500 lost about a third in a "
    "single month in 2020), yet practitioners routinely reason about the present by analogy (“this looks like 2008”). "
    "Classical portfolio selection {cite:markowitz52} assumes stable return moments and is estimated on a trailing window, so it "
    "adapts slowly to such shifts; the naive 1/N rule is surprisingly hard to beat out of sample {cite:demiguel09}. "
    "Regime-switching models {cite:hamilton89,angbekaert02} and, more recently, deep reinforcement learning (RL) "
    "{cite:moody01,jiang17} have been proposed to make allocation adaptive."
)
P.p(
    "Two practical needs follow. First, a decision-maker wants an *explainable* comparison of the present (or any chosen "
    "period) against catalogued crises that states how closely, and on what evidence, the period resembles each of them, or "
    "that it resembles none and should be treated as an ordinary regime. Second, a researcher wants an honest test of whether "
    "making an allocation policy regime-aware actually improves risk-adjusted performance under transaction costs and a strict "
    "walk-forward protocol. This work builds and evaluates a single platform for both."
)
P.h("1.2 Problem statement", 2)
P.p("We study two coupled problems on daily multi-asset price data.")
P.bullets([
    "**P1. Historical-analogue comparison.** Given any symbol and date range (a window of arbitrary length, from about a month "
    "to several years), quantify whether its price path resembles one of three catalogued crises (dot-com, 2008 GFC, COVID-19) "
    "or a normal regime, report calibrated evidence (similarity confidence, a visual overlay with a real historical window, "
    "return/volatility/drawdown, regime composition) and, when no crisis matches, a normal-regime baseline.",
    "**P2. Regime-aware allocation.** Determine whether an adaptive policy that observes regime probabilities and "
    "asset-level momentum and volatility improves risk-adjusted outcomes over equal-weight and rolling mean-variance "
    "benchmarks when trained only on pre-crisis data and evaluated walk-forward through the crisis and recovery.",
])
P.h("1.3 Research gap", 2)
P.p(
    "The literature reviewed for this project (Section 2) reports adaptive RL allocators that outperform static baselines, but "
    "leaves gaps that this work targets. (a) Many RL studies use a single asset or simplified settings and ignore transaction "
    "costs and liquidity {cite:rlapm25}. (b) Static mean-variance and early RL models do not respond to regime shifts in real "
    "time. (c) Reported returns are often not accompanied by robust risk-adjusted benchmarking across varied market conditions, "
    "multi-seed variability or leakage controls. (d) High-performing deep models trade interpretability for performance "
    "{cite:regimerl25}. (e) We are not aware of a length-invariant, explainable analogue-matching tool for comparing arbitrary "
    "periods with named crises, coupled to an allocation benchmark."
)
P.h("1.4 Objectives and key contributions", 2)
P.p("The objectives are to (O1) detect market regimes without labelled crash dates; (O2) build an explainable historical-pattern "
    "comparison for arbitrary periods; (O3) implement an adaptive multi-asset allocator whose reward includes drawdown and "
    "turnover (transaction-cost) terms; (O4) benchmark against mean-variance and equal-weight with Sharpe ratio, drawdown and "
    "return; and (O5) assess robustness across bull, bear and crisis regimes. The contributions are:")
P.bullets([
    "**C1.** An open, runnable platform (FastAPI back end, React dashboard) with live data ingestion, a calibrated offline "
    "fallback, an HMM regime view, a pattern-comparison report and a backtesting workbench.",
    "**C2.** A length-invariant shape representation (cumulative return, short-window volatility and running drawdown resampled "
    "to 40 points) with a Transformer encoder, a stratified, class-balanced reference corpus and a nearest-neighbour similarity "
    "score with a purity-weighted confidence.",
    "**C3.** A leakage-controlled evaluation protocol for analogue matching (blocked, purged cross-validation and "
    "leave-one-crisis-out), together with simple-feature and untrained-encoder ablations.",
    "**C4.** A walk-forward allocation benchmark with multi-seed ablations and stationary-bootstrap intervals, including a "
    "corrected dot-com design.",
    "**C5.** Transparent negative findings and their diagnosis (an invalid train/test split producing a constant gold "
    "portfolio; no advantage of a trained over an untrained encoder; no robust advantage of the adaptive policy).",
])

# =================================================================== 2 RELATED WORK
P.h("2. Related Work / Literature Review", 1)
P.h("2.1 Portfolio selection and baselines", 2)
P.p(
    "Mean-variance optimisation {cite:markowitz52} selects weights that trade expected return against variance; estimated on a "
    "short window it is noisy, but remains the standard benchmark. DeMiguel et al. {cite:demiguel09} showed that the equal-weight "
    "1/N rule is rarely outperformed out of sample by optimised portfolios given realistic estimation error, which is why both "
    "are used as baselines here. Risk-adjusted performance is conventionally summarised by the Sharpe ratio {cite:sharpe94}, "
    "maximum drawdown and the Calmar ratio."
)
P.h("2.2 Regime detection", 2)
P.p(
    "Hamilton {cite:hamilton89} introduced Markov-switching models for nonstationary series, and Ang and Bekaert "
    "{cite:angbekaert02} showed that regimes matter for international asset allocation. Gaussian hidden Markov models (HMMs) "
    "{cite:rabiner89} fitted by expectation–maximisation provide an unsupervised way to label each day with a latent state; we "
    "use a three-state HMM on return, rolling volatility and drawdown features and relabel states by their empirical statistics."
)
P.h("2.3 Reinforcement learning for portfolio management", 2)
P.p(
    "Direct reinforcement learning for trading {cite:moody01} was extended by deep RL frameworks for portfolio management "
    "{cite:jiang17} using policy-gradient and actor–critic methods such as DDPG {cite:lillicrap16} and PPO {cite:schulman17}. "
    "The four recent studies surveyed in the project proposal are summarised in Table 1: Zhang {cite:zhang25} combines a graph "
    "attention network with three specialised agents; the DRL-APO framework {cite:drlapo25} uses a CNN–LSTM encoder with PPO; "
    "RL-APM {cite:rlapm25} applies DDPG to single-stock allocation with technical indicators; and a regime-aware study "
    "{cite:regimerl25} uses Transformer-PPO and LSTM models with regime detection."
)
P.table(
    ["Study", "Objective", "Methodology", "Strengths / limitations (as reported)"],
    [["Zhang (2025) {cite:zhang25}", "Adapt to time-varying asset correlations", "Graph-attention network + 3 specialised RL agents",
      "High returns; high complexity. Outperforms mean-variance and equal-weight"],
     ["DRL-APO (2025) {cite:drlapo25}", "Adaptive optimisation for dynamic markets", "CNN–LSTM encoder + PPO policy gradient",
      "Captures temporal patterns; price-only inputs. More adaptive than static models"],
     ["RL-APM (2025) {cite:rlapm25}", "Real-time single-stock allocation", "DDPG agent + SMA, RSI, MACD indicators",
      "Continuous control; ignores costs and liquidity. Improves on static baselines"],
     ["Regime-aware RL (2025) {cite:regimerl25}", "Regime-adaptive, explainable allocation", "Transformer-PPO and LSTM with regime detection",
      "Best returns; LSTM more interpretable. Regime context improves stability"]],
    "Related studies on adaptive RL portfolio optimisation (summarised from the project proposal’s literature review; results are "
    "as reported by the cited authors and are not reproduced here).",
    widths=[1.35, 1.35, 1.6, 2.0], font=8.5, align=["l", "l", "l", "l"])
P.h("2.4 Transformers and learned representations of time series", 2)
P.p(
    "The Transformer {cite:vaswani17} replaces recurrence with self-attention (Eq. 5 below), allowing every position in a "
    "sequence to attend to every other. Zerveas et al. {cite:zerveas21} showed that Transformer encoders can be pre-trained "
    "without labels on multivariate time series and used as representation learners. We follow this self-supervised "
    "autoencoding idea at a deliberately small scale: the encoder compresses a resampled window to a 16-dimensional vector."
)
P.h("2.5 Density-based clustering and analogue retrieval", 2)
P.p(
    "HDBSCAN {cite:campello13,mcinnes17} extends DBSCAN to variable-density clusters via a hierarchy of density levels and "
    "explicitly labels low-density points as noise, which is attractive when the question is “does this belong to any known "
    "group at all?”. We use the scikit-learn implementation {cite:pedregosa11} to examine the structure of the reference corpus "
    "and, as detailed in Section 3.3, score new windows directly against labelled corpus points."
)
P.h("2.6 Evolution strategies", 2)
P.p(
    "Salimans et al. {cite:salimans17} showed that natural-evolution-style strategies, which estimate a gradient from the rewards "
    "of randomly perturbed parameter vectors, are a scalable, gradient-free alternative to RL training. We use this method "
    "because it needs no automatic-differentiation framework for the allocator and trains in seconds on a laptop CPU."
)
P.h("2.7 Positioning", 2)
P.p(
    "Unlike the surveyed RL studies, which report returns of adaptive agents, this work (i) treats historical-analogue "
    "comparison as a first-class, explainable output, (ii) evaluates it with leakage controls and ablations, and (iii) "
    "reports multi-seed and bootstrap evidence for the allocator, including where it fails. We make no claim to reproduce or "
    "exceed the cited results; the implementation uses evolution strategies rather than PPO/DDPG, as discussed in Section 5."
)

# =================================================================== 3 METHODS
P.h("3. Materials and Methods / Methodology", 1)
P.h("3.1 Dataset and data collection", 2)
P.p(
    "All experiments use real daily adjusted-close prices obtained from the Yahoo Finance chart interface {cite:yahoo}. "
    "Requests are issued with explicit period timestamps (Yahoo’s “max” range silently coarsens daily bars to roughly monthly bars "
    "for long histories), retried with back-off, and cached per symbol on disk for 15 minutes. A calibrated synthetic fallback "
    "exists for the three case studies so that the demonstration never fails when the feed is unreachable; it was *not* used "
    "for any result reported here (every backtest in this paper ran on live data). The three crises are defined by calendar "
    "windows (Table 2): the dot-com bust (10 March 2000 – 9 October 2002), the 2008 GFC (9 October 2007 – 9 March 2009) and the "
    "COVID-19 crash (19 February – 7 April 2020)."
)
rows = []
for k in CASES:
    c = BT["cases"][k]
    rows.append([CNAME[k], ", ".join(c["assets"]), f"{c['date_range']['start']} – {c['date_range']['end']}", c["n_days"],
                 f"{c['crisis_window'][0]} – {c['crisis_window'][1]}", f"{c['n_train_days']} / {c['n_test_days']}", c["split"]])
P.table(["Setting", "Assets", "Data window", "Trading days", "Crisis window", "Train / test days", "Split date"], rows,
        "Backtest settings. The “as deployed” dot-com setting uses gold futures (GC=F), whose history starts in August 2000, after "
        "the crisis began; the corrected setting replaces it by the long-history Treasury fund VUSTX so that a genuine pre-crisis "
        "training segment exists.", widths=[1.1, 1.15, 1.2, 0.55, 1.2, 0.6, 0.7], font=7.5, align=["l", "l", "c", "c", "c", "c", "c"])
rows = []
for k in CASES:
    for a, s in BT["dataset"][k].items():
        rows.append([CNAME[k], a, f"{s['start']} – {s['end']}", s["n"], pct(s["ann_return"]), pct(s["ann_vol"]), pct(s["max_drawdown"])])
P.table(["Setting", "Asset", "Period", "N", "Ann. return", "Ann. volatility", "Max drawdown"], rows,
        "Descriptive statistics of the assets over each backtest data window (adjusted close; annualised with 252 trading days).",
        widths=[1.15, 0.65, 1.55, 0.45, 0.8, 0.9, 0.85], font=7.8, align=["l", "l", "c", "c", "c", "c", "c"])
P.p(
    f"For analogue matching, a reference corpus of **{sum(M['corpus'].values())} labelled windows** is built from the three case "
    f"studies (dot-com via QQQ, GFC and COVID-19 via SPY) plus four calm reference periods (SPY 2003–2006, 2012–2015, "
    f"2016–2019; QQQ 2012–2015): {M['corpus']['normal']} normal, {M['corpus']['dotcom']} dot-com, {M['corpus']['gfc']} GFC and "
    f"{M['corpus']['covid']} COVID-19 windows (Section 3.3.3)."
)
P.h("3.2 Preprocessing", 2)
P.p(
    "Prices are aligned on shared trading dates and daily simple returns are formed for the backtests. For regime detection, "
    "each day is described by the log return, a 20-day annualised rolling volatility and the drawdown from a 60-day rolling "
    "peak:"
)
P.equation(r"r_t=\ln\frac{P_t}{P_{t-1}},\qquad \sigma_t=\sqrt{252}\,\mathrm{std}\,(r_{t-19},\ldots,r_t),\qquad d_t=\frac{P_t}{\max_{s\in[t-59,\,t]}P_s}-1")
P.p(
    "The features are standardised before fitting. For analogue matching, a window of *n* trading days (any *n* ≥ 20) is "
    "converted to three channels measured from the window’s own start (cumulative return, 5-day rolling annualised "
    "volatility and running drawdown) and linearly interpolated onto 40 equally spaced points of the normalised time axis, "
    "so that a 33-day crash and a 600-day bear market become comparable *shapes*:"
)
P.equation(r"c_\tau=\frac{P_\tau}{P_1}-1,\quad v_\tau=\sqrt{252}\,\mathrm{std}(\rho_{\tau-4:\tau}),\quad g_\tau=\frac{P_\tau}{\max_{s\leq\tau}P_s}-1,\qquad \tilde{X}\in\mathbb{R}^{40\times 3}")
P.p("where ρ denotes daily returns inside the window. Channels are standardised with corpus-level means and standard deviations "
    "before entering the encoder.")

P.h("3.3 Proposed method", 2)
P.h("3.3.1 System architecture", 3)
P.p(
    "The platform has four layers (Fig. 1): a React 19 / Vite / Recharts dashboard; a FastAPI service layer exposing eight REST "
    "endpoints; an analytics layer containing the HMM, the pattern matcher, the adaptive allocator and the backtest engine; and a "
    "data layer with the Yahoo Finance client, an on-disk cache and the labelled synthetic fallback. Appendix A shows the "
    "dashboard. The fitted matcher is built once per server process (about 25 s) and cached; subsequent comparisons take "
    "about 0.1 s."
)
P.figure(FIG / "fig_architecture.png", "Layered architecture of the platform: presentation (React), service (FastAPI), "
         "analytics (HMM, pattern matcher, adaptive allocator, backtest engine) and data layers.", width=5.6)

P.h("3.3.2 Regime detection (HMM)", 3)
P.p(
    "A three-state diagonal-covariance Gaussian HMM {cite:rabiner89} is fitted by expectation–maximisation (300 iterations, "
    "fixed seed) on the standardised features:"
)
P.equation(r"z_t\mid z_{t-1}=i\sim\mathrm{Cat}(A_{i\cdot}),\qquad \mathbf{x}_t\mid z_t=j\sim\mathcal{N}\left(\mu_j,\mathrm{diag}(\sigma_j^{2})\right)")
P.p(
    "Because state indices are arbitrary, states are relabelled by their empirical statistics: the state with the highest mean "
    "volatility is *Crisis*; of the remaining two, the one with the higher mean return is *Bull* and the other *Bear*. The model "
    "is never given crash dates. Its posterior probabilities are used as part of the allocation policy’s state."
)
P.h("3.3.3 Historical-pattern comparison", 3)
P.p(
    "**Encoder.** A small Transformer encoder {cite:vaswani17} maps the 40 × 3 shape to a 16-dimensional embedding "
    "(Fig. 2): a linear projection to *d*_{model} = 32 with a learned positional embedding, two encoder layers with four attention "
    "heads, a 64-unit feed-forward block with GELU {cite:hendrycks16} and layer normalisation {cite:ba16}, mean pooling over the 40 steps "
    "and a linear map to 16 dimensions. The encoder is trained self-supervised as an autoencoder with a symmetric decoder (used "
    "only in training), minimising"
)
P.equation(r"\mathcal{L}=\frac{1}{N}\sum_{i=1}^{N}\left\|\tilde{X}_i-g_\phi\left(f_\theta(\tilde{X}_i)\right)\right\|_F^{2}")
P.p("with Adam {cite:kingma15} (learning rate 2·10^{−3}, 120 full-batch epochs) in PyTorch {cite:paszke19}. Self-attention is "
    "the scaled dot-product of Vaswani et al.:")
P.equation(r"\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V")
P.figure(FIG / "fig_transformer.png", "Transformer shape encoder. A 40 × 3 resampled window is encoded to a 16-dimensional embedding; "
         "the decoder exists only to provide the self-supervised reconstruction loss.", width=6.2)
P.p(
    "**Reference corpus.** Overlapping windows are cut from each source series at 21 lengths between 10 and 600 trading days "
    "with a stride of max(3, length/5) days, rounded down. A window receives an event label if the crisis fills at least 50% of the *window*, or "
    "if the window contains at least 80% of the *crisis* while being at most four times its length (so that the short COVID-19 "
    "crash is represented by windows longer than 68 days); windows with 5–50% overlap are dropped as ambiguous and the rest are "
    "labelled normal. Event classes are capped at 120 windows and the normal class at 2.5 times the event total, with capping "
    "stratified by window length so that sparse, distinctive long windows are not crowded out by many short ones."
)
P.p(
    "**Corpus structure.** HDBSCAN is run on the embeddings over a grid of nine (min_cluster_size, min_samples) settings; the "
    "setting is chosen to maximise a score that rewards each event having its own cluster in which it forms a ≥ 60% majority, "
    "high size-weighted purity and low noise, and penalises fragmentation. With the deployed corpus it yields "
    f"{M['matcher']['n_clusters']} clusters. The clustering is used to inspect the corpus; scoring does not depend on cluster "
    "centroids, because the centroid of an arbitrarily shaped density cluster can lie in empty space and about 16% of the "
    "corpus is HDBSCAN noise that no centroid can represent."
)
P.p(
    "**Scoring.** For a query embedding *z* and a label ℓ, let N_k^ℓ(z) be its *k* = 8 nearest corpus windows carrying label ℓ, "
    "*s*_ℓ the label’s in-class spacing (the 75th percentile, over its members, of the mean distance to their eight nearest "
    "same-label neighbours) and p_ℓ(z) an inverse-distance-weighted purity among the 15 nearest windows overall:"
)
P.equation(r"m_\ell(z)=\frac{1}{k}\sum_{j\in\mathcal{N}_k^{\ell}(z)}\left\|z-z_j\right\|_2,\qquad \pi_\ell(z)=\exp\left[-\frac{1}{2}\left(\frac{m_\ell(z)}{s_\ell}\right)^{2}\right]")
P.equation(r"p_\ell(z)=\frac{\sum_{j\in\mathcal{N}_{15}(z)}w_j\,\mathbb{1}[y_j=\ell]}{\sum_{j\in\mathcal{N}_{15}(z)}w_j},\quad w_j=\frac{1}{\left\|z-z_j\right\|_2+10^{-3}},\qquad C_\ell(z)=\pi_\ell(z)\,p_\ell(z)")
P.p(
    "In *targeted* mode the confidence C_ℓ is reported for one chosen event; in *auto* mode the label with the largest C_ℓ "
    "wins. A match requires m_ℓ ≤ 1.4 s_ℓ, C_ℓ ≥ 0.20 and, in auto mode, ℓ ≠ normal with p_ℓ > 0.6; otherwise the report says that "
    "no crisis resembles the period and gives the normal-class confidence and baseline instead. Every match is accompanied by "
    "the nearest same-label reference window (for the overlay), the period’s annualised return, volatility, maximum drawdown, "
    "the HMM regime mix, and the mean 21-trading-day forward return of the nearest same-label windows. Fig. 3 summarises the pipeline."
)
P.figure(FIG / "fig_pipeline_matcher.png", "Historical-pattern comparison pipeline: length-invariant shape, Transformer embedding, "
         "labelled reference corpus, nearest-neighbour similarity scoring and the explanatory report.", width=6.0)

P.h("3.3.4 Allocation strategies and the adaptive policy", 3)
P.p(
    "Three strategies are compared on identical rebalance dates (every 21 trading days, i.e. approximately monthly) and an "
    "identical proportional cost of 5 basis points on turnover. **Equal-Weight** holds 1/N. **Mean-Variance** maximises the "
    "Sharpe ratio of a long-only, fully invested portfolio using the previous 63 trading days (sequential least-squares "
    "programming; equal-weight if the optimiser fails). The **Adaptive** policy is a two-layer perceptron mapping the state "
    "*s*_k (the three HMM regime probabilities, 21-day momentum and 21-day annualised volatility of each asset) to long-only "
    "weights:"
)
P.equation(r"\mathbf{w}_k=\mathrm{softmax}\left(W_2\tanh\left(W_1\mathbf{s}_k+\mathbf{b}_1\right)+\mathbf{b}_2\right)")
P.p(
    "It is trained by OpenAI-style evolution strategies {cite:salimans17}: at each of 40 generations, 32 antithetic Gaussian "
    "perturbations (σ = 0.1) of the flattened parameters are evaluated by simulating the policy through the *training* "
    "segment, rewards are rank-normalised, and the parameters move along the reward-weighted perturbation direction "
    "(step 0.05). The reward penalises drawdown and turnover so that costs enter the objective:"
)
P.equation(r"R(\theta)=\mathrm{SR}-2\,\mathrm{MDD}-0.5\,\overline{\mathrm{TO}},\qquad \theta\leftarrow\theta+\frac{\alpha}{N\sigma}\sum_{i=1}^{N}\hat{F}_i\,\varepsilon_i")
P.figure(FIG / "fig_backtest_protocol.png", "Walk-forward protocol and allocation policy. The policy is trained only on the pre-crisis "
         "segment by evolution strategies and then evaluated without retraining through the crisis and recovery against the "
         "two baselines.", width=6.0)

P.h("3.4 Experimental setup", 2)
P.p(
    "All code runs on an Apple M3 laptop (8 cores, 8 GB RAM, macOS 26) with Python 3.12.7, PyTorch 2.5.1, scikit-learn 1.5.2, "
    "hmmlearn 0.3.3, SciPy 1.13.1, NumPy 1.26.4 and pandas 2.2; the dashboard uses React 19, Vite 8 and Recharts 3. "
    "No GPU is used. The experiments are:"
)
P.bullets([
    "**E1. Backtests.** Four settings (Table 2), single deterministic run (policy seed 42) for headline tables, with a "
    "stationary bootstrap {cite:politis94} (5,000 resamples, mean block length 10 days) for the Sharpe-ratio difference between the "
    "adaptive policy and each baseline on the test segment.",
    "**E2. Allocator ablation.** Six seeds × four settings × three variants: *full* (regime probabilities, momentum and "
    "volatility), *no-regime* (regime probabilities replaced by a constant, removing all regime information) and *untrained* "
    "(random-initialised policy, 0 generations).",
    "**E3. Matcher evaluation.** (a) Five-fold *blocked* cross-validation: within each label, windows are sorted by start date "
    "and split into five contiguous time blocks; every training window overlapping in calendar time with any test window is "
    "purged, and the encoder is re-trained on the training windows only. (b) *Leave-one-event-out*: train on the other two "
    "events plus normal (purged against the held-out event’s dates), and test whether the unseen crisis is predicted as any crisis "
    "class rather than normal. (c) Qualitative, in-sample queries (Table 5).",
    "**E4. Embedding ablation.** The same distance-weighted 8-NN classifier on: the trained Transformer (three encoder seeds), "
    "an untrained random-initialised Transformer (three seeds), the raw 120-dimensional shape, PCA-16 of the raw shape, and seven "
    "hand-crafted statistics (final and extreme cumulative return, mean and peak rolling volatility, minimum and mean drawdown).",
])
P.table(
    ["Component", "Setting"],
    [["HMM", "3 states, diagonal covariance, 300 EM iterations, seed 7; features: log return, 20-day volatility, 60-day drawdown"],
     ["Shape", "40 points × 3 channels; windows 10–600 days (21 lengths); stride max(3, length/5)"],
     ["Transformer", "d_model 32, 4 heads, 2 layers, FFN 64 (GELU), learned positional embedding, mean-pool, 16-d output; Adam, lr 2·10^{−3}, 120 epochs"],
     ["Corpus", "Event cap 120/class (length-stratified); normal cap 2.5× events; ≥50% window overlap or ≥80% crisis coverage (≤4× length)"],
     ["HDBSCAN", "9-point grid on (min_cluster_size, min_samples); score = 10·#events covered + 5·purity − 3·noise − 0.05·#clusters"],
     ["Scoring", "k = 8 same-label neighbours; Gaussian kernel on spacing s_ℓ; purity over 15 nearest; match: m ≤ 1.4 s, C ≥ 0.20"],
     ["Backtest", "Rebalance every 21 trading days; cost 5 bps on turnover; risk-free rate 2%; 252 trading days/year"],
     ["Mean-Variance", "63-day window, long-only max-Sharpe (SLSQP), fallback equal-weight"],
     ["ES policy", "Hidden 16 (tanh); 32 antithetic perturbations × 40 generations; σ 0.1; step 0.05; reward SR − 2·MDD − 0.5·turnover"]],
    "Key hyper-parameters.", widths=[1.2, 5.3], font=8.3, align=["l", "l"])

P.h("3.5 Evaluation metrics", 2)
P.p(
    "Performance is reported on the out-of-sample (test) segment, from the crisis start to the end of the window, and, where "
    "stated, the full period. With daily portfolio returns *r* and value path V:"
)
P.equation(r"\mathrm{SR}=\sqrt{252}\,\frac{\overline{r-r_f/252}}{\mathrm{std}(r)},\qquad \mathrm{MDD}=\min_t\left(\frac{V_t}{\max_{s\leq t}V_s}-1\right),\qquad \mathrm{Calmar}=\frac{\mathrm{AnnRet}}{|\mathrm{MDD}|}")
P.equation(r"\mathrm{TO}_k=\left\|\mathbf{w}_k-\mathbf{w}_{k-1}\right\|_1,\qquad V\leftarrow V\left(1-c\,\mathrm{TO}_k\right),\ c=5\times10^{-4},\qquad \Delta=\mathrm{SR}_{\mathrm{adaptive}}-\mathrm{SR}_{\mathrm{baseline}}")
P.p(
    "Total return, annualised return and annualised volatility are also reported, and *turnover* is the mean ℓ¹ weight change per "
    "rebalance. Because the event classes are far smaller than the normal class, matcher performance is reported as *balanced "
    "accuracy* (mean per-class recall; chance = 0.25 for four classes) and macro-F1, together with the *crisis recall* (share of "
    "event windows predicted as any event) and the *false-alarm rate* (share of normal windows predicted as an event); plain "
    "accuracy is shown only for completeness because always predicting “normal” already reaches 70%."
)

# =================================================================== 4 RESULTS
P.h("4. Results and Discussion", 1)
P.h("4.1 Experimental results", 2)
P.h("4.1.1 Regime detection", 3)
P.p(
    "The unsupervised HMM labels the sustained high-volatility stretches of all three crises as Crisis (Fig. 5): the red band "
    "covers the 2008–09 collapse, the October 2002 bottom of the dot-com bust and the March 2020 crash, without any crash dates "
    "being supplied. Crisis days are a minority of each window (19%, 14% and 8%), and Bear states dominate the slow phases "
    "before and after, so the model separates high-volatility turmoil from gradual decline. The labels are noisier in calm "
    "periods, where short Bull/Bear alternations occur; the HMM is therefore used as a descriptive overlay and as an input to "
    "the allocator, not as a trading signal."
)
P.figure(FIG / "fig_regimes.png", "Unsupervised three-state HMM regimes on SPY for (a) the dot-com bust, (b) the 2008 GFC and (c) the "
         "COVID-19 crash. Dashed lines mark the known crisis start and end, which the model never sees.", width=6.2)

P.h("4.1.2 Historical-pattern comparison", 3)
P.p(
    "**In-sample illustration.** Table 5 and Fig. 6 show the comparison report on five representative queries. The three crisis "
    f"windows are matched to their own events with confidences of {pct(q['COVID-19 crash window']['targeted']['confidence'], 0)} "
    f"(COVID-19), {pct(q['2008 GFC window']['targeted']['confidence'], 0)} (GFC) and "
    f"{pct(q['Dot-com bust window']['targeted']['confidence'], 0)} (dot-com), and two ordinary periods (2013–2015 and the most recent "
    "months) are labelled normal. These queries, however, overlap the periods from which the corpus was built, so the matches are "
    "partly circular; they illustrate the report, and the leakage-controlled tests below measure generalisation."
)
rows = []
for name, v in q.items():
    t = v["targeted"]
    rows.append([name, f"{v['symbol']} {v['start']} → {v['end']}", v["days"],
                 (f"{t['event']} ({pct(t['confidence'], 0)})" if v["target"] else f"{'normal' if not v['auto']['event'] else v['auto']['event']} ({pct(v['auto']['confidence'], 0)})"),
                 pct(v["stats"]["ann_return"]), pct(v["stats"]["max_dd"]),
                 f"{pct(v['stats']['regime_mix']['bull'], 0)}/{pct(v['stats']['regime_mix']['bear'], 0)}/{pct(v['stats']['regime_mix']['crisis'], 0)}",
                 pct(v["fwd21"], 1)])
P.table(["Query", "Symbol, range", "Days", "Result (confidence)", "Ann. return", "Max DD", "HMM bull/bear/crisis", "Next-21d baseline"], rows,
        "Comparison-report output for five example queries (in-sample: the periods overlap the corpus). Crisis rows are scored in "
        "targeted mode against the corresponding event; the other two use auto-detect mode. Next-21d baseline is the mean 21-trading-day "
        "forward return of the nearest same-label reference windows.",
        widths=[1.0, 1.3, 0.4, 0.95, 0.6, 0.55, 0.85, 0.65], font=7.5, align=["l", "l", "c", "c", "c", "c", "c", "c"])
P.figure(FIG / "fig_overlays.png", "Overlay of each query’s normalised cumulative-return path (solid) and the path of its nearest same-label "
         "reference window (dashed), as shown in the dashboard’s comparison report.", width=6.2)
P.p(
    "**Embedding space.** Fig. 7 projects the 16-dimensional embeddings. Normal windows form the dense bulk, whereas event windows "
    "occupy a visibly separate but overlapping region in which dot-com, GFC and COVID-19 windows are intermingled, consistent "
    "with the observation that crash-then-recovery shapes are intrinsically similar across crises."
)
P.figure(FIG / "fig_embedding.png", "Corpus embeddings coloured by label: (a) PCA and (b) t-SNE projections of the 965 16-dimensional "
         "vectors.", width=6.0)
P.p(
    "**Leakage-controlled evaluation.** Table 6 reports blocked, purged cross-validation. All variants reach about 75–76% accuracy, "
    "but this is almost entirely the normal class (recall 99%); balanced accuracy is only "
    f"{TR['balanced_acc']:.2f} (chance 0.25) for the trained Transformer. Dot-com windows are recovered in "
    f"{pct(TR['recall']['dotcom'], 0)} of cases, whereas GFC and COVID-19 windows are almost never identified "
    f"(recall {pct(TR['recall']['gfc'], 0)} and {pct(TR['recall']['covid'], 0)}); the confusion matrix (Fig. 9) shows that they are "
    "mostly assigned to dot-com or normal. Held-out *crisis versus normal* discrimination is better: "
    f"{pct(TR['crisis_recall'], 0)} of event windows are predicted as some crisis, with a false-alarm rate of {pct(TR['crisis_fpr'])}. "
    "Thus, across unseen time blocks the representation captures “crisis-like” but not “which crisis”."
)
rows = []
for n_, v in cv.items():
    rows.append([n_, f"{v['balanced_acc']:.3f} ± {v['balanced_acc_sd']:.3f}", f"{v['macro_f1']:.3f} ± {v['macro_f1_sd']:.3f}",
                 pct(v["acc"], 1), pct(v["crisis_recall"], 1), pct(v["crisis_fpr"], 1),
                 f"{pct(v['recall']['dotcom'], 0)} / {pct(v['recall']['gfc'], 0)} / {pct(v['recall']['covid'], 0)} / {pct(v['recall']['normal'], 0)}"])
P.table(["Features (8-NN)", "Balanced acc.", "Macro-F1", "Accuracy", "Crisis recall", "False alarm", "Recall dot-com / GFC / COVID / normal"], rows,
        "Blocked, purged 5-fold cross-validation of event identification (four classes). Mean ± s.d. over three encoder seeds for the two "
        "Transformer variants; the other variants are deterministic. Chance balanced accuracy is 0.25.",
        widths=[1.6, 0.85, 0.85, 0.6, 0.6, 0.6, 1.4], font=7.6, align=["l", "c", "c", "c", "c", "c", "c"], bold_rows=(0,))
P.figure(FIG / "fig_cv_ablation.png", "Embedding ablation under blocked cross-validation: (a) balanced accuracy and macro-F1 and (b) crisis "
         "recall and false-alarm rate for the five feature variants (error bars: s.d. over encoder seeds).", width=6.0)
P.figure(FIG / "fig_confusion.png", "Pooled confusion matrix of the trained-Transformer 8-NN classifier under blocked cross-validation "
         "(counts and row percentages).", width=3.4)
P.p(
    "**Unseen crises.** In leave-one-event-out tests (Table 7) the trained encoder predicted "
    f"{pct(lo['dotcom']['Transformer (trained)']['crisis_like_rate'], 0)} of dot-com windows, "
    f"{pct(lo['gfc']['Transformer (trained)']['crisis_like_rate'], 0)} of GFC windows and "
    f"{pct(lo['covid']['Transformer (trained)']['crisis_like_rate'], 0)} of COVID-19 windows as crisis-like although no window of "
    "that event was available for training. The GFC, a slow, grinding decline whose short sub-windows resemble ordinary "
    "periods, is hardest. The result supports using the tool to ask whether a period “looks like a crisis” in general."
)
rows = []
for ev, nm in (("dotcom", "Dot-com"), ("gfc", "2008 GFC"), ("covid", "COVID-19")):
    r = [nm, lo[ev]["n_test"], lo[ev]["n_train"]]
    for v in cv:
        r.append(pct(lo[ev][v]["crisis_like_rate"], 0))
    rows.append(r)
P.table(["Held-out event", "Test", "Train"] + [v.replace(" (", "\n(") for v in cv], rows,
        "Leave-one-event-out: share of the unseen crisis’ windows predicted as crisis-like (any event class), by feature variant.",
        widths=[0.95, 0.4, 0.45, 0.95, 0.95, 0.85, 0.85, 0.95], font=7.5, align=["l"] + ["c"] * 7)

P.h("4.1.3 Walk-forward allocation backtests", 3)
P.p(
    "Table 8 and Figs. 10–11 give the out-of-sample results. The picture is mixed and, importantly, depends on the experimental "
    "design. **Dot-com as deployed:** the adaptive portfolio reaches a Sharpe ratio of "
    f"{BT['cases']['dotcom_deployed']['metrics_test']['adaptive']['sharpe_ratio']:.2f} and a maximum drawdown of "
    f"{pct(BT['cases']['dotcom_deployed']['metrics_test']['adaptive']['max_drawdown'])}, beating both baselines; but its weights are "
    "100% gold (GC=F) at every one of 40 rebalances and its turnover is zero: the gold futures series starts in August 2000, "
    "after the crisis began, so the “pre-crisis” training segment is only 30 trading days and the policy collapses to a "
    "constant, buy-and-hold gold position (which happened to rise). **Dot-com corrected** (SPY, QQQ, VUSTX; 196 genuine "
    "pre-crisis training days): the adaptive policy loses badly (Sharpe "
    f"{BT['cases']['dotcom_corrected']['metrics_test']['adaptive']['sharpe_ratio']:.2f}, maximum drawdown "
    f"{pct(BT['cases']['dotcom_corrected']['metrics_test']['adaptive']['max_drawdown'])}), because a policy fitted to the bubble "
    "years concentrates in QQQ and flips between QQQ and bonds (Fig. 12). **2008 GFC:** mean-variance has the best Sharpe "
    f"({BT['cases']['gfc']['metrics_test']['mean_variance']['sharpe_ratio']:.2f}) and the adaptive policy the lowest volatility "
    f"({pct(BT['cases']['gfc']['metrics_test']['adaptive']['annualized_volatility'])}) but not the shallowest drawdown. "
    "**COVID-19:** mean-variance again leads on Sharpe, while the adaptive policy is close to equal-weight."
)
rows, bold = [], []
for k in CASES:
    c = BT["cases"][k]
    best = max(c["metrics_test"], key=lambda s: c["metrics_test"][s]["sharpe_ratio"])
    for s in ("equal_weight", "mean_variance", "adaptive"):
        m = c["metrics_test"][s]
        rows.append([CNAME[k] if s == "equal_weight" else "", SNAME[s], pct(m["total_return"]), pct(m["annualized_return"]),
                     pct(m["annualized_volatility"]), f"{m['sharpe_ratio']:.2f}", pct(m["max_drawdown"]), f"{m['calmar_ratio']:.2f}",
                     f"{c['turnover'][s]:.2f}"])
        if s == best:
            bold.append(len(rows) - 1)
P.table(["Setting", "Strategy", "Total return", "Ann. return", "Ann. vol.", "Sharpe", "Max DD", "Calmar", "Turnover"], rows,
        "Out-of-sample (test-segment) performance of the three strategies, single run (policy seed 42). The row with the highest "
        "Sharpe ratio in each setting is in bold. Turnover is the mean ℓ¹ weight change per monthly rebalance (equal-weight "
        "targets never change, so it reports 0; trades that merely restore 1/N after price drift are not counted).",
        widths=[1.15, 0.95, 0.65, 0.65, 0.6, 0.5, 0.6, 0.5, 0.55], font=7.8, align=["l", "l"] + ["c"] * 7, bold_rows=tuple(bold))
P.figure(FIG / "fig_backtest_curves.png", "Walk-forward portfolio value for the three strategies in the four settings. Dashed line: "
         "train/test split; shaded: known crisis window.", width=6.2)
P.figure(FIG / "fig_metrics_bars.png", "Out-of-sample Sharpe ratio, maximum drawdown and annualised volatility by setting and strategy.", width=6.2)
P.figure(FIG / "fig_allocations.png", "Adaptive-policy weights over time (seed 42). In the 2008 GFC the policy moves almost entirely to "
         "long Treasuries (TLT) in late 2008 (a flight-to-quality response) but in the corrected dot-com setting it alternates "
         "between QQQ and bonds.", width=6.2)

P.h("4.2 Tables, graphs and figures", 2)
P.p(
    "The figures referred to above are generated by scripts in the repository (paper/scripts) from the saved experiment outputs "
    "(paper/results), so every number in the tables is reproducible with the commands in Appendix B. Appendix A reproduces the "
    "dashboard’s three tabs, in which the same tables and charts (value curves, metrics table, allocation chart, HMM regimes and the "
    "comparison report) are interactive."
)

P.h("4.3 Criterion-wise analysis", 2)
crit = [("Total return", "total_return", max), ("Annualised volatility (lowest)", "annualized_volatility", min),
        ("Sharpe ratio", "sharpe_ratio", max), ("Max drawdown (shallowest)", "max_drawdown", max), ("Calmar ratio", "calmar_ratio", max)]
ab = {"equal_weight": "EW", "mean_variance": "MV", "adaptive": "AD"}
rows = []
for nm, key, fn in crit:
    r = [nm]
    for k in CASES:
        ms = BT["cases"][k]["metrics_test"]
        r.append(ab[fn(ms, key=lambda s: ms[s][key])])
    rows.append(r)
P.table(["Criterion"] + [CNAME[k] for k in CASES], rows,
        "Best strategy per criterion and setting on the test segment (seed-42 run). EW = Equal-Weight, MV = Mean-Variance, AD = Adaptive.",
        widths=[2.0, 1.1, 1.1, 1.1, 1.1], font=8.3, align=["l", "c", "c", "c", "c"])
P.p(
    "Mean-variance is best on return, Sharpe, Calmar ratio and maximum drawdown in three of four settings; the adaptive "
    "policy is best on volatility in three settings (the criterion on which it is explicitly rewarded through the drawdown and "
    "turnover terms), but, as Section 4.6 shows, this is not stable across seeds. Equal-weight is never best and has the deepest drawdown of the three strategies in three of four settings (all but the corrected "
    "dot-com), but it is the cheapest to operate (no optimisation, no change in target weights). "
    "On *interpretability*, the HMM and the comparison report are transparent by construction (stated evidence, an overlay of a real "
    "reference window), whereas the ES-trained policy is a small black box. On *runtime*, a backtest takes 8–17 s and a cached "
    "comparison about 0.1 s on the laptop (first-use matcher build ≈ 25 s)."
)

P.h("4.4 Comparison with existing methods", 2)
P.p(
    "A like-for-like numerical comparison with the studies in Table 1 is not possible: they use different datasets, assets, "
    "periods, costs and protocols, and their code and data were not available to us. We therefore compare against the "
    "standard benchmarks of the same protocol (equal-weight and mean-variance; Section 4.1.3) and position the platform "
    "qualitatively against the reviewed approaches (Table 10), using only the properties those studies report. The principal "
    "difference is not performance but scope and candour: the platform adds an explainable analogue-comparison output and an "
    "evaluation that includes seeds, ablations and leakage controls; it does *not* demonstrate that its allocator beats the baselines."
)
P.table(
    ["Capability", "Mean-Variance / 1/N", "Zhang [GAT multi-agent]", "DRL-APO", "RL-APM", "Regime-aware RL", "This work"],
    [["Multi-asset allocation", "Yes", "Yes", "Yes", "No (single stock)", "Yes", "Yes (3–4 assets)"],
     ["Explicit regime detection", "No", "Not reported", "No", "No", "Yes", "Yes (HMM)"],
     ["Transaction costs in objective", "No", "Not reported", "Not reported", "Ignored", "Not reported", "Yes (turnover + 5 bps)"],
     ["Historical-analogue comparison", "No", "No", "No", "No", "No", "Yes (Transformer + HDBSCAN)"],
     ["Interpretable output", "Yes", "Low", "Low", "Medium", "Medium (LSTM)", "Yes (report + overlay)"],
     ["Leakage-controlled / multi-seed evaluation", "n/a", "Not reported", "Not reported", "Not reported", "Not reported", "Yes"],
     ["Reported to beat static baselines", "n/a", "Yes (reported)", "Yes (reported)", "Yes (reported)", "Yes (reported)", "No (this paper)"]],
    "Qualitative capability comparison. Entries for other studies are limited to what Table 1 records; “Not reported” means the "
    "project literature review did not record the property, not that the study lacks it.",
    widths=[1.45, 0.85, 0.85, 0.7, 0.75, 0.85, 1.05], font=7.4, align=["l"] + ["c"] * 6)

P.h("4.5 Interpretation of results", 2)
P.bullets([
    "**Regimes are detectable; the label is not a forecast.** The HMM recovers the turbulent phases of all three crises without "
    "labels, but its day-level states are noisy and gave no consistent benefit when removed from the allocator’s state (Section 4.7).",
    "**Shape similarity is a crisis detector, not a crisis identifier.** Windows from crises look alike once reduced to normalised "
    "shapes (Fig. 7), so discriminating dot-com from GFC from COVID-19 across held-out blocks fails, while “crisis-like versus "
    "normal” works with few false alarms. The report’s confidence should therefore be read as proximity to a family of crash "
    "shapes, not as the probability that the period *is* that event.",
    "**The Transformer is not the source of the signal.** An untrained random encoder, the raw shape and seven summary "
    "statistics perform as well as the trained Transformer. This is expected for a 16-dimensional embedding trained on under a "
    "thousand strongly overlapping windows; the encoder acts as a fixed random projection that preserves distances.",
    "**Adaptive allocation did not beat simple baselines.** Mean-variance is the strongest or near-strongest strategy in three "
    "of four settings. Apparent wins are fragile: the dot-com “success” was a gold position produced by an invalid split, and the "
    "volatility advantage depends on the seed.",
    "**Experimental design dominates the conclusion.** A one-line change in the data (a bond fund instead of gold futures) turns "
    "the best strategy into the worst, underlining the need for the diagnostics reported here.",
])

P.h("4.6 Statistical analysis", 2)
rows = []
for k in CASES:
    b = BT["cases"][k]["bootstrap"]
    for base, lab in (("mean_variance", "vs Mean-Variance"), ("equal_weight", "vs Equal-Weight")):
        d = b[f"adaptive_minus_{base}"]
        rows.append([CNAME[k] if base == "mean_variance" else "", lab, f"{d['sharpe_diff']:+.2f}", f"[{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]",
                     f"{d['p_diff_le_0']:.2f}"])
P.table(["Setting", "Adaptive minus …", "ΔSharpe", "95% bootstrap CI", "P(Δ ≤ 0)"], rows,
        "Paired stationary-bootstrap (5,000 resamples, mean block 10 days) intervals for the test-segment Sharpe-ratio difference "
        "between the adaptive policy (seed 42) and each baseline.", widths=[1.5, 1.4, 0.8, 1.5, 0.8], font=8.3, align=["l", "l", "c", "c", "c"])
P.p(
    "Every interval contains zero: with test segments of 473–957 daily returns, none of the observed Sharpe differences is "
    "statistically distinguishable from zero, in either direction, so neither the dot-com “win” nor the other “losses” should be "
    "over-interpreted as evidence about the method. Seed-to-seed variability is a larger effect than most observed "
    "differences (Table 12: standard deviations of 0.08–0.23 Sharpe units in three settings), and the blocked cross-validation "
    f"standard deviation over encoder seeds is small (≤ {max(v['balanced_acc_sd'] for v in cv.values()):.3f} balanced accuracy), so the "
    "matcher conclusions are stable even though the allocator’s are not. Eight comparisons are reported without multiplicity "
    "correction; given that no interval excludes zero, correction would not change any conclusion."
)

P.h("4.7 Ablation study", 2)
P.p(
    "**Allocator.** Table 12 and Fig. 13 give mean ± s.d. test-segment results over six seeds. Removing the regime features "
    "(no-regime) does not systematically change the outcome: it lowers the GFC Sharpe ratio slightly and raises the COVID-19 one, "
    "both within one standard deviation. More tellingly, the *untrained* random policy matches or exceeds the trained policy in the "
    "GFC and COVID-19 settings (Sharpe 0.59 vs 0.36 and 1.02 vs 0.90) and has lower volatility in COVID-19 (14.1% vs 19.2%): "
    "evolution-strategies training on the short pre-crisis segment overfits and does not generalise to the crisis. In the "
    "as-deployed dot-com setting the *trained* variants are seed-independent because they converge to the same constant gold "
    "portfolio, while the untrained policy is not."
)
rows, bold = [], []
for k in CASES:
    sub = abl[abl.case == k]
    s0 = sub[(sub.cond == "full") & (sub.seed == 0)].iloc[0]
    rows.append([CNAME[k], "Equal-Weight", f"{s0.equal_weight_sharpe:.2f}", pct(s0.equal_weight_vol), pct(s0.equal_weight_mdd), "n/a"])
    rows.append(["", "Mean-Variance", f"{s0.mean_variance_sharpe:.2f}", pct(s0.mean_variance_vol), pct(s0.mean_variance_mdd), "n/a"])
    floor = min(s0.equal_weight_vol, s0.mean_variance_vol)
    for c, lab in (("full", "Adaptive (full)"), ("no_regime", "Adaptive (no regime)"), ("untrained", "Adaptive (untrained)")):
        x = sub[sub.cond == c]
        rows.append(["", lab, f"{x.adaptive_sharpe.mean():.2f} ± {x.adaptive_sharpe.std():.2f}",
                     f"{pct(x.adaptive_vol.mean())} ± {pct(x.adaptive_vol.std())}",
                     f"{pct(x.adaptive_mdd.mean())} ± {pct(x.adaptive_mdd.std())}", f"{(x.adaptive_vol < floor).mean() * 100:.0f}%"])
P.table(["Setting", "Strategy / variant", "Sharpe", "Ann. volatility", "Max drawdown", "Seeds with lowest volatility"], rows,
        "Allocator ablation: test-segment results, mean ± s.d. over six seeds (baselines are deterministic). The last column is the share "
        "of seeds in which the adaptive variant’s volatility is lower than that of both baselines.",
        widths=[1.15, 1.35, 1.0, 1.15, 1.15, 0.95], font=7.6, align=["l", "l", "c", "c", "c", "c"], shade_rows=tuple(range(0, len(rows), 5)))
P.figure(FIG / "fig_seed_ablation.png", "Out-of-sample Sharpe ratio of the adaptive policy across six seeds for the full, no-regime and untrained "
         "variants; dashed lines are the two baselines.", width=6.2)
P.p(
    "**Matcher.** Table 6 and Fig. 8 show that replacing the trained Transformer by an untrained encoder, by the raw shape, by PCA "
    "or by seven summary statistics changes balanced accuracy by at most 0.022 (Table 6); leave-one-event-out "
    "rates (Table 7) are likewise similar, with summary statistics highest for COVID-19 (96% vs 93% trained), the untrained encoder "
    "highest for dot-com (96% vs 90%) and the trained encoder above the raw features for dot-com and COVID-19. The honest reading is that the signal lives in the resampled shape "
    "itself, and the Transformer neither adds to nor removes it at this data scale."
)

# =================================================================== 5 LIMITATIONS
P.h("5. Limitations", 1)
P.bullets([
    "**Few events, strong dependence.** Three crises and four calm periods yield 965 *overlapping* windows with a heavily imbalanced "
    "normal class; blocked purging reduces but cannot remove dependence, and generalisation to other crises (1987, 1998, 2022) "
    "is untested. The three crises are defined by the authors’ calendar choices.",
    "**In-sample illustrations.** The example queries in Table 5 overlap the corpus; only the blocked and leave-one-event-out "
    "results measure generalisation.",
    "**Allocator.** Evolution strategies replace the PPO/DDPG agents of the reviewed literature, with a small state, 40 generations "
    "and short training segments; the negative result applies to this implementation, not to deep RL in general. Only long-only "
    "portfolios of three or four assets were studied, with a single cost assumption (5 bps charged on changes in *target* weights, not on trades that undo price drift) and no liquidity model.",
    "**Experimental design.** The as-deployed dot-com setting is invalid (30 training days) and is reported only to document the "
    "diagnosis; the corrected setting uses a mutual-fund proxy (VUSTX) for bonds. Test segments contain one crisis each.",
    "**Run-to-run sensitivity.** Re-running the COVID-19 backtest after a Yahoo data refresh changed the adaptive Sharpe ratio "
    "from 0.85 to 1.05 with identical code and seeds, probably through the HMM’s sensitivity to tiny data revisions; headline "
    "numbers should be read with the seed ranges in Table 12.",
    "**Data.** Yahoo Finance’s endpoint is unofficial and may rate-limit or change; adjusted prices are revised retroactively.",
    "**Scope.** The tool is descriptive and illustrative; it is not investment advice, and the next-21-day baselines are "
    "historical averages of few overlapping windows, not forecasts.",
])

# =================================================================== 6 CONCLUSION
P.h("6. Conclusion and Future Work", 1)
P.p(
    "We built and evaluated a platform that compares any market period with the dot-com, 2008 and COVID-19 crises through a "
    "length-invariant shape representation, a Transformer encoder and nearest-neighbour scoring over an HDBSCAN-structured "
    "corpus, alongside an HMM regime view and a walk-forward allocation workbench. The evaluation yields a clear, if modest, "
    "picture: the comparison reliably separates crisis-like from normal periods (crisis recall "
    f"{pct(TR['crisis_recall'], 0)} at {pct(TR['crisis_fpr'])} false alarms) and recognises unseen crises as crisis-like, but it cannot "
    "yet say *which* crisis a period resembles, and neither a trained Transformer nor the regime-aware ES allocator outperforms "
    "simple alternatives. We regard the diagnostics (the invalid-split failure, the untrained-encoder and untrained-policy "
    "controls, and the seed and bootstrap analysis) as useful results in their own right."
)
P.p("**Future work.** (i) Extend the corpus to more crises and markets and report event-level (not window-level) cross-validation; "
    "(ii) replace the single-crisis labels by a continuous severity or multi-label target and add shape descriptors that "
    "preserve timescale (e.g. the original duration) so that fast and slow crashes can be distinguished; (iii) pre-train "
    "the encoder on a much larger unlabelled corpus of windows, or compare contrastive objectives, before claiming a benefit from "
    "attention; (iv) implement PPO/DDPG agents under the same protocol with longer training windows, rolling retraining, "
    "regularisation of the policy and a realistic cost and liquidity model; (v) add automated tests and make the dot-com "
    "configuration valid by default; and (vi) evaluate the dashboard with potential users.")

# =================================================================== DECLARATIONS
P.h("Declarations", 1)
P.p("*The statements below are drafts prepared for the authors to confirm or amend before submission; the target journal’s "
    "exact declaration wording and requirements should be checked.*", justify=False)
P.p("**Funding.** No external funding was received for this work. *(To be confirmed by the authors.)*", justify=False)
P.p("**Conflict of interest / competing interests.** The authors declare no competing interests. *(To be confirmed by the authors.)*", justify=False)
P.p("**Ethical approval.** Not applicable: the study uses only publicly available market price data and involves no human "
    "participants or animals.", justify=False)
P.p("**Consent to participate.** Not applicable.", justify=False)
P.p("**Consent for publication.** Not applicable.", justify=False)
P.p("**Data availability.** All price data are publicly available from Yahoo Finance {cite:yahoo}. The derived experiment outputs "
    "(backtest.json, ablation.json, matcher.json, embeddings) are included in the project’s paper/results directory.", justify=False)
P.p("**Code availability.** The full source code (backend, dashboard, experiment and figure scripts) is in the project repository "
    "*[public repository URL to be added by the authors on release]*; commands to reproduce all results are listed in Appendix B.", justify=False)
P.p("**Author contributions.** *[To be completed by the authors using the journal’s contribution taxonomy, e.g. conceptualisation, "
    "software, formal analysis, writing, supervision.]* Dr. Arti Tiwari served as the faculty guide of the underlying course project.", justify=False)

# =================================================================== REFERENCES (filled after all citations collected)
P.h("References", 1)
P.reference_list()

# =================================================================== APPENDICES
P.page_break()
P.h("Appendix A. Software screenshots", 1)
P.p("The dashboard (React/Vite, dark theme) as captured with a headless browser against the running back end on 7 October 2026.")
P.figure(small_img(FIG / "ss_backtest.png"), "Portfolio Backtest tab (2008 GFC): value curves with the train/test split and crisis window, "
         "performance-metrics table and the adaptive policy’s allocation over time.", width=5.6)
P.figure(small_img(FIG / "ss_regime_covid.png"), "Regime Detection tab with the Historical Pattern Comparison report: the COVID-19 window "
         "(15 January – 15 May 2020, SPY) scored against the COVID-19 reference windows, with overlay, statistics and insights.", width=5.6)
P.figure(small_img(FIG / "ss_about.png"), "About / Tech Stack tab describing the method and its stack.", width=5.0)
P.h("Appendix B. Reproducibility", 1)
P.p("From the project root (Python 3.12 and Node 18+; no API keys):", justify=False)
for line in ["./run.sh                       # starts the API (127.0.0.1:8000) and the dashboard (127.0.0.1:5173)",
             "backend/.venv/bin/python paper/scripts/exp_backtest.py     # E1: backtests, bootstrap, dataset statistics",
             "backend/.venv/bin/python paper/scripts/exp_ablation.py     # E2: 6 seeds × 4 settings × 3 variants",
             "backend/.venv/bin/python paper/scripts/exp_matcher.py      # E3/E4: blocked CV, leave-one-event-out, embeddings",
             "python paper/scripts/fig_diagrams.py; python paper/scripts/fig_results.py   # figures",
             "backend/.venv/bin/python paper/scripts/build_paper.py      # this manuscript"]:
    para = P.doc.add_paragraph()
    para.paragraph_format.space_after = Pt(1)
    r = para.add_run(line); r.font.name = "Courier New"; r.font.size = Pt(8.5)
P.save(OUT)
print("saved", OUT, "| figures", P.fig_n, "tables", P.tab_n, "equations", P.eq_n, "refs", len(P.cite_order))
